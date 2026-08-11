#!/usr/bin/env python3
"""
EXACT V2.1 PYTORCH <-> ONNX PARITY AUDIT

This version does NOT reconstruct the Transformer architecture manually.

Instead it:
1. Locates the existing V2.1/training model source in the project.
2. Imports the original model class from that source.
3. Builds the model from checkpoint tensor dimensions/signature.
4. Loads the ORIGINAL Controlled V2.1 checkpoint.
5. Uses the same locked CSV and project vocab/labels.
6. Compares original PyTorch logits against the exported ONNX logits.

If the original model source cannot be discovered or instantiated exactly,
the script stops rather than producing a misleading parity number.
"""

from pathlib import Path
import ast
import importlib.util
import inspect
import json
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import onnxruntime as ort
from sklearn.metrics import accuracy_score, classification_report


ROOT = Path("/Users/shuklam/IntentClassifier/semantic_project/57semanitc")

LOCKED = ROOT / "v3_57intent_locked_eval" / "locked_test_57intent.csv"
CKPT = ROOT / "v3_57intent_v2_1_controlled" / "student_v3_57intent_v2_1_best_fp32.pt"
ONNX = ROOT / "v3_57intent_v2_1_onnx" / "v2_1_57intent_fp32.onnx"
VOCAB = ROOT / "vocab.json"
LABELS = ROOT / "labels.json"

OUT = ROOT / "v3_57intent_v2_1_exact_parity"
OUT.mkdir(parents=True, exist_ok=True)

DETAIL = OUT / "exact_pytorch_vs_onnx_rows.csv"
MISMATCH = OUT / "exact_prediction_mismatches.csv"
SUMMARY = OUT / "exact_parity_summary.json"

MAX_LEN = 24
NCLASS = 57


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_vocab(p):
    x = load_json(p)
    if isinstance(x, list):
        return {str(v): i for i, v in enumerate(x)}
    if isinstance(x, dict):
        if isinstance(x.get("stoi"), dict):
            return x["stoi"]
        if isinstance(x.get("vocab"), dict):
            return x["vocab"]
        if all(isinstance(v, int) for v in x.values()):
            return x
    raise RuntimeError(f"Unsupported vocab format: {p}")


def load_labels(p):
    x = load_json(p)
    if isinstance(x, list):
        return x
    if isinstance(x, dict):
        if isinstance(x.get("labels"), list):
            return x["labels"]
        if all(str(k).isdigit() for k in x):
            return [v for k, v in sorted(x.items(), key=lambda z: int(z[0]))]
        if all(isinstance(v, int) for v in x.values()):
            out = [None] * (max(x.values()) + 1)
            for k, v in x.items():
                out[v] = k
            return out
    raise RuntimeError(f"Unsupported labels format: {p}")


def find_col(df, names):
    m = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n in m:
            return m[n]
    return None


def checkpoint_state():
    x = torch.load(CKPT, map_location="cpu")
    if isinstance(x, dict) and "state_dict" in x:
        x = x["state_dict"]
    if not isinstance(x, dict):
        raise RuntimeError("Unsupported checkpoint format")
    return x


def discover_python_sources():
    preferred = [
        ROOT / "train_v3_57intent_v2_1_controlled.py",
        ROOT / "train_v3_57intent_v2_1.py",
        ROOT / "train_v3_57intent_v2_targeted.py",
    ]

    files = []
    for p in preferred:
        if p.exists():
            files.append(p)

    for p in sorted(ROOT.glob("*.py")):
        if p not in files and re.search(r"(train|v2.?1|controlled|targeted)", p.name, re.I):
            files.append(p)

    return files


def source_model_classes(path):
    """
    AST discovery prevents importing arbitrary helper scripts first.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = []
            for b in node.bases:
                if isinstance(b, ast.Name):
                    bases.append(b.id)
                elif isinstance(b, ast.Attribute):
                    bases.append(b.attr)

            if (
                "Module" in bases
                or "Model" in node.name
                or "Encoder" in node.name
                or "Student" in node.name
                or "Classifier" in node.name
            ):
                names.append(node.name)

    return names


def import_module(path):
    name = "_exact_parity_" + re.sub(r"\W+", "_", path.stem)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def infer_dimensions(state):
    d = {}

    if "embedding.weight" in state:
        d["vocab_size"] = int(state["embedding.weight"].shape[0])
        d["d_model"] = int(state["embedding.weight"].shape[1])

    if "position.weight" in state:
        d["max_len"] = int(state["position.weight"].shape[0])

    m = re.match(r"encoder\.layers\.(\d+)\.", "")
    layers = []
    for k in state:
        q = re.match(r"encoder\.layers\.(\d+)\.", k)
        if q:
            layers.append(int(q.group(1)))
    if layers:
        d["num_layers"] = max(layers) + 1

    if "encoder.layers.0.linear1.weight" in state:
        d["ff_dim"] = int(state["encoder.layers.0.linear1.weight"].shape[0])

    if "classifier.0.weight" in state:
        d["classifier0_out"] = int(state["classifier.0.weight"].shape[0])
        d["classifier0_in"] = int(state["classifier.0.weight"].shape[1])

    if "classifier.3.weight" in state:
        d["num_classes"] = int(state["classifier.3.weight"].shape[0])

    return d


def try_construct(cls, dims):
    """
    Try constructor signatures commonly used by the project.
    We only accept a model if its state_dict loads STRICTLY.
    """
    sig = inspect.signature(cls)
    params = sig.parameters

    candidates = [
        dict(
            vocab_size=dims.get("vocab_size"),
            num_classes=NCLASS,
            d_model=dims.get("d_model"),
            max_len=dims.get("max_len", MAX_LEN),
            num_layers=dims.get("num_layers", 2),
        ),
        dict(
            vocab_size=dims.get("vocab_size"),
            num_classes=NCLASS,
            embed_dim=dims.get("d_model"),
            max_len=dims.get("max_len", MAX_LEN),
        ),
        dict(
            vocab_size=dims.get("vocab_size"),
            num_classes=NCLASS,
            hidden_dim=dims.get("d_model"),
            max_len=dims.get("max_len", MAX_LEN),
        ),
        dict(
            vocab_size=dims.get("vocab_size"),
            num_classes=NCLASS,
            d_model=dims.get("d_model"),
        ),
        dict(
            vocab_size=dims.get("vocab_size"),
            n_classes=NCLASS,
            d_model=dims.get("d_model"),
        ),
        dict(
            vocab_size=dims.get("vocab_size"),
            num_classes=NCLASS,
        ),
    ]

    cleaned = []
    for kw in candidates:
        kw = {
            k: v for k, v in kw.items()
            if v is not None and k in params
        }
        cleaned.append(kw)

    # Also try positional values for simple constructors.
    attempts = cleaned + [
        (dims.get("vocab_size"), NCLASS),
        (dims.get("vocab_size"), dims.get("d_model"), NCLASS),
    ]

    for args in attempts:
        try:
            if isinstance(args, dict):
                model = cls(**args)
            else:
                model = cls(*args)

            if not isinstance(model, nn.Module):
                continue

            return model
        except Exception:
            continue

    return None


def load_exact_model(state, dims):
    sources = discover_python_sources()

    print("\n--- MODEL SOURCE DISCOVERY ---")
    for p in sources:
        print(p)

    failures = []

    for source in sources:
        try:
            class_names = source_model_classes(source)
            if not class_names:
                continue

            module = import_module(source)

            for class_name in class_names:
                cls = getattr(module, class_name, None)
                if cls is None or not inspect.isclass(cls):
                    continue

                try:
                    model = try_construct(cls, dims)
                    if model is None:
                        continue

                    model.load_state_dict(state, strict=True)
                    model.eval()

                    print("\nEXACT MODEL FOUND")
                    print(f"Source : {source}")
                    print(f"Class  : {class_name}")
                    return model, source, class_name

                except Exception as exc:
                    failures.append(
                        f"{source.name}::{class_name}: {exc}"
                    )

        except Exception as exc:
            failures.append(
                f"{source.name}: import failed: {exc}"
            )

    msg = (
        "\nCould not reconstruct the original V2.1 model exactly.\n"
        "The script intentionally refuses to calculate a fake parity result.\n\n"
        "Checkpoint dimensions:\n"
        + json.dumps(dims, indent=2)
        + "\n\nAttempts:\n"
        + "\n".join(failures[-30:])
    )
    raise RuntimeError(msg)


def encode_default(text, vocab):
    toks = re.findall(
        r"\w+|[^\w\s]",
        str(text).lower().strip(),
        flags=re.UNICODE,
    )

    unk = vocab.get("<unk>", vocab.get("[UNK]", vocab.get("UNK", 1)))
    pad = vocab.get("<pad>", vocab.get("[PAD]", vocab.get("PAD", 0)))

    ids = [int(vocab.get(t, unk)) for t in toks[:MAX_LEN]]
    ids += [int(pad)] * (MAX_LEN - len(ids))

    return np.asarray(ids[:MAX_LEN], dtype=np.int64)


def main():
    print("=" * 78)
    print("EXACT V2.1 PYTORCH <-> ONNX PARITY AUDIT")
    print("=" * 78)

    for p in [LOCKED, CKPT, ONNX, VOCAB, LABELS]:
        if not p.exists():
            raise FileNotFoundError(f"Missing:\n{p}")

    vocab = load_vocab(VOCAB)
    labels = load_labels(LABELS)

    if len(labels) != NCLASS:
        raise RuntimeError(
            f"Expected {NCLASS} labels, got {len(labels)}"
        )

    df = pd.read_csv(LOCKED)
    tc = find_col(df, ["text", "utterance", "phrase", "query", "sentence", "input"])
    lc = find_col(df, ["label", "intent", "target", "class"])

    if tc is None or lc is None:
        raise RuntimeError(f"Could not find text/label columns: {list(df.columns)}")

    texts = df[tc].fillna("").astype(str).tolist()
    true_names = df[lc].astype(str).tolist()

    l2i = {x: i for i, x in enumerate(labels)}
    unknown = sorted(set(true_names) - set(l2i))
    if unknown:
        raise RuntimeError(f"Unknown locked labels: {unknown}")

    X = np.stack([encode_default(x, vocab) for x in texts])
    y = np.asarray([l2i[x] for x in true_names], dtype=np.int64)

    state = checkpoint_state()
    dims = infer_dimensions(state)

    print("\n--- CHECKPOINT ---")
    print(json.dumps(dims, indent=2))

    model, source, class_name = load_exact_model(state, dims)

    # PyTorch reference.
    with torch.no_grad():
        pt_logits = model(torch.from_numpy(X)).cpu().numpy().astype(np.float32)

    if pt_logits.shape != (len(X), NCLASS):
        raise RuntimeError(
            f"Exact PyTorch model produced {pt_logits.shape}; "
            f"expected {(len(X), NCLASS)}"
        )

    pt_pred = pt_logits.argmax(axis=1)
    pt_acc = accuracy_score(y, pt_pred)

    # ONNX reference.
    sess = ort.InferenceSession(
        str(ONNX),
        providers=["CPUExecutionProvider"],
    )
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]

    print("\n--- ONNX CONTRACT ---")
    print("Input :", inp.name, inp.type, inp.shape)
    print("Output:", out.name, out.type, out.shape)

    if inp.shape != [1, MAX_LEN]:
        raise RuntimeError(f"Expected ONNX [1,{MAX_LEN}], got {inp.shape}")

    onnx_rows = []
    t0 = time.perf_counter()

    for i in range(len(X)):
        one = X[i:i + 1]
        z = sess.run([out.name], {inp.name: one})[0]
        z = np.asarray(z, dtype=np.float32)

        if z.shape != (1, NCLASS):
            raise RuntimeError(
                f"ONNX row {i}: {z.shape}; expected (1,{NCLASS})"
            )

        onnx_rows.append(z[0])

    onnx_time = time.perf_counter() - t0
    ox_logits = np.stack(onnx_rows)
    ox_pred = ox_logits.argmax(axis=1)
    ox_acc = accuracy_score(y, ox_pred)

    diff = np.abs(pt_logits - ox_logits)

    pt_norm = np.linalg.norm(pt_logits, axis=1)
    ox_norm = np.linalg.norm(ox_logits, axis=1)
    cosine = np.sum(pt_logits * ox_logits, axis=1) / np.maximum(
        pt_norm * ox_norm, 1e-12
    )

    same = pt_pred == ox_pred

    print("\n" + "=" * 78)
    print("EXACT PARITY RESULT")
    print("=" * 78)
    print(f"PyTorch accuracy          : {pt_acc * 100:.4f}%")
    print(f"ONNX accuracy             : {ox_acc * 100:.4f}%")
    print(f"Accuracy delta            : {(ox_acc - pt_acc) * 100:+.4f} pp")
    print(f"Max absolute logit diff   : {diff.max():.10f}")
    print(f"Mean absolute logit diff  : {diff.mean():.10f}")
    print(f"Median absolute diff      : {np.median(diff):.10f}")
    print(f"P99 absolute diff         : {np.percentile(diff,99):.10f}")
    print(f"Mean cosine similarity    : {cosine.mean():.10f}")
    print(f"Min cosine similarity     : {cosine.min():.10f}")
    print(f"Prediction agreement      : {same.mean()*100:.4f}%")
    print(f"Prediction mismatches     : {(~same).sum()} / {len(X)}")

    details = pd.DataFrame({
        "row": np.arange(len(X)),
        "text": texts,
        "true_intent": true_names,
        "pytorch_prediction": [labels[i] for i in pt_pred],
        "onnx_prediction": [labels[i] for i in ox_pred],
        "prediction_same": same,
        "max_abs_logit_diff": diff.max(axis=1),
        "mean_abs_logit_diff": diff.mean(axis=1),
        "cosine_similarity": cosine,
    })
    details.to_csv(DETAIL, index=False)
    details[~details["prediction_same"]].to_csv(MISMATCH, index=False)

    summary = {
        "status": "EXACT_PARITY_AUDIT_COMPLETE",
        "model_source": str(source),
        "model_class": class_name,
        "rows": len(X),
        "pytorch_accuracy": float(pt_acc),
        "onnx_accuracy": float(ox_acc),
        "accuracy_delta_onnx_minus_pytorch": float(ox_acc - pt_acc),
        "max_abs_logit_diff": float(diff.max()),
        "mean_abs_logit_diff": float(diff.mean()),
        "median_abs_logit_diff": float(np.median(diff)),
        "p99_abs_logit_diff": float(np.percentile(diff, 99)),
        "mean_cosine_similarity": float(cosine.mean()),
        "min_cosine_similarity": float(cosine.min()),
        "prediction_agreement": float(same.mean()),
        "prediction_mismatch_count": int((~same).sum()),
        "locked_test_used": True,
        "training_performed": False,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(DETAIL)
    print(MISMATCH)
    print(SUMMARY)
    print("\nSTATUS: EXACT V2.1 PYTORCH <-> ONNX PARITY AUDIT COMPLETE")


if __name__ == "__main__":
    main()
