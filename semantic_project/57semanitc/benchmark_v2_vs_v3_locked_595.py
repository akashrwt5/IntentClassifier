#!/usr/bin/env python3
"""
LOCKED 595 BENCHMARK
Compare:
  - Canonical V3 ONNX
  - V3 V2-targeted FP32 PyTorch checkpoint

IMPORTANT:
  - The 595-row CSV is evaluation-only.
  - No training.
  - No threshold fitting.
  - No ONNX export of V2.
  - Canonical V3 identity:
      Accuracy = 96.3025%
      Macro F1 = 96.3117%
      CSV SHA256 =
      755f7b05e7d0c3cf08ae301f98445170b334c4c9a2dd84c030f409f52ba89528
      Token SHA256 =
      13b782c0631e986b3b7649084676bc27e8c5281b1eea1813e4c81102c71a9242

Run from 57semanitc:
  python3 benchmark_v2_vs_v3_locked_595.py

If auto-discovery cannot find the canonical V3 ONNX, edit V3_ONNX below.
"""

from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
import onnxruntime as ort
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent

LOCKED_CSV = ROOT / "unseen_semantic_stress_test.csv"
V2_CHECKPOINT = ROOT / "v3_57intent_v2_model" / "student_v3_57intent_v2_best_fp32.pt"
VOCAB_PATH = ROOT / "vocab.json"
LABELS_PATH = ROOT / "labels.json"

EXPECTED_CSV_SHA256 = "755f7b05e7d0c3cf08ae301f98445170b334c4c9a2dd84c030f409f52ba89528"
EXPECTED_TOKEN_SHA256 = "13b782c0631e986b3b7649084676bc27e8c5281b1eea1813e4c81102c71a9242"

# If auto-discovery picks the wrong file, set this explicitly.
V3_ONNX = None

OUT_DIR = ROOT / "v3_vs_v2_locked_595_benchmark"
OUT_DIR.mkdir(exist_ok=True)

MAX_LEN = 24
VOCAB_SIZE = 895
NUM_CLASSES = 57
EMBED_DIM = 64
HEADS = 4
LAYERS = 2
FF_DIM = 128
DROPOUT = 0.10


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_columns(columns):
    lower = {str(c).strip().lower(): c for c in columns}
    text = next((lower[x] for x in ["text", "utterance", "query", "sentence", "input"] if x in lower), None)
    label = next((lower[x] for x in ["intent", "label", "category", "class"] if x in lower), None)
    if text is None or label is None:
        raise RuntimeError(f"Could not detect text/intent columns: {list(columns)}")
    return text, label


def load_vocab():
    obj = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    if "token_to_id" in obj:
        vocab = obj["token_to_id"]
    elif "vocab" in obj and isinstance(obj["vocab"], dict):
        vocab = obj["vocab"]
    elif all(isinstance(v, int) for v in obj.values()):
        vocab = obj
    else:
        raise RuntimeError("Unsupported vocab.json format")
    if len(vocab) != VOCAB_SIZE:
        raise RuntimeError(f"Expected vocab {VOCAB_SIZE}, got {len(vocab)}")
    return vocab


def load_labels():
    obj = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        labels = [str(x) for x in obj]
    elif isinstance(obj.get("labels"), list):
        labels = [str(x) for x in obj["labels"]]
    elif isinstance(obj.get("id_to_label"), dict):
        labels = [v for _, v in sorted((int(k), str(v)) for k, v in obj["id_to_label"].items())]
    elif isinstance(obj.get("label_to_id"), dict):
        labels = [k for _, k in sorted((int(v), str(k)) for k, v in obj["label_to_id"].items())]
    else:
        raise RuntimeError("Unsupported labels.json format")
    if len(labels) != NUM_CLASSES:
        raise RuntimeError(f"Expected {NUM_CLASSES} labels, got {len(labels)}")
    return labels


def tokenize(text, vocab):
    ids = []
    for token in str(text).lower().split():
        token = token.strip(".,!?;:\"'()[]{}")
        if token:
            ids.append(int(vocab.get(token, 1)))
    ids = ids[:MAX_LEN]
    ids += [0] * (MAX_LEN - len(ids))
    return ids


class V3Student57(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.position = nn.Embedding(MAX_LEN, EMBED_DIM)

        layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=HEADS,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=LAYERS)
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.classifier = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(EMBED_DIM, NUM_CLASSES),
        )

    def forward(self, x):
        pad = x.eq(0)
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.position(pos)
        h = self.encoder(h, src_key_padding_mask=pad)
        valid = (~pad).unsqueeze(-1).float()
        pooled = (h * valid).sum(1) / valid.sum(1).clamp(min=1.0)
        return self.classifier(self.norm(pooled))


def load_v2():
    model = V3Student57()
    state = torch.load(V2_CHECKPOINT, map_location="cpu")
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def find_v3_onnx():
    if V3_ONNX:
        p = Path(V3_ONNX)
        if not p.exists():
            raise FileNotFoundError(f"Configured V3_ONNX not found: {p}")
        return p

    candidates = [
        ROOT / "v3_57intent_onnx" / "v3_57intent_fp32.onnx",
        ROOT / "v3_57intent_onnx" / "model.onnx",
        ROOT / "v3_57intent_onnx" / "v3_57intent.onnx",
    ]
    candidates += sorted(ROOT.glob("v3_57intent_onnx/**/*.onnx"))

    # Avoid selecting an unrelated ONNX file.
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        raise FileNotFoundError(
            "Canonical V3 ONNX not found. Set V3_ONNX at the top of this script."
        )
    return candidates[0]


def main():
    print("=" * 78)
    print("LOCKED 595 — V3 vs V2 TARGETED")
    print("=" * 78)

    for p in [LOCKED_CSV, V2_CHECKPOINT, VOCAB_PATH, LABELS_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Missing:\n{p}")

    csv_hash = sha256_file(LOCKED_CSV)
    print("CSV SHA256:", csv_hash)

    if csv_hash != EXPECTED_CSV_SHA256:
        raise RuntimeError(
            "LOCKED CSV SHA256 MISMATCH.\n"
            f"Expected: {EXPECTED_CSV_SHA256}\n"
            f"Got     : {csv_hash}"
        )

    vocab = load_vocab()
    labels = load_labels()
    label_to_id = {x: i for i, x in enumerate(labels)}

    df = pd.read_csv(LOCKED_CSV)
    text_col, label_col = detect_columns(df.columns)
    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "intent"]
    df["text"] = df["text"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()

    if len(df) != 595:
        raise RuntimeError(f"Expected 595 rows, got {len(df)}")

    unknown = sorted(set(df["intent"]) - set(labels))
    if unknown:
        raise RuntimeError(f"Locked CSV contains labels absent from labels.json: {unknown}")

    X = np.asarray([tokenize(x, vocab) for x in df["text"]], dtype=np.int64)
    token_hash = hashlib.sha256(X.tobytes()).hexdigest()
    print("Token tensor SHA256:", token_hash)

    if token_hash != EXPECTED_TOKEN_SHA256:
        raise RuntimeError(
            "TOKEN SHA256 MISMATCH.\n"
            f"Expected: {EXPECTED_TOKEN_SHA256}\n"
            f"Got     : {token_hash}"
        )

    y = np.asarray([label_to_id[x] for x in df["intent"]], dtype=np.int64)

    # ---------------- V3 ONNX ----------------
    v3_path = find_v3_onnx()
    print("\nCanonical V3 ONNX:", v3_path)

    session = ort.InferenceSession(
        str(v3_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    v3_logits = session.run(
        [output_name],
        {input_name: X},
    )[0]

    # Some ONNX exports are fixed [1,24]. In that case run one row at a time.
    if v3_logits.shape[0] != 595:
        all_logits = []
        for i in range(595):
            out = session.run(
                [output_name],
                {input_name: X[i:i+1]},
            )[0]
            all_logits.append(out[0])
        v3_logits = np.asarray(all_logits, dtype=np.float32)

    v3_pred = np.argmax(v3_logits, axis=1)
    v3_acc = accuracy_score(y, v3_pred)
    v3_f1 = f1_score(y, v3_pred, average="macro", zero_division=0)

    # ---------------- V2 PyTorch ----------------
    device = torch.device("cpu")
    v2 = load_v2().to(device)

    with torch.no_grad():
        v2_logits = v2(
            torch.from_numpy(X).to(device)
        ).cpu().numpy()

    v2_pred = np.argmax(v2_logits, axis=1)
    v2_acc = accuracy_score(y, v2_pred)
    v2_f1 = f1_score(y, v2_pred, average="macro", zero_division=0)

    print("\n--- LOCKED 595 RESULT ---")
    print(f"V3 Accuracy   : {v3_acc*100:.4f}%")
    print(f"V3 Macro F1   : {v3_f1*100:.4f}%")
    print(f"V2 Accuracy   : {v2_acc*100:.4f}%")
    print(f"V2 Macro F1   : {v2_f1*100:.4f}%")
    print(f"Accuracy delta: {(v2_acc-v3_acc)*100:+.4f} pp")
    print(f"Macro F1 delta: {(v2_f1-v3_f1)*100:+.4f} pp")

    print("\n--- V2 CLASSIFICATION REPORT ---")
    print(
        classification_report(
            y,
            v2_pred,
            labels=np.arange(NUM_CLASSES),
            target_names=labels,
            digits=4,
            zero_division=0,
        )
    )

    # Critical intent rows = one representative row per intent, same style as prior audit.
    critical_indices = []
    for intent_id in range(NUM_CLASSES):
        idx = np.where(y == intent_id)[0]
        if len(idx):
            critical_indices.append(idx[0])

    v3_critical = accuracy_score(y[critical_indices], v3_pred[critical_indices])
    v2_critical = accuracy_score(y[critical_indices], v2_pred[critical_indices])

    print("--- CRITICAL ---")
    print("Rows:", len(critical_indices))
    print(f"V3 critical accuracy: {v3_critical*100:.2f}%")
    print(f"V2 critical accuracy: {v2_critical*100:.2f}%")

    # Row-level comparison
    rows = []
    for i in range(len(df)):
        rows.append({
            "row": i,
            "text": df.iloc[i]["text"],
            "true_intent": df.iloc[i]["intent"],
            "v3_predicted_intent": labels[int(v3_pred[i])],
            "v2_predicted_intent": labels[int(v2_pred[i])],
            "v3_correct": bool(v3_pred[i] == y[i]),
            "v2_correct": bool(v2_pred[i] == y[i]),
            "v3_confidence": float(
                np.exp(v3_logits[i] - np.max(v3_logits[i])).max()
                / np.exp(v3_logits[i] - np.max(v3_logits[i])).sum()
            ),
            "v2_confidence": float(
                np.exp(v2_logits[i] - np.max(v2_logits[i])).max()
                / np.exp(v2_logits[i] - np.max(v2_logits[i])).sum()
            ),
        })

    details = pd.DataFrame(rows)
    details.to_csv(
        OUT_DIR / "locked_595_v2_vs_v3_details.csv",
        index=False,
    )

    confusion_v2 = pd.DataFrame(
        confusion_matrix(
            y,
            v2_pred,
            labels=np.arange(NUM_CLASSES),
        ),
        index=labels,
        columns=labels,
    )
    confusion_v2.to_csv(
        OUT_DIR / "v2_confusion_matrix.csv"
    )

    regressions = details[
        (details["v3_correct"]) &
        (~details["v2_correct"])
    ].copy()

    recoveries = details[
        (~details["v3_correct"]) &
        (details["v2_correct"])
    ].copy()

    regressions.to_csv(
        OUT_DIR / "v2_regressions_vs_v3.csv",
        index=False,
    )
    recoveries.to_csv(
        OUT_DIR / "v2_recoveries_vs_v3.csv",
        index=False,
    )

    summary = {
        "locked_csv": str(LOCKED_CSV),
        "csv_sha256": csv_hash,
        "token_sha256": token_hash,
        "rows": 595,
        "v3_canonical_accuracy": float(v3_acc),
        "v3_canonical_macro_f1": float(v3_f1),
        "v2_accuracy": float(v2_acc),
        "v2_macro_f1": float(v2_f1),
        "accuracy_delta_pp": float((v2_acc - v3_acc) * 100),
        "macro_f1_delta_pp": float((v2_f1 - v3_f1) * 100),
        "critical_rows": len(critical_indices),
        "v3_critical_accuracy": float(v3_critical),
        "v2_critical_accuracy": float(v2_critical),
        "v3_to_v2_regressions": int(len(regressions)),
        "v3_to_v2_recoveries": int(len(recoveries)),
        "quality_gate": {
            "accuracy_ge_v3": bool(v2_acc >= v3_acc),
            "macro_f1_ge_v3": bool(v2_f1 >= v3_f1),
            "critical_not_worse": bool(v2_critical >= v3_critical),
        },
    }

    (OUT_DIR / "v2_vs_v3_locked_595_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n--- QUALITY GATE ---")
    print("Accuracy >= canonical V3 :", "PASS" if v2_acc >= v3_acc else "FAIL")
    print("Macro F1 >= canonical V3 :", "PASS" if v2_f1 >= v3_f1 else "FAIL")
    print("Critical not worse       :", "PASS" if v2_critical >= v3_critical else "FAIL")

    if (
        v2_acc >= v3_acc
        and v2_f1 >= v3_f1
        and v2_critical >= v3_critical
    ):
        print("\nSTATUS: V2 PASSES LOCKED-595 QUALITY GATE")
        print("Next: ONNX export + PyTorch/ONNX parity.")
    else:
        print("\nSTATUS: V2 FAILS LOCKED-595 QUALITY GATE")
        print("Keep canonical V3. Do NOT export V2 to ONNX yet.")

    print("\nSaved:")
    print(OUT_DIR / "locked_595_v2_vs_v3_details.csv")
    print(OUT_DIR / "v2_confusion_matrix.csv")
    print(OUT_DIR / "v2_regressions_vs_v3.csv")
    print(OUT_DIR / "v2_recoveries_vs_v3.csv")
    print(OUT_DIR / "v2_vs_v3_locked_595_summary.json")


if __name__ == "__main__":
    main()
