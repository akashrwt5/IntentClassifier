#!/usr/bin/env python3
"""
V2.1 ONNX FP32 LOCKED 57-INTENT BENCHMARK

Uses:
  ONNX:
    v3_57intent_v2_1_onnx/v2_1_57intent_fp32.onnx

  Locked test:
    v3_57intent_locked_eval/locked_test_57intent.csv

  Vocab / labels:
    vocab.json
    labels.json

No training.
No checkpoint loading.
No dataset modification.

IMPORTANT:
The tokenizer is intentionally kept compatible with the compact V3
whitespace/subword-style vocabulary contract:
  - lowercase
  - whitespace tokenization
  - unknown -> <unk>
  - padding -> <pad>
  - truncate/pad to MAX_LEN
"""

from pathlib import Path
import json
import re
import time

import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


ROOT = Path(__file__).resolve().parent

ONNX_PATH = ROOT / "v3_57intent_v2_1_onnx" / "v2_1_57intent_fp32.onnx"
LOCKED_CSV = ROOT / "v3_57intent_locked_eval" / "locked_test_57intent.csv"
VOCAB_PATH = ROOT / "vocab.json"
LABELS_PATH = ROOT / "labels.json"

OUT_DIR = ROOT / "v3_57intent_v2_1_onnx_locked_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_PATH = OUT_DIR / "locked_predictions_onnx_v2_1.csv"
REPORT_PATH = OUT_DIR / "classification_report_onnx_v2_1.txt"
CM_PATH = OUT_DIR / "confusion_matrix_onnx_v2_1.csv"
SUMMARY_PATH = OUT_DIR / "benchmark_summary_onnx_v2_1.json"

MAX_LEN = 24


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_vocab(path):
    obj = load_json(path)

    if isinstance(obj, dict):
        # Common formats:
        # {"word": id}
        if all(isinstance(v, int) for v in obj.values()):
            return obj

        # {"itos": [...], "stoi": {...}}
        if isinstance(obj.get("stoi"), dict):
            return obj["stoi"]

        # {"vocab": {"word": id}}
        if isinstance(obj.get("vocab"), dict):
            return obj["vocab"]

    if isinstance(obj, list):
        return {tok: i for i, tok in enumerate(obj)}

    raise ValueError(f"Unsupported vocab.json format: {path}")


def load_labels(path):
    obj = load_json(path)

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        # id -> label
        if all(str(k).isdigit() for k in obj.keys()):
            return [obj[str(i)] for i in range(len(obj))]

        # {"labels": [...]}
        if isinstance(obj.get("labels"), list):
            return obj["labels"]

        # label -> id
        if all(isinstance(v, int) for v in obj.values()):
            n = max(obj.values()) + 1
            out = [None] * n
            for label, idx in obj.items():
                out[idx] = label
            return out

    raise ValueError(f"Unsupported labels.json format: {path}")


def find_column(df, candidates):
    lower = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def tokenize(text, vocab):
    text = str(text).strip().lower()
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

    unk_id = (
        vocab.get("<unk>")
        if "<unk>" in vocab
        else vocab.get("[UNK]", vocab.get("UNK", 1))
    )

    pad_id = (
        vocab.get("<pad>")
        if "<pad>" in vocab
        else vocab.get("[PAD]", vocab.get("PAD", 0))
    )

    ids = [int(vocab.get(tok, unk_id)) for tok in tokens[:MAX_LEN]]
    ids = ids[:MAX_LEN]

    if len(ids) < MAX_LEN:
        ids += [int(pad_id)] * (MAX_LEN - len(ids))

    return ids


def main():
    print("=" * 78)
    print("V2.1 ONNX FP32 LOCKED 57-INTENT BENCHMARK")
    print("=" * 78)

    for path in [ONNX_PATH, LOCKED_CSV, VOCAB_PATH, LABELS_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing:\n{path}")

    vocab = load_vocab(VOCAB_PATH)
    labels = load_labels(LABELS_PATH)

    if len(labels) != 57:
        raise RuntimeError(
            f"Expected 57 labels, found {len(labels)}"
        )

    df = pd.read_csv(LOCKED_CSV)

    text_col = find_column(
        df,
        ["text", "utterance", "phrase", "query", "sentence", "input"]
    )
    label_col = find_column(
        df,
        ["label", "intent", "target", "class"]
    )

    if text_col is None or label_col is None:
        raise RuntimeError(
            f"Could not identify text/label columns.\n"
            f"Columns found: {list(df.columns)}"
        )

    texts = df[text_col].fillna("").astype(str).tolist()
    y_true_names = df[label_col].astype(str).tolist()

    label_to_id = {name: i for i, name in enumerate(labels)}

    unknown = sorted(set(y_true_names) - set(label_to_id))
    if unknown:
        raise RuntimeError(
            "Locked CSV contains labels absent from labels.json:\n"
            + "\n".join(unknown)
        )

    X = np.asarray(
        [tokenize(t, vocab) for t in texts],
        dtype=np.int64,
    )

    y_true = np.asarray(
        [label_to_id[x] for x in y_true_names],
        dtype=np.int64,
    )

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print("\n--- MODEL CONTRACT ---")
    print(f"Input name : {input_name}")
    print(f"Input shape: {session.get_inputs()[0].shape}")
    print(f"Input type : {session.get_inputs()[0].type}")
    print(f"Output name: {output_name}")
    print(f"Output shape: {session.get_outputs()[0].shape}")
    print(f"Rows       : {len(X)}")

    t0 = time.perf_counter()

    logits = session.run(
        [output_name],
        {input_name: X},
    )[0]

    elapsed = time.perf_counter() - t0

    logits = np.asarray(logits)
    y_pred = logits.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)

    report_text = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(labels)),
        target_names=labels,
        digits=4,
        zero_division=0,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(labels)),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    macro_f1 = report_dict["macro avg"]["f1-score"]
    weighted_f1 = report_dict["weighted avg"]["f1-score"]

    print("\n--- V2.1 ONNX LOCKED TEST RESULT ---")
    print(f"Accuracy   : {acc * 100:.4f}%")
    print(f"Macro F1   : {macro_f1 * 100:.4f}%")
    print(f"Weighted F1: {weighted_f1 * 100:.4f}%")

    print("\nClassification report:")
    print(report_text)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(labels)),
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    )
    cm_df.to_csv(CM_PATH)

    pred_df = pd.DataFrame({
        "text": texts,
        "true_intent": y_true_names,
        "predicted_intent": [labels[i] for i in y_pred],
        "confidence": logits.max(axis=1),
    })
    pred_df.to_csv(PRED_PATH, index=False)

    REPORT_PATH.write_text(
        report_text,
        encoding="utf-8",
    )

    rows_per_sec = len(X) / elapsed if elapsed > 0 else 0.0
    ms_per_row = elapsed * 1000 / len(X) if len(X) else 0.0

    summary = {
        "model": "V2.1 Controlled ONNX FP32",
        "onnx_path": str(ONNX_PATH.resolve()),
        "locked_test": str(LOCKED_CSV.resolve()),
        "rows": int(len(X)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "total_time_sec": float(elapsed),
        "rows_per_sec": float(rows_per_sec),
        "ms_per_row": float(ms_per_row),
        "input_shape": [1, MAX_LEN],
        "input_type": "int64",
        "output_shape": [1, 57],
        "output_type": "float32",
        "training_performed": False,
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n--- INFERENCE SPEED ---")
    print(f"Total rows : {len(X)}")
    print(f"Total time : {elapsed:.4f} sec")
    print(f"Rows/sec   : {rows_per_sec:.2f}")
    print(f"ms/row     : {ms_per_row:.4f}")

    print("\nSaved:")
    print(PRED_PATH)
    print(REPORT_PATH)
    print(CM_PATH)
    print(SUMMARY_PATH)

    print("\nSTATUS:")
    print("V2.1 ONNX LOCKED 57-INTENT BENCHMARK COMPLETE")


if __name__ == "__main__":
    main()
