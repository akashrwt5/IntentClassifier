#!/usr/bin/env python3
"""
FINAL V2.1 ONNX FP32 LOCKED 57-INTENT BENCHMARK

Fix:
The exported ONNX model has a fixed batch dimension:
    input_ids: int64 [1, 24]

Therefore this script NEVER sends the full locked dataset as [N, 24].
It performs one ONNX inference per row using [1, 24].

No training.
No checkpoint loading.
No modification of the locked test.
"""

from pathlib import Path
import json
import re
import time

import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


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
EXPECTED_CLASSES = 57


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_vocab(path):
    obj = load_json(path)

    if isinstance(obj, dict):
        if all(isinstance(v, int) for v in obj.values()):
            return obj

        if isinstance(obj.get("stoi"), dict):
            return obj["stoi"]

        if isinstance(obj.get("vocab"), dict):
            return obj["vocab"]

    if isinstance(obj, list):
        return {token: i for i, token in enumerate(obj)}

    raise ValueError(f"Unsupported vocab.json format: {path}")


def load_labels(path):
    obj = load_json(path)

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        if isinstance(obj.get("labels"), list):
            return obj["labels"]

        if all(str(k).isdigit() for k in obj.keys()):
            items = sorted(
                ((int(k), v) for k, v in obj.items()),
                key=lambda x: x[0],
            )
            return [v for _, v in items]

        if all(isinstance(v, int) for v in obj.values()):
            n = max(obj.values()) + 1
            labels = [None] * n
            for label, idx in obj.items():
                labels[idx] = label
            return labels

    raise ValueError(f"Unsupported labels.json format: {path}")


def find_column(df, candidates):
    normalized = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    return None


def tokenize(text, vocab):
    text = str(text).strip().lower()

    # Same simple tokenizer contract used by the compact model benchmark.
    tokens = re.findall(
        r"\w+|[^\w\s]",
        text,
        flags=re.UNICODE,
    )

    unk_id = vocab.get(
        "<unk>",
        vocab.get(
            "[UNK]",
            vocab.get("UNK", 1),
        ),
    )

    pad_id = vocab.get(
        "<pad>",
        vocab.get(
            "[PAD]",
            vocab.get("PAD", 0),
        ),
    )

    ids = [
        int(vocab.get(token, unk_id))
        for token in tokens[:MAX_LEN]
    ]

    if len(ids) < MAX_LEN:
        ids += [int(pad_id)] * (MAX_LEN - len(ids))

    return np.asarray(
        ids[:MAX_LEN],
        dtype=np.int64,
    )


def main():
    print("=" * 78)
    print("FINAL V2.1 ONNX FP32 LOCKED 57-INTENT BENCHMARK")
    print("=" * 78)

    required = [
        ONNX_PATH,
        LOCKED_CSV,
        VOCAB_PATH,
        LABELS_PATH,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing:\n{path}")

    labels = load_labels(LABELS_PATH)
    vocab = load_vocab(VOCAB_PATH)

    if len(labels) != EXPECTED_CLASSES:
        raise RuntimeError(
            f"Expected {EXPECTED_CLASSES} labels, "
            f"found {len(labels)}"
        )

    df = pd.read_csv(LOCKED_CSV)

    text_col = find_column(
        df,
        [
            "text",
            "utterance",
            "phrase",
            "query",
            "sentence",
            "input",
        ],
    )

    label_col = find_column(
        df,
        [
            "label",
            "intent",
            "target",
            "class",
        ],
    )

    if text_col is None or label_col is None:
        raise RuntimeError(
            "Could not identify text/label columns.\n"
            f"Available columns: {list(df.columns)}"
        )

    texts = (
        df[text_col]
        .fillna("")
        .astype(str)
        .tolist()
    )

    true_names = (
        df[label_col]
        .astype(str)
        .tolist()
    )

    label_to_id = {
        label: idx
        for idx, label in enumerate(labels)
    }

    unknown = sorted(
        set(true_names) - set(label_to_id)
    )

    if unknown:
        raise RuntimeError(
            "Locked CSV contains labels absent from labels.json:\n"
            + "\n".join(unknown)
        )

    # Build [N,24] only for bookkeeping.
    # IMPORTANT: it is NOT passed directly to the fixed-batch ONNX model.
    X = np.stack(
        [tokenize(text, vocab) for text in texts],
        axis=0,
    ).astype(np.int64)

    y_true = np.asarray(
        [label_to_id[label] for label in true_names],
        dtype=np.int64,
    )

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]

    input_name = input_meta.name
    output_name = output_meta.name

    print("\n--- ONNX CONTRACT ---")
    print(f"Input name : {input_name}")
    print(f"Input type : {input_meta.type}")
    print(f"Input shape: {input_meta.shape}")
    print(f"Output name: {output_name}")
    print(f"Output type: {output_meta.type}")
    print(f"Output shape: {output_meta.shape}")

    # Hard safety check. This prevents the exact error seen previously.
    actual_shape = input_meta.shape

    if len(actual_shape) != 2:
        raise RuntimeError(
            f"Unexpected ONNX input rank: {actual_shape}"
        )

    if actual_shape[0] != 1:
        raise RuntimeError(
            "This final benchmark expects the exported fixed-batch "
            f"model [1,24], but ONNX reports {actual_shape}."
        )

    if actual_shape[1] != MAX_LEN:
        raise RuntimeError(
            f"Expected sequence length {MAX_LEN}, "
            f"but ONNX reports {actual_shape}."
        )

    if input_meta.type != "tensor(int64)":
        raise RuntimeError(
            f"Expected int64 ONNX input, got {input_meta.type}"
        )

    print("\n--- INPUT SAFETY CHECK ---")
    print("Locked matrix shape :", X.shape)
    print("Per-row ONNX shape  : (1, 24)")
    print("Full [N,24] feed    : DISABLED")
    print("Batching error guard : ENABLED")

    # ------------------------------------------------------------------
    # CRITICAL SECTION
    # ------------------------------------------------------------------
    # The ONNX model requires exactly [1,24].
    # Every row is reshaped to [1,24] before session.run().
    # ------------------------------------------------------------------

    logits_rows = []

    start = time.perf_counter()

    for i in range(len(X)):
        row = X[i:i + 1]

        if row.shape != (1, MAX_LEN):
            raise RuntimeError(
                f"Internal shape error at row {i}: {row.shape}"
            )

        result = session.run(
            [output_name],
            {
                input_name: row,
            },
        )

        logits_one = np.asarray(
            result[0],
            dtype=np.float32,
        )

        if logits_one.shape != (1, EXPECTED_CLASSES):
            raise RuntimeError(
                f"Unexpected output at row {i}: "
                f"{logits_one.shape}; expected "
                f"(1,{EXPECTED_CLASSES})"
            )

        logits_rows.append(logits_one[0])

        if (i + 1) % 500 == 0 or i + 1 == len(X):
            print(
                f"Progress: {i + 1}/{len(X)}"
            )

    elapsed = time.perf_counter() - start

    logits = np.stack(
        logits_rows,
        axis=0,
    )

    y_pred = logits.argmax(axis=1)

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=np.arange(EXPECTED_CLASSES),
        target_names=labels,
        digits=4,
        zero_division=0,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=np.arange(EXPECTED_CLASSES),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    macro_f1 = report_dict["macro avg"]["f1-score"]
    weighted_f1 = report_dict["weighted avg"]["f1-score"]

    print("\n--- V2.1 ONNX LOCKED TEST RESULT ---")
    print(f"Accuracy   : {accuracy * 100:.4f}%")
    print(f"Macro F1   : {macro_f1 * 100:.4f}%")
    print(f"Weighted F1: {weighted_f1 * 100:.4f}%")

    print("\nClassification report:")
    print(report_text)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(EXPECTED_CLASSES),
    )

    pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    ).to_csv(CM_PATH)

    pred_df = pd.DataFrame({
        "text": texts,
        "true_intent": true_names,
        "predicted_intent": [
            labels[idx]
            for idx in y_pred
        ],
        "confidence": logits.max(axis=1),
    })

    pred_df.to_csv(
        PRED_PATH,
        index=False,
    )

    REPORT_PATH.write_text(
        report_text,
        encoding="utf-8",
    )

    rows_per_sec = (
        len(X) / elapsed
        if elapsed > 0
        else 0.0
    )

    ms_per_row = (
        elapsed * 1000 / len(X)
        if len(X)
        else 0.0
    )

    model_size_mb = (
        ONNX_PATH.stat().st_size
        / (1024 * 1024)
    )

    summary = {
        "model": "V2.1 Controlled ONNX FP32",
        "onnx_path": str(
            ONNX_PATH.resolve()
        ),
        "locked_test": str(
            LOCKED_CSV.resolve()
        ),
        "rows": int(len(X)),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "total_time_sec": float(elapsed),
        "rows_per_sec": float(rows_per_sec),
        "ms_per_row": float(ms_per_row),
        "model_size_mb": float(model_size_mb),
        "input_type": input_meta.type,
        "input_shape": [1, MAX_LEN],
        "output_type": output_meta.type,
        "output_shape": [1, EXPECTED_CLASSES],
        "fixed_batch": True,
        "batching_method": "one_row_at_a_time",
        "training_performed": False,
        "locked_test_used": True,
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n--- INFERENCE SPEED ---")
    print(f"Total rows : {len(X)}")
    print(f"Total time : {elapsed:.4f} sec")
    print(f"Rows/sec   : {rows_per_sec:.2f}")
    print(f"ms/row     : {ms_per_row:.4f}")
    print(f"Model size : {model_size_mb:.3f} MB")

    print("\nSaved:")
    print(PRED_PATH)
    print(REPORT_PATH)
    print(CM_PATH)
    print(SUMMARY_PATH)

    print("\nSTATUS:")
    print("V2.1 ONNX LOCKED 57-INTENT BENCHMARK COMPLETE")


if __name__ == "__main__":
    main()
