#!/usr/bin/env python3

"""
FINAL 57-INTENT V3 ONNX BENCHMARK

Runs the fixed-batch ONNX model one row at a time, so an ONNX model
with input shape [1,24] is never incorrectly fed [N,24].

Expected model:
    v3_57intent_onnx/v3_semantic_student_57intent_fp32.onnx

Expected input:
    input_ids : int64 [1,24]

Expected output:
    logits    : float32 [1,57]

The benchmark uses the EXISTING vocab.json and labels.json.
It does NOT retrain the model and does NOT rebuild the vocabulary.

Usage:
    python3 benchmark_v3_57intent_onnx.py

If your CSV has a different filename/path, change CSV_PATH below.
"""

from pathlib import Path
import csv
import json
import hashlib
import time

import numpy as np
import onnxruntime as ort


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/Users/shuklam/IntentClassifier/semantic_project/57semanitc"
)

ONNX_PATH = (
    ROOT
    / "v3_57intent_onnx"
    / "v3_semantic_student_57intent_fp32.onnx"
)

# CHANGE THIS if your 57-intent CSV has another name/location.
CSV_PATH = (
    ROOT
    / "train.csv"
)

VOCAB_PATH = (
    ROOT
    / "vocab.json"
)

LABELS_PATH = (
    ROOT
    / "labels.json"
)

OUTPUT_DIR = (
    ROOT
    / "v3_57intent_onnx_benchmark"
)

DETAILS_CSV = (
    OUTPUT_DIR
    / "onnx_predictions.csv"
)

REPORT_TXT = (
    OUTPUT_DIR
    / "classification_report.txt"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "benchmark_summary.json"
)


# ============================================================
# EXPECTED MODEL CONTRACT
# ============================================================

EXPECTED_VOCAB_SIZE = 895
EXPECTED_SEQ_LEN = 24
EXPECTED_CLASSES = 57


# ============================================================
# TOKENIZER
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_vocab(vocab_obj):
    """
    Supports common vocab formats:
      {"word": id}
      {"token_to_id": {"word": id}}
      {"vocab": {"word": id}}
    """

    if not isinstance(vocab_obj, dict):
        raise RuntimeError(
            f"Unsupported vocab format: {type(vocab_obj)}"
        )

    for key in ("token_to_id", "vocab", "stoi"):
        value = vocab_obj.get(key)
        if isinstance(value, dict):
            return value

    # Raw token -> integer mapping.
    if all(
        isinstance(v, int)
        for v in vocab_obj.values()
    ):
        return vocab_obj

    raise RuntimeError(
        "Could not find token->id mapping in vocab.json."
    )


def normalize_labels(labels_obj):
    """
    Supports:
      ["intent0", "intent1", ...]
      {"0": "intent0", ...}
      {"intent0": 0, ...}
      {"labels": [...]}
      {"label_to_id": {...}}
    """

    if isinstance(labels_obj, list):
        return [str(x) for x in labels_obj]

    if not isinstance(labels_obj, dict):
        raise RuntimeError(
            "Unsupported labels.json format."
        )

    for key in ("labels", "id_to_label"):
        value = labels_obj.get(key)
        if isinstance(value, list):
            return [str(x) for x in value]

        if isinstance(value, dict):
            pairs = []
            for k, v in value.items():
                try:
                    idx = int(k)
                except ValueError:
                    continue
                pairs.append((idx, str(v)))

            if pairs:
                pairs.sort()
                return [v for _, v in pairs]

    for key in ("label_to_id", "intent_to_id"):
        value = labels_obj.get(key)
        if isinstance(value, dict):
            pairs = [
                (int(v), str(k))
                for k, v in value.items()
            ]
            pairs.sort()
            return [v for _, v in pairs]

    # Try raw dict.
    if all(
        isinstance(v, int)
        for v in labels_obj.values()
    ):
        pairs = [
            (int(v), str(k))
            for k, v in labels_obj.items()
        ]
        pairs.sort()
        return [v for _, v in pairs]

    if all(
        str(k).isdigit()
        for k in labels_obj.keys()
    ):
        pairs = [
            (int(k), str(v))
            for k, v in labels_obj.items()
        ]
        pairs.sort()
        return [v for _, v in pairs]

    raise RuntimeError(
        "Could not determine label ordering from labels.json."
    )


def tokenize(text, vocab, max_len=24):
    """
    Conservative tokenizer matching the simple V3 word-vocabulary
    convention: lowercase whitespace/punctuation-separated tokens.

    If your original training tokenizer used a custom normalization,
    replace ONLY this function with that exact tokenizer.
    """

    text = str(text).strip().lower()

    # Match common punctuation-separated tokenization without
    # requiring an external tokenizer package.
    cleaned = (
        text
        .replace(",", " ")
        .replace(".", " ")
        .replace("!", " ")
        .replace("?", " ")
        .replace(":", " ")
        .replace(";", " ")
        .replace("'", " ")
        .replace('"', " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace("[", " ")
        .replace("]", " ")
        .replace("/", " ")
        .replace("-", " ")
        .replace("_", " ")
    )

    tokens = cleaned.split()

    unk_id = (
        vocab.get("<unk>")
        if "<unk>" in vocab
        else vocab.get("[UNK]", 1)
    )

    pad_id = (
        vocab.get("<pad>")
        if "<pad>" in vocab
        else vocab.get("[PAD]", 0)
    )

    ids = [
        int(vocab.get(token, unk_id))
        for token in tokens[:max_len]
    ]

    if len(ids) < max_len:
        ids.extend(
            [pad_id] * (max_len - len(ids))
        )

    return ids


# ============================================================
# CSV
# ============================================================

def detect_columns(fieldnames):
    lowered = {
        str(x).strip().lower(): x
        for x in fieldnames
    }

    text_candidates = [
        "text",
        "utterance",
        "query",
        "sentence",
        "input",
    ]

    label_candidates = [
        "intent",
        "label",
        "category",
        "class",
    ]

    text_col = None
    label_col = None

    for candidate in text_candidates:
        if candidate in lowered:
            text_col = lowered[candidate]
            break

    for candidate in label_candidates:
        if candidate in lowered:
            label_col = lowered[candidate]
            break

    if text_col is None or label_col is None:
        raise RuntimeError(
            "Could not detect CSV columns.\n"
            f"Columns found: {fieldnames}\n"
            "Expected text/intent (or equivalent) columns."
        )

    return text_col, label_col


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"\nCSV not found:\n{path}\n\n"
            "Edit CSV_PATH at the top of this script."
        )

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise RuntimeError(
                "CSV has no header."
            )

        text_col, label_col = detect_columns(
            reader.fieldnames
        )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            text = str(
                row.get(text_col, "")
            ).strip()

            label = str(
                row.get(label_col, "")
            ).strip()

            if not text or not label:
                continue

            rows.append(
                {
                    "row_number": row_number,
                    "text": text,
                    "label": label,
                }
            )

    if not rows:
        raise RuntimeError(
            "CSV contains zero usable rows."
        )

    return rows


# ============================================================
# METRICS
# ============================================================

def confusion_matrix(y_true, y_pred, n):
    cm = np.zeros(
        (n, n),
        dtype=np.int64,
    )

    for t, p in zip(
        y_true,
        y_pred,
    ):
        if 0 <= t < n and 0 <= p < n:
            cm[t, p] += 1

    return cm


def classification_metrics(
    y_true,
    y_pred,
    n_classes,
):
    cm = confusion_matrix(
        y_true,
        y_pred,
        n_classes,
    )

    rows = []

    total = len(y_true)
    correct = int(
        sum(
            int(a == b)
            for a, b in zip(
                y_true,
                y_pred,
            )
        )
    )

    for i in range(n_classes):

        tp = int(cm[i, i])
        support = int(cm[i].sum())

        predicted = int(
            cm[:, i].sum()
        )

        fn = support - tp
        fp = predicted - tp

        precision = (
            tp / predicted
            if predicted
            else 0.0
        )

        recall = (
            tp / support
            if support
            else 0.0
        )

        f1 = (
            2.0 * precision * recall
            / (precision + recall)
            if precision + recall
            else 0.0
        )

        accuracy = (
            recall
            if support
            else 0.0
        )

        rows.append(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "accuracy": accuracy,
            }
        )

    macro_precision = float(
        np.mean(
            [r["precision"] for r in rows]
        )
    )

    macro_recall = float(
        np.mean(
            [r["recall"] for r in rows]
        )
    )

    macro_f1 = float(
        np.mean(
            [r["f1"] for r in rows]
        )
    )

    weighted_f1 = (
        sum(
            r["f1"] * r["support"]
            for r in rows
        )
        / total
        if total
        else 0.0
    )

    return {
        "accuracy": (
            correct / total
            if total
            else 0.0
        ),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "rows": rows,
        "cm": cm,
    }


def format_report(
    labels,
    metrics,
):
    lines = []

    lines.append(
        "precision    recall  f1-score   support"
    )
    lines.append("")

    for label, row in zip(
        labels,
        metrics["rows"],
    ):
        lines.append(
            f"{label:<36}"
            f"{row['precision']:>8.4f}"
            f"{row['recall']:>10.4f}"
            f"{row['f1']:>10.4f}"
            f"{row['support']:>10d}"
        )

    total = sum(
        r["support"]
        for r in metrics["rows"]
    )

    lines.append("")
    lines.append(
        f"accuracy{'':>31}"
        f"{metrics['accuracy']:>10.4f}"
        f"{total:>10d}"
    )

    lines.append(
        f"macro avg{'':>27}"
        f"{metrics['macro_precision']:>10.4f}"
        f"{metrics['macro_recall']:>10.4f}"
        f"{metrics['macro_f1']:>10.4f}"
        f"{total:>10d}"
    )

    lines.append(
        f"weighted avg{'':>24}"
        f"{metrics['macro_precision']:>10.4f}"
        f"{metrics['macro_recall']:>10.4f}"
        f"{metrics['weighted_f1']:>10.4f}"
        f"{total:>10d}"
    )

    return "\n".join(lines)


# ============================================================
# HASH
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)

    return h.hexdigest()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("V3 57-INTENT ONNX BENCHMARK")
    print("=" * 78)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # File checks
    # --------------------------------------------------------

    for path, name in (
        (ONNX_PATH, "ONNX"),
        (CSV_PATH, "CSV"),
        (VOCAB_PATH, "vocab.json"),
        (LABELS_PATH, "labels.json"),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found:\n{path}"
            )

    print("\nONNX:")
    print(ONNX_PATH)

    print("\nCSV:")
    print(CSV_PATH)

    print("\nVocab:")
    print(VOCAB_PATH)

    print("\nLabels:")
    print(LABELS_PATH)

    # --------------------------------------------------------
    # Load vocab / labels
    # --------------------------------------------------------

    vocab = normalize_vocab(
        load_json(VOCAB_PATH)
    )

    labels = normalize_labels(
        load_json(LABELS_PATH)
    )

    print("\n--- MODEL METADATA ---")
    print(
        f"Vocabulary size: {len(vocab)}"
    )
    print(
        f"Labels         : {len(labels)}"
    )

    if len(vocab) != EXPECTED_VOCAB_SIZE:
        raise RuntimeError(
            f"Expected vocab size "
            f"{EXPECTED_VOCAB_SIZE}, "
            f"got {len(vocab)}."
        )

    if len(labels) != EXPECTED_CLASSES:
        raise RuntimeError(
            f"Expected {EXPECTED_CLASSES} labels, "
            f"got {len(labels)}."
        )

    # --------------------------------------------------------
    # ONNX session
    # --------------------------------------------------------

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=[
            "CPUExecutionProvider"
        ],
    )

    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]

    print("\n--- ONNX CONTRACT ---")

    print(
        "Input name :",
        input_meta.name,
    )
    print(
        "Input shape:",
        input_meta.shape,
    )
    print(
        "Input type :",
        input_meta.type,
    )
    print(
        "Output name:",
        output_meta.name,
    )
    print(
        "Output shape:",
        output_meta.shape,
    )
    print(
        "Output type :",
        output_meta.type,
    )

    if input_meta.name != "input_ids":
        raise RuntimeError(
            "Unexpected ONNX input name."
        )

    if input_meta.shape != [
        1,
        24,
    ]:
        raise RuntimeError(
            "Expected fixed ONNX input [1,24]."
        )

    if output_meta.shape != [
        1,
        57,
    ]:
        raise RuntimeError(
            "Expected fixed ONNX output [1,57]."
        )

    # --------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------

    rows = load_csv(
        CSV_PATH
    )

    print("\n--- DATASET ---")
    print(
        f"Rows: {len(rows)}"
    )

    # --------------------------------------------------------
    # Label mapping
    # --------------------------------------------------------

    label_to_id = {
        label: i
        for i, label in enumerate(
            labels
        )
    }

    unknown_labels = sorted(
        {
            row["label"]
            for row in rows
            if row["label"]
            not in label_to_id
        }
    )

    if unknown_labels:
        print(
            "\nUNKNOWN CSV LABELS:"
        )

        for label in unknown_labels:
            print(
                f"  {label}"
            )

        raise RuntimeError(
            "CSV contains labels not present "
            "in labels.json."
        )

    # --------------------------------------------------------
    # One-row-at-a-time inference
    # --------------------------------------------------------

    y_true = []
    y_pred = []
    confidences = []

    predictions = []

    start = time.perf_counter()

    for index, row in enumerate(
        rows,
        start=1,
    ):

        token_ids = tokenize(
            row["text"],
            vocab,
            EXPECTED_SEQ_LEN,
        )

        input_array = np.asarray(
            [token_ids],
            dtype=np.int64,
        )

        if input_array.shape != (
            1,
            24,
        ):
            raise RuntimeError(
                f"Bad input shape at row "
                f"{row['row_number']}: "
                f"{input_array.shape}"
            )

        logits = session.run(
            [output_meta.name],
            {
                input_meta.name:
                input_array
            },
        )[0]

        if logits.shape != (
            1,
            57,
        ):
            raise RuntimeError(
                f"Bad ONNX output shape: "
                f"{logits.shape}"
            )

        logits_1d = logits[0]

        # Stable softmax.
        shifted = (
            logits_1d
            - np.max(logits_1d)
        )

        exp = np.exp(
            shifted
        )

        probs = exp / np.sum(exp)

        pred_id = int(
            np.argmax(probs)
        )

        confidence = float(
            probs[pred_id]
        )

        true_id = label_to_id[
            row["label"]
        ]

        y_true.append(
            true_id
        )

        y_pred.append(
            pred_id
        )

        confidences.append(
            confidence
        )

        predictions.append(
            {
                "row_number":
                    row["row_number"],
                "text":
                    row["text"],
                "true_intent":
                    row["label"],
                "predicted_intent":
                    labels[pred_id],
                "true_id":
                    true_id,
                "predicted_id":
                    pred_id,
                "correct":
                    int(true_id == pred_id),
                "confidence":
                    confidence,
            }
        )

        if (
            index % 100 == 0
            or index == len(rows)
        ):
            print(
                f"Progress: "
                f"{index}/{len(rows)}"
            )

    elapsed = (
        time.perf_counter()
        - start
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = classification_metrics(
        y_true,
        y_pred,
        len(labels),
    )

    print(
        "\n--- ONNX TEST RESULT ---"
    )

    print(
        f"Accuracy : "
        f"{metrics['accuracy'] * 100:.4f}%"
    )

    print(
        f"Macro F1 : "
        f"{metrics['macro_f1'] * 100:.4f}%"
    )

    print(
        f"Weighted F1 : "
        f"{metrics['weighted_f1'] * 100:.4f}%"
    )

    report = format_report(
        labels,
        metrics,
    )

    print(
        "\nClassification report:"
    )
    print(report)

    # --------------------------------------------------------
    # Default Fallback
    # --------------------------------------------------------

    fallback_candidates = [
        "Default Fallback Intent",
        "default fallback intent",
        "defaultFallbackIntent",
        "default_fallback_intent",
        "default.fallback",
    ]

    fallback_id = None

    for candidate in fallback_candidates:

        if candidate in label_to_id:
            fallback_id = label_to_id[
                candidate
            ]
            break

    fallback_summary = None

    if fallback_id is not None:

        r = metrics["rows"][
            fallback_id
        ]

        fallback_summary = {
            "label": labels[
                fallback_id
            ],
            "id": fallback_id,
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "support": r["support"],
        }

        print(
            "\n--- DEFAULT FALLBACK ---"
        )

        print(
            "Intent    :",
            fallback_summary["label"],
        )

        print(
            "Precision :",
            f"{r['precision'] * 100:.2f}%",
        )

        print(
            "Recall    :",
            f"{r['recall'] * 100:.2f}%",
        )

        print(
            "F1        :",
            f"{r['f1'] * 100:.2f}%",
        )

        print(
            "Support   :",
            r["support"],
        )

    else:

        print(
            "\nDefault Fallback Intent "
            "not found in labels.json."
        )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    conf = np.asarray(
        confidences,
        dtype=np.float64,
    )

    print(
        "\n--- CONFIDENCE ---"
    )

    print(
        f"Mean   : {np.mean(conf):.6f}"
    )

    print(
        f"Median : {np.median(conf):.6f}"
    )

    print(
        f"Min    : {np.min(conf):.6f}"
    )

    print(
        f"P05    : {np.percentile(conf, 5):.6f}"
    )

    print(
        f"P25    : {np.percentile(conf, 25):.6f}"
    )

    print(
        f"P75    : {np.percentile(conf, 75):.6f}"
    )

    print(
        f"P95    : {np.percentile(conf, 95):.6f}"
    )

    print(
        f"Max    : {np.max(conf):.6f}"
    )

    # --------------------------------------------------------
    # Speed
    # --------------------------------------------------------

    print(
        "\n--- INFERENCE SPEED ---"
    )

    print(
        f"Total rows : {len(rows)}"
    )

    print(
        f"Total time : {elapsed:.4f} sec"
    )

    print(
        f"Rows/sec   : "
        f"{len(rows) / elapsed:.2f}"
    )

    print(
        f"ms/row     : "
        f"{elapsed * 1000 / len(rows):.4f}"
    )

    # --------------------------------------------------------
    # Save prediction details
    # --------------------------------------------------------

    with open(
        DETAILS_CSV,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row_number",
                "text",
                "true_intent",
                "predicted_intent",
                "true_id",
                "predicted_id",
                "correct",
                "confidence",
            ],
        )

        writer.writeheader()

        writer.writerows(
            predictions
        )

    REPORT_TXT.write_text(
        report,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = {
        "model": str(
            ONNX_PATH.resolve()
        ),
        "model_sha256": sha256_file(
            ONNX_PATH
        ),
        "dataset": str(
            CSV_PATH.resolve()
        ),
        "dataset_sha256": sha256_file(
            CSV_PATH
        ),
        "vocab_size": len(vocab),
        "num_classes": len(labels),
        "rows": len(rows),
        "input": {
            "name": input_meta.name,
            "shape": [1, 24],
            "dtype": "int64",
        },
        "output": {
            "name": output_meta.name,
            "shape": [1, 57],
            "dtype": "float32",
        },
        "accuracy": metrics["accuracy"],
        "macro_precision":
            metrics["macro_precision"],
        "macro_recall":
            metrics["macro_recall"],
        "macro_f1":
            metrics["macro_f1"],
        "weighted_f1":
            metrics["weighted_f1"],
        "fallback":
            fallback_summary,
        "confidence": {
            "mean": float(np.mean(conf)),
            "median": float(np.median(conf)),
            "min": float(np.min(conf)),
            "p05": float(np.percentile(conf, 5)),
            "p25": float(np.percentile(conf, 25)),
            "p75": float(np.percentile(conf, 75)),
            "p95": float(np.percentile(conf, 95)),
            "max": float(np.max(conf)),
        },
        "inference": {
            "total_seconds": elapsed,
            "rows_per_second":
                len(rows) / elapsed,
            "milliseconds_per_row":
                elapsed * 1000 / len(rows),
        },
        "benchmark_mode":
            "fixed [1,24] one-row-at-a-time",
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n--- SAVED ---"
    )

    print(
        DETAILS_CSV
    )

    print(
        REPORT_TXT
    )

    print(
        SUMMARY_JSON
    )

    print(
        "\n" + "=" * 78
    )

    print(
        "57-INTENT ONNX BENCHMARK COMPLETE"
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()
