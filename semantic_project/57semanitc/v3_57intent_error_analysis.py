#!/usr/bin/env python3

"""
V3 57-INTENT ERROR ANALYSIS

Purpose:
    Analyze the current 57-intent FP32 ONNX model without retraining.

Inputs:
    - train.csv
    - vocab.json
    - labels.json
    - v3_57intent_onnx/v3_semantic_student_57intent_fp32.onnx

Outputs:
    v3_57intent_error_analysis/
        error_analysis_summary.json
        per_intent_metrics.csv
        confusion_matrix.csv
        top_confusion_pairs.csv
        confusion_examples.csv
        low_confidence_errors.csv
        fallback_errors.csv

IMPORTANT:
    This uses the same simple V3 tokenizer convention:
        lowercase
        whitespace split
        strip punctuation
        unknown -> 1
        padding -> 0
        max length -> 24

    It does NOT train or modify the model.
"""

from pathlib import Path
import csv
import json
import hashlib

import numpy as np
import onnxruntime as ort


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/Users/shuklam/IntentClassifier/semantic_project/57semanitc"
)

CSV_PATH = ROOT / "train.csv"
VOCAB_PATH = ROOT / "vocab.json"
LABELS_PATH = ROOT / "labels.json"

ONNX_PATH = (
    ROOT
    / "v3_57intent_onnx"
    / "v3_semantic_student_57intent_fp32.onnx"
)

OUTPUT_DIR = ROOT / "v3_57intent_error_analysis"


# ============================================================
# CONTRACT
# ============================================================

VOCAB_SIZE = 895
NUM_CLASSES = 57
MAX_LEN = 24
PAD_ID = 0
UNK_ID = 1


# ============================================================
# JSON / HASH
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ============================================================
# VOCAB / LABELS
# ============================================================

def normalize_vocab(obj):
    if not isinstance(obj, dict):
        raise RuntimeError("Unsupported vocab.json format.")

    for key in ("token_to_id", "vocab", "stoi"):
        value = obj.get(key)
        if isinstance(value, dict):
            return value

    if all(isinstance(v, int) for v in obj.values()):
        return obj

    raise RuntimeError(
        "Could not find token -> id mapping in vocab.json."
    )


def normalize_labels(obj):
    if isinstance(obj, list):
        return [str(x) for x in obj]

    if not isinstance(obj, dict):
        raise RuntimeError("Unsupported labels.json format.")

    for key in ("labels", "id_to_label"):
        value = obj.get(key)

        if isinstance(value, list):
            return [str(x) for x in value]

        if isinstance(value, dict):
            pairs = []
            for k, v in value.items():
                try:
                    pairs.append((int(k), str(v)))
                except Exception:
                    pass
            if pairs:
                pairs.sort()
                return [v for _, v in pairs]

    for key in ("label_to_id", "intent_to_id"):
        value = obj.get(key)

        if isinstance(value, dict):
            pairs = [
                (int(v), str(k))
                for k, v in value.items()
            ]
            pairs.sort()
            return [v for _, v in pairs]

    raise RuntimeError(
        "Could not determine label ordering."
    )


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text, vocab):
    cleaned = (
        str(text)
        .lower()
        .strip()
    )

    tokens = []

    for token in cleaned.split():
        token = token.strip(
            ".,!?;:\"'()[]{}"
        )

        if token:
            tokens.append(token)

    ids = [
        int(vocab.get(token, UNK_ID))
        for token in tokens[:MAX_LEN]
    ]

    if len(ids) < MAX_LEN:
        ids += [
            PAD_ID
        ] * (
            MAX_LEN - len(ids)
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

    for x in text_candidates:
        if x in lowered:
            text_col = lowered[x]
            break

    for x in label_candidates:
        if x in lowered:
            label_col = lowered[x]
            break

    if text_col is None or label_col is None:
        raise RuntimeError(
            f"Could not detect text/intent columns. "
            f"Columns={fieldnames}"
        )

    return text_col, label_col


def load_rows(path):
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

        for row_num, row in enumerate(
            reader,
            start=2,
        ):

            text = str(
                row.get(text_col, "")
            ).strip()

            label = str(
                row.get(label_col, "")
            ).strip()

            if text and label:
                rows.append(
                    {
                        "row_number": row_num,
                        "text": text,
                        "label": label,
                    }
                )

    if not rows:
        raise RuntimeError(
            "No usable rows found."
        )

    return rows


# ============================================================
# SOFTMAX
# ============================================================

def softmax(logits):
    logits = np.asarray(
        logits,
        dtype=np.float64,
    )

    x = logits - np.max(logits)
    e = np.exp(x)

    return e / e.sum()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("V3 57-INTENT ERROR ANALYSIS")
    print("=" * 78)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    for path in (
        CSV_PATH,
        VOCAB_PATH,
        LABELS_PATH,
        ONNX_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    print("\nFiles:")
    print("CSV   :", CSV_PATH)
    print("VOCAB :", VOCAB_PATH)
    print("LABEL :", LABELS_PATH)
    print("ONNX  :", ONNX_PATH)

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    vocab = normalize_vocab(
        load_json(VOCAB_PATH)
    )

    labels = normalize_labels(
        load_json(LABELS_PATH)
    )

    if len(vocab) != VOCAB_SIZE:
        raise RuntimeError(
            f"Expected vocab {VOCAB_SIZE}, "
            f"got {len(vocab)}"
        )

    if len(labels) != NUM_CLASSES:
        raise RuntimeError(
            f"Expected labels {NUM_CLASSES}, "
            f"got {len(labels)}"
        )

    label_to_id = {
        x: i
        for i, x in enumerate(labels)
    }

    rows = load_rows(
        CSV_PATH
    )

    print(
        f"\nRows : {len(rows)}"
    )

    # --------------------------------------------------------
    # ONNX
    # --------------------------------------------------------

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=[
            "CPUExecutionProvider"
        ],
    )

    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]

    if inp.name != "input_ids":
        raise RuntimeError(
            f"Unexpected input: {inp.name}"
        )

    if inp.shape != [1, 24]:
        raise RuntimeError(
            f"Expected [1,24], got {inp.shape}"
        )

    if out.shape != [1, 57]:
        raise RuntimeError(
            f"Expected [1,57], got {out.shape}"
        )

    # --------------------------------------------------------
    # Arrays
    # --------------------------------------------------------

    y_true = []
    y_pred = []
    confidence = []

    details = []

    print(
        "\nRunning inference..."
    )

    for index, row in enumerate(
        rows,
        start=1,
    ):

        true_label = row["label"]

        if true_label not in label_to_id:
            raise RuntimeError(
                f"Unknown label at CSV row "
                f"{row['row_number']}: "
                f"{true_label}"
            )

        true_id = label_to_id[
            true_label
        ]

        token_ids = tokenize(
            row["text"],
            vocab,
        )

        x = np.asarray(
            [token_ids],
            dtype=np.int64,
        )

        logits = session.run(
            [out.name],
            {
                inp.name: x
            },
        )[0][0]

        probs = softmax(
            logits
        )

        pred_id = int(
            np.argmax(probs)
        )

        pred_label = labels[
            pred_id
        ]

        conf = float(
            probs[pred_id]
        )

        y_true.append(
            true_id
        )

        y_pred.append(
            pred_id
        )

        confidence.append(
            conf
        )

        details.append(
            {
                "row_number":
                    row["row_number"],
                "text":
                    row["text"],
                "true_intent":
                    true_label,
                "predicted_intent":
                    pred_label,
                "true_id":
                    true_id,
                "predicted_id":
                    pred_id,
                "correct":
                    int(
                        true_id == pred_id
                    ),
                "confidence":
                    conf,
            }
        )

        if (
            index % 500 == 0
            or index == len(rows)
        ):
            print(
                f"Progress: "
                f"{index}/{len(rows)}"
            )

    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.int64,
    )

    confidence = np.asarray(
        confidence,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES,
        ),
        dtype=np.int64,
    )

    for t, p in zip(
        y_true,
        y_pred,
    ):
        cm[t, p] += 1

    # --------------------------------------------------------
    # Per-intent metrics
    # --------------------------------------------------------

    per_intent = []

    for i, label in enumerate(labels):

        tp = int(cm[i, i])
        support = int(
            cm[i].sum()
        )

        predicted_total = int(
            cm[:, i].sum()
        )

        precision = (
            tp / predicted_total
            if predicted_total
            else 0.0
        )

        recall = (
            tp / support
            if support
            else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if precision + recall
            else 0.0
        )

        per_intent.append(
            {
                "intent": label,
                "id": i,
                "support": support,
                "correct": tp,
                "accuracy": recall,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predicted_total":
                    predicted_total,
            }
        )

    # --------------------------------------------------------
    # Top confusion pairs
    # --------------------------------------------------------

    pairs = []

    for true_id in range(
        NUM_CLASSES
    ):

        for pred_id in range(
            NUM_CLASSES
        ):

            if true_id == pred_id:
                continue

            count = int(
                cm[
                    true_id,
                    pred_id
                ]
            )

            if count <= 0:
                continue

            pairs.append(
                {
                    "true_intent":
                        labels[true_id],
                    "predicted_intent":
                        labels[pred_id],
                    "true_id":
                        true_id,
                    "predicted_id":
                        pred_id,
                    "count":
                        count,
                    "true_support":
                        int(
                            cm[true_id].sum()
                        ),
                    "rate_of_true_intent":
                        count
                        / max(
                            1,
                            int(
                                cm[true_id].sum()
                            )
                        ),
                }
            )

    pairs.sort(
        key=lambda x: (
            -x["count"],
            -x["rate_of_true_intent"],
        )
    )

    # --------------------------------------------------------
    # Confusion examples
    # --------------------------------------------------------

    examples = []

    for pair in pairs:

        t = pair["true_id"]
        p = pair["predicted_id"]

        matching = [
            x
            for x in details
            if (
                x["true_id"] == t
                and x["predicted_id"] == p
            )
        ]

        # Keep up to 10 examples per pair.
        for x in matching[:10]:

            examples.append(
                {
                    "true_intent":
                        x["true_intent"],
                    "predicted_intent":
                        x["predicted_intent"],
                    "count_for_pair":
                        pair["count"],
                    "confidence":
                        x["confidence"],
                    "row_number":
                        x["row_number"],
                    "text":
                        x["text"],
                }
            )

    # --------------------------------------------------------
    # Low-confidence errors
    # --------------------------------------------------------

    low_conf_errors = [
        x
        for x in details
        if x["correct"] == 0
    ]

    low_conf_errors.sort(
        key=lambda x:
            x["confidence"]
    )

    # --------------------------------------------------------
    # Fallback errors
    # --------------------------------------------------------

    fallback_names = {
        "Default Fallback Intent",
        "default fallback intent",
        "defaultFallbackIntent",
        "default_fallback_intent",
        "default.fallback",
    }

    fallback_label = None

    for label in labels:
        if label in fallback_names:
            fallback_label = label
            break

    fallback_errors = []

    if fallback_label:

        for x in details:

            if (
                x["true_intent"]
                == fallback_label
                and x["correct"] == 0
            ):

                fallback_errors.append(
                    x
                )

            elif (
                x["predicted_intent"]
                == fallback_label
                and x["correct"] == 0
            ):

                fallback_errors.append(
                    x
                )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    total = len(y_true)

    correct = int(
        np.sum(
            y_true == y_pred
        )
    )

    accuracy = (
        correct / total
    )

    macro_f1 = float(
        np.mean(
            [
                x["f1"]
                for x in per_intent
            ]
        )
    )

    weighted_f1 = float(
        sum(
            x["f1"] * x["support"]
            for x in per_intent
        )
        / total
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        "\n" + "=" * 78
    )

    print(
        "OVERALL"
    )

    print(
        "=" * 78
    )

    print(
        f"Accuracy   : "
        f"{accuracy * 100:.4f}%"
    )

    print(
        f"Macro F1   : "
        f"{macro_f1 * 100:.4f}%"
    )

    print(
        f"Weighted F1: "
        f"{weighted_f1 * 100:.4f}%"
    )

    print(
        "\n" + "=" * 78
    )

    print(
        "WEAKEST INTENTS BY F1"
    )

    print(
        "=" * 78
    )

    weakest = sorted(
        per_intent,
        key=lambda x: x["f1"],
    )

    for x in weakest[:15]:

        print(
            f"{x['intent']:<42} "
            f"F1={x['f1']*100:6.2f}% "
            f"Recall={x['recall']*100:6.2f}% "
            f"Support={x['support']}"
        )

    print(
        "\n" + "=" * 78
    )

    print(
        "TOP 30 CONFUSION PAIRS"
    )

    print(
        "=" * 78
    )

    for i, pair in enumerate(
        pairs[:30],
        start=1,
    ):

        print(
            f"{i:02d}. "
            f"{pair['true_intent']} "
            f"-> "
            f"{pair['predicted_intent']} "
            f"| count={pair['count']} "
            f"| rate="
            f"{pair['rate_of_true_intent']*100:.2f}%"
        )

    print(
        "\n" + "=" * 78
    )

    print(
        "LOW-CONFIDENCE ERRORS"
    )

    print(
        "=" * 78
    )

    for x in low_conf_errors[:30]:

        print(
            f"{x['confidence']:.4f} | "
            f"{x['true_intent']} -> "
            f"{x['predicted_intent']} | "
            f"{x['text']}"
        )

    # --------------------------------------------------------
    # Save per-intent
    # --------------------------------------------------------

    per_intent_path = (
        OUTPUT_DIR
        / "per_intent_metrics.csv"
    )

    with open(
        per_intent_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "intent",
                "id",
                "support",
                "correct",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "predicted_total",
            ],
        )

        writer.writeheader()
        writer.writerows(
            per_intent
        )

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    cm_path = (
        OUTPUT_DIR
        / "confusion_matrix.csv"
    )

    with open(
        cm_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            ["true\\pred"] + labels
        )

        for i, label in enumerate(
            labels
        ):

            writer.writerow(
                [label]
                + cm[i].tolist()
            )

    # --------------------------------------------------------
    # Save pairs
    # --------------------------------------------------------

    pairs_path = (
        OUTPUT_DIR
        / "top_confusion_pairs.csv"
    )

    with open(
        pairs_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        fieldnames = [
            "true_intent",
            "predicted_intent",
            "true_id",
            "predicted_id",
            "count",
            "true_support",
            "rate_of_true_intent",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            pairs
        )

    # --------------------------------------------------------
    # Save examples
    # --------------------------------------------------------

    examples_path = (
        OUTPUT_DIR
        / "confusion_examples.csv"
    )

    with open(
        examples_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        fieldnames = [
            "true_intent",
            "predicted_intent",
            "count_for_pair",
            "confidence",
            "row_number",
            "text",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            examples
        )

    # --------------------------------------------------------
    # Low confidence
    # --------------------------------------------------------

    low_path = (
        OUTPUT_DIR
        / "low_confidence_errors.csv"
    )

    with open(
        low_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        fieldnames = [
            "row_number",
            "text",
            "true_intent",
            "predicted_intent",
            "true_id",
            "predicted_id",
            "correct",
            "confidence",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            low_conf_errors
        )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    fallback_path = (
        OUTPUT_DIR
        / "fallback_errors.csv"
    )

    with open(
        fallback_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        fieldnames = [
            "row_number",
            "text",
            "true_intent",
            "predicted_intent",
            "true_id",
            "predicted_id",
            "correct",
            "confidence",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            fallback_errors
        )

    # --------------------------------------------------------
    # Full summary
    # --------------------------------------------------------

    summary = {
        "model": str(
            ONNX_PATH.resolve()
        ),
        "model_sha256":
            sha256_file(
                ONNX_PATH
            ),
        "dataset": str(
            CSV_PATH.resolve()
        ),
        "dataset_sha256":
            sha256_file(
                CSV_PATH
            ),
        "vocab_size":
            len(vocab),
        "num_classes":
            len(labels),
        "rows":
            total,
        "accuracy":
            accuracy,
        "macro_f1":
            macro_f1,
        "weighted_f1":
            weighted_f1,
        "incorrect_rows":
            int(total - correct),
        "confusion_pair_count":
            len(pairs),
        "weakest_15":
            weakest[:15],
        "top_30_confusion_pairs":
            pairs[:30],
        "fallback_label":
            fallback_label,
        "fallback_error_count":
            len(fallback_errors),
    }

    summary_path = (
        OUTPUT_DIR
        / "error_analysis_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print(
        "\n" + "=" * 78
    )

    print(
        "ERROR ANALYSIS COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        "\nSaved:"
    )

    print(per_intent_path)
    print(cm_path)
    print(pairs_path)
    print(examples_path)
    print(low_path)
    print(fallback_path)
    print(summary_path)

    print(
        "\nNO TRAINING WAS PERFORMED."
    )


if __name__ == "__main__":
    main()
