#!/usr/bin/env python3

"""
E5-SMALL-V2 — LOCKED 57-INTENT BENCHMARK
==========================================

Evaluates the already-trained E5-small-v2 + Logistic Regression
classifier on the existing LOCKED 57-intent test.

IMPORTANT:
- Does NOT retrain.
- Does NOT modify the classifier.
- Does NOT use the locked test for training.
- Does NOT quantize.
- Does NOT export ONNX.
- Uses the saved E5 classifier from the previous baseline run.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path("/Users/shuklam/IntentClassifier/semantic_project/57semanitc")

MODEL_DIR = ROOT / "v3_57intent_e5_small_v2"

CLASSIFIER_PATH = MODEL_DIR / "e5_logistic_classifier.joblib"

LABEL_MAP_PATH = MODEL_DIR / "label_map.json"

LOCKED_TEST = ROOT / "v3_57intent_locked_eval" / "locked_test_57intent.csv"

OUTPUT_DIR = ROOT / "v3_57intent_e5_small_v2_locked_benchmark"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_PATH = OUTPUT_DIR / "locked_classification_report.txt"

PREDICTIONS_PATH = OUTPUT_DIR / "locked_predictions.csv"

CONFUSION_PATH = OUTPUT_DIR / "confusion_matrix.csv"

SUMMARY_PATH = OUTPUT_DIR / "locked_benchmark_summary.json"

MODEL_NAME = "intfloat/e5-small-v2"


# ============================================================
# HELPERS
# ============================================================


def find_column(df, candidates):

    lookup = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:

        if candidate in lookup:
            return lookup[candidate]

    return None


def load_labels():

    with open(
        LABEL_MAP_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        obj = json.load(f)

    if "id_to_label" in obj:

        id_to_label = obj["id_to_label"]

        return [id_to_label[str(i)] for i in range(len(id_to_label))]

    if "label_to_id" in obj:

        label_to_id = obj["label_to_id"]

        return [
            label
            for label, _ in sorted(
                label_to_id.items(),
                key=lambda x: x[1],
            )
        ]

    raise RuntimeError("Could not find label mapping in label_map.json")


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 78)
    print("E5-SMALL-V2 — LOCKED 57-INTENT BENCHMARK")
    print("=" * 78)

    # --------------------------------------------------------
    # Verify files
    # --------------------------------------------------------

    required = [
        CLASSIFIER_PATH,
        LABEL_MAP_PATH,
        LOCKED_TEST,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(f"Missing required file:\n{path}")

    # --------------------------------------------------------
    # Load classifier + labels
    # --------------------------------------------------------

    print()
    print("Loading saved classifier...")

    classifier = joblib.load(CLASSIFIER_PATH)

    labels = load_labels()

    print(f"Classes: {len(labels)}")

    if len(labels) != 57:

        raise RuntimeError(f"Expected 57 labels, got {len(labels)}")

    # --------------------------------------------------------
    # Load locked test
    # --------------------------------------------------------

    df = pd.read_csv(LOCKED_TEST)

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

    if text_col is None:

        raise RuntimeError(f"Text column not found: {list(df.columns)}")

    if label_col is None:

        raise RuntimeError(f"Label column not found: {list(df.columns)}")

    texts = df[text_col].fillna("").astype(str).tolist()

    true_labels = df[label_col].astype(str).tolist()

    print()
    print(f"Locked test rows: {len(texts)}")

    # --------------------------------------------------------
    # Validate labels
    # --------------------------------------------------------

    unknown = sorted(set(true_labels) - set(labels))

    if unknown:

        raise RuntimeError(
            "Locked test contains labels absent " f"from the trained classifier:\n{unknown}"
        )

    # --------------------------------------------------------
    # Load E5
    # --------------------------------------------------------

    print()
    print(f"Loading embedding model: {MODEL_NAME}")

    encoder = SentenceTransformer(MODEL_NAME)

    # --------------------------------------------------------
    # Encode locked test
    # --------------------------------------------------------

    e5_texts = ["query: " + text for text in texts]

    print()
    print("Encoding locked test...")

    start_encode = time.perf_counter()

    X = encoder.encode(
        e5_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    encode_time = time.perf_counter() - start_encode

    print()
    print(f"Embedding shape: {X.shape}")

    # --------------------------------------------------------
    # Classifier inference
    # --------------------------------------------------------

    print()
    print("Running classifier inference...")

    start_inference = time.perf_counter()

    predictions = classifier.predict(X)

    inference_time = time.perf_counter() - start_inference

    probabilities = None

    if hasattr(classifier, "predict_proba"):

        probabilities = classifier.predict_proba(X)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        true_labels,
        predictions,
    )

    report = classification_report(
        true_labels,
        predictions,
        labels=labels,
        target_names=labels,
        zero_division=0,
    )

    cm = confusion_matrix(
        true_labels,
        predictions,
        labels=labels,
    )

    print()
    print("=" * 78)
    print("E5-SMALL-V2 LOCKED TEST RESULT")
    print("=" * 78)

    print()
    print(f"Accuracy   : {accuracy * 100:.4f}%")

    print(f"Rows       : {len(texts)}")

    print(f"Embedding time : {encode_time:.4f} sec")

    print(f"Classifier time: {inference_time:.4f} sec")

    print(f"Classifier rows/sec: " f"{len(texts) / inference_time:.2f}")

    print(f"Classifier ms/row: " f"{(inference_time / len(texts)) * 1000:.4f}")

    print()
    print("Classification report:")
    print(report)

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    result_df = pd.DataFrame(
        {
            "text": texts,
            "true_intent": true_labels,
            "predicted_intent": predictions,
            "correct": [
                a == b
                for a, b in zip(
                    true_labels,
                    predictions,
                )
            ],
        }
    )

    if probabilities is not None:

        max_confidence = probabilities.max(axis=1)

        result_df["confidence"] = max_confidence

    result_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    ).to_csv(CONFUSION_PATH)

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        f.write("E5-SMALL-V2 LOCKED 57-INTENT BENCHMARK\n")

        f.write("=" * 78 + "\n\n")

        f.write(f"Model: {MODEL_NAME}\n")

        f.write(f"Rows: {len(texts)}\n")

        f.write(f"Accuracy: {accuracy * 100:.4f}%\n")

        f.write(f"Embedding time: {encode_time:.6f} sec\n")

        f.write(f"Classifier time: {inference_time:.6f} sec\n")

        f.write(f"Classifier rows/sec: " f"{len(texts) / inference_time:.2f}\n")

        f.write(f"Classifier ms/row: " f"{(inference_time / len(texts)) * 1000:.6f}\n\n")

        f.write("Classification report:\n\n")

        f.write(report)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "status": "E5-SMALL-V2 LOCKED BENCHMARK COMPLETE",
        "model": MODEL_NAME,
        "classifier": str(CLASSIFIER_PATH),
        "locked_test": str(LOCKED_TEST),
        "rows": len(texts),
        "num_intents": len(labels),
        "accuracy": float(accuracy),
        "accuracy_percent": float(accuracy * 100),
        "embedding_time_seconds": float(encode_time),
        "classifier_inference_time_seconds": float(inference_time),
        "classifier_rows_per_second": float(len(texts) / inference_time),
        "classifier_ms_per_row": float((inference_time / len(texts)) * 1000),
        "retraining": False,
        "quantization": False,
        "onnx": False,
        "synthetic_text": False,
        "labels_changed": False,
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("STATUS: E5-SMALL-V2 LOCKED BENCHMARK COMPLETE")
    print("=" * 78)

    print()
    print("Saved:")
    print(PREDICTIONS_PATH)
    print(REPORT_PATH)
    print(CONFUSION_PATH)
    print(SUMMARY_PATH)

    print()
    print("IMPORTANT:")
    print("Training: NOT performed")
    print("Locked test: evaluation only")
    print("Quantization: NO")
    print("ONNX: NO")
    print("Synthetic text: NO")
    print("Labels changed: NO")


if __name__ == "__main__":
    main()
