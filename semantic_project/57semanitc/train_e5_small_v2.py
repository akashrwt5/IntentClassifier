#!/usr/bin/env python3
"""
E5-SMALL-V2 BASELINE
====================

Purpose:
    Train a clean 57-intent classifier using E5-small-v2 embeddings.

Input:
    /Users/shuklam/IntentClassifier/semantic_project/57semanitc/train.csv

Expected columns:
    text
    intent

Important:
    - Locked 57-intent test is NOT read.
    - No quantization.
    - No ONNX.
    - No synthetic text.
    - No label changes.
    - Stratified train/validation split only.
    - E5-small-v2 is used as a frozen embedding model.
    - Logistic Regression is used as the intent classifier.

Output:
    v3_57intent_e5_small_v2/
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/Users/shuklam/IntentClassifier/semantic_project/57semanitc"
)

TRAIN_CSV = ROOT / "train.csv"

OUT = ROOT / "v3_57intent_e5_small_v2"
OUT.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_X = OUT / "e5_train_embeddings.npy"
LABELS_Y = OUT / "e5_train_labels.npy"
CLASSIFIER = OUT / "e5_logistic_classifier.joblib"
LABEL_MAP = OUT / "label_map.json"
REPORT = OUT / "validation_report.txt"
SUMMARY = OUT / "training_summary.json"
CONFUSION = OUT / "confusion_matrix.csv"

MODEL_NAME = "intfloat/e5-small-v2"

TEST_SIZE = 0.20
RANDOM_STATE = 42

# Start with a strong linear baseline.
# C can later be tuned, but do not tune on the locked test.
LOGREG_C = 4.0


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("E5-SMALL-V2 — 57 INTENT BASELINE")
    print("=" * 78)

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"Missing training CSV:\n{TRAIN_CSV}"
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(TRAIN_CSV)

    required = {"text", "intent"}

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing required columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df[["text", "intent"]].copy()

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["intent"] = df["intent"].fillna("").astype(str).str.strip()

    df = df[
        (df["text"] != "")
        & (df["intent"] != "")
    ].reset_index(drop=True)

    print()
    print(f"Dataset rows : {len(df)}")
    print(f"Intent count : {df['intent'].nunique()}")

    counts = df["intent"].value_counts()

    print()
    print("Intent distribution:")
    print(counts.to_string())

    if df["intent"].nunique() != 57:
        raise RuntimeError(
            f"Expected 57 intents, found "
            f"{df['intent'].nunique()}"
        )

    # --------------------------------------------------------
    # Stratified split
    # --------------------------------------------------------

    train_df, val_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["intent"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print()
    print(f"Train rows : {len(train_df)}")
    print(f"Val rows   : {len(val_df)}")

    # --------------------------------------------------------
    # Load E5
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("Loading E5-small-v2")
    print("=" * 78)

    print(f"Model: {MODEL_NAME}")

    model = SentenceTransformer(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # E5 encoding
    # --------------------------------------------------------
    # Official E5 usage for retrieval-style text embeddings
    # uses the "query: " prefix. We use it consistently for
    # every utterance in this classifier baseline.
    # --------------------------------------------------------

    train_texts = [
        "query: " + text
        for text in train_df["text"].tolist()
    ]

    val_texts = [
        "query: " + text
        for text in val_df["text"].tolist()
    ]

    print()
    print("Encoding training data...")

    start = time.perf_counter()

    X_train = model.encode(
        train_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    train_encode_time = (
        time.perf_counter() - start
    )

    print()
    print("Encoding validation data...")

    start = time.perf_counter()

    X_val = model.encode(
        val_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    val_encode_time = (
        time.perf_counter() - start
    )

    print()
    print(f"Embedding shape: {X_train.shape}")
    print(f"Embedding dim  : {X_train.shape[1]}")
    print(
        f"Train encoding : "
        f"{train_encode_time:.3f} sec"
    )
    print(
        f"Val encoding   : "
        f"{val_encode_time:.3f} sec"
    )

    # Save embeddings for reproducibility.
    np.save(
        EMBEDDINGS_X,
        np.vstack([X_train, X_val])
    )

    # --------------------------------------------------------
    # Label mapping
    # --------------------------------------------------------

    labels = sorted(
        df["intent"].unique().tolist()
    )

    label_to_id = {
        label: i
        for i, label in enumerate(labels)
    }

    id_to_label = {
        str(i): label
        for label, i in label_to_id.items()
    }

    y_train = np.asarray([
        label_to_id[x]
        for x in train_df["intent"]
    ])

    y_val = np.asarray([
        label_to_id[x]
        for x in val_df["intent"]
    ])

    np.save(
        LABELS_Y,
        np.concatenate([y_train, y_val])
    )

    LABEL_MAP.write_text(
        json.dumps(
            {
                "label_to_id": label_to_id,
                "id_to_label": id_to_label,
                "num_classes": len(labels),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Logistic Regression classifier
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("Training Logistic Regression classifier")
    print("=" * 78)

    print(f"C = {LOGREG_C}")
    print("class_weight = balanced")

    classifier = LogisticRegression(
        C=LOGREG_C,
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )

    start = time.perf_counter()

    classifier.fit(
        X_train,
        y_train,
    )

    train_time = (
        time.perf_counter() - start
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("E5-SMALL-V2 VALIDATION")
    print("=" * 78)

    start = time.perf_counter()

    pred = classifier.predict(
        X_val
    )

    inference_time = (
        time.perf_counter() - start
    )

    accuracy = accuracy_score(
        y_val,
        pred,
    )

    report = classification_report(
        y_val,
        pred,
        target_names=labels,
        zero_division=0,
    )

    print()
    print(
        f"Accuracy : {accuracy * 100:.4f}%"
    )

    print()
    print(report)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_val,
        pred,
    )

    pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    ).to_csv(
        CONFUSION
    )

    # --------------------------------------------------------
    # Save classifier
    # --------------------------------------------------------

    import joblib

    joblib.dump(
        classifier,
        CLASSIFIER
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    with open(
        REPORT,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "E5-SMALL-V2 57-INTENT BASELINE\n"
        )

        f.write("=" * 78 + "\n\n")

        f.write(
            f"Model: {MODEL_NAME}\n"
        )

        f.write(
            f"Dataset: {TRAIN_CSV}\n"
        )

        f.write(
            f"Total rows: {len(df)}\n"
        )

        f.write(
            f"Intents: {len(labels)}\n"
        )

        f.write(
            f"Train rows: {len(train_df)}\n"
        )

        f.write(
            f"Validation rows: {len(val_df)}\n"
        )

        f.write(
            f"Embedding dimension: "
            f"{X_train.shape[1]}\n"
        )

        f.write(
            f"Accuracy: {accuracy * 100:.4f}%\n"
        )

        f.write(
            f"Train time: {train_time:.4f} sec\n"
        )

        f.write(
            f"Validation inference time: "
            f"{inference_time:.4f} sec\n"
        )

        f.write("\nClassification report:\n\n")
        f.write(report)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "status":
            "E5-SMALL-V2 BASELINE COMPLETE",

        "model":
            MODEL_NAME,

        "dataset":
            str(TRAIN_CSV),

        "total_rows":
            int(len(df)),

        "num_intents":
            int(len(labels)),

        "train_rows":
            int(len(train_df)),

        "validation_rows":
            int(len(val_df)),

        "embedding_dimension":
            int(X_train.shape[1]),

        "accuracy":
            float(accuracy),

        "accuracy_percent":
            float(accuracy * 100),

        "logistic_regression_C":
            LOGREG_C,

        "class_weight":
            "balanced",

        "locked_test_used":
            False,

        "quantization":
            False,

        "onnx":
            False,

        "synthetic_text":
            False,

        "labels_changed":
            False,

        "train_encoding_seconds":
            float(train_encode_time),

        "validation_encoding_seconds":
            float(val_encode_time),

        "classifier_training_seconds":
            float(train_time),

        "validation_inference_seconds":
            float(inference_time),
    }

    SUMMARY.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("E5-SMALL-V2 BASELINE COMPLETE")
    print("=" * 78)

    print()
    print(
        f"Validation Accuracy : "
        f"{accuracy * 100:.4f}%"
    )

    print()
    print("Saved:")

    print(EMBEDDINGS_X)
    print(LABELS_Y)
    print(CLASSIFIER)
    print(LABEL_MAP)
    print(REPORT)
    print(CONFUSION)
    print(SUMMARY)

    print()
    print("IMPORTANT:")
    print("Locked 57-intent test: NOT USED")
    print("Quantization: NO")
    print("ONNX: NO")
    print("Synthetic text: NO")
    print("Labels changed: NO")


if __name__ == "__main__":
    main()
