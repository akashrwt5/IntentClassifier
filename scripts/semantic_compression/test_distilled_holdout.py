#!/usr/bin/env python3
"""
Test Distilled Model on Honest Holdout Set
==========================================
Evaluates the final 9 MB Stage 2 Contrastive distilled ONNX model and its
Logistic Regression classifier head against the purely unseen holdout dataset
(`language_packs/en/holdout_honest.csv`).
"""

import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, f1_score

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from transformers import AutoTokenizer
from packages.runtime.nlu_engine.inference import OrtEmbedderBackend, OrtIntentBackend

# Configuration
DISTILLED_MODEL_PATH = (
    Path(__file__).parent
    / "output_models"
    / "stage2_contrastive_bge_small_onnx"
    / "model_quantized.onnx"
)
DISTILLED_CLF_PATH = Path(__file__).parent / "output_models" / "classifier_head.pkl"
DISTILLED_TOKENIZER_PATH = (
    Path(__file__).parent / "output_models" / "stage2_contrastive_bge_small_onnx"
)

TFIDF_MODEL_PATH = BASE_DIR / "models" / "intent" / "en" / "model.onnx"
TFIDF_LABELS_PATH = BASE_DIR / "models" / "intent" / "en" / "labels.json"

HOLDOUT_CSV_PATH = BASE_DIR / "language_packs" / "en" / "holdout_honest.csv"


def embed_onnx(backend, texts, tokenizer):
    """Batch embed a list of texts using the ONNX backend."""
    vecs = []
    for text in texts:
        encoded = tokenizer(
            text, max_length=64, truncation=True, padding="max_length", return_tensors="np"
        )
        input_ids = encoded["input_ids"]
        # Clamp out of bounds IDs due to fast tokenizer mismatch with pruned matrix
        input_ids[input_ids >= 10000] = 100
        attention_mask = encoded["attention_mask"]
        token_type_ids = encoded.get("token_type_ids", np.zeros_like(input_ids))

        token_embeddings = backend.embed_tokens(input_ids, attention_mask, token_type_ids)

        # CLS pooling
        vec = token_embeddings[0]

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        vecs.append(vec)
    return np.array(vecs)


def evaluate_model(name, true_labels, predictions, total_time, n_samples):
    print(f"\n==================================================")
    print(f"Holdout Evaluation Results: {name}")
    print(f"==================================================")

    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="macro", zero_division=0)

    print(f"Total Test Phrases : {n_samples}")
    print(f"Overall Accuracy   : {acc:.4f}")
    print(f"Macro-F1 Score     : {f1:.4f}")
    print(f"Total Latency      : {total_time:.2f}s ({(total_time/n_samples)*1000:.2f} ms/query)")
    print(f"==================================================\n")


def main():
    if not HOLDOUT_CSV_PATH.exists():
        print(f"❌ Error: Holdout dataset not found at {HOLDOUT_CSV_PATH}")
        sys.exit(1)

    print(f"Loading Holdout Dataset: {HOLDOUT_CSV_PATH.name}")
    df = pd.read_csv(HOLDOUT_CSV_PATH, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    df["text"] = df["text"].astype(str).str.lower().str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()

    texts = df["text"].tolist()
    true_labels = df["intent"].tolist()
    print(f"Loaded {len(texts)} test phrases.\n")

    # ---------------------------------------------------------
    # 1. TF-IDF Evaluation
    # ---------------------------------------------------------
    print("Loading TF-IDF Baseline Model...")
    with open(TFIDF_LABELS_PATH, "r") as f:
        tfidf_labels = json.load(f)

    tfidf_backend = OrtIntentBackend(TFIDF_MODEL_PATH, n_labels=len(tfidf_labels))

    print("Evaluating TF-IDF Model...")
    tfidf_predictions = []
    start_time = time.time()
    for text in texts:
        logits = tfidf_backend.tfidf_logits(text)
        pred_idx = int(np.argmax(logits))
        tfidf_predictions.append(tfidf_labels[pred_idx])
    tfidf_time = time.time() - start_time

    evaluate_model("TF-IDF Baseline", true_labels, tfidf_predictions, tfidf_time, len(texts))

    # ---------------------------------------------------------
    # 2. Distilled Semantic Model Evaluation
    # ---------------------------------------------------------
    print("Loading 9 MB Distilled Semantic Model...")
    distilled_backend = OrtEmbedderBackend(DISTILLED_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(DISTILLED_TOKENIZER_PATH)
    with open(DISTILLED_CLF_PATH, "rb") as f:
        clf = pickle.load(f)

    print("Evaluating Distilled Model...")
    start_time = time.time()

    # Process in batches
    X = []
    batch_size = 500
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        X.append(embed_onnx(distilled_backend, batch_texts, tokenizer))

    X = np.vstack(X)
    probs = clf.predict_proba(X)
    distilled_predictions = [clf.classes_[np.argmax(p)] for p in probs]
    distilled_time = time.time() - start_time

    evaluate_model(
        "9 MB Distilled Semantic (Stage 2)",
        true_labels,
        distilled_predictions,
        distilled_time,
        len(texts),
    )


if __name__ == "__main__":
    main()
