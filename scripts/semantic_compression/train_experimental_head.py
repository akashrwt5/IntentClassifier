#!/usr/bin/env python3
"""
Trains a Logistic Regression classification head on top of the newly generated
experimental compression models (Track 1 and Track 3) using the EXACT same
strategy, splits, and hyperparameters as packages/buildtime/nlu_training/train_semantic_head.py
"""

import sys
import pickle
import time
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from artifact import ArtifactContractError, encode, head_path, load_encoder

# --- Outside this directory -------------------------------------------------
# Data inputs only. Lifting this directory into another project means
# repointing these; no code is imported from outside.
DATA_PATH = BASE_DIR / "language_packs" / "en" / "train.csv"
OOS_PATH = BASE_DIR / "language_packs" / "en" / "extras" / "oos_2.csv"
HOLDOUT_PATH = BASE_DIR / "datasets" / "semantic_holdout_2.csv"

FALLBACK_INTENT = "Default Fallback Intent"

MODELS = {
    "Track 1 (Pruned L3)": Path(__file__).parent
    / "output_models"
    / "track1_pruned_l3"
    / "onnx_quantized"
    / "model_quantized.onnx",
    "Track 3 (SVD L6)": Path(__file__).parent
    / "output_models"
    / "track3_svd_l6"
    / "onnx_quantized"
    / "model_quantized.onnx",
    "Track 1 (Stage 1 Distilled)": Path(__file__).parent
    / "output_models"
    / "final_distilled_onnx"
    / "model_quantized.onnx",
    "Track 1 (Stage 2 Contrastive)": Path(__file__).parent
    / "output_models"
    / "stage2_contrastive_bge_small_onnx"
    / "model_quantized.onnx",
}


def get_training_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}")

    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    data.columns = [c.strip().lower() for c in data.columns]
    data["text"] = data["text"].astype(str).str.lower().str.strip()
    data["intent"] = data["intent"].astype(str).str.strip()
    data = data.dropna().drop_duplicates(subset=["text", "intent"])

    in_scope = data[data["intent"] != FALLBACK_INTENT]
    noisy_fb = data[data["intent"] == FALLBACK_INTENT]  # noisy eval only

    MAX_PER_INTENT = 250
    in_scope = (
        in_scope.groupby("intent")
        .tail(MAX_PER_INTENT)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    oos_texts = []
    if OOS_PATH.exists():
        oos = pd.read_csv(OOS_PATH, encoding="utf-8-sig")
        oos.columns = [c.strip().lower() for c in oos.columns]
        oos["text"] = oos["text"].astype(str).str.lower().str.strip()
        oos = oos.dropna(subset=["text"]).drop_duplicates(subset=["text"])

        if HOLDOUT_PATH.exists():
            hold = pd.read_csv(HOLDOUT_PATH, encoding="utf-8-sig")
            hold.columns = [c.strip().lower() for c in hold.columns]
            htext = next(
                (
                    c
                    for c in hold.columns
                    if c.strip().lower() in ("text", "utterance", "query", "sentence", "phrase")
                ),
                hold.columns[0],
            )
            hold_set = set(hold[htext].astype(str).str.lower().str.strip())
            oos = oos[~oos["text"].isin(hold_set)]
        oos_texts = oos["text"].tolist()

    texts = in_scope["text"].tolist() + oos_texts
    intents = in_scope["intent"].tolist() + [FALLBACK_INTENT] * len(oos_texts)

    return texts, intents, noisy_fb["text"].tolist()


def main():
    print("Loading datasets with Production Strategy...")
    texts, intents, noisy_fb_texts = get_training_data()
    print(f"Training phrases: {len(texts)}  Intents: {len(set(intents))}")

    for name, path in MODELS.items():
        print(f"\n{'='*50}")
        print(f"Training Head for {name} (Production Setup)")
        print(f"{'='*50}")

        if not path.exists():
            print(f"Skipping {name} (Model not built yet).")
            continue

        # Tokenizer, pooling and id-range are read from the artifact itself
        # (artifact.py), not chosen here. Three scripts used to choose them
        # independently and all three chose wrong for at least one export.
        try:
            encoder = load_encoder(path)
        except ArtifactContractError as exc:
            print(f"Skipping {name} -- artifact contract not satisfied:\n  {exc}")
            continue
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            continue
        print(f"   {encoder.summary()}")

        print("1. Generating Embeddings for training data...")
        X = []
        for i in range(0, len(texts), 500):
            X.append(encode(encoder, texts[i : i + 500]))
            print(f"   Embedded {min(i+500, len(texts))} / {len(texts)} samples")

        X = np.vstack(X)
        y = np.array(intents)

        # Exact same hold-out strategy
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

        print("2. Fitting Logistic Regression classifier (C=3.0, balanced)...")
        clf = LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced")
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        macro_f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)

        print(f"\n── Held-out evaluation (15% never seen during training) ──")
        print(f"  Overall accuracy: {acc:.3f}")
        print(f"  Macro-F1 score:   {macro_f1:.3f}")

        # Retrain on 100% of data just like the production script
        print("\n3. Retraining on full dataset...")
        final_clf = LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced")
        final_clf.fit(X, y)

        # Beside the model, not one level up: two exports share that parent
        # and used to overwrite each other's head.
        save_path = head_path(path)
        with open(save_path, "wb") as f:
            pickle.dump(final_clf, f)

        print(f"✅ Saved head to {save_path}")


if __name__ == "__main__":
    main()
