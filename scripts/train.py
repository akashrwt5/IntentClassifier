#!/usr/bin/env python3
"""
Train intent classification model and export to ONNX.

Usage:
    python scripts/train.py

Reads:  data/intent_data_new.csv
Writes: models/intent_model.onnx, models/intent_labels.pkl
"""

import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

# ---------- Paths ----------
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "intent_data_new.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

ONNX_PATH = MODELS_DIR / "intent_model.onnx"
LABELS_PATH = MODELS_DIR / "intent_labels.pkl"

# ---------- 1. Load & clean data ----------
data = pd.read_csv(DATA_PATH, encoding="utf-8-sig", header=0)

data.columns = [c.strip().lower() for c in data.columns]
data["text"] = data["text"].astype(str).str.lower().str.strip()
data["intent"] = data["intent"].astype(str).str.strip()   # preserve exact Dialogflow casing
data = data.dropna()
data = data.drop_duplicates(subset=["text", "intent"])

print(f"Total samples: {len(data)}")
print(f"Intents: {sorted(data['intent'].unique())}")
print(f"\nSamples per intent (raw):")
print(data["intent"].value_counts().to_string())

# ---------- 1b. Cap over-represented intents ----------
MAX_PER_INTENT = 500
data = (
    pd.concat([
        g.sample(min(len(g), MAX_PER_INTENT), random_state=42)
        for _, g in data.groupby("intent")
    ])
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)
print(f"\nSamples per intent (capped at {MAX_PER_INTENT}):")
print(data["intent"].value_counts().to_string())

X = data["text"]
y = data["intent"]

# ---------- 2. Train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# ---------- 3. Save intent labels ----------
labels = sorted(y.unique())
joblib.dump(labels, str(LABELS_PATH))

# ---------- 4. Build pipeline ----------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )),
    ("clf", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        C=15.0
    ))
])

# ---------- 5. Cross-validation ----------
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="accuracy")
print(f"\nCross-val accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")

# ---------- 6. Train on full training set ----------
pipeline.fit(X_train, y_train)

# ---------- 7. Evaluate on test set ----------
y_pred = pipeline.predict(X_test)
print(f"\nTest set accuracy: {np.mean(y_pred == y_test):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=labels))

# ---------- 8. Retrain on ALL data for final export ----------
pipeline.fit(X, y)

# ---------- 9. Export to ONNX ----------
initial_type = [("input", StringTensorType([None, 1]))]

onnx_model = convert_sklearn(
    pipeline,
    initial_types=initial_type,
    options={id(pipeline.named_steps["clf"]): {"zipmap": False}}
)

with open(ONNX_PATH, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"\n✅ Model exported to {ONNX_PATH}")
print(f"✅ Labels saved to {LABELS_PATH}")
print(f"✅ Intent labels: {labels}")
print(f"✅ Model size: {ONNX_PATH.stat().st_size / 1024:.1f} KB")
