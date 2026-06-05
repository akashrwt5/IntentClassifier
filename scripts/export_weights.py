#!/usr/bin/env python3
"""Export trained intent classifier weights to JSON for Swift on-device inference.

Usage:
    python scripts/export_weights.py

Reads:  data/intent_data_new.csv
Writes: models/intent_classifier_weights.json
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "intent_data_new.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
WEIGHTS_PATH = MODELS_DIR / "intent_classifier_weights.json"

data = pd.read_csv(DATA_PATH, encoding="utf-8", header=0)
data.columns = [c.strip().lower() for c in data.columns]
data["text"] = data["text"].astype(str).str.lower().str.strip()
data["intent"] = data["intent"].astype(str).str.strip()
data = data.dropna()

print(f"Total samples: {len(data)}")
print(f"Intents: {sorted(data['intent'].unique())}")

X, y = data["text"], data["intent"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(f"Test accuracy: {np.mean(y_pred == y_test):.3f}")
print(classification_report(y_test, y_pred))

pipeline.fit(X, y)
tfidf = pipeline.named_steps["tfidf"]
clf   = pipeline.named_steps["clf"]

weights = {
    "labels": clf.classes_.tolist(),
    "vocab": tfidf.vocabulary_,
    "idf": [round(x, 6) for x in tfidf.idf_.tolist()],
    "coef": [[round(x, 6) for x in row] for row in clf.coef_.tolist()],
    "intercept": [round(x, 6) for x in clf.intercept_.tolist()],
    "conf_threshold": 0.70,
    "conf_gap_threshold": 0.20,
    "genai_base_url": "https://genai.yourcompany.com/chat?query="
}

with open(WEIGHTS_PATH, "w") as f:
    json.dump(weights, f, separators=(',', ':'))

print(f"\n✅ Exported to {WEIGHTS_PATH} ({WEIGHTS_PATH.stat().st_size/1024:.1f} KB)")
print(f"✅ Labels: {weights['labels']}")
print(f"✅ Vocab size: {len(weights['vocab'])}")
