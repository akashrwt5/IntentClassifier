#!/usr/bin/env python3
"""Train on the full Dialogflow-imported dataset and export weights for Swift."""

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
WEIGHTS_PATH = BASE_DIR / "models" / "intent_classifier_weights.json"
BASE_DIR.joinpath("models").mkdir(exist_ok=True)

data = pd.read_csv(DATA_PATH, encoding="utf-8-sig", header=0)
data.columns = [c.strip().lower() for c in data.columns]
data["text"] = data["text"].astype(str).str.lower().str.strip()
data["intent"] = data["intent"].astype(str).str.upper().str.strip()
data = data.dropna().drop_duplicates(subset=["text", "intent"])

print(f"Total samples: {len(data)}")
print(f"Intents: {sorted(data['intent'].unique())}")

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

X, y = data["text"], data["intent"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=15.0))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(f"\nTest accuracy: {np.mean(y_pred == y_test):.3f}")
print(classification_report(y_test, y_pred))

# Retrain on all data then export
pipeline.fit(X, y)
tfidf = pipeline.named_steps["tfidf"]
clf   = pipeline.named_steps["clf"]

weights = {
    "labels": clf.classes_.tolist(),
    "vocab": {k: int(v) for k, v in tfidf.vocabulary_.items()},
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