#!/usr/bin/env python3
"""
Train intent classification model and export to ONNX.

Usage:
    python scripts/train.py              # default: version 3 (enhanced data)
    python scripts/train.py --version 1  # original files
    python scripts/train.py --version 2  # corrected _2 files
    python scripts/train.py --version 3  # enhanced: _2 + augmented phrases (default)
    python scripts/train.py -v 1

Reads:  data/intent_data_new[_2][_enhanced].csv
Writes: models/intent_model.onnx, models/intent_labels.pkl
"""

import argparse
import json as _json
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

# ---------- Args ----------
_parser = argparse.ArgumentParser(description="Train TF-IDF intent model")
_parser.add_argument("--version", "-v", type=int, choices=[1, 2, 3], default=3,
                     help="Data version: 1=original files, 2=corrected _2 files, "
                          "3=enhanced (_2 + augmented phrases, default)")
_args = _parser.parse_args()

# ---------- Paths ----------
BASE_DIR = Path(__file__).parent.parent
if _args.version == 1:
    DATA_PATH    = BASE_DIR / "data" / "intent_data_new.csv"
    HOLDOUT_PATH = BASE_DIR / "data" / "semantic_holdout_100.csv"
elif _args.version == 2:
    DATA_PATH    = BASE_DIR / "data" / "intent_data_new_2.csv"
    HOLDOUT_PATH = BASE_DIR / "data" / "semantic_holdout_2.csv"
else:
    # v3: corrected _2 data augmented with hand-written conversational
    # paraphrases via build_augmented_data.py. Holdout stays the v2 set so
    # the leakage guard below proves the augmentation never copies a test phrase.
    DATA_PATH    = BASE_DIR / "data" / "intent_data_new_2_enhanced.csv"
    HOLDOUT_PATH = BASE_DIR / "data" / "semantic_holdout_2.csv"

print(f"Data version: {_args.version}  |  {DATA_PATH.name}  |  holdout: {HOLDOUT_PATH.name}")
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

ONNX_PATH = MODELS_DIR / "intent_model.onnx"
LABELS_PATH = MODELS_DIR / "intent_labels.pkl"
LABELS_JSON_PATH = MODELS_DIR / "intent_labels.json"
PIPELINE_PATH = MODELS_DIR / "intent_pipeline.pkl"

# Accuracy regression gate. Gates on the TF-IDF model's TEST-SPLIT accuracy
# (held-back examples from the same distribution) — the standard generalization
# estimate for this stage. If it drops below the floor the build fails and
# nothing is exported, stopping a bad retrain from silently shipping.
#
# NOTE: the gate is NOT on the 100-utterance paraphrase holdout. That holdout
# is intentionally built to need the semantic stage, so TF-IDF ALONE scores
# ~0.50 on it — a pipeline metric, not a TF-IDF metric. The final model's
# holdout score is still computed and recorded in the manifest for visibility.
# Override via env: MIN_TEST_ACCURACY=0.80 python scripts/train.py
import os as _os
MIN_TEST_ACCURACY = float(_os.environ.get("MIN_TEST_ACCURACY", "0.85"))

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

# ---------- 1b. Guard: no holdout leakage ----------
if HOLDOUT_PATH.exists():
    holdout_raw = pd.read_csv(HOLDOUT_PATH, encoding="utf-8-sig", header=0)
    holdout_raw.columns = [c.strip().lower() for c in holdout_raw.columns]
    # Accept any common text-column name
    _text_col = next((c for c in holdout_raw.columns if c in ("text", "utterance", "query", "sentence", "phrase")), None)
    if _text_col is None:
        _text_col = holdout_raw.columns[0]
        print(f"  [holdout] no standard text column found, using first column: '{_text_col}'")
    holdout_texts = set(holdout_raw[_text_col].astype(str).str.lower().str.strip())
    leaked = set(data["text"]) & holdout_texts
    if leaked:
        raise RuntimeError(
            f"Holdout leakage detected — {len(leaked)} utterance(s) appear in both "
            f"training data and the permanent holdout set. Remove them from "
            f"intent_data_new.csv before retraining:\n  {sorted(leaked)[:5]}"
        )
    print(f"\nHoldout guard: 0 leaks detected ({len(holdout_texts)} holdout utterances checked).")

# ---------- 1c. Cap over-represented intents (deterministic keep-last) ----------
MAX_PER_INTENT = 500
data = (
    data.groupby("intent")
        .tail(MAX_PER_INTENT)          # keep newest rows when over cap; preserves all columns
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
# min_df=2 drops hapax legomena that inflate vocab without improving generalisation.
# CalibratedClassifierCV (isotonic) corrects the overconfident probabilities from C=15.
base_lr = LogisticRegression(max_iter=3000, class_weight="balanced", C=15.0)
calibrated_lr = CalibratedClassifierCV(base_lr, method="isotonic", cv=3)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )),
    ("clf", calibrated_lr)
])

# ---------- 5. Cross-validation ----------
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="accuracy")
print(f"\nCross-val accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")

# ---------- 6. Train on full training set ----------
pipeline.fit(X_train, y_train)

# ---------- 7. Evaluate on held-out test split + accuracy gate ----------
y_pred = pipeline.predict(X_test)
test_acc = float(np.mean(y_pred == y_test))
print(f"\nTest split accuracy (NOT the permanent holdout): {test_acc:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=labels))

if test_acc < MIN_TEST_ACCURACY:
    raise RuntimeError(
        f"Accuracy regression gate FAILED: test-split accuracy {test_acc:.2f} "
        f"is below the floor {MIN_TEST_ACCURACY:.2f}. Nothing was exported. "
        f"Investigate the training data before retrying (or lower "
        f"MIN_TEST_ACCURACY only with justification)."
    )
print(f"✅ Accuracy gate passed (test-split floor {MIN_TEST_ACCURACY:.2f}).")

# ---------- 8. Evaluate the TRAIN-SPLIT model on the permanent holdout ----------
# Note: this is the X_train model. The FINAL gate (section 9b) re-evaluates the
# exported model, which is trained on all data — that is the number that ships.
_holdout = None
if HOLDOUT_PATH.exists():
    hdf = pd.read_csv(HOLDOUT_PATH, encoding="utf-8-sig", header=0)
    hdf.columns = [c.strip().lower() for c in hdf.columns]
    _htext = next((c for c in hdf.columns if c in ("text", "utterance", "query", "sentence", "phrase")), hdf.columns[0])
    _hint  = next((c for c in hdf.columns if c in ("intent", "label", "class")), hdf.columns[1])
    hdf[_htext] = hdf[_htext].astype(str).str.lower().str.strip()
    hdf = hdf.dropna(subset=[_htext, _hint])
    _holdout = (hdf, _htext, _hint)
    h_pred = pipeline.predict(hdf[_htext])
    h_acc = np.mean(h_pred == hdf[_hint].str.strip())
    print(f"\n(train-split model) HOLDOUT accuracy: {h_acc:.2f} ({int(h_acc*len(hdf))}/{len(hdf)})")

# ---------- 9. Retrain on ALL data for final export ----------
pipeline.fit(X, y)

# ---------- 9b. Record the FINAL (exported) model's TF-IDF-only holdout ----------
# Informational + recorded in the manifest (fixes the "reported accuracy belongs
# to a discarded model" issue). This is the TF-IDF stage in isolation, so the
# number is low by design — end-to-end pipeline accuracy is validated separately
# by scripts/test_holdout.py.
final_holdout_acc = None
if _holdout is not None:
    hdf, _htext, _hint = _holdout
    f_pred = pipeline.predict(hdf[_htext])
    final_holdout_acc = float(np.mean(f_pred == hdf[_hint].str.strip()))
    n = len(hdf)
    print(f"\n(final exported model) TF-IDF-only HOLDOUT accuracy: "
          f"{final_holdout_acc:.2f} ({int(final_holdout_acc*n)}/{n}) "
          f"— pipeline accuracy is validated by test_holdout.py")

# ---------- 10. Export to ONNX ----------
initial_type = [("input", StringTensorType([None, 1]))]

onnx_model = convert_sklearn(
    pipeline,
    initial_types=initial_type,
    options={id(pipeline.named_steps["clf"]): {"zipmap": False}}
)

with open(ONNX_PATH, "wb") as f:
    f.write(onnx_model.SerializeToString())

with open(LABELS_JSON_PATH, "w") as f:
    _json.dump(labels, f, indent=2)

joblib.dump(pipeline, str(PIPELINE_PATH))

print(f"\n✅ Model exported to {ONNX_PATH}")
print(f"✅ Labels saved to {LABELS_PATH}")
print(f"✅ Labels JSON saved to {LABELS_JSON_PATH}")
print(f"✅ Pipeline saved to {PIPELINE_PATH}")
print(f"✅ Intent labels: {labels}")
print(f"✅ Model size: {ONNX_PATH.stat().st_size / 1024:.1f} KB")

# ---------- 11. Write model bundle manifest ----------
import sys
sys.path.insert(0, str(BASE_DIR / "scripts"))
from nlu.manifest import generate_manifest
_meta = {"holdout_accuracy": round(final_holdout_acc, 4)} if final_holdout_acc is not None else None
generate_manifest(BASE_DIR, meta=_meta)
print("✅ Manifest written to models/manifest.json")
