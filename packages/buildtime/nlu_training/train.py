#!/usr/bin/env python3
"""
Train a per-language intent model and export it to ONNX.

One language per run. There is no combined "multilingual" model: each Language
Pack carries its own model, so training is per language by construction.

Usage:
    python -m nlu_training.train              # English (default)
    python -m nlu_training.train --lang fr    # French, once datasets/fr/ exists

Reads:  datasets/<lang>/train.csv               (build-time input, never ships)
Writes: models/intent/<lang>/model.onnx         (mirrors the in-bundle layout,
        models/intent/<lang>/labels.pkl          so assemble_pack copies it
        models/intent/<lang>/labels.json         straight into the pack)
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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

from nlu_training.leakage import find_leaks, leak_report

# Shared surface-form normaliser (contraction expansion + apostrophe removal).
# MUST be the same function the runtime applies, or the exported ONNX and the
# in-memory model diverge on apostrophe inputs. It lives in the runtime package
# because inference ships it; training imports it so both agree. The path
# bootstrap that makes this importable lives once in nlu_training/__init__.py.
from nlu_engine.text_norm import normalize_text

# ---------- Args ----------
_parser = argparse.ArgumentParser(description="Train TF-IDF intent model")
_parser.add_argument("--lang", "-l", default="en",
                     help="language to train (default: en). Reads "
                          "language_packs/<lang>/train.csv — adding a language means "
                          "adding that directory, not editing this script.")
_args = _parser.parse_args()

# ---------- Paths ----------
# Per-language layout: language_packs/<lang>/. Training data is BUILD-TIME input and
# never ships — the pack records only its sha256 in bundle.json's `training`
# block. Adding a language is adding language_packs/<lang>/train.csv; this script does
# not learn about it.
BASE_DIR = Path(__file__).resolve().parents[3]
LANG = _args.lang
DATA_DIR     = BASE_DIR / "language_packs" / LANG
DATA_PATH    = DATA_DIR / "train.csv"
HOLDOUT_PATH = DATA_DIR / "holdout_leakage_guard.csv"

if not DATA_PATH.exists():
    available = sorted(p.name for p in (BASE_DIR / "language_packs").iterdir()
                       if p.is_dir() and not p.name.startswith("_")
                       and (p / "train.csv").exists())
    raise SystemExit(
        f"No training data for {LANG!r}: {DATA_PATH} not found.\n"
        f"Languages with data: {available or '(none)'}\n"
        f"To add one, create language_packs/{LANG}/train.csv (text,intent).")

print(f"Language: {LANG}  |  {DATA_PATH.relative_to(BASE_DIR)}  |  "
      f"holdout: {HOLDOUT_PATH.name}")

# Output mirrors the IN-BUNDLE layout (models/intent/<lang>/...) so
# assemble_pack copies the tree straight into the pack with no renaming. Model
# artifacts stay OUT of git; the pack is how they travel.
MODELS_DIR = BASE_DIR / "models" / "intent" / LANG
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ONNX_PATH = MODELS_DIR / "model.onnx"
LABELS_PATH = MODELS_DIR / "labels.pkl"
LABELS_JSON_PATH = MODELS_DIR / "labels.json"
PIPELINE_PATH = MODELS_DIR / "pipeline.pkl"

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
data["text"] = data["text"].astype(str).map(normalize_text)  # lower+strip+contractions
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
    holdout_texts = holdout_raw[_text_col].astype(str).map(normalize_text).tolist()
    # NORMALISED comparison (case, punctuation, spacing). A raw-string compare
    # missed any pair differing only by a trailing '?' — which is how a
    # 99.9%-leaked English holdout passed this guard for so long (Review-F5
    # blocker B9). One shared definition lives in leakage.py so train.py, the
    # calibration fitter and holdout construction cannot drift apart.
    leaked = find_leaks(data["text"], holdout_texts)
    if leaked:
        raise RuntimeError(
            leak_report(leaked, len(holdout_texts), source="permanent holdout")
            + f"\nRemove them from {DATA_PATH.name} before retraining."
        )
    print(f"\n{leak_report([], len(holdout_texts), source='permanent holdout')}")

# ---------- 1c. Per-intent row cap (DISABLED) ----------
# `None` = no cap. It was 500, applied with `.tail()`, which DELETED rows — they
# reached neither training nor the test split. `Default Fallback Intent` went 1191 ->
# 500, and a catch-all class is defined by its LEXICAL VARIETY rather than by a
# pattern, so cutting 58% of it cut 58% of the evidence that anything is out of
# scope. That is where out-of-scope detection lives: the word which makes an
# utterance out of scope is a rare, specific one, so it is the first thing a cap
# removes and then `min_df=2` finishes off.
#
# Measured on holdout_honest.csv: uncapping moves OOS recall 69.2% -> 81.0% and
# accuracy 90.5% -> 92.0%. Both improve — `class_weight="balanced"` below was
# already doing the imbalance job the cap was for, and doing it without
# destroying data.
#
# This is NOT the train/test split. Step 2 below is unaffected and still holds
# out 20%; `holdout_honest.csv` remains a separate leak-guarded file (step 1b).
# The cap only decided which rows exist at all.
#
# The value lives in `nlu_training.fit_calibration` so the trainer, the fitters
# and the exporters cannot drift apart.
from nlu_training.fit_calibration import MAX_PER_INTENT, cap_per_intent  # noqa: E402
data = cap_per_intent(data).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"\nSamples per intent (cap={MAX_PER_INTENT}):")
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
# Plain LogisticRegression: the over-confident C=15 probabilities are calibrated
# downstream by single-parameter temperature scaling (one scalar T fit on the
# device-equivalent logits by scripts/export_ios_weights.py), NOT by wrapping the LR
# in CalibratedClassifierCV. A plain LR is required so the exported ONNX can emit raw
# decision-function logits (raw_scores=True, see export below) for the server to scale.
# NOTE: word analyzer only. A word+char_wb FeatureUnion scores ~3pts higher on the
# hard paraphrase holdout, but skl2onnx cannot export char-analyzer TfidfVectorizers
# (the engine/iOS run the ONNX model), so it is not deployable here. The semantic
# (MiniLM) stage already covers sub-word generalisation at runtime.
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )),
    ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=15.0))
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
    hdf[_htext] = hdf[_htext].astype(str).map(normalize_text)  # same norm as train
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
# raw_scores=True exposes the raw decision_function logits (NOT a baked-in softmax)
# as the 'probabilities' output, so the server/device can apply softmax(logits / T)
# with the temperature T fitted at export time. Dividing softmaxed probabilities by T
# would be mathematically wrong, hence the plain-LR + raw_scores combination.
initial_type = [("input", StringTensorType([None, 1]))]

onnx_model = convert_sklearn(
    pipeline,
    initial_types=initial_type,
    options={id(pipeline.named_steps["clf"]): {"zipmap": False, "raw_scores": True}}
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
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[3] / 'packages' / 'runtime'))
from nlu_engine.manifest import generate_manifest
_meta = {"holdout_accuracy": round(final_holdout_acc, 4)} if final_holdout_acc is not None else None
generate_manifest(BASE_DIR, meta=_meta)
print("✅ Manifest written to models/manifest.json")
