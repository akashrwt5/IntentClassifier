#!/usr/bin/env python3
"""
Multilingual TF-IDF intent model generator.

This is a SELF-CONTAINED sibling of scripts/train.py. It never imports from or
modifies scripts/train.py — it only mirrors that script's proven pipeline design
(TF-IDF word 1-2 grams, capped classes, stratified split, accuracy gate) so the
multilingual models behave consistently with the production English model.

It can build:
  * one model per language (en / fr / de / ...), and
  * one combined "multilingual" model trained on every registered language
    (exported as multilingual_intent_model.onnx).

Adding a new language is one line in the LANGUAGES registry below — the design
target is "any number of languages".

────────────────────────────────────────────────────────────────────────────
Usage
────────────────────────────────────────────────────────────────────────────
    # build every per-language model AND the combined multilingual model
    python multilingual/train_multilingual.py --all

    # build a single language from its registered data file
    python multilingual/train_multilingual.py --language fr

    # build a single language from an explicit data file (overrides registry)
    python multilingual/train_multilingual.py --language fr --data path/to/fr.csv

    # build only the combined model
    python multilingual/train_multilingual.py --language multilingual

    # tune the gate / cap
    python multilingual/train_multilingual.py --all --min-accuracy 0.75
    MIN_TEST_ACCURACY=0.75 python multilingual/train_multilingual.py --all

Input data format (every file): a CSV with header `text,intent`.

────────────────────────────────────────────────────────────────────────────
Outputs  (one self-contained folder per model under multilingual/models/)
────────────────────────────────────────────────────────────────────────────
    multilingual/models/<name>/
        <name>_intent_model.onnx                ONNX model (onnxruntime / server path)
        <name>_intent_pipeline.pkl              fitted sklearn Pipeline
        <name>_intent_labels.json               label list (JSON)
        <name>_intent_labels.pkl                label list (joblib)
        <name>_intent_classifier_weights.json   raw TF-IDF + LR weights (on-device path)
        manifest.json                           SHA-256 of every artifact above
    multilingual/test/<name>_holdout.csv        deterministic held-out test split

────────────────────────────────────────────────────────────────────────────
⚠️  DEFERRED — iOS / Core ML / Swift parity  (see multilingual/README.md)
────────────────────────────────────────────────────────────────────────────
The current iOS Stage-2 intent classifier is built from
intent_classifier_weights.json, NOT from the .onnx file (coremltools cannot
convert the calibrated sklearn pipeline or the ONNX string-input TF-IDF
subgraph — see scripts/export_coreml.py). The production server model uses
CalibratedClassifierCV(isotonic); this multilingual script intentionally uses a
PLAIN (uncalibrated) LogisticRegression so a clean classifier_weights.json can
be emitted. The per-class isotonic calibration tables, per-language Core ML
(.mlpackage) export, and multilingual-vocab handling are NOT done here yet and
must be handled later for full Swift/Core ML parity.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

sys.path.insert(0, str(Path(__file__).parent))
from text_norm import normalize_text   # shared lowercase + ASCII accent-folding

# ───────────────────────── Paths & registry ──────────────────────────────────
BASE_DIR = Path(__file__).parent              # .../multilingual
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
TEST_DIR = BASE_DIR / "test"

# Add a language = add one line here. The combined "multilingual" model is the
# concatenation of every entry in this registry.
LANGUAGES = {
    "en": DATA_DIR / "en.csv",   # data/04_GENERATED_MASTER_training_data.csv
    "fr": DATA_DIR / "fr.csv",   # data/Generated_Master_training_French_Data.csv
    "de": DATA_DIR / "de.csv",   # pva_intent_german.csv master (promoted from pending/)
    "da": DATA_DIR / "da.csv",   # pva_intent_danish.csv master (promoted from pending/)
}

COMBINED_NAME = "multilingual"   # -> multilingual_intent_model.onnx

# Class cap and gate mirror scripts/train.py. The gate default is slightly lower
# than train.py's 0.85 because some language sets (e.g. German: many intents,
# fewer rows each) generalise less; override with --min-accuracy / MIN_TEST_ACCURACY.
MAX_PER_INTENT = 500
DEFAULT_MIN_ACCURACY = float(os.environ.get("MIN_TEST_ACCURACY", "0.80"))


# ───────────────────────── Data helpers ──────────────────────────────────────
def load_clean(path: Path) -> pd.DataFrame:
    """Load and normalise a `text,intent` CSV the same way train.py does."""
    df = pd.read_csv(path, encoding="utf-8-sig", header=0)
    df.columns = [c.strip().lower() for c in df.columns]
    if "text" not in df.columns or "intent" not in df.columns:
        raise ValueError(
            f"{path} must have 'text' and 'intent' columns; found {list(df.columns)}"
        )
    df["text"] = df["text"].astype(str).map(normalize_text)   # lowercase + ASCII fold
    df["intent"] = df["intent"].astype(str).str.strip()   # preserve Dialogflow casing
    df = df.dropna(subset=["text", "intent"])
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset=["text", "intent"])
    return df


def cap_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic keep-last cap, identical strategy to train.py."""
    return (
        df.groupby("intent")
        .tail(MAX_PER_INTENT)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )


def load_data_for(name: str, explicit: Path | None) -> pd.DataFrame:
    """Resolve the training frame for a model name.

    - explicit --data wins.
    - COMBINED_NAME concatenates every registered language.
    - otherwise look the name up in LANGUAGES.
    """
    if explicit is not None:
        print(f"  [{name}] using explicit data file: {explicit}")
        return load_clean(explicit)

    if name == COMBINED_NAME:
        frames = []
        for lang, path in LANGUAGES.items():
            f = load_clean(path)
            f["__lang__"] = lang
            frames.append(f)
            print(f"  [{name}] + {lang}: {len(f)} rows from {path.name}")
        combined = pd.concat(frames, ignore_index=True)
        # Drop the helper column before training; it was only for the log above.
        return combined.drop(columns="__lang__")

    if name not in LANGUAGES:
        raise SystemExit(
            f"Unknown language '{name}'. Known: {sorted(LANGUAGES)} "
            f"(or '{COMBINED_NAME}'). Pass --data to use a custom file."
        )
    return load_clean(LANGUAGES[name])


# ───────────────────────── Export helpers ────────────────────────────────────
def export_weights_json(pipeline: Pipeline, labels, out_path: Path):
    """Dump raw TF-IDF + LR weights for the on-device (Swift) inference path.

    Mirrors scripts/export_weights.py. Requires a PLAIN LogisticRegression
    (CalibratedClassifierCV would not expose .coef_). The isotonic calibration
    tables used by the production iOS build are NOT emitted here — see the
    DEFERRED note in this file's docstring.
    """
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    weights = {
        "labels": clf.classes_.tolist(),
        "vocab": {k: int(v) for k, v in tfidf.vocabulary_.items()},
        "idf": [round(x, 6) for x in tfidf.idf_.tolist()],
        "coef": [[round(x, 6) for x in row] for row in clf.coef_.tolist()],
        "intercept": [round(x, 6) for x in clf.intercept_.tolist()],
        "ngram_range": list(tfidf.ngram_range),
        "sublinear_tf": bool(tfidf.sublinear_tf),
        "conf_threshold": 0.70,
        "conf_gap_threshold": 0.20,
        "calibration": None,   # DEFERRED: isotonic tables for Swift parity
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(weights, f, separators=(",", ":"))


def write_manifest(model_dir: Path):
    """SHA-256 every artifact in the model folder (excluding the manifest itself)."""
    manifest = {}
    for p in sorted(model_dir.iterdir()):
        if p.name == "manifest.json" or not p.is_file():
            continue
        manifest[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    with open(model_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# ───────────────────────── Train one model ───────────────────────────────────
def train_one(name: str, df: pd.DataFrame, min_accuracy: float) -> bool:
    """Train, gate, and export a single model. Returns True on success."""
    print(f"\n{'='*70}\nMODEL: {name}\n{'='*70}")

    df = cap_classes(df)
    print(f"Samples: {len(df)} | Intents: {df['intent'].nunique()}")

    # Need at least 2 examples per class for a stratified split.
    counts = df["intent"].value_counts()
    too_few = counts[counts < 2]
    if len(too_few):
        print(f"  [warn] dropping {len(too_few)} intent(s) with <2 samples: "
              f"{too_few.index.tolist()[:8]}{'...' if len(too_few) > 8 else ''}")
        df = df[df["intent"].isin(counts[counts >= 2].index)].reset_index(drop=True)

    X, y = df["text"], df["intent"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Plain (uncalibrated) LR so coef_/idf_ are exportable to classifier_weights.json.
    # Word analyzer only — skl2onnx cannot export char-analyzer TfidfVectorizers.
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=15.0)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    test_acc = float(np.mean(y_pred == y_test))
    print(f"\nTest-split accuracy: {test_acc:.3f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    if test_acc < min_accuracy:
        print(f"❌ Accuracy gate FAILED for '{name}': {test_acc:.3f} < {min_accuracy:.2f}. "
              f"Nothing exported for this model.")
        return False
    print(f"✅ Accuracy gate passed ({test_acc:.3f} >= {min_accuracy:.2f}).")

    # Persist the held-out split so the test script has real eval data.
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    holdout = pd.DataFrame({"text": X_test, "intent": y_test})
    holdout.to_csv(TEST_DIR / f"{name}_holdout.csv", index=False, encoding="utf-8")

    # NOTE: we export the model fitted on X_train only — we deliberately do NOT
    # refit on all data here. The test split written above must stay genuinely
    # unseen by the exported model, otherwise test_multilingual_models.py would
    # score training data (leakage) and report inflated accuracy. train.py can
    # refit on all data because it validates against a SEPARATE permanent holdout
    # file; we have no such per-language file, so the train/test split is our
    # holdout and must not leak into the shipped model.
    labels = pipeline.named_steps["clf"].classes_.tolist()

    model_dir = MODELS_DIR / name
    model_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = model_dir / f"{name}_intent_model.onnx"
    if name == COMBINED_NAME:
        onnx_path = model_dir / "multilingual_intent_model.onnx"

    onnx_model = convert_sklearn(
        pipeline,
        initial_types=[("input", StringTensorType([None, 1]))],
        options={id(pipeline.named_steps["clf"]): {"zipmap": False}},
    )
    onnx_path.write_bytes(onnx_model.SerializeToString())

    joblib.dump(pipeline, str(model_dir / f"{name}_intent_pipeline.pkl"))
    joblib.dump(labels, str(model_dir / f"{name}_intent_labels.pkl"))
    with open(model_dir / f"{name}_intent_labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    export_weights_json(pipeline, labels, model_dir / f"{name}_intent_classifier_weights.json")
    write_manifest(model_dir)

    print(f"✅ Exported '{name}' → {model_dir}/  "
          f"(onnx {onnx_path.stat().st_size/1024:.1f} KB, {len(labels)} labels)")
    return True


# ───────────────────────── CLI ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train multilingual TF-IDF intent models")
    parser.add_argument("--language", "-l",
                        help=f"Language code {sorted(LANGUAGES)} or '{COMBINED_NAME}'.")
    parser.add_argument("--data", "-d", type=Path,
                        help="Explicit training CSV (overrides the registry path).")
    parser.add_argument("--all", action="store_true",
                        help="Build every per-language model AND the combined model.")
    parser.add_argument("--min-accuracy", type=float, default=DEFAULT_MIN_ACCURACY,
                        help=f"Accuracy gate floor (default {DEFAULT_MIN_ACCURACY}).")
    args = parser.parse_args()

    if not args.all and not args.language:
        parser.error("pass --all, or --language <code> [--data <file>].")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        targets = list(LANGUAGES) + [COMBINED_NAME]
    else:
        targets = [args.language]

    results = {}
    for name in targets:
        # --data only applies to a single explicit --language target.
        explicit = args.data if (not args.all and args.language == name) else None
        df = load_data_for(name, explicit)
        results[name] = train_one(name, df, args.min_accuracy)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'}  {name}")
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
