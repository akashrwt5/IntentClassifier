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
subgraph — see scripts/export_coreml.py). This multilingual script trains a
PLAIN LogisticRegression and calibrates its confidence with single-parameter
**temperature scaling**: one scalar `T` per model, fit by bounded NLL
minimization on a held-out calibration split, applied at inference as
`softmax(logits / T)`. Temperature scaling is rank-preserving (argmax is
unchanged), so intent selection is identical to the raw logits — only the
confidence used by the 0.70 GenAI-fallback gate is rescaled. `T` is persisted in
the exported classifier_weights.json ("temperature"); a missing key means
T = 1.0 (plain softmax) for backward compatibility. See
multilingual/TEMPERATURE_SCALING_DECISION.md for the pivot rationale (away from
per-class isotonic calibration).

Caveat: class_weight="balanced" shifts the LR's implied class priors; `T`
corrects logit *sharpness*, not that prior shift. This is acceptable for a
single-threshold confidence gate but is not full calibration. The per-language
Core ML (.mlpackage) export and multilingual-vocab pruning are handled
separately (see scripts/export_ios_weights.py).
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
from scipy.optimize import minimize_scalar
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

COMBINED_NAME = "multilingual"        # -> multilingual_intent_model.onnx
SMALL_NAME = "multilingual_small"     # -> multilingual_small_intent_model.onnx

# TF-IDF recipes.
#   * The FULL recipe (word 1-2 grams, no vocab cap) is what every per-language
#     model and the combined `multilingual` model use — it mirrors train.py.
#   * The SMALL recipes build a deliberately compact combined model
#     (`multilingual_small`) from the SAME data, trading a sliver of accuracy
#     for a much smaller artifact. Two variants are offered (pick with
#     --small-recipe); both were validated on the combined holdout:
#       - "unigram"   : ngram (1,1), full vocab   -> ~1.4 MB ONNX, acc ~0.863
#                       (matches the 5.3 MB full model's accuracy at ~¼ the size
#                       because single words already carry the intent signal;
#                       bigrams added size, not accuracy)
#       - "maxfeat10k": ngram (1,2), top 10k terms -> ~3.2 MB ONNX, acc ~0.854
#                       (keeps only the 10,000 most frequent terms)
#     Note: trigrams (1,3) were tested and rejected — they nearly doubled size
#     (5.3 -> 9.5 MB) AND lowered accuracy, so no recipe uses them.
FULL_TFIDF = dict(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
SMALL_RECIPES = {
    "unigram":    dict(ngram_range=(1, 1), min_df=2, sublinear_tf=True),
    "maxfeat10k": dict(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=10000),
}
DEFAULT_SMALL_RECIPE = "unigram"

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

    The per-intent cap is applied HERE, and for the combined model it is applied
    PER LANGUAGE before concatenation. This is critical: capping after concat
    (with groupby.tail) would keep only the last language's rows for any intent
    that exceeds the cap across languages — e.g. reminders.add has 832 rows ×4
    languages, so a post-concat cap of 500 would retain 500 Danish rows and ZERO
    English/French/German, dropping "remind"/"rappelle"/"erinnere" from the
    vocabulary entirely. Capping each language first keeps every language's
    vocabulary represented and balanced.
    """
    if explicit is not None:
        print(f"  [{name}] using explicit data file: {explicit}")
        return cap_classes(load_clean(explicit))

    if name in (COMBINED_NAME, SMALL_NAME):
        # multilingual_small trains on the SAME combined/deduped data as the full
        # multilingual model; only its TF-IDF recipe (set in main) differs.
        frames = []
        for lang, path in LANGUAGES.items():
            f = cap_classes(load_clean(path))   # cap PER LANGUAGE, before concat
            f["__lang__"] = lang
            frames.append(f)
            print(f"  [{name}] + {lang}: {len(f)} rows from {path.name} (capped per intent)")
        combined = pd.concat(frames, ignore_index=True).drop(columns="__lang__")
        # Dedup ACROSS languages. load_clean dedups within each file, but some
        # datasets (notably de.csv) contain untranslated English phrases that are
        # byte-identical to en.csv rows. Without this cross-language dedup, those
        # duplicate (text, intent) pairs get scattered across the train/test split
        # — one copy in train, an identical copy in the holdout — which is data
        # leakage that inflates the combined model's measured accuracy. Keep the
        # first occurrence (registry order) so the phrase stays attributed once.
        before = len(combined)
        combined = combined.drop_duplicates(subset=["text", "intent"]).reset_index(drop=True)
        removed = before - len(combined)
        if removed:
            print(f"  [{name}] dropped {removed} cross-language duplicate (text,intent) rows "
                  f"-> {len(combined)} unique rows (prevents train/test leakage)")
        return combined

    if name not in LANGUAGES:
        raise SystemExit(
            f"Unknown language '{name}'. Known: {sorted(LANGUAGES)} "
            f"(or '{COMBINED_NAME}'). Pass --data to use a custom file."
        )
    return cap_classes(load_clean(LANGUAGES[name]))


# ───────────────────── Temperature scaling helpers ───────────────────────────
T_BOUNDS = (0.05, 10.0)   # bounded search range for the scalar temperature `T`
ECE_BINS = 15             # 15-bin equal-width reliability diagram (top-1 conf)


def extract_lr(clf):
    """Return (coef, intercept, classes) from a plain LogisticRegression.

    The pipeline now ships a plain LR (no CalibratedClassifierCV wrapper), so the
    coef/intercept are taken verbatim — they are the exact weights the linear layer
    and the exported ONNX both run, so the device logit reproduces the server logit
    bit-for-bit. Confidence calibration is handled separately by the scalar
    temperature `T` (see fit_temperature), not by reshaping these weights.
    """
    return clf.coef_, clf.intercept_, clf.classes_


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise numerically-stable softmax (subtract per-row max before exp)."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _nll(logits: np.ndarray, y_idx: np.ndarray, T: float) -> float:
    """Mean negative log-likelihood (cross-entropy) of the true class under
    softmax(logits / T). This is the primary calibration metric and the objective
    `T` minimizes."""
    probs = _stable_softmax(logits / T)
    p_true = probs[np.arange(len(y_idx)), y_idx]
    return float(-np.log(np.clip(p_true, 1e-12, 1.0)).mean())


def _ece(logits: np.ndarray, y_idx: np.ndarray, T: float, n_bins: int = ECE_BINS) -> float:
    """Expected Calibration Error: 15-bin equal-width over top-1 confidence vs
    accuracy. Diagnostic/secondary metric (NLL is primary)."""
    probs = _stable_softmax(logits / T)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_idx).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_idx)
    for lo, hi in zip(edges[:-1], edges[1:]):
        # last bin is closed on the right so conf == 1.0 is counted
        in_bin = (conf > lo) & (conf <= hi) if hi < 1.0 else (conf > lo) & (conf <= hi + 1e-9)
        m = in_bin.sum()
        if m == 0:
            continue
        ece += (m / n) * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def fit_temperature(logits: np.ndarray, y_idx: np.ndarray) -> float:
    """Fit the scalar temperature `T` that minimizes NLL on the calibration logits.

    Uses scipy bounded scalar minimization over T_BOUNDS. `logits` are the raw
    decision_function scores (== device logits for these full-vocab models, since no
    pruning happens here); `y_idx` is the true-class index in classifier-class order.
    Lets the data set T's direction — no hardcoded expectation.
    """
    res = minimize_scalar(lambda T: _nll(logits, y_idx, T),
                          bounds=T_BOUNDS, method="bounded")
    return float(res.x)


def export_weights_json(pipeline: Pipeline, labels, out_path: Path,
                        coef, intercept, temperature: float):
    """Dump TF-IDF + plain-LR weights (+ scalar temperature) for the Swift path.

    `coef`/`intercept` are the plain-LR weights from extract_lr (in `labels` order) —
    the same LR the ONNX runs. `temperature` is the single scalar applied on-device as
    softmax(logits / T) so the confidence matches the server's temperature-scaled
    output. A consumer that does not find "temperature" treats it as 1.0 (plain
    softmax) for backward compatibility.
    """
    tfidf = pipeline.named_steps["tfidf"]
    weights = {
        "labels": list(labels),
        "vocab": {k: int(v) for k, v in tfidf.vocabulary_.items()},
        "idf": [round(x, 6) for x in tfidf.idf_.tolist()],
        "coef": [[round(float(x), 6) for x in row] for row in np.asarray(coef).tolist()],
        "intercept": [round(float(x), 6) for x in np.asarray(intercept).tolist()],
        "ngram_range": list(tfidf.ngram_range),
        "sublinear_tf": bool(tfidf.sublinear_tf),
        "normalize": "l2",     # Swift L2-normalises the TF-IDF vector before the linear layer
        "conf_threshold": 0.70,
        "conf_gap_threshold": 0.20,
        "temperature": round(float(temperature), 6),   # softmax(logits / T) on-device
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
def train_one(name: str, df: pd.DataFrame, min_accuracy: float,
              tfidf_kwargs: dict) -> bool:
    """Train, gate, and export a single model. Returns True on success.

    `tfidf_kwargs` selects the TF-IDF recipe: FULL_TFIDF for the per-language and
    combined models, or a SMALL_RECIPES entry for `multilingual_small`.
    """
    print(f"\n{'='*70}\nMODEL: {name}\n{'='*70}")

    # NOTE: the per-intent cap is applied in load_data_for (per-language for the
    # combined model), NOT here. Re-capping the already-concatenated combined
    # frame would re-truncate it back to a single language. df arrives capped.
    print(f"Samples: {len(df)} | Intents: {df['intent'].nunique()}")

    # Need at least 2 examples per class for a stratified split.
    counts = df["intent"].value_counts()
    too_few = counts[counts < 2]
    if len(too_few):
        print(f"  [warn] dropping {len(too_few)} intent(s) with <2 samples: "
              f"{too_few.index.tolist()[:8]}{'...' if len(too_few) > 8 else ''}")
        df = df[df["intent"].isin(counts[counts >= 2].index)].reset_index(drop=True)

    X, y = df["text"], df["intent"]
    # 3-WAY SPLIT (no leakage): train (fit LR) → calibration (fit T) → test (report).
    # First peel off the test set (20%), then peel a calibration set off the remainder
    # (20% of the remainder ≈ 16% overall). `T` is fit ONLY on the calibration split and
    # every calibration metric is reported ONLY on the untouched test split, so `T` is
    # never fit and scored on the same data. Stratify both splits to preserve the class
    # mix; fall back to unstratified only if a split is too small to stratify.
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    cal_counts = y_tmp.value_counts()
    strat_cal = y_tmp if cal_counts.min() >= 2 else None
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_tmp, y_tmp, test_size=0.2, random_state=42, stratify=strat_cal
    )
    print(f"Train: {len(X_train)} | Calibration: {len(X_cal)} | Test: {len(X_test)}")

    # Plain LogisticRegression (C=15) — confidence is calibrated post-hoc by the scalar
    # temperature `T` (fit below), NOT by wrapping the LR. Because no CalibratedClassifierCV
    # sits in the pipeline, the exported ONNX/device linear layer runs these exact
    # coef/intercept, and `decision_function` yields the raw logits that `T` rescales.
    # Word analyzer only — skl2onnx cannot export char-analyzer TfidfVectorizers.
    print(f"TF-IDF recipe: {tfidf_kwargs}")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_kwargs)),
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

    # ── Temperature scaling: fit T on the calibration split, report on the test split.
    # decision_function gives the raw logits in clf.classes_ order; these are the
    # device-equivalent logits for these full-vocab models (no pruning here), so the
    # T fit here is the device-path T.
    clf = pipeline.named_steps["clf"]
    cls_idx = {c: i for i, c in enumerate(clf.classes_)}
    cal_logits = np.asarray(pipeline.decision_function(X_cal))
    test_logits = np.asarray(pipeline.decision_function(X_test))
    y_cal_idx = np.array([cls_idx[c] for c in y_cal])
    y_test_idx = np.array([cls_idx[c] for c in y_test])

    temperature = fit_temperature(cal_logits, y_cal_idx)

    nll_raw, nll_temp = _nll(test_logits, y_test_idx, 1.0), _nll(test_logits, y_test_idx, temperature)
    ece_raw, ece_temp = _ece(test_logits, y_test_idx, 1.0), _ece(test_logits, y_test_idx, temperature)
    # Rank-preserving: argmax(logits / T) == argmax(logits), so temp accuracy == raw.
    acc_raw = float((test_logits.argmax(axis=1) == y_test_idx).mean())
    acc_temp = float(((test_logits / temperature).argmax(axis=1) == y_test_idx).mean())
    print(f"  [temperature] T = {temperature:.4f}  (fit on {len(y_cal_idx)} cal samples, "
          f"reported on {len(y_test_idx)} test samples)")
    print(f"  [temperature] NLL  raw {nll_raw:.4f} → temp {nll_temp:.4f}  "
          f"({'✅ improved' if nll_temp < nll_raw else '⚠️ NOT improved'})")
    print(f"  [temperature] ECE  raw {ece_raw:.4f} → temp {ece_temp:.4f}  "
          f"({'✅ improved' if ece_temp < ece_raw else '⚠️ NOT improved'})")
    print(f"  [temperature] argmax acc  raw {acc_raw:.4f} → temp {acc_temp:.4f}  "
          f"({'✅ rank-preserving' if acc_temp >= acc_raw else '❌ rank CHANGED'})")

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
    labels = clf.classes_.tolist()

    model_dir = MODELS_DIR / name
    model_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = model_dir / f"{name}_intent_model.onnx"
    if name == COMBINED_NAME:
        onnx_path = model_dir / "multilingual_intent_model.onnx"

    # raw_scores=True makes the ONNX 'probabilities' output expose the raw
    # decision_function logits (NOT a softmax) so the server can apply
    # softmax(logits / T) itself. Without it, skl2onnx bakes a softmax into the
    # graph and dividing those probabilities by T would be mathematically wrong.
    onnx_model = convert_sklearn(
        pipeline,
        initial_types=[("input", StringTensorType([None, 1]))],
        options={id(pipeline.named_steps["clf"]): {"zipmap": False, "raw_scores": True}},
    )
    onnx_path.write_bytes(onnx_model.SerializeToString())

    joblib.dump(pipeline, str(model_dir / f"{name}_intent_pipeline.pkl"))
    joblib.dump(labels, str(model_dir / f"{name}_intent_labels.pkl"))
    with open(model_dir / f"{name}_intent_labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    # On-device path: plain-LR coefs (the same LR the ONNX runs) + the scalar
    # temperature `T`, applied on-device as softmax(logits / T) to match the server.
    coef, intercept, _ = extract_lr(clf)
    export_weights_json(pipeline, labels,
                        model_dir / f"{name}_intent_classifier_weights.json",
                        coef, intercept, temperature)
    write_manifest(model_dir)

    print(f"✅ Exported '{name}' → {model_dir}/  "
          f"(onnx {onnx_path.stat().st_size/1024:.1f} KB, {len(labels)} labels)")
    return True


# ───────────────────────── CLI ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train multilingual TF-IDF intent models")
    parser.add_argument("--language", "-l",
                        help=f"Language code {sorted(LANGUAGES)}, '{COMBINED_NAME}', "
                             f"or '{SMALL_NAME}'.")
    parser.add_argument("--data", "-d", type=Path,
                        help="Explicit training CSV (overrides the registry path).")
    parser.add_argument("--all", action="store_true",
                        help="Build every per-language model, the combined model, "
                             "AND the compact multilingual_small model.")
    parser.add_argument("--small-recipe", choices=sorted(SMALL_RECIPES),
                        default=DEFAULT_SMALL_RECIPE,
                        help=f"TF-IDF recipe for '{SMALL_NAME}' "
                             f"(default '{DEFAULT_SMALL_RECIPE}': ngram (1,1), ~1.4 MB; "
                             f"'maxfeat10k': ngram (1,2) top-10k terms, ~3.2 MB).")
    parser.add_argument("--min-accuracy", type=float, default=DEFAULT_MIN_ACCURACY,
                        help=f"Accuracy gate floor (default {DEFAULT_MIN_ACCURACY}).")
    args = parser.parse_args()

    if not args.all and not args.language:
        parser.error("pass --all, or --language <code> [--data <file>].")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        targets = list(LANGUAGES) + [COMBINED_NAME, SMALL_NAME]
    else:
        targets = [args.language]

    results = {}
    for name in targets:
        # --data only applies to a single explicit --language target.
        explicit = args.data if (not args.all and args.language == name) else None
        # multilingual_small uses the chosen compact recipe; everything else uses
        # the full word 1-2 gram recipe.
        tfidf_kwargs = SMALL_RECIPES[args.small_recipe] if name == SMALL_NAME else FULL_TFIDF
        df = load_data_for(name, explicit)
        results[name] = train_one(name, df, args.min_accuracy, tfidf_kwargs)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'}  {name}")
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
