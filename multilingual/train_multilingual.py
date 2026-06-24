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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
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


# ───────────────────────── Export helpers ────────────────────────────────────
CAL_ROUND = 4   # decimal places for calibration breakpoints (matches production)


def extract_lr(clf):
    """Return (coef, intercept, classes) from a plain LR or a CalibratedClassifierCV.

    Mirrors scripts/export_ios_weights.py::_extract_lr. The shipped pipeline uses
    CalibratedClassifierCV(isotonic, cv=3, ensemble=False), so calibrated_classifiers_
    holds a SINGLE base LR refit on all the data. We return that one LR's coef/intercept
    verbatim — these are the exact weights the exported ONNX runs internally, so the
    on-device linear layer reproduces the ONNX logit bit-for-bit (no fold averaging).
    The isotonic maps (see fit_calibration) then rescale that logit to the calibrated
    probability the ONNX emits, keeping the two paths consistent.

    The averaging branch below is retained only as a defensive fallback for an
    ensemble=True classifier; with ensemble=False it averages over a single estimator
    (i.e. a no-op) and prints the parity-preserving message.
    """
    if hasattr(clf, "coef_"):
        return clf.coef_, clf.intercept_, clf.classes_
    fold_clfs = []
    for cc in clf.calibrated_classifiers_:
        est = getattr(cc, "estimator", None)
        if est is None:
            est = getattr(cc, "base_estimator", None)   # older sklearn attr name
        fold_clfs.append(est)
    coef = np.mean([fc.coef_ for fc in fold_clfs], axis=0)
    intercept = np.mean([fc.intercept_ for fc in fold_clfs], axis=0)
    classes = fold_clfs[0].classes_
    if len(fold_clfs) == 1:
        print("  [weights] CalibratedClassifierCV(ensemble=False): single base-LR "
              "refit on all data → device logit == ONNX logit (exact parity).")
    else:
        print(f"  [weights] CalibratedClassifierCV(ensemble=True): averaged base-LR "
              f"coef across {len(fold_clfs)} folds (approximate parity).")
    return coef, intercept, classes


def fit_calibration(pipeline: Pipeline, X_train, labels, coef, intercept):
    """Fit per-class isotonic maps: device-logit → calibrated prob.

    Mirrors scripts/export_ios_weights.py::_fit_calibration. The device logit is
    computed from the single base-LR weights (exactly what Swift ships, and exactly
    what the ONNX runs internally under ensemble=False) over the sklearn L2-normalised
    TF-IDF vector; the target is the calibrated pipeline's own predict_proba — i.e.
    precisely what the ONNX/server emits. Swift applies these maps to its logits and
    renormalises, matching the server's calibrated confidence so the 0.70 threshold
    behaves identically on-device.

    Returns {"method": "isotonic_logit", "maps": [...]} in `labels` order.
    """
    # coef/intercept come from extract_lr in clf.classes_ order == `labels` order.
    Xtf = pipeline.named_steps["tfidf"].transform(X_train)     # (N, n_feat), L2-normed
    coef = np.asarray(coef, dtype=np.float64)
    intercept = np.asarray(intercept, dtype=np.float64)
    logits = np.asarray(Xtf @ coef.T) + intercept             # (N, n_classes), labels order

    srv = pipeline.predict_proba(X_train)                     # (N, n_classes) clf order
    clf = pipeline.named_steps["clf"]
    col = {c: i for i, c in enumerate(clf.classes_)}
    srv = srv[:, [col[lbl] for lbl in labels]]

    maps = []
    for k in range(len(labels)):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(logits[:, k], srv[:, k])
        xs = [round(float(v), CAL_ROUND) for v in ir.X_thresholds_]
        ys = [round(float(v), CAL_ROUND) for v in ir.y_thresholds_]
        maps.append({"x": xs, "y": ys})

    n_pts = sum(len(m["x"]) for m in maps)
    print(f"  [calibration] fitted isotonic maps: {len(maps)} classes, {n_pts} "
          f"breakpoints (device-logit → calibrated-prob).")
    return {"method": "isotonic_logit", "maps": maps}


def export_weights_json(pipeline: Pipeline, labels, out_path: Path,
                        coef, intercept, calibration):
    """Dump TF-IDF + single base-LR weights (+ isotonic calibration) for the Swift path.

    Mirrors scripts/export_ios_weights.py. `coef`/`intercept` are the single base-LR
    weights from extract_lr (in `labels` order) — the same LR the ONNX runs under
    ensemble=False; `calibration` is the isotonic map block so the on-device softmax
    matches the server's calibrated confidence.
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
        "calibration": calibration,   # {"method": "isotonic_logit", "maps": [...]}
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # CalibratedClassifierCV(isotonic, cv=3, ensemble=False) — calibrates the
    # over-confident C=15 LR. ensemble=False is deliberate and load-bearing for the
    # on-device path: it fits ONE base LR on all the data (cross_val_predict only
    # supplies unbiased scores for the calibrator), so calibrated_classifiers_ holds
    # a SINGLE estimator + a SINGLE isotonic calibrator per class — not a 3-fold
    # ensemble. Consequences:
    #   • the exported ONNX bakes one LR (≈1/3 the size of the cv=3 ensemble), and
    #   • the device logit (from that one LR's coef/intercept, see extract_lr) is the
    #     EXACT logit the ONNX's internal LR computes — no fold-averaging
    #     approximation — so Swift and ONNX agree by construction (parity).
    # With ensemble=True the device had to average 3 fold-LRs and fit a separate map,
    # which drifted ~30% of utterances across the 0.70 threshold in conformance.
    # Word analyzer only — skl2onnx cannot export char-analyzer TfidfVectorizers.
    print(f"TF-IDF recipe: {tfidf_kwargs}")
    base_lr = LogisticRegression(max_iter=3000, class_weight="balanced", C=15.0)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_kwargs)),
        ("clf", CalibratedClassifierCV(base_lr, method="isotonic", cv=3,
                                       ensemble=False)),
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
    # On-device path: take the single base-LR coefs (ensemble=False → same LR the
    # ONNX runs), then fit isotonic calibration (device-logit → calibrated prob) so
    # Swift matches the calibrated ONNX output.
    coef, intercept, _ = extract_lr(pipeline.named_steps["clf"])
    calibration = fit_calibration(pipeline, X_train, labels, coef, intercept)
    export_weights_json(pipeline, labels,
                        model_dir / f"{name}_intent_classifier_weights.json",
                        coef, intercept, calibration)
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
