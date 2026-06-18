#!/usr/bin/env python3
"""
Export TF-IDF + LogisticRegression weights to iOS-compatible JSON.

The iOS app performs inference natively using:
  tfidf_vector = {vocab[token]: idf[vocab[token]] * log(1 + count) for token in tokens}
  scores[i] = dot(coef[i], tfidf_vector) + intercept[i]
  probabilities = softmax(scores)

Vocabulary pruning (--top-per-class):
  For each class keep the top N features by abs(coefficient). Take the union
  across all classes. This keeps discriminative features for every intent
  while discarding low-signal terms, shrinking the file significantly.

Usage:
    python scripts/export_ios_weights.py
    python scripts/export_ios_weights.py --top-per-class 25 --out models/intent_classifier_weights.json

Reads:  models/intent_pipeline.pkl  (saved by train.py)
        models/intent_labels.pkl
Writes: models/intent_classifier_weights.json
"""

import argparse
import json
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
PIPELINE_PATH = BASE_DIR / "models" / "intent_pipeline.pkl"
LABELS_PATH   = BASE_DIR / "models" / "intent_labels.pkl"
DATA_PATH     = BASE_DIR / "data"   / "intent_data_new.csv"

CONF_THRESHOLD     = 0.70
CONF_GAP_THRESHOLD = 0.20
GENAI_BASE_URL     = "https://genai.yourcompany.com/chat?query="

ROUND = 4   # decimal places — 4 dp is sufficient for LR inference


def _load_or_train_pipeline():
    if PIPELINE_PATH.exists():
        print(f"Loading pipeline from {PIPELINE_PATH}")
        return joblib.load(str(PIPELINE_PATH))

    print(f"Pipeline not found at {PIPELINE_PATH} — training from {DATA_PATH}")
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig", header=0)
    data.columns = [c.strip().lower() for c in data.columns]
    data["text"]   = data["text"].astype(str).str.lower().str.strip()
    data["intent"] = data["intent"].astype(str).str.strip()
    data = data.dropna().drop_duplicates(subset=["text", "intent"])

    MAX_PER_INTENT = 500
    data = (
        pd.concat([g.sample(min(len(g), MAX_PER_INTENT), random_state=42)
                   for _, g in data.groupby("intent")])
        .sample(frac=1, random_state=42).reset_index(drop=True)
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf",   LogisticRegression(max_iter=3000, class_weight="balanced", C=15.0))
    ])
    pipeline.fit(data["text"], data["intent"])
    joblib.dump(pipeline, str(PIPELINE_PATH))
    print(f"Pipeline saved to {PIPELINE_PATH}")
    return pipeline


def _select_features(coef_matrix: np.ndarray, top_per_class: int) -> np.ndarray:
    """Return sorted array of feature indices: union of top-N per class by |coef|."""
    selected = set()
    for row in coef_matrix:
        top_idx = np.argpartition(np.abs(row), -top_per_class)[-top_per_class:]
        selected.update(top_idx.tolist())
    return np.array(sorted(selected), dtype=int)


def _extract_lr(clf):
    """
    Return (coef_, intercept_, classes_) from a plain LogisticRegression or a
    CalibratedClassifierCV wrapping one.  For the calibrated case we average
    the base LR weights across folds — the server applies per-fold isotonic
    calibration on the decision-function margin, which cannot be reproduced
    bit-for-bit on iOS.  Instead we fit a single per-class isotonic map on the
    averaged model's own logits (see `_fit_calibration`) and ship it alongside
    the weights, so the iOS softmax can be recalibrated to match the server.
    """
    if hasattr(clf, "coef_"):
        return clf.coef_, clf.intercept_, clf.classes_

    # CalibratedClassifierCV — average base LR weights across CV folds
    fold_clfs = [cc.estimator for cc in clf.calibrated_classifiers_]
    coef      = np.mean([fc.coef_      for fc in fold_clfs], axis=0)
    intercept = np.mean([fc.intercept_ for fc in fold_clfs], axis=0)
    classes   = fold_clfs[0].classes_
    print(f"  [ios export] CalibratedClassifierCV detected — averaging coef across "
          f"{len(fold_clfs)} folds for iOS base weights (isotonic calibration shipped separately).")
    return coef, intercept, classes


# ──────────────────────────────────────────────────────────────────────────────
# iOS-side calibration
# ──────────────────────────────────────────────────────────────────────────────

def _swift_tokenize(text: str):
    """Replicate IntentClassifierService.tokenize(): lowercase, split on
    non-alphanumerics, then unigrams + adjacent bigrams (single chars kept)."""
    import re
    words = [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w]
    tokens = list(words)
    for i in range(len(words) - 1):
        tokens.append(words[i] + " " + words[i + 1])
    return tokens


def _device_logits(texts, vocab, idf, coef, intercept):
    """Reproduce the on-device Stage-2 logits exactly: sublinear-TF over the
    pruned vocab, L2-normalised on the pruned subspace, then the linear layer.
    Fitting calibration on these (not sklearn's full-vocab vector) keeps the
    isotonic maps faithful to what Swift actually computes."""
    idf       = np.asarray(idf, dtype=np.float64)
    coef      = np.asarray(coef, dtype=np.float64)       # (n_classes, n_feat)
    intercept = np.asarray(intercept, dtype=np.float64)  # (n_classes,)
    n_feat    = len(idf)
    out = np.empty((len(texts), coef.shape[0]))
    for r, text in enumerate(texts):
        counts = {}
        for tok in _swift_tokenize(text):
            j = vocab.get(tok)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        vec = np.zeros(n_feat)
        for j, c in counts.items():
            vec[j] = (1.0 + np.log(c)) * idf[j]
        norm = np.sqrt((vec * vec).sum())
        if norm > 0:
            vec /= norm
        out[r] = coef @ vec + intercept
    return out


def _fit_calibration(pipeline, labels, vocab, idf, coef, intercept):
    """Fit one isotonic map per class: device logit → server-calibrated prob.

    The server (CalibratedClassifierCV, isotonic) and the iOS averaged-LR softmax
    disagree because iOS ships uncalibrated probabilities. We learn, per class,
    the monotonic function that maps the device's own logit margin onto the
    server's calibrated probability, fit over the training texts. iOS then applies
    these maps to its logits and renormalises, recovering server-level confidence
    (so genuinely-correct mid-confidence intents clear the 0.70 threshold).

    Returns a list (in `labels` order) of {"x": [...], "y": [...]} breakpoints
    defining a clamped piecewise-linear curve, or None if the data is unavailable.
    """
    from sklearn.isotonic import IsotonicRegression

    if not DATA_PATH.exists():
        print(f"  [ios export] WARN: {DATA_PATH} not found — skipping calibration export.")
        return None

    import pandas as pd
    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig", header=0)
    data.columns = [c.strip().lower() for c in data.columns]
    texts = data["text"].astype(str).str.lower().str.strip().tolist()

    logits = _device_logits(texts, vocab, idf, coef, intercept)   # (N, n_classes) labels order

    # Server calibrated probabilities, reordered into `labels` order.
    srv = pipeline.predict_proba(texts)                            # (N, n_classes) clf order
    clf = pipeline.named_steps["clf"]
    col = {cls: i for i, cls in enumerate(clf.classes_)}
    srv = srv[:, [col[lbl] for lbl in labels]]

    maps = []
    for k in range(len(labels)):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(logits[:, k], srv[:, k])
        xs = [round(float(v), ROUND) for v in ir.X_thresholds_]
        ys = [round(float(v), ROUND) for v in ir.y_thresholds_]
        maps.append({"x": xs, "y": ys})

    n_pts = sum(len(m["x"]) for m in maps)
    print(f"  [ios export] Fitted isotonic calibration: {len(maps)} classes, "
          f"{n_pts} breakpoints (device-logit → server-prob).")
    return maps


def export(out_path: Path, top_per_class: int):
    pipeline = _load_or_train_pipeline()
    labels   = joblib.load(str(LABELS_PATH))

    tfidf = pipeline.named_steps["tfidf"]
    clf   = pipeline.named_steps["clf"]

    coef_, intercept_, classes_ = _extract_lr(clf)
    class_to_row = {cls: i for i, cls in enumerate(classes_)}

    # Full coef matrix ordered by labels
    full_coef = np.array([coef_[class_to_row[lbl]] for lbl in labels])  # (n_classes, n_features)

    # Select discriminative feature subset
    feat_idx = _select_features(full_coef, top_per_class)
    pruned_coef = full_coef[:, feat_idx]                   # (n_classes, n_selected)
    pruned_idf  = tfidf.idf_[feat_idx]                     # (n_selected,)

    # Build a remapped vocab: token -> new sequential index
    idx_to_token = {v: k for k, v in tfidf.vocabulary_.items()}
    new_vocab = {idx_to_token[old_idx]: new_idx
                 for new_idx, old_idx in enumerate(feat_idx)}

    coef      = [[round(float(v), ROUND) for v in row] for row in pruned_coef]
    idf       = [round(float(v), ROUND) for v in pruned_idf]
    intercept = [round(float(intercept_[class_to_row[lbl]]), ROUND) for lbl in labels]

    # Per-class isotonic calibration (device logit → server-calibrated prob),
    # fit on the exact pruned weights we just built so it matches Swift's logits.
    calibration_maps = _fit_calibration(pipeline, labels, new_vocab, idf, coef, intercept)

    payload = {
        "labels":             labels,
        "vocab":              new_vocab,
        "idf":                idf,
        "coef":               coef,
        "intercept":          intercept,
        "conf_threshold":     CONF_THRESHOLD,
        "conf_gap_threshold": CONF_GAP_THRESHOLD,
        "genai_base_url":     GENAI_BASE_URL,
        # iOS inference must L2-normalise the TF-IDF vector before scoring:
        #   norm = sqrt(sum(v*v)); if norm > 0: vec /= norm
        "normalize":          "l2",
    }
    if calibration_maps is not None:
        # iOS: calibrated[k] = interp(logit[k], maps[k].x, maps[k].y) clamped to
        # the endpoints; then divide by the sum across classes. Falls back to
        # plain softmax when this key is absent (older bundles).
        payload["calibration"] = {
            "method": "isotonic_logit",
            "maps":   calibration_maps,
        }

    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_kb = out_path.stat().st_size / 1024
    print(f"✅ Exported {len(labels)} intents, vocab={len(new_vocab)} "
          f"(pruned from {len(tfidf.vocabulary_)} using top {top_per_class}/class), "
          f"coef={len(coef)}x{len(coef[0])} → {out_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(BASE_DIR / "models" / "intent_classifier_weights.json"))
    parser.add_argument("--top-per-class", type=int, default=30,
                        help="Top N features per class by |coef| to keep (default: 30)")
    args = parser.parse_args()
    export(Path(args.out), args.top_per_class)
