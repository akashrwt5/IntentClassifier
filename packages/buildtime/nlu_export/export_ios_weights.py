#!/usr/bin/env python3
"""
Export TF-IDF + LogisticRegression weights to iOS-compatible JSON.

The iOS app performs inference natively using:
  tfidf_vector = {vocab[token]: idf[vocab[token]] * log(1 + count) for token in tokens}
  scores[i] = dot(coef[i], tfidf_vector) + intercept[i]
  probabilities = softmax(scores / T)          # T = "temperature" (scalar)

Confidence is calibrated by single-parameter temperature scaling: one scalar T,
fit by bounded NLL minimization on the device-equivalent logits (the pruned-vocab,
L2-normalised logits Swift actually computes — see _device_logits), applied at
inference as softmax(scores / T). It is rank-preserving (argmax unchanged), so the
predicted intent is identical to the raw logits; only the confidence used by the
0.70 GenAI-fallback gate is rescaled. A consumer that does not find "temperature"
falls back to T = 1.0 (plain softmax). This replaces the previous per-class
isotonic calibration; see multilingual/TEMPERATURE_SCALING_DECISION.md.

Vocabulary pruning (--top-per-class):
  For each class keep the top N features by abs(coefficient). Take the union
  across all classes. This keeps discriminative features for every intent
  while discarding low-signal terms, shrinking the file significantly.

Usage:
    python scripts/export_ios_weights.py
    python scripts/export_ios_weights.py --top-per-class 25 --out models/intent_classifier_weights.json

Reads:  models/intent/en/pipeline.pkl  (saved by train.py)
        models/intent/en/labels.pkl
Writes: models/intent_classifier_weights.json
"""

import argparse
import json
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
PIPELINE_PATH = None
LABELS_PATH   = None
DATA_PATH     = None

CONF_THRESHOLD     = 0.70
CONF_GAP_THRESHOLD = 0.20
GENAI_BASE_URL     = "https://genai.yourcompany.com/chat?query="

ROUND = 4   # decimal places — 4 dp is sufficient for LR inference
T_BOUNDS = (0.05, 10.0)   # bounded search range for the scalar temperature `T`


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
    Return (coef_, intercept_, classes_) from a plain LogisticRegression.

    The pipeline now ships a plain LR (no CalibratedClassifierCV wrapper): confidence
    is calibrated post-hoc by the scalar temperature `T` (see _fit_temperature), not
    by reshaping these weights, so the coef/intercept are taken verbatim — exactly the
    weights Swift and the exported ONNX both run.
    """
    return clf.coef_, clf.intercept_, clf.classes_


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


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise numerically-stable softmax (subtract per-row max before exp)."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _nll(logits: np.ndarray, y_idx: np.ndarray, T: float) -> float:
    """Mean negative log-likelihood (cross-entropy) of the true class under
    softmax(logits / T) — the objective `T` minimizes."""
    probs = _stable_softmax(logits / T)
    p_true = probs[np.arange(len(y_idx)), y_idx]
    return float(-np.log(np.clip(p_true, 1e-12, 1.0)).mean())


def _fit_temperature(labels, vocab, idf, coef, intercept):
    """Fit the scalar temperature `T` on the DEVICE-equivalent logits.

    Req #2: `T` must be fit and validated on the logits Swift actually computes —
    the pruned-vocab, L2-on-pruned-subspace logits from _device_logits — not the
    server's full-vocab logits, so the shipped `T` keeps the device confidence on the
    right side of the 0.70 gate. We minimize NLL of the true class (bounded scalar
    search over T_BOUNDS) against the training labels.

    Caveat: this fits `T` on the same texts that trained the LR (the production export
    has no separate held-out split here). Temperature is a single parameter — far less
    prone to overfit than the per-class isotonic maps it replaces — and the
    untouched-test-set NLL/ECE validation lives in the multilingual training path
    (train_multilingual.py) and the conformance suite. Returns 1.0 (identity, plain
    softmax) if the data is unavailable.

    Note the class_weight="balanced" caveat: `T` corrects logit sharpness, not the
    balanced-prior shift; acceptable for a single-threshold confidence gate.
    """
    from scipy.optimize import minimize_scalar

    if not DATA_PATH.exists():
        print(f"  [ios export] WARN: {DATA_PATH} not found — shipping T=1.0 (plain softmax).")
        return 1.0

    import pandas as pd
    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig", header=0)
    data.columns = [c.strip().lower() for c in data.columns]
    data = data.dropna(subset=["text", "intent"])
    texts = data["text"].astype(str).str.lower().str.strip().tolist()
    intents = data["intent"].astype(str).str.strip().tolist()

    # Device logits in `labels` order; map each true intent to its label index.
    logits = _device_logits(texts, vocab, idf, coef, intercept)   # (N, n_classes) labels order
    lbl_idx = {lbl: i for i, lbl in enumerate(labels)}
    keep = [i for i, c in enumerate(intents) if c in lbl_idx]
    logits = logits[keep]
    y_idx = np.array([lbl_idx[intents[i]] for i in keep])

    res = minimize_scalar(lambda T: _nll(logits, y_idx, T),
                          bounds=T_BOUNDS, method="bounded")
    T = float(res.x)

    nll_raw, nll_temp = _nll(logits, y_idx, 1.0), _nll(logits, y_idx, T)
    print(f"  [ios export] Fitted temperature T={T:.4f} on {len(y_idx)} device logits  "
          f"(NLL {nll_raw:.4f} → {nll_temp:.4f}).")
    return round(T, 6)


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

    # Scalar temperature, fit on the exact pruned weights we just built so it
    # matches the logits Swift computes (req #2: device-equivalent logits).
    temperature = _fit_temperature(labels, new_vocab, idf, coef, intercept)

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
        # iOS: probabilities = softmax(scores / temperature). Falls back to
        # T = 1.0 (plain softmax) when this key is absent (older bundles).
        "temperature":        temperature,
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
    parser.add_argument("--lang", default="en", help="Language code (e.g., en)")
    parser.add_argument("--out", help="Output path (default: models/intent/{lang}/intent_classifier_weights.json)")
    parser.add_argument("--top-per-class", type=int, default=30,
                        help="Top N features per class by |coef| to keep (default: 30)")
    args = parser.parse_args()
    
    PIPELINE_PATH = BASE_DIR / "models" / "intent" / args.lang / "pipeline.pkl"
    LABELS_PATH   = BASE_DIR / "models" / "intent" / args.lang / "labels.pkl"
    DATA_PATH     = BASE_DIR / "language_packs" / args.lang / "train.csv"
    
    out_path = Path(args.out) if args.out else BASE_DIR / "models" / "intent" / args.lang / "intent_classifier_weights.json"
    export(out_path, args.top_per_class)
