#!/usr/bin/env python3
"""
Fit the confidence-calibration temperature `T` for the SERVER/ONNX featurizer.

Confidence is `softmax(logits / T)`. `T` is rank-preserving — it can never change
which intent wins, only how confident the engine is that it won. That confidence
then drives five separate gates (fire-vs-GenAI, slot acceptance, flow
interruption, semantic agreement, semantic rescue), so a wrong `T` silently
mis-tunes all of them at once.

WHY THIS SCRIPT EXISTS
----------------------
Two `T` values existed for English and neither was fit correctly:

  packs/en/intent_model/weights.json   T=0.796  fit on DEVICE logits (1370-term
                                                pruned vocab, iOS/Swift path) but
                                                applied by the Python engine to
                                                FULL-vocab ONNX logits.
  config/calibration.json              T=0.621  fit on the right featurizer, but
                                                on a set that is 99.6% training
                                                data — and read by nothing.

Temperature is a property of a **(model, featurizer)** pair, not of a language.
Each featurizer needs its own `T`, fit on the logits it will actually calibrate,
using data the model has not memorised.

METHOD — out-of-fold (OOF)
--------------------------
Post-hoc calibration must be fit on held-out predictions. Rather than sacrifice a
split, every row is scored by a fold-model that never saw it, giving leakage-free
logits for the whole dataset. `T` is then the bounded scalar minimising NLL over
those logits. This is standard practice and costs seconds at this data size.

The featurizer here mirrors scripts/train.py exactly, so the fitted `T` calibrates
the same logits the shipped ONNX emits.

USAGE
    python scripts/fit_calibration.py                 # fit + report, write nothing
    python scripts/fit_calibration.py --write         # write packs/en/intent_model/calibration.json
    python scripts/fit_calibration.py --lang en --folds 5

The iOS/device `T` is fit separately by scripts/export_ios_weights.py against the
pruned device logits — that separation is correct and deliberate. Do not unify them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "04_GENERATED_MASTER_training_data.csv"
# Evaluation sets the calibration data must NOT overlap. Fitting on these would
# burn the only honest measurement of generalisation the project has.
EVAL_SETS = [
    BASE_DIR / "data" / "semantic_holdout_2.csv",
    BASE_DIR / "data" / "semantic_holdout_100.csv",
]
T_BOUNDS = (0.05, 10.0)

# Must mirror scripts/train.py — the fitted T only calibrates logits produced by
# this exact featurizer. If train.py changes, change this together with it.
MAX_PER_INTENT = 500
TFIDF_KW = {"ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
LR_KW = {"max_iter": 3000, "class_weight": "balanced", "C": 15.0}


def _norm(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", str(s).lower().strip()))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eval_leakage_mask(texts):
    """Return (keep_mask, leaked_examples, n_eval_checked).

    Rows that also appear in an evaluation set are EXCLUDED from calibration —
    fitting on them would invalidate every number later measured against those
    sets. Calibration historically had no guard at all, which is how a
    99.6%-leaked set became the basis for the shipped temperature.

    Matching is normalised (case + punctuation). Note that scripts/train.py's own
    guard compares raw strings, so it does NOT catch pairs differing only by a
    trailing '?' — those leaks reach the model even though training reports clean.
    """
    normed = [_norm(t) for t in texts]
    leaked_norm, checked = set(), 0
    for path in EVAL_SETS:
        if not path.exists():
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        col = next((c for c in df.columns
                    if c.strip().lower() in ("utterance", "text", "query", "phrase")), df.columns[0])
        evalset = {_norm(t) for t in df[col]}
        checked += len(evalset)
        leaked_norm |= (set(normed) & evalset)
    keep = np.array([n not in leaked_norm for n in normed])
    examples = [t for t, n in zip(texts, normed) if n in leaked_norm][:5]
    return keep, examples, checked


def oof_logits(X, y, folds: int, seed: int = 0):
    """Logits for every row, each produced by a model that never saw that row."""
    classes = np.array(sorted(set(y)))
    index = {c: i for i, c in enumerate(classes)}
    out = np.zeros((len(y), len(classes)), dtype=np.float64)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(skf.split(X, y), 1):
        pipe = Pipeline([("tfidf", TfidfVectorizer(**TFIDF_KW)),
                         ("clf", LogisticRegression(**LR_KW))]).fit(X[tr], y[tr])
        scores = pipe.decision_function(X[te])
        for j, cls in enumerate(pipe.classes_):
            out[te, index[cls]] = scores[:, j]
        print(f"  fold {k}/{folds} done ({len(tr)} train / {len(te)} scored)")
    return out, classes, np.array([index[v] for v in y])


def fit_temperature(logits: np.ndarray, y_idx: np.ndarray) -> float:
    def nll(T: float) -> float:
        s = logits / T
        s = s - s.max(axis=1, keepdims=True)
        e = np.exp(s)
        return float(-np.mean(np.log(e[np.arange(len(y_idx)), y_idx] / e.sum(axis=1))))
    return float(minimize_scalar(nll, bounds=T_BOUNDS, method="bounded").x)


def expected_calibration_error(logits, y_idx, T: float, bins: int = 10) -> float:
    s = logits / T
    s = s - s.max(axis=1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=1, keepdims=True)
    conf, pred = p.max(axis=1), p.argmax(axis=1)
    correct = (pred == y_idx).astype(float)
    ece = 0.0
    for i in range(bins):
        m = (conf > i / bins) & (conf <= (i + 1) / bins)
        if m.sum():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fit the server/ONNX calibration temperature")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write", action="store_true",
                    help="Write packs/<lang>/intent_model/calibration.json (default: report only)")
    args = ap.parse_args()

    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig").dropna(subset=["text", "intent"])
    data = (data.groupby("intent").tail(MAX_PER_INTENT)
                .sample(frac=1, random_state=42).reset_index(drop=True))
    X, y = data["text"].astype(str).values, data["intent"].values
    print(f"Calibration data: {len(X)} rows, {len(set(y))} intents "
          f"(capped at {MAX_PER_INTENT}/intent, mirroring train.py)")

    keep, leaked_examples, n_checked = eval_leakage_mask(X)
    n_leaked = int((~keep).sum())
    if n_leaked:
        print(f"\n  !! LEAKAGE: {n_leaked} training row(s) also appear in an evaluation set.")
        print("     Excluded from calibration. These still reach the MODEL, because")
        print("     train.py's guard compares raw strings and misses punctuation-only")
        print("     differences (e.g. a trailing '?'). Fix at source.")
        for t in leaked_examples:
            print(f"       - {t!r}")
        X, y = X[keep], y[keep]
    print(f"\nLeakage guard: {n_leaked} excluded, checked against {n_checked} evaluation utterances.")

    print(f"\nBuilding out-of-fold logits ({args.folds}-fold):")
    logits, classes, y_idx = oof_logits(X, y, args.folds, args.seed)

    T = fit_temperature(logits, y_idx)
    ece_fit = expected_calibration_error(logits, y_idx, T)
    ece_one = expected_calibration_error(logits, y_idx, 1.0)
    print(f"\nFitted temperature  T = {T:.6f}")
    print(f"  ECE @ T=1.0 (uncalibrated) : {ece_one:.4f}")
    print(f"  ECE @ fitted T             : {ece_fit:.4f}")

    out = BASE_DIR / "packs" / args.lang / "intent_model" / "calibration.json"
    payload = {
        "_note": ("Confidence calibration for the SERVER/ONNX featurizer: "
                  "softmax(logits / temperature). Fit out-of-fold on data the model "
                  "did not memorise, and guarded against evaluation-set leakage. "
                  "The iOS/device temperature is fit separately in "
                  "scripts/export_ios_weights.py against pruned device logits — "
                  "the two are expected to differ and must not be unified."),
        "temperature": round(T, 6),
        "ece": round(ece_fit, 4),
        "ece_uncalibrated": round(ece_one, 4),
        "provenance": {
            "method": "out-of-fold NLL minimisation (bounded scalar)",
            "folds": args.folds,
            "seed": args.seed,
            "n_samples": len(X),
            "n_intents": len(classes),
            "featurizer": f"TfidfVectorizer({TFIDF_KW}) + LogisticRegression({LR_KW})",
            "source": str(DATA_PATH.relative_to(BASE_DIR)),
            "source_sha256": _sha256(DATA_PATH),
            "fitted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fitted_by": "scripts/fit_calibration.py",
        },
    }
    if args.write:
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {out.relative_to(BASE_DIR)}")
        print("NOTE: the runtime does not read this file yet — IntentClassifier still "
              "takes T from weights.json. Wiring it up is a separate, gated change "
              "because it shifts all five confidence thresholds at once.")
    else:
        print("\n(report only — pass --write to update the pack artifact)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
