#!/usr/bin/env python3
"""
Charter B2 — fit the confidence-calibration temperature `T` out-of-fold.

Confidence is `softmax(logits / T)`. `T` is RANK-PRESERVING: dividing by a
positive scalar cannot change which intent wins, only how confident the engine
is that it won. That confidence then drives every gate in the system — the 0.70
fire threshold, the 0.80 confirm gate, slot acceptance, flow interruption,
semantic agreement — so a wrong `T` silently mis-tunes all of them at once and
no test fails.

WHY THIS EXISTS
---------------
Two English temperatures existed and NEITHER was fit correctly (Review-F5
blocker B8):

  models/intent_classifier_weights.json   T=0.796  fit on DEVICE logits (pruned
                                                   1370-term vocab, iOS path)
                                                   but applied by the Python
                                                   engine to FULL-vocab ONNX
                                                   logits.
  config/calibration.json                 T=0.621  fit on the right featurizer,
                                                   but on a set that was 99.9%
                                                   training data — and read by
                                                   nothing at runtime.

Charter B1 then measured the cost: on an honest holdout the model's ACCURACY was
essentially unchanged (0.9007 vs 0.907) while ECE was **2.4x worse** (0.0441 vs
0.018). The leakage was never inflating accuracy — it was concealing
miscalibration. Fixing `T` is therefore the highest-leverage change available.

Temperature is a property of a **(model, featurizer)** pair, not of a language.
Each featurizer needs its own `T`, fit on the logits it will actually calibrate,
using data the model has not memorised.

METHOD — out-of-fold
--------------------
Post-hoc calibration must be fit on held-out predictions. Rather than sacrifice
another split on top of B1's holdout, every training row is scored by a fold
model that never saw it, giving leakage-free logits for the whole set. `T` is
then the bounded scalar minimising NLL over those logits.

The featurizer below mirrors `train.py` exactly — same n-grams, same min_df,
same class_weight, same C, same per-intent cap — so the fitted `T` calibrates
the same logits the shipped ONNX emits. If train.py changes, change this with it
(`test_calibration.py` asserts they agree).

USAGE
    python -m nlu_training.fit_calibration --lang en            # fit + report
    python -m nlu_training.fit_calibration --lang en --write    # persist
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

from nlu_training.leakage import normalize_text

BASE_DIR = Path(__file__).resolve().parents[3]
T_BOUNDS = (0.05, 10.0)
DEFAULT_FOLDS = 5
SEED = 0

# MUST mirror nlu_training/train.py. The fitted T only calibrates logits from
# this exact featurizer; a mismatch here reproduces blocker B8 in a new place.
MAX_PER_INTENT = 500
TFIDF_KW = {"ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
LR_KW = {"max_iter": 3000, "class_weight": "balanced", "C": 15.0}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eval_sets(lang: str) -> list[Path]:
    """Evaluation sets the calibration data must NOT overlap.

    Fitting on these would burn the only honest measurement of generalisation
    the project has — and would recreate exactly the failure B1 just fixed.
    """
    ds = BASE_DIR / "datasets" / lang
    return [p for p in (ds / "holdout_honest.csv",
                        ds / "holdout_leakage_guard.csv",
                        ds / "holdout_paraphrase.csv") if p.exists()]


def eval_leakage_mask(texts, lang: str):
    """(keep_mask, leaked_examples, n_eval_checked).

    Rows that also appear in an evaluation set are EXCLUDED from calibration.
    Calibration historically had NO guard at all, which is how a 99.6%-leaked
    set became the basis for a shipped temperature.
    """
    normed = [normalize_text(t) for t in texts]
    leaked, checked = set(), 0
    for path in eval_sets(lang):
        df = pd.read_csv(path, encoding="utf-8-sig")
        col = next((c for c in df.columns
                    if c.strip().lower() in ("utterance", "text", "query", "phrase")),
                   df.columns[0])
        evalset = {normalize_text(t) for t in df[col]}
        checked += len(evalset)
        leaked |= (set(normed) & evalset)
    keep = np.array([n not in leaked for n in normed])
    examples = [t for t, n in zip(texts, normed) if n in leaked][:5]
    return keep, examples, checked


def oof_logits(X, y, folds: int, seed: int = SEED):
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
        print(f"  fold {k}/{folds}: {len(tr)} train / {len(te)} scored")
    return out, classes, np.array([index[v] for v in y])


def _softmax(logits: np.ndarray, T: float) -> np.ndarray:
    s = logits / T
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, y_idx: np.ndarray) -> float:
    def nll(T: float) -> float:
        p = _softmax(logits, T)
        return float(-np.mean(np.log(np.clip(p[np.arange(len(y_idx)), y_idx], 1e-12, None))))
    return float(minimize_scalar(nll, bounds=T_BOUNDS, method="bounded").x)


def expected_calibration_error(probs: np.ndarray, y_idx: np.ndarray,
                               n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_idx
    ece, n = 0.0, len(y_idx)
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (conf > lo) & (conf <= hi) if b else (conf <= hi)
        if m.sum():
            ece += m.sum() / n * abs(conf[m].mean() - correct[m].mean())
    return float(ece)


def fit(lang: str, folds: int, write: bool) -> int:
    data_path = BASE_DIR / "datasets" / lang / "train.csv"
    if not data_path.exists():
        print(f"FAIL: no training data for {lang!r}: {data_path}")
        return 1

    df = pd.read_csv(data_path, encoding="utf-8-sig").dropna(subset=["text", "intent"])
    df["text"] = df["text"].astype(str).str.lower().str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["text", "intent"])
    df = df.groupby("intent").tail(MAX_PER_INTENT).reset_index(drop=True)

    keep, leaked, checked = eval_leakage_mask(df["text"].values, lang)
    if leaked:
        print(f"  excluded {(~keep).sum()} row(s) present in an evaluation set, "
              f"e.g. {leaked[:2]}")
    df = df[keep]
    print(f"  leakage guard: {checked} evaluation utterances checked, "
          f"{(~keep).sum()} excluded")

    # StratifiedKFold needs at least `folds` members per class.
    counts = df["intent"].value_counts()
    if (counts < folds).any():
        thin = counts[counts < folds]
        print(f"FAIL: {len(thin)} intent(s) have fewer than {folds} rows: "
              f"{dict(thin)}")
        return 1

    X, y = df["text"].values, df["intent"].values
    print(f"  fitting on {len(y)} rows / {len(set(y))} intents, {folds}-fold OOF")
    logits, classes, y_idx = oof_logits(X, y, folds)

    T = fit_temperature(logits, y_idx)
    ece_before = expected_calibration_error(_softmax(logits, 1.0), y_idx)
    ece_after = expected_calibration_error(_softmax(logits, T), y_idx)

    print(f"\n  temperature      : {T:.6f}")
    print(f"  ECE uncalibrated : {ece_before:.4f}")
    print(f"  ECE calibrated   : {ece_after:.4f}")
    if ece_after >= ece_before:
        print("  WARNING: calibration did not improve ECE — investigate before shipping.")

    payload = {
        "_note": "Confidence calibration for the SERVER/ONNX featurizer: "
                 "softmax(logits / temperature). Fit OUT-OF-FOLD on data the "
                 "model has not memorised, with evaluation sets excluded. The "
                 "iOS/device temperature is fit separately against pruned device "
                 "logits — the two calibrate different featurizers and must NOT "
                 "be unified (Review-F5 blocker B8).",
        "temperature": round(T, 6),
        "ece": round(ece_after, 4),
        "ece_uncalibrated": round(ece_before, 4),
        "provenance": {
            "method": "out-of-fold NLL minimisation (bounded scalar)",
            "folds": folds,
            "seed": SEED,
            "n_samples": int(len(y)),
            "n_intents": int(len(set(y))),
            "featurizer": f"TfidfVectorizer({TFIDF_KW}) + LogisticRegression({LR_KW})",
            "max_per_intent": MAX_PER_INTENT,
            "source": str(data_path.relative_to(BASE_DIR)),
            "source_sha256": _sha256(data_path),
            "eval_sets_excluded": [str(p.relative_to(BASE_DIR)) for p in eval_sets(lang)],
            "fitted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fitted_by": "nlu_training.fit_calibration",
        },
    }

    if not write:
        print("\n  (not written — pass --write to persist)")
        return 0

    out = BASE_DIR / "models" / "intent" / lang / "calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(BASE_DIR)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    return fit(a.lang, a.folds, a.write)


if __name__ == "__main__":
    sys.exit(main())
