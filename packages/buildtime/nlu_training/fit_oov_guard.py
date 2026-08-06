#!/usr/bin/env python3
"""Fit `oov_reject_ratio` — the out-of-vocabulary guard — out-of-fold.

WHAT THE GUARD DOES
-------------------
Refuses to act on an utterance when too large a share of its tokens are absent
from the model's vocabulary, however confident the classifier is about the rest.

WHY A CONFIDENCE THRESHOLD CANNOT DO THIS JOB
---------------------------------------------
TF-IDF's vocabulary is a fixed set of slots. A token outside it is not weighed
and dismissed — there is nowhere to put it, so the sentence arrives without it:

    'turn off'          -> 3 non-zero features
    'turn off toshiba'  -> 3 non-zero features, cosine 1.000000, bit-identical

The two inputs are the same input. No threshold separates them, because the
model is never asked the question. `help me find a paper` reduces to
`help me find` — on which `Help_FindMyHearingAids` is the RIGHT answer,
confidently held, and wrong about the sentence the user actually said.

The word that makes an utterance out of scope is almost always rare and
specific — a brand, an object, a topic — so it is exactly the kind of word a
finite vocabulary lacks. Its absence is evidence, and this recovers it.

METHOD — out-of-fold, and the vocabulary matters
------------------------------------------------
The ratio is computed against a FOLD model's vocabulary, never the shipped one.
The shipped vocabulary was built from every training row, so scoring training
rows against it reports an out-of-vocabulary share no user will ever see —
which would fit a threshold to a fiction and push it far too low.

Each fold therefore refits the featurizer on its own training rows and scores
the held-out rows against that. Refitting per fold is the point, not overhead.

THE TRADE
---------
Raising the ratio (a laxer guard) lets more out-of-scope utterances reach an
action. Lowering it (a stricter guard) refuses more real commands, because a
legitimate command can also carry an unseen word. The costs are not equal and
are stated as arguments, not assumed — see `fit_decision_ladder` for the same
reasoning applied to the fire threshold.

USAGE
    python -m nlu_training.fit_oov_guard --lang en
    python -m nlu_training.fit_oov_guard --lang en --cost-wrong 10 --cost-reject 3
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from nlu_training.fit_calibration import (
    LR_KW, SEED, TFIDF_KW, cap_per_intent, eval_leakage_mask,
)
from nlu_engine.text_norm import normalize_text as featurize_text

BASE_DIR = Path(__file__).resolve().parents[3]
OOS = "Default Fallback Intent"

# Must match `IntentClassifier._TOKEN_RE`, which in turn matches sklearn's
# default `token_pattern`. Splitting text differently from the featurizer would
# count words the model was never offered.
TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")

# Deliberately fine. The fold-space value is multiplied by a ~2.7x correction
# before it ships, so a coarse grid becomes a coarse ANSWER: at 0.05 steps the
# shipped threshold carries +/-0.14 of quantisation error, which is wider than
# the range worth arguing about.
GRID = [round(x, 3) for x in np.arange(0.02, 0.601, 0.01)]
# 1.01 = never bypass (strict ratio-only guard), included so the sweep can
# choose it if the confidence condition turns out not to earn its place.
BYPASS_GRID = [0.80, 0.90, 0.95, 0.97, 0.99, 1.01]


def oov_ratio(text: str, vocab: set[str]) -> float:
    tokens = TOKEN_RE.findall(text.lower())
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t not in vocab) / len(tokens)


def oof_predictions(X, y, folds: int, T: float):
    """(pred, conf, ratio) per row, each from a model that never saw that row.

    `ratio` uses the FOLD's vocabulary — see the module docstring.
    """
    n = len(y)
    pred = np.empty(n, dtype=object)
    conf = np.zeros(n)
    ratio = np.zeros(n)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for k, (tr, te) in enumerate(skf.split(X, y), 1):
        pipe = Pipeline([("tfidf", TfidfVectorizer(**TFIDF_KW)),
                         ("clf", LogisticRegression(**LR_KW))]).fit(X[tr], y[tr])
        vocab = {w for w in pipe[0].vocabulary_ if " " not in w}
        z = pipe[-1].decision_function(pipe[0].transform(X[te])) / T
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p = p / p.sum(axis=1, keepdims=True)
        top = p.argmax(axis=1)
        pred[te] = pipe[-1].classes_[top]
        conf[te] = p[np.arange(len(te)), top]
        for j, i in enumerate(te):
            ratio[i] = oov_ratio(X[i], vocab)
        print(f"  fold {k}/{folds}: {len(tr)} train / {len(te)} scored, "
              f"vocab {len(vocab)} unigrams")
    return pred, conf, ratio


def fold_space_warning(X, ratio_oof, shipped_vocab) -> str:
    """Why the number above is a fold-space value and must not be shipped as-is.

    READ THIS BEFORE TRUSTING THE RECOMMENDATION.

    Out-of-fold is the right estimator for a CONFIDENCE threshold: a fold
    model's confidence and the shipped model's confidence are the same
    quantity. It is NOT for this one. The out-of-vocabulary ratio is defined
    against a vocabulary, and neither vocabulary available here is the one the
    guard will face:

      * against a FOLD vocabulary (what this fits), held-out rows carry words
        the fold model never saw but the shipped model has — ratios come out
        several times too high, so the threshold comes out too strict;
      * against the SHIPPED vocabulary, every training row is inside it BY
        CONSTRUCTION, so almost every row scores exactly 0.0 and the
        distribution has no tail to place a threshold in at all.

    Two corrections were tried and both fail for that one reason: rescaling by
    the ratio of the means (dominated by the shared mass at 0.0, so the
    quotient says nothing about the tail) and quantile matching (the shipped-
    vocabulary distribution has no usable quantiles above 0.0). They are not
    reported here because a plausible-looking wrong number is worse than an
    admitted gap — that is the failure this whole guard exists to undo.

    WHAT WOULD ACTUALLY FIT IT: a sample of real user utterances scored against
    the shipped vocabulary. Field telemetry already records `oov_ratio` on every
    blocked turn (`nlu.oov_guard`), so the distribution can be measured in
    production and this becomes fittable. Until then the shipped value is
    PROVISIONAL and should say so where it lives.
    """
    fold_mean = float(ratio_oof.mean())
    runtime = np.array([oov_ratio(x, shipped_vocab) for x in X])
    zero_share = float((runtime == 0.0).mean())
    return (f"fold-space mean ratio {fold_mean:.4f}; against the shipped "
            f"vocabulary {zero_share * 100:.1f}% of the same rows score exactly "
            f"0.0, which is why neither distribution can be mapped onto the "
            f"other")


def fit(lang: str, folds: int, fire: float, c_wrong: float, c_reject: float) -> int:
    calib = BASE_DIR / "models" / "intent" / lang / "calibration.json"
    if not calib.exists():
        print(f"FAIL: no calibration for {lang!r}. Run fit_calibration first.")
        return 1
    T = json.loads(calib.read_text(encoding="utf-8"))["temperature"]

    data_path = BASE_DIR / "language_packs" / lang / "train.csv"
    df = pd.read_csv(data_path, encoding="utf-8-sig").dropna(subset=["text", "intent"])
    df["text"] = df["text"].astype(str).map(featurize_text)
    df["intent"] = df["intent"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["text", "intent"])
    df = cap_per_intent(df)
    keep, _leaked, _checked = eval_leakage_mask(df["text"].values, lang)
    df = df[keep]

    X, y = df["text"].values, df["intent"].values
    print(f"  {len(y)} rows / {len(set(y))} intents, {folds}-fold OOF, T={T}, FIRE={fire}")
    pred, conf, ratio = oof_predictions(X, y, folds, T)

    truth = y
    fires = (pred != OOS) & (conf >= fire)
    is_oos = truth == OOS

    # Two dimensions, because the ratio alone cannot tell an entity value from a
    # foreign topic. Entity values are unknown BY NATURE — a contact name, a
    # brand, a free-text reminder topic can never all be in a finite vocabulary
    # — so a bare ratio refuses "send a message to john" (oov 0.25, conf 1.000)
    # alongside "help me find a paper" (oov 0.25, conf 0.771). The confidence
    # separates them: an unknown token is evidence only when the REST of the
    # utterance is also ambiguous, so the guard stands down above `bypass`.
    print(f"\n  {'ratio':>7}{'bypass':>8}{'OOS->action':>13}{'wrongAct':>10}"
          f"{'okFire':>9}{'lost':>7}{'cost':>9}")
    best = None
    rows = []
    for r in GRID:
        for bypass in BYPASS_GRID:
            blocked = (ratio >= r) & (conf < bypass)
            f = fires & ~blocked
            leak = int((f & is_oos).sum())
            wrong = int((f & ~is_oos & (pred != truth)).sum())
            ok = int((f & (pred == truth)).sum())
            lost = int((fires & blocked & (pred == truth)).sum())
            cost = leak * c_wrong + wrong * c_wrong + lost * c_reject
            rows.append((cost, r, bypass, leak, wrong, ok, lost))
            if best is None or cost < best[0]:
                best = (cost, r, bypass, leak, wrong, ok, lost)
    for cost, r, bypass, leak, wrong, ok, lost in sorted(rows)[:12]:
        print(f"  {r:>7.2f}{bypass:>8.2f}{leak:>13}{wrong:>10}{ok:>9}{lost:>7}{cost:>9.0f}")
    print("  (12 lowest-cost settings)")

    _cost, r, bypass, leak, wrong, ok, lost = best

    # Translate out of fold space into the space the engine actually runs in.
    print(f"\n  FOLD-SPACE best: ratio {r:.2f}, bypass {bypass:.2f}")
    print(f"    {leak} out-of-scope utterances still reach an action, "
          f"{wrong} wrong actions,")
    print(f"    {ok} correct fires, {lost} correct commands refused")
    print(f"    costs used: wrong action={c_wrong}, refusal={c_reject}")

    import joblib
    pipe_path = BASE_DIR / "models" / "intent" / lang / "pipeline.pkl"
    print()
    print("  ratio: DO NOT SHIP THIS NUMBER DIRECTLY")
    if pipe_path.exists():
        shipped = {w for w in joblib.load(str(pipe_path))[0].vocabulary_ if " " not in w}
        print(f"    {fold_space_warning(X, ratio, shipped)}.")
    print("    The out-of-vocabulary ratio is defined against a vocabulary, and")
    print("    neither vocabulary available at fit time is the one the guard will")
    print("    face — see `fold_space_warning` for why the two obvious")
    print("    corrections do not work. Choose the shipped value deliberately,")
    print("    mark it PROVISIONAL where it lives, and re-fit from field")
    print("    telemetry (`nlu.oov_guard` records oov_ratio per blocked turn).")
    print()
    print(f"  bypass_confidence = {bypass:.2f} IS fit soundly — it is a confidence,")
    print("    the one quantity out-of-fold estimates correctly here.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fire", type=float, default=0.70,
                    help="the fire threshold this guard sits in front of")
    # Relative, not absolute — only the ratio matters. A refusal costs the user
    # a repeat; a wrong action changes the device. PLACEHOLDERS FOR A PRODUCT
    # DECISION, same as fit_decision_ladder.
    ap.add_argument("--cost-wrong", type=float, default=10.0)
    ap.add_argument("--cost-reject", type=float, default=3.0)
    a = ap.parse_args(argv)
    return fit(a.lang, a.folds, a.fire, a.cost_wrong, a.cost_reject)


if __name__ == "__main__":
    sys.exit(main())
