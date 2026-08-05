#!/usr/bin/env python3
"""Fit the uncertainty-confirmation gate out-of-fold.

WHAT THE GATE DOES
------------------
When the classifier predicts a STATE-CHANGING intent below
``uncertain_confirm.below_confidence``, the engine asks before acting
("Just to be sure — turn the volume down?") instead of firing. A wrong
prediction inside the band becomes a question, not a wrong action.

WHY IT IS FIT HERE AND NOT ON THE HOLDOUT
-----------------------------------------
Sweeping the band on ``holdout_honest.csv`` and then reporting the resulting
wrong-action count on that same file fits a threshold to the one unbiased
estimate the project has — the same failure as blocker B9, one layer up. So the
band is fit on OUT-OF-FOLD predictions over ``train.csv``: every row scored by a
fold model that never saw it, which gives leakage-free confidences without
sacrificing data or retraining the shipped model. The holdout is then used ONCE,
to confirm.

THE TRADE BEING OPTIMISED
-------------------------
Raising the band converts wrong actions into confirmations, and also converts
some CORRECT actions into confirmations. Those costs are not equal:

  * a wrong action fires something the user did not ask for;
  * a confirmation costs one extra turn and the command still executes.

So this maximises wrong actions caught subject to a ceiling on how much correct
traffic gets an extra turn (``--max-friction``, default 10%).

USAGE
    python -m nlu_training.fit_confirm_gate --lang en
    python -m nlu_training.fit_confirm_gate --lang en --max-friction 0.10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nlu_training.fit_calibration import (
    cap_per_intent, eval_leakage_mask, oof_logits,
)

# The FEATURISATION normaliser the trainer applies (contraction expansion +
# apostrophe folding), imported under a distinct name because
# `nlu_training.leakage.normalize_text` is a DIFFERENT function used for
# set-comparison during leakage checks. Confusing the two silently changes what
# is being fitted.
from nlu_engine.text_norm import normalize_text as featurize_text

BASE_DIR = Path(__file__).resolve().parents[3]
GRID = [round(x, 2) for x in np.arange(0.70, 1.005, 0.01)]


def is_actionable(label: str) -> bool:
    return bool(label) and not label.startswith(("help.", "sys."))


def is_read_only(label: str, read_only_intents: set[str]) -> bool:
    return label in read_only_intents or label.rsplit(".", 1)[-1] == "query"


def state_changing(schema: dict) -> set[str]:
    """Intents that change device/app state — what the gate must cover."""
    ro = {"device.status.battery"}
    return {i for i in schema.get("intents", {})
            if is_actionable(i) and not is_read_only(i, ro)}


def fit(lang: str, folds: int, max_friction: float) -> int:
    schema = json.loads((BASE_DIR / "content" / "nlu_schema.json")
                        .read_text(encoding="utf-8"))
    sc = state_changing(schema)
    calib_path = BASE_DIR / "models" / "intent" / lang / "calibration.json"
    if not calib_path.exists():
        print(f"FAIL: no calibration for {lang!r}. Run fit_calibration first.")
        return 1
    T = json.loads(calib_path.read_text(encoding="utf-8"))["temperature"]

    data_path = BASE_DIR / "datasets" / lang / "train.csv"
    df = pd.read_csv(data_path, encoding="utf-8-sig").dropna(subset=["text", "intent"])
    # MUST mirror train.py, which applies text_norm.normalize_text before
    # fitting and before the ONNX export. Lower+strip alone leaves
    # apostrophes in, so the fitted values would describe a featurizer the
    # shipped model does not use — blocker B8 in a new place.
    df["text"] = df["text"].astype(str).map(featurize_text)
    df["intent"] = df["intent"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["text", "intent"])
    df = cap_per_intent(df)
    keep, _leaked, _checked = eval_leakage_mask(df["text"].values, lang)
    df = df[keep]

    counts = df["intent"].value_counts()
    if (counts < folds).any():
        print(f"FAIL: intents with < {folds} rows: {dict(counts[counts < folds])}")
        return 1

    X, y = df["text"].values, df["intent"].values
    print(f"  {len(y)} rows / {len(set(y))} intents, {folds}-fold OOF, T={T}")
    logits, classes, y_idx = oof_logits(X, y, folds)

    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(axis=1, keepdims=True)
    top = p.argmax(axis=1)
    conf = p.max(axis=1)
    pred = classes[top]
    truth = classes[y_idx]

    # Only predictions of a state-changing intent reach the gate at all.
    gated = np.array([lbl in sc for lbl in pred])
    wrong = gated & (pred != truth)
    right = gated & (pred == truth)
    n_wrong, n_right = int(wrong.sum()), int(right.sum())
    print(f"  OOF state-changing predictions: {n_right} correct / {n_wrong} wrong\n")

    print(f"  {'below':>7}{'wrong caught':>14}{'%wrong':>9}"
          f"{'correct asked':>15}{'friction':>10}")
    best = None
    for b in GRID:
        caught = int((wrong & (conf < b)).sum())
        asked = int((right & (conf < b)).sum())
        friction = asked / max(n_right, 1)
        mark = ""
        if friction <= max_friction and (best is None or caught > best[1]):
            best, mark = (b, caught, asked, friction), ""
        if abs(b * 100 % 5) < 1e-6:
            print(f"  {b:>7.2f}{caught:>14}{caught/max(n_wrong,1)*100:>8.1f}%"
                  f"{asked:>15}{friction*100:>9.1f}%{mark}")

    if best is None:
        print(f"\n  no band satisfies friction <= {max_friction:.0%}")
        return 1
    b, caught, asked, friction = best
    print(f"\n  RECOMMENDED below_confidence = {b:.2f}")
    print(f"    catches {caught}/{n_wrong} OOF wrong state-changing predictions "
          f"({caught/max(n_wrong,1)*100:.1f}%)")
    print(f"    asks about {asked}/{n_right} correct ones ({friction*100:.1f}% friction)")
    print(f"    gate must cover all {len(sc)} state-changing intents")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-friction", type=float, default=0.10,
                    help="max share of CORRECT state-changing fires that may "
                         "become a confirmation (default 0.10)")
    a = ap.parse_args(argv)
    return fit(a.lang, a.folds, a.max_friction)


if __name__ == "__main__":
    sys.exit(main())
