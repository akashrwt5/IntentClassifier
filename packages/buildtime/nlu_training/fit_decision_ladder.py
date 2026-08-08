#!/usr/bin/env python3
"""Fit the decision ladder — FIRE and FLOOR — out-of-fold.

THE LADDER
----------
    conf >= FIRE           -> FULFILL
    FLOOR <= conf < FIRE   -> CONFIRM   (recovery: this would otherwise reject)
    conf <  FLOOR          -> FALLBACK

Two numbers, replacing the previous arrangement of `confidence_threshold`,
`uncertain_confirm.below_confidence`, `uncertain_confirm.confirm_floor` and a
hand-curated 14-intent gated list.

WHY THE OLD SHAPE WAS WRONG (not just mis-tuned)
------------------------------------------------
`below_confidence` (0.91) sat ABOVE the fire threshold (0.70), so it converted
commands that would have fired into questions. Measured on the honest holdout,
that band produced 103 friction turns against 16 useful catches — 85% of every
confirmation shown to a user was asked about a CORRECT prediction.

A band BELOW the fire threshold does the opposite: it converts turns that would
have been REJECTED into questions. Same word, opposite mechanism. Only the lower
one is kept, and `FLOOR` is named for what it does rather than for a gate.

WHY THIS REPLACES fit_confirm_gate.py
-------------------------------------
That script swept one threshold and counted two outcomes. It could not see
rejections at all, which is why the friction side of the trade was invisible
while it grew — budgeting one side of a two-sided trade is how the suite stayed
green at 85% friction. This sweeps both thresholds and counts all four turn
outcomes.

ARBITRATION IS SIMULATED, NOT IGNORED
-------------------------------------
At runtime `classifier.classify` arbitrates between the keyword rules and the
model: a corroborated keyword hit reports the model's calibrated probability, a
contested one reports `CONTESTED_CONFIDENCE`. Fitting against raw OOF model
confidence would therefore fit a distribution the runtime never produces. The
keyword rules are replayed here so the fitted confidences are the ones the
engine will actually compare against these thresholds.

SEMANTIC RESCUE
---------------
FIRE is semantic-independent: the rescue stage only runs BELOW the fire
threshold, so it cannot affect the wrong-action trade FIRE controls.

FLOOR is NOT. It splits confirm-from-reject among sub-FIRE turns, and semantic
rescue intercepts exactly those before FLOOR is consulted. The value fitted here
therefore describes the **semantic-disabled** configuration, which is what this
tree currently runs (`NLUEngine.semantic is None` — the MiniLM artifacts do not
load). Re-run with the stage live before shipping a build that enables it.

USAGE
    python -m nlu_training.fit_decision_ladder --lang en
    python -m nlu_training.fit_decision_ladder --lang en --max-friction 0.05
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
from nlu_engine.classifier import IntentClassifier, _compile_keyword_rules, _is_negated

BASE_DIR = Path(__file__).resolve().parents[3]

FIRE_GRID = [float(round(x, 2)) for x in np.arange(0.50, 1.001, 0.01)]
FLOOR_GRID = [float(round(x, 2)) for x in np.arange(0.30, 0.901, 0.05)]

OOS = "Default Fallback Intent"


def keyword_intent(rules, text: str, cues) -> str | None:
    """Replay `IntentClassifier._keyword_match` over the fit corpus.

    Kept structurally identical to the runtime loop; if that changes, this must
    change with it or the fit stops describing the shipped engine.
    """
    t = text.lower().strip()
    for rule in rules:
        kind = rule.get("type")
        if kind == "contains":
            hit = next((term for term in rule["terms"] if term in t), None)
            if hit and not _is_negated(t, hit, cues):
                return rule["intent"]
        elif kind == "exact":
            if t in rule["terms"]:
                return rule["intent"]
        elif kind == "regex":
            if rule["pattern"].search(t):
                if rule["not_pattern"] is None or not rule["not_pattern"].search(t):
                    return rule["intent"]
    return None


def arbitrate(rules, texts, model_pred, model_conf, cues, contested):
    """Apply the runtime's rule/model arbitration to OOF model predictions."""
    pred = list(model_pred)
    conf = list(model_conf)
    n_corroborated = n_contested = 0
    for i, text in enumerate(texts):
        kw = keyword_intent(rules, text, cues)
        if kw is None:
            continue
        if kw == model_pred[i]:
            n_corroborated += 1          # label and confidence both stand
        else:
            n_contested += 1
            pred[i] = kw                 # rule owns the label
            conf[i] = contested          # ...but not a calibrated confidence
    return np.array(pred), np.array(conf), n_corroborated, n_contested


def outcomes(pred, conf, truth, fire: float, floor: float):
    """(wrong_fire, right_fire, friction, useful, lost) for one ladder setting.

    friction — a CORRECT prediction that had to be confirmed (an extra turn)
    useful   — a WRONG prediction caught by a confirmation (an averted action)
    lost     — a valid command that fell below FLOOR and was rejected outright
    """
    is_oos = truth == OOS
    fires = conf >= fire
    asks = (conf >= floor) & ~fires
    rejects = ~fires & ~asks
    correct = pred == truth

    # Firing on an utterance whose truth is OOS is itself a wrong action, so
    # `is_oos` is deliberately not excluded here — only from `lost`, where
    # rejecting an OOS utterance is the correct outcome rather than a loss.
    wrong_fire = int((fires & ~correct).sum())
    right_fire = int((fires & correct).sum())
    friction = int((asks & correct).sum())
    useful = int((asks & ~correct).sum())
    lost = int((rejects & ~is_oos).sum())
    return wrong_fire, right_fire, friction, useful, lost


def fit(lang: str, folds: int, max_friction: float, contested: float,
        c_wrong: float, c_ask: float, c_reject: float) -> int:
    schema_path = BASE_DIR / "language_packs" / lang / "nlu_schema.json"
    if not schema_path.exists():
        schema_path = BASE_DIR / "content" / "nlu_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    calib_path = BASE_DIR / "models" / "intent" / lang / "calibration.json"
    if not calib_path.exists():
        print(f"FAIL: no calibration for {lang!r}. Run fit_calibration first.")
        return 1
    T = json.loads(calib_path.read_text(encoding="utf-8"))["temperature"]

    data_path = BASE_DIR / "language_packs" / lang / "train.csv"
    if not data_path.exists():
        data_path = BASE_DIR / "datasets" / lang / "train.csv"
    df = pd.read_csv(data_path, encoding="utf-8-sig").dropna(subset=["text", "intent"])
    # MUST mirror train.py, which applies text_norm.normalize_text before
    # fitting and before the ONNX export. Lower+strip alone leaves apostrophes
    # in, so the fitted values would describe a featurizer the shipped model
    # does not use — blocker B8 in a new place.
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
    model_pred = classes[p.argmax(axis=1)]
    model_conf = p.max(axis=1)
    truth = classes[y_idx]

    rules = _compile_keyword_rules(schema)
    cues = tuple(schema.get("negation_cues") or ())
    if not cues:
        from nlu_engine.classifier import _DEFAULT_NEGATIONS
        cues = _DEFAULT_NEGATIONS
    pred, conf, n_corr, n_cont = arbitrate(
        rules, X, model_pred, model_conf, cues, contested)

    kw_total = n_corr + n_cont
    print(f"  keyword rules fire on {kw_total} rows "
          f"({kw_total / len(y) * 100:.1f}%): "
          f"{n_corr} corroborated / {n_cont} contested (conf := {contested})")
    if n_cont:
        cont_mask = conf == contested
        print(f"  contested rows are {(pred[cont_mask] == truth[cont_mask]).mean() * 100:.1f}% "
              f"correct — this is what decides which band they belong in")
    print()

    n_correct_total = int((pred == truth).sum())

    # ---- sweep -------------------------------------------------------------
    # Minimise EXPECTED COST, not any single outcome. A threshold is a proxy
    # for a cost ratio; leaving the ratio implicit is what let this ladder drift
    # to 85% friction with a green suite. The three costs are arguments because
    # they are a product and clinical judgement, not an ML one.
    best = None
    rows = []
    for fire in FIRE_GRID:
        for floor in FLOOR_GRID:
            if floor >= fire:
                continue
            wf, rf, fr, us, lost = outcomes(pred, conf, truth, fire, floor)
            friction_rate = fr / max(n_correct_total, 1)
            cost = wf * c_wrong + (fr + us) * c_ask + lost * c_reject
            rows.append((fire, floor, wf, rf, fr, us, lost, friction_rate, cost))
            if friction_rate <= max_friction:
                if best is None or cost < best[0]:
                    best = (cost, fire, floor, wf, rf, fr, us, lost, friction_rate)

    print(f"  costs: wrong action={c_wrong}  confirmation={c_ask}  rejection={c_reject}")
    print(f"  {'FIRE':>6}{'FLOOR':>7}{'wrongAct':>10}{'okFire':>9}"
          f"{'friction':>10}{'useful':>8}{'lost':>7}{'friction%':>11}{'cost':>9}")
    top = sorted(rows, key=lambda r: r[8])[:12]
    for fire, floor, wf, rf, fr, us, lost, fr_rate, cost in top:
        print(f"  {fire:>6.2f}{floor:>7.2f}{wf:>10}{rf:>9}"
              f"{fr:>10}{us:>8}{lost:>7}{fr_rate * 100:>10.1f}%{cost:>9.0f}")
    print("  (12 lowest-cost settings)")

    if best is None:
        print(f"\n  no ladder satisfies friction <= {max_friction:.0%}")
        return 1
    _cost, fire, floor, wf, rf, fr, us, lost, fr_rate = best
    print(f"\n  RECOMMENDED  FIRE = {fire:.2f}   FLOOR = {floor:.2f}")
    print(f"    {wf} wrong actions, {rf} correct fires")
    print(f"    {fr + us} confirmations: {us} averted a wrong action, "
          f"{fr} were friction ({fr_rate * 100:.1f}% of correct predictions)")
    print(f"    {lost} valid commands rejected outright")
    print()
    # The recommendation moves with the cost ratios, so show how much. A value
    # that is stable across plausible ratios can be adopted; one that swings is
    # a signal that the ratio has to be decided deliberately rather than
    # inherited from this script's defaults.
    print("  sensitivity — FIRE/FLOOR under other cost ratios:")
    for cw, ca, cr in ((5, 1, 3), (10, 1, 3), (20, 1, 3), (10, 1, 1), (10, 2, 3)):
        alt = min(((wf2 * cw + (fr2 + us2) * ca + lo2 * cr, f2, fl2)
                   for f2, fl2, wf2, _rf2, fr2, us2, lo2, frr, _c in rows
                   if frr <= max_friction), key=lambda t: t[0])
        print(f"    wrong={cw:>2} ask={ca} reject={cr}  ->  "
              f"FIRE {alt[1]:.2f} / FLOOR {alt[2]:.2f}")
    print()
    print("  These are OUT-OF-FOLD estimates on train.csv. Confirm ONCE on")
    print("  holdout_honest.csv, then stop touching it — tuning against the")
    print("  holdout is blocker B9.")
    print("  FLOOR assumes semantic rescue is DISABLED. Re-fit before shipping")
    print("  a build that enables it — the stage intercepts exactly the")
    print("  sub-FIRE turns FLOOR is deciding about.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-friction", type=float, default=1.0,
                    help="optional guardrail: max share of CORRECT predictions "
                         "that may become a confirmation. Off by default — the "
                         "cost model below is the real objective")
    ap.add_argument("--contested", type=float,
                    default=IntentClassifier.CONTESTED_CONFIDENCE,
                    help="confidence assigned when a keyword rule and the model "
                         "disagree (default: the runtime's current value)")
    # Relative, not absolute — only the ratios matter. Defaults encode: a wrong
    # action costs ~10 confirmations, and a rejection ~3, because a confirmation
    # still completes the command in one extra turn while a rejection completes
    # nothing. THESE ARE PLACEHOLDERS FOR A PRODUCT DECISION.
    ap.add_argument("--cost-wrong", type=float, default=10.0)
    ap.add_argument("--cost-ask", type=float, default=1.0)
    ap.add_argument("--cost-reject", type=float, default=3.0)
    a = ap.parse_args(argv)
    return fit(a.lang, a.folds, a.max_friction, a.contested,
               a.cost_wrong, a.cost_ask, a.cost_reject)


if __name__ == "__main__":
    sys.exit(main())
