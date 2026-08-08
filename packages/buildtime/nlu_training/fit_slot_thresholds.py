#!/usr/bin/env python3
"""Fit the two slot-flow thresholds, with provenance.

Both values shipped without any recorded derivation, which is how the
confirmation band came to sit at 0.80 against a temperature that had already
been shown wrong (Review-F5 blocker B8). These are the last two unfitted
confidence knobs in the English path.

------------------------------------------------------------------------------
1. slot_confidence_threshold  (default 0.60)
------------------------------------------------------------------------------
An intent that OWNS SLOTS fires at a lower bar than a fire-and-forget one,
because entering the flow only produces a QUESTION ("What is the name of the
memory?"). The user sees the question and can walk away; nothing changes on the
device. So the two errors are asymmetric:

  * too HIGH -> a valid slot-bearing command is deflected to fallback; the user
    is told the system did not understand something it did understand.
  * too LOW  -> the user is asked an irrelevant question. Mildly annoying,
    recoverable, and NOT a wrong action.

The frontier is very flat (every bar from 0.30 to 0.90 keeps >= 89% recall for
<= 0.31% of turns), so a single argmax point would be over-fitting a curve that
barely moves. Instead the admissible RANGE is computed — recall >= 95%,
irrelevant asks <= ``--max-irrelevant`` of all turns, and never above
``confidence_threshold`` — and its midpoint taken, which is robust to where in
the band the value lands.

------------------------------------------------------------------------------
2. interrupt_threshold  (was a hardcoded 0.75 in engine.py)
------------------------------------------------------------------------------
Mid-slot-filling, if the classifier predicts a DIFFERENT intent at or above this
bar, the pending flow is abandoned and the new intent handled instead. The
failure mode that matters is mistaking a SLOT ANSWER for a new command: the
engine asks "What do you want to be reminded?", the user says "take medication",
and if that scores highly as some other intent the flow is destroyed.

The engine's own comment named that exact case, and the negatives were already
in the repo: enum surface forms in ``content/nlu_entities.json`` are the things a
user says while ANSWERING a prompt.

Auditing them found a live wrong-action bug that no single-turn holdout could
see. Several MEMORY names are also commands, so "What is the name of the memory?"
answered with "mute" fired Cmd.VolumeMute at 0.980 — the device muted instead
of switching to the Mute memory. "quiet" muted too; "telephone" rang the phone.
No bar can fix that ("tinnitus" and "mask" classify at 1.000), so the engine now
resolves it by PRECEDENCE: ``NLUEngine._answers_awaited_slot`` treats a valid
value for the awaited slot as the answer, whatever it classifies as.

That leaves this bar responsible only for OPEN free-text slots, where every
utterance is a valid value so no such test exists. Only three such phrases would
interrupt at all, and all three resolve to read-only ``help.*`` intents, so the
bar is no longer doing safety work — precedence is. It is therefore fitted for
genuine-topic-switch recall under a hard safety constraint: zero STATE-CHANGING
leaks, admit >= 95% of genuine commands, then the highest such bar.

The old 0.75 was justified as "lowered from 0.85 after isotonic calibration".
The pipeline calibrates by temperature scaling now (T from ``fit_calibration``),
so that value was tuned against a confidence scale that no longer exists — and
as a class constant no language pack could override it.

USAGE
    python -m nlu_training.fit_slot_thresholds --lang en
    python -m nlu_training.fit_slot_thresholds --lang en --write
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nlu_training.fit_calibration import (
    cap_per_intent, _softmax, eval_leakage_mask, oof_logits,
)
from nlu_training.leakage import normalize_text

# The FEATURISATION normaliser the trainer applies (contraction expansion +
# apostrophe folding), imported under a distinct name because
# `nlu_training.leakage.normalize_text` is a DIFFERENT function used for
# set-comparison during leakage checks. Confusing the two silently changes what
# is being fitted.
from nlu_engine.text_norm import normalize_text as featurize_text

BASE_DIR = Path(__file__).resolve().parents[3]
GRID = [round(x, 2) for x in np.arange(0.30, 1.005, 0.01)]


def _schema() -> dict:
    return json.loads((BASE_DIR / "content" / "nlu_schema.json")
                      .read_text(encoding="utf-8"))


def slot_bearing(schema: dict) -> set[str]:
    return {name for name, cfg in schema.get("intents", {}).items()
            if cfg.get("slots")}


def entity_owners(schema: dict) -> dict[str, set[str]]:
    """entity name -> the intents that collect it as a slot."""
    owners: dict[str, set[str]] = {}
    for name, cfg in schema.get("intents", {}).items():
        for slot in cfg.get("slots", []) or []:
            owners.setdefault(slot["entity"], set()).add(name)
    return owners


def unprotected_slot_answers(schema: dict) -> list[tuple[str, str, set[str]]]:
    """(phrase, entity, owning intents) for answers the threshold must still judge.

    `NLUEngine._answers_awaited_slot` already protects CLOSED enum slots: an
    utterance that resolves to a valid value for the awaited slot is treated as
    the answer and cannot interrupt, whatever it classifies as. That fix removed
    the dangerous cases ("mute" and "quiet" as memory names used to MUTE the
    device at 0.98).

    What the bar still has to decide is the OPEN free-text slots, where every
    utterance is a valid value so no such test is possible. Those are the only
    genuine negatives left, so they are the only ones fitted against.
    """
    ents = json.loads((BASE_DIR / "content" / "nlu_entities.json")
                      .read_text(encoding="utf-8"))
    owners = entity_owners(schema)
    out: list[tuple[str, str, set[str]]] = []
    for entity, spec in ents.items():
        if not isinstance(spec, dict) or spec.get("type") != "enum":
            continue
        if not spec.get("open"):
            continue        # closed enum — protected by precedence, not by the bar
        for synonyms in spec.get("values", {}).values():
            for s in synonyms:
                if s and s.strip():
                    out.append((s.lower().strip(), entity, owners.get(entity, set())))
    return out


def _oof(lang: str, folds: int, T: float):
    path = BASE_DIR / "datasets" / lang / "train.csv"
    df = pd.read_csv(path, encoding="utf-8-sig").dropna(subset=["text", "intent"])
    # MUST mirror train.py, which applies text_norm.normalize_text before
    # fitting and before the ONNX export. Lower+strip alone leaves
    # apostrophes in, so the fitted values would describe a featurizer the
    # shipped model does not use — blocker B8 in a new place.
    df["text"] = df["text"].astype(str).map(featurize_text)
    df["intent"] = df["intent"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["text", "intent"])
    df = cap_per_intent(df)
    keep, _l, _c = eval_leakage_mask(df["text"].values, lang)
    df = df[keep]
    counts = df["intent"].value_counts()
    if (counts < folds).any():
        raise SystemExit(f"intents with < {folds} rows: {dict(counts[counts < folds])}")
    X, y = df["text"].values, df["intent"].values
    logits, classes, y_idx = oof_logits(X, y, folds)
    probs = _softmax(logits, T)
    return X, classes, y_idx, probs


def fit(lang: str, folds: int, max_irrelevant: float, write: bool) -> int:
    schema = _schema()
    calib_path = BASE_DIR / "models" / "intent" / lang / "calibration.json"
    if not calib_path.exists():
        print(f"FAIL: no calibration for {lang!r}; run fit_calibration first.")
        return 1
    T = json.loads(calib_path.read_text(encoding="utf-8"))["temperature"]
    slotted = slot_bearing(schema)
    print(f"  T={T}  slot-bearing intents: {sorted(slotted)}\n")

    texts, classes, y_idx, probs = _oof(lang, folds, T)
    pred = classes[probs.argmax(axis=1)]
    conf = probs.max(axis=1)
    truth = classes[y_idx]

    # ---------------- 1. slot_confidence_threshold -------------------------
    enters = np.array([p in slotted for p in pred])
    good = enters & (pred == truth)
    bad = enters & (pred != truth)
    n_good, n_bad = int(good.sum()), int(bad.sum())
    n_turns = len(conf)
    print(f"  OOF predictions of a slot-bearing intent: "
          f"{n_good} correct / {n_bad} wrong  (of {n_turns} turns)")
    # The cost is measured against ALL traffic, not against the 22 wrong
    # predictions: an irrelevant question is only a problem in proportion to how
    # often a user meets one. Expressing it as a share of the wrong predictions
    # would let 22 rare rows veto recall for 965 common ones.
    print(f"\n  {'thresh':>7}{'correct kept':>14}{'recall':>9}"
          f"{'irrelevant asks':>17}{'% of traffic':>14}")
    # The frontier here is very flat, so picking the single best point would be
    # over-fitting a curve that barely moves. Instead take the RANGE that meets
    # both stated criteria and recommend its midpoint — a value robust to where
    # in the band it lands, and reported as such.
    base = schema.get("confidence_threshold", 0.70)
    MIN_RECALL = 0.95
    ok = []
    for t in GRID:
        kept = int((good & (conf >= t)).sum())
        asks = int((bad & (conf >= t)).sum())
        share = asks / max(n_turns, 1)
        recall = kept / max(n_good, 1)
        # A slot-bearing intent must never need MORE confidence than a
        # fire-and-forget one — that inverts the design intent, since entering a
        # flow only asks a question while firing changes device state.
        if recall >= MIN_RECALL and share <= max_irrelevant and t <= base:
            ok.append((t, kept, asks, share, recall))
        if abs(t * 100 % 10) < 1e-6:
            print(f"  {t:>7.2f}{kept:>14}{recall*100:>8.1f}%"
                  f"{asks:>17}{share*100:>13.2f}%")
    if not ok:
        print(f"\n  no threshold with recall >= {MIN_RECALL:.0%}, irrelevant asks "
              f"<= {max_irrelevant:.2%} of traffic, and value <= {base}")
        return 1
    lo, hi = ok[0][0], ok[-1][0]
    st = round((lo + hi) / 2, 2)
    chosen = min(ok, key=lambda r: abs(r[0] - st))
    st, kept, asks, share, recall = chosen
    print(f"\n  admissible range [{lo:.2f}, {hi:.2f}]  "
          f"(recall >= {MIN_RECALL:.0%}, asks <= {max_irrelevant:.2%} of turns, "
          f"<= base {base})")
    print(f"  RECOMMENDED slot_confidence_threshold = {st:.2f}  (midpoint)")
    print(f"    keeps {kept}/{n_good} correct slot-flow entries ({recall*100:.1f}%)")
    print(f"    admits {asks} irrelevant questions = {share*100:.2f}% of turns")

    # ---------------- 2. interrupt_threshold -------------------------------
    print("\n" + "-" * 70)
    candidates = unprotected_slot_answers(schema)
    train_norm = {normalize_text(t) for t in texts}
    held = [(p, e, own) for p, e, own in candidates
            if normalize_text(p) not in train_norm]
    print(f"  open-slot answer phrases: {len(candidates)} "
          f"({len(held)} absent from train.csv, so genuinely out-of-sample)")

    from nlu_engine.classifier import IntentClassifier, _stable_softmax
    md = BASE_DIR / "models" / "intent" / lang
    clf = IntentClassifier(model_path=md / "model.onnx",
                           labels_path=md / "labels.pkl",
                           calibration_path=md / "calibration.json")
    labels = list(clf.labels)

    def predict(phrase: str) -> tuple[str, float]:
        s = clf.backend.tfidf_logits(phrase)
        if isinstance(s, dict):
            s = np.array([s[lbl] for lbl in labels], dtype=float)
        p = _stable_softmax(np.asarray(s, dtype=float) / T)
        i = int(p.argmax())
        return labels[i], float(p[i])

    # A phrase is only a negative if it would ACTUALLY interrupt: the engine
    # requires a different intent that is not the OOS label. A recurrence answer
    # scoring 0.856 for Default Fallback Intent is already safe.
    neg_rows = []
    for phrase, entity, owners in held:
        intent, c = predict(phrase)
        if intent != "Default Fallback Intent" and intent not in owners:
            neg_rows.append((c, phrase, intent, entity))
    neg = np.array([c for c, *_ in neg_rows]) if neg_rows else np.array([0.0])
    print(f"  of those, {len(neg_rows)} would interrupt (different non-OOS intent):")
    for c, phrase, intent, entity in sorted(neg_rows, reverse=True)[:8]:
        print(f"     {c:.3f}  {phrase!r:<26} -> {intent}  [@{entity}]")
    # Positives: genuine commands, scored out-of-fold so memorisation cannot
    # inflate them. A real topic switch names an actionable intent.
    actionable = np.array([not p.startswith(("help.", "sys.")) for p in pred])
    pos = conf[actionable & (pred == truth)]

    print(f"\n  answers (must NOT interrupt): median {np.median(neg):.3f}  "
          f"max {neg.max():.3f}   n={len(neg_rows)}")
    print(f"  commands (SHOULD interrupt) : median {np.median(pos):.3f}  "
          f"p10 {np.percentile(pos, 10):.3f}   n={len(pos)}")
    print(f"\n  {'thresh':>7}{'answers leaking':>17}{'commands admitted':>19}{'rate':>8}")
    # A bar that admits nothing leaks nothing, so minimising leakage alone drives
    # this to 1.0 and disables interruption entirely. Three criteria instead, in
    # priority order:
    #
    #   1. HARD SAFETY — no answer that would fire a STATE-CHANGING intent may
    #      clear the bar. Destroying a flow to show a help page is recoverable;
    #      destroying it to change the device is the class of bug that
    #      `_answers_awaited_slot` was added to fix.
    #   2. A genuine topic switch must still be honoured (>= MIN_ADMIT).
    #   3. Among the rest, take the HIGHEST bar — fewest flows destroyed.
    #
    # Note the safety constraint is now cheap to satisfy: closed enum slots are
    # protected by precedence, so every remaining negative maps to a read-only
    # help.* intent except one scoring 0.491, well under any usable bar.
    MIN_ADMIT = 0.95
    state_changing = {i for i in schema.get("intents", {})
                      if not i.startswith(("help.", "sys."))
                      and i != "Cmd.BatteryLevel"
                      and i.rsplit(".", 1)[-1] != "query"}
    sc_neg = np.array([c for c, _p, intent, _e in neg_rows
                       if intent in state_changing] or [0.0])
    admissible = []
    for t in GRID:
        leak = int((neg >= t).sum())
        adm = int((pos >= t).sum())
        sc_leak = int((sc_neg >= t).sum())
        if sc_leak == 0 and adm / max(len(pos), 1) >= MIN_ADMIT:
            admissible.append((t, leak, adm))
        if abs(t * 100 % 10) < 1e-6:
            print(f"  {t:>7.2f}{leak:>17}{adm:>19}{adm/max(len(pos),1)*100:>7.1f}%")
    if not admissible:
        print(f"\n  no bar with zero state-changing leaks that still admits "
              f">= {MIN_ADMIT:.0%} of genuine commands")
        return 1
    it, leak, adm = max(admissible)          # highest qualifying bar
    leak_rate = leak / max(len(neg_rows), 1)
    print(f"\n  criteria: zero STATE-CHANGING leaks, admit >= {MIN_ADMIT:.0%} of "
          f"genuine commands, then take the highest such bar")
    print(f"\n  RECOMMENDED interrupt_threshold = {it:.2f}")
    print(f"    {leak}/{len(neg)} slot answers would wrongly interrupt "
          f"({leak_rate*100:.1f}%)")
    print(f"    {adm}/{len(pos)} genuine commands can interrupt "
          f"({adm/max(len(pos),1)*100:.1f}%)")

    payload = {
        "_note": "Slot-flow confidence thresholds, fitted. slot_confidence_threshold "
                 "is fit OUT-OF-FOLD over train.csv; interrupt_threshold is fit "
                 "against enum slot-answer surface forms (the negatives) versus "
                 "out-of-fold genuine commands (the positives). Neither is tuned "
                 "on holdout_honest.csv.",
        "slot_confidence_threshold": st,
        "interrupt_threshold": it,
        "provenance": {
            "method": "out-of-fold NLL-calibrated confidence sweep",
            "folds": folds, "temperature": T,
            "slot_confidence": {
                "objective": f"midpoint of the range with recall >= 95%, "
                             f"irrelevant asks <= {max_irrelevant:.2%} of turns, "
                             f"and value <= confidence_threshold (a slot intent "
                             f"must never need MORE confidence than a "
                             f"fire-and-forget one)",
                "admissible_range": [lo, hi],
                "correct_kept": kept, "correct_total": n_good,
                "irrelevant_admitted": asks, "irrelevant_total": n_bad,
            },
            "interrupt": {
                "objective": "zero STATE-CHANGING slot-answer leaks (hard), "
                             "admit >= 95% of genuine commands, then the highest "
                             "such bar (fewest flows destroyed)",
                "negatives": "OPEN free-text entity surface forms in "
                             "content/nlu_entities.json, absent from train.csv, "
                             "that classify as a different non-OOS intent. Closed "
                             "enum slots are protected by precedence instead "
                             "(NLUEngine._answers_awaited_slot), not by this bar.",
                "n_negatives": int(len(neg_rows)), "leaking": leak,
                "leaking_are_read_only": True,
                "n_positives": int(len(pos)), "admitted": adm,
                "supersedes": "hardcoded NLUEngine.INTERRUPT_THRESHOLD = 0.75, "
                              "justified as 'lowered from 0.85 after isotonic "
                              "calibration' — the pipeline uses temperature "
                              "scaling, so that scale no longer exists",
            },
            "source": f"datasets/{lang}/train.csv",
            "fitted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fitted_by": "nlu_training.fit_slot_thresholds",
        },
    }
    if not write:
        print("\n  (not written — pass --write to persist)")
        return 0
    out = BASE_DIR / "models" / "intent" / lang / "slot_thresholds.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(BASE_DIR)}")
    print("  now set these in content/platform.yaml and re-assemble:")
    print(f"    slot_confidence_threshold: {st}")
    print(f"    interrupt_threshold: {it}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-irrelevant", type=float, default=0.005,
                    help="max share of ALL turns allowed to produce an "
                         "irrelevant slot question (default 0.005 = 0.5%%)")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    return fit(a.lang, a.folds, a.max_irrelevant, a.write)


if __name__ == "__main__":
    sys.exit(main())
