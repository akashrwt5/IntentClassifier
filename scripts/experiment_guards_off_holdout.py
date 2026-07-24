#!/usr/bin/env python3
"""EXPERIMENT (non-destructive): replay the English holdout with the polarity
guards ON vs OFF, to isolate where the MODEL itself fails (no guard involved).

Nothing is written to config — the guard list is cleared on a live engine
instance only. Answers: (1) does disabling the guard help or hurt overall,
and (2) exactly which utterances the model gets wrong on its own.
"""
from __future__ import annotations
import csv
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packages" / "runtime"))
from nlu_engine import NLUEngine  # noqa: E402

HOLD = REPO / "multilingual" / "test" / "en_holdout.csv"
STATE_CHANGING_PREFIXES = ("device.volume", "device.memory", "streaming.session",
                           "messaging.message", "reminders.task",
                           "translation.session", "transcription.session", "find.phone")


def is_state_changing(label: str) -> bool:
    return bool(label) and label.startswith(STATE_CHANGING_PREFIXES)


def replay(disable_guards: bool):
    eng = NLUEngine()
    if disable_guards:
        eng._polarity_guards = []            # turn the polarity guard OFF
    rows = list(csv.DictReader(open(HOLD, encoding="utf-8")))
    fired_correct = 0
    fired_total = 0
    wrong_actions = []          # (text, truth, fired) on state-changing intents
    model_errors = Counter()    # confusion pairs truth->pred (classifier only)
    for r in rows:
        text, truth = r["text"], r["intent"]
        res = eng.handle(f"x-{text[:12]}", text)
        # classifier-only prediction (what the model believes, pre-gates)
        mpred, _ = eng.classifier.classify(text)
        if mpred != truth:
            model_errors[f"{truth}  ->  {mpred}"] += 1
        if res.type == "FULFILL" and res.intent:
            fired_total += 1
            if res.intent == truth:
                fired_correct += 1
            elif is_state_changing(res.intent):
                wrong_actions.append((text, truth, res.intent))
    return len(rows), fired_total, fired_correct, wrong_actions, model_errors


def main():
    print("Replaying en_holdout.csv (1461 utterances)\n")
    for label, off in [("GUARDS ON  (current)", False), ("GUARDS OFF (model only)", True)]:
        n, ft, fc, wa, me = replay(off)
        print(f"=== {label} ===")
        print(f"    fired FULFILL: {ft}   correct: {fc}   fired-accuracy: {fc/ft:.3f}")
        print(f"    wrong STATE-CHANGING actions: {len(wa)}")
        for text, truth, fired in wa[:12]:
            print(f"       WRONG: {text!r}  truth={truth}  fired={fired}")
        print()
    # model-only confusion map (same regardless of guards — the classifier)
    print("=== MODEL-ONLY top confusions (classifier argmax != truth) ===")
    _, _, _, _, me = replay(True)
    for pair, c in me.most_common(15):
        print(f"    {c:3d}   {pair}")
    print(f"    total model errors: {sum(me.values())} / 1461  "
          f"(model accuracy {1 - sum(me.values())/1461:.3f})")


if __name__ == "__main__":
    main()
