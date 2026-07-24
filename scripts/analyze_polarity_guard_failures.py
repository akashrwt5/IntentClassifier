#!/usr/bin/env python3
"""Data-driven audit: where does the polarity guard change the model's answer,
and does it help or hurt? Runs the REAL English holdout through the model and
every guard rule. No assumptions — every number below comes from the 1,461
holdout utterances.
"""
from __future__ import annotations
import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packages" / "runtime"))
from nlu_engine import NLUEngine  # noqa: E402

# The FULL original guard set (incl. the up/down rules we retired for English),
# so we can measure each rule's real-world help/hurt on the holdout.
ALL_GUARDS = [
    (r"\bmute\b(?!\s+off)", "device.volume.unmute", "device.volume.mute"),
    (r"\bunmute\b|\bun-mute\b|\bmute\s+off\b|\boff\s+mute\b|\bturn\s+(the\s+)?mute\s+off\b",
     "device.volume.mute", "device.volume.unmute"),
    (r"\b(quiet(er)?|lower|softer|decrease|reduce|down)\b",
     "device.volume.increase", "device.volume.decrease"),
    (r"\b(loud(er)?|higher|increase|raise|up)\b",
     "device.volume.decrease", "device.volume.increase"),
    (r"\b(stop|end|quit|finish|turn off)\b",
     "streaming.session.start", "streaming.session.stop"),
    (r"\b(start|begin|turn on)\b",
     "streaming.session.stop", "streaming.session.start"),
]
COMPILED = [(re.compile(p, re.I), b, r) for p, b, r in ALL_GUARDS]


def guard_once(text, pred):
    """Apply the SAME logic the engine uses (with the both-cue abstain fix),
    but over the full 6-rule set, and report which rule fired."""
    low = text.lower()
    hits = []
    for rx, blocked, redirect in COMPILED:
        if blocked != pred or not rx.search(low):
            continue
        opposite = any(b2 == redirect and r2 == pred and rx2.search(low)
                       for rx2, b2, r2 in COMPILED)
        if not opposite:
            hits.append((redirect, blocked))
    if len(hits) == 1:
        return hits[0]  # (redirect, blocked_rule)
    return (pred, None)


def main():
    eng = NLUEngine()
    rows = list(csv.DictReader(open(REPO / "multilingual" / "test" / "en_holdout.csv",
                                    encoding="utf-8")))
    # per-rule tally: fired, helped (model wrong->guard right), hurt (model right->guard wrong)
    from collections import defaultdict
    tally = defaultdict(lambda: {"fired": 0, "helped": 0, "hurt": 0, "neutral": 0,
                                 "hurt_ex": [], "help_ex": []})
    for r in rows:
        text, truth = r["text"], r["intent"]
        pred, _ = eng.classifier.classify(text)
        redirect, rule = guard_once(text, pred)
        if rule is None or redirect == pred:
            continue
        t = tally[rule]
        t["fired"] += 1
        if pred != truth and redirect == truth:
            t["helped"] += 1
            if len(t["help_ex"]) < 4:
                t["help_ex"].append((text, truth, pred, redirect))
        elif pred == truth and redirect != truth:
            t["hurt"] += 1
            if len(t["hurt_ex"]) < 6:
                t["hurt_ex"].append((text, truth, pred, redirect))
        else:
            t["neutral"] += 1
    print(f"Holdout utterances: {len(rows)}\n")
    for rule in sorted(tally):
        t = tally[rule]
        print(f"### guard rule blocked={rule}")
        print(f"    fired={t['fired']}  helped={t['helped']}  HURT={t['hurt']}  neutral={t['neutral']}")
        for text, truth, pred, red in t["hurt_ex"]:
            print(f"      HURT : {text!r}  truth={truth}  model={pred} -> guard={red}")
        for text, truth, pred, red in t["help_ex"]:
            print(f"      help : {text!r}  truth={truth}  model={pred} -> guard={red}")
        print()


if __name__ == "__main__":
    main()
