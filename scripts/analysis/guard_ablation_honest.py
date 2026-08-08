#!/usr/bin/env python3
"""Re-derive the English guard decisions on the HONEST holdout.

WHY
---
`content/platform.yaml` records an owner directive of 2026-07-24 in which each
guard was kept or removed "on holdout evidence". That holdout was
`multilingual/test/en_holdout.csv` — 1460 of its 1461 rows appear verbatim in
the training data (Review-F5 blocker B9). On memorised text the model answers
correctly, so a guard that corrects a genuine model error looks like a guard
that overrides a correct prediction. The evidence inverted the conclusion.

The same tainted holdout underwrites:
  * polarity_guards REMOVED       ("3 vs 4 wrong state-changing actions")
  * help_marker_guard KEPT        ("44 misfires fixed, 0 correct diverted")
  * semantic rescue left OFF      ("+150 recovered / +5 wrong per 1,461 turns")

This replays each configuration through the full engine against
`datasets/en/holdout_honest.csv` so the decisions rest on data the model has
not memorised.

MEASUREMENT ONLY — every override is applied to an in-memory schema. No file on
disk is modified and no policy is changed.

USAGE
    PYTHONPATH=packages/buildtime:packages/runtime \\
        python scripts/analysis/guard_ablation_honest.py [--out report.json]
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in ("packages/runtime", "packages/buildtime"):
    sys.path.insert(0, str(REPO / p))

from nlu_training.wrong_action_harness import (  # noqa: E402
    is_actionable, is_read_only,
)

# The six rules removed on 2026-07-24, recovered from content/platform.yaml at
# commit 835083f9 (the last revision that carried them).
POLARITY_GUARDS = [
    {"blocked_intent": "Cmd.VolumeUnmute", "pattern": r"\bmute\b",
     "redirect_intent": "Cmd.VolumeMute"},
    {"blocked_intent": "Cmd.VolumeMute", "pattern": r"\bunmute\b|\bun-mute\b",
     "redirect_intent": "Cmd.VolumeUnmute"},
    {"blocked_intent": "Cmd.VolumeIncrease",
     "pattern": r"\b(quiet(er)?|lower|softer|decrease|reduce|down)\b",
     "redirect_intent": "Cmd.VolumeDecrease"},
    {"blocked_intent": "Cmd.VolumeDecrease",
     "pattern": r"\b(loud(er)?|higher|increase|raise|up)\b",
     "redirect_intent": "Cmd.VolumeIncrease"},
    {"blocked_intent": "Cmd.StreamingStart",
     "pattern": r"\b(stop|end|quit|finish|turn off)\b",
     "redirect_intent": "Cmd.StreamingStop"},
    {"blocked_intent": "Cmd.StreamingStop",
     "pattern": r"\b(start|begin|turn on)\b",
     "redirect_intent": "Cmd.StreamingStart"},
]


def load_holdout() -> list[dict]:
    path = REPO / "datasets" / "en" / "holdout_honest.csv"
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def run(name: str, rows: list[dict], *, polarity: bool, help_marker: bool,
        semantic: bool) -> dict:
    """Replay the holdout under one guard configuration."""
    from nlu_engine import NLUEngine

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eng = NLUEngine(model_name="en", language="en", semantic_enabled=semantic)

    # Rebuild the guard state from an overridden copy of the loaded schema.
    # Shapes must match NLUEngine.__init__ exactly: polarity guards are
    # (compiled_pattern, blocked_intent, redirect_intent) 3-tuples.
    schema = copy.deepcopy(eng.schema)
    schema["polarity_guards"] = POLARITY_GUARDS if polarity else []
    if not help_marker:
        schema["help_marker_guard"] = {}
    eng.schema = schema
    eng._polarity_guards = [
        (re.compile(g["pattern"], re.IGNORECASE), g["blocked_intent"],
         g["redirect_intent"])
        for g in schema.get("polarity_guards", [])
    ]
    hmg = schema.get("help_marker_guard", {})
    markers = hmg.get("markers")
    eng._help_markers = re.compile(markers, re.IGNORECASE) if markers else None
    eng._help_pairs = dict(hmg.get("pairs", {}))

    counts = {"turns": 0, "wrong_actions": 0, "wrong_queries": 0,
              "confirm_gated_wrong": 0, "fulfilled": 0, "correct_fulfilled": 0,
              "prompted": 0, "fallback": 0}
    failures = []
    for i, row in enumerate(rows):
        text, truth = row["text"], row["intent"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = eng.handle(f"abl-{name}-{i}", text)
        counts["turns"] += 1
        intent = r.intent or ""
        if r.type == "FULFILL":
            counts["fulfilled"] += 1
            if intent == truth:
                counts["correct_fulfilled"] += 1
            elif is_actionable(intent):
                if is_read_only(intent):
                    counts["wrong_queries"] += 1
                else:
                    counts["wrong_actions"] += 1
                    failures.append({"text": text, "truth": truth,
                                     "fired": intent,
                                     "conf": round(r.confidence or 0, 3)})
        elif r.type == "CONFIRM":
            if intent != truth and is_actionable(intent) and not is_read_only(intent):
                counts["confirm_gated_wrong"] += 1
        elif r.type == "PROMPT":
            counts["prompted"] += 1
        elif r.type == "FALLBACK":
            counts["fallback"] += 1
    counts["failures"] = failures
    return counts


CONFIGS = [
    # name,                       polarity, help_marker, semantic
    ("baseline (as shipped)",        False, True,  False),
    ("+ polarity guards restored",   True,  True,  False),
    ("- help_marker_guard",          False, False, False),
    ("+ semantic rescue",            False, True,  True),
    ("+ polarity + semantic",        True,  True,  True),
    ("+ polarity, - help_marker",    True,  False, False),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPO / "tests/parity/oracle_honest_en/guard_ablation.json")
    args = ap.parse_args(argv)

    rows = load_holdout()
    print(f"honest holdout: {len(rows)} turns\n")
    report = {"_note": "Guard decisions re-derived on the honest holdout after "
                       "Review-F5 blocker B9 invalidated the evidence base. "
                       "Measurement only — no policy file was modified.",
              "holdout": "datasets/en/holdout_honest.csv",
              "turns": len(rows), "configs": {}}

    hdr = f"{'configuration':<30}{'wrong':>7}{'correct':>9}{'gated':>7}{'fallback':>10}"
    print(hdr)
    print("-" * len(hdr))
    for name, pol, hm, sem in CONFIGS:
        # "- help_marker_guard" and "no guards at all" coincide while polarity
        # is already empty; keep both rows so the table reads unambiguously.
        c = run(name, rows, polarity=pol, help_marker=hm, semantic=sem)
        report["configs"][name] = {k: v for k, v in c.items() if k != "failures"}
        report["configs"][name]["failures"] = c["failures"][:15]
        print(f"{name:<30}{c['wrong_actions']:>7}{c['correct_fulfilled']:>9}"
              f"{c['confirm_gated_wrong']:>7}{c['fallback']:>10}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"\nwrote {args.out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
