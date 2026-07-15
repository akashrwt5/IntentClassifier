#!/usr/bin/env python3
"""Engine-in-the-loop wrong-action harness — the SYSTEM-level budget measure.

Replays every holdout utterance through ``NLUEngine.handle`` (fresh session
per turn) and counts what the charter's ≤5 budget actually governs: turns
where the SYSTEM would fire a wrong action. Unlike the raw-classifier
number in ``nlu_training.evaluate`` (an upper bound), this sees the whole
defense stack: keyword tiers, calibrated thresholds, slot prompts,
confirmation gates, and OOS fallback.

Counting rules (single-turn replay):
- ``FULFILL`` with an actionable intent ≠ truth  → **wrong action** (fired).
- ``CONFIRM`` with intent ≠ truth                → confirmation-gated wrong
  guess (the gate did its job; reported separately, not budget-charged).
- ``PROMPT``                                     → no action fired (slot flow).
- ``FALLBACK`` / low-confidence                  → deflected, no action.
- help.* / sys.* results never fire device actions.

Semantic rescue is DISABLED for this run: local semantic artifacts are
gitignored and may carry the pre-migration label space (known-issues.md);
measuring through stale artifacts would be noise. Re-run with --semantic
after regenerating them.

Usage:
    PYTHONPATH=packages/buildtime:packages/runtime \\
        python -m nlu_training.wrong_action_harness [--langs en fr de da] \\
        [--semantic] [--out wrong_action_report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HOLDOUT_DIR = REPO / "multilingual" / "test"

FALLBACK_LABEL = "sys.oos.fallback"
WRONG_ACTION_BUDGET = 5
GATE_WAIVERS = {"da"}


def is_actionable(label: str) -> bool:
    return bool(label) and not label.startswith(("help.", "sys."))


def replay_language(lang: str, semantic: bool) -> dict:
    import csv

    sys.path.insert(0, str(REPO / "packages" / "runtime"))
    from nlu_engine import NLUEngine

    engine = NLUEngine(model_name=lang, language=lang)
    if not semantic:
        engine.semantic = None

    counts = {"turns": 0, "wrong_actions": 0, "confirm_gated_wrong": 0,
              "fulfilled": 0, "prompted": 0, "fallback": 0, "confirmed": 0}
    per_domain: dict[str, int] = {}
    examples: list[dict] = []

    with open(HOLDOUT_DIR / f"{lang}_holdout.csv", newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            text, truth = row["text"], row["intent"]
            session = f"harness-{lang}-{i}"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = engine.handle(session, text)
            counts["turns"] += 1
            rtype = r.type
            intent = r.intent or ""
            if rtype == "FULFILL":
                counts["fulfilled"] += 1
                if is_actionable(intent) and intent != truth:
                    counts["wrong_actions"] += 1
                    domain = intent.split(".")[0]
                    per_domain[domain] = per_domain.get(domain, 0) + 1
                    if len(examples) < 10:
                        examples.append({"lang": lang, "text": text,
                                         "truth": truth, "fired": intent,
                                         "confidence": round(r.confidence or 0, 3)})
            elif rtype == "CONFIRM":
                counts["confirmed"] += 1
                if is_actionable(intent) and intent != truth:
                    counts["confirm_gated_wrong"] += 1
            elif rtype == "PROMPT":
                counts["prompted"] += 1
            elif rtype == "FALLBACK":
                counts["fallback"] += 1
    counts["per_domain"] = per_domain
    counts["examples"] = examples
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--langs", nargs="+", default=["en", "fr", "de", "da"])
    ap.add_argument("--semantic", action="store_true",
                    help="enable semantic rescue (only after regenerating artifacts)")
    ap.add_argument("--out", type=Path, default=Path("wrong_action_report.json"))
    args = ap.parse_args(argv)

    report = {"semantic_enabled": args.semantic, "per_language": {}, "examples": []}
    for lang in args.langs:
        c = replay_language(lang, args.semantic)
        report["examples"].extend(c.pop("examples"))
        report["per_language"][lang] = c
        print(f"  {lang}: turns={c['turns']} fired={c['fulfilled']} "
              f"WRONG={c['wrong_actions']} confirm-gated-wrong={c['confirm_gated_wrong']} "
              f"prompted={c['prompted']} fallback={c['fallback']} "
              f"per-domain={c['per_domain']}")

    shipped = [lg for lg in args.langs if lg not in GATE_WAIVERS]
    total = sum(report["per_language"][lg]["wrong_actions"] for lg in shipped)
    report["wrong_actions_shipped_langs"] = total
    report["budget"] = WRONG_ACTION_BUDGET
    report["budget_met"] = total <= WRONG_ACTION_BUDGET
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\n  SYSTEM wrong actions (shipped langs): {total}  "
          f"budget ≤{WRONG_ACTION_BUDGET}  met={report['budget_met']}  → {args.out}")
    return 0 if report["budget_met"] else 1


if __name__ == "__main__":
    sys.exit(main())
