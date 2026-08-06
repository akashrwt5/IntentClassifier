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

Semantic rescue is controlled by the ONE flag (engine semantic_enabled /
NLU_SEMANTIC_RESCUE / schema semantic_rescue_enabled). The harness passes
--semantic straight through; default off for the deterministic core
measurement.

The holdout it replays is the FROZEN honest holdout at
``datasets/<lang>/holdout_honest.csv`` (charter B1). The previous source,
``multilingual/test/<lang>_holdout.csv``, had 1460 of 1461 rows present verbatim
in the training data (Review-F5 blocker B9), so every wrong-action count
measured against it was a memorisation replay: the engine had seen the exact
strings, so its confidence — and therefore every gate this harness exercises —
was not the confidence it will show in the field. There is deliberately NO
fallback to that file; a missing honest holdout is an error, not a reason to
quietly measure the wrong thing.

Usage:
    PYTHONPATH=packages/buildtime:packages/runtime \\
        python -m nlu_training.wrong_action_harness [--langs en] \\
        [--semantic] [--out wrong_action_report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

FALLBACK_LABEL = "Default Fallback Intent"
WRONG_ACTION_BUDGET = 5
GATE_WAIVERS = {"da"}


def holdout_path(lang: str) -> Path:
    """The frozen honest holdout for `lang` (charter B1)."""
    return REPO / "datasets" / lang / "holdout_honest.csv"


# WHY THESE ARE EXPLICIT SETS AND NOT PREFIX TESTS
#
# These predicates decide what counts as a wrong ACTION — the medical safety
# budget, `WRONG_ACTION_BUDGET`, the report card's `wrong_action_count`, and
# blocker B1 all reduce to them.
#
# They used to read the taxonomy: `not label.startswith(("help.", "sys."))` and
# a `.query` suffix test. That worked while labels were `domain.object.action`,
# because that taxonomy ENCODED actionability in the name. The migration to
# `Cmd.*` / `Help_*` / `Default Fallback Intent` does not, and the prefix tests
# did not fail — they silently started answering True for every help intent and
# for the fallback, so displaying help content and deflecting to GenAI both
# began counting as device actions.
#
# A predicate whose meaning depends on a naming convention will break again the
# next time the convention moves, quietly, in the direction of a worse-looking
# metric. So the sets are enumerated, derived once from the taxonomy that DID
# encode this (`legacy_label_map.json`, modern -> legacy), and a test asserts
# they still cover the shipped label space.
NON_ACTIONABLE_PREFIXES = ("Help_", "help.", "sys.")
NON_ACTIONABLE_INTENTS = frozenset({"Default Fallback Intent"})


def is_actionable(label: str) -> bool:
    """Could acting on this label do something to the device or the app?

    False for help content (shows an article) and for the fallback (routes to
    GenAI narration). Neither can fire an action, so a wrong one is a missed or
    deflected turn rather than a safety event.
    """
    if not label:
        return False
    if label in NON_ACTIONABLE_INTENTS:
        return False
    return not label.startswith(NON_ACTIONABLE_PREFIXES)


# Read-only intents report information but change NO device or app state (they
# read a value back to the user), so a wrong one is a query-ACCURACY miss, not a
# safety event. The medical wrong-ACTION budget governs state changes — firing
# the wrong irreversible/stateful command — so read-only misses are tracked
# separately (``wrong_queries``) and NOT charged to the budget.
#
# Enumerated for the reason above. Under `domain.object.action` these were the
# `.query` suffix plus `device.status.battery`; the same nine intents under the
# current taxonomy are the eight activity readings and the battery level.
READ_ONLY_SUFFIXES = ("query",)
READ_ONLY_INTENTS = frozenset({
    "Cmd.BatteryLevel",
    "Cmd.ActivityAerobics", "Cmd.ActivityCalories", "Cmd.ActivityCycle",
    "Cmd.ActivityExercise", "Cmd.ActivityRun", "Cmd.ActivityStand",
    "Cmd.ActivityStep", "Cmd.ActivityWalk",
})


def is_read_only(label: str) -> bool:
    if not label:
        return False
    if label in READ_ONLY_INTENTS:
        return True
    return label.rsplit(".", 1)[-1] in READ_ONLY_SUFFIXES


def is_state_changing(label: str) -> bool:
    """A wrong prediction here is a SAFETY event (the budget governs these):
    an actionable intent that also changes device/app state."""
    return is_actionable(label) and not is_read_only(label)


def replay_language(lang: str, semantic: bool) -> dict:
    import csv

    sys.path.insert(0, str(REPO / "packages" / "runtime"))
    from nlu_engine import NLUEngine

    path = holdout_path(lang)
    if not path.exists():
        raise SystemExit(
            f"no honest holdout for {lang!r}: {path.relative_to(REPO)}\n"
            f"  Build one with: python scripts/ci/build_honest_holdout.py --lang {lang}\n"
            f"  This harness deliberately does NOT fall back to "
            f"multilingual/test/{lang}_holdout.csv — that set is 99.9% training "
            f"data (Review-F5 blocker B9) and any number from it is meaningless.")

    engine = NLUEngine(model_name=lang, language=lang, semantic_enabled=semantic)

    counts = {"turns": 0, "wrong_actions": 0, "wrong_queries": 0,
              "confirm_gated_wrong": 0, "fulfilled": 0, "prompted": 0,
              "fallback": 0, "confirmed": 0}
    per_domain: dict[str, int] = {}
    examples: list[dict] = []

    with open(path, newline="", encoding="utf-8-sig") as f:
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
                if intent != truth and is_actionable(intent):
                    if is_read_only(intent):
                        # Wrong read-only query: an accuracy miss, not a safety
                        # event — tracked, but NOT charged to the wrong-action budget.
                        counts["wrong_queries"] += 1
                    else:
                        counts["wrong_actions"] += 1
                        domain = intent.split(".")[0]
                        per_domain[domain] = per_domain.get(domain, 0) + 1
                        if len(examples) < 10:
                            examples.append({"lang": lang, "text": text,
                                             "truth": truth, "fired": intent,
                                             "confidence": round(r.confidence or 0, 3)})
            elif rtype == "CONFIRM":
                counts["confirmed"] += 1
                if is_state_changing(intent) and intent != truth:
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
    # English only by default: it is the only language with a frozen honest
    # holdout and a retrained model. fr/de/da still have to be rebuilt on the
    # per-language layout before a number from them means anything.
    ap.add_argument("--langs", nargs="+", default=["en"])
    ap.add_argument("--semantic", action="store_true",
                    help="enable semantic rescue (only after regenerating artifacts)")
    ap.add_argument("--out", type=Path, default=Path("wrong_action_report.json"))
    args = ap.parse_args(argv)

    # Provenance: a wrong-action count is only interpretable alongside the
    # holdout it was measured on and the temperature that drove every gate.
    # The previous number (41) carried neither, so it survived a leaked holdout
    # and a mis-fit temperature without anyone being able to tell.
    import hashlib

    report = {"semantic_enabled": args.semantic, "per_language": {},
              "examples": [], "provenance": {}}
    for lang in args.langs:
        hp = holdout_path(lang)
        calib = REPO / "models" / "intent" / lang / "calibration.json"
        report["provenance"][lang] = {
            "holdout": str(hp.relative_to(REPO)) if hp.exists() else None,
            "holdout_sha256": hashlib.sha256(hp.read_bytes()).hexdigest()
            if hp.exists() else None,
            "temperature": json.loads(calib.read_text(encoding="utf-8"))["temperature"]
            if calib.exists() else None,
        }
        c = replay_language(lang, args.semantic)
        report["examples"].extend(c.pop("examples"))
        report["per_language"][lang] = c
        print(f"  {lang}: turns={c['turns']} fired={c['fulfilled']} "
              f"WRONG-ACTION={c['wrong_actions']} (wrong-query={c['wrong_queries']}, "
              f"not budget-charged) confirm-gated-wrong={c['confirm_gated_wrong']} "
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
