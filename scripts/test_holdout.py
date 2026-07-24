#!/usr/bin/env python3
"""
Held-out benchmark — measures TRUE generalization AND gates the build.

Runs the FULL NLU pipeline (keyword + TF-IDF + semantic) against
the holdout CSV: paraphrases deliberately NOT in training data,
so the score reflects unseen-phrasing behavior.

This is the AUTHORITATIVE accuracy gate. Unlike train.py's gate (which
measures the TF-IDF stage in isolation), this measures the end-to-end
pipeline — the number that predicts real user-facing behavior.

Run from the repo root:
    python scripts/test_holdout.py              # v2 holdout, report only
    python scripts/test_holdout.py --version 1  # original 100-row holdout
    python scripts/test_holdout.py --version 2  # 345-row all-intent holdout (default)
    python scripts/test_holdout.py --verbose    # show every phrase
    python scripts/test_holdout.py --strict     # GATE: exit 1 if below budget
    python scripts/test_holdout.py --json out.json   # machine-readable results

Gate budget (override via env):
    MIN_HOLDOUT_TOTAL=88        minimum correct (absolute count)
    MAX_WRONG_ACTION=5          max confidently-wrong intents (worse than GenAI)
The wrong-action budget is the safety-critical metric: firing the wrong
command on a hearing aid is a user-harm event, unlike a safe GenAI handoff.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from nlu.engine import NLUEngine

_parser = argparse.ArgumentParser(description="Held-out NLU benchmark")
_parser.add_argument("--version", "-v", type=int, choices=[1, 2], default=2,
                     help="Holdout version: 1=semantic_holdout_100.csv, 2=semantic_holdout_2.csv (default: 2)")
_parser.add_argument("--verbose",  action="store_true")
_parser.add_argument("--strict",   action="store_true")
_parser.add_argument("--json",     metavar="FILE")
_args = _parser.parse_args()

_BASE = Path(__file__).parent.parent
if _args.version == 1:
    HOLDOUT_PATH = _BASE / "data" / "semantic_holdout_100.csv"
    # v1 gate: 88/100 correct, ≤5 wrong-action
    _default_min   = 88
    _default_wrong = 5
else:
    HOLDOUT_PATH = _BASE / "data" / "semantic_holdout_2.csv"
    # v2 gate: ~75% of 345 correct (258), ≤15 wrong-action
    # Lower % floor because v2 contains many HARD novel paraphrases across 59 intents
    _default_min   = int(os.environ.get("MIN_HOLDOUT_TOTAL", "258"))
    _default_wrong = int(os.environ.get("MAX_WRONG_ACTION", "15"))

MIN_HOLDOUT_TOTAL = int(os.environ.get("MIN_HOLDOUT_TOTAL", str(_default_min)))
MAX_WRONG_ACTION  = int(os.environ.get("MAX_WRONG_ACTION",  str(_default_wrong)))

GREEN, RED, YELLOW, CYAN, RESET, BOLD = (
    "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[0m", "\033[1m"
)


def main():
    verbose   = _args.verbose
    strict    = _args.strict
    json_path = _args.json

    n_rows = len(pd.read_csv(HOLDOUT_PATH))
    print(f"\n{BOLD}Held-out Benchmark — {n_rows} never-trained paraphrases "
          f"[v{_args.version}: {HOLDOUT_PATH.name}]{RESET}\n")
    engine = NLUEngine(enable_semantic=True)  # holdout exercises the full pipeline incl. semantic (off by default)
    df = pd.read_csv(HOLDOUT_PATH)

    rows = []
    for _, row in df.iterrows():
        r = engine.handle("holdout", row["utterance"])
        engine.reset("holdout")
        actual = r.intent if r.type != "FALLBACK" else "Default Fallback Intent"
        stage = ("semantic" if r.semantic_rescue
                 else ("genai" if r.type == "FALLBACK" else "tfidf"))
        rows.append({
            "utterance": row["utterance"],
            "expected": row["expected_intent"],
            "actual": actual,
            "conf": r.confidence,
            "stage": stage,
            "correct": actual == row["expected_intent"],
        })

    results = pd.DataFrame(rows)

    # Per-intent breakdown
    print(f"{BOLD}{'Intent':<28} {'Score':>6}   Stage of correct answers{RESET}")
    print("─" * 70)
    for intent in results["expected"].unique():
        sub = results[results["expected"] == intent]
        ok = sub["correct"].sum()
        stages = dict(sub[sub["correct"]]["stage"].value_counts())
        total_for_intent = len(sub)
        colour = GREEN if ok == total_for_intent else (YELLOW if ok >= total_for_intent * 0.6 else RED)
        print(f"{intent:<28} {colour}{ok:>3}/{len(sub)}{RESET}   {stages}")

    # Failures (or everything with --verbose)
    shown = results if verbose else results[~results["correct"]]
    label = "All results" if verbose else "Failures"
    print(f"\n{BOLD}{label}:{RESET}")
    for _, w in shown.iterrows():
        mark = f"{GREEN}✅{RESET}" if w["correct"] else f"{RED}❌{RESET}"
        print(f"  {mark} [{w['stage']:8}] conf={w['conf']:.2f}  "
              f"expected={w['expected']:<24} got={w['actual']:<24}")
        print(f"       \"{w['utterance']}\"")

    total = int(results["correct"].sum())
    n = len(results)
    # Wrong-action = confidently fired the wrong intent (worse than falling to GenAI)
    wrong_action = len(results[(~results["correct"]) & (results["stage"] != "genai")])
    safe_miss = n - total - wrong_action
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  Total: {total}/{n} ({100*total/n:.1f}%)   "
          f"Wrong-action misses: {wrong_action}   "
          f"Safe misses (fell to GenAI): {safe_miss}{RESET}")
    print(f"{BOLD}{'='*70}{RESET}\n")

    if json_path:
        Path(json_path).write_text(json.dumps({
            "total": total, "n": n, "wrong_action": wrong_action,
            "safe_miss": safe_miss,
            "min_total": MIN_HOLDOUT_TOTAL, "max_wrong_action": MAX_WRONG_ACTION,
        }, indent=2))
        print(f"  [holdout] results → {json_path}")

    # Authoritative gate: total floor AND wrong-action ceiling must both hold.
    failures = []
    if total < MIN_HOLDOUT_TOTAL:
        failures.append(f"total {total} < floor {MIN_HOLDOUT_TOTAL}")
    if wrong_action > MAX_WRONG_ACTION:
        failures.append(f"wrong-action {wrong_action} > budget {MAX_WRONG_ACTION}")

    if failures:
        msg = "; ".join(failures)
        if strict:
            print(f"{RED}{BOLD}  GATE FAILED: {msg}{RESET}\n")
            sys.exit(1)
        print(f"{YELLOW}  (gate would fail in --strict: {msg}){RESET}\n")
    elif strict:
        print(f"{GREEN}  GATE PASSED "
              f"(total ≥ {MIN_HOLDOUT_TOTAL}, wrong-action ≤ {MAX_WRONG_ACTION}){RESET}\n")


if __name__ == "__main__":
    main()
