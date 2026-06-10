#!/usr/bin/env python3
"""
Held-out benchmark — measures TRUE generalization.

Runs the full NLU pipeline against data/semantic_holdout_100.csv:
100 paraphrases that are deliberately NOT in the training data, so the
score reflects how well the system handles phrasings it has never seen.

Run from the repo root:
    python scripts/test_holdout.py
    python scripts/test_holdout.py --verbose   # show every phrase, not just failures
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from nlu.engine import NLUEngine

HOLDOUT_PATH = Path(__file__).parent.parent / "data" / "semantic_holdout_100.csv"

GREEN, RED, YELLOW, CYAN, RESET, BOLD = (
    "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[0m", "\033[1m"
)


def main():
    verbose = "--verbose" in sys.argv

    print(f"\n{BOLD}Held-out Benchmark — 100 never-trained paraphrases{RESET}\n")
    engine = NLUEngine()
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
        colour = GREEN if ok >= 8 else (YELLOW if ok >= 5 else RED)
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

    total = results["correct"].sum()
    # Wrong-action = confidently fired the wrong intent (worse than falling to GenAI)
    wrong_action = len(results[(~results["correct"]) & (results["stage"] != "genai")])
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  Total: {total}/{len(results)} ({100*total/len(results):.1f}%)   "
          f"Wrong-action misses: {wrong_action}   "
          f"Safe misses (fell to GenAI): {len(results) - total - wrong_action}{RESET}")
    print(f"{BOLD}{'='*70}{RESET}\n")


if __name__ == "__main__":
    main()
