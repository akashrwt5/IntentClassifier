#!/usr/bin/env python3
"""
Robustness Benchmark Suite for Intent and BIO Slot Tagger.

Tests the model against the 7 real-world failure categories:
1. Spelling Mistakes
2. Slang / Colloquialisms
3. Speech-to-Text (ASR) Errors
4. Missing Punctuation
5. Contractions
6. Conversational Fillers
7. Word Variations

Usage:
    python scripts/benchmark_robustness.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "runtime"))

from test_joint_model import JointONNXModel

BENCHMARK_CASES = {
    "1. Spelling Mistakes": [
        ("set a remndr to take my medcine tomorow at 5pm", "take my medcine"),
        ("remind me to call the docotr at 9am", "call the docotr"),
        ("dont let me forget to pick up the perscription", "pick up the perscription"),
        ("make a remider to by groseries tomorrow", "by groseries"),
    ],
    "2. Slang": [
        ("yo bro drop a reminder to hit the gym at 6pm", "hit the gym"),
        ("gimme a ping to call mom tonight at 8", "call mom"),
        ("jot down a reminder to pay electricity bill tomorrow", "pay electricity bill"),
        ("hit me up to grab some groceries tomorrow at 11am", "grab some groceries"),
    ],
    "3. Speech-to-Text (ASR) Errors": [
        ("set a remind her to take medicine tomorrow at 5 pm", "take medicine"),
        ("remind me two take medicine tomorrow at 5 pm", "take medicine"),
        ("set reminder for doctor appointment at for pm", "doctor appointment"),
        ("set a reminder to by milk and eggs", "by milk and eggs"),
    ],
    "4. Missing Punctuation": [
        ("set reminder to take medicine tomorrow at 5pm dont forget", "take medicine"),
        ("remind me to call mom at 8pm please thanks", "call mom"),
        ("set alarm for 6am wake up for flight", "wake up for flight"),
    ],
    "5. Contractions": [
        ("don't let me forget to take my pills at 9am", "take my pills"),
        ("i've gotta remember to pick up dry cleaning tomorrow", "pick up dry cleaning"),
        ("i'd like a reminder to stretch in 20 minutes", "stretch"),
        ("can't forget to submit the report by friday", "submit the report"),
    ],
    "6. Conversational Fillers": [
        ("um like please set a reminder to water the plants tomorrow at 9", "water the plants"),
        ("you know uhhh remind me to buy groceries at 4pm", "buy groceries"),
        ("well basically set an alarm to wake up at 7am", "wake up"),
        ("hey so yeah can you remind me to check the oven in 15 minutes", "check the oven"),
    ],
    "7. Word Variations": [
        (
            "schedule an appointment with the dentist next tuesday at 3pm",
            "appointment with the dentist",
        ),
        ("log an alert to take medication at 8pm", "take medication"),
        ("keep track of feeding the pets tonight at 7", "feeding the pets"),
        ("put on my calendar to review quarterly results tomorrow", "review quarterly results"),
    ],
}


def run_benchmark(model=None):
    if model is None:
        model = JointONNXModel()

    total_cases = 0
    total_passed = 0
    category_scores = {}

    print("=" * 75)
    print("🧪 EVALUATING MODEL ON ROBUSTNESS BENCHMARK (7 CATEGORIES)")
    print("=" * 75)

    for category, cases in BENCHMARK_CASES.items():
        print(f"\n📂 Category: {category}")
        print("-" * 75)
        cat_passed = 0

        for text, expected in cases:
            total_cases += 1
            res = model.predict(text)
            actual = (res["title"] or "").lower().strip()
            exp = expected.lower().strip()

            # Pass condition: clean containment or high overlap
            intent_ok = (
                "remind" in res["intent"].lower()
                or "alarm" in res["intent"].lower()
                or "schedule" in res["intent"].lower()
            )
            title_ok = (exp in actual) or (actual in exp and len(actual) >= len(exp) * 0.6)

            if intent_ok and title_ok:
                cat_passed += 1
                total_passed += 1
                status = "✅ PASS"
            else:
                status = "❌ FAIL"

            print(f'  [{status}] Input   : "{text}"')
            print(f"         Intent  : {res['intent']} ({'OK' if intent_ok else 'WRONG'})")
            print(f'         Expected: "{exp}"')
            print(f"         Actual  : \"{actual or '(None)'}\"\n")

        pct = (cat_passed / len(cases)) * 100
        category_scores[category] = (cat_passed, len(cases), pct)
        print(f"  👉 Score for {category}: {cat_passed}/{len(cases)} ({pct:.1f}%)")

    print("\n" + "=" * 75)
    print("📊 FINAL BENCHMARK SUMMARY")
    print("=" * 75)
    for cat, (p, t, pct) in category_scores.items():
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        print(f"  {cat:<32} : {p}/{t} ({pct:5.1f}%)  [{bar}]")
    print("-" * 75)
    overall_pct = (total_passed / total_cases) * 100
    print(
        f"  ⭐ OVERALL ACCURACY              : {total_passed}/{total_cases} ({overall_pct:.1f}%)\n"
    )
    return category_scores


if __name__ == "__main__":
    run_benchmark()
