#!/usr/bin/env python3
"""
Test TF-IDF classifier ONLY (no semantic fallback, no keyword matching).

This isolates the raw TF-IDF model's performance on holdout data.
Use this to understand what the semantic stage needs to rescue.

Run:
    python scripts/test_tfidf_only.py
    python scripts/test_tfidf_only.py --verbose  # show all cases
"""

import argparse
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
PIPELINE_PATH = BASE_DIR / "models" / "intent_pipeline.pkl"
HOLDOUT_PATH = BASE_DIR / "data" / "semantic_holdout_2.csv"

BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

def main():
    parser = argparse.ArgumentParser(description="Test TF-IDF only accuracy")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all predictions")
    args = parser.parse_args()

    print(f"\n{BOLD}TF-IDF ONLY Accuracy Test (No Semantic Fallback){RESET}\n")

    # Load pipeline and holdout data
    pipeline = joblib.load(str(PIPELINE_PATH))
    holdout = pd.read_csv(HOLDOUT_PATH)

    print(f"Testing on {len(holdout)} holdout utterances\n")

    # Run TF-IDF predictions
    results = []
    for _, row in holdout.iterrows():
        utterance = row['utterance'].lower().strip()
        expected = row['expected_intent']

        # Direct TF-IDF prediction
        predicted = pipeline.predict([utterance])[0]
        proba = pipeline.predict_proba([utterance])[0]
        confidence = float(max(proba))

        correct = predicted == expected
        results.append({
            'utterance': utterance,
            'expected': expected,
            'predicted': predicted,
            'confidence': confidence,
            'correct': correct
        })

    results_df = pd.DataFrame(results)

    # Overall accuracy
    correct_count = int(results_df['correct'].sum())
    total = len(results_df)
    accuracy = correct_count / total

    print(f"{BOLD}📊 TF-IDF Only Results:{RESET}")
    print(f"  Correct: {correct_count}/{total} ({accuracy*100:.1f}%)\n")

    # Per-intent breakdown
    print(f"{BOLD}Per-Intent Accuracy:{RESET}")
    print(f"{'Intent':<35} {'Accuracy':<15} {'Avg Confidence':<15}")
    print("─" * 65)

    for intent in sorted(results_df['expected'].unique()):
        intent_results = results_df[results_df['expected'] == intent]
        intent_acc = intent_results['correct'].sum() / len(intent_results)
        correct = int(intent_results['correct'].sum())
        total_for_intent = len(intent_results)
        avg_conf = intent_results['confidence'].mean()

        color = GREEN if intent_acc == 1.0 else RED if intent_acc < 0.7 else ""
        print(f"{intent:<35} {color}{correct}/{total_for_intent} ({intent_acc*100:>3.0f}%){RESET} {avg_conf:>14.2f}")

    # Failure analysis
    failures = results_df[~results_df['correct']]
    if len(failures) > 0:
        low_conf_failures = failures[failures['confidence'] < 0.70]
        high_conf_failures = failures[failures['confidence'] >= 0.70]

        print(f"\n{BOLD}Failure Analysis:{RESET}")
        print(f"  Total failures: {len(failures)}")
        print(f"  • Low confidence (<0.70): {len(low_conf_failures)} (should be rescued by semantic)")
        print(f"  • High confidence (≥0.70): {len(high_conf_failures)} (confident errors)")

        if args.verbose:
            print(f"\n{BOLD}❌ All Failures:{RESET}")
            for _, row in failures.sort_values('confidence', ascending=False).iterrows():
                print(f"  [{row['confidence']:.2f}] {row['expected']:<30} → {row['predicted']:<30}")
                print(f"       \"{row['utterance']}\"")
        else:
            print(f"\n{BOLD}❌ Top 10 Failures (by confidence):{RESET}")
            for _, row in failures.sort_values('confidence', ascending=False).head(10).iterrows():
                print(f"  [{row['confidence']:.2f}] {row['expected']:<30} → {row['predicted']:<30}")
                print(f"       \"{row['utterance'][:60]}\"")

    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}TF-IDF Accuracy: {accuracy*100:.1f}%{RESET}")
    print(f"{BOLD}Expected with semantic: >{(accuracy + len(low_conf_failures)/total)*100:.0f}% (if all low-conf rescued){RESET}")
    print(f"{BOLD}{'='*70}{RESET}\n")

if __name__ == "__main__":
    main()
