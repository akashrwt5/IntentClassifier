#!/usr/bin/env python3
"""
Scaffold Lever A data changes: identify tail classes, suggest OOS expansion,
and prepare a template for the expanded holdout covering all 60 intents.

Pure Python — no external dependencies.

Run from repo root:
    python scripts/scaffold_data_correction.py
"""

import csv
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = BASE_DIR / "data" / "01_source_base_training_data.csv"
OOS_PATH = BASE_DIR / "data" / "semantic_oos.csv"
HOLDOUT_PATH = BASE_DIR / "data" / "semantic_holdout_100.csv"

def read_csv(path):
    """Read CSV and return list of dicts."""
    if not path.exists():
        return []
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        return list(reader)

# Load training data
data = read_csv(DATA_PATH)
in_scope = [row for row in data if row.get('intent', '').strip() != 'Default Fallback Intent']

print("=" * 70)
print("LEVER A DATA CORRECTION SCAFFOLD")
print("=" * 70)

# ── 1. Identify tail classes (< 40 rows) ──
print("\n1. TAIL CLASSES (need diversity boost to ≥40 rows):\n")
intent_counts = Counter(row['intent'] for row in in_scope)
sorted_intents = sorted(intent_counts.items(), key=lambda x: x[1])
tail_intents = [i for i, c in sorted_intents if c < 40]

total_need = 0
for intent, count in sorted_intents:
    if count < 40:
        need = 40 - count
        total_need += need
        print(f"  {intent[:40]:40s}  {count:3d} rows  → need {need:2d} more")

print(f"\n  TOTAL: {len(tail_intents)} intents under 40 rows")
print(f"         need {total_need} new paraphrases")

# ── 2. OOS expansion suggestions ──
print("\n2. SEMANTIC_OOS.CSV EXPANSION (156 → 400-600):\n")
oos = read_csv(OOS_PATH)
current_oos = len(oos)
target_oos = 500
need_oos = target_oos - current_oos

print(f"  Current OOS phrases: {current_oos}")
print(f"  Target: {target_oos}")
print(f"  Need to add: {need_oos} hard-negative phrases")
print("\n  Suggested categories for hard negatives:")
print("    • Cooking/food queries (e.g., 'how do I make pasta', 'pasta recipe')")
print("    • Weather (e.g., 'what's the weather', 'will it rain tomorrow')")
print("    • General knowledge (e.g., 'who won the world cup', 'capital of france')")
print("    • Unit conversion (e.g., 'convert 10 dollars to euros')")
print("    • Product Q&A near boundary (e.g., 'how to clean my device', 'battery life specs')")
print("    • Entertainment (e.g., 'tell me a joke', 'sing a song')")
print("\n  Template row structure:")
print("    text,intent")
print("    how do I make pasta,Default Fallback Intent")
print("    ...\n")

# ── 3. Holdout expansion scaffold ──
print("3. SEMANTIC_HOLDOUT_100.CSV EXPANSION (10 intents → 60 intents):\n")
holdout = read_csv(HOLDOUT_PATH)
holdout_intents = set(row.get('expected_intent', row.get('intent', '')).strip() for row in holdout)
all_intents = set(row['intent'] for row in in_scope)
unmeasured = all_intents - holdout_intents

print(f"  Current holdout coverage: {len(holdout_intents)} intents")
print(f"  Measured intents: {sorted(holdout_intents)}")
print(f"\n  UNMEASURED INTENTS ({len(unmeasured)}):")
for intent in sorted(unmeasured):
    count = len([r for r in in_scope if r['intent'] == intent])
    print(f"    {intent[:50]:50s}  {count:3d} training rows")

print(f"\n  Rebuild plan:")
print(f"    • Add ~5-10 rows per intent (aim for ~300-600 total)")
print(f"    • For well-represented intents, pick edge cases / paraphrases")
print(f"    • For tail intents, use existing training data (already small)")
print(f"    • Tag 'difficulty' column: 'easy' (in training data) vs 'hard' (paraphrase/edge case)")
print(f"\n  Template row structure:")
print(f"    utterance,expected_intent,difficulty")
print(f"    mute the audio,Cmd.VolumeMute,easy")
print(f"    silence please,Cmd.VolumeMute,hard")
print(f"    ...\n")

# ── 4. Per-intent distribution ──
print("4. FULL INTENT DISTRIBUTION (after capping at 250):\n")
print(f"  Intent counts (after MAX_PER_INTENT=250 cap):")
print(f"  {'Intent':50s}  {'Original':>8s}  {'Post-cap':>8s}")
sep = "  " + "-" * 50 + "  " + "-" * 8 + "  " + "-" * 8
print(sep)

post_cap_total = 0
for intent, original in sorted_intents:
    capped = min(original, 250)
    post_cap_total += capped
    marker = " (capped)" if original > 250 else ""
    print(f"  {intent[:50]:50s}  {original:8d}  {capped:8d}{marker}")

print(f"\n  Summary: {len(intent_counts)} intents → {post_cap_total} training samples (post-cap)")
print(f"           vs {len(in_scope)} current")

print("\n" + "=" * 70)
print("NEXT STEPS:")
print("=" * 70)
print("1. Add tail-class diversity to 01_source_base_training_data.csv (target ≥40 per intent)")
print("2. Expand data/semantic_oos.csv with 244+ hard negatives → 400+ total")
print("3. Rebuild data/semantic_holdout_100.csv to cover all 60 intents")
print("4. Run: python scripts/train_semantic_head.py")
print("\nDo NOT commit data changes until reviewing samples for accuracy.\n")
