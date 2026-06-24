#!/usr/bin/env python3
"""
Dissolve the Cmd.Health intent.

All 43 Cmd.Health rows are activity-metric queries (step/stand/walk/exercise)
that duplicate the dedicated Cmd.Activity* intents. This script:
  1. Reassigns every Cmd.Health row to its matching activity intent.
  2. Removes the Cmd.Health intent definition from data/nlu_schema.json.

Reassignment rule (first match wins):
    step      -> Cmd.ActivityStep
    stand/stood-> Cmd.ActivityStand
    walk      -> Cmd.ActivityWalk
    exercise/workout/dance -> Cmd.ActivityExercise

Run:
    python scripts/dissolve_cmd_health.py --dry-run
    python scripts/dissolve_cmd_health.py
"""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

BASE_DIR    = Path(__file__).parent.parent
DATA_PATH   = BASE_DIR / "data" / "01_source_base_training_data.csv"
SCHEMA_PATH = BASE_DIR / "data" / "nlu_schema.json"
SRC_INTENT  = "Cmd.Health"


def target(text: str) -> str:
    tl = text.lower()
    if re.search(r"\bstep", tl):                            return "Cmd.ActivityStep"
    if re.search(r"\bstand|\bstood", tl):                   return "Cmd.ActivityStand"
    if re.search(r"\bwalk", tl):                            return "Cmd.ActivityWalk"
    if re.search(r"\bexercis|work\s*out|working out", tl):  return "Cmd.ActivityExercise"
    if re.search(r"\bdanc", tl):                            return "Cmd.ActivityExercise"
    return "Cmd.ActivityExercise"  # safe default (none expected to hit this)


def main():
    dry_run = "--dry-run" in sys.argv

    with open(DATA_PATH, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    existing_pairs = {(r["text"].strip().lower(), r["intent"].strip()) for r in rows}

    moved = Counter()
    dropped_dupe = []
    out_rows = []
    for r in rows:
        if r["intent"].strip() != SRC_INTENT:
            out_rows.append(r)
            continue
        tgt = target(r["text"])
        key = (r["text"].strip().lower(), tgt)
        # If the same phrase already exists under the target intent, drop the
        # duplicate instead of creating one.
        if key in existing_pairs:
            dropped_dupe.append((r["text"], tgt))
            continue
        existing_pairs.add(key)
        out_rows.append({"text": r["text"], "intent": tgt})
        moved[tgt] += 1

    print("=" * 64)
    print("DISSOLVE Cmd.Health")
    print("=" * 64)
    print(f"\nReassigned {sum(moved.values())} rows:")
    for tgt, n in sorted(moved.items(), key=lambda x: -x[1]):
        print(f"   +{n:3d} -> {tgt}")
    if dropped_dupe:
        print(f"\nDropped {len(dropped_dupe)} exact duplicates (already in target intent):")
        for t, tgt in dropped_dupe:
            print(f"   - [{tgt}] {t}")

    remaining = sum(1 for r in out_rows if r["intent"].strip() == SRC_INTENT)
    print(f"\nRows still labelled {SRC_INTENT}: {remaining}")

    # Schema edit
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)
    in_schema = SRC_INTENT in schema.get("intents", {})
    print(f"{SRC_INTENT} present in nlu_schema.json /intents: {in_schema}")

    if dry_run:
        print("\n[dry-run] No files written.")
        return

    with open(DATA_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\n✅ Wrote {len(out_rows)} rows to {DATA_PATH}")

    if in_schema:
        del schema["intents"][SRC_INTENT]
        with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"✅ Removed {SRC_INTENT} from {SCHEMA_PATH} "
              f"({len(schema['intents'])} intents remain)")


if __name__ == "__main__":
    main()
