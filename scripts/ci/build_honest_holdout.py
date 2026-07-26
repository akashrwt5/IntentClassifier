#!/usr/bin/env python3
"""
Charter B1 — partition a language's training data into an HONEST holdout.

WHY
---
The English figures this project steered by (macro-F1 0.896, accuracy 0.907,
ECE 0.018) were measured on `multilingual/test/en_holdout.csv`, of which
**1460 of 1461 rows appear verbatim in the training data** — Review-F5 blocker
B9. That is a memorisation score, not a generalisation estimate, and every
threshold tuned against it inherited the error.

This script produces a holdout that is disjoint by construction, so the numbers
measured on it mean what they say.

TWO CORRECTNESS PROPERTIES
--------------------------
1. **Split by NORMALISED TEXT, not by row.** `train.csv` contains 83 texts that
   appear more than once. A naive row-level split would put one copy in train
   and another in the holdout — leakage inside what looks like a clean
   partition, which is exactly how B9 happened. Rows are grouped by their
   normalised form and whole groups move together.

2. **Stratified by intent.** Every intent must appear on both sides or its
   recall is unmeasurable. The smallest English intent has 53 rows, so a 15%
   holdout leaves ~8 per intent — thin but real. Intents too small to split
   safely stay wholly in train and are reported, rather than being silently
   dropped from evaluation.

The holdout is FROZEN once written: its sha256 goes into the manifest so any
later change is visible. It must never be trained on, and never tuned against
beyond reporting.

USAGE
    python scripts/ci/build_honest_holdout.py --lang en --dry-run
    python scripts/ci/build_honest_holdout.py --lang en
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "packages" / "buildtime"))
from nlu_training.leakage import find_leaks, normalize_text  # noqa: E402

HOLDOUT_FRACTION = 0.15
MIN_ROWS_TO_SPLIT = 20      # below this an intent keeps all its rows for training
SEED = 42


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(lang: str, fraction: float, dry_run: bool) -> int:
    ds = REPO / "datasets" / lang
    train_path = ds / "train.csv"
    holdout_path = ds / "holdout_honest.csv"
    manifest_path = ds / "holdout_honest.manifest.json"

    if not train_path.exists():
        print(f"FAIL: {train_path} not found")
        return 1
    if holdout_path.exists():
        print(f"FAIL: {holdout_path.name} already exists. The holdout is FROZEN "
              f"once built — re-splitting silently changes every number measured "
              f"against it. Delete it deliberately if you really mean to.")
        return 1

    with train_path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fields, rows = list(reader.fieldnames or []), list(reader)

    # Group by normalised text so duplicates cannot straddle the split.
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[normalize_text(r["text"])].append(r)

    by_intent: dict[str, list[str]] = defaultdict(list)
    for key, grp in groups.items():
        by_intent[grp[0]["intent"].strip()].append(key)

    rng = random.Random(SEED)
    held_keys: set[str] = set()
    too_small: list[str] = []
    for intent, keys in sorted(by_intent.items()):
        if len(keys) < MIN_ROWS_TO_SPLIT:
            too_small.append(f"{intent} ({len(keys)})")
            continue
        keys = sorted(keys)
        rng.shuffle(keys)
        n = max(1, round(len(keys) * fraction))
        held_keys.update(keys[:n])

    train_rows = [r for k, g in groups.items() if k not in held_keys for r in g]
    hold_rows = [r for k, g in groups.items() if k in held_keys for r in g]

    print(f"language        : {lang}")
    print(f"source          : {len(rows)} rows / {len(groups)} distinct texts")
    print(f"train           : {len(train_rows)} rows")
    print(f"holdout         : {len(hold_rows)} rows "
          f"({len(hold_rows)/len(rows):.1%}) across "
          f"{len({r['intent'].strip() for r in hold_rows})} intents")
    if too_small:
        print(f"kept wholly in train (< {MIN_ROWS_TO_SPLIT} distinct texts): {too_small}")

    leaks = find_leaks([r["text"] for r in train_rows],
                       [r["text"] for r in hold_rows])
    print(f"leakage check   : {len(leaks)} overlapping utterance(s)")
    if leaks:
        print(f"  FAIL — the split is not disjoint: {leaks[:5]}")
        return 1

    if dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    for path, data in ((train_path, train_rows), (holdout_path, hold_rows)):
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(data)

    manifest = {
        "_note": "FROZEN honest holdout (charter B1). Disjoint from train.csv by "
                 "normalised text. Never train on it; never tune against it "
                 "beyond reporting. Changing either file invalidates every "
                 "number measured here — the hashes below make that visible.",
        "language": lang,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": "stratified by intent, grouped by normalised text",
        "seed": SEED,
        "holdout_fraction_requested": fraction,
        "rows": {"train": len(train_rows), "holdout": len(hold_rows)},
        "intents": {"train": len({r["intent"].strip() for r in train_rows}),
                    "holdout": len({r["intent"].strip() for r in hold_rows})},
        "intents_kept_wholly_in_train": too_small,
        "sha256": {"train.csv": _sha256(train_path),
                   holdout_path.name: _sha256(holdout_path)},
        "supersedes": "multilingual/test/en_holdout.csv — 1460/1461 rows appeared "
                      "verbatim in training data (Review-F5 blocker B9)",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {holdout_path.relative_to(REPO)}")
    print(f"wrote {manifest_path.relative_to(REPO)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en")
    ap.add_argument("--fraction", type=float, default=HOLDOUT_FRACTION)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    return build(a.lang, a.fraction, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
