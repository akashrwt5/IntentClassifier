#!/usr/bin/env python3
"""
One-shot: reorganise `datasets/` into the per-language layout.

WHY
---
Training data is BUILD-TIME input; a Language Pack is a RUNTIME artifact. The
bundle spec already encodes that split — `bundle.json`'s `training` block
carries `dataset_hashes`, i.e. the fingerprint of the data that produced the
model, never the data itself. A `.nlu` ships to a hearing aid; 6.7 MB of user
utterances has no business on the device.

So training data lives here, keyed by language, and the pack records only its
hash:

    datasets/<lang>/
        train.csv                  REQUIRED  text,intent
        holdout_leakage_guard.csv  the set train.py checks against for leakage
        holdout_paraphrase.csv     hard paraphrases (needs the semantic stage)
        oos.csv / oos_2.csv        out-of-scope
        benchmark_250.csv          scored benchmark
        sources/                   the inputs that generate train.csv

Adding a language is then: create `datasets/<lang>/train.csv`, train, ship. No
engine change and no data on device.

WHAT MOVES TO _archive/
-----------------------
Owner decision (2026-07-26): the combined-multilingual and Dialogflow corpora
are no longer used. English trains from `04_GENERATED_MASTER_training_data.csv`;
other languages will supply their own datasets. Those files are ARCHIVED rather
than deleted — this data was recovered from near-loss days ago and deleting it
again in the same week would be careless. Archive, confirm nothing needs it,
delete later.

LABEL MIGRATION
---------------
Three recovered files still carry the pre-ND-3 59-label space. The holdouts are
migrated here through `docs/Review-F5/capability-map.json` — the same
machine-checked map the bootstrap used — because a holdout referencing intents
that no longer exist cannot evaluate anything. `multilingual/pending/*` is
deliberately NOT migrated: it is unprocessed inbox data and goes to the archive
as-is.

USAGE
    python scripts/ci/relayout_datasets.py --dry-run
    python scripts/ci/relayout_datasets.py
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DS = REPO / "datasets"
CAPABILITY_MAP = REPO / "docs" / "Review-F5" / "capability-map.json"
SCHEMA = REPO / "content" / "nlu_schema.json"

# source -> destination, relative to datasets/
MOVES: dict[str, str] = {
    "04_GENERATED_MASTER_training_data.csv": "en/train.csv",
    "semantic_holdout_2.csv":                "en/holdout_leakage_guard.csv",
    "semantic_holdout_100.csv":              "en/holdout_paraphrase.csv",
    "semantic_oos.csv":                      "en/oos.csv",
    "semantic_oos_2.csv":                    "en/oos_2.csv",
    "semantic_benchmark_250.csv":            "en/benchmark_250.csv",
    "confirmation_fixtures.csv":             "en/confirmation_fixtures.csv",
    "01_source_base_training_data.csv":      "en/sources/01_source_base_training_data.csv",
    "02_source_manual_corrections.csv":      "en/sources/02_source_manual_corrections.csv",
    "03_generated_augmented_phrases.csv":    "en/sources/03_generated_augmented_phrases.csv",
    "label_migration_map.json":              "en/sources/label_migration_map.json",
    "semantic_holdout_expansion_template.csv": "en/sources/holdout_expansion_template.csv",
    "semantic_oos_expansion_template.csv":     "en/sources/oos_expansion_template.csv",
}

# Retired by owner decision — archived, not deleted.
ARCHIVE: list[str] = [
    "multilingual",
    "dialogflowData",
    "intent_data_new.csv",
    "Generated_Master_training_Danish_Data.csv",
    "Generated_Master_training_French_Data.csv",
]

# Files whose labels must be migrated 59 -> 57 as part of the move.
# strict=True  -> an unmappable label is a hard failure (evaluation sets: a row
#                 scored against a non-existent intent is silently meaningless).
# strict=False -> unmappable rows are dropped with a report (authoring
#                 templates: a dissolved label there just misleads an author).
MIGRATE_LABELS: dict[str, tuple[str, bool]] = {
    "en/holdout_leakage_guard.csv":              ("expected_intent", True),
    "en/sources/holdout_expansion_template.csv": ("expected_intent", False),
}


def _migration() -> tuple[dict[str, str], set[str]]:
    intents = json.loads(CAPABILITY_MAP.read_text(encoding="utf-8"))["intents"]
    mapping = {o: s["proposed_intent"] for o, s in intents.items() if s.get("proposed_intent")}
    dropped = {o for o, s in intents.items() if not s.get("proposed_intent")}
    return mapping, dropped


def _migrate_file(path: Path, col: str, mapping: dict[str, str],
                  dropped: set[str], shipped: set[str],
                  strict: bool) -> tuple[int, int, list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fields, rows = list(reader.fieldnames or []), list(reader)
    if col not in fields:
        return 0, 0
    kept, removed, unmappable = [], 0, []
    for row in rows:
        old = (row.get(col) or "").strip()
        if old in dropped:
            removed += 1
            continue
        new = mapping.get(old, old)
        if new not in shipped:
            # Neither mapped nor already current: the label was DISSOLVED by
            # ND-3 (e.g. Cmd.Health) and has no successor to point at.
            unmappable.append(old)
            if not strict:
                removed += 1
                continue
        row[col] = new
        kept.append(row)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(kept)
    return len(kept), removed, sorted(set(unmappable))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    missing = [s for s in MOVES if not (DS / s).exists()]
    if missing:
        print(f"FAIL: expected source files are absent: {missing}")
        return 1

    print("MOVES")
    for src, dst in MOVES.items():
        print(f"  {src:<45} -> {dst}")
    print("\nARCHIVE (retired, not deleted)")
    for a in ARCHIVE:
        print(f"  {a:<45} -> _archive/{a}")
    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    for src, dst in MOVES.items():
        target = DS / dst
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(DS / src), str(target))

    arch = DS / "_archive"
    arch.mkdir(exist_ok=True)
    for a in ARCHIVE:
        s = DS / a
        if s.exists():
            shutil.move(str(s), str(arch / a))

    mapping, dropped = _migration()
    shipped = set(json.loads(SCHEMA.read_text(encoding="utf-8"))["intents"])
    print("\nLABEL MIGRATION (59 -> 57)")
    for rel, (col, strict) in MIGRATE_LABELS.items():
        path = DS / rel
        if not path.exists():
            continue
        kept, removed, unmappable = _migrate_file(
            path, col, mapping, dropped, shipped, strict)
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
        labs = {r[col].strip() for r in rows if r.get(col)}
        stale = labs - shipped
        note = f"dissolved: {unmappable}" if unmappable else "OK"
        print(f"  {rel:<45} {kept} rows (-{removed})  {note}")
        if stale:
            print(f"    FAIL: labels absent from the 57-intent schema: {sorted(stale)[:5]}")
            return 1

    print("\nRESULT")
    for lang_dir in sorted(p for p in DS.iterdir() if p.is_dir() and not p.name.startswith("_")):
        n = len(list(lang_dir.rglob("*.csv")))
        print(f"  datasets/{lang_dir.name}/  {n} csv files")
    print(f"  datasets/_archive/  {len(list(arch.rglob('*.csv')))} csv files (retired)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
