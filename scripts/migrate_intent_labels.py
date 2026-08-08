#!/usr/bin/env python3
"""Migrate intent labels from domain.object.action → legacy Cmd.*/Help_* format.

Reads the mapping from legacy_label_map.json and applies it across all relevant
files in the IntentClassifier codebase.

Usage:
    python scripts/migrate_intent_labels.py --dry-run   # preview changes
    python scripts/migrate_intent_labels.py --apply      # apply changes
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAP_PATH = REPO / "packages" / "runtime" / "nlu_engine" / "legacy_label_map.json"


def load_mapping() -> dict[str, str]:
    """Load modern->legacy intent mapping from legacy_label_map.json."""
    data = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    return dict(data["map"])


def migrate_csv(path: Path, mapping: dict[str, str], intent_col: str = "intent",
                dry_run: bool = True) -> int:
    """Replace intent labels in a CSV file. Returns count of replacements."""
    text = path.read_text(encoding="utf-8")
    # Detect line ending
    newline = "\r\n" if "\r\n" in text else "\n"

    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        return 0

    # Find the actual intent column name
    fieldnames = list(rows[0].keys())
    col = None
    for candidate in [intent_col, "expected_intent"]:
        if candidate in fieldnames:
            col = candidate
            break
    if col is None:
        return 0

    count = 0
    for row in rows:
        old = row[col]
        if old in mapping:
            row[col] = mapping[old]
            count += 1

    if count > 0 and not dry_run:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator=newline)
        writer.writeheader()
        writer.writerows(rows)
        path.write_text(buf.getvalue(), encoding="utf-8")

    return count


def migrate_text_file(path: Path, mapping: dict[str, str],
                      dry_run: bool = True) -> int:
    """Replace intent label strings in a text file (JSON, YAML, PY, etc.).
    
    Replaces only whole-word occurrences using word boundary matching.
    Returns count of replacements.
    """
    text = path.read_text(encoding="utf-8")
    count = 0

    # Sort by length descending to avoid partial matches
    for modern, legacy in sorted(mapping.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(re.escape(modern))
        matches = pattern.findall(text)
        if matches:
            text = pattern.sub(legacy, text)
            count += len(matches)

    if count > 0 and not dry_run:
        path.write_text(text, encoding="utf-8")

    return count


def collect_csv_files() -> list[tuple[Path, str]]:
    """Collect all CSV files that need intent column migration.
    
    Returns list of (path, intent_column_name).
    """
    files = []

    # language_packs/en CSVs
    lp = REPO / "language_packs" / "en"
    files.append((lp / "train.csv", "intent"))
    files.append((lp / "holdout_honest.csv", "intent"))
    files.append((lp / "holdout_leakage_guard.csv", "expected_intent"))

    # Source CSVs
    sources = lp / "extras" / "sources"
    if sources.exists():
        for f in sources.glob("*.csv"):
            header = f.read_text(encoding="utf-8").split("\n")[0]
            if "expected_intent" in header:
                files.append((f, "expected_intent"))
            elif "intent" in header:
                files.append((f, "intent"))

    # OOS CSVs
    extras = lp / "extras"
    if extras.exists():
        for f in extras.glob("*.csv"):
            if f.parent == sources:
                continue
            header = f.read_text(encoding="utf-8").split("\n")[0]
            if "expected_intent" in header:
                files.append((f, "expected_intent"))
            elif "intent" in header:
                files.append((f, "intent"))

    # datasets/ CSVs
    datasets = REPO / "datasets"
    if datasets.exists():
        for f in datasets.rglob("*.csv"):
            header = f.read_text(encoding="utf-8").split("\n")[0]
            if "expected_intent" in header:
                files.append((f, "expected_intent"))
            elif "intent" in header:
                files.append((f, "intent"))

    return files


def collect_text_files() -> list[Path]:
    """Collect all non-CSV text files that need intent string replacement."""
    files = []
    skip_dirs = {".venv", ".git", ".mypy_cache", ".ruff_cache", ".pytest_cache",
                 "dist", "__pycache__", "node_modules"}

    # JSON files
    for f in REPO.rglob("*.json"):
        if any(d in f.parts for d in skip_dirs):
            continue
        if f.name == "legacy_label_map.json":
            continue
        if f.name == "label_migration_map.json":
            continue
        files.append(f)

    # YAML files
    for f in REPO.rglob("*.yaml"):
        if any(d in f.parts for d in skip_dirs):
            continue
        files.append(f)
    for f in REPO.rglob("*.yml"):
        if any(d in f.parts for d in skip_dirs):
            continue
        files.append(f)

    # Python files
    for f in REPO.rglob("*.py"):
        if any(d in f.parts for d in skip_dirs):
            continue
        files.append(f)

    return files


def main():
    parser = argparse.ArgumentParser(description="Migrate intent labels")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Preview changes")
    group.add_argument("--apply", action="store_true", help="Apply changes")
    args = parser.parse_args()

    dry_run = args.dry_run
    mapping = load_mapping()
    total_replacements = 0
    files_changed = 0

    print(f"{'DRY RUN' if dry_run else 'APPLYING'} -- {len(mapping)} intent mappings loaded")
    print("=" * 70)

    # Phase 1: CSV files (column-aware replacement)
    print("\n CSV FILES (column-aware replacement)")
    print("-" * 50)
    csv_files = collect_csv_files()
    for path, col in csv_files:
        if not path.exists():
            continue
        count = migrate_csv(path, mapping, intent_col=col, dry_run=dry_run)
        if count > 0:
            rel = path.relative_to(REPO)
            print(f"  OK {rel}: {count} replacements (col={col})")
            total_replacements += count
            files_changed += 1

    # Phase 2: Text files (string replacement)
    print("\n TEXT FILES (string replacement)")
    print("-" * 50)
    text_files = collect_text_files()
    for path in text_files:
        if not path.exists():
            continue
        try:
            count = migrate_text_file(path, mapping, dry_run=dry_run)
        except (UnicodeDecodeError, PermissionError):
            continue
        if count > 0:
            rel = path.relative_to(REPO)
            print(f"  OK {rel}: {count} replacements")
            total_replacements += count
            files_changed += 1

    print("\n" + "=" * 70)
    print(f"{'DRY RUN' if dry_run else 'APPLIED'}: {total_replacements} total replacements across {files_changed} files")

    if dry_run:
        print("\nRun with --apply to make changes.")


if __name__ == "__main__":
    main()
