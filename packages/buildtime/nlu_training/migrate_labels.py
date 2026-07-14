#!/usr/bin/env python3
"""Label-space migration (ND-3, owner-approved 2026-07-14, option A).

Source of truth: docs/Review-F5/capability-map.json (ADR-002 §A3-aligned).
Emits datasets/label_migration_map.json and rewrites, in place:

- every dataset/holdout CSV with an ``intent`` column (datasets/,
  datasets/multilingual/, multilingual/test/*_holdout.csv),
- content/nlu_schema.json + content/localization/nlu_schema.*.json
  (intent keys, keyword_triggers, any intent cross-references),
- rows of the two REMOVED dialogue-act labels (``Cmd.SendMessage - yes/no``)
  are extracted to datasets/confirmation_fixtures.csv (polarity-tagged),
  never discarded.

Renames only, plus the two removals — no merges, no splits, so per-class
metrics stay comparable to the recorded baseline. Provenance dirs
(datasets/dialogflowData, datasets/multilingual/pending) are untouched.

Usage:
    python packages/buildtime/nlu_training/migrate_labels.py --dry-run
    python packages/buildtime/nlu_training/migrate_labels.py --apply
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CAP_MAP = REPO / "docs" / "Review-F5" / "capability-map.json"
OUT_MAP = REPO / "datasets" / "label_migration_map.json"
FIXTURES = REPO / "datasets" / "confirmation_fixtures.csv"

REMOVED_POLARITY = {"Cmd.SendMessage - yes": "yes", "Cmd.SendMessage - no": "no"}


def _dissolve_cmd_health(text: str) -> str:
    """Straggler cleanup: datasets/intent_data_new.csv still holds 43
    'Cmd.Health' rows that dissolve_cmd_health.py (which only processed the
    base file) never reassigned. Same documented first-match rule."""
    import re
    tl = text.lower()
    if re.search(r"\bstep", tl):
        return "Cmd.ActivityStep"
    if re.search(r"\bstand|\bstood", tl):
        return "Cmd.ActivityStand"
    if re.search(r"\bwalk", tl):
        return "Cmd.ActivityWalk"
    return "Cmd.ActivityExercise"

CSV_FILES = [
    *(REPO / "datasets").glob("*.csv"),
    *(REPO / "datasets" / "multilingual").glob("*.csv"),
    *(REPO / "multilingual" / "test").glob("*_holdout.csv"),
]

SCHEMA_FILES = [
    REPO / "content" / "nlu_schema.json",
    *(REPO / "content" / "localization").glob("nlu_schema.*.json"),
]


def load_map() -> tuple[dict[str, str], set[str]]:
    cap = json.loads(CAP_MAP.read_text())
    rename: dict[str, str] = {}
    removed: set[str] = set()
    for old, entry in cap["intents"].items():
        new = entry["proposed_intent"]
        if new is None:
            removed.add(old)
        else:
            rename[old] = new
    assert removed == set(REMOVED_POLARITY), removed
    return rename, removed


def lang_of_csv(path: Path) -> str:
    name = path.name.lower()
    for code, keys in {"fr": ("french", "fr"), "de": ("german", "de"),
                       "da": ("danish", "da")}.items():
        if any(k in name for k in keys):
            return code
    return "en"


def migrate_csv(path: Path, rename: dict, removed: set, apply: bool,
                fixtures: list) -> dict | None:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return None
    header = rows[0]
    if "intent" not in header:
        return None
    idx = header.index("intent")
    text_idx = header.index("text") if "text" in header else None
    n_renamed = n_removed = n_unknown = 0
    out = [header]
    for row in rows[1:]:
        if len(row) <= idx:
            out.append(row)
            continue
        label = row[idx]
        if label == "Cmd.Health":  # pre-migration dissolution (see helper)
            row = list(row)
            row[idx] = _dissolve_cmd_health(row[text_idx] if text_idx is not None else "")
            label = row[idx]
        if label in removed:
            n_removed += 1
            fixtures.append([row[text_idx] if text_idx is not None else "",
                             lang_of_csv(path), REMOVED_POLARITY[label], path.name])
            continue  # dropped from source, preserved in fixtures
        if label in rename:
            row = list(row)
            row[idx] = rename[label]
            n_renamed += 1
        elif label not in rename.values():
            n_unknown += 1
        out.append(row)
    if apply and (n_renamed or n_removed):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(out)
    return {"file": str(path.relative_to(REPO)), "renamed": n_renamed,
            "removed_to_fixtures": n_removed, "unknown_labels": n_unknown}


def migrate_schema(path: Path, rename: dict, removed: set, apply: bool) -> dict:
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    n_keys = 0
    if "intents" in data:
        new_intents = {}
        for k, v in data["intents"].items():
            if k in removed:
                continue
            new_intents[rename.get(k, k)] = v
            if k in rename:
                n_keys += 1
        data["intents"] = new_intents
    # deep string replacement for cross-references (keyword_triggers, followups)
    def walk(obj):
        if isinstance(obj, dict):
            return {k: walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(v) for v in obj if not (isinstance(v, dict)
                    and v.get("intent") in removed)]
        if isinstance(obj, str) and obj in rename:
            return rename[obj]
        return obj
    data = walk(data)
    if apply:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    blob = json.dumps(data, ensure_ascii=False)
    residue = [o for o in list(rename) + list(removed) if f'"{o}"' in blob]
    return {"file": str(path.relative_to(REPO)), "keys_renamed": n_keys,
            "residual_old_labels": residue}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    rename, removed = load_map()
    report = {"renames": len(rename), "removed": sorted(removed), "csv": [], "schema": []}
    fixtures: list = []
    for path in sorted(set(CSV_FILES)):
        r = migrate_csv(path, rename, removed, args.apply, fixtures)
        if r and (r["renamed"] or r["removed_to_fixtures"] or r["unknown_labels"]):
            report["csv"].append(r)
    for path in sorted(set(SCHEMA_FILES)):
        report["schema"].append(migrate_schema(path, rename, removed, args.apply))

    if args.apply:
        OUT_MAP.write_text(json.dumps(
            {"$comment": "ND-3 label migration (option A). old -> new; null = removed "
                         "(dialogue act dissolved into confirmation lexicons).",
             "applied": "2026-07-14",
             "map": {**rename, **{k: None for k in removed}}},
            indent=2) + "\n")
        with open(FIXTURES, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["text", "lang", "polarity", "source_file"])
            w.writerows(fixtures)
        report["fixtures_written"] = len(fixtures)

    print(json.dumps(report, indent=2))
    bad = [r for r in report["csv"] if r["unknown_labels"]]
    residue = [r for r in report["schema"] if r["residual_old_labels"]]
    if bad or residue:
        print(f"\nPROBLEMS: unknown_labels in {len(bad)} csvs; residue in "
              f"{len(residue)} schemas", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
