#!/usr/bin/env python3
"""
Bootstrap the English training corpus into `datasets/` without DVC.

WHY THIS EXISTS
---------------
The real datasets are DVC-managed and the configured remote is a LOCAL path
(`.dvc/config` -> `url = ../../dvc-store`), which exists only on the owner's
machine. Any other environment — CI, a fresh clone, a scheduled routine — gets
no training data at all, so every model, calibration and safety gate silently
skips (Review-F5 blocker B2).

This script unblocks the *English* work in the meantime. It materialises
`datasets/` from a tracked, provenance-stamped snapshot under
`data/bootstrap/en/`, so English training / calibration / evaluation can run in
any clone. It is a STOPGAP, not a replacement for DVC.

WHAT THE SNAPSHOT IS, AND WHAT IT IS NOT
----------------------------------------
The snapshot is the English master from the reference branch
`claude/claude-setup-architecture-ebqobs-Temperaturescaling-fixes`, migrated
from the old 59-label space to the shipped 57-label `domain.object.action`
taxonomy using `docs/Review-F5/capability-map.json` (which is machine-checked
to cover all 59 old labels exactly).

It is NOT the authoritative dataset:
  - it predates this branch's post-migration data-quality passes;
  - it is English only (fr/de/da masters are not recoverable from git);
  - it is the corpus from which the leaked `en_holdout.csv` was drawn, so it is
    the right input for BUILDING an honest holdout but must never be used as
    one.

Numbers produced from it are provisional. The authoritative baseline still
requires the real datasets. See `data/bootstrap/en/README.md`.

SAFETY
------
The script REFUSES to overwrite real data. If `datasets/` already holds a
training master (i.e. `dvc pull` succeeded), it no-ops and exits 0. Silently
replacing real data with a stale snapshot is exactly the class of error
Review-F5 is fixing, so it fails safe by default; `--force` is required to
override and says so loudly.

USAGE
    python scripts/ci/bootstrap_en_data.py            # materialise datasets/
    python scripts/ci/bootstrap_en_data.py --check    # verify only, write nothing
    python scripts/ci/bootstrap_en_data.py --build    # regenerate the tracked
                                                      # snapshot from the ref branch
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BOOTSTRAP = REPO / "data" / "bootstrap" / "en"
DATASETS = REPO / "datasets"
CAPABILITY_MAP = REPO / "docs" / "Review-F5" / "capability-map.json"
SCHEMA = REPO / "content" / "nlu_schema.json"

REF_BRANCH = "origin/claude/claude-setup-architecture-ebqobs-Temperaturescaling-fixes"

# Files copied out of the reference branch, with the column holding the label.
SOURCES: dict[str, str] = {
    "04_GENERATED_MASTER_training_data.csv": "intent",
    "semantic_holdout_2.csv": "expected_intent",
    "semantic_holdout_100.csv": "expected_intent",
    "semantic_oos.csv": "intent",
}
MASTER = "04_GENERATED_MASTER_training_data.csv"
# Rows whose label the migration DROPS (dialogue acts that dissolve into the
# confirmation flow, ND-3 Change 1). Preserved, never discarded.
CONFIRMATION_FIXTURES = "confirmation_fixtures.csv"
PROVENANCE = "PROVENANCE.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_migration() -> tuple[dict[str, str], set[str]]:
    """(old_label -> new_label, labels deliberately dropped)."""
    intents = json.loads(CAPABILITY_MAP.read_text(encoding="utf-8"))["intents"]
    mapping, dropped = {}, set()
    for old, spec in intents.items():
        new = spec.get("proposed_intent")
        if new:
            mapping[old] = new
        else:
            dropped.add(old)
    return mapping, dropped


def _shipped_labels() -> set[str]:
    return set(json.loads(SCHEMA.read_text(encoding="utf-8"))["intents"])


def _read(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def _write(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# --build : regenerate the tracked snapshot from the reference branch
# --------------------------------------------------------------------------- #

def build() -> int:
    mapping, dropped = _load_migration()
    shipped = _shipped_labels()
    BOOTSTRAP.mkdir(parents=True, exist_ok=True)

    ref_sha = subprocess.run(["git", "rev-parse", REF_BRANCH], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout.strip()

    prov: dict = {
        "_note": "Provisional English bootstrap corpus. NOT the authoritative "
                 "dataset — see README.md. Regenerate with "
                 "`python scripts/ci/bootstrap_en_data.py --build`.",
        "source_branch": REF_BRANCH,
        "source_commit": ref_sha,
        "migration_map": "docs/Review-F5/capability-map.json",
        "migrated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dropped_labels": sorted(dropped),
        "files": {},
    }

    held_all: list[dict] = []
    for name, label_col in SOURCES.items():
        raw = subprocess.run(["git", "show", f"{REF_BRANCH}:data/{name}"], cwd=REPO,
                             capture_output=True, check=True).stdout
        tmp = BOOTSTRAP / f".{name}.src"
        tmp.write_bytes(raw)
        source_sha = _sha256(tmp)
        fields, rows = _read(tmp)
        tmp.unlink()

        kept, held = [], []
        for row in rows:
            old = (row.get(label_col) or "").strip()
            if old in dropped:
                held.append(row)
                continue
            new = mapping.get(old)
            if new is None:
                print(f"FAIL: {name}: label {old!r} is in neither the map nor the "
                      f"dropped set — the migration is not exhaustive.")
                return 1
            row[label_col] = new
            kept.append(row)

        _write(BOOTSTRAP / name, fields, kept)
        prov["files"][name] = {
            "source_sha256": source_sha,
            "sha256": _sha256(BOOTSTRAP / name),
            "rows_in": len(rows),
            "rows_out": len(kept),
            "rows_held_back": len(held),
            "label_column": label_col,
        }

        # ND-3: dialogue-act rows are preserved as confirmation fixtures, from
        # EVERY source file — a holdout row expecting a label that no longer
        # exists cannot be scored, but it must not vanish either.
        for row in held:
            row["_origin_file"] = name
        held_all.extend(held)

        print(f"  {name}: {len(rows)} -> {len(kept)} rows "
              f"({len(held)} held back as confirmation fixtures)")

    if held_all:
        fixture_fields = ["_origin_file"] + [
            f for f in dict.fromkeys(k for r in held_all for k in r)
            if f != "_origin_file"
        ]
        _write(BOOTSTRAP / CONFIRMATION_FIXTURES, fixture_fields, held_all)
        prov["files"][CONFIRMATION_FIXTURES] = {
            "sha256": _sha256(BOOTSTRAP / CONFIRMATION_FIXTURES),
            "rows_out": len(held_all),
            "origin": "dialogue-act rows dropped by the ND-3 migration, from all "
                      "source files; retained for confirmation-flow regression use",
        }

    # Exhaustiveness check: the migrated master must land inside the shipped space.
    _, master_rows = _read(BOOTSTRAP / MASTER)
    produced = {r["intent"].strip() for r in master_rows}
    if produced - shipped:
        print(f"FAIL: migrated labels absent from content/nlu_schema.json: "
              f"{sorted(produced - shipped)}")
        return 1
    prov["labels_produced"] = len(produced)
    prov["labels_shipped"] = len(shipped)
    prov["labels_shipped_without_data"] = sorted(shipped - produced)

    (BOOTSTRAP / PROVENANCE).write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
    print(f"OK: snapshot built — {len(produced)}/{len(shipped)} shipped labels covered")
    if prov["labels_shipped_without_data"]:
        print(f"  NOTE: no rows for {prov['labels_shipped_without_data']}")
    return 0


# --------------------------------------------------------------------------- #
# default : materialise datasets/ from the tracked snapshot
# --------------------------------------------------------------------------- #

def _real_data_present() -> bool:
    """True when datasets/ already holds a master that did NOT come from here."""
    target = DATASETS / MASTER
    if not target.exists():
        return False
    prov_path = BOOTSTRAP / PROVENANCE
    if not prov_path.exists():
        return True
    expected = json.loads(prov_path.read_text(encoding="utf-8"))["files"][MASTER]["sha256"]
    return _sha256(target) != expected


def materialise(check_only: bool, force: bool) -> int:
    if not (BOOTSTRAP / PROVENANCE).exists():
        print("FAIL: no tracked snapshot. Run with --build first.")
        return 1

    prov = json.loads((BOOTSTRAP / PROVENANCE).read_text(encoding="utf-8"))

    # Integrity: the tracked snapshot must match its recorded hashes.
    for name, meta in prov["files"].items():
        path = BOOTSTRAP / name
        if not path.exists():
            print(f"FAIL: snapshot file missing: {path.relative_to(REPO)}")
            return 1
        if _sha256(path) != meta["sha256"]:
            print(f"FAIL: {name} does not match its recorded sha256 — the snapshot "
                  f"was edited by hand. Regenerate with --build.")
            return 1

    if _real_data_present() and not force:
        print("Real datasets/ content is present (dvc pull succeeded, or the master "
              "was replaced). Leaving it untouched — the bootstrap snapshot is a "
              "stale stopgap and must never overwrite authoritative data.")
        return 0

    if check_only:
        print(f"OK: snapshot verified ({len(prov['files'])} files). Nothing written.")
        return 0

    DATASETS.mkdir(parents=True, exist_ok=True)
    for name in prov["files"]:
        (DATASETS / name).write_bytes((BOOTSTRAP / name).read_bytes())
    print(f"OK: materialised {len(prov['files'])} files into datasets/ "
          f"from the bootstrap snapshot ({prov['source_commit'][:8]}).")
    print("  PROVISIONAL DATA — English only, pre-dates this branch's data-quality "
          "passes. Do not publish these numbers as the authoritative baseline.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true",
                    help="regenerate the tracked snapshot from the reference branch")
    ap.add_argument("--check", action="store_true",
                    help="verify the snapshot, write nothing")
    ap.add_argument("--force", action="store_true",
                    help="overwrite datasets/ even if real data is present (dangerous)")
    args = ap.parse_args(argv)
    return build() if args.build else materialise(args.check, args.force)


if __name__ == "__main__":
    sys.exit(main())
