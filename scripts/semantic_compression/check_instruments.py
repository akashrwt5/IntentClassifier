#!/usr/bin/env python3
"""CI guard for the evaluation instruments. Exit non-zero if a ruler has moved.

WHAT THIS PREVENTS
------------------
Every defect in this project's register produced a confident number rather than
a crash, and the ones that survived longest were the ones where a measuring
instrument changed underneath a measurement. B9 is the canonical case: the
honest holdout's manifest was frozen at ``cc46010``, then 77 training rows were
added (``ce0d469``) and every label in both files was rewritten (``af4a88b``) at
an unchanged row count. Nothing failed. The manifest simply stopped describing
the files it named, and every number measured afterwards inherited the error.

FIVE CHECKS
-----------
1. **Holdout manifest freshness.** train.csv and holdout_honest.csv still hash to
   what the manifest says.
2. **Exact disjointness.** No normalised holdout utterance appears in train.csv.
   This is the property the honest holdout exists to have; it is cheap to check
   and catastrophic to lose.
3. **Partition integrity.** dev_near + dev_hard reconstruct holdout_honest
   exactly -- no dropped row, no row in both.
4. **Split reproducibility.** The derived sets recompute from their inputs and
   match their own manifest hashes.
5. **dev_hard contamination.** No training row near-duplicates a dev_hard row.

Check 5 is the one that matters later. Today it passes trivially, because
dev_hard is *defined* as the holdout rows with no near-duplicate in train.csv.
Its purpose is the Super Dataset: when tens of thousands of generated rows enter
training, some will paraphrase a dev_hard row, and dev_hard will quietly stop
being hard. The only visible symptom would be that every model suddenly looks
better. This check turns that into a build failure that names the offending
training rows, so the fix is to drop them from training -- a filter over new
data, not a rebuild of the instrument.

USAGE
    python3 check_instruments.py              # CI mode
    python3 check_instruments.py --refreeze   # deliberate: re-pin the manifest
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import split_dev_sets  # noqa: E402
from instruments import (  # noqa: E402
    near_duplicate_flags,
    normalize_text,
    read_rows,
    sha256_file,
)

PACK = split_dev_sets.PACK
TRAIN_CSV = split_dev_sets.TRAIN_CSV
HOLDOUT_CSV = split_dev_sets.HOLDOUT_CSV
HOLDOUT_MANIFEST = PACK / "extras" / "holdout_honest.manifest.json"
DEV_HARD = PACK / "dev_hard.csv"
DEV_NEAR = PACK / "dev_near.csv"

MAX_LISTED = 10


class Failure(list):
    """Collected problems. Kept as a list so every check runs before reporting."""


def check_manifest_freshness(problems: Failure) -> None:
    if not HOLDOUT_MANIFEST.exists():
        problems.append(f"{HOLDOUT_MANIFEST.name}: missing")
        return
    m = json.loads(HOLDOUT_MANIFEST.read_text(encoding="utf-8"))
    for name, path in (("train.csv", TRAIN_CSV), ("holdout_honest.csv", HOLDOUT_CSV)):
        recorded = m.get("sha256", {}).get(name)
        actual = sha256_file(path)
        if recorded != actual:
            rows = sum(1 for _ in path.open("rb")) - 1
            want = m.get("rows", {}).get(name.split(".")[0].replace("holdout_honest", "holdout"))
            problems.append(
                f"{name}: hash does not match {HOLDOUT_MANIFEST.name} "
                f"(manifest rows={want}, on disk={rows}). Every number measured against "
                f"this holdout since the manifest was frozen is measured against a "
                f"different file than the one the manifest describes."
            )


def check_exact_disjointness(problems: Failure) -> None:
    _, train_rows = read_rows(TRAIN_CSV)
    _, hold_rows = read_rows(HOLDOUT_CSV)
    train_norm = {normalize_text(r["text"]) for r in train_rows}
    train_norm.discard("")
    leaks = sorted({r["text"] for r in hold_rows if normalize_text(r["text"]) in train_norm})
    if leaks:
        problems.append(
            f"holdout_honest.csv: {len(leaks)} utterances also appear in train.csv "
            f"after normalisation. The honest holdout is no longer honest. "
            f"First {min(len(leaks), MAX_LISTED)}: " + "; ".join(leaks[:MAX_LISTED])
        )


def check_partition(problems: Failure) -> None:
    for path in (DEV_NEAR, DEV_HARD):
        if not path.exists():
            problems.append(f"{path.name}: missing -- run split_dev_sets.py")
            return
    _, hold = read_rows(HOLDOUT_CSV)
    _, near = read_rows(DEV_NEAR)
    _, hard = read_rows(DEV_HARD)

    def key(rows):
        return sorted((r["text"], r["intent"]) for r in rows)

    if len(near) + len(hard) != len(hold):
        problems.append(
            f"partition: dev_near ({len(near)}) + dev_hard ({len(hard)}) "
            f"!= holdout_honest ({len(hold)})"
        )
    if key(near + hard) != key(hold):
        problems.append(
            "partition: dev_near + dev_hard do not reconstruct holdout_honest.csv "
            "row for row -- a row was dropped, duplicated or edited"
        )


def check_dev_hard_contamination(problems: Failure) -> None:
    if not DEV_HARD.exists():
        return
    _, train_rows = read_rows(TRAIN_CSV)
    _, hard_rows = read_rows(DEV_HARD)
    matches = near_duplicate_flags(
        [r["text"] for r in hard_rows],
        [r["text"] for r in train_rows],
        split_dev_sets.THRESHOLD,
    )
    hits = [(r["text"], m) for r, m in zip(hard_rows, matches, strict=True) if m is not None]
    if hits:
        listed = "\n".join(
            f"        dev_hard: {h!r}\n        train  : {t!r}" for h, t in hits[:MAX_LISTED]
        )
        problems.append(
            f"dev_hard is contaminated: {len(hits)} of {len(hard_rows)} rows now have a "
            f"near-duplicate in train.csv (Jaccard >= {split_dev_sets.THRESHOLD}). "
            f"dev_hard has stopped being hard.\n"
            f"      Fix by removing the offending TRAINING rows, not by re-splitting "
            f"dev_hard -- re-splitting retires every number measured on it so far.\n"
            f"      First {min(len(hits), MAX_LISTED)}:\n{listed}"
        )


def refreeze() -> int:
    """Re-pin the holdout manifest to the files as they stand, keeping the record."""
    m = json.loads(HOLDOUT_MANIFEST.read_text(encoding="utf-8"))
    _, train_rows = read_rows(TRAIN_CSV)
    _, hold_rows = read_rows(HOLDOUT_CSV)

    before = dict(m.get("sha256", {}))
    before_rows = dict(m.get("rows", {}))
    now_sha = {"train.csv": sha256_file(TRAIN_CSV), "holdout_honest.csv": sha256_file(HOLDOUT_CSV)}
    if before == now_sha and before_rows == {"train": len(train_rows), "holdout": len(hold_rows)}:
        print("manifest already matches the files on disk -- nothing to refreeze.")
        return 0

    m.setdefault("amendments", []).append(
        {
            "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "by": "scripts/semantic_compression/check_instruments.py --refreeze",
            "what": (
                "Re-pinned the hashes and row counts to the files as they stand. The "
                "partition itself was NOT rebuilt and the holdout was NOT re-split: the "
                "train/holdout boundary is the one built at cc46010, and exact normalised "
                "disjointness is re-verified by check_instruments.py at every run."
            ),
            "why_it_had_drifted": [
                "The 2026-07-26T12:40:30Z label amendment edited rows in both files "
                "without refreshing the hashes recorded above it, so the manifest has "
                "been describing pre-amendment files ever since.",
                "ce0d469 added 77 rows to train.csv (8,353 -> 8,430).",
                "af4a88b rewrote every label in both files from the modern taxonomy back "
                "to Cmd.*/Help_*; holdout_honest.csv changed content at an unchanged "
                "1,470 rows, which is why the drift was invisible.",
                "b6c2e83 moved the pack to language_packs/en/ and carried this manifest "
                "to extras/ without regenerating it.",
            ],
            "superseded": {"sha256": before, "rows": before_rows},
        }
    )
    m["rows"] = {"train": len(train_rows), "holdout": len(hold_rows)}
    m["intents"] = {
        "train": len({r["intent"].strip() for r in train_rows}),
        "holdout": len({r["intent"].strip() for r in hold_rows}),
    }
    m["sha256"] = now_sha
    HOLDOUT_MANIFEST.write_text(
        json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"re-pinned {HOLDOUT_MANIFEST}")
    print(f"  train.csv          {before_rows.get('train')} -> {len(train_rows)} rows")
    print(f"  holdout_honest.csv {before_rows.get('holdout')} -> {len(hold_rows)} rows")
    print("  previous hashes preserved under amendments[].superseded")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--refreeze",
        action="store_true",
        help="deliberately re-pin the holdout manifest to the current files, "
        "recording what drifted and why in amendments[]",
    )
    args = ap.parse_args(argv)

    if args.refreeze:
        return refreeze()

    problems = Failure()
    checks = (
        ("holdout manifest freshness", check_manifest_freshness),
        ("exact train/holdout disjointness", check_exact_disjointness),
        ("dev_near + dev_hard partition holdout_honest", check_partition),
        ("dev_hard contamination by train.csv", check_dev_hard_contamination),
    )
    for label, fn in checks:
        n_before = len(problems)
        fn(problems)
        print(f"  {'FAIL' if len(problems) > n_before else 'ok  '}  {label}")

    n_before = len(problems)
    if split_dev_sets.check(split_dev_sets.THRESHOLD, PACK) != 0:
        problems.append("dev split does not reproduce from its inputs (see above)")
    print(f"  {'FAIL' if len(problems) > n_before else 'ok  '}  dev split reproducibility")

    if problems:
        print(f"\n{len(problems)} problem(s):\n")
        for p in problems:
            print(f"  - {p}\n")
        return 1
    print("\nAll instrument checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
