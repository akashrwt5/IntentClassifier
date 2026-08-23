#!/usr/bin/env python3
"""Derive dev_near / dev_hard from the honest holdout. Deterministic, re-runnable.

WHAT THIS PRODUCES AND WHY
--------------------------
``holdout_honest.csv`` is disjoint from ``train.csv`` by exact normalised text --
0 leaks, verified -- but 44.7% of its rows are near-duplicates of a training
utterance: same words, different arrangement. Shared words are what a lexical
model scores on, so those rows reward memorisation and say almost nothing about
the paraphrase generalisation this project is spending megabytes to buy. Mixed
into one number they drag the whole instrument toward the thing it is supposed
to be measuring against.

So the holdout is partitioned, not filtered:

  dev_near.csv   near-duplicate rows. Regression detection only -- it answers
                 "did this change break something that used to work", and it is
                 NOT a generalisation estimate. Never gate on it.
  dev_hard.csv   the remainder. The primary decision instrument for P2-P8.

A DERIVATION, NOT A FILE
------------------------
Near-duplication is a relation between the holdout and ``train.csv``, not a
property of a row. Add rows to training and a dev_hard row can become a
near-duplicate without anything in dev_hard changing -- it stops being hard
while still looking untouched. A hand-frozen CSV cannot express that; a script
plus a manifest can, which is why this exists as code.

That does NOT mean re-running it whenever training data moves. ``dev_hard`` is
frozen for the whole of P2-P8: a ruler that changes mid-run makes each phase
incomparable with the last. When the Super Dataset lands it enters TRAINING
only, and ``check_instruments.py`` fails the build on any training row that
near-duplicates a dev_hard row -- so the landing is a filter over new training
rows, not a rebuild of the instrument.

USAGE
    python3 split_dev_sets.py --dry-run      # measure and report, write nothing
    python3 split_dev_sets.py                # write the sets and the manifest
    python3 split_dev_sets.py --check        # CI: recompute and diff, write nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from instruments import (  # noqa: E402
    minimum_detectable_effect,
    near_duplicate_flags,
    read_rows,
    sha256_file,
    write_rows,
)

HERE = Path(__file__).resolve().parent

# --- Outside this directory ---------------------------------------------
# Evaluation data lives with the language pack, not with the compression code.
# Lifting this directory into another project means repointing these two.
REPO = HERE.parents[1]
PACK = REPO / "language_packs" / "en"
TRAIN_CSV = PACK / "train.csv"
HOLDOUT_CSV = PACK / "holdout_honest.csv"
# ------------------------------------------------------------------------

THRESHOLD = 0.8
# Reported alongside every MDE. 0.15 is the working assumption for two encoders
# of similar quality on the same rows; the manifest carries all three so a
# reader can pick the one matching an actual observed discordance.
DISCORDANCE_GRID = (0.10, 0.15, 0.25)

CHARTERS = {
    "dev_hard": (
        "PRIMARY DECISION INSTRUMENT for P2-P8. Holdout rows with no near-duplicate "
        "in train.csv. Decide with McNemar's test on discordant items, never by "
        "comparing two accuracy numbers. Frozen for the duration of the plan."
    ),
    "dev_near": (
        "REGRESSION DETECTION ONLY. Holdout rows that near-duplicate a training "
        "utterance. A high score here is evidence of memorisation, not of "
        "generalisation. Never gate a release on it and never report it alone."
    ),
}


def build(threshold: float, out_dir: Path) -> dict:
    for path in (TRAIN_CSV, HOLDOUT_CSV):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")

    _, train_rows = read_rows(TRAIN_CSV)
    fields, hold_rows = read_rows(HOLDOUT_CSV)

    matches = near_duplicate_flags(
        [r["text"] for r in hold_rows], [r["text"] for r in train_rows], threshold
    )

    near = [r for r, m in zip(hold_rows, matches, strict=True) if m is not None]
    hard = [r for r, m in zip(hold_rows, matches, strict=True) if m is None]
    assert len(near) + len(hard) == len(hold_rows)

    hard_intents = Counter(r["intent"].strip() for r in hard)
    all_intents = Counter(r["intent"].strip() for r in hold_rows)
    unrepresented = sorted(i for i in all_intents if i not in hard_intents)
    thin = sorted(i for i, n in hard_intents.items() if n < 5)

    manifest = {
        "_note": (
            "Derived by split_dev_sets.py. Deterministic: no seed, no sampling. "
            "Re-running on unchanged inputs reproduces these files byte for byte. "
            "dev_hard is FROZEN for P2-P8 -- see the module docstring before "
            "regenerating it."
        ),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": {
            "near_duplicate": "token-set Jaccard >= threshold on normalised text",
            "threshold": threshold,
            "normaliser": "instruments.normalize_text (vendored from "
            "packages/buildtime/nlu_training/leakage.py; parity asserted "
            "by test_instruments.py)",
            "exactness": "prefix filtering prunes candidates but computes the full "
            "Jaccard for each survivor -- no pair is missed",
        },
        "inputs": {
            "train.csv": {"rows": len(train_rows), "sha256": sha256_file(TRAIN_CSV)},
            "holdout_honest.csv": {"rows": len(hold_rows), "sha256": sha256_file(HOLDOUT_CSV)},
        },
        "outputs": {
            "dev_near.csv": {"rows": len(near), "charter": CHARTERS["dev_near"]},
            "dev_hard.csv": {"rows": len(hard), "charter": CHARTERS["dev_hard"]},
        },
        "near_duplicate_share": round(len(near) / len(hold_rows), 4) if hold_rows else None,
        "minimum_detectable_effect": {
            name: {
                f"discordance_{d:.2f}": round(minimum_detectable_effect(n, d), 4)
                for d in DISCORDANCE_GRID
            }
            for name, n in (
                ("dev_hard", len(hard)),
                ("dev_near", len(near)),
                ("holdout_honest", len(hold_rows)),
            )
        },
        "dev_hard_intent_coverage": {
            "intents_in_holdout": len(all_intents),
            "intents_in_dev_hard": len(hard_intents),
            "intents_absent_from_dev_hard": unrepresented,
            "intents_with_fewer_than_5_rows": thin,
        },
    }

    written = {}
    if out_dir is not None:
        write_rows(out_dir / "dev_near.csv", fields, near)
        write_rows(out_dir / "dev_hard.csv", fields, hard)
        for name in ("dev_near.csv", "dev_hard.csv"):
            manifest["outputs"][name]["sha256"] = sha256_file(out_dir / name)
            written[name] = out_dir / name
        (out_dir / "dev_split.manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        written["dev_split.manifest.json"] = out_dir / "dev_split.manifest.json"

    return {"manifest": manifest, "written": written}


def report(manifest: dict) -> str:
    m = manifest
    lines = [
        f"holdout_honest.csv       {m['inputs']['holdout_honest.csv']['rows']:>6} rows",
        f"  near-duplicate         {m['outputs']['dev_near.csv']['rows']:>6} rows"
        f"   ({m['near_duplicate_share'] * 100:.1f}%)  -> dev_near.csv",
        f"  clean                  {m['outputs']['dev_hard.csv']['rows']:>6} rows"
        f"           -> dev_hard.csv",
        "",
        "minimum detectable effect (McNemar, alpha=0.05, 80% power)",
        f"{'instrument':<18}{'n':>7}" + "".join(f"{f'disc {d:.2f}':>12}" for d in DISCORDANCE_GRID),
    ]
    sizes = {
        "dev_hard": m["outputs"]["dev_hard.csv"]["rows"],
        "dev_near": m["outputs"]["dev_near.csv"]["rows"],
        "holdout_honest": m["inputs"]["holdout_honest.csv"]["rows"],
    }
    for name in m["minimum_detectable_effect"]:
        # Recomputed rather than read back from the manifest: the manifest stores
        # 4 decimals, and rounding 0.036525 -> 0.0365 -> "0.036" loses a digit the
        # plan's power table depends on. Round once, from the full value.
        row = f"{name:<18}{sizes[name]:>7}"
        for d in DISCORDANCE_GRID:
            row += f"{minimum_detectable_effect(sizes[name], d):>12.3f}"
        lines.append(row)

    cov = m["dev_hard_intent_coverage"]
    lines += [
        "",
        f"dev_hard covers {cov['intents_in_dev_hard']} of "
        f"{cov['intents_in_holdout']} holdout intents",
    ]
    if cov["intents_absent_from_dev_hard"]:
        lines.append(
            "  ABSENT (recall unmeasurable on dev_hard): "
            + ", ".join(cov["intents_absent_from_dev_hard"])
        )
    if cov["intents_with_fewer_than_5_rows"]:
        lines.append(
            "  thin (<5 rows, per-intent recall is noise): "
            + ", ".join(cov["intents_with_fewer_than_5_rows"])
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=PACK,
        help="where the derived sets are written (default: the en language pack, "
        "beside the holdout they come from)",
    )
    ap.add_argument("--dry-run", action="store_true", help="measure and report, write nothing")
    ap.add_argument(
        "--check",
        action="store_true",
        help="recompute and compare against what is on disk; non-zero on any "
        "difference. Writes nothing. This is the CI mode.",
    )
    args = ap.parse_args(argv)

    if args.check:
        return check(args.threshold, args.out_dir)

    result = build(args.threshold, None if args.dry_run else args.out_dir)
    print(report(result["manifest"]))
    if args.dry_run:
        print("\n(dry run -- nothing written)")
    else:
        print()
        for path in result["written"].values():
            print(f"wrote {path}")
    return 0


def check(threshold: float, out_dir: Path) -> int:
    """Recompute and diff against disk. The instrument must be reproducible."""
    manifest_path = out_dir / "dev_split.manifest.json"
    if not manifest_path.exists():
        print(f"FAIL: {manifest_path} not found -- run split_dev_sets.py first")
        return 1
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    fresh = build(threshold, None)["manifest"]

    problems = []
    for name in ("train.csv", "holdout_honest.csv"):
        a, b = on_disk["inputs"][name], fresh["inputs"][name]
        if a["sha256"] != b["sha256"]:
            problems.append(
                f"{name} changed since the split was frozen "
                f"({a['rows']} rows -> {b['rows']} rows). Every number measured on "
                f"dev_hard predates this change."
            )
    for name in ("dev_near.csv", "dev_hard.csv"):
        want = on_disk["outputs"][name]["rows"]
        got = fresh["outputs"][name]["rows"]
        if want != got:
            problems.append(f"{name}: manifest says {want} rows, recomputation gives {got}")
        path = out_dir / name
        if not path.exists():
            problems.append(f"{name}: missing from {out_dir}")
        elif on_disk["outputs"][name].get("sha256") != sha256_file(path):
            problems.append(f"{name}: on-disk file does not match its manifest hash")

    if problems:
        print("FAIL -- the derived instruments no longer match their manifest:\n")
        for p in problems:
            print(f"  - {p}")
        print("\nIf the inputs changed deliberately, re-run split_dev_sets.py and say why")
        print("in the commit. Note that doing so retires every dev_hard number measured")
        print("so far -- see R7 in the plan.")
        return 1

    print("OK -- dev_near and dev_hard reproduce from their inputs, hashes match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
