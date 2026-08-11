#!/usr/bin/env python3
"""
Build a class-balanced copy of train.csv.

Strategy (chosen 2026-08-10):
  - Majority classes are DOWNSAMPLED to CAP (random, seeded).
  - Minority classes are left untouched (NO duplication, NO synthetic text).
  - `Default Fallback Intent` is EXEMPT from the cap.

    Why exempt: it is the OOD / rejection class. Its semantic space is unbounded
    (every non-command sentence in the world), unlike e.g. Cmd.VolumeIncrease.
    Capping it to 200 would drop its share from 14.1% to ~3%, which is the most
    likely way to regress OOD rejection — and the ship bar says OOD rejection
    must never regress.

Output has the SAME schema as train.csv: text,intent

Nothing is invented: every output row is a verbatim row from the source.

  !!  SOURCE IS reference_train.csv, NOT train.csv  !!

  train.csv (8,430 rows) CONTAINS ALL 1,686 locked-test rows verbatim
  (100% overlap, verified 2026-08-10). Anything trained on train.csv and then
  scored on locked_test_57intent.csv is reporting memorisation, not accuracy.

  create_locked_57intent_split.py already warned about this in its own
  docstring and emitted the clean 80% split as reference_train.csv
  (6,744 rows, 0 overlap with the locked test). That is the correct base.

Usage:
    python build_balanced_train.py                 # cap=200, clean source
    python build_balanced_train.py --cap 150
    python build_balanced_train.py --source train.csv   # will FAIL the leak guard
"""

import argparse
import collections
import csv
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent

SRC = HERE / "v3_57intent_locked_eval" / "reference_train.csv"
LOCKED = HERE / "v3_57intent_locked_eval" / "locked_test_57intent.csv"

FALLBACK_INTENT = "Default Fallback Intent"
SEED = 42


def read_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=200, help="max rows per intent")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--no-exempt-fallback",
        action="store_true",
        help="also cap Default Fallback Intent (NOT recommended)",
    )
    ap.add_argument("--source", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=HERE / "train_balanced.csv")
    ap.add_argument(
        "--allow-leak",
        action="store_true",
        help="do not fail when output overlaps the locked test (debug only)",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)

    rows = read_rows(args.source)
    if not rows:
        raise SystemExit(f"no rows in {args.source}")

    by_intent: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_intent[r["intent"]].append(r)

    # ---------------- integrity checks on the SOURCE ----------------
    seen: dict[tuple[str, str], int] = collections.Counter(
        (norm(r["text"]), r["intent"]) for r in rows
    )
    dup_rows = sum(v - 1 for v in seen.values() if v > 1)

    text_to_intents: dict[str, set] = collections.defaultdict(set)
    for r in rows:
        text_to_intents[norm(r["text"])].add(r["intent"])
    conflicts = {t: i for t, i in text_to_intents.items() if len(i) > 1}

    # ---------------- balance ----------------
    kept: list[dict] = []
    per_class: dict[str, dict] = {}

    for intent in sorted(by_intent):
        pool = by_intent[intent]
        exempt = intent == FALLBACK_INTENT and not args.no_exempt_fallback

        if exempt or len(pool) <= args.cap:
            chosen = list(pool)
            action = "exempt" if exempt else "kept"
        else:
            chosen = rng.sample(pool, args.cap)
            action = "downsampled"

        kept.extend(chosen)
        per_class[intent] = {
            "before": len(pool),
            "after": len(chosen),
            "action": action,
        }

    # keep output order stable and shuffled (so no intent-ordered blocks)
    rng.shuffle(kept)

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "intent"])
        w.writeheader()
        for r in kept:
            w.writerow({"text": r["text"], "intent": r["intent"]})

    # ---------------- leakage guard vs the locked test ----------------
    leak_count = None
    if LOCKED.exists():
        locked = read_rows(LOCKED)
        tcol = next((c for c in locked[0] if c.lower() in ("text", "utterance")), None)
        if tcol:
            locked_texts = {norm(r[tcol]) for r in locked}
            leak = {norm(r["text"]) for r in kept} & locked_texts
            leak_count = len(leak)

    after = collections.Counter(r["intent"] for r in kept)
    vals = sorted(after.values())

    summary = {
        "source": str(args.source),
        "output": str(args.out),
        "strategy": "cap-only downsample, minority untouched, fallback exempt",
        "cap": args.cap,
        "seed": args.seed,
        "fallback_exempt": not args.no_exempt_fallback,
        "duplicated_rows_created": 0,
        "synthetic_text": False,
        "rows_before": len(rows),
        "rows_after": len(kept),
        "intents": len(after),
        "min_per_intent": vals[0],
        "max_per_intent": vals[-1],
        "imbalance_ratio_before": round(
            max(len(v) for v in by_intent.values()) / min(len(v) for v in by_intent.values()),
            2,
        ),
        "imbalance_ratio_after": round(vals[-1] / vals[0], 2),
        # the headline ratio is dominated by the deliberately-exempt fallback
        # class; this is the number that describes the real intents.
        "imbalance_ratio_after_excluding_fallback": round(
            max(v for k, v in after.items() if k != FALLBACK_INTENT)
            / min(v for k, v in after.items() if k != FALLBACK_INTENT),
            2,
        ),
        "fallback_share_before": round(len(by_intent[FALLBACK_INTENT]) / len(rows) * 100, 1),
        "fallback_share_after": round(after[FALLBACK_INTENT] / len(kept) * 100, 1),
        "source_duplicate_rows": dup_rows,
        "source_label_conflicts": len(conflicts),
        "locked_test_overlap_rows": leak_count,
        "per_class": per_class,
    }

    report = args.out.with_name(args.out.stem + "_summary.json")
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"rows   {len(rows)} -> {len(kept)}")
    print(f"ratio  {summary['imbalance_ratio_before']}x -> {summary['imbalance_ratio_after']}x")
    print(f"min/max per intent: {vals[0]} / {vals[-1]}")
    print(
        f"downsampled classes: {sum(1 for v in per_class.values() if v['action'] == 'downsampled')}"
    )
    print(f"duplicated rows created: 0")
    if leak_count is not None:
        status = "OK" if leak_count == 0 else "!! LEAK !!"
        print(f"locked-test overlap: {leak_count} rows  {status}")
    print(f"\nwrote {args.out}")
    print(f"wrote {report}")

    if leak_count and not args.allow_leak:
        raise SystemExit(
            f"\nABORT: {leak_count} output rows also appear in the locked test.\n"
            f"Source was {args.source}.\n"
            f"Use v3_57intent_locked_eval/reference_train.csv as the source, or "
            f"pass --allow-leak if you really mean it."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
