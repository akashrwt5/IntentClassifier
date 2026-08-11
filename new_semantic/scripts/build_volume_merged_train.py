#!/usr/bin/env python3
"""
Merge base train.csv + audio_volume_rows.csv + volume_polarity_hard_negatives.csv
into data/en/train_vol5.csv with leak guard protection against all eval sets.

Output: data/en/train_vol5.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts.common import load_rows, token_key  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=config.DATA / "en" / "train_vol5.csv",
    )
    args = ap.parse_args()

    base_rows = load_rows(config.TRAIN_CSV)
    seen = {token_key(t): l for t, l in base_rows}

    # Load eval sets for leak guard
    eval_seen = set()
    for p in (
        config.LOCKED_TEST,
        config.STRESS_TEST,
        config.OOD_TEST,
        config.OOV_TEST,
        config.DATA / "eval" / "typo_test_en.csv",
    ):
        if p.exists():
            eval_seen |= {token_key(t) for t, _ in load_rows(p)}

    volume_sources = [
        config.DATA / "en" / "audio_volume_rows.csv",
        config.DATA / "en" / "volume_polarity_hard_negatives.csv",
    ]

    added = 0
    dropped_leak = 0
    for src_path in volume_sources:
        if not src_path.exists():
            print(f"Warning: {src_path} does not exist, skipping.")
            continue
        v_rows = load_rows(src_path)
        for text, intent in v_rows:
            tk = token_key(text)
            if tk in eval_seen:
                dropped_leak += 1
                continue
            if tk not in seen:
                seen[tk] = (text, intent)
                added += 1

    final_rows = []
    # Reassemble preserving original order where possible
    for tk, val in seen.items():
        if isinstance(val, tuple):
            final_rows.append(val)
        else:
            # base row was string intent
            pass

    # Simple clean rebuild
    dedup = {}
    for t, l in base_rows:
        tk = token_key(t)
        if tk not in eval_seen:
            dedup[tk] = (t, l)

    for src_path in volume_sources:
        if src_path.exists():
            for t, l in load_rows(src_path):
                tk = token_key(t)
                if tk not in eval_seen and tk not in dedup:
                    dedup[tk] = (t, l)

    merged_list = list(dedup.values())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "intent"])
        writer.writerows(merged_list)

    print(f"Base train rows : {len(base_rows)}")
    print(f"Volume added    : +{added} rows")
    print(f"Total merged    : {len(merged_list)} rows")
    print(f"Leak guard drops: {dropped_leak}")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
