#!/usr/bin/env python3
"""
Produce the final, leak-free English training file.

    train_merged.csv  ──(remove every row that appears in ANY eval set)──►  data/en/train.csv

A phrase cannot be on both sides. `--protect` decides which side wins:

  --protect eval      (default)  drop the row from TRAINING; eval sets untouched
  --protect training            keep ALL training rows; drop the row from the
                                EVAL set instead, and rewrite that eval file

`--protect training` is the right choice when the training corpus is the asset
you care about and the eval set is regenerable. It shrinks the eval set, so
scores are no longer comparable with earlier runs on the old eval file — the
summary records the new row count so this stays visible.

Matching is done on the student tokenizer's view (punctuation discarded), because
that is what the model actually sees: "volume up" and "volume up?" are one input.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --source ../semantic_project/57semanitc/train_merged.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts.common import load_rows, read_csv, token_key  # noqa: E402

DEFAULT_SOURCE = (
    Path(__file__).resolve().parents[2] / "semantic_project" / "57semanitc" / "train_merged.csv"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--out", type=Path, default=config.TRAIN_CSV)
    ap.add_argument(
        "--protect",
        choices=["eval", "training"],
        default="training",
        help="which side keeps a phrase that appears on both sides",
    )
    args = ap.parse_args()

    rows = read_csv(args.source)
    print(f"source rows : {len(rows)}  ({args.source.name})")

    eval_keys: dict[str, set] = {}
    for name, path in (
        ("locked", config.LOCKED_TEST),
        ("stress", config.STRESS_TEST),
        ("ood", config.OOD_TEST),
    ):
        if path.exists():
            eval_keys[name] = {token_key(t) for t, _ in load_rows(path)}
            print(f"  eval '{name}': {len(eval_keys[name])} unique keys")

    all_eval = set().union(*eval_keys.values()) if eval_keys else set()
    train_keys = {token_key(r["text"]) for r in rows if token_key(r["text"])}

    eval_rows_removed: Counter = Counter()
    if args.protect == "training":
        # training corpus wins: strip the collisions out of the EVAL files
        for name, path in (
            ("locked", config.LOCKED_TEST),
            ("stress", config.STRESS_TEST),
            ("ood", config.OOD_TEST),
        ):
            if not path.exists():
                continue
            ev = read_csv(path)
            fields = list(ev[0].keys())
            tcol = next(c for c in fields if c.lower() in ("text", "utterance", "phrase"))
            keep_ev = [r for r in ev if token_key(r[tcol]) not in train_keys]
            n_drop = len(ev) - len(keep_ev)
            if n_drop:
                with open(path, "w", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
                    w.writerows(keep_ev)
                eval_rows_removed[name] = n_drop
                eval_keys[name] = {token_key(r[tcol]) for r in keep_ev}
                print(f"  eval '{name}': dropped {n_drop} colliding rows -> {len(keep_ev)}")
        all_eval = set().union(*eval_keys.values()) if eval_keys else set()

    kept, removed = [], Counter()
    seen = set()
    dups = 0
    for r in rows:
        k = token_key(r["text"])
        if not k:
            continue
        if k in all_eval:
            for name, keys in eval_keys.items():
                if k in keys:
                    removed[name] += 1
                    break
            continue
        if k in seen:
            dups += 1
            continue
        seen.add(k)
        kept.append(r)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "intent"])
        w.writeheader()
        for r in kept:
            w.writerow({"text": r["text"], "intent": r["intent"]})

    counts = Counter(r["intent"] for r in kept)
    vals = sorted(counts.values())
    no_fb = {k: v for k, v in counts.items() if k != config.FALLBACK_INTENT}

    # final independent verification
    kept_keys = {token_key(r["text"]) for r in kept}
    residual = {n: len(kept_keys & k) for n, k in eval_keys.items()}

    summary = {
        "source": str(args.source),
        "output": str(args.out),
        "source_rows": len(rows),
        "protect": args.protect,
        "removed_from_training": dict(removed),
        "removed_from_eval_sets": dict(eval_rows_removed),
        "removed_duplicates": dups,
        "final_rows": len(kept),
        "intents": len(counts),
        "min_per_intent": vals[0],
        "max_per_intent": vals[-1],
        "imbalance_ratio": round(vals[-1] / vals[0], 2),
        "imbalance_ratio_excluding_fallback": round(max(no_fb.values()) / min(no_fb.values()), 2),
        "intents_under_50_rows": sum(1 for v in vals if v < 50),
        "residual_eval_overlap": residual,
        "class_weights": {
            k: round(len(kept) / (len(counts) * v), 4) for k, v in sorted(counts.items())
        },
        "per_intent_counts": dict(counts.most_common()),
    }
    report = config.REPORTS / "prepare_data.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nprotect mode           : {args.protect}")
    print(f"removed from TRAINING  : {dict(removed) or '{}'}")
    print(f"removed from EVAL sets : {dict(eval_rows_removed) or '{}'}")
    print(f"removed (duplicates)   : {dups}")
    print(f"FINAL                  : {len(kept)} rows / {len(counts)} intents")
    print(f"  min/max per intent   : {vals[0]} / {vals[-1]}")
    print(f"  intents under 50 rows: {summary['intents_under_50_rows']}")
    print(f"  residual overlap     : {residual}")
    print(f"\nwrote {args.out}")
    print(f"wrote {report}")

    if any(residual.values()):
        raise SystemExit("ABORT: eval rows still present in training data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
