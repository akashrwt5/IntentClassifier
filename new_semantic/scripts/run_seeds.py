#!/usr/bin/env python3
"""
Run one config across N seeds and report the MEAN, not a single lucky run.

Why this exists: with an identical config, seeds 42/1/7 produced OOD fallback
rates of 0.6551 / 0.3127 / 0.2134 — a 44-point spread. Every single-seed
comparison between v1-v5 was therefore uninterpretable. Nothing gets compared
on one seed again.

Usage:
    python scripts/run_seeds.py --name base --seeds 42 1 7
    python scripts/run_seeds.py --name unkaug --seeds 42 1 7 -- --unk-aug 0.04
    python scripts/run_seeds.py --compare base unkaug
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

METRICS = [
    ("locked.accuracy", "locked acc"),
    ("locked.macro_recall", "macro recall"),
    ("stress.accuracy", "stress acc"),
    ("ood.fallback_rate", "OOD raw"),
]


def dig(d, path):
    for k in path.split("."):
        d = d.get(k, {})
    return d if isinstance(d, (int, float)) else None


def collect(name, seeds):
    out = []
    for s in seeds:
        p = config.REPORTS / f"eval_{name}_s{s}.json"
        if p.exists():
            out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def summarize(name, seeds):
    evals = collect(name, seeds)
    if not evals:
        return None
    row = {"name": name, "n_seeds": len(evals)}
    for key, label in METRICS:
        vals = [dig(e, key) for e in evals]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        row[label] = {
            "mean": round(st.mean(vals), 4),
            "sd": round(st.stdev(vals), 4) if len(vals) > 1 else 0.0,
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }
    # best-threshold numbers, averaged
    rej, acc = [], []
    for e in evals:
        sw = e.get("threshold_sweep") or []
        if sw:
            b = max(sw, key=lambda s: s["harmonic"])
            rej.append(b["ood_reject"])
            acc.append(b["in_scope_acc"])
    if rej:
        row["OOD reject@best"] = {
            "mean": round(st.mean(rej), 4),
            "sd": round(st.stdev(rej), 4) if len(rej) > 1 else 0.0,
        }
        row["in-scope@best"] = {
            "mean": round(st.mean(acc), 4),
            "sd": round(st.stdev(acc), 4) if len(acc) > 1 else 0.0,
        }
    return row


def print_row(row):
    print(f"\n{row['name']}  (n={row['n_seeds']} seeds)")
    for k, v in row.items():
        if isinstance(v, dict):
            rng = f"  [{v['min']:.4f}-{v['max']:.4f}]" if "min" in v else ""
            print(f"  {k:<18} {v['mean']:.4f}  ± {v['sd']:.4f}{rng}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="config name; tags become <name>_s<seed>")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7])
    ap.add_argument("--compare", nargs="+", help="summarise existing configs")
    ap.add_argument(
        "extra", nargs=argparse.REMAINDER, help="args after -- are passed to train_en.py"
    )
    args = ap.parse_args()

    if args.compare:
        rows = [r for n in args.compare if (r := summarize(n, args.seeds))]
        for r in rows:
            print_row(r)
        if len(rows) >= 2:
            print("\n=== IS THE DIFFERENCE REAL? ===")
            print("(a gap smaller than ~2x the pooled sd is not evidence)")
            base = rows[0]
            for other in rows[1:]:
                print(f"\n  {base['name']} vs {other['name']}")
                for _, label in METRICS:
                    if label not in base or label not in other:
                        continue
                    d = other[label]["mean"] - base[label]["mean"]
                    pooled = (base[label]["sd"] + other[label]["sd"]) / 2
                    verdict = "REAL" if abs(d) > 2 * pooled and pooled > 0 else "not significant"
                    print(f"    {label:<16} {d:+.4f}   (pooled sd {pooled:.4f})  {verdict}")
        return 0

    if not args.name:
        ap.error("--name required unless --compare is used")

    passthru = [a for a in args.extra if a != "--"]
    for s in args.seeds:
        tag = f"{args.name}_s{s}"
        print(f"\n{'=' * 60}\n  {tag}\n{'=' * 60}")
        # check=True raises CalledProcessError, whose traceback points only at
        # subprocess.run — the real cause is the child's output above it, and
        # that framing has already sent one debugging session the wrong way.
        r = subprocess.run(
            [sys.executable, "scripts/train_en.py", "--tag", tag, "--seed", str(s)] + passthru
        )
        if r.returncode:
            raise SystemExit(
                f"\ntrain_en.py failed for {tag} (exit {r.returncode}). "
                f"The real error is printed above this line."
            )
        r = subprocess.run([sys.executable, "scripts/evaluate.py", "--tag", tag, "--sweep"])
        if r.returncode:
            raise SystemExit(
                f"\nevaluate.py failed for {tag} (exit {r.returncode}). "
                f"The error is printed above this line."
            )

    row = summarize(args.name, args.seeds)
    if row:
        print_row(row)
        (config.REPORTS / f"seeds_{args.name}.json").write_text(
            json.dumps(row, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
