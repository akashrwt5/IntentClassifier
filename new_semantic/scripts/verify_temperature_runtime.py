#!/usr/bin/env python3
"""
Measure what wiring `temperature` into StudentSemantic actually changed.

Temperature scaling is rank-preserving, so it CANNOT change argmax. That is
exactly why the omission survived: every accuracy check kept passing while the
engine gated on a miscalibrated confidence. The flip side is that turning it on
is not a no-op either — the gate is a threshold on the confidence, and T changes
the confidence, so the SAME gate number is a different gate once T is applied.

This script reports both halves of that:

  1. PARITY   argmax must be identical at T=1 and T=fitted, on every eval row.
              A single disagreement means something other than scaling happened.
  2. EFFECT   at a fixed gate, how many rows cross that were previously rejected
              (and vice versa), and what it does to in-scope accuracy and OOD
              rejection.

It runs the INSTALLED artifact through the INSTALLED runtime class, not a torch
reimplementation, so what it reports is what the device would do. No torch
required.

Usage:
    python scripts/verify_temperature_runtime.py
    python scripts/verify_temperature_runtime.py --gate 0.40 0.50 0.60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts.common import load_rows  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
INSTALL = REPO / "models" / "semantic_student" / "en"


def load_runtime():
    """Import StudentSemantic by path — importing the package would pull in
    engine -> classifier -> joblib/sklearn, none of which this needs."""
    import importlib.util

    src = REPO / "packages" / "runtime" / "nlu_engine" / "semantic.py"
    spec = importlib.util.spec_from_file_location("_sem_runtime", src)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {src}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", type=float, nargs="+", default=[0.40, 0.50, 0.60])
    args = ap.parse_args()

    mod = load_runtime()
    if not (INSTALL / "student.onnx").exists():
        raise SystemExit(f"no student installed at {INSTALL}")

    s = mod.StudentSemantic(INSTALL)
    T = s.temperature
    print(f"installed student : {INSTALL}")
    print(f"fitted temperature: {T}")
    if T == 1.0:
        print("\nT is 1.0 — nothing to verify. Run calibrate.py --apply first.")
        return 0

    sets = [
        ("locked", config.LOCKED_TEST, False),
        ("stress", config.STRESS_TEST, False),
        ("ood", config.OOD_TEST, True),
    ]

    total_flips = 0
    rows_seen = 0
    print()

    for name, path, is_ood in sets:
        if not path.exists():
            continue
        rows = load_rows(path)
        texts = [t for t, _ in rows]
        gold = [g for _, g in rows]

        s.temperature = T
        cal = [s.classify(t) for t in texts]
        s.temperature = 1.0
        raw = [s.classify(t) for t in texts]
        s.temperature = T  # leave it as installed

        flips = sum(1 for a, b in zip(cal, raw) if a[0] != b[0])
        total_flips += flips
        rows_seen += len(rows)

        c_cal = np.array([c for _, c in cal])
        c_raw = np.array([c for _, c in raw])
        i_cal = [i for i, _ in cal]

        print(f"{name.upper():<8} {len(rows):>5} rows")
        print(
            f"   argmax flips T=1 -> T={T:<5}      {flips:>5}"
            f"   {'OK — rank-preserving' if flips == 0 else '<-- BUG: not pure scaling'}"
        )
        print(
            f"   mean confidence                  "
            f"{c_raw.mean():.4f} -> {c_cal.mean():.4f}"
        )

        for g in args.gate:
            crossed = int(((c_raw < g) & (c_cal >= g)).sum())
            dropped = int(((c_raw >= g) & (c_cal < g)).sum())
            if is_ood:
                fb = mod.FALLBACK_INTENT
                r_raw = float(np.mean([(i == fb) or (c < g) for (i, c) in raw]))
                r_cal = float(np.mean([(i == fb) or (c < g) for (i, c) in cal]))
                print(
                    f"   gate {g:.2f}  OOD reject   {r_raw:.4f} -> {r_cal:.4f}"
                    f"  ({r_cal - r_raw:+.4f})   +{crossed} crossed / -{dropped} dropped"
                )
            else:
                a_raw = float(
                    np.mean([(i == y) and (c >= g) for (i, c), y in zip(raw, gold)])
                )
                a_cal = float(
                    np.mean([(i == y) and (c >= g) for i, y, c in zip(i_cal, gold, c_cal)])
                )
                print(
                    f"   gate {g:.2f}  accuracy     {a_raw:.4f} -> {a_cal:.4f}"
                    f"  ({a_cal - a_raw:+.4f})   +{crossed} crossed / -{dropped} dropped"
                )
        print()

    print("=" * 66)
    if total_flips:
        print(f"FAIL: {total_flips} argmax flips across {rows_seen} rows.")
        print("Temperature scaling cannot change argmax. Something else changed.")
        return 1
    print(f"PASS: 0 argmax flips across {rows_seen} rows — pure scaling confirmed.")
    print()
    print("Read the gate rows above as a WARNING, not a result: the gate in")
    print("meta.json was chosen on the T=1 scale. Rows that 'crossed' are rows")
    print("the device now answers that it previously refused. Re-pick the gate:")
    print("    python scripts/select_policy.py --tags semfz_s1 --reveal-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
