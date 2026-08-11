#!/usr/bin/env python3
"""
Compare models on the one thing the other eval sets do not measure:
does the model understand a word it has never seen?

Every row of the OOV probe contains a word outside the training vocabulary but
is unambiguous to a person ("elevate the volume"). The current shipped student
scores 9.1% here — it treats `elevate` and `heighten` as the same token,
`[UNK]`, and rejects both with identical confidence.

Also reported: accuracy on the ordinary eval sets, because a model that learns
general word meanings could plausibly LOSE in-domain sharpness. Both numbers
have to be read together.

Usage:
    python scripts/eval_oov.py --tags unkaug_s1 sem_s1 semfz_s1
    python scripts/eval_oov.py --tags sem_s1 --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts.common import load_rows  # noqa: E402
from scripts.try_student import Student  # noqa: E402


def score(st: Student, rows, gate: float, ood=False):
    correct = rejected = 0
    detail = []
    for text, gold in rows:
        p, _ = st.predict(text)
        i = int(p.argmax())
        pred, conf = st.labels[i], float(p[i])
        is_rej = pred == config.FALLBACK_INTENT or conf < gate
        if ood:
            correct += int(is_rej)
        else:
            correct += int(pred == gold and not is_rej)
        rejected += int(is_rej)
        detail.append((text, gold, pred, conf, is_rej))
    n = max(len(rows), 1)
    return correct / n, rejected / n, detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--gate", type=float, default=0.40)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not config.OOV_TEST.exists():
        raise SystemExit("run scripts/build_oov_testset.py first")

    sets = [
        ("oov", config.OOV_TEST, False),
        ("stress", config.STRESS_TEST, False),
        ("locked", config.LOCKED_TEST, False),
        ("ood", config.OOD_TEST, True),
    ]

    print(f"gate {args.gate}\n")
    header = f"{'tag':<14}{'backend':<9}{'vocab':>7}{'MB':>7}" + "".join(
        f"{n:>10}" for n, _, _ in sets
    )
    print(header)
    print("-" * len(header))

    out = {}
    for tag in args.tags:
        try:
            st = Student(tag)
        except SystemExit as e:
            print(f"{tag:<14} skipped: {e}")
            continue
        except ImportError:
            print(
                f"{tag:<14} skipped: only a .pt checkpoint exists and torch is "
                f"not installed here. Export it first:\n"
                f"               python scripts/export_onnx.py --tag {tag} "
                f"--threshold 0.40 --skip-int8"
            )
            continue
        row, det = {}, {}
        for name, path, ood in sets:
            if not path.exists():
                continue
            rows = load_rows(path)
            if not ood:
                rows = [(t, g) for t, g in rows if g in st.labels]
            acc, rej, d = score(st, rows, args.gate, ood)
            row[name] = round(acc, 4)
            det[name] = d
        out[tag] = {
            "vocab": len(st.vocab),
            "mb": round(st.size_mb, 3),
            "backend": st.backend,
            **row,
        }
        print(
            f"{tag:<14}{st.backend:<9}{len(st.vocab):>7}{st.size_mb:>7.2f}"
            + "".join(f"{row.get(n, float('nan')):>10.4f}" for n, _, _ in sets)
        )

        if args.verbose:
            print(f"\n  --- {tag} on the OOV probe ---")
            for text, gold, pred, conf, rej in det.get("oov", []):
                mark = "OK " if (pred == gold and not rej) else "XX "
                print(
                    f"   {mark} {text:<34} -> {pred:<24} {conf:.3f}"
                    f"{'  [rejected]' if rej else ''}"
                )
            print()

    p = config.REPORTS / "oov_comparison.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")
    print("\noov = authored probe, diagnostic only. Read it WITH stress/locked:")
    print("a model that gains on oov but loses on stress has traded in-domain")
    print("sharpness for general vocabulary, which may not be a good trade.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
