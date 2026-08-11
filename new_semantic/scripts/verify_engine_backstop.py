#!/usr/bin/env python3
"""
Run the REAL NLUEngine over the eval sets with the backstop ON and OFF.

Why this exists: the policy was measured in this folder's own scripts, but it is
IMPLEMENTED in packages/runtime/nlu_engine/engine.py. Those are two independent
pieces of code. A unit test proves the branch is reachable; only this proves the
shipped engine actually reproduces the measured behaviour.

IMPORTANT — what this does and does not measure
-----------------------------------------------
The engine's Stage 3 is `SemanticFallback`, which loads the OLD artifacts
(models/semantic_head.npz + minilm-l6-v2.onnx). It is NOT the 0.166 MB student
trained in this folder — wiring that in is a separate, still-pending change.

So the numbers here answer: "does the backstop help the pipeline as it ships
today?" They will not equal DECISION.md's table, which used the new student.

Usage:
    python scripts/verify_engine_backstop.py
    python scripts/verify_engine_backstop.py --limit 200      # quicker
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts.common import load_rows  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
for p in ("packages/buildtime", "packages/runtime"):
    sys.path.insert(0, str(REPO / p))

MIGRATION = REPO / "datasets" / "label_migration_map.json"


def _to_old_space():
    raw = json.loads(MIGRATION.read_text(encoding="utf-8"))["map"]
    return {new: old for old, new in raw.items() if new}


def run(backstop: float, sets, rev):
    from nlu_engine.engine import NLUEngine

    eng = NLUEngine(language="en")
    eng._stage2_backstop = backstop  # the only thing that changes
    out = {}
    for name, rows, is_ood in sets:
        genai = correct = 0
        for text, gold in rows:
            try:
                r = eng.handle("verify-session", text)
            except TypeError:
                r = eng.handle(text)
            is_fb = r.type == "FALLBACK"
            if is_ood:
                correct += int(is_fb)
            else:
                got = rev.get(r.intent, r.intent)
                correct += int((not is_fb) and got == gold)
            genai += int(is_fb)
        out[name] = {
            "rows": len(rows),
            "score": round(correct / len(rows), 4),
            "fallback_rate": round(genai / len(rows), 4),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--backstop", type=float, default=0.30)
    args = ap.parse_args()

    rev = _to_old_space()
    sets = []
    for name, path, is_ood in (
        ("stress", config.STRESS_TEST, False),
        ("locked", config.LOCKED_TEST, False),
        ("ood", config.OOD_TEST, True),
    ):
        if not path.exists():
            continue
        rows = load_rows(path)
        if args.limit:
            rows = rows[: args.limit]
        sets.append((name, rows, is_ood))

    print(
        "Stage 3 in the engine is the OLD MiniLM head, not the new student.\n"
        "These numbers show the backstop's effect on the pipeline AS SHIPPED.\n"
    )

    off = run(0.0, sets, rev)
    on = run(args.backstop, sets, rev)

    print(f"{'set':<8}{'rows':>7}{'OFF':>10}{'ON':>10}{'delta':>10}" f"{'  fallback OFF->ON':>22}")
    for name, _, is_ood in sets:
        o, n = off[name], on[name]
        label = "OOD reject" if is_ood else "accuracy"
        print(
            f"{name:<8}{o['rows']:>7}{o['score']:>10.4f}{n['score']:>10.4f}"
            f"{n['score'] - o['score']:>+10.4f}"
            f"{o['fallback_rate']:>12.4f}->{n['fallback_rate']:.4f}   ({label})"
        )

    report = config.REPORTS / "engine_backstop_verification.json"
    report.write_text(
        json.dumps(
            {
                "backstop": args.backstop,
                "stage3_model": "legacy MiniLM semantic head",
                "off": off,
                "on": on,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {report}")

    reg = [n for n, _, ood in sets if not ood and on[n]["score"] < off[n]["score"] - 0.005]
    if reg:
        print(
            f"\nREGRESSION on {reg} — the engine is not reproducing the "
            f"measured behaviour; do not enable the backstop in the pack."
        )
        return 1
    print("\nno in-scope regression")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
