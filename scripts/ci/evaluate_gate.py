#!/usr/bin/env python3
"""
The ACCURACY GATE (train-and-gate.yml).

Runs the labeled holdout through the engine, computes accuracy / macro-F1 /
wrong-action count, and compares them to the thresholds in
config/gate_thresholds.json. Writes a report card (the approval artifact) and
exits NON-ZERO if any threshold fails — so a model that trains but is not good
enough is blocked, never shipped. On a medical device this hard stop is the point.

Usage:
    python scripts/ci/evaluate_gate.py [--thresholds PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "packages"))
logging.disable(logging.CRITICAL)

FALLBACK = "Default Fallback Intent"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def evaluate(thresholds: dict) -> dict:
    from nlu import NLUEngine  # imported here so path setup above applies
    from sklearn.metrics import f1_score

    holdout = ROOT / thresholds.get("holdout", "data/semantic_holdout_100.csv")
    rows = list(csv.DictReader(open(holdout, encoding="utf-8")))
    engine = NLUEngine(enable_semantic=bool(thresholds.get("enable_semantic", True)))

    y_true, y_pred = [], []
    wrong_action = 0
    for r in rows:
        expected = r.get("expected_intent") or r.get("intent")
        res = engine.handle("gate", r["utterance"])
        engine.reset("gate")
        pred = res.intent or FALLBACK
        y_true.append(expected)
        y_pred.append(pred)
        # A "wrong action" is a confident WRONG *command* (Cmd.*) — the dangerous
        # class on a device (it would actually change hearing-aid state).
        if pred != expected and pred.startswith("Cmd."):
            wrong_action += 1

    n = len(rows)
    accuracy = sum(a == b for a, b in zip(y_true, y_pred)) / n if n else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if n else 0.0

    checks = {
        "accuracy": (accuracy, thresholds["min_accuracy"], accuracy >= thresholds["min_accuracy"]),
        "macro_f1": (macro_f1, thresholds["min_macro_f1"], macro_f1 >= thresholds["min_macro_f1"]),
        "wrong_action": (wrong_action, thresholds["max_wrong_action"],
                         wrong_action <= thresholds["max_wrong_action"]),
    }
    passed = all(ok for *_, ok in checks.values())

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "holdout": str(holdout.relative_to(ROOT)),
        "n": n,
        "enable_semantic": bool(thresholds.get("enable_semantic", True)),
        "metrics": {"accuracy": round(accuracy, 4), "macro_f1": round(macro_f1, 4),
                    "wrong_action": wrong_action},
        "thresholds": {k: thresholds[k] for k in ("min_accuracy", "min_macro_f1", "max_wrong_action")},
        "checks": {k: {"value": v, "bar": bar, "pass": ok} for k, (v, bar, ok) in checks.items()},
        "passed": passed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", default=str(ROOT / "config" / "gate_thresholds.json"))
    ap.add_argument("--out", default=str(ROOT / "dist" / "report_card.json"))
    args = ap.parse_args()

    thresholds = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    report = evaluate(thresholds)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    m = report["metrics"]
    print(f"accuracy={m['accuracy']} macro_f1={m['macro_f1']} wrong_action={m['wrong_action']}")
    for name, c in report["checks"].items():
        print(f"  {'PASS' if c['pass'] else 'FAIL'}  {name}: {c['value']} vs bar {c['bar']}")
    print(f"report card -> {out}")
    if not report["passed"]:
        print("\nGATE FAILED — model blocked, not shippable.")
        return 1
    print("\nGATE PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
