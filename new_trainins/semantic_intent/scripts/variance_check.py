"""Measure the noise floor before trusting any improvement.

Two runs of the identical configuration — same data, same seed — produced
macro-F1 0.9129 and 0.8984. sklearn's MLP is seeded, but the BLAS matmuls
underneath it are multi-threaded, so float accumulation order varies between
runs and 400 epochs is long enough for that to land in a different place.

That makes a 1.45-point swing achievable with no change at all, which is larger
than most of the differences we have been reading as results. Every "this batch
helped / hurt" judgement is worthless until the size of that swing is known.

    python scripts/variance_check.py --encoder bge-small-en-v1.5 --seeds 5

Report mean +- std. A change is only real if it clears roughly two standard
deviations; anything smaller is indistinguishable from re-running the same
command twice.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from calibration import ece, fit_temperature, softmax  # noqa: E402
from encoders import get_encoder  # noqa: E402
from evaluate_model import run_all  # noqa: E402
from pipeline import DATA, IntentModel, decision_logits  # noqa: E402
from calibration import SafetyGate, select_operating_point  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

METRICS = [
    ("test_accuracy", lambda r: r["standard_test"]["accuracy"]),
    ("test_macro_f1", lambda r: r["standard_test"]["macro_f1"]),
    ("test_ece", lambda r: r["standard_test"]["ece"]),
    ("contextual", lambda r: r["contextual"]["accuracy"]),
    ("accessories", lambda r: r["accessories"]["accuracy"]),
    ("minimal_pair", lambda r: r["minimal_pairs"]["pair_accuracy"]),
    ("hard_negative", lambda r: r["hard_negatives"]["accuracy"]),
    ("negation", lambda r: r["negation"]["accuracy"]),
    ("stt", lambda r: r["stt"]["accuracy"]),
    # OOD was missing from this list until now, which is the actual reason it
    # never had a noise floor. The story told for months was "the suite is only
    # 45 rows"; the suite being small explains why a floor would be WIDE, not
    # why one was never computed. It was never computed because nothing asked
    # for it. Both are now fixed: 286 rows, and measured every seed.
    ("ood_rejection", lambda r: r["ood"]["rejection_rate"]),
    ("ood_rejection_near", lambda r: r["ood"].get("rejection_near", float("nan"))),
    ("ood_rejection_far", lambda r: r["ood"].get("rejection_far", float("nan"))),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="bge-small-en-v1.5")
    ap.add_argument("--classifier", default="mlp")
    ap.add_argument("--train", default="train_augmented")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--exclude-sources", nargs="*", default=None)
    ap.add_argument("--out", default="reports/variance.json")
    args = ap.parse_args()

    train = pd.read_csv(DATA / f"{args.train}.csv")
    if args.exclude_sources and "source" in train.columns:
        train = train[~train["source"].astype(str).str.startswith(
            tuple(args.exclude_sources))].reset_index(drop=True)
    val = pd.read_csv(DATA / "validation.csv")

    # Encode once. The encoder is frozen, so re-encoding per seed would only
    # burn time — the variance being measured lives in the classifier head.
    enc = get_encoder(args.encoder)
    if getattr(enc, "needs_fit", False):
        enc.fit(train["text"].tolist())
    Xtr = enc.encode(train["text"].tolist())
    Xva = enc.encode(val["text"].tolist())
    print(f"{len(train)} train rows, {args.seeds} seeds, encoder {args.encoder}")

    runs = []
    for seed in range(args.seeds):
        m = IntentModel(enc, args.classifier, seed=seed)
        m.clf.fit(Xtr, train["intent"].tolist())
        m.labels = list(m.clf.classes_)
        lg = decision_logits(m.clf, Xva)
        yva = m.y_index(val["intent"].tolist())
        m.temperature = fit_temperature(lg, yva)
        p = softmax(lg / m.temperature)
        op = select_operating_point(p, yva)
        m.gate = SafetyGate(op["conf_threshold"], op["margin_threshold"],
                            m.labels, temperature=m.temperature)
        r = run_all(m)
        runs.append({name: fn(r) for name, fn in METRICS})
        print(f"  seed {seed}: " + "  ".join(
            f"{k}={v:.4f}" for k, v in list(runs[-1].items())[:3]))

    print(f"\n{'metric':16s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s} "
          f"{'2-sigma':>9s}")
    summary = {}
    for name, _ in METRICS:
        vals = [r[name] for r in runs]
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        summary[name] = dict(mean=statistics.mean(vals), std=sd,
                             min=min(vals), max=max(vals),
                             two_sigma=2 * sd)
        print(f"{name:16s} {statistics.mean(vals):8.4f} {sd:8.4f} "
              f"{min(vals):8.4f} {max(vals):8.4f} {2*sd:9.4f}")

    (ROOT / args.out).write_text(json.dumps(
        dict(n_seeds=args.seeds, train_rows=len(train),
             excluded=args.exclude_sources, runs=runs, summary=summary),
        indent=2))
    print(f"\nA change smaller than the 2-sigma column is not evidence of "
          f"anything.\n-> {args.out}")


if __name__ == "__main__":
    main()
