"""
Evaluate the *exported* model — the artifact that actually ships.

`train.py` reports numbers from the Python objects it just fitted. This module
loads `semantic_intent.onnx` through the same runtime class the app uses, so a
regression in export, metadata, thresholds or tokenisation shows up here and
not on a device.

    python -m semantic_intent.evaluate --data datasets/balanced_intents_final.xlsx

Reports: held-out accuracy, hard paraphrases, antonym pairs, out-of-scope
rejection, confidence calibration, and per-utterance latency.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .eval_sets import ANTONYM_PAIRS, HARD_PARAPHRASES, KNOWN_GAPS, OUT_OF_SCOPE
from .predict import DEFAULT_MODEL, DEFAULT_VOCAB, SemanticIntentClassifier

BASE_DIR = Path(__file__).resolve().parents[1]


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d:.1%})" if d else "n/a"


def evaluate(clf: SemanticIntentClassifier, data_path: Path | None, seed: int = 0) -> dict:
    out: dict = {}

    # ------------------------------------------------ held-out corpus
    if data_path:
        from . import data as data_mod

        df = data_mod.grouped_split(data_mod.load(data_path), seed=seed)
        test = df[df.split == "test"]
        preds = clf.predict_batch(test.text.tolist())
        correct = np.array([p.intent == g for p, g in zip(preds, test.intent)])
        conf = np.array([p.confidence for p in preds])
        acc = float(correct.mean())
        out["test_accuracy"] = acc
        print(f"\nHeld-out (grouped) accuracy : {acc:.4f}  n={len(test)}")

        # Where is it wrong, and does it know?
        print(
            f"  mean confidence  correct={conf[correct].mean():.3f}  "
            f"wrong={conf[~correct].mean() if (~correct).any() else float('nan'):.3f}"
        )
        if (~correct).any():
            from collections import Counter

            pairs = Counter(
                (g, p.intent) for p, g, ok in zip(preds, test.intent, correct) if not ok
            )
            print("  top confusions:")
            for (gold, got), n in pairs.most_common(5):
                print(f"    {n:3d}x  {gold}  ->  {got}")
            out["top_confusions"] = [
                {"gold": g, "predicted": p, "count": n} for (g, p), n in pairs.most_common(5)
            ]

    # ------------------------------------------------ hard paraphrases
    preds = clf.predict_batch([t for t, _ in HARD_PARAPHRASES])
    hits = [p.intent == g for p, (_, g) in zip(preds, HARD_PARAPHRASES)]
    print(f"\nHard paraphrases (unseen wording): {_pct(sum(hits), len(hits))}")
    out["hard_accuracy"] = sum(hits) / len(hits)
    for (text, gold), p, ok in zip(HARD_PARAPHRASES, preds, hits):
        if not ok:
            print(
                f"  miss  {text[:44]:46s} gold={gold:24s} got={p.intent} " f"({p.confidence:.2f})"
            )

    # ------------------------------------------------ antonym pairs
    print("\nAntonym pairs (same words, opposite intent):")
    ok_pairs = 0
    for a, ga, b, gb in ANTONYM_PAIRS:
        pa, pb = clf.predict(a), clf.predict(b)
        good = pa.intent == ga and pb.intent == gb
        ok_pairs += good
        tag = "OK  " if good else "FAIL"
        print(f"  {tag} {a[:40]:42s} -> {pa.intent:24s} {pa.confidence:.2f}")
        print(f"       {b[:40]:42s} -> {pb.intent:24s} {pb.confidence:.2f}")
    out["antonym_pairs"] = f"{ok_pairs}/{len(ANTONYM_PAIRS)}"

    # ------------------------------------------------ out of scope
    oos = [t for t in OUT_OF_SCOPE if t.strip()]
    preds = clf.predict_batch(oos)
    rejected = [not p.accepted for p in preds]
    print(f"\nOut-of-scope rejection: {_pct(sum(rejected), len(rejected))}")
    out["oos_rejection"] = sum(rejected) / len(rejected)
    for text, p, rej in zip(oos, preds, rejected):
        if not rej:
            print(
                f"  leaked  {text[:44]:46s} -> {p.intent} "
                f"(conf={p.confidence:.2f} ood={p.ood_score:.2f})"
            )

    print("\nKnown capability gaps (rejection is the correct behaviour):")
    for gap, examples in KNOWN_GAPS.items():
        got = [clf.predict(e) for e in examples]
        n_rej = sum(not g.accepted for g in got)
        print(f"  {gap:38s} rejected {n_rej}/{len(got)}")

    # ------------------------------------------------ latency
    probe = [t for t, _ in HARD_PARAPHRASES][:20]
    for _ in range(3):
        clf.predict_batch(probe)  # warm
    timings = []
    for text in probe * 5:
        t0 = time.perf_counter()
        clf.predict(text)
        timings.append((time.perf_counter() - t0) * 1000)
    timings = np.array(timings)
    print(
        f"\nLatency per utterance: mean={timings.mean():.1f}ms  "
        f"p50={np.percentile(timings, 50):.1f}ms  p95={np.percentile(timings, 95):.1f}ms"
    )
    out["latency_ms"] = {
        "mean": float(timings.mean()),
        "p50": float(np.percentile(timings, 50)),
        "p95": float(np.percentile(timings, 95)),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    ap.add_argument("--data", type=Path, help="corpus for the held-out split")
    ap.add_argument("--seed", type=int, default=0, help="must match training seed")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    clf = SemanticIntentClassifier(args.model, args.vocab)
    size_mb = args.model.stat().st_size / 1e6
    print(f"model: {args.model.name}  {size_mb:.1f} MB  " f"{len(clf.labels)} intents")
    print(f"gates: conf>={clf.conf_threshold:.3f}  ood>={clf.ood_threshold:.3f}")

    results = evaluate(clf, args.data, args.seed)
    results["model_mb"] = size_mb

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
