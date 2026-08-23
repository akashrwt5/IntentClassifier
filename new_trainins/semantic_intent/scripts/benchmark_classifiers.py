"""Phases 11-13 — encoder x classifier benchmark on identical splits.

Selection rule (fixed in advance so it cannot be rationalized afterwards):
    1. validation macro-F1 is primary
    2. validation ECE after temperature scaling breaks ties within 0.005 macro-F1
    3. size and latency break remaining ties
The challenge suites are REPORTED for every candidate but are NOT part of the
selection rule — they are held-out tests, and choosing on them would be the
"tune on the test set" mistake the plan forbids.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from calibration import (
    SafetyGate,
    ece,
    fit_temperature,  # noqa: E402
    select_operating_point,
    softmax,
)
from encoders import discover_local_encoders, get_encoder, measure_latency  # noqa: E402
from evaluate_model import headline, run_all  # noqa: E402
from pipeline import DATA, IntentModel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CLASSIFIERS = ["logreg", "logreg_balanced", "linsvm", "linsvm_balanced", "mlp"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="*", default=None)
    ap.add_argument("--classifiers", nargs="*", default=CLASSIFIERS)
    ap.add_argument("--target-precision", type=float, default=0.97)
    ap.add_argument("--min-coverage", type=float, default=0.50)
    ap.add_argument("--train", default="train")
    ap.add_argument("--out", default="reports/benchmark.json")
    args = ap.parse_args()

    encoders = args.encoders or (["tfidf-svd"] + list(discover_local_encoders()))
    train = pd.read_csv(DATA / f"{args.train}.csv")
    val = pd.read_csv(DATA / "validation.csv")
    print(f"train={len(train)} val={len(val)} encoders={encoders}")

    results = []
    for enc_spec in encoders:
        t0 = time.perf_counter()
        try:
            base_encoder = get_encoder(enc_spec)
        except Exception as e:  # noqa: BLE001
            print(f"!! skipping encoder {enc_spec}: {e}")
            continue
        # fit the encoder once, reuse embeddings across classifiers
        if getattr(base_encoder, "needs_fit", False):
            base_encoder.fit(train["text"].tolist())
        Xtr = base_encoder.encode(train["text"].tolist())
        Xva = base_encoder.encode(val["text"].tolist())
        enc_meta = base_encoder.meta()
        lat = measure_latency(base_encoder, val["text"].tolist())
        print(
            f"\n== {enc_spec}  dim={enc_meta['dim']} "
            f"encode_all={time.perf_counter()-t0:.1f}s  p50={lat['p50_ms']}ms"
        )

        for clf_spec in args.classifiers:
            m = IntentModel(base_encoder, clf_spec)
            t1 = time.perf_counter()
            m.clf.fit(Xtr, train["intent"].tolist())
            m.labels = list(m.clf.classes_)
            fit_s = time.perf_counter() - t1

            from pipeline import decision_logits

            logits_va = decision_logits(m.clf, Xva)
            yva = m.y_index(val["intent"].tolist())

            raw_conf = softmax(logits_va).max(1)
            correct = softmax(logits_va).argmax(1) == yva
            ece_raw = ece(raw_conf, correct)

            T = fit_temperature(logits_va, yva)
            m.temperature = T
            p_cal = softmax(logits_va / T)
            ece_cal = ece(p_cal.max(1), p_cal.argmax(1) == yva)

            op = select_operating_point(
                p_cal, yva, target_precision=args.target_precision, min_coverage=args.min_coverage
            )
            m.gate = SafetyGate(
                op["conf_threshold"], op["margin_threshold"], m.labels, temperature=T
            )

            from sklearn.metrics import accuracy_score, f1_score

            pred = np.array(m.labels)[p_cal.argmax(1)]
            val_acc = accuracy_score(val["intent"], pred)
            val_f1 = f1_score(val["intent"], pred, average="macro", zero_division=0)

            suites = run_all(m)
            row = dict(
                encoder=enc_spec,
                classifier=clf_spec,
                val_accuracy=round(float(val_acc), 4),
                val_macro_f1=round(float(val_f1), 4),
                ece_raw=round(ece_raw, 4),
                ece_calibrated=round(ece_cal, 4),
                temperature=round(T, 4),
                conf_threshold=op["conf_threshold"],
                margin_threshold=op["margin_threshold"],
                val_coverage=round(op["coverage"], 4),
                val_accepted_precision=round(op["precision"], 4),
                target_met=op["target_met"],
                fit_seconds=round(fit_s, 2),
                encoder_dim=enc_meta["dim"],
                encoder_fp32_mb=enc_meta.get("fp32_mb"),
                latency_p50_ms=lat["p50_ms"],
                latency_p90_ms=lat["p90_ms"],
                test_accuracy=suites["standard_test"]["accuracy"],
                test_macro_f1=suites["standard_test"]["macro_f1"],
                test_ece=suites["standard_test"]["ece"],
                contextual=suites["contextual"]["accuracy"],
                minimal_pair_item=suites["minimal_pairs"]["accuracy"],
                minimal_pair_pair=suites["minimal_pairs"]["pair_accuracy"],
                hard_negative=suites["hard_negatives"]["accuracy"],
                negation=suites["negation"]["accuracy"],
                stt=suites["stt"]["accuracy"],
                ood_rejection=suites["ood"]["rejection_rate"],
                ood_false_acceptance=suites["ood"]["false_acceptance"],
                gated_coverage=suites["gated_test"]["coverage"],
                gated_accepted_precision=suites["gated_test"]["accepted_precision"],
            )
            results.append(row)
            print(
                f"  {clf_spec:17s} valF1={val_f1:.4f} T={T:5.2f} "
                f"ECE {ece_raw:.4f}->{ece_cal:.4f} | {headline(suites)}"
            )

    df = pd.DataFrame(results)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    df.to_csv(out.with_suffix(".csv"), index=False)

    # selection rule
    if len(df):
        best_f1 = df["val_macro_f1"].max()
        pool = df[df["val_macro_f1"] >= best_f1 - 0.005]
        pool = pool.sort_values(["ece_calibrated", "encoder_fp32_mb", "latency_p50_ms"])
        win = pool.iloc[0]
        print(
            f"\nWINNER: {win['encoder']} + {win['classifier']} "
            f"(valF1={win['val_macro_f1']:.4f}, ECE={win['ece_calibrated']:.4f})"
        )
        (ROOT / "reports" / "selected.json").write_text(
            json.dumps(
                dict(
                    encoder=win["encoder"],
                    classifier=win["classifier"],
                    rule="max val_macro_f1, tie<=0.005 broken by "
                    "calibrated ECE then size then latency",
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
