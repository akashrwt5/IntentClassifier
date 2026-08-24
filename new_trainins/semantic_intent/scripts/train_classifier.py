"""Phase 14 + 18 — train, calibrate, gate and freeze one candidate.

    python scripts/train_classifier.py --encoder tfidf-svd --classifier mlp \
        --train train_augmented --out models/final

Calibration temperature and both gate thresholds come from the validation
split only. The test split is touched exactly once, by evaluate/report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from calibration import (SafetyGate, coverage_precision_curve, ece,  # noqa: E402
                         fit_temperature, margin_of, select_operating_point, softmax)
from encoders import get_encoder, measure_latency  # noqa: E402
from evaluate_model import headline, run_all  # noqa: E402
from pipeline import DATA, FALLBACK, IntentModel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="student-h256-l4")
    ap.add_argument("--classifier", default="mlp")
    ap.add_argument("--train", default="train_augmented")
    ap.add_argument("--out", default="models/final_student_256")
    ap.add_argument("--target-precision", type=float, default=0.97)
    ap.add_argument("--min-coverage", type=float, default=0.50)
    ap.add_argument("--max-fallback-leak", type=float, default=0.07,
                    help="max share of validation rows labelled as the "
                         "reject class that the gate may still accept")
    ap.add_argument("--high-risk-precision", type=float, default=0.99,
                    help="precision target for intents whose mistakes "
                         "the user may not notice or be able to undo")
    ap.add_argument("--ood-method", default="mahalanobis",
                    choices=["mahalanobis", "energy", "none"])
    ap.add_argument("--exclude-sources", nargs="*", default=None,
                    help="drop augmentation blocks by source prefix, e.g. "
                         "--exclude-sources F12 — for isolating whether a "
                         "batch helped or hurt, instead of guessing")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train = pd.read_csv(DATA / f"{args.train}.csv")
    if args.exclude_sources and "source" in train.columns:
        before = len(train)
        drop = train["source"].astype(str).str.startswith(
            tuple(args.exclude_sources))
        train = train[~drop].reset_index(drop=True)
        print(f"excluded {', '.join(args.exclude_sources)}: "
              f"{before} -> {len(train)} rows")
    val = pd.read_csv(DATA / "validation.csv")

    model = IntentModel(get_encoder(args.encoder), args.classifier, seed=args.seed)
    model.fit(train["text"], train["intent"])
    print(f"fitted {args.encoder} + {args.classifier} on {len(train)} rows "
          f"in {model.fit_seconds}s; {len(model.labels)} classes")

    logits_va = model.logits(val["text"].tolist())
    yva = model.y_index(val["intent"].tolist())
    ece_raw = ece(softmax(logits_va).max(1), softmax(logits_va).argmax(1) == yva)
    T = fit_temperature(logits_va, yva)
    model.temperature = T
    p_cal = softmax(logits_va / T)
    ece_cal = ece(p_cal.max(1), p_cal.argmax(1) == yva)
    print(f"temperature={T:.4f}  ECE {ece_raw:.4f} -> {ece_cal:.4f}")

    # --- OOD scorer, fitted on TRAIN embeddings only -----------------------
    from calibration import (fit_per_risk_thresholds, ood_ablation,
                             select_operating_point_3d)
    from ood_score import OODScorer

    model.ood = OODScorer(args.ood_method).fit(
        model.embed(train["text"].tolist()), train["intent"].tolist(),
        logits=model.logits(train["text"].tolist())
        if args.ood_method == "energy" else None)
    ood_va = model.ood_scores(texts=val["text"].tolist())
    reject_idx = (model.labels.index(FALLBACK) if FALLBACK in model.labels else -1)

    if ood_va is None:
        op = select_operating_point(p_cal, yva, args.target_precision,
                                    args.min_coverage)
        op["ood_threshold"] = float("inf")
        op["fallback_leak"] = float("nan")
        pct = None
    else:
        # Candidate cut-offs are percentiles of the TRAINING score distribution:
        # "reject anything less like the training data than the most unusual
        # N% of it". inf keeps the off-setting in the search so the OOD signal
        # has to earn its place rather than being assumed useful.
        grid = [model.ood.threshold_at(q) for q in (90, 95, 97.5, 99, 99.5)]
        grid = sorted(set(g for g in grid if np.isfinite(g))) + [float("inf")]
        # Diagnostic first: does the score separate ANYTHING? A gate built on
        # a score with AUROC near 0.5 is theatre — it costs coverage and buys
        # nothing. Reported before the ablation so the ablation is read in
        # context. Held-out suites are used for REPORTING only here; no
        # threshold is fitted on them.
        from sklearn.metrics import roc_auc_score
        s_in = ood_va[val["intent"] != FALLBACK]
        diag = {}
        for name, path in (("val fallback", None),
                           ("held-out OOD", "ood_test.csv"),
                           ("STT noise", "stt_test.csv")):
            other = (ood_va[val["intent"] == FALLBACK] if path is None
                     else model.ood_scores(
                         texts=pd.read_csv(DATA / path)["text"].tolist()[:600]))
            if other is None or len(other) == 0:
                continue
            yy = np.r_[np.ones(len(other)), np.zeros(len(s_in))]
            diag[name] = float(roc_auc_score(yy, np.r_[other, s_in]))
        print("\nOOD score separability (AUROC vs in-domain; 0.5 = useless):")
        for k, v in diag.items():
            verdict = ("useful" if v >= 0.80 else
                       "weak" if v >= 0.65 else "no signal")
            print(f"  {k:14s} {v:.3f}   {verdict}")

        print("\nOOD ablation on validation (what each cut-off costs and buys):")
        print(f"{'ood_thr':>10s} {'conf':>6s} {'coverage':>9s} "
              f"{'precision':>10s} {'fallback_leak':>14s}")
        for r in ood_ablation(p_cal, yva, ood_va, grid, reject_idx,
                              args.target_precision):
            print(f"{str(r['ood_threshold']):>10s} {r['conf']:>6.3f} "
                  f"{r['coverage']:>9.4f} {r['precision']:>10.4f} "
                  f"{r['fallback_leak']:>14.4f}")
        print()
        op = select_operating_point_3d(p_cal, yva, ood_va, grid, reject_idx,
                                       args.target_precision, args.min_coverage,
                                       max_fallback_leak=args.max_fallback_leak)
        pct = next((q for q in (90, 95, 97.5, 99, 99.5)
                    if abs(model.ood.threshold_at(q) - op["ood_threshold"]) < 1e-9),
                   None)

    # --- per-risk confidence thresholds -----------------------------------
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "intents.yaml").read_text())
    risk_of = {k: v.get("risk", "normal") for k, v in cfg["intents"].items()}
    conf_by_risk = fit_per_risk_thresholds(
        p_cal, yva, risk_of, model.labels, op["conf_threshold"],
        targets={"high": args.high_risk_precision,
                 "normal": args.target_precision})

    model.gate = SafetyGate(op["conf_threshold"], op["margin_threshold"],
                            model.labels, temperature=T,
                            ood_threshold=op["ood_threshold"],
                            ood_percentile=pct,
                            risk_of=risk_of, conf_by_risk=conf_by_risk)
    high = sorted(cfg["risk_tiers"]["high"])
    print(f"per-risk thresholds: normal>={conf_by_risk['normal']:.3f}  "
          f"high>={conf_by_risk['high']:.3f}")
    print(f"      high-risk intents ({len(high)}): {', '.join(high)}")
    print(f"gate: conf>={op['conf_threshold']} margin>={op['margin_threshold']} "
          f"ood<={op['ood_threshold']:.3f}"
          f"{f' (train p{pct})' if pct else ' (OOD signal not selected)'}")
    print(f"      val precision={op['precision']:.4f} coverage={op['coverage']:.4f} "
          f"fallback_leak={op.get('fallback_leak', float('nan')):.4f} "
          f"target_met={op['target_met']} "
          f"<- measured at conf>={op['conf_threshold']}, NOT the shipped gate")

    # The line above reports the operating point's own numbers, which are
    # computed at op["conf_threshold"]. The gate does not ship that threshold:
    # fit_per_risk_thresholds then raises `normal` on top of it, and `normal`
    # covers 52 of the 57 intents. Reporting the pre-per-risk figure as "val
    # coverage" overstates what the gate will do — in one run it printed 0.6827
    # while the shipped gate delivered 0.431 on test, and that discrepancy was
    # mistaken for a validation/test generalisation gap for several rounds.
    # It was not a gap. It was two different thresholds.
    _pred = p_cal.argmax(1)
    _conf = p_cal.max(1)
    _tier = np.array([risk_of.get(model.labels[i], "normal") for i in _pred])
    _need = np.array([conf_by_risk.get(t, op["conf_threshold"]) for t in _tier])
    _acc = ((_pred != model.labels.index(FALLBACK))
            & (_conf >= _need)
            & (margin_of(p_cal) >= op["margin_threshold"]))
    _correct = _pred == yva
    _actionable = yva != model.labels.index(FALLBACK)
    print(f"      SHIPPED GATE on validation: "
          f"precision={_correct[_acc].mean():.4f} "
          f"coverage={_acc[_actionable].mean():.4f} "
          f"(normal>={conf_by_risk['normal']:.3f})")
    if _acc[_actionable].mean() < op["coverage"] - 0.05:
        print(f"      NOTE: the per-risk step cost "
              f"{op['coverage'] - _acc[_actionable].mean():.4f} of coverage. "
              f"It applies a SECOND precision target on top of the operating "
              f"point's; lower --target-precision or raise --max-fallback-leak "
              f"if that trade is wrong for the product.")

    out = ROOT / args.out
    model.save(out)

    curve = coverage_precision_curve(p_cal, yva, op["margin_threshold"])
    (ROOT / "reports" / "operating_point.json").write_text(json.dumps(
        dict(temperature=T, ece_raw=ece_raw, ece_calibrated=ece_cal,
             operating_point=op, coverage_precision_curve=curve), indent=2))

    res = run_all(model)
    (ROOT / "reports" / "final_suites.json").write_text(
        json.dumps(res, indent=2, default=float))
    print(headline(res))
    lat = measure_latency(model.encoder, val["text"].tolist())
    print(f"encoder latency p50={lat['p50_ms']}ms p90={lat['p90_ms']}ms")
    print(f"saved -> {out}")

    print("\n--- Auto-exporting ONNX & Bundling ---")
    try:
        import zipfile
        from export_onnx import export_sklearn, export_transformer, quantize, write_runtime_config

        out_onnx = out / "onnx"
        out_onnx.mkdir(parents=True, exist_ok=True)
        is_transformer = hasattr(model.encoder, "tok")
        if is_transformer:
            p = export_transformer(model, out_onnx, 64)
        else:
            p = export_sklearn(model, out_onnx)
        write_runtime_config(model, out_onnx, 64)
        q = quantize(p, keep_embeddings_fp32=False)
        print(f"Auto-exported to {out_onnx}")

        # Bundle for release
        bundle_path = out / "release_bundle.zip"
        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(q, q.name)
            z.write(out_onnx / "runtime_config.json", "runtime_config.json")
            if is_transformer:
                tok_dir = out_onnx / "tokenizer"
                for f in tok_dir.iterdir():
                    if f.is_file():
                        z.write(f, f"tokenizer/{f.name}")
        print(f"Bundled Android/iOS release package -> {bundle_path}")
    except Exception as e:
        print(f"Auto-export skipped: {e}")


if __name__ == "__main__":
    main()
