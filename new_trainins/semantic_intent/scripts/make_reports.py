"""Render reports/benchmark.md, calibration.md and final_evaluation.md from
the JSON artifacts produced by the benchmark / training / parity scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import df_to_markdown  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"

BASELINE = dict(
    name="INT8 semantic student (existing)",
    intents=11,
    size_mb=0.236,
    unseen_semantic_acc=0.9429,
    unseen_semantic_f1=0.9428,
    contextual_acc=0.9062,
    contextual_f1=0.9078,
    ood_rejection=0.3438,
)


def _load(name):
    p = REP / name
    return json.loads(p.read_text()) if p.exists() else None


def benchmark_md() -> str:
    rows = _load("benchmark.json") or []
    aug = _load("benchmark_aug.json") or []
    L = [
        "# Encoder x Classifier Benchmark\n",
        "Selection rule, fixed before any numbers were seen:\n",
        "1. validation macro-F1 is primary\n"
        "2. calibrated ECE breaks ties within 0.005 macro-F1\n"
        "3. size then latency break what remains\n",
        "The challenge suites below are **reported, not used for selection**. "
        "Choosing a model on them would be the 'tune on the test set' mistake "
        "the plan rules out in Section 33.\n",
    ]

    def table(rs, title):
        if not rs:
            return []
        df = pd.DataFrame(rs)
        cols = [
            "encoder",
            "classifier",
            "val_macro_f1",
            "test_accuracy",
            "test_macro_f1",
            "ece_raw",
            "ece_calibrated",
            "temperature",
            "contextual",
            "minimal_pair_pair",
            "hard_negative",
            "negation",
            "stt",
            "ood_rejection",
            "gated_coverage",
            "gated_accepted_precision",
            "latency_p50_ms",
        ]
        cols = [c for c in cols if c in df.columns]
        return [f"\n## {title}\n", df_to_markdown(df[cols]), ""]

    L += table(rows, "Trained on the original split (`train.csv`)")
    L += table(aug, "Trained with targeted augmentation (`train_augmented.csv`)")

    if rows and aug:
        a = pd.DataFrame(rows).set_index(["encoder", "classifier"])
        b = pd.DataFrame(aug).set_index(["encoder", "classifier"])
        common = a.index.intersection(b.index)
        metrics = [
            "negation",
            "stt",
            "ood_rejection",
            "minimal_pair_pair",
            "hard_negative",
            "contextual",
            "test_macro_f1",
        ]
        delta = (b.loc[common, metrics] - a.loc[common, metrics]).round(4)
        L += [
            "\n## What the targeted augmentation changed\n",
            "Delta = augmented - original, on identical held-out suites.\n",
            df_to_markdown(delta, index=True),
            "",
            "The augmentation was built to fix negation, STT robustness and "
            "near-OOD, and those are the columns that move. Minimal pairs and "
            "hard negatives barely move, which is the expected result for a "
            "bag-of-n-grams encoder: direction reversal under paraphrase is a "
            "semantic problem, not a lexical one, so it is the encoder that "
            "has to change.\n",
        ]
    return "\n".join(L)


def calibration_md() -> str:
    op = _load("operating_point.json")
    suites = _load("final_suites.json")
    if not op or not suites:
        return "# Confidence Calibration\n\n_Not yet generated._\n"
    L = [
        "# Confidence Calibration and the Operating Point\n",
        "## What 'calibrated' means here\n",
        "> 90% confidence must not mean 'the model feels 90% sure'. It must "
        "mean that among predictions in the 90% band, roughly 90% are correct.\n",
        "The table below is the direct test of that claim on held-out data.\n",
        f"\n## Temperature scaling\n",
        f"- fitted temperature **T = {op['temperature']:.4f}** (validation split only)",
        f"- validation ECE **{op['ece_raw']:.4f} -> {op['ece_calibrated']:.4f}**",
        "",
    ]
    if op["temperature"] > 1:
        L.append("T > 1 means the raw scores were **over-confident** and had to " "be softened.\n")
    else:
        L.append(
            "T < 1 means the raw scores were **under-confident** and had to " "be sharpened.\n"
        )

    rel = suites["standard_test"]["reliability"]
    L += [
        "## Reliability diagram (held-out test)\n",
        "| confidence bin | n | mean confidence | actual accuracy | gap |",
        "|---|---|---|---|---|",
    ]
    for r in rel:
        if not r["n"]:
            continue
        L.append(
            f"| {r['bin_lo']:.1f}–{r['bin_hi']:.1f} | {r['n']} | "
            f"{r['avg_conf']:.3f} | {r['accuracy']:.3f} | {r['gap']:+.3f} |"
        )
    L += [
        "",
        f"Test ECE **{suites['standard_test']['ece']:.4f}**, "
        f"MCE {suites['standard_test']['mce']:.4f}, "
        f"Brier {suites['standard_test']['brier']:.4f}.\n",
    ]

    o = op["operating_point"]
    L += [
        "## Operating point\n",
        "Chosen on validation for precision first: on a hearing aid, "
        "executing the wrong command is worse than asking the user to repeat.\n",
        f"- confidence threshold **{o['conf_threshold']}**",
        f"- top1-top2 margin threshold **{o['margin_threshold']}**",
        f"- validation precision among accepted **{o['precision']:.4f}** "
        f"at coverage **{o['coverage']:.4f}**",
        f"- target of {o['target_precision']:.0%} precision met: **{o['target_met']}**",
        "",
        "### Coverage / precision trade-off\n",
        "| threshold | coverage | precision among accepted | n |",
        "|---|---|---|---|",
    ]
    for c in op["coverage_precision_curve"]:
        L.append(
            f"| {c['threshold']:.2f} | {c['coverage']:.3f} | " f"{c['precision']:.4f} | {c['n']} |"
        )
    L += [
        "",
        "## Why margin is checked as well as confidence\n",
        "```text\nincrease = 0.51\ndecrease = 0.48\nmargin   = 0.03\n```\n",
        "Top-1 confidence alone would call that a decision. The margin says "
        "it is a coin flip between two opposite commands, which is exactly "
        "the case where a hearing aid should ask again.\n",
    ]
    return "\n".join(L)


def final_md() -> str:
    suites = _load("final_suites.json")
    parity = _load("parity.json")
    sel = _load("selected.json")
    if not suites:
        return "# Final Evaluation\n\n_Not yet generated._\n"
    s, g = suites["standard_test"], suites.get("gated_test", {})
    ood = suites.get("ood", {})
    L = ["# Final Evaluation\n"]
    if sel:
        L.append(f"Model: **{sel['encoder']} + {sel['classifier']}**\n")
    L += [
        "## Evaluation matrix\n",
        "| suite | purpose | result |",
        "|---|---|---|",
        f"| Standard test | general classification | acc **{s['accuracy']:.4f}**, macro-F1 **{s['macro_f1']:.4f}** |",
        f"| Contextual | long/natural requests | acc {suites['contextual']['accuracy']:.4f} |",
        f"| Minimal pairs | opposite-intent separation | item {suites['minimal_pairs']['accuracy']:.4f}, both-sides {suites['minimal_pairs']['pair_accuracy']:.4f} |",
        f"| Hard negatives | shortcut resistance | acc {suites['hard_negatives']['accuracy']:.4f} |",
        f"| Negation | scope handling (P2/P3) | acc {suites['negation']['accuracy']:.4f} |",
        f"| STT | recognition noise | acc {suites['stt']['accuracy']:.4f} |",
        f"| OOD | unknown rejection | rejection {ood.get('rejection_rate', float('nan')):.4f}, false acceptance {ood.get('false_acceptance', float('nan')):.4f} |",
        f"| Calibration | trustworthy confidence | ECE {s['ece']:.4f} |",
        "",
    ]
    if g:
        L += [
            "## The production statement\n",
            f"> At the selected operating threshold, accepted predictions "
            f"achieve **{g['accepted_precision']:.2%} precision** on the "
            f"held-out test set, with **{g['coverage']:.2%} coverage**.\n",
            f"- false execution rate (wrong **and** accepted): " f"**{g['false_execution']:.4f}**",
            f"- false rejection rate (right but gated out): " f"{g['false_rejection']:.4f}",
            "",
        ]
    if "per_group_accuracy" in suites["negation"]:
        L += ["## Negation policy breakdown\n", "| policy | accuracy |", "|---|---|"]
        for k, v in suites["negation"]["per_group_accuracy"].items():
            L.append(f"| {k} | {v:.4f} |")
        L.append("")
    if "per_axis_accuracy" in suites["minimal_pairs"]:
        L += ["## Minimal pairs by axis\n", "| axis | item accuracy |", "|---|---|"]
        for k, v in suites["minimal_pairs"]["per_axis_accuracy"].items():
            L.append(f"| {k} | {v:.4f} |")
        L.append("")
    if parity:
        L += [
            "## Python vs ONNX parity\n",
            "| build | max abs delta | top-1 agreement | gate agreement | size MB | within tolerance |",
            "|---|---|---|---|---|---|",
        ]
        for tag, r in parity.items():
            L.append(
                f"| {tag} | {r['max_abs_delta']:.3e} | {r['top1_agreement']:.5f} | "
                f"{r['gate_agreement']:.5f} | {r['size_mb']} | {r['within_tolerance']} |"
            )
        L.append("")
    L += [
        "## Comparison against the existing baseline\n",
        "| metric | baseline (INT8 semantic student) | this run |",
        "|---|---|---|",
        f"| intents | {BASELINE['intents']} | 57 |",
        f"| model size | {BASELINE['size_mb']} MB | see parity table |",
        f"| contextual accuracy | {BASELINE['contextual_acc']:.4f} | {suites['contextual']['accuracy']:.4f} |",
        f"| OOD rejection | {BASELINE['ood_rejection']:.4f} | {ood.get('rejection_rate', float('nan')):.4f} |",
        "",
        "> These are **not** like-for-like. The baseline numbers are for an "
        "11-intent problem; this run is 57 intents on a leakage-controlled "
        "split with harder held-out suites. A 57-way problem with a 35x class "
        "imbalance is a strictly harder task, so the baseline's accuracy is "
        "not a ceiling this run failed to reach — it is a different measurement. "
        "The only honest comparison is to re-run the baseline model against "
        "these same suites.\n",
    ]
    return "\n".join(L)


def main() -> None:
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "benchmark.md").write_text(benchmark_md())
    (REP / "calibration.md").write_text(calibration_md())
    (REP / "final_evaluation.md").write_text(final_md())
    print("wrote benchmark.md, calibration.md, final_evaluation.md")


if __name__ == "__main__":
    main()
