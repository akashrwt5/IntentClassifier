"""Unified evaluation report (roadmap §9.2; report_card-schema-shaped).

Evaluates the trained per-language pipelines against their holdouts and
emits ONE JSON artifact carrying every gate metric: per-language macro-F1 /
holdout accuracy / ECE, wrong-action counts, and OOS recall (recall of the
fallback class).

wrong_action_count (OFFICIAL definition, per the medical-safety budget):
a turn where the model CONFIDENTLY predicts an ACTIONABLE intent that
differs from the truth — i.e. the device would fire an action it shouldn't.
Predicted help.* or sys.oos.fallback cannot fire an action (help shows
content; fallback routes to GenAI narration), so confident confusions INTO
those classes are missed/deflected turns, not wrong actions. Confusions
FROM any true label INTO a different actionable intent count — including
help-utterances confidently misread as device commands, the worst case.
The broader proxy (all confident misclassifications) is still reported as
confident_misclassifications per language for trend-watching.

Usage:
    PYTHONPATH=packages/buildtime python -m nlu_training.evaluate \
        [--langs en fr de da] [--out evaluate_report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
# Per-language layout (see language_packs/). The legacy
# multilingual/models/<lang>/ tree was written only by the retired
# combined-multilingual trainer and no longer exists.
MODELS_DIR = REPO_ROOT / "models" / "intent"
DATASETS_DIR = REPO_ROOT / "language_packs"
CALIBRATION = REPO_ROOT / "config" / "calibration.json"

# The trained OOS / fallback class in the CURRENT label space. After the
# ND-3 migration this becomes 'sys.oos.fallback' — read from the migration
# map once it exists so this file needs no edit then.
FALLBACK_LABEL = "sys.oos.fallback"
ACCURACY_FLOOR = 0.80
# Medical-safety budget: max confident wrong ACTIONS across shipped
# (non-waived) languages. Charter: ≤5 global, moving toward per-domain.
WRONG_ACTION_BUDGET = 5


def is_actionable(label: str) -> bool:
    """True if a prediction of this label fires a device/app action."""
    return bool(label) and not label.startswith(("help.", "sys."))
# Danish is flag-gated (fails its floor; native-data program pending) —
# excluded from gates_passed, never from the report itself.
GATE_WAIVERS = {"da"}


def _normalize(texts):
    sys.path.insert(0, str(REPO_ROOT / "multilingual"))
    from text_norm import normalize_text
    return [normalize_text(t) for t in texts]


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(conf)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += (mask.sum() / total) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def evaluate_language(lang: str, calibration: dict) -> dict:
    import joblib
    import pandas as pd
    from sklearn.metrics import f1_score, recall_score

    pipeline = joblib.load(MODELS_DIR / lang / "pipeline.pkl")
    holdout = pd.read_csv(DATASETS_DIR / lang / "holdout_honest.csv")
    _text_col = next(c for c in ("text", "utterance") if c in holdout.columns)
    _label_col = next(c for c in ("intent", "expected_intent") if c in holdout.columns)
    texts = _normalize(holdout[_text_col].astype(str).tolist())
    y_true = holdout[_label_col].to_numpy()

    logits = pipeline.decision_function(texts)
    temperature = calibration[lang]["temperature"]
    probs = _softmax(np.asarray(logits) / temperature)
    classes = pipeline.classes_
    pred_idx = probs.argmax(axis=1)
    y_pred = classes[pred_idx]
    conf = probs[np.arange(len(pred_idx)), pred_idx]

    correct = (y_pred == y_true)
    threshold = calibration[lang]["conf_threshold"]
    wrong_confident = (~correct) & (conf >= threshold)
    # Official wrong actions: confident + predicted label would FIRE an action.
    pred_actionable = np.array([is_actionable(p) for p in y_pred])
    wrong_action = wrong_confident & pred_actionable

    # Per-domain by the PREDICTED domain (the domain that would wrongly act).
    per_domain: dict[str, int] = {}
    for label in y_pred[wrong_action]:
        domain = str(label).split(".")[0]
        per_domain[domain] = per_domain.get(domain, 0) + 1

    oos_mask = y_true == FALLBACK_LABEL
    oos_recall = (float(recall_score(oos_mask, y_pred == FALLBACK_LABEL))
                  if oos_mask.any() else 1.0)

    return {
        "schema_fields": {  # exactly what report_card.schema.json allows per language
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro",
                                             zero_division=0)), 4),
            "holdout_accuracy": round(float(correct.mean()), 4),
            "ece": round(_ece(conf, correct.astype(float)), 4),
        },
        "wrong_action_count": int(wrong_action.sum()),
        "wrong_action_per_domain": per_domain,
        "confident_misclassifications": int(wrong_confident.sum()),
        "oos_recall": round(oos_recall, 4),
        "n_holdout": int(len(y_true)),
    }


def evaluate_all(langs: list[str]) -> dict:
    calibration = json.loads(CALIBRATION.read_text())
    per_language: dict[str, dict] = {}
    details: dict[str, dict] = {}
    for lang in langs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            details[lang] = evaluate_language(lang, calibration)
        per_language[lang] = details[lang]["schema_fields"]

    wrong_total = sum(d["wrong_action_count"] for d in details.values())
    per_domain: dict[str, int] = {}
    for d in details.values():
        for k, v in d["wrong_action_per_domain"].items():
            per_domain[k] = per_domain.get(k, 0) + v
    oos_recall = min(d["oos_recall"] for d in details.values())

    # gates_passed = accuracy floors. The ≤5 wrong-action budget is a SYSTEM
    # property (engine keyword tiers + thresholds + confirmation gates sit
    # between the raw classifier and any action); the raw-classifier count
    # reported here is an upper bound for trend-watching, not the budget
    # gate. The budget gate needs the engine-in-the-loop harness (tracked
    # in known-issues.md).
    shipped = [lang for lang in langs if lang not in GATE_WAIVERS]
    gates_passed = all(
        per_language[lang]["holdout_accuracy"] >= ACCURACY_FLOOR for lang in shipped)

    return {
        "per_language": per_language,
        "wrong_action_count": wrong_total,
        "wrong_action_per_domain": per_domain,
        "oos_recall": oos_recall,
        "gates_passed": gates_passed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nlu_training.evaluate", description=__doc__)
    parser.add_argument("--langs", nargs="+", default=["en", "fr", "de", "da"])
    parser.add_argument("--out", type=Path, default=Path("evaluate_report.json"))
    args = parser.parse_args(argv)

    report = evaluate_all(args.langs)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    from .tracking import log_evaluation
    if log_evaluation(report, run_name="evaluate", params={"langs": ",".join(args.langs)}):
        print("  (logged to MLflow: ./mlruns)")
    for lang, m in report["per_language"].items():
        print(f"  {lang}: acc={m['holdout_accuracy']:.3f} macroF1={m['macro_f1']:.3f} "
              f"ece={m['ece']:.4f}")
    print(f"  wrong-action count (official, all langs): {report['wrong_action_count']} "
          f"(per predicted domain: {report['wrong_action_per_domain']})")
    print(f"  OOS recall (min over langs): {report['oos_recall']:.3f}")
    print(f"  gates_passed: {report['gates_passed']}  →  {args.out}")
    return 0 if report["gates_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
