#!/usr/bin/env python3
"""
V6 PRODUCTION CONFIDENCE / FALLBACK CALIBRATION

Uses ONLY the original train.csv to create a deterministic held-out
validation split. The locked 1686-row test is NEVER read.

Goal:
- Measure V6 confidence behavior.
- Find a conservative fallback threshold.
- Compare baseline argmax vs thresholded predictions.
- Evaluate OOD fixture separately.
- Do not modify V6 model files.

Policy:
If max probability < threshold:
    Default Fallback Intent
Otherwise:
    keep classifier argmax prediction.

The threshold is selected using validation data only.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report


SEED = 20260809
PROJECT = Path("/Users/shuklam/IntentClassifier/semantic_project/57semanitc")

V6_ENCODER = PROJECT / "v6_e5_english_production_vocab/e5_base_v6_finetuned"
V6_CLASSIFIER = PROJECT / "v6_e5_english_production_vocab/e5_base_v6_logistic_classifier.joblib"
TRAIN_CSV = PROJECT / "train.csv"

OUT_DIR = PROJECT / "v6_confidence_calibration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FALLBACK = "Default Fallback Intent"
BATCH_SIZE = 64


def clean(x):
    return " ".join(str(x).strip().split())


def e5_query(x):
    return "query: " + clean(x)


def main():
    print("=" * 78)
    print("V6 PRODUCTION CONFIDENCE / FALLBACK CALIBRATION")
    print("=" * 78)

    for p in [V6_ENCODER, V6_CLASSIFIER, TRAIN_CSV]:
        if not p.exists():
            raise FileNotFoundError(p)

    df = pd.read_csv(TRAIN_CSV)
    if "text" not in df.columns or "intent" not in df.columns:
        raise RuntimeError(f"Expected text/intent columns, found {list(df.columns)}")

    df = df[["text", "intent"]].copy()
    df["text"] = df["text"].map(clean)
    df["intent"] = df["intent"].map(clean)
    df = df[(df.text != "") & (df.intent != "")].drop_duplicates("text").reset_index(drop=True)

    if df.intent.nunique() != 57:
        raise RuntimeError(f"Expected 57 intents, found {df.intent.nunique()}")

    # Same deterministic split family used by V6 training.
    _, val = train_test_split(
        df,
        test_size=0.10,
        random_state=SEED,
        stratify=df["intent"],
    )
    val = val.reset_index(drop=True)

    print(f"Validation rows : {len(val)}")
    print(f"Intents         : {val.intent.nunique()}")

    model = SentenceTransformer(str(V6_ENCODER))
    model.max_seq_length = 64
    clf = joblib.load(V6_CLASSIFIER)

    embeddings = model.encode(
        [e5_query(x) for x in val.text],
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    probs = clf.predict_proba(embeddings)
    classes = list(clf.classes_)
    pred_ids = np.argmax(probs, axis=1)
    pred_labels = [classes[i] for i in pred_ids]
    confidence = probs.max(axis=1)

    y_true = val.intent.to_numpy()

    base_acc = accuracy_score(y_true, pred_labels)
    base_macro = f1_score(y_true, pred_labels, average="macro", zero_division=0)
    base_weighted = f1_score(y_true, pred_labels, average="weighted", zero_division=0)

    rows = []

    # Conservative threshold sweep.
    thresholds = np.arange(0.00, 1.001, 0.01)

    for t in thresholds:
        thresholded = [
            FALLBACK if c < t else p
            for p, c in zip(pred_labels, confidence)
        ]

        rows.append({
            "threshold": float(t),
            "accuracy": float(accuracy_score(y_true, thresholded)),
            "macro_f1": float(f1_score(y_true, thresholded, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, thresholded, average="weighted", zero_division=0)),
            "fallback_rate": float(np.mean(np.array(thresholded) == FALLBACK)),
        })

    sweep = pd.DataFrame(rows)

    # We prioritize accuracy first, then macro F1.
    best = sweep.sort_values(
        ["accuracy", "macro_f1", "weighted_f1"],
        ascending=False,
    ).iloc[0]

    # Also choose the best threshold among those within 0.20 pp of
    # maximum accuracy, preferring the higher threshold (more conservative).
    max_acc = sweep["accuracy"].max()
    near = sweep[sweep["accuracy"] >= max_acc - 0.0020]
    conservative = near.sort_values(
        ["threshold", "macro_f1", "weighted_f1"],
        ascending=[False, False, False],
    ).iloc[0]

    recommended = conservative

    t = float(recommended["threshold"])
    final_pred = [
        FALLBACK if c < t else p
        for p, c in zip(pred_labels, confidence)
    ]

    final_report = classification_report(
        y_true,
        final_pred,
        digits=4,
        zero_division=0,
    )

    # Confidence distribution.
    conf_df = pd.DataFrame({
        "text": val.text,
        "true_intent": y_true,
        "argmax_prediction": pred_labels,
        "confidence": confidence,
    })
    conf_df["thresholded_prediction"] = final_pred
    conf_df["argmax_correct"] = (
        conf_df.true_intent == conf_df.argmax_prediction
    )
    conf_df["thresholded_correct"] = (
        conf_df.true_intent == conf_df.thresholded_prediction
    )

    sweep.to_csv(OUT_DIR / "threshold_sweep.csv", index=False)
    conf_df.to_csv(OUT_DIR / "validation_confidence.csv", index=False)

    summary = {
        "validation_rows": int(len(val)),
        "baseline_accuracy": float(base_acc),
        "baseline_macro_f1": float(base_macro),
        "baseline_weighted_f1": float(base_weighted),
        "best_accuracy_threshold": float(best.threshold),
        "best_accuracy": float(best.accuracy),
        "best_accuracy_macro_f1": float(best.macro_f1),
        "best_accuracy_weighted_f1": float(best.weighted_f1),
        "recommended_threshold": t,
        "recommended_accuracy": float(recommended.accuracy),
        "recommended_macro_f1": float(recommended.macro_f1),
        "recommended_weighted_f1": float(recommended.weighted_f1),
        "recommended_fallback_rate": float(recommended.fallback_rate),
        "locked_test_used": False,
        "locked_test_used_for_threshold_selection": False,
        "ood_used_for_threshold_selection": False,
    }

    (OUT_DIR / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    (OUT_DIR / "recommended_threshold_report.txt").write_text(
        "# V6 CONFIDENCE / FALLBACK CALIBRATION\n\n"
        f"Baseline accuracy   : {base_acc*100:.4f}%\n"
        f"Baseline macro F1   : {base_macro*100:.4f}%\n"
        f"Baseline weighted F1: {base_weighted*100:.4f}%\n\n"
        f"Recommended threshold: {t:.2f}\n"
        f"Thresholded accuracy   : {recommended.accuracy*100:.4f}%\n"
        f"Thresholded macro F1   : {recommended.macro_f1*100:.4f}%\n"
        f"Thresholded weighted F1: {recommended.weighted_f1*100:.4f}%\n"
        f"Validation fallback rate: {recommended.fallback_rate*100:.4f}%\n\n"
        "Thresholded classification report:\n"
        f"{final_report}\n",
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("V6 CALIBRATION RESULT")
    print("=" * 78)
    print(f"Baseline accuracy      : {base_acc*100:.4f}%")
    print(f"Baseline macro F1      : {base_macro*100:.4f}%")
    print(f"Baseline weighted F1   : {base_weighted*100:.4f}%")
    print()
    print(f"Recommended threshold  : {t:.2f}")
    print(f"Thresholded accuracy   : {recommended.accuracy*100:.4f}%")
    print(f"Thresholded macro F1   : {recommended.macro_f1*100:.4f}%")
    print(f"Thresholded weighted F1: {recommended.weighted_f1*100:.4f}%")
    print(f"Validation fallback    : {recommended.fallback_rate*100:.2f}%")
    print()
    print("Saved:")
    print(OUT_DIR / "threshold_sweep.csv")
    print(OUT_DIR / "validation_confidence.csv")
    print(OUT_DIR / "calibration_summary.json")
    print(OUT_DIR / "recommended_threshold_report.txt")
    print()
    print("IMPORTANT: locked 1686-row test was NOT read.")
    print("STATUS: V6 CONFIDENCE CALIBRATION COMPLETE")


if __name__ == "__main__":
    main()
