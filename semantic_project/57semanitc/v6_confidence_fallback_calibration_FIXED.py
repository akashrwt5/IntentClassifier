#!/usr/bin/env python3
"""
V6 CONFIDENCE / FALLBACK CALIBRATION — FIXED

Fix:
The V6 LogisticRegression was trained with integer class IDs, while the
validation ground truth contains intent strings. The previous script
compared integers/IDs against strings, causing sklearn's mixed-target error.

This version loads V6 label_map.json and converts classifier predictions
back to intent strings before scoring.

The locked 1686-row test is NOT read.
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
PROJECT = Path(
    "/Users/shuklam/IntentClassifier/semantic_project/57semanitc"
)

V6_ENCODER = (
    PROJECT
    / "v6_e5_english_production_vocab"
    / "e5_base_v6_finetuned"
)

V6_CLASSIFIER = (
    PROJECT
    / "v6_e5_english_production_vocab"
    / "e5_base_v6_logistic_classifier.joblib"
)

V6_LABEL_MAP = (
    PROJECT
    / "v6_e5_english_production_vocab"
    / "label_map.json"
)

TRAIN_CSV = PROJECT / "train.csv"

OUT_DIR = PROJECT / "v6_confidence_calibration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FALLBACK = "Default Fallback Intent"
BATCH_SIZE = 64


def clean(x):
    return " ".join(str(x).strip().split())


def e5_query(x):
    return "query: " + clean(x)


def load_label_map():
    data = json.loads(
        V6_LABEL_MAP.read_text(
            encoding="utf-8"
        )
    )

    # V6 label_map is {"0": "IntentName", ...}
    return {
        int(k): str(v)
        for k, v in data.items()
    }


def main():

    print("=" * 78)
    print("V6 CONFIDENCE / FALLBACK CALIBRATION — FIXED")
    print("=" * 78)

    for p in [
        V6_ENCODER,
        V6_CLASSIFIER,
        V6_LABEL_MAP,
        TRAIN_CSV,
    ]:
        if not p.exists():
            raise FileNotFoundError(
                f"Required path not found:\n{p}"
            )

    df = pd.read_csv(TRAIN_CSV)

    if "text" not in df.columns or "intent" not in df.columns:
        raise RuntimeError(
            f"Expected text/intent columns, found {list(df.columns)}"
        )

    df = df[
        ["text", "intent"]
    ].copy()

    df["text"] = df["text"].map(clean)
    df["intent"] = df["intent"].map(clean)

    df = df[
        (df["text"] != "")
        & (df["intent"] != "")
    ].drop_duplicates(
        "text"
    ).reset_index(drop=True)

    if df["intent"].nunique() != 57:
        raise RuntimeError(
            f"Expected 57 intents, found {df['intent'].nunique()}"
        )

    # IMPORTANT:
    # This is the same deterministic validation split family used by V6.
    _, val = train_test_split(
        df,
        test_size=0.10,
        random_state=SEED,
        stratify=df["intent"],
    )

    val = val.reset_index(drop=True)

    print(
        f"Validation rows : {len(val)}"
    )
    print(
        f"Validation intents: {val['intent'].nunique()}"
    )

    label_map = load_label_map()

    print(
        f"Label map entries: {len(label_map)}"
    )

    if len(label_map) != 57:
        raise RuntimeError(
            f"Expected 57 label-map entries, found {len(label_map)}"
        )

    model = SentenceTransformer(
        str(V6_ENCODER)
    )

    model.max_seq_length = 64

    clf = joblib.load(
        V6_CLASSIFIER
    )

    # ---------------------------------------------------------------
    # ENCODE
    # ---------------------------------------------------------------

    print()
    print(
        "Encoding validation data..."
    )

    embeddings = model.encode(
        [
            e5_query(x)
            for x in val["text"]
        ],
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # ---------------------------------------------------------------
    # CLASSIFIER
    # ---------------------------------------------------------------

    probs = clf.predict_proba(
        embeddings
    )

    # LogisticRegression.classes_ contains the integer IDs used during
    # V6 training. Convert those IDs to actual intent strings.
    class_ids = [
        int(x)
        for x in clf.classes_
    ]

    predicted_ids = np.argmax(
        probs,
        axis=1,
    )

    pred_labels = [
        label_map[
            class_ids[i]
        ]
        for i in predicted_ids
    ]

    confidence = probs.max(
        axis=1
    )

    y_true = val[
        "intent"
    ].to_numpy()

    # ---------------------------------------------------------------
    # SANITY CHECK
    # ---------------------------------------------------------------

    unknown_predictions = sorted(
        set(pred_labels)
        - set(label_map.values())
    )

    if unknown_predictions:
        raise RuntimeError(
            "Unknown predicted intents: "
            + str(unknown_predictions)
        )

    print()
    print(
        "Prediction/label sanity check:"
    )
    print(
        f"True labels type       : {type(y_true[0]).__name__}"
    )
    print(
        f"Prediction labels type : {type(pred_labels[0]).__name__}"
    )
    print(
        f"Prediction classes     : {len(set(pred_labels))}"
    )

    # ---------------------------------------------------------------
    # BASELINE
    # ---------------------------------------------------------------

    base_acc = accuracy_score(
        y_true,
        pred_labels,
    )

    base_macro = f1_score(
        y_true,
        pred_labels,
        average="macro",
        zero_division=0,
    )

    base_weighted = f1_score(
        y_true,
        pred_labels,
        average="weighted",
        zero_division=0,
    )

    # ---------------------------------------------------------------
    # THRESHOLD SWEEP
    # ---------------------------------------------------------------

    rows = []

    thresholds = np.arange(
        0.00,
        1.001,
        0.01,
    )

    for threshold in thresholds:

        thresholded = [
            (
                FALLBACK
                if c < threshold
                else p
            )
            for p, c in zip(
                pred_labels,
                confidence,
            )
        ]

        rows.append({
            "threshold": float(threshold),
            "accuracy": float(
                accuracy_score(
                    y_true,
                    thresholded,
                )
            ),
            "macro_f1": float(
                f1_score(
                    y_true,
                    thresholded,
                    average="macro",
                    zero_division=0,
                )
            ),
            "weighted_f1": float(
                f1_score(
                    y_true,
                    thresholded,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "fallback_rate": float(
                np.mean(
                    np.array(thresholded)
                    == FALLBACK
                )
            ),
        })

    sweep = pd.DataFrame(
        rows
    )

    # Best validation accuracy.
    best = sweep.sort_values(
        [
            "accuracy",
            "macro_f1",
            "weighted_f1",
        ],
        ascending=False,
    ).iloc[0]

    # Conservative choice:
    # among thresholds within 0.20 percentage points of max accuracy,
    # prefer the highest threshold.
    max_acc = sweep[
        "accuracy"
    ].max()

    near = sweep[
        sweep["accuracy"]
        >= max_acc - 0.002
    ]

    recommended = near.sort_values(
        [
            "threshold",
            "macro_f1",
            "weighted_f1",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).iloc[0]

    threshold = float(
        recommended["threshold"]
    )

    final_pred = [
        (
            FALLBACK
            if c < threshold
            else p
        )
        for p, c in zip(
            pred_labels,
            confidence,
        )
    ]

    final_report = classification_report(
        y_true,
        final_pred,
        digits=4,
        zero_division=0,
    )

    # ---------------------------------------------------------------
    # PER-ROW CONFIDENCE FILE
    # ---------------------------------------------------------------

    confidence_df = pd.DataFrame({
        "text": val["text"],
        "true_intent": y_true,
        "argmax_prediction": pred_labels,
        "confidence": confidence,
    })

    confidence_df[
        "thresholded_prediction"
    ] = final_pred

    confidence_df[
        "argmax_correct"
    ] = (
        confidence_df[
            "true_intent"
        ]
        == confidence_df[
            "argmax_prediction"
        ]
    )

    confidence_df[
        "thresholded_correct"
    ] = (
        confidence_df[
            "true_intent"
        ]
        == confidence_df[
            "thresholded_prediction"
        ]
    )

    # ---------------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------------

    sweep.to_csv(
        OUT_DIR
        / "threshold_sweep.csv",
        index=False,
    )

    confidence_df.to_csv(
        OUT_DIR
        / "validation_confidence.csv",
        index=False,
    )

    summary = {
        "validation_rows": int(
            len(val)
        ),
        "baseline_accuracy": float(
            base_acc
        ),
        "baseline_macro_f1": float(
            base_macro
        ),
        "baseline_weighted_f1": float(
            base_weighted
        ),
        "best_accuracy_threshold": float(
            best["threshold"]
        ),
        "best_accuracy": float(
            best["accuracy"]
        ),
        "best_accuracy_macro_f1": float(
            best["macro_f1"]
        ),
        "best_accuracy_weighted_f1": float(
            best["weighted_f1"]
        ),
        "recommended_threshold": threshold,
        "recommended_accuracy": float(
            recommended["accuracy"]
        ),
        "recommended_macro_f1": float(
            recommended["macro_f1"]
        ),
        "recommended_weighted_f1": float(
            recommended["weighted_f1"]
        ),
        "recommended_fallback_rate": float(
            recommended["fallback_rate"]
        ),
        "locked_test_used": False,
        "locked_test_used_for_threshold_selection": False,
        "ood_used_for_threshold_selection": False,
    }

    (
        OUT_DIR
        / "calibration_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        OUT_DIR
        / "recommended_threshold_report.txt"
    ).write_text(
        "# V6 CONFIDENCE / FALLBACK CALIBRATION\n\n"
        f"Baseline accuracy   : {base_acc * 100:.4f}%\n"
        f"Baseline macro F1   : {base_macro * 100:.4f}%\n"
        f"Baseline weighted F1: {base_weighted * 100:.4f}%\n\n"
        f"Recommended threshold: {threshold:.2f}\n"
        f"Thresholded accuracy   : {recommended['accuracy'] * 100:.4f}%\n"
        f"Thresholded macro F1   : {recommended['macro_f1'] * 100:.4f}%\n"
        f"Thresholded weighted F1: {recommended['weighted_f1'] * 100:.4f}%\n"
        f"Validation fallback rate: {recommended['fallback_rate'] * 100:.4f}%\n\n"
        "Thresholded classification report:\n"
        f"{final_report}\n",
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(
        "V6 CALIBRATION RESULT"
    )
    print("=" * 78)

    print(
        f"Baseline accuracy      : {base_acc * 100:.4f}%"
    )
    print(
        f"Baseline macro F1      : {base_macro * 100:.4f}%"
    )
    print(
        f"Baseline weighted F1   : {base_weighted * 100:.4f}%"
    )

    print()
    print(
        f"Recommended threshold  : {threshold:.2f}"
    )
    print(
        f"Thresholded accuracy   : {recommended['accuracy'] * 100:.4f}%"
    )
    print(
        f"Thresholded macro F1   : {recommended['macro_f1'] * 100:.4f}%"
    )
    print(
        f"Thresholded weighted F1: {recommended['weighted_f1'] * 100:.4f}%"
    )
    print(
        f"Validation fallback    : {recommended['fallback_rate'] * 100:.2f}%"
    )

    print()
    print("Saved:")
    print(
        OUT_DIR
        / "threshold_sweep.csv"
    )
    print(
        OUT_DIR
        / "validation_confidence.csv"
    )
    print(
        OUT_DIR
        / "calibration_summary.json"
    )
    print(
        OUT_DIR
        / "recommended_threshold_report.txt"
    )

    print()
    print(
        "IMPORTANT: locked 1686-row test was NOT read."
    )
    print(
        "STATUS: V6 CONFIDENCE CALIBRATION COMPLETE"
    )


if __name__ == "__main__":
    main()
