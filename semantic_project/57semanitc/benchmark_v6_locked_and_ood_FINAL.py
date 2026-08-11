#!/usr/bin/env python3
"""
V6 EXACT LOCKED 57-INTENT BENCHMARK + OOD TEST

V6:
 /Users/shuklam/IntentClassifier/semantic_project/57semanitc/
 v6_e5_english_production_vocab/e5_base_v6_finetuned

Classifier:
 e5_base_v6_logistic_classifier.joblib

Locked test:
 v3_57intent_locked_eval/locked_test_57intent.csv

OOD:
 Reuses the existing V3 negative-test fixture if present.
 It is used ONLY for evaluation.

IMPORTANT:
- No training.
- No fine-tuning.
- No modification of V6.
- Locked test is evaluation only.
- No quantization.
- No ONNX.
- English only.
"""

from pathlib import Path
import json
import time
import re

import numpy as np
import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


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

LOCKED_CSV = (
    PROJECT
    / "v3_57intent_locked_eval"
    / "locked_test_57intent.csv"
)

OOD_CANDIDATES = [
    PROJECT
    / "v3_57intent_e5_distilled_v3_negative_test"
    / "negative_test_v3_results.csv",
    PROJECT
    / "v3_57intent_e5_distilled_v2_negative_test"
    / "negative_test_results.csv",
    PROJECT
    / "v3_57intent_negative_test"
    / "negative_test_results.csv",
]

OUT_DIR = (
    PROJECT
    / "v6_locked_and_ood_benchmark"
)
OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BATCH_SIZE = 64


def clean(x):
    return " ".join(
        str(x).strip().split()
    )


def e5_query(x):
    return "query: " + clean(x)


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_locked():
    if not LOCKED_CSV.exists():
        raise FileNotFoundError(
            f"Locked CSV not found:\n{LOCKED_CSV}"
        )

    df = pd.read_csv(LOCKED_CSV)

    text_col = find_col(
        df,
        ["text", "utterance", "query", "sentence"],
    )
    label_col = find_col(
        df,
        ["intent", "label", "true_intent", "expected_intent"],
    )

    if not text_col or not label_col:
        raise RuntimeError(
            "Locked CSV must contain text + intent columns. "
            f"Found: {list(df.columns)}"
        )

    out = df[
        [text_col, label_col]
    ].copy()

    out.columns = [
        "text",
        "intent",
    ]

    out["text"] = out["text"].map(clean)
    out["intent"] = out["intent"].map(clean)

    return out


def load_ood():
    for path in OOD_CANDIDATES:
        if path.exists():
            df = pd.read_csv(path)

            text_col = find_col(
                df,
                ["text", "utterance", "query", "sentence"],
            )

            if text_col:
                out = df.copy()
                out["text"] = out[
                    text_col
                ].map(clean)
                return out, path

    # If an existing fixture is unavailable, use a small manually
    # defined evaluation-only OOD set. These are NOT training data.
    phrases = [
        "play some music",
        "show me a funny video",
        "send an email",
        "book me a flight",
        "take me to the airport",
        "go to the airport tomorrow at 9 pm",
        "turn off the television",
        "turn off the lights",
        "turn off my phone",
        "open the camera",
        "what is the weather",
        "call my friend",
        "set an alarm for tomorrow",
        "navigate to the airport",
        "order me a taxi",
        "play a podcast",
        "open youtube",
        "turn on the television",
        "check my bank account",
        "what time is it",
    ]

    return (
        pd.DataFrame({"text": phrases}),
        None,
    )


def load_label_map():
    if not V6_LABEL_MAP.exists():
        raise FileNotFoundError(
            f"V6 label_map.json not found:\n{V6_LABEL_MAP}"
        )

    data = json.loads(
        V6_LABEL_MAP.read_text(
            encoding="utf-8"
        )
    )

    # Expected format: {"0": "intent", ...}
    return {
        int(k): v
        for k, v in data.items()
    }


def main():

    print("=" * 78)
    print(
        "V6 EXACT LOCKED 57-INTENT BENCHMARK + OOD TEST"
    )
    print("=" * 78)

    for path in [
        V6_ENCODER,
        V6_CLASSIFIER,
        V6_LABEL_MAP,
        LOCKED_CSV,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required path not found:\n{path}"
            )

    print()
    print(f"V6 Encoder   : {V6_ENCODER}")
    print(f"V6 Classifier : {V6_CLASSIFIER}")
    print(f"Locked CSV    : {LOCKED_CSV}")

    # ---------------------------------------------------------------
    # LOAD
    # ---------------------------------------------------------------

    label_map = load_label_map()

    classifier = joblib.load(
        V6_CLASSIFIER
    )

    model = SentenceTransformer(
        str(V6_ENCODER)
    )

    model.max_seq_length = 64

    locked = load_locked()

    print()
    print(
        f"Locked rows  : {len(locked)}"
    )
    print(
        f"Locked intents: {locked['intent'].nunique()}"
    )

    if len(locked) != 1686:
        print(
            "WARNING: locked row count is not 1686."
        )

    if locked["intent"].nunique() != 57:
        print(
            "WARNING: locked intent count is not 57."
        )

    # ---------------------------------------------------------------
    # LOCKED INFERENCE
    # ---------------------------------------------------------------

    print()
    print(
        "Running V6 locked inference..."
    )

    start = time.perf_counter()

    embeddings = model.encode(
        [
            e5_query(x)
            for x in locked["text"]
        ],
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    embedding_time = (
        time.perf_counter() - start
    )

    start = time.perf_counter()

    pred_ids = classifier.predict(
        embeddings
    )

    classifier_time = (
        time.perf_counter() - start
    )

    predictions = [
        label_map.get(
            int(x),
            str(x),
        )
        for x in pred_ids
    ]

    y_true = locked[
        "intent"
    ].tolist()

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )

    labels = sorted(
        set(y_true)
        | set(predictions)
    )

    report = classification_report(
        y_true,
        predictions,
        labels=labels,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=labels,
    )

    locked_output = locked.copy()

    locked_output[
        "prediction"
    ] = predictions

    locked_output[
        "correct"
    ] = (
        locked_output[
            "intent"
        ]
        == locked_output[
            "prediction"
        ]
    )

    locked_output[
        "confidence"
    ] = classifier.predict_proba(
        embeddings
    ).max(axis=1)

    locked_path = (
        OUT_DIR
        / "locked_predictions_v6.csv"
    )

    report_path = (
        OUT_DIR
        / "locked_test_report_v6.txt"
    )

    cm_path = (
        OUT_DIR
        / "confusion_matrix_v6.csv"
    )

    locked_output.to_csv(
        locked_path,
        index=False,
    )

    pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    ).to_csv(
        cm_path
    )

    rows_per_sec = (
        len(locked)
        / max(
            embedding_time
            + classifier_time,
            1e-9,
        )
    )

    report_path.write_text(
        "# V6 EXACT LOCKED 57-INTENT TEST\n\n"
        f"Accuracy   : {accuracy * 100:.4f}%\n"
        f"Macro F1   : {macro_f1 * 100:.4f}%\n"
        f"Weighted F1: {weighted_f1 * 100:.4f}%\n"
        f"Rows       : {len(locked)}\n"
        f"Embedding time : {embedding_time:.4f} sec\n"
        f"Classifier time: {classifier_time:.4f} sec\n"
        f"Rows/sec       : {rows_per_sec:.2f}\n\n"
        "Classification report:\n"
        f"{report}\n",
        encoding="utf-8",
    )

    # ---------------------------------------------------------------
    # OOD
    # ---------------------------------------------------------------

    print()
    print(
        "Running V6 OOD / negative test..."
    )

    ood, ood_source = load_ood()

    ood_embeddings = model.encode(
        [
            e5_query(x)
            for x in ood["text"]
        ],
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    ood_ids = classifier.predict(
        ood_embeddings
    )

    ood_pred = [
        label_map.get(
            int(x),
            str(x),
        )
        for x in ood_ids
    ]

    ood_probs = classifier.predict_proba(
        ood_embeddings
    )

    ood_conf = ood_probs.max(
        axis=1
    )

    ood_output = ood.copy()

    ood_output[
        "prediction"
    ] = ood_pred

    ood_output[
        "confidence"
    ] = ood_conf

    ood_output[
        "is_fallback"
    ] = (
        ood_output[
            "prediction"
        ]
        == "Default Fallback Intent"
    )

    ood_path = (
        OUT_DIR
        / "ood_predictions_v6.csv"
    )

    ood_output.to_csv(
        ood_path,
        index=False,
    )

    fallback_count = int(
        ood_output[
            "is_fallback"
        ].sum()
    )

    ood_total = len(
        ood_output
    )

    fallback_rate = (
        fallback_count
        / ood_total
        if ood_total
        else 0.0
    )

    non_fallback = ood_output[
        ~ood_output[
            "is_fallback"
        ]
    ].copy()

    # ---------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------

    summary = {
        "model": "V6 E5 English production vocabulary",
        "locked_csv": str(LOCKED_CSV),
        "locked_rows": int(len(locked)),
        "locked_intents": int(
            locked["intent"].nunique()
        ),
        "locked_accuracy": float(
            accuracy
        ),
        "locked_macro_f1": float(
            macro_f1
        ),
        "locked_weighted_f1": float(
            weighted_f1
        ),
        "embedding_time_sec": float(
            embedding_time
        ),
        "classifier_time_sec": float(
            classifier_time
        ),
        "rows_per_sec": float(
            rows_per_sec
        ),
        "ood_source": (
            str(ood_source)
            if ood_source
            else "built_in_evaluation_only_fixture"
        ),
        "ood_rows": int(
            ood_total
        ),
        "ood_fallback_count": int(
            fallback_count
        ),
        "ood_fallback_rate": float(
            fallback_rate
        ),
        "ood_non_fallback_count": int(
            len(non_fallback)
        ),
        "locked_test_used_for_training": False,
        "quantization": False,
        "onnx": False,
        "training_performed": False,
    }

    summary_path = (
        OUT_DIR
        / "benchmark_summary_v6.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------------
    # PRINT
    # ---------------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "V6 EXACT LOCKED 57-INTENT RESULT"
    )
    print("=" * 78)

    print(
        f"Accuracy   : {accuracy * 100:.4f}%"
    )
    print(
        f"Macro F1   : {macro_f1 * 100:.4f}%"
    )
    print(
        f"Weighted F1: {weighted_f1 * 100:.4f}%"
    )

    print()
    print(
        "--- OOD / NEGATIVE TEST ---"
    )
    print(
        f"Total OOD rows       : {ood_total}"
    )
    print(
        f"Fallback predictions : {fallback_count}"
    )
    print(
        f"Fallback rate        : {fallback_rate * 100:.2f}%"
    )

    print()

    if len(non_fallback):
        print(
            "--- NON-FALLBACK OOD PREDICTIONS ---"
        )
        print(
            non_fallback[
                [
                    "text",
                    "prediction",
                    "confidence",
                ]
            ].to_string(
                index=False
            )
        )
    else:
        print(
            "NONE — all OOD queries predicted Default Fallback Intent."
        )

    print()
    print(
        "--- INFERENCE SPEED ---"
    )
    print(
        f"Total rows : {len(locked)}"
    )
    print(
        f"Embedding time : {embedding_time:.4f} sec"
    )
    print(
        f"Classifier time: {classifier_time:.4f} sec"
    )
    print(
        f"Rows/sec       : {rows_per_sec:.2f}"
    )

    print()
    print("Saved:")
    print(locked_path)
    print(report_path)
    print(cm_path)
    print(ood_path)
    print(summary_path)

    print()
    print(
        "STATUS: V6 LOCKED + OOD BENCHMARK COMPLETE"
    )


if __name__ == "__main__":
    main()
