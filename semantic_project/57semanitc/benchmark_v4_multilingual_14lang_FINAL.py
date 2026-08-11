#!/usr/bin/env python3
"""
V4 MULTILINGUAL E5-BASE — 14-LANGUAGE EVALUATION

Purpose:
    Evaluate the already-trained V4 multilingual E5-base intent model
    separately for each language and overall.

IMPORTANT:
- No training.
- No synthetic text generation.
- No labels changed.
- No locked 1686-row test is read.
- This script requires a REAL multilingual evaluation CSV.

Required CSV:
    /Users/shuklam/IntentClassifier/semantic_project/57semanitc/
    multilingual_14_test.csv

Required columns:
    language,text,intent

Example:
    language,text,intent
    English,make the volume louder,Cmd.VolumeIncrease
    Hindi,आवाज़ तेज कर दो,Cmd.VolumeIncrease
    ...

The CSV must contain exactly 14 unique language values and the same
57 intent labels used by V4.

The script reports:
    - overall accuracy / macro F1 / weighted F1
    - per-language accuracy / macro F1 / weighted F1
    - per-intent metrics
    - language x intent accuracy matrix
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


PROJECT = Path(
    "/Users/shuklam/IntentClassifier/semantic_project/57semanitc"
)

V4_DIR = PROJECT / "v4_multilingual_e5_base_semantic"

ENCODER_DIR = V4_DIR / "e5_base_finetuned"
CLASSIFIER_PATH = V4_DIR / "e5_base_logistic_classifier.joblib"
LABEL_MAP_PATH = V4_DIR / "label_map.json"

# Put the real 14-language evaluation CSV here.
EVAL_CSV = PROJECT / "multilingual_14_test.csv"

OUT_DIR = PROJECT / "v4_multilingual_14lang_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
EXPECTED_LANGUAGES = 14
EXPECTED_INTENTS = 57


def normalize_text(x):
    return str(x).strip().replace("\n", " ").replace("\r", " ")


def add_e5_prefix(text):
    return "query: " + normalize_text(text)


def load_labels():
    obj = json.loads(
        LABEL_MAP_PATH.read_text(encoding="utf-8")
    )

    if all(str(k).isdigit() for k in obj.keys()):
        return [obj[str(i)] for i in range(len(obj))]

    return [
        k for k, _ in sorted(
            obj.items(),
            key=lambda kv: int(kv[1]),
        )
    ]


def main():
    print("=" * 72)
    print("V4 MULTILINGUAL E5-BASE — 14-LANGUAGE EVALUATION")
    print("=" * 72)

    for path in [
        ENCODER_DIR,
        CLASSIFIER_PATH,
        LABEL_MAP_PATH,
        EVAL_CSV,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required path not found:\n{path}"
            )

    df = pd.read_csv(EVAL_CSV)

    required = {"language", "text", "intent"}
    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing columns: {sorted(missing)}\n"
            "Required: language,text,intent"
        )

    df = df[["language", "text", "intent"]].copy()

    for col in ["language", "text", "intent"]:
        df[col] = df[col].map(normalize_text)

    df = df[
        (df["language"] != "")
        & (df["text"] != "")
        & (df["intent"] != "")
    ].reset_index(drop=True)

    languages = sorted(df["language"].unique().tolist())
    labels = load_labels()

    print()
    print(f"Rows      : {len(df)}")
    print(f"Languages : {len(languages)}")
    print(f"Intents   : {df['intent'].nunique()}")
    print()
    print("Languages:")
    for language in languages:
        print(f"  - {language}")

    if len(languages) != EXPECTED_LANGUAGES:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_LANGUAGES} languages, "
            f"found {len(languages)}."
        )

    if len(labels) != EXPECTED_INTENTS:
        raise RuntimeError(
            f"V4 label map has {len(labels)} labels; "
            f"expected {EXPECTED_INTENTS}."
        )

    unknown_labels = sorted(
        set(df["intent"]) - set(labels)
    )

    if unknown_labels:
        raise RuntimeError(
            "Evaluation CSV contains labels not present in V4:\n"
            + "\n".join(unknown_labels)
        )

    print()
    print("Loading V4 multilingual E5 encoder...")
    model = SentenceTransformer(str(ENCODER_DIR))

    classifier = joblib.load(CLASSIFIER_PATH)

    texts = [
        add_e5_prefix(x)
        for x in df["text"].tolist()
    ]

    print()
    print("Encoding multilingual evaluation set...")

    start = time.perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    embedding_time = time.perf_counter() - start

    pred_indices = classifier.predict(embeddings)

    predictions = [
        labels[int(i)]
        for i in pred_indices
    ]

    df["prediction"] = predictions
    df["correct"] = (
        df["intent"] == df["prediction"]
    )

    # ---------------------------------------------------------------
    # OVERALL
    # ---------------------------------------------------------------

    overall_accuracy = accuracy_score(
        df["intent"],
        df["prediction"],
    )

    overall_macro = f1_score(
        df["intent"],
        df["prediction"],
        labels=labels,
        average="macro",
        zero_division=0,
    )

    overall_weighted = f1_score(
        df["intent"],
        df["prediction"],
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    # ---------------------------------------------------------------
    # PER LANGUAGE
    # ---------------------------------------------------------------

    language_rows = []

    for language in languages:
        part = df[df["language"] == language]

        language_rows.append({
            "language": language,
            "rows": len(part),
            "accuracy": accuracy_score(
                part["intent"],
                part["prediction"],
            ),
            "macro_f1": f1_score(
                part["intent"],
                part["prediction"],
                labels=labels,
                average="macro",
                zero_division=0,
            ),
            "weighted_f1": f1_score(
                part["intent"],
                part["prediction"],
                labels=labels,
                average="weighted",
                zero_division=0,
            ),
        })

    language_df = pd.DataFrame(
        language_rows
    ).sort_values(
        "accuracy"
    )

    # ---------------------------------------------------------------
    # PER INTENT
    # ---------------------------------------------------------------

    intent_report = classification_report(
        df["intent"],
        df["prediction"],
        labels=labels,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    intent_rows = []

    for label in labels:
        metrics = intent_report[label]

        intent_rows.append({
            "intent": label,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1-score"],
            "support": int(metrics["support"]),
        })

    intent_df = pd.DataFrame(
        intent_rows
    ).sort_values("f1")

    # ---------------------------------------------------------------
    # LANGUAGE x INTENT
    # ---------------------------------------------------------------

    lang_intent_rows = []

    for language in languages:
        for intent in labels:
            part = df[
                (df["language"] == language)
                & (df["intent"] == intent)
            ]

            if len(part) == 0:
                continue

            lang_intent_rows.append({
                "language": language,
                "intent": intent,
                "rows": len(part),
                "accuracy": float(
                    part["correct"].mean()
                ),
            })

    lang_intent_df = pd.DataFrame(
        lang_intent_rows
    )

    # ---------------------------------------------------------------
    # FAILURE ANALYSIS
    # ---------------------------------------------------------------

    failures = df[
        ~df["correct"]
    ].copy()

    # ---------------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------------

    predictions_path = (
        OUT_DIR / "multilingual_14_predictions.csv"
    )
    language_path = (
        OUT_DIR / "per_language_metrics.csv"
    )
    intent_path = (
        OUT_DIR / "per_intent_metrics.csv"
    )
    lang_intent_path = (
        OUT_DIR / "language_intent_metrics.csv"
    )
    failures_path = (
        OUT_DIR / "multilingual_14_failures.csv"
    )

    df.to_csv(
        predictions_path,
        index=False,
    )

    language_df.to_csv(
        language_path,
        index=False,
    )

    intent_df.to_csv(
        intent_path,
        index=False,
    )

    lang_intent_df.to_csv(
        lang_intent_path,
        index=False,
    )

    failures.to_csv(
        failures_path,
        index=False,
    )

    report = classification_report(
        df["intent"],
        df["prediction"],
        labels=labels,
        target_names=labels,
        digits=4,
        zero_division=0,
    )

    report_text = f"""# V4 MULTILINGUAL E5-BASE — 14-LANGUAGE EVALUATION

Rows       : {len(df)}
Languages  : {len(languages)}
Intents    : {df['intent'].nunique()}

Overall Accuracy   : {overall_accuracy * 100:.4f}%
Overall Macro F1   : {overall_macro * 100:.4f}%
Overall Weighted F1: {overall_weighted * 100:.4f}%

Embedding time     : {embedding_time:.4f} sec
Rows/sec           : {len(df) / embedding_time:.2f}
ms/row             : {embedding_time * 1000 / len(df):.4f}

PER-LANGUAGE RESULTS
--------------------
{language_df.to_string(index=False)}

PER-INTENT CLASSIFICATION REPORT
--------------------------------
{report}
"""

    report_path = (
        OUT_DIR / "multilingual_14_report.txt"
    )

    report_path.write_text(
        report_text,
        encoding="utf-8",
    )

    summary = {
        "model": "multilingual-e5-base",
        "evaluation_csv": str(EVAL_CSV),
        "rows": int(len(df)),
        "languages": languages,
        "num_languages": len(languages),
        "num_intents": int(df["intent"].nunique()),
        "accuracy": float(overall_accuracy),
        "macro_f1": float(overall_macro),
        "weighted_f1": float(overall_weighted),
        "embedding_time_sec": float(embedding_time),
        "rows_per_sec": float(
            len(df) / embedding_time
        ),
        "ms_per_row": float(
            embedding_time * 1000 / len(df)
        ),
        "synthetic_text": False,
        "training_performed": False,
        "locked_test_used": False,
        "quantization": False,
        "onnx": False,
    }

    summary_path = (
        OUT_DIR / "multilingual_14_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("V4 MULTILINGUAL 14-LANGUAGE RESULT")
    print("=" * 72)

    print(
        f"Accuracy   : {overall_accuracy * 100:.4f}%"
    )
    print(
        f"Macro F1   : {overall_macro * 100:.4f}%"
    )
    print(
        f"Weighted F1: {overall_weighted * 100:.4f}%"
    )

    print()
    print("--- PER LANGUAGE ---")
    print(
        language_df.to_string(index=False)
    )

    print()
    print("--- WORST 10 INTENTS ---")
    print(
        intent_df.head(10).to_string(index=False)
    )

    print()
    print("--- FAILURES ---")

    if len(failures) == 0:
        print("NONE")
    else:
        print(
            failures[
                [
                    "language",
                    "text",
                    "intent",
                    "prediction",
                ]
            ].to_string(index=False)
        )

    print()
    print("Saved:")
    print(predictions_path)
    print(language_path)
    print(intent_path)
    print(lang_intent_path)
    print(failures_path)
    print(report_path)
    print(summary_path)

    print()
    print(
        "STATUS: V4 MULTILINGUAL 14-LANGUAGE "
        "EVALUATION COMPLETE"
    )


if __name__ == "__main__":
    main()
