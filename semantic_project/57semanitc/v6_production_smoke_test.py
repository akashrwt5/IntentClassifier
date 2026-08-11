#!/usr/bin/env python3
"""
V6 PRODUCTION SMOKE TEST

Evaluation only. No training and no model modification.

Builds a production-style English test set from EXISTING labeled/evaluation
files on disk. It does NOT generate synthetic text.

Sources are discovered from known project locations:
  - language_packs/en/holdout_honest.csv
  - language_packs/en/holdout_paraphrase.csv
  - language_packs/en/extras/semantic_holdout_2.csv
  - language_packs/en/extras/holdout_leakage_guard.csv
  - semantic_project/57semanitc/unseen_semantic_stress_test.csv

The script:
  1. Loads V6 E5 encoder + LogisticRegression.
  2. Reads only evaluation sources (not train.csv).
  3. Keeps rows whose intent belongs to V6's 57 labels.
  4. Removes exact duplicate texts.
  5. Runs V6 inference.
  6. Reports overall/per-intent metrics.
  7. Reports short-query behavior.
  8. Reports low-confidence predictions.
  9. Reports non-fallback predictions on OOD-like rows when a source has no
     usable ground-truth intent.
 10. Saves every prediction for manual review.

IMPORTANT:
- The locked 1686-row test is NEVER read.
- train.csv is NEVER read.
- No synthetic phrases are generated.
- No threshold is tuned here.
"""

from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)


PROJECT = Path(
    "/Users/shuklam/IntentClassifier/semantic_project/57semanitc"
)

ROOT = Path(
    "/Users/shuklam/IntentClassifier"
)

V6_DIR = (
    PROJECT
    / "v6_e5_english_production_vocab"
)

ENCODER = (
    V6_DIR
    / "e5_base_v6_finetuned"
)

CLASSIFIER = (
    V6_DIR
    / "e5_base_v6_logistic_classifier.joblib"
)

LABEL_MAP = (
    V6_DIR
    / "label_map.json"
)

OUT_DIR = (
    PROJECT
    / "v6_production_smoke_test"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FALLBACK = "Default Fallback Intent"
BATCH_SIZE = 64

# Existing evaluation sources only.
SOURCE_CANDIDATES = [
    ROOT
    / "language_packs/en/holdout_honest.csv",

    ROOT
    / "language_packs/en/holdout_paraphrase.csv",

    ROOT
    / "language_packs/en/extras/semantic_holdout_2.csv",

    ROOT
    / "language_packs/en/extras/holdout_leakage_guard.csv",

    PROJECT
    / "unseen_semantic_stress_test.csv",
]


def clean(x):
    return " ".join(
        str(x).strip().split()
    )


def e5_query(x):
    return "query: " + clean(x)


def find_column(df, candidates):
    lower = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    return None


def load_label_map():
    data = json.loads(
        LABEL_MAP.read_text(
            encoding="utf-8"
        )
    )

    return {
        int(k): str(v)
        for k, v in data.items()
    }


def load_sources(label_set):
    frames = []
    source_info = []

    for path in SOURCE_CANDIDATES:

        if not path.exists():
            continue

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(
                f"SKIP {path}: {exc}"
            )
            continue

        text_col = find_column(
            df,
            [
                "text",
                "utterance",
                "query",
                "sentence",
            ],
        )

        intent_col = find_column(
            df,
            [
                "intent",
                "label",
                "true_intent",
                "expected_intent",
            ],
        )

        if text_col is None:
            print(
                f"SKIP {path}: no text column"
            )
            continue

        tmp = pd.DataFrame({
            "text": df[text_col].map(clean),
            "intent": (
                df[intent_col].map(clean)
                if intent_col is not None
                else pd.Series(
                    [None] * len(df)
                )
            ),
        })

        tmp["source"] = str(path)

        tmp = tmp[
            tmp["text"] != ""
        ].copy()

        if intent_col is not None:
            tmp["intent"] = tmp["intent"].replace(
                {
                    "nan": None,
                    "None": None,
                    "": None,
                }
            )

            # Keep only labels known by V6.
            tmp = tmp[
                tmp["intent"].isin(
                    label_set
                )
            ].copy()

        if len(tmp) == 0:
            continue

        frames.append(tmp)

        source_info.append({
            "path": str(path),
            "rows_loaded": int(len(tmp)),
            "has_intent": bool(
                intent_col is not None
            ),
        })

    if not frames:
        raise RuntimeError(
            "No usable evaluation sources found."
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    # Exact text de-duplication across sources.
    combined = combined.drop_duplicates(
        subset=["text"],
        keep="first",
    ).reset_index(drop=True)

    return combined, source_info


def main():

    print("=" * 78)
    print(
        "V6 PRODUCTION SMOKE TEST"
    )
    print("=" * 78)

    for path in [
        ENCODER,
        CLASSIFIER,
        LABEL_MAP,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required path not found:\n{path}"
            )

    label_map = load_label_map()
    label_set = set(
        label_map.values()
    )

    if len(label_set) != 57:
        raise RuntimeError(
            f"Expected 57 V6 labels, found {len(label_set)}"
        )

    data, source_info = load_sources(
        label_set
    )

    print()
    print(
        f"Smoke-test rows : {len(data)}"
    )
    print(
        f"Known intents   : {data['intent'].notna().sum()}"
    )
    print(
        f"57-intent coverage: "
        f"{data.loc[data.intent.notna(), 'intent'].nunique()}/57"
    )

    print()
    print(
        "Sources:"
    )
    for item in source_info:
        print(
            f"  {item['rows_loaded']:4d} | "
            f"{item['path']}"
        )

    # ---------------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------------

    print()
    print(
        "Loading V6..."
    )

    model = SentenceTransformer(
        str(ENCODER)
    )
    model.max_seq_length = 64

    clf = joblib.load(
        CLASSIFIER
    )

    # ---------------------------------------------------------------
    # INFERENCE
    # ---------------------------------------------------------------

    start = time.perf_counter()

    embeddings = model.encode(
        [
            e5_query(x)
            for x in data["text"]
        ],
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    embedding_time = (
        time.perf_counter()
        - start
    )

    start = time.perf_counter()

    probabilities = clf.predict_proba(
        embeddings
    )

    classifier_time = (
        time.perf_counter()
        - start
    )

    class_ids = [
        int(x)
        for x in clf.classes_
    ]

    pred_indices = np.argmax(
        probabilities,
        axis=1,
    )

    predictions = [
        label_map[
            class_ids[i]
        ]
        for i in pred_indices
    ]

    confidence = probabilities.max(
        axis=1
    )

    top3_idx = np.argsort(
        probabilities,
        axis=1,
    )[:, -3:][:, ::-1]

    top3_text = []

    for row in top3_idx:
        parts = []

        for i in row:
            intent = label_map[
                class_ids[int(i)]
            ]
            score = float(
                probabilities[
                    len(top3_text),
                    int(i),
                ]
            )

            parts.append(
                f"{intent}={score:.4f}"
            )

        top3_text.append(
            " | ".join(parts)
        )

    result = data.copy()

    result[
        "prediction"
    ] = predictions

    result[
        "confidence"
    ] = confidence

    result[
        "correct"
    ] = (
        result["intent"]
        == result["prediction"]
    )

    result[
        "fallback"
    ] = (
        result["prediction"]
        == FALLBACK
    )

    result[
        "word_count"
    ] = (
        result["text"]
        .str.split()
        .str.len()
    )

    result[
        "char_count"
    ] = result["text"].str.len()

    result[
        "top3"
    ] = top3_text

    result[
        "low_confidence_lt_0_50"
    ] = (
        result["confidence"]
        < 0.50
    )

    result[
        "low_confidence_lt_0_70"
    ] = (
        result["confidence"]
        < 0.70
    )

    result[
        "low_confidence_lt_0_90"
    ] = (
        result["confidence"]
        < 0.90
    )

    # ---------------------------------------------------------------
    # LABELED METRICS
    # ---------------------------------------------------------------

    labeled = result[
        result["intent"].notna()
    ].copy()

    if len(labeled) > 0:

        y_true = labeled[
            "intent"
        ].tolist()

        y_pred = labeled[
            "prediction"
        ].tolist()

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        macro_f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        weighted_f1 = f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        report = classification_report(
            y_true,
            y_pred,
            digits=4,
            zero_division=0,
        )

    else:
        accuracy = None
        macro_f1 = None
        weighted_f1 = None
        report = (
            "No labeled evaluation rows."
        )

    # ---------------------------------------------------------------
    # SHORT QUERY ANALYSIS
    # ---------------------------------------------------------------

    short = result[
        result["word_count"] <= 3
    ].copy()

    short_labeled = short[
        short["intent"].notna()
    ].copy()

    if len(short_labeled) > 0:
        short_accuracy = accuracy_score(
            short_labeled["intent"],
            short_labeled["prediction"],
        )
    else:
        short_accuracy = None

    # ---------------------------------------------------------------
    # PER-INTENT
    # ---------------------------------------------------------------

    per_intent = []

    for intent in sorted(
        label_set
    ):

        subset = labeled[
            labeled["intent"]
            == intent
        ]

        if len(subset) == 0:
            continue

        per_intent.append({
            "intent": intent,
            "support": int(len(subset)),
            "accuracy": float(
                accuracy_score(
                    subset["intent"],
                    subset["prediction"],
                )
            ),
            "mean_confidence": float(
                subset["confidence"].mean()
            ),
            "fallback_rate": float(
                subset["fallback"].mean()
            ),
            "low_confidence_lt_0_50": float(
                subset[
                    "low_confidence_lt_0_50"
                ].mean()
            ),
            "low_confidence_lt_0_70": float(
                subset[
                    "low_confidence_lt_0_70"
                ].mean()
            ),
        })

    per_intent_df = pd.DataFrame(
        per_intent
    )

    # ---------------------------------------------------------------
    # LOW-CONFIDENCE REVIEW
    # ---------------------------------------------------------------

    low_conf = result[
        result["confidence"] < 0.70
    ].sort_values(
        "confidence"
    )

    # Incorrect labeled examples first.
    errors = labeled[
        ~labeled["correct"]
    ].sort_values(
        "confidence"
    )

    # ---------------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------------

    result.to_csv(
        OUT_DIR
        / "production_smoke_predictions.csv",
        index=False,
    )

    per_intent_df.to_csv(
        OUT_DIR
        / "per_intent_smoke_metrics.csv",
        index=False,
    )

    low_conf.to_csv(
        OUT_DIR
        / "low_confidence_review.csv",
        index=False,
    )

    errors.to_csv(
        OUT_DIR
        / "smoke_errors.csv",
        index=False,
    )

    summary = {
        "model": "V6 E5 English production vocabulary",
        "rows": int(len(result)),
        "labeled_rows": int(len(labeled)),
        "intent_coverage": int(
            labeled["intent"].nunique()
        ),
        "accuracy": (
            float(accuracy)
            if accuracy is not None
            else None
        ),
        "macro_f1": (
            float(macro_f1)
            if macro_f1 is not None
            else None
        ),
        "weighted_f1": (
            float(weighted_f1)
            if weighted_f1 is not None
            else None
        ),
        "short_query_rows": int(
            len(short)
        ),
        "short_query_accuracy": (
            float(short_accuracy)
            if short_accuracy is not None
            else None
        ),
        "low_confidence_lt_0_50": int(
            (result["confidence"] < 0.50).sum()
        ),
        "low_confidence_lt_0_70": int(
            (result["confidence"] < 0.70).sum()
        ),
        "low_confidence_lt_0_90": int(
            (result["confidence"] < 0.90).sum()
        ),
        "fallback_count": int(
            result["fallback"].sum()
        ),
        "fallback_rate": float(
            result["fallback"].mean()
        ),
        "embedding_time_sec": float(
            embedding_time
        ),
        "classifier_time_sec": float(
            classifier_time
        ),
        "locked_test_read": False,
        "train_csv_read": False,
        "synthetic_text_generated": False,
        "threshold_tuned": False,
    }

    (
        OUT_DIR
        / "smoke_test_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        OUT_DIR
        / "smoke_test_report.txt"
    ).write_text(
        "# V6 PRODUCTION SMOKE TEST\n\n"
        f"Rows: {len(result)}\n"
        f"Labeled rows: {len(labeled)}\n"
        f"Intent coverage: {labeled['intent'].nunique()}/57\n\n"
        f"Accuracy: "
        f"{accuracy * 100:.4f}%\n"
        if accuracy is not None
        else "# V6 PRODUCTION SMOKE TEST\n\nNo labeled rows.\n",
        encoding="utf-8",
    )

    # Append additional report details.
    with (
        OUT_DIR
        / "smoke_test_report.txt"
    ).open(
        "a",
        encoding="utf-8",
    ) as f:

        if accuracy is not None:
            f.write(
                f"Macro F1: "
                f"{macro_f1 * 100:.4f}%\n"
                f"Weighted F1: "
                f"{weighted_f1 * 100:.4f}%\n"
                f"Short-query rows: "
                f"{len(short)}\n"
                f"Short-query accuracy: "
                f"{short_accuracy * 100:.4f}%\n"
                if short_accuracy is not None
                else ""
            )

            f.write(
                f"Fallback rate: "
                f"{result['fallback'].mean() * 100:.4f}%\n"
                f"Confidence < 0.50: "
                f"{(result['confidence'] < 0.50).sum()}\n"
                f"Confidence < 0.70: "
                f"{(result['confidence'] < 0.70).sum()}\n"
                f"Confidence < 0.90: "
                f"{(result['confidence'] < 0.90).sum()}\n\n"
            )

            f.write(
                "Classification report:\n"
            )
            f.write(report)

    # ---------------------------------------------------------------
    # PRINT
    # ---------------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "V6 PRODUCTION SMOKE TEST RESULT"
    )
    print("=" * 78)

    if accuracy is not None:
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
        f"Intent coverage : "
        f"{labeled['intent'].nunique()}/57"
    )
    print(
        f"Short queries   : {len(short)}"
    )

    if short_accuracy is not None:
        print(
            f"Short accuracy  : "
            f"{short_accuracy * 100:.4f}%"
        )

    print()
    print(
        "--- CONFIDENCE ---"
    )
    print(
        f"< 0.50 : "
        f"{(result['confidence'] < 0.50).sum()}"
    )
    print(
        f"< 0.70 : "
        f"{(result['confidence'] < 0.70).sum()}"
    )
    print(
        f"< 0.90 : "
        f"{(result['confidence'] < 0.90).sum()}"
    )

    print()
    print(
        "--- FALLBACK ---"
    )
    print(
        f"Fallback count: "
        f"{result['fallback'].sum()}"
    )
    print(
        f"Fallback rate : "
        f"{result['fallback'].mean() * 100:.2f}%"
    )

    print()
    print(
        "--- TOP 20 LOW-CONFIDENCE ROWS ---"
    )

    cols = [
        "text",
        "intent",
        "prediction",
        "confidence",
        "top3",
    ]

    print(
        result.sort_values(
            "confidence"
        )[cols].head(20).to_string(
            index=False
        )
    )

    print()
    print(
        "--- TOP 20 ERRORS ---"
    )

    if len(errors):
        print(
            errors[
                [
                    "text",
                    "intent",
                    "prediction",
                    "confidence",
                    "top3",
                ]
            ].head(20).to_string(
                index=False
            )
        )
    else:
        print(
            "NONE"
        )

    print()
    print(
        "--- INFERENCE SPEED ---"
    )
    print(
        f"Embedding time : "
        f"{embedding_time:.4f} sec"
    )
    print(
        f"Classifier time: "
        f"{classifier_time:.4f} sec"
    )
    print(
        f"Rows/sec       : "
        f"{len(result) / max(embedding_time + classifier_time, 1e-9):.2f}"
    )

    print()
    print("Saved:")
    print(
        OUT_DIR
        / "production_smoke_predictions.csv"
    )
    print(
        OUT_DIR
        / "per_intent_smoke_metrics.csv"
    )
    print(
        OUT_DIR
        / "low_confidence_review.csv"
    )
    print(
        OUT_DIR
        / "smoke_errors.csv"
    )
    print(
        OUT_DIR
        / "smoke_test_summary.json"
    )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "Locked 1686-row test was NOT read."
    )
    print(
        "train.csv was NOT read."
    )
    print(
        "No synthetic text was generated."
    )
    print(
        "No threshold was tuned."
    )
    print()
    print(
        "STATUS: V6 PRODUCTION SMOKE TEST COMPLETE"
    )


if __name__ == "__main__":
    main()
