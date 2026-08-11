#!/usr/bin/env python3
"""
V6 — English E5 production-vocabulary refinement.

Base:
    V4 multilingual-e5-base fine-tuned encoder.

Adds:
    - original train.csv
    - approved/reviewed production vocabulary candidates from:
      v6_english_vocab_review/candidate_vocab_review.csv
    - existing V5 hard-negative examples when available

Important:
    - English only
    - 57 intents unchanged
    - canonical 1686-row locked test is NEVER used for training/validation
    - locked-test texts are used ONLY for exact-text leakage exclusion
    - no synthetic generation
    - no quantization
    - no ONNX
    - V4 is never overwritten

The candidate file is consumed only from rows with:
    status == REVIEW_REQUIRED

Run after reviewing candidate_vocab_review.csv.
"""

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers import losses

from datasets import Dataset

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

import joblib


SEED = 20260809
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT = Path(
    "/Users/shuklam/IntentClassifier/semantic_project/57semanitc"
)

V4_ENCODER = (
    PROJECT
    / "v4_multilingual_e5_base_semantic"
    / "e5_base_finetuned"
)

TRAIN_CSV = PROJECT / "train.csv"

VOCAB_CSV = (
    PROJECT
    / "v6_english_vocab_review"
    / "candidate_vocab_review.csv"
)

HARD_CSV = (
    PROJECT
    / "v3_57intent_e5_distilled_v3_hard_negative"
    / "hard_negative_examples.csv"
)

LOCKED_CSV = (
    PROJECT
    / "v3_57intent_locked_eval"
    / "locked_test_57intent.csv"
)

OUT_DIR = (
    PROJECT
    / "v6_e5_english_production_vocab"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EXPECTED_INTENTS = 57

MAX_SEQ_LENGTH = 64
BATCH_SIZE = 16
CLASSIFIER_BATCH_SIZE = 64

EPOCHS = 2
LEARNING_RATE = 7e-6

# Production candidates are intentionally repeated in the contrastive
# training source so the model pays attention to newly covered phrasing.
CANDIDATE_PAIR_MULTIPLIER = 3


def clean(x):
    return (
        str(x)
        .strip()
        .replace("\n", " ")
        .replace("\r", " ")
    )


def norm(x):
    return " ".join(
        clean(x).lower().split()
    )


def e5(x):
    return "query: " + clean(x)


def find_text_label_columns(df):
    text_candidates = [
        "text",
        "utterance",
        "query",
        "sentence",
    ]

    label_candidates = [
        "intent",
        "label",
        "expected_intent",
        "true_intent",
    ]

    text_col = next(
        (
            c for c in text_candidates
            if c in df.columns
        ),
        None,
    )

    label_col = next(
        (
            c for c in label_candidates
            if c in df.columns
        ),
        None,
    )

    if text_col is None or label_col is None:
        raise RuntimeError(
            "Could not identify text/intent columns. "
            f"Found columns: {list(df.columns)}"
        )

    return text_col, label_col


def load_labeled_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    text_col, label_col = find_text_label_columns(
        df
    )

    out = df[
        [text_col, label_col]
    ].copy()

    out.columns = [
        "text",
        "intent",
    ]

    out["text"] = out[
        "text"
    ].map(clean)

    out["intent"] = out[
        "intent"
    ].map(clean)

    out = out[
        (out["text"] != "")
        & (out["intent"] != "")
    ]

    out["text_norm"] = out[
        "text"
    ].map(norm)

    return out.drop_duplicates(
        "text_norm"
    ).reset_index(drop=True)


def load_vocab_candidates(path):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {
        "intent",
        "text",
        "status",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise RuntimeError(
            "Vocabulary candidate CSV is missing: "
            f"{sorted(missing)}"
        )

    # Only explicitly reviewed candidate rows.
    df = df[
        df["status"].astype(str)
        == "REVIEW_REQUIRED"
    ].copy()

    df["text"] = df[
        "text"
    ].map(clean)

    df["intent"] = df[
        "intent"
    ].map(clean)

    df["text_norm"] = df[
        "text"
    ].map(norm)

    df = df[
        (df["text_norm"] != "")
        & (df["intent"] != "")
    ]

    return df[
        [
            "text",
            "intent",
            "text_norm",
        ]
    ].drop_duplicates(
        "text_norm"
    ).reset_index(drop=True)


def make_positive_pairs(
    normal_df,
    candidate_df,
    hard_df=None,
):
    rng = np.random.default_rng(
        SEED
    )

    examples = []

    # Normal training data.
    for intent, group in normal_df.groupby(
        "intent"
    ):
        texts = group[
            "text"
        ].tolist()

        if len(texts) < 2:
            continue

        for i, anchor in enumerate(texts):
            j = int(
                rng.integers(
                    0,
                    len(texts),
                )
            )

            if texts[j] == anchor:
                j = (j + 1) % len(texts)

            examples.append(
                InputExample(
                    texts=[
                        e5(anchor),
                        e5(texts[j]),
                    ]
                )
            )

    # Production candidates get multiple pair opportunities.
    if len(candidate_df):
        combined = pd.concat(
            [
                normal_df,
                candidate_df[
                    [
                        "text",
                        "intent",
                        "text_norm",
                    ]
                ],
            ],
            ignore_index=True,
        ).drop_duplicates(
            "text_norm"
        )

        for intent, group in combined.groupby(
            "intent"
        ):
            base_texts = normal_df[
                normal_df["intent"] == intent
            ]["text"].tolist()

            if len(base_texts) == 0:
                continue

            candidates = candidate_df[
                candidate_df["intent"] == intent
            ]["text"].tolist()

            for anchor in candidates:
                for _ in range(
                    CANDIDATE_PAIR_MULTIPLIER
                ):
                    positive = base_texts[
                        int(
                            rng.integers(
                                0,
                                len(base_texts),
                            )
                        )
                    ]

                    if positive == anchor:
                        continue

                    examples.append(
                        InputExample(
                            texts=[
                                e5(anchor),
                                e5(positive),
                            ]
                        )
                    )

    # Existing hard-negative data contributes same-intent semantic pairs.
    if hard_df is not None and len(hard_df):
        for intent, group in hard_df.groupby(
            "intent"
        ):
            texts = group[
                "text"
            ].tolist()

            if len(texts) < 2:
                continue

            for anchor in texts:
                positive = texts[
                    int(
                        rng.integers(
                            0,
                            len(texts),
                        )
                    )
                ]

                if positive == anchor:
                    continue

                examples.append(
                    InputExample(
                        texts=[
                            e5(anchor),
                            e5(positive),
                        ]
                    )
                )

    rng.shuffle(
        examples
    )

    return examples


def evaluate(
    classifier,
    embeddings,
    labels,
    y_true,
):
    pred = classifier.predict(
        embeddings
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                pred,
                labels=list(
                    range(len(labels))
                ),
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                pred,
                labels=list(
                    range(len(labels))
                ),
                average="weighted",
                zero_division=0,
            )
        ),
        "pred": pred,
    }


def main():

    print("=" * 78)
    print(
        "V6 ENGLISH E5 PRODUCTION-VOCABULARY REFINEMENT"
    )
    print("=" * 78)

    for path in [
        V4_ENCODER,
        TRAIN_CSV,
        VOCAB_CSV,
        LOCKED_CSV,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required path not found:\n{path}"
            )

    # ---------------------------------------------------------------
    # LOAD ORIGINAL TRAIN
    # ---------------------------------------------------------------

    train_all = load_labeled_csv(
        TRAIN_CSV
    )

    if train_all[
        "intent"
    ].nunique() != EXPECTED_INTENTS:
        raise RuntimeError(
            "train.csv intent count mismatch: "
            f"{train_all['intent'].nunique()}"
        )

    # ---------------------------------------------------------------
    # LOCKED TEST = LEAKAGE GUARD ONLY
    # ---------------------------------------------------------------

    locked = load_labeled_csv(
        LOCKED_CSV
    )

    locked_texts = set(
        locked["text_norm"]
    )

    before = len(train_all)

    train_all = train_all[
        ~train_all[
            "text_norm"
        ].isin(locked_texts)
    ].copy()

    removed = before - len(
        train_all
    )

    # ---------------------------------------------------------------
    # LOAD PRODUCTION VOCABULARY
    # ---------------------------------------------------------------

    candidates = load_vocab_candidates(
        VOCAB_CSV
    )

    unknown_candidate_intents = sorted(
        set(
            candidates["intent"]
        )
        - set(
            train_all["intent"]
        )
    )

    if unknown_candidate_intents:
        raise RuntimeError(
            "Candidate vocabulary contains "
            "unknown intents:\n"
            + "\n".join(
                unknown_candidate_intents
            )
        )

    # Candidate leakage guard.
    candidates = candidates[
        ~candidates[
            "text_norm"
        ].isin(locked_texts)
    ].copy()

    # Do not duplicate text already in original training.
    original_train_texts = set(
        train_all["text_norm"]
    )

    candidates = candidates[
        ~candidates[
            "text_norm"
        ].isin(original_train_texts)
    ].copy()

    # ---------------------------------------------------------------
    # HARD NEGATIVES
    # ---------------------------------------------------------------

    hard = None

    if HARD_CSV.exists():
        hard = load_labeled_csv(
            HARD_CSV
        )

        hard = hard[
            ~hard[
                "text_norm"
            ].isin(locked_texts)
        ].copy()

        # Remove duplicates with normal training and candidates.
        occupied = set(
            train_all["text_norm"]
        ) | set(
            candidates["text_norm"]
        )

        hard = hard[
            ~hard[
                "text_norm"
            ].isin(occupied)
        ].copy()

    print()
    print(
        f"Original train rows          : {before}"
    )
    print(
        f"Locked overlaps removed      : {removed}"
    )
    print(
        f"Base train rows              : {len(train_all)}"
    )
    print(
        f"New production candidates    : {len(candidates)}"
    )
    print(
        f"Hard-negative rows            : "
        f"{len(hard) if hard is not None else 0}"
    )
    print(
        f"Intents                       : "
        f"{train_all['intent'].nunique()}"
    )

    # ---------------------------------------------------------------
    # NORMAL VALIDATION
    #
    # Validation is from original train.csv only.
    # Production candidates are never used as validation.
    # ---------------------------------------------------------------

    train_df, val_df = train_test_split(
        train_all,
        test_size=0.10,
        random_state=SEED,
        stratify=train_all[
            "intent"
        ],
    )

    train_df = train_df.reset_index(
        drop=True
    )

    val_df = val_df.reset_index(
        drop=True
    )

    # Candidates are added to training after validation split.
    train_plus_candidates = pd.concat(
        [
            train_df,
            candidates[
                [
                    "text",
                    "intent",
                    "text_norm",
                ]
            ],
        ],
        ignore_index=True,
    ).drop_duplicates(
        "text_norm"
    )

    # Add hard-negative training rows if available.
    if hard is not None and len(hard):
        train_plus_candidates = pd.concat(
            [
                train_plus_candidates,
                hard[
                    [
                        "text",
                        "intent",
                        "text_norm",
                    ]
                ],
            ],
            ignore_index=True,
        ).drop_duplicates(
            "text_norm"
        )

    # ---------------------------------------------------------------
    # POSITIVE PAIRS
    # ---------------------------------------------------------------

    pairs = make_positive_pairs(
        train_df,
        candidates,
        hard_df=hard,
    )

    if len(pairs) < 100:
        raise RuntimeError(
            f"Too few training pairs: {len(pairs)}"
        )

    print()
    print(
        f"Contrastive positive pairs: {len(pairs)}"
    )

    pair_dataset = Dataset.from_dict(
        {
            "sentence1": [
                x.texts[0]
                for x in pairs
            ],
            "sentence2": [
                x.texts[1]
                for x in pairs
            ],
        }
    )

    # ---------------------------------------------------------------
    # LOAD V4
    # ---------------------------------------------------------------

    print()
    print(
        "Loading V4 multilingual-e5-base..."
    )

    model = SentenceTransformer(
        str(V4_ENCODER)
    )

    model.max_seq_length = (
        MAX_SEQ_LENGTH
    )

    # ---------------------------------------------------------------
    # CONSERVATIVE FINE-TUNING
    # ---------------------------------------------------------------

    loss = losses.MultipleNegativesRankingLoss(
        model
    )

    args = SentenceTransformerTrainingArguments(
        output_dir=str(
            OUT_DIR / "trainer_output"
        ),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.10,
        fp16=False,
        bf16=False,
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
        seed=SEED,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=pair_dataset,
        loss=loss,
    )

    print()
    print(
        "Starting V6 conservative refinement..."
    )

    trainer.train()

    encoder_out = (
        OUT_DIR
        / "e5_base_v6_finetuned"
    )

    model.save(
        str(encoder_out)
    )

    # ---------------------------------------------------------------
    # CLASSIFIER DATA
    # ---------------------------------------------------------------

    print()
    print(
        "Encoding V6 classifier training data..."
    )

    train_embeddings = model.encode(
        [
            e5(x)
            for x in train_plus_candidates[
                "text"
            ]
        ],
        batch_size=CLASSIFIER_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    val_embeddings = model.encode(
        [
            e5(x)
            for x in val_df["text"]
        ],
        batch_size=CLASSIFIER_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    labels = sorted(
        train_all[
            "intent"
        ].unique()
    )

    label_to_id = {
        label: i
        for i, label in enumerate(
            labels
        )
    }

    y_train = np.array([
        label_to_id[x]
        for x in train_plus_candidates[
            "intent"
        ]
    ])

    y_val = np.array([
        label_to_id[x]
        for x in val_df[
            "intent"
        ]
    ])

    print()
    print(
        "Training Logistic Regression..."
    )

    classifier = LogisticRegression(
        max_iter=3000,
        C=4.0,
        solver="lbfgs",
    )

    classifier.fit(
        train_embeddings,
        y_train,
    )

    # ---------------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------------

    val_result = evaluate(
        classifier,
        val_embeddings,
        labels,
        y_val,
    )

    val_pred = val_result[
        "pred"
    ]

    report = classification_report(
        y_val,
        val_pred,
        labels=list(
            range(len(labels))
        ),
        target_names=labels,
        digits=4,
        zero_division=0,
    )

    # ---------------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------------

    classifier_path = (
        OUT_DIR
        / "e5_base_v6_logistic_classifier.joblib"
    )

    label_map_path = (
        OUT_DIR / "label_map.json"
    )

    validation_predictions_path = (
        OUT_DIR
        / "validation_predictions.csv"
    )

    report_path = (
        OUT_DIR
        / "validation_report.txt"
    )

    summary_path = (
        OUT_DIR
        / "training_summary.json"
    )

    joblib.dump(
        classifier,
        classifier_path,
    )

    label_map_path.write_text(
        json.dumps(
            {
                str(i): label
                for i, label in enumerate(
                    labels
                )
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    val_output = val_df[
        [
            "text",
            "intent",
        ]
    ].copy()

    val_output[
        "prediction"
    ] = [
        labels[int(x)]
        for x in val_pred
    ]

    val_output[
        "correct"
    ] = (
        val_output[
            "intent"
        ]
        == val_output[
            "prediction"
        ]
    )

    val_output.to_csv(
        validation_predictions_path,
        index=False,
    )

    report_path.write_text(
        "# V6 ENGLISH PRODUCTION VOCABULARY REFINEMENT\n\n"
        f"Validation Accuracy   : "
        f"{val_result['accuracy'] * 100:.4f}%\n"
        f"Validation Macro F1   : "
        f"{val_result['macro_f1'] * 100:.4f}%\n"
        f"Validation Weighted F1: "
        f"{val_result['weighted_f1'] * 100:.4f}%\n\n"
        "Classification report:\n"
        f"{report}\n",
        encoding="utf-8",
    )

    summary = {
        "model": "multilingual-e5-base",
        "base_encoder": str(V4_ENCODER),
        "output_encoder": str(encoder_out),
        "training_csv": str(TRAIN_CSV),
        "vocabulary_csv": str(VOCAB_CSV),
        "hard_negative_csv": (
            str(HARD_CSV)
            if HARD_CSV.exists()
            else None
        ),
        "locked_csv": str(LOCKED_CSV),
        "base_train_rows": int(
            len(train_df)
        ),
        "production_candidates_added": int(
            len(candidates)
        ),
        "hard_negative_rows_added": int(
            len(hard)
            if hard is not None
            else 0
        ),
        "classifier_training_rows": int(
            len(train_plus_candidates)
        ),
        "validation_rows": int(
            len(val_df)
        ),
        "num_intents": int(
            len(labels)
        ),
        "validation_accuracy": float(
            val_result["accuracy"]
        ),
        "validation_macro_f1": float(
            val_result["macro_f1"]
        ),
        "validation_weighted_f1": float(
            val_result["weighted_f1"]
        ),
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "candidate_pair_multiplier": (
            CANDIDATE_PAIR_MULTIPLIER
        ),
        "locked_test_used_for_training": False,
        "locked_test_used_for_validation": False,
        "locked_texts_used_for_leakage_guard": True,
        "synthetic_text": False,
        "multilingual_data": False,
        "quantization": False,
        "onnx": False,
        "v4_overwritten": False,
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(
        "V6 ENGLISH PRODUCTION VOCABULARY "
        "REFINEMENT RESULT"
    )
    print("=" * 78)

    print(
        f"Validation Accuracy   : "
        f"{val_result['accuracy'] * 100:.4f}%"
    )

    print(
        f"Validation Macro F1   : "
        f"{val_result['macro_f1'] * 100:.4f}%"
    )

    print(
        f"Validation Weighted F1: "
        f"{val_result['weighted_f1'] * 100:.4f}%"
    )

    print()
    print(
        f"Production candidates : {len(candidates)}"
    )

    print(
        f"Hard-negative rows    : "
        f"{len(hard) if hard is not None else 0}"
    )

    print()
    print("Saved:")
    print(encoder_out)
    print(classifier_path)
    print(label_map_path)
    print(validation_predictions_path)
    print(report_path)
    print(summary_path)

    print()
    print(
        "STATUS: V6 ENGLISH PRODUCTION "
        "VOCABULARY TRAINING COMPLETE"
    )

    print()
    print(
        "IMPORTANT: V6 has NOT been tested on the "
        "locked 1686-row benchmark yet."
    )


if __name__ == "__main__":
    main()
