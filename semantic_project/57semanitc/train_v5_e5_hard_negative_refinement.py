#!/usr/bin/env python3
"""
V5 — English-only E5 semantic hard-negative refinement.

Goal:
    Improve real-world English intent boundaries without changing the
    57-intent label set.

Starting point:
    V4 multilingual-e5-base fine-tuned encoder.

Data:
    - train.csv: original labeled training data
    - V3 hard_negative_examples.csv: existing controlled hard-negative data
    - canonical locked_test_57intent.csv: READ ONLY for leakage exclusion;
      its text is never used as training/validation data.

Training:
    - supervised triplet refinement:
        anchor = hard/normal English utterance
        positive = another utterance with the same intent
        negative = utterance from a different intent
    - V4 encoder is fine-tuned conservatively
    - Logistic Regression is retrained on V5 embeddings
    - validation is held out from train.csv
    - hard-negative evaluation is held out from hard-negative examples

No:
    - multilingual data
    - synthetic text generation
    - label changes
    - locked-test training
    - quantization
    - ONNX
"""

from pathlib import Path
import json
import random
import time

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
    confusion_matrix,
)

import joblib


SEED = 20260809
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT = Path(
    "/Users/shuklam/IntentClassifier/semantic_project/57semanitc"
)

V4_DIR = PROJECT / "v4_multilingual_e5_base_semantic"

V4_ENCODER = V4_DIR / "e5_base_finetuned"

TRAIN_CSV = PROJECT / "train.csv"

LOCKED_CSV = (
    PROJECT
    / "v3_57intent_locked_eval"
    / "locked_test_57intent.csv"
)

HARD_CSV = (
    PROJECT
    / "v3_57intent_e5_distilled_v3_hard_negative"
    / "hard_negative_examples.csv"
)

OUT_DIR = PROJECT / "v5_e5_english_hard_negative_refinement"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEQ_LENGTH = 64
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 1e-5

EXPECTED_INTENTS = 57


def clean(x):
    return (
        str(x)
        .strip()
        .replace("\n", " ")
        .replace("\r", " ")
    )


def e5(x):
    return "query: " + clean(x)


def find_text_label_columns(df):
    text_candidates = ["text", "utterance", "query", "sentence"]
    label_candidates = ["intent", "label", "expected_intent", "true_intent"]

    text_col = next(
        (c for c in text_candidates if c in df.columns),
        None,
    )
    label_col = next(
        (c for c in label_candidates if c in df.columns),
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

    text_col, label_col = find_text_label_columns(df)

    out = df[[text_col, label_col]].copy()
    out.columns = ["text", "intent"]

    out["text"] = out["text"].map(clean)
    out["intent"] = out["intent"].map(clean)

    out = out[
        (out["text"] != "")
        & (out["intent"] != "")
    ]

    return out.drop_duplicates("text").reset_index(drop=True)


def make_positive_pairs(df, per_row=1):
    rng = np.random.default_rng(SEED)
    examples = []

    for intent, group in df.groupby("intent"):
        texts = group["text"].tolist()

        if len(texts) < 2:
            continue

        for i, anchor in enumerate(texts):
            for _ in range(per_row):
                j = int(rng.integers(0, len(texts)))
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

    rng.shuffle(examples)
    return examples


def main():
    print("=" * 78)
    print("V5 ENGLISH E5 HARD-NEGATIVE REFINEMENT")
    print("=" * 78)

    for path in [
        V4_ENCODER,
        TRAIN_CSV,
        LOCKED_CSV,
        HARD_CSV,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required path not found:\n{path}"
            )

    # ---------------------------------------------------------------
    # LOAD TRAINING DATA
    # ---------------------------------------------------------------

    train_all = load_labeled_csv(TRAIN_CSV)

    if train_all["intent"].nunique() != EXPECTED_INTENTS:
        raise RuntimeError(
            f"train.csv has {train_all['intent'].nunique()} intents; "
            f"expected {EXPECTED_INTENTS}."
        )

    # ---------------------------------------------------------------
    # LEAKAGE GUARD
    # ---------------------------------------------------------------

    locked = load_labeled_csv(LOCKED_CSV)

    locked_texts = set(locked["text"])

    before = len(train_all)

    train_all = train_all[
        ~train_all["text"].isin(locked_texts)
    ].copy()

    removed = before - len(train_all)

    print()
    print(f"Original train rows        : {before}")
    print(f"Locked-test overlaps removed: {removed}")
    print(f"Training rows after guard   : {len(train_all)}")

    # ---------------------------------------------------------------
    # HOLD OUT VALIDATION FROM NORMAL TRAIN DATA
    # ---------------------------------------------------------------

    train_df, val_df = train_test_split(
        train_all,
        test_size=0.10,
        random_state=SEED,
        stratify=train_all["intent"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # ---------------------------------------------------------------
    # HARD NEGATIVES
    # ---------------------------------------------------------------

    hard = load_labeled_csv(HARD_CSV)

    # Never train on locked-test utterances.
    hard = hard[
        ~hard["text"].isin(locked_texts)
    ].copy()

    # Never let exact duplicates cross into the normal source.
    train_df = train_df[
        ~train_df["text"].isin(set(hard["text"]))
    ].copy()

    if hard["intent"].nunique() != EXPECTED_INTENTS:
        print(
            "WARNING: hard-negative file does not contain all 57 intents."
        )

    # Reserve 20% of hard negatives for targeted evaluation.
    hard_train, hard_eval = train_test_split(
        hard,
        test_size=0.20,
        random_state=SEED,
        stratify=hard["intent"],
    )

    hard_train = hard_train.reset_index(drop=True)
    hard_eval = hard_eval.reset_index(drop=True)

    print()
    print(f"Normal train rows  : {len(train_df)}")
    print(f"Validation rows    : {len(val_df)}")
    print(f"Hard train rows    : {len(hard_train)}")
    print(f"Hard eval rows     : {len(hard_eval)}")
    print(f"Intents            : {train_all['intent'].nunique()}")

    # ---------------------------------------------------------------
    # POSITIVE PAIR DATASET
    # ---------------------------------------------------------------

    # Use both normal training data and hard training data.
    pair_source = pd.concat(
        [train_df, hard_train],
        ignore_index=True,
    ).drop_duplicates("text")

    pairs = make_positive_pairs(
        pair_source,
        per_row=1,
    )

    if len(pairs) < 100:
        raise RuntimeError(
            f"Only {len(pairs)} positive pairs were built."
        )

    print()
    print(f"Contrastive positive pairs: {len(pairs)}")

    pair_dataset = Dataset.from_dict(
        {
            "sentence1": [
                x.texts[0] for x in pairs
            ],
            "sentence2": [
                x.texts[1] for x in pairs
            ],
        }
    )

    # ---------------------------------------------------------------
    # LOAD V4
    # ---------------------------------------------------------------

    print()
    print("Loading V4 multilingual-e5-base encoder...")

    model = SentenceTransformer(
        str(V4_ENCODER)
    )

    model.max_seq_length = MAX_SEQ_LENGTH

    # ---------------------------------------------------------------
    # FINE-TUNE
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
    print("Starting V5 conservative refinement...")
    trainer.train()

    encoder_out = OUT_DIR / "e5_base_v5_finetuned"
    model.save(str(encoder_out))

    # ---------------------------------------------------------------
    # CLASSIFIER TRAINING
    # ---------------------------------------------------------------

    print()
    print("Encoding V5 training data...")

    train_embeddings = model.encode(
        [
            e5(x)
            for x in train_df["text"]
        ],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    val_embeddings = model.encode(
        [
            e5(x)
            for x in val_df["text"]
        ],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print()
    print("Training Logistic Regression...")

    labels = sorted(
        train_all["intent"].unique()
    )

    label_to_id = {
        label: i
        for i, label in enumerate(labels)
    }

    y_train = np.array([
        label_to_id[x]
        for x in train_df["intent"]
    ])

    y_val = np.array([
        label_to_id[x]
        for x in val_df["intent"]
    ])

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
    # NORMAL VALIDATION
    # ---------------------------------------------------------------

    val_pred = classifier.predict(
        val_embeddings
    )

    val_acc = accuracy_score(
        y_val,
        val_pred,
    )

    val_macro = f1_score(
        y_val,
        val_pred,
        average="macro",
        zero_division=0,
    )

    val_weighted = f1_score(
        y_val,
        val_pred,
        average="weighted",
        zero_division=0,
    )

    # ---------------------------------------------------------------
    # HARD-NEGATIVE EVALUATION
    # ---------------------------------------------------------------

    print()
    print("Evaluating held-out hard negatives...")

    hard_embeddings = model.encode(
        [
            e5(x)
            for x in hard_eval["text"]
        ],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    hard_pred = classifier.predict(
        hard_embeddings
    )

    y_hard = np.array([
        label_to_id[x]
        for x in hard_eval["intent"]
    ])

    hard_acc = accuracy_score(
        y_hard,
        hard_pred,
    )

    hard_macro = f1_score(
        y_hard,
        hard_pred,
        average="macro",
        zero_division=0,
    )

    # ---------------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------------

    classifier_path = (
        OUT_DIR
        / "e5_base_v5_logistic_classifier.joblib"
    )

    joblib.dump(
        classifier,
        classifier_path,
    )

    label_map = {
        str(i): label
        for label, i in label_to_id.items()
    }

    label_map_path = (
        OUT_DIR / "label_map.json"
    )

    label_map_path.write_text(
        json.dumps(
            label_map,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    val_pred_labels = [
        labels[int(i)]
        for i in val_pred
    ]

    hard_pred_labels = [
        labels[int(i)]
        for i in hard_pred
    ]

    val_predictions = val_df.copy()
    val_predictions["prediction"] = val_pred_labels
    val_predictions["correct"] = (
        val_predictions["intent"]
        == val_predictions["prediction"]
    )

    hard_predictions = hard_eval.copy()
    hard_predictions["prediction"] = hard_pred_labels
    hard_predictions["correct"] = (
        hard_predictions["intent"]
        == hard_predictions["prediction"]
    )

    val_predictions.to_csv(
        OUT_DIR / "validation_predictions.csv",
        index=False,
    )

    hard_predictions.to_csv(
        OUT_DIR / "hard_negative_predictions.csv",
        index=False,
    )

    report = classification_report(
        y_val,
        val_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        digits=4,
        zero_division=0,
    )

    (OUT_DIR / "validation_report.txt").write_text(
        (
            "# V5 ENGLISH E5 HARD-NEGATIVE REFINEMENT\n\n"
            f"Validation Accuracy   : {val_acc * 100:.4f}%\n"
            f"Validation Macro F1   : {val_macro * 100:.4f}%\n"
            f"Validation Weighted F1: {val_weighted * 100:.4f}%\n"
            f"Hard-Negative Accuracy: {hard_acc * 100:.4f}%\n"
            f"Hard-Negative Macro F1: {hard_macro * 100:.4f}%\n\n"
            "Classification report:\n"
            f"{report}"
        ),
        encoding="utf-8",
    )

    summary = {
        "version": "v5_english_hard_negative_refinement",
        "base_encoder": str(V4_ENCODER),
        "output_encoder": str(encoder_out),
        "train_csv": str(TRAIN_CSV),
        "hard_negative_csv": str(HARD_CSV),
        "locked_csv_read_for_leakage_guard_only": str(LOCKED_CSV),
        "locked_test_rows_used_for_training": 0,
        "labels_changed": False,
        "synthetic_text": False,
        "multilingual_training": False,
        "quantization": False,
        "onnx": False,
        "intents": len(labels),
        "normal_train_rows": len(train_df),
        "validation_rows": len(val_df),
        "hard_train_rows": len(hard_train),
        "hard_eval_rows": len(hard_eval),
        "positive_pairs": len(pairs),
        "validation_accuracy": float(val_acc),
        "validation_macro_f1": float(val_macro),
        "validation_weighted_f1": float(val_weighted),
        "hard_negative_accuracy": float(hard_acc),
        "hard_negative_macro_f1": float(hard_macro),
    }

    (OUT_DIR / "training_summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("V5 ENGLISH HARD-NEGATIVE REFINEMENT RESULT")
    print("=" * 78)
    print(
        f"Validation Accuracy   : {val_acc * 100:.4f}%"
    )
    print(
        f"Validation Macro F1   : {val_macro * 100:.4f}%"
    )
    print(
        f"Validation Weighted F1: {val_weighted * 100:.4f}%"
    )
    print(
        f"Hard-Negative Accuracy: {hard_acc * 100:.4f}%"
    )
    print(
        f"Hard-Negative Macro F1: {hard_macro * 100:.4f}%"
    )

    print()
    print("Saved:")
    print(encoder_out)
    print(classifier_path)
    print(label_map_path)
    print(OUT_DIR / "validation_predictions.csv")
    print(OUT_DIR / "hard_negative_predictions.csv")
    print(OUT_DIR / "validation_report.txt")
    print(OUT_DIR / "training_summary.json")

    print()
    print("IMPORTANT:")
    print("English-only refinement.")
    print("Locked test was NOT used for training.")
    print("Locked texts were used only for leakage exclusion.")
    print("No labels changed.")
    print("No synthetic text generated.")
    print("No quantization.")
    print("No ONNX.")

    print()
    print(
        "NEXT STEP: if V5 improves hard-negative performance "
        "without materially hurting validation, benchmark V5 "
        "on the exact 1686-row locked test."
    )


if __name__ == "__main__":
    main()
