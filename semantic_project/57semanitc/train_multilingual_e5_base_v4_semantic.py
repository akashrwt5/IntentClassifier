#!/usr/bin/env python3
"""
V4 — Multilingual E5-base semantic intent model.

Chosen backbone:
    intfloat/multilingual-e5-base

Why:
- 768-dimensional embeddings
- 12 transformer layers
- multilingual model supporting 100 languages
- suitable for semantic matching / intent-style classification
- no model-size constraint for this experiment

Pipeline:
    train.csv
       ↓
    supervised contrastive fine-tuning
       ↓
    E5 embeddings
       ↓
    Logistic Regression classifier
       ↓
    validation benchmark

IMPORTANT:
- Locked 1686-row test is NOT read.
- No synthetic text.
- No labels are changed.
- No quantization.
- No ONNX.
- Existing V3 checkpoint is untouched.

Expected train.csv columns:
    text, intent
"""

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
    InputExample,
)
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

PROJECT = Path(
    "/Users/shuklam/IntentClassifier/semantic_project/57semanitc"
)

TRAIN_CSV = PROJECT / "train.csv"

OUT_DIR = (
    PROJECT
    / "v4_multilingual_e5_base_semantic"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_NAME = "intfloat/multilingual-e5-base"

SEED = 42

VAL_SIZE = 0.10

MAX_SEQ_LENGTH = 64

EPOCHS = 3

BATCH_SIZE = 16

LEARNING_RATE = 2e-5

RANDOM_STATE = 42


# ---------------------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def normalize_text(x):
    return (
        str(x)
        .strip()
        .replace("\n", " ")
        .replace("\r", " ")
    )


def add_e5_prefix(text):
    # E5 model card recommends query:/passage: prefixes.
    # For intent utterances, query: is appropriate.
    return "query: " + normalize_text(text)


def make_positive_pairs(train_df):
    """
    Build same-intent positive pairs.

    Each row is paired with another utterance from
    the same intent. This teaches semantic equivalence
    instead of only keyword matching.
    """

    rng = np.random.default_rng(SEED)

    examples = []

    for intent, group in train_df.groupby("intent"):

        texts = group["text"].astype(str).tolist()

        if len(texts) < 2:
            continue

        indices = np.arange(len(texts))
        rng.shuffle(indices)

        for i in range(len(indices)):

            a = indices[i]

            b = indices[
                (i + 1) % len(indices)
            ]

            if a == b:
                continue

            examples.append(
                InputExample(
                    texts=[
                        add_e5_prefix(texts[a]),
                        add_e5_prefix(texts[b]),
                    ]
                )
            )

    rng.shuffle(examples)

    return examples


def evaluate_classifier(
    model,
    classifier,
    label_encoder,
    texts,
    y_true,
):
    embeddings = model.encode(
        [
            add_e5_prefix(x)
            for x in texts
        ],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    y_pred = classifier.predict(
        embeddings
    )

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

    labels = list(
        range(
            len(label_encoder.classes_)
        )
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=label_encoder.classes_,
        digits=4,
        zero_division=0,
    )

    return (
        accuracy,
        macro_f1,
        weighted_f1,
        report,
        y_pred,
        embeddings,
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("=" * 72)
    print(
        "V4 MULTILINGUAL E5-BASE SEMANTIC TRAINING"
    )
    print("=" * 72)

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"train.csv not found:\n{TRAIN_CSV}"
        )

    print()
    print(f"Backbone   : {MODEL_NAME}")
    print(f"Train CSV  : {TRAIN_CSV}")
    print(f"Output     : {OUT_DIR}")

    # ---------------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------------

    df = pd.read_csv(
        TRAIN_CSV
    )

    required = {
        "text",
        "intent",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise RuntimeError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = df[
        ["text", "intent"]
    ].copy()

    df["text"] = df[
        "text"
    ].map(normalize_text)

    df["intent"] = df[
        "intent"
    ].map(normalize_text)

    df = df[
        (df["text"] != "")
        & (df["intent"] != "")
    ].reset_index(drop=True)

    num_intents = df[
        "intent"
    ].nunique()

    print()
    print(
        f"Samples    : {len(df)}"
    )
    print(
        f"Intents    : {num_intents}"
    )

    if num_intents != 57:
        raise RuntimeError(
            f"Expected 57 intents, found {num_intents}."
        )

    # ---------------------------------------------------------------
    # LABEL ENCODING
    # ---------------------------------------------------------------

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(
        df["intent"]
    )

    label_map = {
        str(i): label
        for i, label in enumerate(
            label_encoder.classes_
        )
    }

    (
        train_idx,
        val_idx,
    ) = train_test_split(
        np.arange(len(df)),
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_df = df.iloc[
        train_idx
    ].reset_index(drop=True)

    val_df = df.iloc[
        val_idx
    ].reset_index(drop=True)

    y_train = y[
        train_idx
    ]

    y_val = y[
        val_idx
    ]

    print()
    print(
        f"Train      : {len(train_df)}"
    )
    print(
        f"Validation : {len(val_df)}"
    )

    # ---------------------------------------------------------------
    # LOAD E5
    # ---------------------------------------------------------------

    print()
    print(
        "Loading multilingual-e5-base..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    model.max_seq_length = (
        MAX_SEQ_LENGTH
    )

    # ---------------------------------------------------------------
    # CONTRASTIVE TRAINING DATA
    # ---------------------------------------------------------------

    print()
    print(
        "Building same-intent semantic pairs..."
    )

    pair_examples = make_positive_pairs(
        train_df
    )

    print(
        f"Positive pairs: {len(pair_examples)}"
    )

    if len(pair_examples) < 100:
        raise RuntimeError(
            "Too few positive pairs."
        )

    # ---------------------------------------------------------------
    # CONTRASTIVE FINE-TUNING
    # ---------------------------------------------------------------

    train_dataset = {
        "sentence1": [
            ex.texts[0]
            for ex in pair_examples
        ],
        "sentence2": [
            ex.texts[1]
            for ex in pair_examples
        ],
    }

    from datasets import Dataset

    train_dataset = Dataset.from_dict(
        train_dataset
    )

    loss = (
        losses.MultipleNegativesRankingLoss(
            model
        )
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
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
        seed=SEED,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )

    print()
    print(
        "Starting semantic fine-tuning..."
    )

    trainer.train()

    # ---------------------------------------------------------------
    # SAVE FINE-TUNED ENCODER
    # ---------------------------------------------------------------

    encoder_dir = (
        OUT_DIR
        / "e5_base_finetuned"
    )

    model.save(
        str(encoder_dir)
    )

    # ---------------------------------------------------------------
    # TRAIN CLASSIFIER ON EMBEDDINGS
    # ---------------------------------------------------------------

    print()
    print(
        "Encoding train/validation data..."
    )

    train_embeddings = model.encode(
        [
            add_e5_prefix(x)
            for x in train_df["text"]
        ],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    val_embeddings = model.encode(
        [
            add_e5_prefix(x)
            for x in val_df["text"]
        ],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print()
    print(
        "Training Logistic Regression classifier..."
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

    y_pred = classifier.predict(
        val_embeddings
    )

    accuracy = accuracy_score(
        y_val,
        y_pred,
    )

    macro_f1 = f1_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_val,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_val,
        y_pred,
        labels=list(
            range(num_intents)
        ),
        target_names=label_encoder.classes_,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_val,
        y_pred,
        labels=list(
            range(num_intents)
        ),
    )

    print()
    print("=" * 72)
    print(
        "V4 MULTILINGUAL E5-BASE VALIDATION"
    )
    print("=" * 72)

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
    print(report)

    # ---------------------------------------------------------------
    # SAVE ARTIFACTS
    # ---------------------------------------------------------------

    classifier_path = (
        OUT_DIR
        / "e5_base_logistic_classifier.joblib"
    )

    joblib.dump(
        classifier,
        classifier_path,
    )

    label_map_path = (
        OUT_DIR
        / "label_map.json"
    )

    label_map_path.write_text(
        json.dumps(
            label_map,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    np.save(
        OUT_DIR
        / "train_embeddings.npy",
        train_embeddings,
    )

    np.save(
        OUT_DIR
        / "validation_embeddings.npy",
        val_embeddings,
    )

    np.save(
        OUT_DIR
        / "validation_labels.npy",
        y_val,
    )

    np.save(
        OUT_DIR
        / "validation_predictions.npy",
        y_pred,
    )

    pd.DataFrame(
        cm,
        index=label_encoder.classes_,
        columns=label_encoder.classes_,
    ).to_csv(
        OUT_DIR
        / "confusion_matrix.csv"
    )

    validation_report = (
        "# V4 MULTILINGUAL E5-BASE VALIDATION\n\n"
        f"Backbone      : {MODEL_NAME}\n"
        f"Samples       : {len(df)}\n"
        f"Intents       : {num_intents}\n"
        f"Train         : {len(train_df)}\n"
        f"Validation    : {len(val_df)}\n"
        f"Accuracy      : {accuracy * 100:.4f}%\n"
        f"Macro F1      : {macro_f1 * 100:.4f}%\n"
        f"Weighted F1   : {weighted_f1 * 100:.4f}%\n\n"
        "Classification report:\n"
        f"{report}"
    )

    (
        OUT_DIR
        / "validation_report.txt"
    ).write_text(
        validation_report,
        encoding="utf-8",
    )

    summary = {
        "backbone": MODEL_NAME,
        "samples": int(len(df)),
        "num_intents": int(num_intents),
        "train_samples": int(len(train_df)),
        "validation_samples": int(len(val_df)),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_seq_length": MAX_SEQ_LENGTH,
        "contrastive_pairs": len(pair_examples),
        "locked_test_used": False,
        "quantization": False,
        "onnx": False,
        "synthetic_text": False,
        "labels_changed": False,
    }

    (
        OUT_DIR
        / "training_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Saved:")
    print(
        encoder_dir
    )
    print(
        classifier_path
    )
    print(
        label_map_path
    )
    print(
        OUT_DIR / "train_embeddings.npy"
    )
    print(
        OUT_DIR / "validation_embeddings.npy"
    )
    print(
        OUT_DIR / "validation_labels.npy"
    )
    print(
        OUT_DIR / "validation_predictions.npy"
    )
    print(
        OUT_DIR / "confusion_matrix.csv"
    )
    print(
        OUT_DIR / "validation_report.txt"
    )
    print(
        OUT_DIR / "training_summary.json"
    )

    print()
    print(
        "STATUS: "
        "V4 MULTILINGUAL E5-BASE SEMANTIC TRAINING COMPLETE"
    )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "Locked 57-intent test: NOT USED"
    )
    print(
        "Quantization: NO"
    )
    print(
        "ONNX: NO"
    )
    print(
        "Synthetic text: NO"
    )
    print(
        "Labels changed: NO"
    )


if __name__ == "__main__":
    main()
