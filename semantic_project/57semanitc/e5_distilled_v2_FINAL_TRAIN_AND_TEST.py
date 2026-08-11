#!/usr/bin/env python3
"""
E5-small -> tiny student V2
FINAL TRAIN + EXACT LOCKED TEST PIPELINE

Training data:
  /Users/shuklam/IntentClassifier/semantic_project/57semanitc/train.csv

Locked test:
  /Users/shuklam/IntentClassifier/semantic_project/57semanitc/
  v3_57intent_locked_eval/locked_test_57intent.csv

IMPORTANT:
- Locked test is NEVER used during training.
- No quantization.
- No ONNX.
- No labels are changed.
- E5-small-v2 is used ONLY during training as the teacher.
- The saved student is a small standalone PyTorch FP32 model.
"""

from pathlib import Path
import json
import re
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT = Path(
    "/Users/shuklam/IntentClassifier/semantic_project/57semanitc"
)

TRAIN_CSV = PROJECT / "train.csv"

LOCKED_CSV = (
    PROJECT
    / "v3_57intent_locked_eval"
    / "locked_test_57intent.csv"
)

OUT_DIR = (
    PROJECT
    / "v3_57intent_e5_distilled_v2_FINAL"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT = (
    OUT_DIR
    / "student_e5_distilled_v2_best_fp32.pt"
)

VOCAB_JSON = (
    OUT_DIR
    / "vocab.json"
)

LABEL_MAP_JSON = (
    OUT_DIR
    / "label_map.json"
)

TRAIN_EMB_NPY = (
    OUT_DIR
    / "teacher_train_embeddings.npy"
)

TEACHER_JSON = (
    OUT_DIR
    / "teacher_summary.json"
)

HISTORY_CSV = (
    OUT_DIR
    / "training_history.csv"
)

VALIDATION_REPORT = (
    OUT_DIR
    / "validation_report.txt"
)

LOCKED_REPORT = (
    OUT_DIR
    / "locked_test_report.txt"
)

LOCKED_PREDICTIONS = (
    OUT_DIR
    / "locked_predictions.csv"
)

LOCKED_CM = (
    OUT_DIR
    / "locked_confusion_matrix.csv"
)

SUMMARY_JSON = (
    OUT_DIR
    / "final_summary.json"
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

MAX_LEN = 24

PAD_ID = 0
UNK_ID = 1

EMBED_DIM = 64
NHEAD = 4
FF_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.10

BATCH_SIZE = 128
EPOCHS = 40
PATIENCE = 7

LR = 2e-3
WEIGHT_DECAY = 1e-4

TEMPERATURE = 2.0

# CE weight. Higher = more emphasis on true labels.
CE_WEIGHT = 0.70

# KL/distillation weight.
KD_WEIGHT = 0.30

VAL_SIZE = 0.15

E5_MODEL_NAME = "intfloat/e5-small-v2"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# TOKENIZER / VOCAB
# ============================================================

def basic_tokens(text):
    text = str(text).lower().strip()

    # Keep words and apostrophes.
    return re.findall(
        r"[a-z0-9]+(?:'[a-z0-9]+)?",
        text,
    )


def build_vocab(texts, min_freq=1):
    counts = {}

    for text in texts:
        for token in basic_tokens(text):
            counts[token] = counts.get(token, 0) + 1

    vocab = {
        "<pad>": PAD_ID,
        "<unk>": UNK_ID,
    }

    next_id = 2

    for token in sorted(counts):
        if counts[token] >= min_freq:
            vocab[token] = next_id
            next_id += 1

    return vocab


def encode_text(text, vocab):
    tokens = basic_tokens(text)[:MAX_LEN]

    ids = [
        vocab.get(token, UNK_ID)
        for token in tokens
    ]

    ids += [
        PAD_ID
    ] * (MAX_LEN - len(ids))

    return ids


def encode_texts(texts, vocab):
    return np.asarray(
        [
            encode_text(text, vocab)
            for text in texts
        ],
        dtype=np.int64,
    )


# ============================================================
# STUDENT MODEL
# ============================================================

class TinyIntentClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        num_classes,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            EMBED_DIM,
            padding_idx=PAD_ID,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NHEAD,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_LAYERS,
        )

        self.norm = nn.LayerNorm(
            EMBED_DIM
        )

        self.classifier = nn.Linear(
            EMBED_DIM,
            num_classes,
        )

    def forward(self, input_ids):

        x = self.embedding(
            input_ids
        )

        padding_mask = input_ids.eq(
            PAD_ID
        )

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        valid = (
            ~padding_mask
        ).unsqueeze(-1).float()

        denom = valid.sum(
            dim=1
        ).clamp_min(1.0)

        x = (
            x * valid
        ).sum(dim=1) / denom

        x = self.norm(x)

        return self.classifier(x)


# ============================================================
# E5 EMBEDDINGS
# ============================================================

def make_e5_inputs(texts):
    # E5 recommends "query:" / "passage:" prefixes.
    return [
        "query: " + str(x)
        for x in texts
    ]


def get_teacher_embeddings(model, texts):
    return model.encode(
        make_e5_inputs(texts),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(
        np.float32
    )


# ============================================================
# TEACHER
# ============================================================

def train_teacher(
    train_embeddings,
    y_train,
):
    teacher = LogisticRegression(
        max_iter=1000,
        C=4.0,
        solver="lbfgs",
    )

    teacher.fit(
        train_embeddings,
        y_train,
    )

    return teacher


# ============================================================
# STUDENT TRAINING
# ============================================================

def evaluate_student(
    model,
    X,
    y,
):
    model.eval()

    inputs = torch.from_numpy(
        X
    )

    all_logits = []

    with torch.no_grad():

        for start in range(
            0,
            len(inputs),
            256,
        ):

            logits = model(
                inputs[
                    start:start + 256
                ]
            )

            all_logits.append(
                logits.cpu().numpy()
            )

    logits = np.concatenate(
        all_logits,
        axis=0,
    )

    pred = logits.argmax(
        axis=1
    )

    acc = accuracy_score(
        y,
        pred,
    )

    macro = f1_score(
        y,
        pred,
        average="macro",
        zero_division=0,
    )

    weighted = f1_score(
        y,
        pred,
        average="weighted",
        zero_division=0,
    )

    return (
        acc,
        macro,
        weighted,
        pred,
        logits,
    )


def train_student(
    model,
    X_train,
    y_train,
    teacher_logits_train,
    X_val,
    y_val,
    teacher_logits_val,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    train_ids = np.arange(
        len(X_train)
    )

    best_val_f1 = -1.0
    best_epoch = -1
    patience_count = 0

    history = []

    best_state = None

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        np.random.shuffle(
            train_ids
        )

        total_loss = 0.0
        batches = 0

        for start in range(
            0,
            len(train_ids),
            BATCH_SIZE,
        ):

            idx = train_ids[
                start:start + BATCH_SIZE
            ]

            xb = torch.from_numpy(
                X_train[idx]
            )

            yb = torch.from_numpy(
                y_train[idx]
            )

            tb = torch.from_numpy(
                teacher_logits_train[idx]
            )

            optimizer.zero_grad()

            student_logits = model(
                xb
            )

            ce = F.cross_entropy(
                student_logits,
                yb,
            )

            kd = F.kl_div(
                F.log_softmax(
                    student_logits
                    / TEMPERATURE,
                    dim=1,
                ),
                F.softmax(
                    tb
                    / TEMPERATURE,
                    dim=1,
                ),
                reduction="batchmean",
            ) * (
                TEMPERATURE
                * TEMPERATURE
            )

            loss = (
                CE_WEIGHT * ce
                + KD_WEIGHT * kd
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            total_loss += float(
                loss.item()
            )

            batches += 1

        (
            val_acc,
            val_f1,
            val_weighted,
            _,
            _,
        ) = evaluate_student(
            model,
            X_val,
            y_val,
        )

        avg_loss = (
            total_loss
            / max(batches, 1)
        )

        history.append(
            {
                "epoch": epoch,
                "loss": avg_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
                "val_weighted_f1": val_weighted,
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"loss={avg_loss:.4f} | "
            f"val={val_acc * 100:.2f}% | "
            f"valF1={val_f1 * 100:.2f}%"
        )

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1
            best_epoch = epoch
            patience_count = 0

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print(
                "Early stopping."
            )
            break

    if best_state is None:
        raise RuntimeError(
            "No best student checkpoint was produced."
        )

    model.load_state_dict(
        best_state
    )

    return (
        history,
        best_epoch,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    seed_everything()

    print("=" * 72)
    print(
        "E5-SMALL -> TINY STUDENT V2 "
        "FINAL TRAIN + LOCKED TEST"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # CHECK INPUTS
    # --------------------------------------------------------

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"Training CSV not found:\n{TRAIN_CSV}"
        )

    if not LOCKED_CSV.exists():
        raise FileNotFoundError(
            f"Canonical locked test not found:\n{LOCKED_CSV}"
        )

    # --------------------------------------------------------
    # LOAD TRAIN
    # --------------------------------------------------------

    train_df = pd.read_csv(
        TRAIN_CSV
    )

    if not {
        "text",
        "intent",
    }.issubset(
        train_df.columns
    ):
        raise ValueError(
            "train.csv must contain "
            "'text' and 'intent' columns."
        )

    train_df = train_df.dropna(
        subset=[
            "text",
            "intent",
        ]
    ).reset_index(
        drop=True
    )

    print()
    print(
        f"Training rows : {len(train_df)}"
    )

    print(
        f"Training intents : "
        f"{train_df['intent'].nunique()}"
    )

    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    labels = sorted(
        train_df["intent"]
        .unique()
        .tolist()
    )

    if len(labels) != 57:
        raise RuntimeError(
            f"Expected 57 training intents, "
            f"found {len(labels)}."
        )

    label_to_id = {
        label: i
        for i, label in enumerate(labels)
    }

    y = np.asarray(
        [
            label_to_id[x]
            for x in train_df["intent"]
        ],
        dtype=np.int64,
    )

    texts = train_df[
        "text"
    ].astype(str).tolist()

    # --------------------------------------------------------
    # STRATIFIED TRAIN / VALIDATION SPLIT
    # --------------------------------------------------------

    (
        text_train,
        text_val,
        y_train,
        y_val,
    ) = train_test_split(
        texts,
        y,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=y,
    )

    print()
    print(
        f"Train split : {len(text_train)}"
    )

    print(
        f"Val split   : {len(text_val)}"
    )

    # --------------------------------------------------------
    # BUILD STUDENT VOCAB FROM TRAINING TEXT ONLY
    # --------------------------------------------------------

    vocab = build_vocab(
        text_train
    )

    VOCAB_JSON.write_text(
        json.dumps(
            vocab,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    LABEL_MAP_JSON.write_text(
        json.dumps(
            {
                str(i): label
                for i, label in enumerate(labels)
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Student vocab size : {len(vocab)}"
    )

    # --------------------------------------------------------
    # STUDENT INPUT IDS
    # --------------------------------------------------------

    X_train = encode_texts(
        text_train,
        vocab,
    )

    X_val = encode_texts(
        text_val,
        vocab,
    )

    # --------------------------------------------------------
    # LOAD E5 TEACHER
    # --------------------------------------------------------

    print()
    print(
        "Loading E5-small-v2 teacher..."
    )

    teacher_model = SentenceTransformer(
        E5_MODEL_NAME
    )

    # --------------------------------------------------------
    # E5 EMBEDDINGS
    # --------------------------------------------------------

    print()
    print(
        "Creating E5 teacher embeddings..."
    )

    teacher_train_embeddings = (
        get_teacher_embeddings(
            teacher_model,
            text_train,
        )
    )

    np.save(
        TRAIN_EMB_NPY,
        teacher_train_embeddings,
    )

    # Validation embeddings
    teacher_val_embeddings = (
        get_teacher_embeddings(
            teacher_model,
            text_val,
        )
    )

    # --------------------------------------------------------
    # TRAIN LOGISTIC TEACHER
    # --------------------------------------------------------

    print()
    print(
        "Training E5 LogisticRegression teacher..."
    )

    teacher = train_teacher(
        teacher_train_embeddings,
        y_train,
    )

    teacher_train_logits = (
        teacher.decision_function(
            teacher_train_embeddings
        )
    )

    teacher_val_logits = (
        teacher.decision_function(
            teacher_val_embeddings
        )
    )

    teacher_val_pred = (
        teacher.predict(
            teacher_val_embeddings
        )
    )

    teacher_val_acc = (
        accuracy_score(
            y_val,
            teacher_val_pred,
        )
    )

    print()
    print(
        "E5 teacher validation accuracy : "
        f"{teacher_val_acc * 100:.4f}%"
    )

    TEACHER_JSON.write_text(
        json.dumps(
            {
                "model": E5_MODEL_NAME,
                "validation_accuracy": float(
                    teacher_val_acc
                ),
                "embedding_dimension": int(
                    teacher_train_embeddings.shape[1]
                ),
                "training_rows": int(
                    len(text_train)
                ),
                "num_intents": 57,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # RELEASE E5 MODEL BEFORE STUDENT TRAINING
    # --------------------------------------------------------

    del teacher_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # STUDENT
    # --------------------------------------------------------

    student = TinyIntentClassifier(
        vocab_size=len(vocab),
        num_classes=57,
    )

    print()
    print(
        "Training distilled student..."
    )

    (
        history,
        best_epoch,
    ) = train_student(
        student,
        X_train,
        y_train,
        teacher_train_logits,
        X_val,
        y_val,
        teacher_val_logits,
    )

    pd.DataFrame(
        history
    ).to_csv(
        HISTORY_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # VALIDATION RESULT
    # --------------------------------------------------------

    (
        val_acc,
        val_macro,
        val_weighted,
        val_pred,
        _,
    ) = evaluate_student(
        student,
        X_val,
        y_val,
    )

    val_report = classification_report(
        y_val,
        val_pred,
        labels=list(range(57)),
        target_names=labels,
        digits=4,
        zero_division=0,
    )

    VALIDATION_REPORT.write_text(
        (
            "# E5 DISTILLED STUDENT V2 "
            "VALIDATION\n\n"
            f"Accuracy   : {val_acc * 100:.4f}%\n"
            f"Macro F1   : {val_macro * 100:.4f}%\n"
            f"Weighted F1: {val_weighted * 100:.4f}%\n"
            f"Best epoch : {best_epoch}\n\n"
            + val_report
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # SAVE STUDENT CHECKPOINT
    # --------------------------------------------------------

    checkpoint_payload = {
        "model_state_dict": student.state_dict(),
        "vocab_size": len(vocab),
        "num_classes": 57,
        "labels": labels,
        "max_len": MAX_LEN,
        "pad_id": PAD_ID,
        "unk_id": UNK_ID,
        "embed_dim": EMBED_DIM,
        "nhead": NHEAD,
        "ff_dim": FF_DIM,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "best_epoch": best_epoch,
        "validation_accuracy": val_acc,
        "validation_macro_f1": val_macro,
        "validation_weighted_f1": val_weighted,
        "teacher": E5_MODEL_NAME,
        "quantization": False,
        "onnx": False,
    }

    torch.save(
        checkpoint_payload,
        CHECKPOINT,
    )

    print()
    print(
        "--- E5 DISTILLED V2 VALIDATION ---"
    )

    print(
        f"Accuracy   : {val_acc * 100:.4f}%"
    )

    print(
        f"Macro F1   : {val_macro * 100:.4f}%"
    )

    print(
        f"Weighted F1: {val_weighted * 100:.4f}%"
    )

    print(
        f"Best epoch : {best_epoch}"
    )

    print()
    print(
        f"Saved checkpoint:\n{CHECKPOINT}"
    )

    # --------------------------------------------------------
    # EXACT LOCKED TEST
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "NOW RUNNING EXACT LOCKED 57-INTENT TEST"
    )
    print("=" * 72)

    locked_df = pd.read_csv(
        LOCKED_CSV
    )

    if not {
        "text",
        "intent",
    }.issubset(
        locked_df.columns
    ):
        raise ValueError(
            "Locked test must contain "
            "'text' and 'intent' columns."
        )

    locked_df = locked_df.dropna(
        subset=[
            "text",
            "intent",
        ]
    ).reset_index(
        drop=True
    )

    if len(locked_df) != 1686:
        raise RuntimeError(
            f"Expected 1686 locked rows, "
            f"found {len(locked_df)}."
        )

    if locked_df["intent"].nunique() != 57:
        raise RuntimeError(
            f"Expected 57 locked intents, "
            f"found {locked_df['intent'].nunique()}."
        )

    unknown = sorted(
        set(
            locked_df["intent"]
        )
        - set(label_to_id)
    )

    if unknown:
        raise RuntimeError(
            "Locked test contains labels absent "
            "from training label map:\n"
            + "\n".join(unknown)
        )

    X_locked = encode_texts(
        locked_df["text"].astype(str).tolist(),
        vocab,
    )

    y_locked = np.asarray(
        [
            label_to_id[x]
            for x in locked_df["intent"]
        ],
        dtype=np.int64,
    )

    start = time.perf_counter()

    (
        locked_acc,
        locked_macro,
        locked_weighted,
        locked_pred,
        locked_logits,
    ) = evaluate_student(
        student,
        X_locked,
        y_locked,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    locked_report = classification_report(
        y_locked,
        locked_pred,
        labels=list(range(57)),
        target_names=labels,
        digits=4,
        zero_division=0,
    )

    print()
    print(
        "--- FINAL E5 DISTILLED V2 LOCKED TEST ---"
    )

    print(
        f"Accuracy   : "
        f"{locked_acc * 100:.4f}%"
    )

    print(
        f"Macro F1   : "
        f"{locked_macro * 100:.4f}%"
    )

    print(
        f"Weighted F1: "
        f"{locked_weighted * 100:.4f}%"
    )

    print()
    print(
        locked_report
    )

    print(
        "--- INFERENCE SPEED ---"
    )

    print(
        f"Total rows : {len(locked_df)}"
    )

    print(
        f"Total time : {elapsed:.4f} sec"
    )

    print(
        f"Rows/sec   : "
        f"{len(locked_df) / elapsed:.2f}"
    )

    print(
        f"ms/row     : "
        f"{elapsed / len(locked_df) * 1000:.4f}"
    )

    # --------------------------------------------------------
    # SAVE LOCKED RESULTS
    # --------------------------------------------------------

    locked_out = locked_df.copy()

    locked_out[
        "true_id"
    ] = y_locked

    locked_out[
        "predicted_id"
    ] = locked_pred

    locked_out[
        "prediction"
    ] = [
        labels[i]
        for i in locked_pred
    ]

    shifted = (
        locked_logits
        - locked_logits.max(
            axis=1,
            keepdims=True,
        )
    )

    probs = np.exp(
        shifted
    )

    probs /= (
        probs.sum(
            axis=1,
            keepdims=True,
        )
    )

    locked_out[
        "confidence"
    ] = probs.max(
        axis=1
    )

    locked_out.to_csv(
        LOCKED_PREDICTIONS,
        index=False,
    )

    LOCKED_REPORT.write_text(
        locked_report,
        encoding="utf-8",
    )

    cm = confusion_matrix(
        y_locked,
        locked_pred,
        labels=list(range(57)),
    )

    pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    ).to_csv(
        LOCKED_CM
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    summary = {
        "train_csv": str(TRAIN_CSV),
        "locked_csv": str(LOCKED_CSV),
        "checkpoint": str(CHECKPOINT),
        "teacher": E5_MODEL_NAME,
        "num_intents": 57,
        "locked_rows": 1686,
        "validation_accuracy": float(val_acc),
        "validation_macro_f1": float(val_macro),
        "validation_weighted_f1": float(val_weighted),
        "locked_accuracy": float(locked_acc),
        "locked_macro_f1": float(locked_macro),
        "locked_weighted_f1": float(locked_weighted),
        "best_epoch": int(best_epoch),
        "locked_inference_seconds": float(elapsed),
        "locked_rows_per_second": float(
            len(locked_df) / elapsed
        ),
        "locked_ms_per_row": float(
            elapsed / len(locked_df) * 1000
        ),
        "quantization": False,
        "onnx": False,
        "synthetic_text": False,
        "labels_changed": False,
        "locked_test_used_for_training": False,
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "============================================================"
    )

    print(
        "FINAL FILES"
    )

    print(
        CHECKPOINT
    )

    print(
        VOCAB_JSON
    )

    print(
        LABEL_MAP_JSON
    )

    print(
        HISTORY_CSV
    )

    print(
        VALIDATION_REPORT
    )

    print(
        LOCKED_PREDICTIONS
    )

    print(
        LOCKED_REPORT
    )

    print(
        LOCKED_CM
    )

    print(
        SUMMARY_JSON
    )

    print()
    print(
        "STATUS: "
        "E5 DISTILLED V2 TRAINING + "
        "EXACT LOCKED TEST COMPLETE"
    )


if __name__ == "__main__":
    main()
