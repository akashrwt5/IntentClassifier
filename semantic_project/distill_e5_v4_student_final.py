#!/usr/bin/env python3
"""
FINAL — E5 SMALL -> COMPACT V4 STUDENT DISTILLATION

Teacher:
  intfloat/multilingual-e5-small (384-d)

Student:
  multilingual E5 tokenizer
  embedding = 128
  Transformer = 4 layers
  heads = 4
  FFN = 256
  max_len = 32
  11 intent classes

Distillation:
  1. supervised intent classification
  2. learned 384 -> 128 teacher projection
  3. cosine embedding distillation
  4. teacher semantic-prototype distillation
  5. extra weighting for short / typo / hard examples

Safety:
  - V3 is NOT modified.
  - locked 595-row unseen test is NOT used for training.
  - no ONNX export.
  - no threshold fitting.
  - this produces an FP32 checkpoint only.

Run:
  python3 distill_e5_v4_student_final.py

Optional:
  python3 distill_e5_v4_student_final.py --epochs 15
"""

from pathlib import Path
import argparse
import json
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

ROOT = Path("/Users/shuklam/IntentClassifier/semantic_project")
DATA = ROOT / "v4_semantic_training"
OUT = ROOT / "e5_v4_student_final"
OUT.mkdir(parents=True, exist_ok=True)

TEACHER_NAME = "intfloat/multilingual-e5-small"

MAX_LEN = 32
STUDENT_DIM = 128
LAYERS = 4
HEADS = 4
FFN = 256

BATCH = 32
EPOCHS = 15
LR = 2e-3
WEIGHT_DECAY = 1e-4

# Loss weights
LAMBDA_EMBED = 1.50
LAMBDA_PROTO = 0.35
LABEL_SMOOTHING = 0.03

SEED = 42

LOCKED_UNSEEN = ROOT / "unseen_semantic_stress_test.csv"


# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize_text(x):
    return re.sub(r"\s+", " ", str(x).strip())


def mean_pool(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------

def load_data():

    train_path = DATA / "v4_training_data.csv"
    dev_path = DATA / "v4_dev_data.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing: {train_path}\n"
            "Run build_v4_semantic_data.py first."
        )

    if not dev_path.exists():
        raise FileNotFoundError(
            f"Missing: {dev_path}\n"
            "Run build_v4_semantic_data.py first."
        )

    train = pd.read_csv(train_path)
    dev = pd.read_csv(dev_path)

    for name, df in [("train", train), ("dev", dev)]:

        if "text" not in df.columns:
            raise ValueError(f"{name} CSV missing 'text' column.")

        if "intent" not in df.columns:
            raise ValueError(f"{name} CSV missing 'intent' column.")

        df["text"] = df["text"].map(normalize_text)
        df["intent"] = df["intent"].astype(str).str.strip()

    return train, dev


def verify_locked_unseen(train, dev):

    if not LOCKED_UNSEEN.exists():

        print(
            "WARNING: locked unseen file not found:\n"
            f"{LOCKED_UNSEEN}"
        )

        return

    unseen = pd.read_csv(LOCKED_UNSEEN)

    if "text" not in unseen.columns:

        print(
            "WARNING: unseen file has no text column. "
            "Overlap check skipped."
        )

        return

    unseen_text = set(
        unseen["text"].map(normalize_text)
    )

    train_overlap = set(train["text"]) & unseen_text
    dev_overlap = set(dev["text"]) & unseen_text

    print(
        "Locked unseen overlap in train:",
        len(train_overlap)
    )

    print(
        "Locked unseen overlap in dev  :",
        len(dev_overlap)
    )

    if train_overlap or dev_overlap:

        raise RuntimeError(
            "\nCRITICAL DATA LEAKAGE\n"
            "595-row unseen set overlaps V4 train/dev.\n"
            "Remove overlap before training."
        )


# ---------------------------------------------------------------------
# STUDENT
# ---------------------------------------------------------------------

class Student(nn.Module):

    def __init__(
        self,
        vocab_size,
        num_classes,
        pad_token_id,
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            STUDENT_DIM,
            padding_idx=pad_token_id,
        )

        self.position = nn.Embedding(
            MAX_LEN,
            STUDENT_DIM,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=STUDENT_DIM,
            nhead=HEADS,
            dim_feedforward=FFN,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=LAYERS,
        )

        self.norm = nn.LayerNorm(
            STUDENT_DIM
        )

        self.semantic_projection = nn.Sequential(
            nn.Linear(
                STUDENT_DIM,
                STUDENT_DIM,
            ),
            nn.GELU(),
            nn.LayerNorm(
                STUDENT_DIM
            ),
        )

        self.classifier = nn.Linear(
            STUDENT_DIM,
            num_classes,
        )

        # IMPORTANT:
        # E5 teacher = 384 dimensions
        # Student = 128 dimensions
        #
        # We learn a projection rather than truncating E5.
        self.teacher_projection = nn.Sequential(
            nn.Linear(
                384,
                256,
            ),
            nn.GELU(),
            nn.Linear(
                256,
                STUDENT_DIM,
            ),
            nn.LayerNorm(
                STUDENT_DIM
            ),
        )


    def encode(
        self,
        input_ids,
        attention_mask,
    ):

        batch_size, seq_len = input_ids.shape

        if seq_len > MAX_LEN:

            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"MAX_LEN={MAX_LEN}"
            )

        positions = torch.arange(
            seq_len,
            device=input_ids.device,
        ).unsqueeze(0)

        x = (
            self.embedding(input_ids)
            +
            self.position(positions)
        )

        padding_mask = (
            attention_mask == 0
        )

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        x = mean_pool(
            x,
            attention_mask,
        )

        x = self.norm(x)

        semantic = self.semantic_projection(x)

        semantic = F.normalize(
            semantic,
            p=2,
            dim=-1,
        )

        return semantic


    def project_teacher(
        self,
        teacher_embedding,
    ):

        projected = self.teacher_projection(
            teacher_embedding
        )

        return F.normalize(
            projected,
            p=2,
            dim=-1,
        )


    def forward(
        self,
        input_ids,
        attention_mask,
    ):

        semantic = self.encode(
            input_ids,
            attention_mask,
        )

        logits = self.classifier(
            semantic
        )

        return semantic, logits


# ---------------------------------------------------------------------
# TOKENIZATION
# ---------------------------------------------------------------------

def tokenize(
    tokenizer,
    texts,
    device,
):

    # E5 works best with query: prefix for user queries.
    prepared = [
        f"query: {x}"
        for x in texts
    ]

    tokens = tokenizer(
        prepared,
        max_length=MAX_LEN,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    return {
        k: v.to(device)
        for k, v in tokens.items()
    }


# ---------------------------------------------------------------------
# TEACHER
# ---------------------------------------------------------------------

@torch.no_grad()
def teacher_encode(
    teacher,
    tokenizer,
    texts,
    device,
):

    tokens = tokenize(
        tokenizer,
        texts,
        device,
    )

    output = teacher(
        **tokens
    )

    embedding = mean_pool(
        output.last_hidden_state,
        tokens["attention_mask"],
    )

    return F.normalize(
        embedding,
        p=2,
        dim=-1,
    )


@torch.no_grad()
def cache_teacher_embeddings(
    teacher,
    tokenizer,
    texts,
    device,
):

    chunks = []

    for start in tqdm(
        range(
            0,
            len(texts),
            BATCH,
        ),
        desc="E5 embeddings",
    ):

        chunk = texts[
            start:start + BATCH
        ]

        emb = teacher_encode(
            teacher,
            tokenizer,
            chunk,
            device,
        )

        chunks.append(
            emb.cpu()
        )

    return torch.cat(
        chunks,
        dim=0,
    )


@torch.no_grad()
def build_teacher_prototypes(
    teacher,
    tokenizer,
    train,
    labels,
    device,
):

    prototypes = []

    print(
        "\nBuilding E5 intent prototypes..."
    )

    for label in labels:

        texts = train.loc[
            train["intent"] == label,
            "text",
        ].tolist()

        # Keep setup deterministic and bounded.
        texts = texts[:512]

        emb = teacher_encode(
            teacher,
            tokenizer,
            texts,
            device,
        )

        prototype = F.normalize(
            emb.mean(
                dim=0,
                keepdim=True,
            ),
            p=2,
            dim=-1,
        )

        prototypes.append(
            prototype.squeeze(0).cpu()
        )

    return torch.stack(
        prototypes,
        dim=0,
    )


# ---------------------------------------------------------------------
# SAMPLE WEIGHTS
# ---------------------------------------------------------------------

def sample_weight(source):

    source = str(source).lower()

    if "hard" in source:
        return 1.50

    if "typo" in source:
        return 1.35

    if "short" in source:
        return 1.35

    if "context" in source:
        return 1.20

    return 1.0


# ---------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    student,
    tokenizer,
    df,
    label_encoder,
    device,
):

    student.eval()

    texts = df["text"].tolist()

    y_true = label_encoder.transform(
        df["intent"]
    )

    predictions = []

    for start in range(
        0,
        len(texts),
        BATCH,
    ):

        tokens = tokenize(
            tokenizer,
            texts[start:start + BATCH],
            device,
        )

        _, logits = student(
            tokens["input_ids"],
            tokens["attention_mask"],
        )

        predictions.extend(
            logits.argmax(
                dim=1
            ).cpu().tolist()
        )

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
    )

    return (
        accuracy,
        macro_f1,
        y_true,
        predictions,
    )


# ---------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    args = parser.parse_args()

    seed_everything()

    device = get_device()

    print(
        "=" * 78
    )

    print(
        "E5 -> COMPACT MULTILINGUAL STUDENT"
    )

    print(
        "=" * 78
    )

    print(
        "Device:",
        device,
    )

    print(
        "Teacher:",
        TEACHER_NAME,
    )

    print(
        "Student:",
        f"{STUDENT_DIM}d / "
        f"{LAYERS} layers / "
        f"{HEADS} heads / "
        f"FFN {FFN}",
    )

    print(
        "Max length:",
        MAX_LEN,
    )

    # -------------------------------------------------------------
    # DATA
    # -------------------------------------------------------------

    train, dev = load_data()

    verify_locked_unseen(
        train,
        dev,
    )

    labels = sorted(
        train["intent"].unique()
    )

    if len(labels) != 11:

        raise RuntimeError(
            f"Expected 11 intents, "
            f"found {len(labels)}:\n{labels}"
        )

    label_encoder = LabelEncoder()

    label_encoder.fit(
        labels
    )

    print(
        "\nTrain rows:",
        len(train),
    )

    print(
        "Dev rows:",
        len(dev),
    )

    print(
        "Intents:",
        len(labels),
    )

    if "language" in train.columns:

        print(
            "\nLanguage distribution:"
        )

        print(
            train[
                "language"
            ].value_counts().to_string()
        )

    else:

        print(
            "\nWARNING: language column "
            "not present."
        )

        print(
            "This run is NOT a 14-language "
            "training evaluation."
        )

    # -------------------------------------------------------------
    # LOAD E5
    # -------------------------------------------------------------

    print(
        "\nLoading multilingual-e5-small..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        TEACHER_NAME
    )

    teacher = AutoModel.from_pretrained(
        TEACHER_NAME
    ).to(device)

    teacher.eval()

    for parameter in teacher.parameters():

        parameter.requires_grad = False

    # -------------------------------------------------------------
    # STUDENT
    # -------------------------------------------------------------

    student = Student(
        vocab_size=tokenizer.vocab_size,
        num_classes=len(labels),
        pad_token_id=tokenizer.pad_token_id,
    ).to(device)

    print(
        "\nStudent parameters:",
        sum(
            p.numel()
            for p in student.parameters()
        ),
    )

    # -------------------------------------------------------------
    # CACHE TEACHER
    # -------------------------------------------------------------

    print(
        "\nCaching E5 teacher embeddings..."
    )

    train_teacher = cache_teacher_embeddings(
        teacher,
        tokenizer,
        train["text"].tolist(),
        device,
    )

    dev_teacher = cache_teacher_embeddings(
        teacher,
        tokenizer,
        dev["text"].tolist(),
        device,
    )

    prototypes = build_teacher_prototypes(
        teacher,
        tokenizer,
        train,
        labels,
        device,
    )

    # -------------------------------------------------------------
    # LABELS
    # -------------------------------------------------------------

    train_y = torch.tensor(
        label_encoder.transform(
            train["intent"]
        ),
        dtype=torch.long,
    )

    dev_y = torch.tensor(
        label_encoder.transform(
            dev["intent"]
        ),
        dtype=torch.long,
    )

    # -------------------------------------------------------------
    # SAMPLE WEIGHTS
    # -------------------------------------------------------------

    if "source" in train.columns:

        weights = torch.tensor(
            [
                sample_weight(x)
                for x in train["source"]
            ],
            dtype=torch.float32,
        )

    else:

        weights = torch.ones(
            len(train),
            dtype=torch.float32,
        )

    # -------------------------------------------------------------
    # OPTIMIZER
    # -------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    loss_function = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING,
        reduction="none",
    )

    best_accuracy = -1.0
    best_f1 = -1.0
    best_state = None

    train_texts = train[
        "text"
    ].tolist()

    # -------------------------------------------------------------
    # TRAINING
    # -------------------------------------------------------------

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        student.train()

        order = np.random.permutation(
            len(train)
        )

        epoch_losses = []

        for start in range(
            0,
            len(order),
            BATCH,
        ):

            indices = order[
                start:start + BATCH
            ]

            batch_text = [
                train_texts[i]
                for i in indices
            ]

            y = train_y[
                indices
            ].to(device)

            teacher_embedding = train_teacher[
                indices
            ].to(device)

            sample_weights = weights[
                indices
            ].to(device)

            tokens = tokenize(
                tokenizer,
                batch_text,
                device,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            student_embedding, logits = student(
                tokens["input_ids"],
                tokens["attention_mask"],
            )

            # -------------------------------------------------
            # LOSS 1: SUPERVISED INTENT
            # -------------------------------------------------

            ce_each = loss_function(
                logits,
                y,
            )

            classification_loss = (
                ce_each * sample_weights
            ).mean()

            # -------------------------------------------------
            # LOSS 2: E5 -> 128-D PROJECTION
            # -------------------------------------------------

            teacher_128 = (
                student.project_teacher(
                    teacher_embedding
                )
            )

            embedding_loss = (
                1.0
                -
                (
                    student_embedding
                    *
                    teacher_128
                ).sum(dim=-1)
            ).mean()

            # -------------------------------------------------
            # LOSS 3: SEMANTIC PROTOTYPE DISTILLATION
            # -------------------------------------------------

            prototype_128 = (
                student.project_teacher(
                    prototypes.to(device)
                )
            )

            teacher_proto_scores = (
                teacher_embedding
                @ prototypes.to(device).T
            )

            student_proto_scores = (
                student_embedding
                @ prototype_128.T
            )

            prototype_loss = F.mse_loss(
                student_proto_scores,
                teacher_proto_scores,
            )

            # -------------------------------------------------
            # TOTAL
            # -------------------------------------------------

            total_loss = (
                classification_loss
                +
                LAMBDA_EMBED
                *
                embedding_loss
                +
                LAMBDA_PROTO
                *
                prototype_loss
            )

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                student.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            epoch_losses.append(
                float(
                    total_loss.detach().cpu()
                )
            )

        # -----------------------------------------------------
        # VALIDATION
        # -----------------------------------------------------

        val_accuracy, val_f1, _, _ = evaluate(
            student,
            tokenizer,
            dev,
            label_encoder,
            device,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"loss={np.mean(epoch_losses):.4f} | "
            f"val={val_accuracy*100:.2f}% | "
            f"valF1={val_f1*100:.2f}%"
        )

        if (
            val_accuracy > best_accuracy
            or (
                val_accuracy == best_accuracy
                and val_f1 > best_f1
            )
        ):

            best_accuracy = val_accuracy
            best_f1 = val_f1

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in student.state_dict().items()
            }

    # -------------------------------------------------------------
    # RESTORE BEST
    # -------------------------------------------------------------

    if best_state is None:

        raise RuntimeError(
            "No best checkpoint was produced."
        )

    student.load_state_dict(
        best_state
    )

    final_accuracy, final_f1, y_true, pred = evaluate(
        student,
        tokenizer,
        dev,
        label_encoder,
        device,
    )

    print(
        "\n" + "=" * 78
    )

    print(
        "FINAL V4 E5 DISTILLED STUDENT"
    )

    print(
        "=" * 78
    )

    print(
        "Validation accuracy:",
        f"{final_accuracy*100:.2f}%",
    )

    print(
        "Validation Macro F1:",
        f"{final_f1*100:.2f}%",
    )

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            y_true,
            pred,
            target_names=label_encoder.classes_,
            digits=4,
        )
    )

    # -------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------

    checkpoint = (
        OUT
        /
        "v4_e5_distilled_student_fp32.pt"
    )

    torch.save(
        {
            "state_dict": student.state_dict(),

            "labels":
                label_encoder.classes_.tolist(),

            "teacher":
                TEACHER_NAME,

            "tokenizer":
                TEACHER_NAME,

            "vocab_size":
                tokenizer.vocab_size,

            "pad_token_id":
                tokenizer.pad_token_id,

            "max_len":
                MAX_LEN,

            "student_dim":
                STUDENT_DIM,

            "layers":
                LAYERS,

            "heads":
                HEADS,

            "ffn":
                FFN,

            "validation_accuracy":
                float(final_accuracy),

            "validation_macro_f1":
                float(final_f1),
        },
        checkpoint,
    )

    manifest = {

        "teacher":
            TEACHER_NAME,

        "student": {
            "embedding_dim":
                STUDENT_DIM,

            "layers":
                LAYERS,

            "heads":
                HEADS,

            "ffn":
                FFN,

            "max_length":
                MAX_LEN,

            "tokenizer":
                TEACHER_NAME,
        },

        "train_rows":
            int(len(train)),

        "validation_rows":
            int(len(dev)),

        "validation_accuracy":
            float(final_accuracy),

        "validation_macro_f1":
            float(final_f1),

        "locked_595_used":
            False,

        "v3_modified":
            False,

        "onnx_exported":
            False,

        "int8_exported":
            False,

        "threshold_fitted":
            False,

        "production_ready":
            False,

        "note":
            "FP32 distillation checkpoint only. "
            "Must pass locked V3 benchmark before export.",
    }

    (
        OUT
        /
        "training_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nSaved checkpoint:"
    )

    print(
        checkpoint
    )

    print(
        "\nV3 modified: NO"
    )

    print(
        "595-row unseen used: NO"
    )

    print(
        "ONNX exported: NO"
    )

    print(
        "INT8 exported: NO"
    )

    print(
        "Production ready: NO"
    )

    print(
        "\nNEXT:"
    )

    print(
        "Benchmark V4 FP32 against the locked V3."
    )


if __name__ == "__main__":
    main()
