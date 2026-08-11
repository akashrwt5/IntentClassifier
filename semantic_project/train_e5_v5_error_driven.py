#!/usr/bin/env python3
"""
V5 — E5 + ERROR-DRIVEN + CONTRASTIVE STUDENT

Goal
----
Train a stronger mobile semantic student while preserving the behavior
that made V3 strong on the locked 595-row unseen test.

Teacher:
    intfloat/multilingual-e5-small

Student:
    compact Transformer, 128-dimensional representation

Training signals:
    1. Hard classification loss
    2. E5 teacher embedding distillation
    3. E5 teacher intent-prototype similarity
    4. Contrastive loss for confusing intent pairs
    5. Critical-command weighting
    6. Hard-negative weighting

Safety
------
- The locked 595-row unseen set is NEVER loaded.
- V3 is NOT modified.
- No ONNX export.
- No INT8.
- Output is a new V5 checkpoint only.

Expected project files:
    e5_v4_teacher/...
    production_calibration_v2/production_hard_negative.csv
    production_calibration_v2/production_contrastive_pairs.csv
    fine_tuned_test_predictions.csv
    tiny_semantic_student_v2_balanced/vocab.json

If a hard-negative/contrastive file is absent, the script continues with
the available training data and clearly reports it.

Run:
    python3 train_e5_v5_error_driven.py

Optional:
    python3 train_e5_v5_error_driven.py --epochs 20
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
from transformers import AutoTokenizer, AutoModel


# ================================================================
# CONFIG
# ================================================================

ROOT = Path(
    "/Users/shuklam/IntentClassifier/semantic_project"
)

TEACHER_NAME = "intfloat/multilingual-e5-small"

NORMAL_SOURCE = (
    ROOT / "fine_tuned_test_predictions.csv"
)

HARD_NEGATIVE_CANDIDATES = [
    ROOT
    / "production_calibration_v2"
    / "production_hard_negative.csv",

    ROOT
    / "production_hardening_v2"
    / "starter_safety_results.csv",

    ROOT
    / "tiny_semantic_student_v3_error_driven"
    / "hard_negative_results.csv",
]

CONTRASTIVE_CANDIDATES = [
    ROOT
    / "production_calibration_v2"
    / "production_contrastive_pairs.csv",

    ROOT
    / "production_calibration_v1"
    / "production_contrastive_pairs.csv",
]

OUTPUT_DIR = (
    ROOT / "e5_v5_error_driven"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 42

MAX_LEN = 32
STUDENT_DIM = 128
LAYERS = 4
HEADS = 4
FFN = 256

BATCH_SIZE = 32

DEFAULT_EPOCHS = 20

LR = 2e-4

WEIGHT_DECAY = 1e-4

ALPHA_CLS = 1.0
ALPHA_KD = 0.45
ALPHA_PROTO = 0.30
ALPHA_CONTRASTIVE = 0.25

HARD_NEGATIVE_WEIGHT = 2.5
CRITICAL_WEIGHT = 2.0

TEMPERATURE = 0.07

VAL_PER_INTENT = 20


# ================================================================
# REPRODUCIBILITY
# ================================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ================================================================
# TEXT
# ================================================================

def normalize_text(text):
    return re.sub(
        r"\s+",
        " ",
        str(text).strip(),
    )


def find_column(df, names):

    lower = {
        str(c).lower(): c
        for c in df.columns
    }

    for name in names:

        if name.lower() in lower:
            return lower[name.lower()]

    return None


# ================================================================
# DATA
# ================================================================

def load_normal_data():

    if not NORMAL_SOURCE.exists():

        raise FileNotFoundError(
            f"Training source not found:\n"
            f"{NORMAL_SOURCE}"
        )

    df = pd.read_csv(
        NORMAL_SOURCE
    )

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
            "target",
        ],
    )

    if text_col is None or intent_col is None:

        raise ValueError(
            "Normal training CSV needs text + intent columns."
        )

    df = df.copy()

    df["text"] = df[text_col].map(
        normalize_text
    )

    df["intent"] = (
        df[intent_col]
        .astype(str)
        .str.strip()
    )

    return df[
        [
            "text",
            "intent",
        ]
    ]


def find_existing(candidates):

    for path in candidates:

        if path.exists():
            return path

    return None


def load_optional_hard_negatives():

    path = find_existing(
        HARD_NEGATIVE_CANDIDATES
    )

    if path is None:

        print(
            "\nHard-negative file: NOT FOUND"
        )

        return pd.DataFrame(
            columns=[
                "text",
                "intent",
            ]
        )

    print(
        "\nHard-negative source:"
    )

    print(path)

    df = pd.read_csv(path)

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
            "target",
            "expected",
        ],
    )

    if text_col is None:

        raise ValueError(
            f"No text column in {path}"
        )

    if intent_col is None:

        # A hard-negative result may contain the
        # expected label under another common name.
        raise ValueError(
            f"No intent/expected label in {path}"
        )

    out = pd.DataFrame(
        {
            "text":
                df[text_col].map(
                    normalize_text
                ),
            "intent":
                df[intent_col]
                .astype(str)
                .str.strip(),
        }
    )

    return out


def load_optional_contrastive():

    path = find_existing(
        CONTRASTIVE_CANDIDATES
    )

    if path is None:

        print(
            "\nContrastive-pair source: NOT FOUND"
        )

        return pd.DataFrame()

    print(
        "\nContrastive-pair source:"
    )

    print(path)

    df = pd.read_csv(path)

    return df


# ================================================================
# CRITICAL COMMAND DETECTOR
# ================================================================

CRITICAL_PHRASES = [
    "make it louder",
    "make it quieter",
    "mute",
    "unmute",
    "turn off",
    "turn the sound back on",
    "completely silent",
    "keep it on",
]


def is_critical(text):

    t = normalize_text(
        text
    ).lower()

    return any(
        phrase in t
        for phrase in CRITICAL_PHRASES
    )


# ================================================================
# STUDENT
# ================================================================

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

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=STUDENT_DIM,
                nhead=HEADS,
                dim_feedforward=FFN,
                dropout=0.10,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=LAYERS,
            )
        )

        self.norm = nn.LayerNorm(
            STUDENT_DIM
        )

        self.semantic_projection = (
            nn.Sequential(
                nn.Linear(
                    STUDENT_DIM,
                    STUDENT_DIM,
                ),
                nn.GELU(),
                nn.LayerNorm(
                    STUDENT_DIM
                ),
            )
        )

        self.classifier = nn.Linear(
            STUDENT_DIM,
            num_classes,
        )

        # E5 = 384 dimensions.
        # Project teacher representation to the
        # student's 128-dimensional semantic space.
        self.teacher_projection = (
            nn.Sequential(
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
        )

    def forward(
        self,
        input_ids,
        attention_mask,
    ):

        _, seq_len = input_ids.shape

        positions = torch.arange(
            seq_len,
            device=input_ids.device,
        ).unsqueeze(0)

        x = (
            self.embedding(
                input_ids
            )
            +
            self.position(
                positions
            )
        )

        padding_mask = (
            attention_mask == 0
        )

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        mask = (
            attention_mask
            .unsqueeze(-1)
            .float()
        )

        x = (
            (x * mask).sum(1)
            /
            mask.sum(1).clamp(
                min=1e-9
            )
        )

        x = self.norm(x)

        semantic = (
            self.semantic_projection(
                x
            )
        )

        semantic = F.normalize(
            semantic,
            p=2,
            dim=-1,
        )

        logits = self.classifier(
            semantic
        )

        return semantic, logits


# ================================================================
# E5 TEACHER
# ================================================================

class E5Teacher:

    def __init__(
        self,
        device,
    ):

        print(
            "\nLoading E5 teacher..."
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                TEACHER_NAME
            )
        )

        self.model = (
            AutoModel.from_pretrained(
                TEACHER_NAME
            )
        )

        self.model.to(
            device
        )

        self.model.eval()

        for parameter in (
            self.model.parameters()
        ):
            parameter.requires_grad = False

        self.device = device

    @torch.no_grad()
    def encode(
        self,
        texts,
        batch_size=32,
    ):

        outputs = []

        for start in range(
            0,
            len(texts),
            batch_size,
        ):

            batch = texts[
                start:
                start + batch_size
            ]

            tokens = self.tokenizer(
                [
                    f"query: {x}"
                    for x in batch
                ],
                max_length=MAX_LEN,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            tokens = {
                k: v.to(
                    self.device
                )
                for k, v in tokens.items()
            }

            model_output = (
                self.model(
                    **tokens
                )
            )

            hidden = (
                model_output
                .last_hidden_state
            )

            mask = (
                tokens[
                    "attention_mask"
                ]
                .unsqueeze(-1)
                .float()
            )

            pooled = (
                (hidden * mask).sum(1)
                /
                mask.sum(1).clamp(
                    min=1e-9
                )
            )

            pooled = F.normalize(
                pooled,
                p=2,
                dim=-1,
            )

            outputs.append(
                pooled.cpu()
            )

        return torch.cat(
            outputs,
            dim=0,
        )


# ================================================================
# VALIDATION SPLIT
# ================================================================

def split_by_intent(
    df,
    val_per_intent=20,
):

    train_parts = []
    val_parts = []

    for intent, group in (
        df.groupby("intent")
    ):

        group = (
            group
            .sample(
                frac=1.0,
                random_state=SEED,
            )
            .reset_index(
                drop=True
            )
        )

        val_n = min(
            val_per_intent,
            max(
                1,
                len(group) // 5,
            ),
        )

        val_parts.append(
            group.iloc[
                :val_n
            ]
        )

        train_parts.append(
            group.iloc[
                val_n:
            ]
        )

    train = pd.concat(
        train_parts,
        ignore_index=True,
    )

    val = pd.concat(
        val_parts,
        ignore_index=True,
    )

    return train, val


# ================================================================
# INTENT PROTOTYPES
# ================================================================

def build_teacher_prototypes(
    teacher_embeddings,
    labels,
    train_labels,
):

    prototypes = []

    for label in labels:

        mask = np.asarray(
            [
                x == label
                for x in train_labels
            ]
        )

        if not mask.any():

            prototypes.append(
                torch.zeros(
                    384
                )
            )

            continue

        p = teacher_embeddings[
            mask
        ].mean(
            dim=0
        )

        p = F.normalize(
            p,
            p=2,
            dim=0,
        )

        prototypes.append(
            p
        )

    return torch.stack(
        prototypes
    )


# ================================================================
# LOSSES
# ================================================================

def weighted_cross_entropy(
    logits,
    target,
    weights,
):

    losses = F.cross_entropy(
        logits,
        target,
        reduction="none",
    )

    return (
        losses * weights
    ).mean()


def prototype_loss(
    student_embedding,
    teacher_embedding,
    prototypes,
    target,
):

    # Teacher says which intent region the
    # sample belongs to. Student must preserve
    # the same prototype geometry.

    projected_teacher = teacher_embedding

    teacher_similarity = (
        projected_teacher
        @
        prototypes.T
    )

    student_similarity = (
        student_embedding
        @
        prototypes.T
    )

    teacher_prob = F.softmax(
        teacher_similarity
        / TEMPERATURE,
        dim=-1,
    )

    student_log_prob = F.log_softmax(
        student_similarity
        / TEMPERATURE,
        dim=-1,
    )

    return F.kl_div(
        student_log_prob,
        teacher_prob,
        reduction="batchmean",
    )


def embedding_kd_loss(
    student_embedding,
    teacher_embedding,
):

    return (
        1.0
        -
        (
            student_embedding
            *
            teacher_embedding
        ).sum(-1)
    ).mean()


def supervised_contrastive_loss(
    embeddings,
    labels,
    temperature=TEMPERATURE,
):

    if len(embeddings) <= 1:
        return embeddings.sum() * 0.0

    z = F.normalize(
        embeddings,
        p=2,
        dim=-1,
    )

    sim = (
        z @ z.T
    ) / temperature

    mask_self = torch.eye(
        len(z),
        device=z.device,
        dtype=torch.bool,
    )

    sim = sim.masked_fill(
        mask_self,
        -1e9,
    )

    labels = labels.view(
        -1,
        1,
    )

    positive = (
        labels == labels.T
    )

    positive = positive.masked_fill(
        mask_self,
        False,
    )

    log_prob = F.log_softmax(
        sim,
        dim=1,
    )

    positive_count = (
        positive.sum(
            dim=1
        )
        .clamp(
            min=1
        )
    )

    loss = -(
        (
            log_prob
            *
            positive.float()
        ).sum(1)
        /
        positive_count
    )

    # Only anchors with at least one
    # positive contribute.
    valid = (
        positive.sum(1)
        > 0
    )

    if not valid.any():
        return embeddings.sum() * 0.0

    return loss[
        valid
    ].mean()


# ================================================================
# DATASET
# ================================================================

class CachedDataset:

    def __init__(
        self,
        df,
        teacher_embeddings,
        label_to_id,
        hard_negative=False,
    ):

        self.df = df.reset_index(
            drop=True
        )

        self.teacher = (
            teacher_embeddings
        )

        self.label_ids = torch.tensor(
            [
                label_to_id[x]
                for x in self.df[
                    "intent"
                ]
            ],
            dtype=torch.long,
        )

        weights = []

        for text in self.df[
            "text"
        ]:

            weight = (
                HARD_NEGATIVE_WEIGHT
                if hard_negative
                else 1.0
            )

            if is_critical(text):
                weight *= (
                    CRITICAL_WEIGHT
                )

            weights.append(
                weight
            )

        self.weights = torch.tensor(
            weights,
            dtype=torch.float32,
        )

    def __len__(self):
        return len(
            self.df
        )


# ================================================================
# TOKEN CACHE
# ================================================================

def tokenize_dataframe(
    tokenizer,
    df,
    device,
):

    tokens = tokenizer(
        [
            f"query: {x}"
            for x in df[
                "text"
            ]
        ],
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    return {
        k: v.to(device)
        for k, v in tokens.items()
    }


# ================================================================
# EVALUATION
# ================================================================

@torch.no_grad()
def evaluate(
    model,
    tokens,
    targets,
    labels,
):

    model.eval()

    _, logits = model(
        tokens["input_ids"],
        tokens["attention_mask"],
    )

    pred = logits.argmax(
        dim=1
    ).cpu().numpy()

    truth = targets.cpu().numpy()

    accuracy = accuracy_score(
        truth,
        pred,
    )

    macro_f1 = f1_score(
        truth,
        pred,
        average="macro",
        zero_division=0,
    )

    return (
        accuracy,
        macro_f1,
        pred,
        truth,
    )


# ================================================================
# MAIN
# ================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    args = parser.parse_args()

    seed_everything(
        SEED
    )

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
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
        "\nSAFETY:"
    )

    print(
        "595-row unseen set is NOT loaded."
    )

    print(
        "V3 checkpoint is NOT modified."
    )

    print(
        "No ONNX export."
    )

    print(
        "No INT8."
    )

    # ------------------------------------------------------------
    # NORMAL DATA
    # ------------------------------------------------------------

    normal = load_normal_data()

    print(
        "\nNormal rows:",
        len(normal),
    )

    labels = sorted(
        normal[
            "intent"
        ].unique()
    )

    label_to_id = {
        label: i
        for i, label in enumerate(
            labels
        )
    }

    print(
        "Intents:",
        len(labels),
    )

    # ------------------------------------------------------------
    # SPLIT NORMAL DATA
    # ------------------------------------------------------------

    normal_train, normal_val = (
        split_by_intent(
            normal,
            VAL_PER_INTENT,
        )
    )

    # ------------------------------------------------------------
    # HARD NEGATIVES
    # ------------------------------------------------------------

    hard = (
        load_optional_hard_negatives()
    )

    # Keep only known intents.
    if len(hard):

        hard = hard[
            hard["intent"].isin(
                labels
            )
        ].reset_index(
            drop=True
        )

    print(
        "Hard-negative rows:",
        len(hard),
    )

    # ------------------------------------------------------------
    # COMBINE TRAIN DATA
    # ------------------------------------------------------------

    train = pd.concat(
        [
            normal_train,
            hard,
        ],
        ignore_index=True,
    )

    # Remove exact duplicate pairs.
    train = train.drop_duplicates(
        subset=[
            "text",
            "intent",
        ]
    ).reset_index(
        drop=True
    )

    print(
        "Final train rows:",
        len(train),
    )

    print(
        "Validation rows:",
        len(normal_val),
    )

    # ------------------------------------------------------------
    # TEACHER
    # ------------------------------------------------------------

    teacher = E5Teacher(
        device
    )

    print(
        "\nCaching E5 embeddings..."
    )

    teacher_train = teacher.encode(
        train[
            "text"
        ].tolist()
    )

    teacher_val = teacher.encode(
        normal_val[
            "text"
        ].tolist()
    )

    # ------------------------------------------------------------
    # STUDENT
    # ------------------------------------------------------------

    print(
        "\nBuilding tokenizer..."
    )

    tokenizer = teacher.tokenizer

    vocab_size = (
        tokenizer.vocab_size
    )

    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id
        is not None
        else 0
    )

    model = Student(
        vocab_size=vocab_size,
        num_classes=len(labels),
        pad_token_id=pad_token_id,
    )

    model.to(
        device
    )

    print(
        "Student parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        ),
    )

    # ------------------------------------------------------------
    # TEACHER PROTOTYPES
    # ------------------------------------------------------------

    prototypes = (
        build_teacher_prototypes(
            teacher_train,
            labels,
            train[
                "intent"
            ].tolist(),
        )
    )

    prototypes = prototypes.to(
        device
    )

    # ------------------------------------------------------------
    # TOKENS
    # ------------------------------------------------------------

    print(
        "\nTokenizing training data..."
    )

    train_tokens = (
        tokenize_dataframe(
            tokenizer,
            train,
            device,
        )
    )

    val_tokens = (
        tokenize_dataframe(
            tokenizer,
            normal_val,
            device,
        )
    )

    train_targets = torch.tensor(
        [
            label_to_id[x]
            for x in train[
                "intent"
            ]
        ],
        dtype=torch.long,
        device=device,
    )

    val_targets = torch.tensor(
        [
            label_to_id[x]
            for x in normal_val[
                "intent"
            ]
        ],
        dtype=torch.long,
        device=device,
    )

    # ------------------------------------------------------------
    # WEIGHTS
    # ------------------------------------------------------------

    sample_weights = []

    for text in train[
        "text"
    ]:

        w = 1.0

        if is_critical(text):
            w *= CRITICAL_WEIGHT

        if text in set(
            hard["text"]
            if len(hard)
            else []
        ):
            w *= HARD_NEGATIVE_WEIGHT

        sample_weights.append(
            w
        )

    sample_weights = torch.tensor(
        sample_weights,
        dtype=torch.float32,
        device=device,
    )

    # ------------------------------------------------------------
    # OPTIMIZER
    # ------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )
    )

    # ------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------

    best_val = -1.0

    best_state = None

    history = []

    n = len(train)

    print(
        "\nStarting V5 training..."
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        order = torch.randperm(
            n,
            device=device,
        )

        epoch_losses = []

        for start in range(
            0,
            n,
            BATCH_SIZE,
        ):

            idx = order[
                start:
                start + BATCH_SIZE
            ]

            input_ids = (
                train_tokens[
                    "input_ids"
                ][idx]
            )

            attention_mask = (
                train_tokens[
                    "attention_mask"
                ][idx]
            )

            target = (
                train_targets[
                    idx
                ]
            )

            teacher_emb = (
                teacher_train[
                    idx.cpu()
                ].to(device)
            )

            weights = (
                sample_weights[
                    idx
                ]
            )

            student_emb, logits = (
                model(
                    input_ids,
                    attention_mask,
                )
            )

            # ----------------------------------------------------
            # 1. HARD CLASSIFICATION
            # ----------------------------------------------------

            loss_cls = (
                weighted_cross_entropy(
                    logits,
                    target,
                    weights,
                )
            )

            # ----------------------------------------------------
            # 2. E5 EMBEDDING KD
            # ----------------------------------------------------

            projected_teacher = (
                model.teacher_projection(
                    teacher_emb
                )
            )

            projected_teacher = (
                F.normalize(
                    projected_teacher,
                    p=2,
                    dim=-1,
                )
            )

            loss_kd = (
                embedding_kd_loss(
                    student_emb,
                    projected_teacher,
                )
            )

            # ----------------------------------------------------
            # 3. PROTOTYPE DISTILLATION
            # ----------------------------------------------------

            loss_proto = (
                prototype_loss(
                    student_emb,
                    projected_teacher,
                    F.normalize(
                        model.teacher_projection(
                            prototypes
                        ),
                        p=2,
                        dim=-1,
                    ),
                    target,
                )
            )

            # ----------------------------------------------------
            # 4. CONTRASTIVE
            # ----------------------------------------------------

            loss_contrastive = (
                supervised_contrastive_loss(
                    student_emb,
                    target,
                )
            )

            loss = (
                ALPHA_CLS
                * loss_cls
                +
                ALPHA_KD
                * loss_kd
                +
                ALPHA_PROTO
                * loss_proto
                +
                ALPHA_CONTRASTIVE
                * loss_contrastive
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            epoch_losses.append(
                float(
                    loss.detach()
                    .cpu()
                )
            )

        scheduler.step()

        val_acc, val_f1, _, _ = (
            evaluate(
                model,
                val_tokens,
                val_targets,
                labels,
            )
        )

        mean_loss = float(
            np.mean(
                epoch_losses
            )
        )

        print(
            f"Epoch {epoch:02d} | "
            f"loss={mean_loss:.4f} | "
            f"val={val_acc*100:.2f}% | "
            f"valF1={val_f1*100:.2f}%"
        )

        history.append(
            {
                "epoch": epoch,
                "loss": mean_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
            }
        )

        # Select by validation F1.
        if val_f1 > best_val:

            best_val = val_f1

            best_state = {
                k: v.detach()
                .cpu()
                .clone()
                for k, v in model.state_dict()
                .items()
            }

    # ------------------------------------------------------------
    # BEST MODEL
    # ------------------------------------------------------------

    if best_state is None:

        raise RuntimeError(
            "No checkpoint was selected."
        )

    model.load_state_dict(
        best_state
    )

    val_acc, val_f1, pred, truth = (
        evaluate(
            model,
            val_tokens,
            val_targets,
            labels,
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "V5 FINAL VALIDATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Validation accuracy : "
        f"{val_acc*100:.2f}%"
    )

    print(
        f"Validation Macro F1 : "
        f"{val_f1*100:.2f}%"
    )

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            truth,
            pred,
            labels=list(
                range(
                    len(labels)
                )
            ),
            target_names=labels,
            digits=4,
            zero_division=0,
        )
    )

    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------

    checkpoint_path = (
        OUTPUT_DIR
        / "v5_e5_error_driven_student_fp32.pt"
    )

    checkpoint = {
        "state_dict":
            model.state_dict(),

        "labels":
            labels,

        "vocab_size":
            vocab_size,

        "pad_token_id":
            pad_token_id,

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

        "teacher":
            TEACHER_NAME,

        "training":
            {
                "normal_rows":
                    len(normal_train),

                "hard_negative_rows":
                    len(hard),

                "total_train_rows":
                    len(train),

                "validation_rows":
                    len(normal_val),

                "epochs":
                    args.epochs,

                "best_val_accuracy":
                    val_acc,

                "best_val_macro_f1":
                    val_f1,

                "alpha_cls":
                    ALPHA_CLS,

                "alpha_kd":
                    ALPHA_KD,

                "alpha_proto":
                    ALPHA_PROTO,

                "alpha_contrastive":
                    ALPHA_CONTRASTIVE,

                "critical_weight":
                    CRITICAL_WEIGHT,

                "hard_negative_weight":
                    HARD_NEGATIVE_WEIGHT,

                "temperature":
                    TEMPERATURE,
            },

        "history":
            history,

        "safety":
            {
                "unseen_595_used":
                    False,

                "v3_modified":
                    False,

                "onnx_exported":
                    False,

                "int8_exported":
                    False,
            },
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    manifest_path = (
        OUTPUT_DIR
        / "v5_training_manifest.json"
    )

    manifest = {
        "checkpoint":
            str(
                checkpoint_path
            ),

        "teacher":
            TEACHER_NAME,

        "normal_source":
            str(
                NORMAL_SOURCE
            ),

        "hard_negative_rows":
            len(hard),

        "validation_rows":
            len(normal_val),

        "unseen_595_used":
            False,

        "v3_modified":
            False,

        "onnx_exported":
            False,

        "int8_exported":
            False,

        "best_validation_accuracy":
            val_acc,

        "best_validation_macro_f1":
            val_f1,
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    history_path = (
        OUTPUT_DIR
        / "v5_training_history.csv"
    )

    pd.DataFrame(
        history
    ).to_csv(
        history_path,
        index=False,
    )

    print(
        "\nSaved checkpoint:"
    )

    print(
        checkpoint_path
    )

    print(
        "\nSaved manifest:"
    )

    print(
        manifest_path
    )

    print(
        "\nSaved history:"
    )

    print(
        history_path
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "V3 was NOT modified."
    )

    print(
        "595-row unseen set was NOT used."
    )

    print(
        "V5 was NOT exported to ONNX."
    )

    print(
        "V5 was NOT quantized."
    )

    print(
        "\nNEXT:"
    )

    print(
        "Benchmark V5 FP32 against the locked V3 "
        "using the untouched 595-row unseen set."
    )


if __name__ == "__main__":
    main()
