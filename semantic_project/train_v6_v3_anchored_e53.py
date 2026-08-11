#!/usr/bin/env python3
"""
V6 — V3-ANCHORED + E5 SEMANTIC DISTILLATION

Production-safe research candidate.

Core idea:
    Start from the proven V3 student checkpoint instead of replacing
    its representation. Add E5-small as a frozen semantic teacher.

Training signals:
    1. Supervised intent classification
    2. E5 semantic distillation through a learned 384 -> 64 adapter
    3. V3-anchor loss to prevent catastrophic drift
    4. Hard-negative weighting
    5. Critical-command weighting
    6. Small contrastive loss

Safety:
    - The locked 595-row unseen test is NEVER loaded.
    - V3 checkpoint is read-only.
    - No ONNX export.
    - No INT8.
    - No threshold fitting.

Important:
    V6 keeps the V3 student architecture:
        vocab   = 895
        dim     = 64
        layers  = 2
        heads   = 4
        FFN     = 128
        max_len = 24

Run:
    python3 train_v6_v3_anchored_e5.py --epochs 12

Then benchmark V6 against the locked V3 before any export.
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
# PATHS
# ================================================================

ROOT = Path(
    "/Users/shuklam/IntentClassifier/semantic_project"
)

V3_CHECKPOINT = (
    ROOT
    / "tiny_semantic_student_v3_error_driven"
    / "student_v3_fp32.pt"
)

NORMAL_SOURCE = (
    ROOT
    / "fine_tuned_test_predictions.csv"
)

HARD_NEGATIVE_CANDIDATES = [
    ROOT
    / "production_calibration_v2"
    / "production_hard_negative.csv",

    ROOT
    / "tiny_semantic_student_v3_error_driven"
    / "hard_negative_results.csv",
]

OUTPUT_DIR = (
    ROOT
    / "v6_v3_anchored_e5"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TEACHER_NAME = (
    "intfloat/multilingual-e5-small"
)

SEED = 42

MAX_LEN = 24
STUDENT_DIM = 64
LAYERS = 2
HEADS = 4
FFN = 128

BATCH_SIZE = 32

DEFAULT_EPOCHS = 12

# Conservative LR because V6 starts from the proven V3 weights.
LR = 3e-5
WEIGHT_DECAY = 1e-4

# Classification remains dominant.
ALPHA_CLS = 1.00

# Small semantic teacher pressure.
ALPHA_E5 = 0.12

# Strong protection against moving away from V3.
ALPHA_V3_ANCHOR = 0.30

# Small supervised contrastive signal.
ALPHA_CONTRASTIVE = 0.08

HARD_NEGATIVE_WEIGHT = 1.75
CRITICAL_WEIGHT = 1.50

TEMPERATURE = 0.10


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
            f"Training source not found:\n{NORMAL_SOURCE}"
        )

    df = pd.read_csv(
        NORMAL_SOURCE
    )

    text_col = find_column(
        df,
        ["text", "utterance", "query", "sentence"],
    )

    intent_col = find_column(
        df,
        ["intent", "label", "target"],
    )

    if text_col is None or intent_col is None:
        raise ValueError(
            "Training CSV needs text + intent columns."
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

    return out.drop_duplicates(
        subset=["text", "intent"]
    ).reset_index(drop=True)


def find_existing(paths):

    for p in paths:
        if p.exists():
            return p

    return None


def load_hard_negatives():

    path = find_existing(
        HARD_NEGATIVE_CANDIDATES
    )

    if path is None:
        print(
            "\nHard-negative source: NOT FOUND"
        )
        return pd.DataFrame(
            columns=["text", "intent"]
        )

    print(
        "\nHard-negative source:"
    )
    print(path)

    df = pd.read_csv(path)

    text_col = find_column(
        df,
        ["text", "utterance", "query", "sentence"],
    )

    intent_col = find_column(
        df,
        ["intent", "label", "target", "expected"],
    )

    if text_col is None or intent_col is None:
        raise ValueError(
            f"Hard-negative CSV needs text + intent/expected: {path}"
        )

    return pd.DataFrame(
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
    ).drop_duplicates(
        subset=["text", "intent"]
    ).reset_index(drop=True)


def split_by_intent(
    df,
    val_per_intent=20,
):

    train_parts = []
    val_parts = []

    for intent, group in df.groupby(
        "intent"
    ):

        group = (
            group
            .sample(
                frac=1.0,
                random_state=SEED,
            )
            .reset_index(drop=True)
        )

        n = min(
            val_per_intent,
            max(
                1,
                len(group) // 5,
            ),
        )

        val_parts.append(
            group.iloc[:n]
        )

        train_parts.append(
            group.iloc[n:]
        )

    return (
        pd.concat(
            train_parts,
            ignore_index=True,
        ),
        pd.concat(
            val_parts,
            ignore_index=True,
        ),
    )


# ================================================================
# CRITICAL COMMANDS
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

    text = normalize_text(
        text
    ).lower()

    return any(
        phrase in text
        for phrase in CRITICAL_PHRASES
    )


# ================================================================
# V3 STUDENT ARCHITECTURE
# ================================================================

class V3Student(nn.Module):

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

        layer = (
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
                layer,
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

        # E5-small = 384 dimensions.
        # Adapter is deliberately small; the V3
        # representation remains the production anchor.
        self.e5_adapter = (
            nn.Sequential(
                nn.Linear(
                    384,
                    128,
                ),
                nn.GELU(),
                nn.Linear(
                    128,
                    STUDENT_DIM,
                ),
                nn.LayerNorm(
                    STUDENT_DIM
                ),
            )
        )

    def encode(
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

        return semantic

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


# ================================================================
# TEACHER
# ================================================================

class E5Teacher:

    def __init__(
        self,
        device,
    ):

        print(
            "\nLoading E5 teacher:"
        )
        print(
            TEACHER_NAME
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

        for p in self.model.parameters():
            p.requires_grad = False

        self.device = device

    @torch.no_grad()
    def encode(
        self,
        texts,
        batch_size=32,
    ):

        result = []

        for start in range(
            0,
            len(texts),
            batch_size,
        ):

            batch = texts[
                start:start + batch_size
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
                k: v.to(self.device)
                for k, v in tokens.items()
            }

            output = self.model(
                **tokens
            )

            hidden = (
                output.last_hidden_state
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

            result.append(
                pooled.cpu()
            )

        return torch.cat(
            result,
            dim=0,
        )


# ================================================================
# TOKENIZER
# ================================================================

def load_v3_vocab():

    candidates = [
        ROOT
        / "tiny_semantic_student_v2_balanced"
        / "vocab.json",

        ROOT
        / "tiny_semantic_student_v3_error_driven"
        / "vocab.json",

        ROOT
        / "tiny_semantic_student_v3_fp32"
        / "vocab.json",

        ROOT / "vocab.json",
    ]

    path = find_existing(
        candidates
    )

    if path is None:

        found = [
            p
            for p in ROOT.rglob(
                "vocab.json"
            )
            if ".venv" not in str(p)
        ]

        if found:
            found.sort(
                key=lambda p: (
                    0
                    if "tiny_semantic_student"
                    in str(p)
                    else 1,
                    len(str(p)),
                )
            )
            path = found[0]

    if path is None:
        raise FileNotFoundError(
            "V3 vocab.json not found."
        )

    obj = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if (
        isinstance(obj, dict)
        and "vocab" in obj
    ):
        obj = obj["vocab"]

    if not isinstance(obj, dict):
        raise ValueError(
            f"Unsupported vocab format: {path}"
        )

    print(
        "\nV3 vocabulary:"
    )
    print(path)
    print(
        "Vocabulary size:",
        len(obj),
    )

    if len(obj) != 895:
        raise RuntimeError(
            f"Expected V3 vocabulary size 895, got {len(obj)}."
        )

    return obj


def tokenize_v3(
    text,
    vocab,
):

    text = normalize_text(
        text
    ).lower()

    unk = vocab.get(
        "<unk>",
        vocab.get(
            "[UNK]",
            1,
        ),
    )

    pad = vocab.get(
        "<pad>",
        vocab.get(
            "[PAD]",
            0,
        ),
    )

    cls = vocab.get(
        "<cls>",
        vocab.get(
            "[CLS]",
            None,
        ),
    )

    sep = vocab.get(
        "<sep>",
        vocab.get(
            "[SEP]",
            None,
        ),
    )

    ids = []

    if cls is not None:
        ids.append(cls)

    for token in text.split():
        ids.append(
            vocab.get(
                token,
                unk,
            )
        )

    if sep is not None:
        ids.append(sep)

    ids = ids[:MAX_LEN]

    if len(ids) < MAX_LEN:
        ids += [
            pad
        ] * (
            MAX_LEN - len(ids)
        )

    return np.asarray(
        ids,
        dtype=np.int64,
    )


# ================================================================
# LOSS
# ================================================================

def contrastive_loss(
    embeddings,
    labels,
):

    if len(embeddings) < 2:
        return embeddings.sum() * 0.0

    z = F.normalize(
        embeddings,
        p=2,
        dim=-1,
    )

    sim = (
        z @ z.T
    ) / TEMPERATURE

    eye = torch.eye(
        len(z),
        device=z.device,
        dtype=torch.bool,
    )

    sim = sim.masked_fill(
        eye,
        -1e9,
    )

    labels = labels.view(
        -1,
        1,
    )

    positive = (
        labels == labels.T
    ).masked_fill(
        eye,
        False,
    )

    log_prob = F.log_softmax(
        sim,
        dim=1,
    )

    count = positive.sum(
        dim=1
    ).clamp(min=1)

    loss = -(
        log_prob
        * positive.float()
    ).sum(1) / count

    valid = (
        positive.sum(1) > 0
    )

    if not valid.any():
        return embeddings.sum() * 0.0

    return loss[
        valid
    ].mean()


# ================================================================
# EVALUATION
# ================================================================

@torch.no_grad()
def evaluate(
    model,
    tokens,
    targets,
):

    model.eval()

    _, logits = model(
        tokens["input_ids"],
        tokens["attention_mask"],
    )

    pred = (
        logits.argmax(
            dim=1
        )
        .cpu()
        .numpy()
    )

    truth = (
        targets
        .cpu()
        .numpy()
    )

    return (
        accuracy_score(
            truth,
            pred,
        ),
        f1_score(
            truth,
            pred,
            average="macro",
            zero_division=0,
        ),
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
        "=" * 78
    )
    print(
        "V6 — V3-ANCHORED + E5 SEMANTIC DISTILLATION"
    )
    print(
        "=" * 78
    )

    print(
        "\nDevice:",
        device,
    )

    print(
        "\nSAFETY LOCKS:"
    )

    print(
        "595-row unseen test: NOT LOADED"
    )

    print(
        "V3 checkpoint: READ ONLY"
    )

    print(
        "ONNX export: NO"
    )

    print(
        "INT8: NO"
    )

    # ------------------------------------------------------------
    # Load V3 checkpoint metadata
    # ------------------------------------------------------------

    if not V3_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"V3 checkpoint not found:\n{V3_CHECKPOINT}"
        )

    v3_ckpt = torch.load(
        V3_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    labels = list(
        v3_ckpt.get(
            "labels",
            v3_ckpt.get(
                "intent_labels",
                [],
            ),
        )
    )

    if len(labels) != 11:
        raise RuntimeError(
            "V3 checkpoint must contain 11 labels."
        )

    vocab_size = int(
        v3_ckpt.get(
            "vocab_size",
            895,
        )
    )

    if vocab_size != 895:
        raise RuntimeError(
            f"V3 checkpoint vocab expected 895, got {vocab_size}."
        )

    print(
        "\nV3 architecture LOCKED:"
    )
    print(
        "vocab=895 | dim=64 | layers=2 | "
        "heads=4 | FFN=128 | max_len=24"
    )

    # ------------------------------------------------------------
    # Normal data
    # ------------------------------------------------------------

    normal = load_normal_data()

    if not set(
        labels
    ).issubset(
        set(
            normal["intent"]
        )
    ):
        raise RuntimeError(
            "Training data does not contain all V3 intents."
        )

    normal_train, normal_val = (
        split_by_intent(
            normal,
            val_per_intent=20,
        )
    )

    # ------------------------------------------------------------
    # Hard negatives
    # ------------------------------------------------------------

    hard = load_hard_negatives()

    hard = hard[
        hard["intent"].isin(
            labels
        )
    ].reset_index(
        drop=True
    )

    print(
        "\nNormal train rows:",
        len(normal_train),
    )

    print(
        "Normal validation rows:",
        len(normal_val),
    )

    print(
        "Hard-negative rows:",
        len(hard),
    )

    train = pd.concat(
        [
            normal_train,
            hard,
        ],
        ignore_index=True,
    ).drop_duplicates(
        subset=[
            "text",
            "intent",
        ]
    ).reset_index(
        drop=True
    )

    print(
        "V6 total train rows:",
        len(train),
    )

    # ------------------------------------------------------------
    # E5 teacher
    # ------------------------------------------------------------

    teacher = E5Teacher(
        device
    )

    print(
        "\nCaching E5 train embeddings..."
    )

    e5_train = teacher.encode(
        train[
            "text"
        ].tolist()
    )

    print(
        "Caching E5 validation embeddings..."
    )

    e5_val = teacher.encode(
        normal_val[
            "text"
        ].tolist()
    )

    # ------------------------------------------------------------
    # V3 tokenizer
    # ------------------------------------------------------------

    vocab = load_v3_vocab()

    pad_id = vocab.get(
        "<pad>",
        vocab.get(
            "[PAD]",
            0,
        ),
    )

    # ------------------------------------------------------------
    # Student
    # ------------------------------------------------------------

    model = V3Student(
        vocab_size=895,
        num_classes=11,
        pad_token_id=pad_id,
    )

    state = v3_ckpt.get(
        "state_dict",
        v3_ckpt,
    )

    # V3 checkpoint should load exactly into
    # the V3 architecture. The E5 adapter is new.
    missing, unexpected = (
        model.load_state_dict(
            state,
            strict=False,
        )
    )

    allowed_missing = {
        name
        for name in model.state_dict()
        if name.startswith(
            "e5_adapter."
        )
    }

    bad_missing = set(
        missing
    ) - allowed_missing

    if bad_missing or unexpected:
        raise RuntimeError(
            "V3 checkpoint compatibility failure.\n"
            f"Missing: {missing}\n"
            f"Unexpected: {unexpected}"
        )

    model.to(
        device
    )

    # Frozen V3 reference used for the real V3-anchor loss.
    frozen_v3 = V3Student(
        vocab_size=895,
        num_classes=11,
        pad_token_id=pad_id,
    )

    frozen_v3.load_state_dict(
        state,
        strict=False,
    )

    frozen_v3.to(device)
    frozen_v3.eval()

    for parameter in frozen_v3.parameters():
        parameter.requires_grad = False

    print(
        "\nV3 weights loaded into V6 student."
    )

    print(
        "Frozen V3 reference loaded for anchor loss."
    )

    # ------------------------------------------------------------
    # Tokenize
    # ------------------------------------------------------------

    train_ids = torch.tensor(
        np.stack(
            [
                tokenize_v3(
                    x,
                    vocab,
                )
                for x in train[
                    "text"
                ]
            ]
        ),
        dtype=torch.long,
        device=device,
    )

    val_ids = torch.tensor(
        np.stack(
            [
                tokenize_v3(
                    x,
                    vocab,
                )
                for x in normal_val[
                    "text"
                ]
            ]
        ),
        dtype=torch.long,
        device=device,
    )

    train_mask = (
        train_ids != pad_id
    ).long()

    val_mask = (
        val_ids != pad_id
    ).long()

    label_to_id = {
        label: i
        for i, label in enumerate(
            labels
        )
    }

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

    e5_train = e5_train.to(
        device
    )

    e5_val = e5_val.to(
        device
    )

    # ------------------------------------------------------------
    # Sample weights
    # ------------------------------------------------------------

    hard_texts = set(
        hard["text"].tolist()
    )

    sample_weights = []

    for text in train[
        "text"
    ]:

        w = 1.0

        if text in hard_texts:
            w *= HARD_NEGATIVE_WEIGHT

        if is_critical(text):
            w *= CRITICAL_WEIGHT

        sample_weights.append(
            w
        )

    sample_weights = torch.tensor(
        sample_weights,
        dtype=torch.float32,
        device=device,
    )

    # ------------------------------------------------------------
    # Optimizer
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

    best_f1 = -1.0
    best_state = None
    history = []

    n = len(train)

    # ------------------------------------------------------------
    # Train
    # ------------------------------------------------------------

    print(
        "\nStarting V6 training..."
    )

    print(
        "Initial model = V3 weights"
    )

    print(
        f"Epochs = {args.epochs}"
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

        losses = []

        for start in range(
            0,
            n,
            BATCH_SIZE,
        ):

            idx = order[
                start:start+BATCH_SIZE
            ]

            ids = train_ids[
                idx
            ]

            mask = train_mask[
                idx
            ]

            target = train_targets[
                idx
            ]

            teacher_emb = e5_train[
                idx
            ]

            weights = sample_weights[
                idx
            ]

            student_emb, logits = (
                model(
                    ids,
                    mask,
                )
            )

            # ----------------------------------------------------
            # 1. Classification
            # ----------------------------------------------------

            ce = F.cross_entropy(
                logits,
                target,
                reduction="none",
            )

            loss_cls = (
                ce * weights
            ).mean()

            # ----------------------------------------------------
            # 2. E5 semantic distillation
            # ----------------------------------------------------

            teacher_projected = (
                model.e5_adapter(
                    teacher_emb
                )
            )

            teacher_projected = F.normalize(
                teacher_projected,
                p=2,
                dim=-1,
            )

            loss_e5 = (
                1.0
                -
                (
                    student_emb
                    * teacher_projected
                ).sum(-1)
            ).mean()

            # ----------------------------------------------------
            # 3. V3 anchor
            #
            # Keep the learned V3 semantic representation close to
            # the proven baseline while E5 supplies a small semantic
            # improvement signal.
            # ----------------------------------------------------

            with torch.no_grad():
                frozen_v3_embedding = (
                    frozen_v3.encode(
                        ids,
                        mask,
                    )
                )

            loss_v3_anchor = (
                1.0
                -
                (
                    student_emb
                    * frozen_v3_embedding
                ).sum(-1)
            ).mean()

            # ----------------------------------------------------
            # 4. Contrastive
            # ----------------------------------------------------

            loss_con = contrastive_loss(
                student_emb,
                target,
            )

            loss = (
                ALPHA_CLS * loss_cls
                +
                ALPHA_E5 * loss_e5
                +
                ALPHA_V3_ANCHOR * loss_v3_anchor
                +
                ALPHA_CONTRASTIVE * loss_con
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                0.75,
            )

            optimizer.step()

            losses.append(
                float(
                    loss.detach()
                    .cpu()
                )
            )

        scheduler.step()

        val_acc, val_f1, _, _ = (
            evaluate(
                model,
                {
                    "input_ids":
                        val_ids,
                    "attention_mask":
                        val_mask,
                },
                val_targets,
            )
        )

        mean_loss = float(
            np.mean(losses)
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

        if val_f1 > best_f1:

            best_f1 = val_f1

            best_state = {
                k:
                    v.detach()
                    .cpu()
                    .clone()
                for k, v
                in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError(
            "No V6 checkpoint selected."
        )

    model.load_state_dict(
        best_state
    )

    val_acc, val_f1, pred, truth = (
        evaluate(
            model,
            {
                "input_ids":
                    val_ids,
                "attention_mask":
                    val_mask,
            },
            val_targets,
        )
    )

    print(
        "\n" + "=" * 78
    )

    print(
        "V6 FINAL VALIDATION"
    )

    print(
        "=" * 78
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
                range(11)
            ),
            target_names=labels,
            digits=4,
            zero_division=0,
        )
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    checkpoint_path = (
        OUTPUT_DIR
        / "v6_v3_anchored_e5_fp32.pt"
    )

    torch.save(
        {
            "state_dict":
                model.state_dict(),

            "labels":
                labels,

            "vocab_size":
                895,

            "pad_token_id":
                pad_id,

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

            "base_checkpoint":
                str(V3_CHECKPOINT),

            "validation_accuracy":
                val_acc,

            "validation_macro_f1":
                val_f1,

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

            "training":
                {
                    "normal_train_rows":
                        len(normal_train),

                    "hard_negative_rows":
                        len(hard),

                    "total_train_rows":
                        len(train),

                    "validation_rows":
                        len(normal_val),

                    "epochs":
                        args.epochs,

                    "lr":
                        LR,

                    "alpha_cls":
                        ALPHA_CLS,

                    "alpha_e5":
                        ALPHA_E5,

                    "alpha_v3_anchor":
                        ALPHA_V3_ANCHOR,

                    "alpha_contrastive":
                        ALPHA_CONTRASTIVE,
                },
        },
        checkpoint_path,
    )

    manifest_path = (
        OUTPUT_DIR
        / "v6_training_manifest.json"
    )

    manifest = {
        "checkpoint":
            str(checkpoint_path),

        "base_v3":
            str(V3_CHECKPOINT),

        "teacher":
            TEACHER_NAME,

        "normal_train_rows":
            len(normal_train),

        "hard_negative_rows":
            len(hard),

        "validation_rows":
            len(normal_val),

        "validation_accuracy":
            val_acc,

        "validation_macro_f1":
            val_f1,

        "unseen_595_used":
            False,

        "v3_modified":
            False,

        "onnx_exported":
            False,

        "int8_exported":
            False,

        "next_step":
            "Benchmark V6 FP32 against locked V3 on untouched 595-row unseen set.",
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
        / "v6_training_history.csv"
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
        "V6 was NOT exported to ONNX."
    )

    print(
        "V6 was NOT quantized."
    )

    print(
        "\nNEXT:"
    )

    print(
        "Benchmark V6 FP32 against locked V3."
    )


if __name__ == "__main__":
    main()
