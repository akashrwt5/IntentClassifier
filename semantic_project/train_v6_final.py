#!/usr/bin/env python3
"""
V6 FINAL — V3-ANCHORED E5 SEMANTIC TRAINING

This version is intentionally rebuilt around the ACTUAL V3 checkpoint
structure observed in the project:

    embedding:       895 -> 64
    transformer:     2 layers, 4 heads, FFN 128
    classifier:      64 -> 64 -> 11

The V3 checkpoint is loaded exactly. No "semantic_projection" and no
"classifier.weight" are expected.

V6 adds ONLY a new E5 adapter:

    multilingual-e5-small: 384 -> 128 -> 64

The 595-row unseen set is never loaded.

No ONNX export.
No INT8.
No threshold fitting.
V3 checkpoint is read-only.

Run:
    python3 train_v6_final.py --epochs 12
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


# ============================================================
# CONFIG
# ============================================================

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
    ROOT / "production_calibration_v2" / "production_hard_negative.csv",
    ROOT / "tiny_semantic_student_v3_error_driven" / "hard_negative_results.csv",
]

OUTPUT_DIR = ROOT / "v6_final_e5"

TEACHER_NAME = "intfloat/multilingual-e5-small"

SEED = 42
MAX_LEN = 24
VOCAB_SIZE = 895
EMBED_DIM = 64
LAYERS = 2
HEADS = 4
FFN = 128
NUM_CLASSES = 11

BATCH_SIZE = 32
DEFAULT_EPOCHS = 12

# Conservative because V3 is already strong.
LR = 2e-5
WEIGHT_DECAY = 1e-4

# V3 behavior remains dominant.
W_CLS = 1.00
W_E5 = 0.08
W_V3_LOGIT_ANCHOR = 0.35
W_V3_EMB_ANCHOR = 0.20
W_CONTRASTIVE = 0.04

TEMPERATURE = 0.10

HARD_WEIGHT = 1.50
CRITICAL_WEIGHT = 1.25


LOCKED_LABELS = [
    "device.memory.change",
    "device.volume.decrease",
    "device.volume.increase",
    "device.volume.mute",
    "device.volume.unmute",
    "find.phone.locate",
    "help.reminder.show",
    "reminders.task.complete",
    "reminders.task.create",
    "streaming.session.start",
    "streaming.session.stop",
]


CRITICAL_PHRASES = [
    "make it louder",
    "make it quieter",
    "mute it",
    "unmute it",
    "turn off",
    "turn the sound back on",
    "completely silent",
    "keep it on",
]


# ============================================================
# UTIL
# ============================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_text(x):
    return re.sub(r"\s+", " ", str(x).strip())


def find_column(df, names):
    lookup = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def is_critical(text):
    t = normalize_text(text).lower()
    return any(p in t for p in CRITICAL_PHRASES)


def find_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


# ============================================================
# DATA
# ============================================================

def load_training_data():
    if not NORMAL_SOURCE.exists():
        raise FileNotFoundError(
            f"Training source not found:\n{NORMAL_SOURCE}"
        )

    df = pd.read_csv(NORMAL_SOURCE)

    text_col = find_column(
        df, ["text", "utterance", "query", "sentence"]
    )
    intent_col = find_column(
        df, ["intent", "label", "target"]
    )

    if text_col is None or intent_col is None:
        raise RuntimeError(
            "Training CSV must contain text/utterance and intent/label columns."
        )

    out = pd.DataFrame({
        "text": df[text_col].map(normalize_text),
        "intent": df[intent_col].astype(str).str.strip(),
    })

    out = out.drop_duplicates(
        subset=["text", "intent"]
    ).reset_index(drop=True)

    missing = sorted(
        set(LOCKED_LABELS) - set(out["intent"])
    )

    if missing:
        raise RuntimeError(
            f"Training data missing intents: {missing}"
        )

    return out


def load_hard_negatives():
    path = find_existing(HARD_NEGATIVE_CANDIDATES)

    if path is None:
        print("\nHard negatives: not found")
        return pd.DataFrame(columns=["text", "intent"])

    df = pd.read_csv(path)

    text_col = find_column(
        df, ["text", "utterance", "query", "sentence"]
    )
    intent_col = find_column(
        df, ["intent", "label", "target", "expected"]
    )

    if text_col is None or intent_col is None:
        raise RuntimeError(
            f"Hard-negative file needs text + intent columns:\n{path}"
        )

    out = pd.DataFrame({
        "text": df[text_col].map(normalize_text),
        "intent": df[intent_col].astype(str).str.strip(),
    })

    out = out[
        out["intent"].isin(LOCKED_LABELS)
    ]

    return out.drop_duplicates(
        subset=["text", "intent"]
    ).reset_index(drop=True)


def split_by_intent(df, val_per_intent=20):
    train_parts = []
    val_parts = []

    for intent, group in df.groupby("intent"):
        group = group.sample(
            frac=1.0,
            random_state=SEED,
        ).reset_index(drop=True)

        n = min(
            val_per_intent,
            max(1, len(group) // 5),
        )

        val_parts.append(group.iloc[:n])
        train_parts.append(group.iloc[n:])

    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(val_parts, ignore_index=True),
    )


# ============================================================
# VOCAB
# ============================================================

def load_vocab():
    candidates = [
        ROOT / "tiny_semantic_student_v2_balanced" / "vocab.json",
        ROOT / "tiny_semantic_student_v3_balanced" / "vocab.json",
        ROOT / "tiny_semantic_student_v3_fp32" / "vocab.json",
    ]

    path = find_existing(candidates)

    if path is None:
        found = list(ROOT.rglob("vocab.json"))
        found = [
            p for p in found
            if ".venv" not in str(p)
        ]
        if not found:
            raise FileNotFoundError("No vocab.json found.")
        path = sorted(found, key=lambda x: len(str(x)))[0]

    obj = json.loads(
        path.read_text(encoding="utf-8")
    )

    if isinstance(obj, dict) and "vocab" in obj:
        obj = obj["vocab"]

    if not isinstance(obj, dict):
        raise RuntimeError(
            f"Unsupported vocabulary format: {path}"
        )

    if len(obj) != VOCAB_SIZE:
        raise RuntimeError(
            f"Expected vocabulary size {VOCAB_SIZE}, got {len(obj)}"
        )

    print("\nVocabulary:")
    print(path)
    print("Vocabulary size:", len(obj))

    return obj


def token_id(vocab, candidates, default):
    for key in candidates:
        if key in vocab:
            return int(vocab[key])
    return default


def tokenize(text, vocab):
    pad_id = token_id(
        vocab,
        ["<pad>", "[PAD]"],
        0,
    )
    unk_id = token_id(
        vocab,
        ["<unk>", "[UNK]"],
        1,
    )

    cls_id = None
    sep_id = None

    for key in ["<cls>", "[CLS]"]:
        if key in vocab:
            cls_id = int(vocab[key])
            break

    for key in ["<sep>", "[SEP]"]:
        if key in vocab:
            sep_id = int(vocab[key])
            break

    ids = []

    if cls_id is not None:
        ids.append(cls_id)

    for token in normalize_text(text).lower().split():
        ids.append(
            int(vocab.get(token, unk_id))
        )

    if sep_id is not None:
        ids.append(sep_id)

    ids = ids[:MAX_LEN]

    if len(ids) < MAX_LEN:
        ids.extend(
            [pad_id] * (MAX_LEN - len(ids))
        )

    return np.asarray(
        ids,
        dtype=np.int64,
    )


# ============================================================
# EXACT V3 MODEL
# ============================================================

class ExactV3(nn.Module):
    """
    Exact structure required by the observed V3 checkpoint.

    classifier:
        classifier.0 = Linear(64, 64)
        classifier.1 = GELU
        classifier.2 = Dropout
        classifier.3 = Linear(64, 11)
    """

    def __init__(self, pad_token_id):
        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            EMBED_DIM,
            padding_idx=pad_token_id,
        )

        self.position = nn.Embedding(
            MAX_LEN,
            EMBED_DIM,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
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
            EMBED_DIM
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                EMBED_DIM,
                EMBED_DIM,
            ),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(
                EMBED_DIM,
                NUM_CLASSES,
            ),
        )

    def encode(
        self,
        input_ids,
        attention_mask,
    ):
        seq_len = input_ids.shape[1]

        pos = torch.arange(
            seq_len,
            device=input_ids.device,
        ).unsqueeze(0)

        x = (
            self.embedding(input_ids)
            + self.position(pos)
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
            (x * mask).sum(dim=1)
            /
            mask.sum(dim=1).clamp(
                min=1e-9
            )
        )

        x = self.norm(x)

        return F.normalize(
            x,
            p=2,
            dim=-1,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
    ):
        emb = self.encode(
            input_ids,
            attention_mask,
        )

        logits = self.classifier(
            emb
        )

        return emb, logits


# ============================================================
# V6 MODEL
# ============================================================

class V6Model(ExactV3):
    def __init__(self, pad_token_id):
        super().__init__(
            pad_token_id
        )

        # ONLY new V6 component.
        self.e5_adapter = nn.Sequential(
            nn.Linear(384, 128),
            nn.GELU(),
            nn.Linear(128, EMBED_DIM),
            nn.LayerNorm(EMBED_DIM),
        )


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in [
            "state_dict",
            "model_state_dict",
            "model",
        ]:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value

    if isinstance(checkpoint, dict):
        tensor_values = [
            v for v in checkpoint.values()
            if torch.is_tensor(v)
        ]
        if tensor_values:
            return checkpoint

    raise RuntimeError(
        "Could not locate a PyTorch state_dict in V3 checkpoint."
    )


def load_exact_v3(path, pad_id, device):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    state = extract_state_dict(
        checkpoint
    )

    # We know the exact keys from the actual checkpoint.
    required = [
        "classifier.0.weight",
        "classifier.0.bias",
        "classifier.3.weight",
        "classifier.3.bias",
    ]

    for key in required:
        if key not in state:
            raise RuntimeError(
                f"Actual V3 checkpoint is missing {key}"
            )

    expected_shapes = {
        "classifier.0.weight": (64, 64),
        "classifier.0.bias": (64,),
        "classifier.3.weight": (11, 64),
        "classifier.3.bias": (11,),
    }

    for key, shape in expected_shapes.items():
        got = tuple(state[key].shape)
        if got != shape:
            raise RuntimeError(
                f"{key}: expected {shape}, got {got}"
            )

    model = ExactV3(
        pad_token_id=pad_id
    )

    model_state = model.state_dict()

    # Copy only exact V3 keys.
    missing = []
    mismatched = []

    for key in model_state:
        if key not in state:
            missing.append(key)
            continue

        if tuple(model_state[key].shape) != tuple(
            state[key].shape
        ):
            mismatched.append(
                (
                    key,
                    tuple(model_state[key].shape),
                    tuple(state[key].shape),
                )
            )

    if missing or mismatched:
        raise RuntimeError(
            "Actual V3 checkpoint does not match the reconstructed "
            "V3 architecture.\n"
            f"Missing: {missing}\n"
            f"Mismatched: {mismatched}"
        )

    for key in model_state:
        model_state[key].copy_(
            state[key]
        )

    model.load_state_dict(
        model_state,
        strict=True,
    )

    model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    print(
        "\nEXACT V3 CHECKPOINT LOAD: PASS"
    )
    print(
        "Classifier: 64 -> 64 -> 11"
    )
    print(
        "V3 parameters: frozen"
    )

    return model, checkpoint


# ============================================================
# E5 TEACHER
# ============================================================

class E5Teacher:
    def __init__(self, device):
        print(
            "\nLoading:",
            TEACHER_NAME,
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

        self.model.to(device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False

        self.device = device

    @torch.no_grad()
    def encode(self, texts):
        chunks = []

        for start in range(
            0,
            len(texts),
            BATCH_SIZE,
        ):
            batch = texts[
                start:start + BATCH_SIZE
            ]

            tokens = self.tokenizer(
                [
                    f"query: {x}"
                    for x in batch
                ],
                max_length=128,
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
                tokens["attention_mask"]
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

            chunks.append(
                pooled.cpu()
            )

        return torch.cat(
            chunks,
            dim=0,
        )


# ============================================================
# LOSS
# ============================================================

def contrastive_loss(emb, labels):
    if len(emb) < 2:
        return emb.sum() * 0.0

    z = F.normalize(
        emb,
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

    same = (
        labels[:, None]
        ==
        labels[None, :]
    )

    positive = same & (~eye)

    valid = positive.sum(1) > 0

    if not valid.any():
        return emb.sum() * 0.0

    logp = F.log_softmax(
        sim,
        dim=1,
    )

    count = positive.sum(1).clamp(
        min=1
    )

    loss = -(
        logp * positive.float()
    ).sum(1) / count

    return loss[valid].mean()


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, ids, mask, targets):
    model.eval()

    _, logits = model(
        ids,
        mask,
    )

    pred = logits.argmax(
        dim=1
    ).cpu().numpy()

    truth = targets.cpu().numpy()

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


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )
    args = parser.parse_args()

    seed_everything(SEED)

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("V6 FINAL — V3-ANCHORED + E5")
    print("=" * 78)

    print("\nDevice:", device)

    print("\nSAFETY LOCKS")
    print("595-row unseen: NOT LOADED")
    print("V3 checkpoint: READ ONLY")
    print("ONNX: NO")
    print("INT8: NO")
    print("Threshold fitting: NO")

    if not V3_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"V3 checkpoint not found:\n{V3_CHECKPOINT}"
        )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    data = load_training_data()
    hard = load_hard_negatives()

    normal_train, val = split_by_intent(
        data,
        val_per_intent=20,
    )

    train = pd.concat(
        [normal_train, hard],
        ignore_index=True,
    ).drop_duplicates(
        subset=["text", "intent"]
    ).reset_index(drop=True)

    print("\nNormal train:", len(normal_train))
    print("Validation:", len(val))
    print("Hard negatives:", len(hard))
    print("V6 train:", len(train))

    # --------------------------------------------------------
    # Vocabulary
    # --------------------------------------------------------

    vocab = load_vocab()

    pad_id = token_id(
        vocab,
        ["<pad>", "[PAD]"],
        0,
    )

    # --------------------------------------------------------
    # Exact V3 frozen model
    # --------------------------------------------------------

    frozen_v3, checkpoint = load_exact_v3(
        V3_CHECKPOINT,
        pad_id,
        device,
    )

    # --------------------------------------------------------
    # V6 starts as EXACT COPY of V3
    # --------------------------------------------------------

    model = V6Model(
        pad_token_id=pad_id
    )

    v3_state = frozen_v3.state_dict()

    model_state = model.state_dict()

    for key in v3_state:
        model_state[key].copy_(
            v3_state[key]
        )

    model.load_state_dict(
        model_state,
        strict=True,
    )

    model.to(device)

    print(
        "\nV6 initialization: exact V3 weights"
    )

    print(
        "New component: E5 adapter 384 -> 128 -> 64"
    )

    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    train_ids = torch.tensor(
        np.stack([
            tokenize(x, vocab)
            for x in train["text"]
        ]),
        dtype=torch.long,
        device=device,
    )

    val_ids = torch.tensor(
        np.stack([
            tokenize(x, vocab)
            for x in val["text"]
        ]),
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
            LOCKED_LABELS
        )
    }

    train_targets = torch.tensor(
        [
            label_to_id[x]
            for x in train["intent"]
        ],
        dtype=torch.long,
        device=device,
    )

    val_targets = torch.tensor(
        [
            label_to_id[x]
            for x in val["intent"]
        ],
        dtype=torch.long,
        device=device,
    )

    # --------------------------------------------------------
    # E5 teacher
    # --------------------------------------------------------

    teacher = E5Teacher(
        device
    )

    print(
        "\nCaching E5 train embeddings..."
    )

    e5_train = teacher.encode(
        train["text"].tolist()
    ).to(device)

    # --------------------------------------------------------
    # Sample weights
    # --------------------------------------------------------

    hard_texts = set(
        hard["text"].tolist()
    )

    weights = []

    for text in train["text"]:
        w = 1.0

        if text in hard_texts:
            w *= HARD_WEIGHT

        if is_critical(text):
            w *= CRITICAL_WEIGHT

        weights.append(w)

    weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    best_f1 = -1.0
    best_state = None
    history = []

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print(
        "\nStarting V6 training..."
    )

    print(
        f"Epochs: {args.epochs}"
    )

    print(
        f"LR: {LR}"
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        order = torch.randperm(
            len(train),
            device=device,
        )

        epoch_losses = []

        for start in range(
            0,
            len(train),
            BATCH_SIZE,
        ):
            idx = order[
                start:start + BATCH_SIZE
            ]

            ids = train_ids[idx]
            mask = train_mask[idx]
            target = train_targets[idx]
            tw = e5_train[idx]
            sw = weights[idx]

            student_emb, logits = model(
                ids,
                mask,
            )

            # 1. Main supervised intent objective.
            ce = F.cross_entropy(
                logits,
                target,
                reduction="none",
            )

            loss_cls = (
                ce * sw
            ).mean()

            # 2. E5 semantic guidance.
            e5_target = model.e5_adapter(
                tw
            )

            e5_target = F.normalize(
                e5_target,
                p=2,
                dim=-1,
            )

            loss_e5 = (
                1.0
                -
                (
                    student_emb
                    * e5_target
                ).sum(-1)
            ).mean()

            # 3. V3 embedding anchor.
            with torch.no_grad():
                v3_emb, v3_logits = frozen_v3(
                    ids,
                    mask,
                )

            loss_emb_anchor = (
                1.0
                -
                (
                    student_emb
                    * v3_emb
                ).sum(-1)
            ).mean()

            # 4. V3 logit anchor.
            loss_logit_anchor = F.mse_loss(
                logits,
                v3_logits,
            )

            # 5. Small contrastive term.
            loss_con = contrastive_loss(
                student_emb,
                target,
            )

            loss = (
                W_CLS * loss_cls
                +
                W_E5 * loss_e5
                +
                W_V3_EMB_ANCHOR * loss_emb_anchor
                +
                W_V3_LOGIT_ANCHOR * loss_logit_anchor
                +
                W_CONTRASTIVE * loss_con
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                0.5,
            )

            optimizer.step()

            epoch_losses.append(
                float(
                    loss.detach().cpu()
                )
            )

        scheduler.step()

        val_acc, val_f1, _, _ = evaluate(
            model,
            val_ids,
            val_mask,
            val_targets,
        )

        mean_loss = float(
            np.mean(epoch_losses)
        )

        print(
            f"Epoch {epoch:02d} | "
            f"loss={mean_loss:.4f} | "
            f"val={val_acc*100:.2f}% | "
            f"valF1={val_f1*100:.2f}%"
        )

        history.append({
            "epoch": epoch,
            "loss": mean_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError(
            "No V6 checkpoint selected."
        )

    model.load_state_dict(
        best_state,
        strict=True,
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    val_acc, val_f1, pred, truth = evaluate(
        model,
        val_ids,
        val_mask,
        val_targets,
    )

    print("\n" + "=" * 78)
    print("V6 FINAL VALIDATION")
    print("=" * 78)

    print(
        f"Validation accuracy : {val_acc*100:.2f}%"
    )

    print(
        f"Validation Macro F1 : {val_f1*100:.2f}%"
    )

    print("\nClassification report:")

    print(
        classification_report(
            truth,
            pred,
            labels=list(range(NUM_CLASSES)),
            target_names=LOCKED_LABELS,
            digits=4,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    checkpoint_path = (
        OUTPUT_DIR
        / "v6_final_e5_fp32.pt"
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": LOCKED_LABELS,
            "vocab_size": VOCAB_SIZE,
            "max_len": MAX_LEN,
            "student_dim": EMBED_DIM,
            "layers": LAYERS,
            "heads": HEADS,
            "ffn": FFN,
            "teacher": TEACHER_NAME,
            "base_v3_checkpoint": str(V3_CHECKPOINT),
            "validation_accuracy": val_acc,
            "validation_macro_f1": val_f1,
            "safety": {
                "unseen_595_used": False,
                "v3_modified": False,
                "onnx_exported": False,
                "int8_exported": False,
                "threshold_fitting": False,
            },
        },
        checkpoint_path,
    )

    manifest_path = (
        OUTPUT_DIR
        / "v6_final_manifest.json"
    )

    manifest = {
        "checkpoint": str(checkpoint_path),
        "base_v3": str(V3_CHECKPOINT),
        "teacher": TEACHER_NAME,
        "validation_accuracy": val_acc,
        "validation_macro_f1": val_f1,
        "normal_train_rows": len(normal_train),
        "hard_negative_rows": len(hard),
        "validation_rows": len(val),
        "unseen_595_used": False,
        "v3_modified": False,
        "onnx_exported": False,
        "int8_exported": False,
        "threshold_fitting": False,
        "next_step": (
            "Benchmark V6 FP32 against locked V3 on the untouched "
            "595-row unseen set."
        ),
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
        / "v6_final_history.csv"
    )

    pd.DataFrame(
        history
    ).to_csv(
        history_path,
        index=False,
    )

    print("\nSaved checkpoint:")
    print(checkpoint_path)

    print("\nSaved manifest:")
    print(manifest_path)

    print("\nSaved history:")
    print(history_path)

    print("\nIMPORTANT:")
    print("V3 was NOT modified.")
    print("595-row unseen set was NOT used.")
    print("V6 was NOT exported to ONNX.")
    print("V6 was NOT quantized.")
    print("\nNEXT: benchmark V6 FP32 against locked V3.")


if __name__ == "__main__":
    main()
