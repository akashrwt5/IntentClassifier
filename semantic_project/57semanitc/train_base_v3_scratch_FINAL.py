#!/usr/bin/env python3
"""
CLEAN SCRATCH BASE V3 — 57 INTENTS

Purpose:
  Train a NEW 57-intent BASE model from scratch using ONLY:
      v3_57intent_locked_eval/reference_train.csv

This is NOT claimed to reproduce the unavailable original V3 training script.
It is a clean reconstructed baseline.

IMPORTANT:
  - locked_test_57intent.csv is NEVER read
  - student_v3_57intent_fp32.pt is NEVER loaded
  - V2 checkpoint is NEVER loaded
  - architecture/tokenizer remain the known 57-intent V3 contract
  - scratch-training recipe is intentionally different from the V2
    fine-tuning recipe because LR=1e-5 was shown to collapse from scratch

Model:
  vocab 895
  embedding 64
  Transformer: 2 layers / 4 heads / FFN 128
  max_len 24
  classifier 64 -> 64 -> 57

Scratch recipe:
  AdamW
  base LR 3e-4
  warmup 2 epochs
  cosine decay
  weight decay 0.01
  label smoothing 0.02
  gradient clipping 1.0
  class-balanced sqrt weighting
  max 40 epochs
  early stopping patience 8 on validation Macro F1

Run:
  python3 train_base_v3_scratch_FINAL.py
"""

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report


ROOT = Path(__file__).resolve().parent

REFERENCE_CSV = ROOT / "v3_57intent_locked_eval" / "reference_train.csv"
LOCKED_TEST_CSV = ROOT / "v3_57intent_locked_eval" / "locked_test_57intent.csv"
VOCAB_PATH = ROOT / "vocab.json"
LABELS_PATH = ROOT / "labels.json"

OUT_DIR = ROOT / "v3_57intent_base_scratch_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = OUT_DIR / "base_v3_scratch_best_fp32.pt"
MANIFEST = OUT_DIR / "training_manifest.json"
HISTORY = OUT_DIR / "training_history.csv"

SEED = 42

BATCH_SIZE = 64
MAX_EPOCHS = 40
PATIENCE = 8

BASE_LR = 3e-4
MIN_LR = 1e-6
WARMUP_EPOCHS = 2
WEIGHT_DECAY = 0.01
DROPOUT = 0.10
LABEL_SMOOTHING = 0.02
GRAD_CLIP = 1.0

VAL_SIZE = 0.15

VOCAB_SIZE = 895
EMBED_DIM = 64
HEADS = 4
LAYERS = 2
FF_DIM = 128
MAX_LEN = 24
NUM_CLASSES = 57


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_vocab(path):
    obj = load_json(path)

    if "token_to_id" in obj:
        vocab = obj["token_to_id"]
    elif "vocab" in obj and isinstance(obj["vocab"], dict):
        vocab = obj["vocab"]
    elif isinstance(obj, dict) and all(isinstance(v, int) for v in obj.values()):
        vocab = obj
    else:
        raise RuntimeError("Unsupported vocab.json format.")

    vocab = {str(k): int(v) for k, v in vocab.items()}

    if len(vocab) != VOCAB_SIZE:
        raise RuntimeError(
            f"Expected vocab size {VOCAB_SIZE}, got {len(vocab)}"
        )

    return vocab


def load_labels(path):
    obj = load_json(path)

    if isinstance(obj, list):
        labels = [str(x) for x in obj]
    elif isinstance(obj.get("labels"), list):
        labels = [str(x) for x in obj["labels"]]
    elif isinstance(obj.get("id_to_label"), dict):
        labels = [
            v for _, v in sorted(
                (int(k), str(v))
                for k, v in obj["id_to_label"].items()
            )
        ]
    elif isinstance(obj.get("label_to_id"), dict):
        labels = [
            v for _, v in sorted(
                (int(v), str(k))
                for k, v in obj["label_to_id"].items()
            )
        ]
    else:
        raise RuntimeError("Unsupported labels.json format.")

    if len(labels) != NUM_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_CLASSES} labels, got {len(labels)}"
        )

    return labels


def tokenize(text, vocab):
    ids = []

    for token in str(text).lower().split():
        token = token.strip(".,!?;:\"'()[]{}")
        if token:
            ids.append(int(vocab.get(token, 1)))

    ids = ids[:MAX_LEN]
    ids += [0] * (MAX_LEN - len(ids))

    return ids


class V3Student57(nn.Module):

    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
        heads=HEADS,
        layers=LAYERS,
        ff_dim=FF_DIM,
        max_len=MAX_LEN,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0,
        )

        self.position = nn.Embedding(
            max_len,
            embed_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):

        pad = x.eq(0)

        pos = torch.arange(
            x.size(1),
            device=x.device,
        ).unsqueeze(0)

        h = self.embedding(x) + self.position(pos)

        h = self.encoder(
            h,
            src_key_padding_mask=pad,
        )

        valid = (~pad).unsqueeze(-1).float()

        pooled = (
            h * valid
        ).sum(dim=1) / valid.sum(
            dim=1
        ).clamp(min=1.0)

        return self.classifier(
            self.norm(pooled)
        )


class TextDataset(Dataset):

    def __init__(self, df, vocab, label_to_id):
        self.x = np.asarray(
            [tokenize(x, vocab) for x in df["text"]],
            dtype=np.int64,
        )

        self.y = np.asarray(
            [label_to_id[x] for x in df["intent"]],
            dtype=np.int64,
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


def make_sqrt_class_weights(train_df, labels, label_to_id):
    counts = np.zeros(len(labels), dtype=np.float64)

    for label, count in train_df["intent"].value_counts().items():
        counts[label_to_id[label]] = float(count)

    if np.any(counts <= 0):
        missing = [
            labels[i]
            for i, c in enumerate(counts)
            if c <= 0
        ]
        raise RuntimeError(
            f"Missing classes in training split: {missing}"
        )

    # Mild balancing: inverse sqrt frequency.
    # This avoids the extreme gradients produced by full inverse frequency.
    weights = np.sqrt(counts.max() / counts)

    # Normalize mean weight to 1.
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32), counts


def evaluate(model, loader, device):

    model.eval()

    ys = []
    ps = []

    with torch.no_grad():

        for x, y in loader:
            x = x.to(device)

            logits = model(x)

            pred = torch.argmax(logits, dim=1).cpu().numpy()

            ps.extend(pred.tolist())
            ys.extend(y.numpy().tolist())

    accuracy = accuracy_score(ys, ps)

    macro_f1 = f1_score(
        ys,
        ps,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        ys,
        ps,
        average="weighted",
        zero_division=0,
    )

    return accuracy, macro_f1, weighted_f1, ys, ps


def lr_for_epoch(epoch):
    """
    Epoch-level warmup + cosine decay.

    Epochs are 1-indexed.
    """
    if epoch <= WARMUP_EPOCHS:
        return BASE_LR * epoch / WARMUP_EPOCHS

    progress = (
        (epoch - WARMUP_EPOCHS)
        / max(1, MAX_EPOCHS - WARMUP_EPOCHS)
    )

    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))

    return MIN_LR + (BASE_LR - MIN_LR) * cosine


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def main():

    seed_everything(SEED)

    # We require the locked file to exist so the split is known,
    # but we NEVER read it.
    for path in (
        REFERENCE_CSV,
        LOCKED_TEST_CSV,
        VOCAB_PATH,
        LABELS_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing:\n{path}")

    vocab = load_vocab(VOCAB_PATH)
    labels = load_labels(LABELS_PATH)

    label_to_id = {
        label: i
        for i, label in enumerate(labels)
    }

    df = pd.read_csv(REFERENCE_CSV)

    if "text" not in df.columns or "intent" not in df.columns:
        raise RuntimeError(
            "reference_train.csv must contain: text,intent"
        )

    df = df[["text", "intent"]].copy()

    df["text"] = df["text"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()

    df = (
        df[
            (df["text"] != "")
            & (df["intent"] != "")
        ]
        .drop_duplicates(["text", "intent"])
        .reset_index(drop=True)
    )

    unknown = sorted(
        set(df["intent"]) - set(labels)
    )

    if unknown:
        raise RuntimeError(
            f"Unknown intents in reference_train.csv:\n{unknown}"
        )

    if df["intent"].nunique() != NUM_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_CLASSES} intents, "
            f"got {df['intent'].nunique()}"
        )

    print("=" * 78)
    print("CLEAN SCRATCH BASE V3 — 57 INTENTS")
    print("=" * 78)

    print(f"Reference CSV : {REFERENCE_CSV}")
    print(f"Locked test   : {LOCKED_TEST_CSV}")
    print("Locked test   : NOT READ")
    print("Initialization: FRESH")
    print("Existing V3   : NOT LOADED")
    print("Existing V2   : NOT LOADED")

    print(f"\nRows    : {len(df)}")
    print(f"Intents : {df['intent'].nunique()}")

    train_df, val_df = train_test_split(
        df,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=df["intent"],
    )

    print(f"Train   : {len(train_df)}")
    print(f"Val     : {len(val_df)}")

    train_ds = TextDataset(
        train_df,
        vocab,
        label_to_id,
    )

    val_ds = TextDataset(
        val_df,
        vocab,
        label_to_id,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    print("\nDevice:", device)

    print("\nMODEL")
    print(f"Vocab      : {VOCAB_SIZE}")
    print(f"Embedding  : {EMBED_DIM}")
    print(f"Layers     : {LAYERS}")
    print(f"Heads      : {HEADS}")
    print(f"FFN        : {FF_DIM}")
    print(f"Max length : {MAX_LEN}")
    print("Classifier : 64 -> 64 -> 57")

    class_weights, counts = make_sqrt_class_weights(
        train_df,
        labels,
        label_to_id,
    )

    print("\nTRAINING RECIPE")
    print(f"Base LR             : {BASE_LR}")
    print(f"Min LR              : {MIN_LR}")
    print(f"Warmup epochs       : {WARMUP_EPOCHS}")
    print(f"Max epochs          : {MAX_EPOCHS}")
    print(f"Weight decay        : {WEIGHT_DECAY}")
    print(f"Label smoothing     : {LABEL_SMOOTHING}")
    print(f"Class balancing     : sqrt inverse frequency")
    print(f"Early-stop patience : {PATIENCE}")

    model = V3Student57()
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_LR,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=LABEL_SMOOTHING,
    )

    best_f1 = -1.0
    best_acc = 0.0
    best_weighted_f1 = 0.0
    best_epoch = 0
    patience_count = 0

    history = []

    for epoch in range(1, MAX_EPOCHS + 1):

        current_lr = lr_for_epoch(epoch)
        set_lr(optimizer, current_lr)

        model.train()

        total_loss = 0.0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)

            loss = criterion(logits, y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            optimizer.step()

            total_loss += loss.item() * x.size(0)

        train_loss = total_loss / len(train_ds)

        val_acc, val_f1, val_weighted_f1, _, _ = evaluate(
            model,
            val_loader,
            device,
        )

        history.append(
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
                "val_weighted_f1": val_weighted_f1,
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"lr={current_lr:.7f} | "
            f"loss={train_loss:.4f} | "
            f"val={val_acc*100:.2f}% | "
            f"valF1={val_f1*100:.2f}%"
        )

        if val_f1 > best_f1 + 1e-5:

            best_f1 = val_f1
            best_acc = val_acc
            best_weighted_f1 = val_weighted_f1
            best_epoch = epoch
            patience_count = 0

            torch.save(
                model.state_dict(),
                CHECKPOINT,
            )

        else:

            patience_count += 1

            if patience_count >= PATIENCE:
                print("\nEarly stopping.")
                break

    best_state = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        best_state,
        strict=True,
    )

    model.to(device)

    val_acc, val_f1, val_weighted_f1, ys, ps = evaluate(
        model,
        val_loader,
        device,
    )

    print("\n" + "=" * 78)
    print("BASE V3 SCRATCH REFERENCE VALIDATION")
    print("=" * 78)

    print(f"Accuracy   : {val_acc*100:.4f}%")
    print(f"Macro F1   : {val_f1*100:.4f}%")
    print(f"Weighted F1: {val_weighted_f1*100:.4f}%")
    print(f"Best epoch : {best_epoch}")

    print("\nClassification report:")

    print(
        classification_report(
            ys,
            ps,
            target_names=labels,
            digits=4,
            zero_division=0,
        )
    )

    pd.DataFrame(history).to_csv(
        HISTORY,
        index=False,
    )

    manifest = {
        "model": "BASE_V3_57_SCRATCH_RECONSTRUCTED",
        "provenance": (
            "Fresh scratch baseline. Original V3 training script "
            "was unavailable."
        ),
        "reference_csv": str(REFERENCE_CSV.resolve()),
        "locked_test_csv": str(LOCKED_TEST_CSV.resolve()),
        "locked_test_read": False,
        "existing_v3_loaded": False,
        "existing_v2_loaded": False,
        "architecture": {
            "vocab_size": VOCAB_SIZE,
            "embedding": EMBED_DIM,
            "layers": LAYERS,
            "heads": HEADS,
            "ffn": FF_DIM,
            "max_len": MAX_LEN,
            "num_classes": NUM_CLASSES,
            "classifier": "64 -> 64 -> 57",
        },
        "tokenizer": {
            "lowercase": True,
            "whitespace_split": True,
            "unknown_id": 1,
            "padding_id": 0,
            "max_len": MAX_LEN,
        },
        "training": {
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "base_lr": BASE_LR,
            "min_lr": MIN_LR,
            "warmup_epochs": WARMUP_EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "label_smoothing": LABEL_SMOOTHING,
            "gradient_clip": GRAD_CLIP,
            "class_weighting": "sqrt_inverse_frequency",
            "validation_size": VAL_SIZE,
            "early_stopping_patience": PATIENCE,
            "selection_metric": "validation_macro_f1",
        },
        "reference_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "best_epoch": int(best_epoch),
        "validation_accuracy": float(val_acc),
        "validation_macro_f1": float(val_f1),
        "validation_weighted_f1": float(val_weighted_f1),
        "onnx_exported": False,
        "int8": False,
    }

    MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSaved checkpoint:")
    print(CHECKPOINT)

    print("\nSaved manifest:")
    print(MANIFEST)

    print("\nSaved history:")
    print(HISTORY)

    print("\nSTATUS:")
    print("SCRATCH BASE V3 TRAINING COMPLETE")
    print("Locked test was NOT read.")
    print("Existing V3/V2 checkpoints were NOT loaded.")
    print("ONNX: NO")
    print("INT8: NO")


if __name__ == "__main__":
    main()
