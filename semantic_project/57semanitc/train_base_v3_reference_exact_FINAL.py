#!/usr/bin/env python3
"""
BASE V3 — 57 INTENTS — REFERENCE-ONLY RETRAIN

This is a reconstructed clean BASE V3.

IMPORTANT:
- Uses ONLY reference_train.csv for training/validation.
- NEVER reads locked_test_57intent.csv.
- NEVER loads the existing full-data student_v3_57intent_fp32.pt.
- Uses the known 57-intent V2 recipe for architecture/tokenization/training.
- Fresh initialization.
- The original V3 training script is unavailable, so this is NOT claimed
  to reproduce the original V3 checkpoint exactly.

Run:
    python3 train_base_v3_reference_exact_FINAL.py
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

OUT_DIR = ROOT / "v3_57intent_base_reference_exact_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = OUT_DIR / "base_v3_reference_exact_best_fp32.pt"
MANIFEST = OUT_DIR / "training_manifest.json"
HISTORY = OUT_DIR / "training_history.csv"

# Known V2 recipe
SEED = 42
BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-5
WEIGHT_DECAY = 0.01
DROPOUT = 0.10
VAL_SIZE = 0.15
PATIENCE = 4

# Exact model contract
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


def load_json(path):
    # FIX: json.loads() receives text, not a Path object.
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
    # FIXED FINAL VERSION
    obj = load_json(path)

    if isinstance(obj, list):
        labels = [str(x) for x in obj]

    elif isinstance(obj.get("labels"), list):
        labels = [str(x) for x in obj["labels"]]

    elif isinstance(obj.get("id_to_label"), dict):
        pairs = sorted(
            (int(k), str(v))
            for k, v in obj["id_to_label"].items()
        )
        labels = [v for _, v in pairs]

    elif isinstance(obj.get("label_to_id"), dict):
        pairs = sorted(
            (int(v), str(k))
            for k, v in obj["label_to_id"].items()
        )
        labels = [v for _, v in pairs]

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
            [
                tokenize(text, vocab)
                for text in df["text"]
            ],
            dtype=np.int64,
        )

        self.y = np.asarray(
            [
                label_to_id[label]
                for label in df["intent"]
            ],
            dtype=np.int64,
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        return (
            torch.tensor(self.x[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


def evaluate(model, loader, device):

    model.eval()

    ys = []
    ps = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)

            logits = model(x)

            pred = torch.argmax(
                logits,
                dim=1,
            ).cpu().numpy()

            ps.extend(pred.tolist())
            ys.extend(y.numpy().tolist())

    accuracy = accuracy_score(ys, ps)

    macro_f1 = f1_score(
        ys,
        ps,
        average="macro",
        zero_division=0,
    )

    return accuracy, macro_f1, ys, ps


def main():

    seed_everything(SEED)

    # Required files.
    for path in (
        REFERENCE_CSV,
        LOCKED_TEST_CSV,
        VOCAB_PATH,
        LABELS_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing:\n{path}"
            )

    # Explicitly DO NOT READ LOCKED_TEST_CSV.
    vocab = load_vocab(VOCAB_PATH)
    labels = load_labels(LABELS_PATH)

    label_to_id = {
        label: i
        for i, label in enumerate(labels)
    }

    df = pd.read_csv(REFERENCE_CSV)

    if "text" not in df.columns or "intent" not in df.columns:
        raise RuntimeError(
            "reference_train.csv must contain columns: text, intent"
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
    print("BASE V3 — 57 INTENTS — REFERENCE-ONLY RETRAIN")
    print("=" * 78)

    print(f"Reference CSV : {REFERENCE_CSV}")
    print(f"Locked test   : {LOCKED_TEST_CSV}")
    print("Locked test   : NOT READ")
    print("Initialization: FRESH")
    print("Existing V3 checkpoint: NOT LOADED")

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

    print("\nMODEL CONTRACT")
    print(f"Vocab       : {VOCAB_SIZE}")
    print(f"Embedding   : {EMBED_DIM}")
    print(f"Layers      : {LAYERS}")
    print(f"Heads       : {HEADS}")
    print(f"FFN         : {FF_DIM}")
    print(f"Max length  : {MAX_LEN}")
    print(f"Classifier  : 64 -> 64 -> {NUM_CLASSES}")

    model = V3Student57()
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.03
    )

    best_f1 = -1.0
    best_epoch = 0
    patience_count = 0

    history = []

    for epoch in range(1, EPOCHS + 1):

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
                1.0,
            )

            optimizer.step()

            total_loss += loss.item() * x.size(0)

        train_loss = total_loss / len(train_ds)

        val_acc, val_f1, _, _ = evaluate(
            model,
            val_loader,
            device,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.4f} | "
            f"val={val_acc*100:.2f}% | "
            f"valF1={val_f1*100:.2f}%"
        )

        if val_f1 > best_f1 + 1e-5:

            best_f1 = val_f1
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

    # Load best validation checkpoint.
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

    val_acc, val_f1, ys, ps = evaluate(
        model,
        val_loader,
        device,
    )

    print("\n" + "=" * 78)
    print("BASE V3 REFERENCE VALIDATION")
    print("=" * 78)

    print(f"Accuracy : {val_acc*100:.4f}%")
    print(f"Macro F1 : {val_f1*100:.4f}%")
    print(f"Best epoch: {best_epoch}")

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
        "model": "BASE_V3_57_REFERENCE_RECONSTRUCTED",
        "status": "reference_only",
        "original_v3_training_script_available": False,
        "existing_full_data_v3_loaded": False,
        "locked_test_read": False,
        "reference_csv": str(REFERENCE_CSV.resolve()),
        "locked_test_csv": str(LOCKED_TEST_CSV.resolve()),
        "checkpoint": str(CHECKPOINT.resolve()),
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
            "epochs_max": EPOCHS,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "label_smoothing": 0.03,
            "gradient_clip": 1.0,
            "validation_size": VAL_SIZE,
            "early_stopping_patience": PATIENCE,
        },
        "reference_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "best_epoch": int(best_epoch),
        "validation_accuracy": float(val_acc),
        "validation_macro_f1": float(val_f1),
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
    print("BASE V3 REFERENCE TRAINING COMPLETE")
    print("Locked test was NOT read.")
    print("Existing full-data V3 checkpoint was NOT loaded.")
    print("ONNX: NO")
    print("INT8: NO")


if __name__ == "__main__":
    main()
