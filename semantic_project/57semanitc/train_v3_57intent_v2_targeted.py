#!/usr/bin/env python3
"""
TRAIN V3 57-INTENT V2 FROM THE EXISTING V3 CHECKPOINT

Goal:
  Improve the existing 57-intent V3 model using targeted hard-error
  oversampling while PRESERVING the V3 architecture.

Architecture contract:
  vocab_size = 895
  embedding  = 64
  layers     = 2
  heads      = 4
  FFN        = 128
  max_len    = 24
  classes    = 57

Model architecture matches the previously used TinySemanticStudent:
  Embedding -> position embedding -> TransformerEncoder -> masked mean
  pooling -> LayerNorm -> Linear(64,64) -> GELU -> Dropout -> Linear(64,57)

The original tokenizer is preserved:
  lowercase -> whitespace split -> punctuation strip
  unknown id = 1
  padding id = 0
  max length = 24

Training:
  - starts from student_v3_57intent_fp32.pt
  - creates a stratified validation split from the V2 training CSV
  - uses hard-error oversampling already present in the dataset
  - uses CrossEntropy only (no E5 runtime, no new architecture)
  - early-stops on validation Macro F1
  - saves best checkpoint only

IMPORTANT:
  Do NOT evaluate on your locked unseen CSV during training.
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


ROOT = Path("/Users/shuklam/IntentClassifier/semantic_project/57semanitc")

TRAIN_CSV = ROOT / "v3_57intent_v2_dataset" / "train_v2_targeted.csv"
BASE_CHECKPOINT = ROOT / "student_v3_57intent_fp32.pt"
VOCAB_PATH = ROOT / "vocab.json"
LABELS_PATH = ROOT / "labels.json"

OUT_DIR = ROOT / "v3_57intent_v2_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CHECKPOINT = OUT_DIR / "student_v3_57intent_v2_best_fp32.pt"
MANIFEST = OUT_DIR / "training_manifest.json"
HISTORY = OUT_DIR / "training_history.csv"

SEED = 42
BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-5
WEIGHT_DECAY = 0.01
DROPOUT = 0.10
VAL_SIZE = 0.15
PATIENCE = 4

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


def load_vocab(path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    if "token_to_id" in obj:
        return obj["token_to_id"]
    if "vocab" in obj and isinstance(obj["vocab"], dict):
        return obj["vocab"]
    if all(isinstance(v, int) for v in obj.values()):
        return obj
    raise RuntimeError("Unsupported vocab.json format.")


def load_labels(path):
    obj = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(obj, list):
        return [str(x) for x in obj]

    if "labels" in obj and isinstance(obj["labels"], list):
        return [str(x) for x in obj["labels"]]

    if "id_to_label" in obj and isinstance(obj["id_to_label"], dict):
        pairs = sorted(
            (int(k), str(v))
            for k, v in obj["id_to_label"].items()
        )
        return [v for _, v in pairs]

    if "label_to_id" in obj and isinstance(obj["label_to_id"], dict):
        pairs = sorted(
            (int(v), str(k))
            for k, v in obj["label_to_id"].items()
        )
        return [v for _, v in pairs]

    raise RuntimeError("Unsupported labels.json format.")


def tokenize(text, vocab):
    ids = []

    for token in str(text).lower().split():
        token = token.strip(
            ".,!?;:\"'()[]{}"
        )

        if token:
            ids.append(
                int(vocab.get(token, 1))
            )

    ids = ids[:MAX_LEN]

    if len(ids) < MAX_LEN:
        ids += [0] * (
            MAX_LEN - len(ids)
        )

    return ids


class V3Student57(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes,
        embed_dim=64,
        heads=4,
        layers=2,
        ff_dim=128,
        max_len=24,
        dropout=0.10,
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

        self.norm = nn.LayerNorm(
            embed_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                embed_dim,
                embed_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                embed_dim,
                num_classes,
            ),
        )

    def forward(self, x):
        pad = x.eq(0)

        pos = torch.arange(
            x.size(1),
            device=x.device,
        ).unsqueeze(0)

        h = (
            self.embedding(x)
            + self.position(pos)
        )

        h = self.encoder(
            h,
            src_key_padding_mask=pad,
        )

        valid = (
            (~pad)
            .unsqueeze(-1)
            .float()
        )

        pooled = (
            h * valid
        ).sum(dim=1) / valid.sum(
            dim=1
        ).clamp(min=1.0)

        return self.classifier(
            self.norm(pooled)
        )


class TextDataset(Dataset):
    def __init__(
        self,
        df,
        vocab,
        label_to_id,
    ):
        self.x = np.asarray(
            [
                tokenize(
                    text,
                    vocab,
                )
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
            torch.tensor(
                self.x[idx],
                dtype=torch.long,
            ),
            torch.tensor(
                self.y[idx],
                dtype=torch.long,
            ),
        )


def evaluate(
    model,
    loader,
    device,
    labels,
):
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

            ps.extend(
                pred.tolist()
            )

            ys.extend(
                y.numpy().tolist()
            )

    acc = accuracy_score(
        ys,
        ps,
    )

    macro_f1 = f1_score(
        ys,
        ps,
        average="macro",
        zero_division=0,
    )

    return (
        acc,
        macro_f1,
        ys,
        ps,
    )


def main():

    seed_everything(SEED)

    for path in (
        TRAIN_CSV,
        BASE_CHECKPOINT,
        VOCAB_PATH,
        LABELS_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing:\n{path}"
            )

    vocab = load_vocab(
        VOCAB_PATH
    )

    labels = load_labels(
        LABELS_PATH
    )

    if len(vocab) != VOCAB_SIZE:
        raise RuntimeError(
            f"Expected vocab {VOCAB_SIZE}, "
            f"got {len(vocab)}"
        )

    if len(labels) != NUM_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_CLASSES} labels, "
            f"got {len(labels)}"
        )

    label_to_id = {
        label: i
        for i, label in enumerate(labels)
    }

    df = pd.read_csv(
        TRAIN_CSV
    )

    df["text"] = (
        df["text"]
        .astype(str)
        .str.strip()
    )

    df["intent"] = (
        df["intent"]
        .astype(str)
        .str.strip()
    )

    df = df[
        (df["text"] != "")
        & (df["intent"] != "")
    ].drop_duplicates(
        ["text", "intent"]
    ).reset_index(drop=True)

    unknown = sorted(
        set(df["intent"])
        - set(labels)
    )

    if unknown:
        raise RuntimeError(
            f"Unknown labels in V2 dataset: {unknown}"
        )

    print("=" * 78)
    print("V3 57-INTENT V2 TRAINING")
    print("=" * 78)

    print(
        f"Rows          : {len(df)}"
    )

    print(
        f"Intents       : {len(labels)}"
    )

    print(
        f"Base checkpoint: {BASE_CHECKPOINT}"
    )

    print(
        "\nArchitecture PRESERVED:"
    )

    print(
        "Embedding : 64"
    )
    print(
        "Layers    : 2"
    )
    print(
        "Heads     : 4"
    )
    print(
        "FFN       : 128"
    )
    print(
        "Classifier: 64 -> 64 -> 57"
    )

    # --------------------------------------------------------
    # Stratified split
    # --------------------------------------------------------

    train_df, val_df = train_test_split(
        df,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=df["intent"],
    )

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

    print(
        "\nDevice:",
        device,
    )

    model = V3Student57(
        vocab_size=VOCAB_SIZE,
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
        heads=HEADS,
        layers=LAYERS,
        ff_dim=FF_DIM,
        max_len=MAX_LEN,
        dropout=DROPOUT,
    )

    state = torch.load(
        BASE_CHECKPOINT,
        map_location="cpu",
    )

    # Support checkpoints saved either directly as state_dict
    # or under a common state_dict/model_state_dict key.
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]

    missing, unexpected = model.load_state_dict(
        state,
        strict=False,
    )

    if missing or unexpected:
        raise RuntimeError(
            "\nV3 checkpoint is NOT architecture-compatible.\n"
            f"Missing: {missing}\n"
            f"Unexpected: {unexpected}"
        )

    print(
        "\nV3 checkpoint loaded EXACTLY."
    )

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # Slight label smoothing improves generalization without changing
    # the architecture.
    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.03
    )

    best_f1 = -1.0
    best_acc = 0.0
    best_epoch = 0
    patience_count = 0
    history = []

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        total_loss = 0.0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(x)

            loss = criterion(
                logits,
                y,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            total_loss += (
                loss.item()
                * x.size(0)
            )

        train_loss = (
            total_loss
            / len(train_ds)
        )

        val_acc, val_f1, _, _ = evaluate(
            model,
            val_loader,
            device,
            labels,
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

        improved = (
            val_f1 > best_f1 + 1e-5
        )

        if improved:

            best_f1 = val_f1
            best_acc = val_acc
            best_epoch = epoch
            patience_count = 0

            torch.save(
                model.state_dict(),
                BEST_CHECKPOINT,
            )

        else:

            patience_count += 1

            if patience_count >= PATIENCE:
                print(
                    "\nEarly stopping."
                )
                break

    # --------------------------------------------------------
    # Reload best
    # --------------------------------------------------------

    best_state = torch.load(
        BEST_CHECKPOINT,
        map_location="cpu",
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
        labels,
    )

    print(
        "\n" + "=" * 78
    )

    print(
        "BEST V2 VALIDATION"
    )

    print(
        "=" * 78
    )

    print(
        f"Accuracy : {val_acc*100:.4f}%"
    )

    print(
        f"Macro F1 : {val_f1*100:.4f}%"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            ys,
            ps,
            target_names=labels,
            digits=4,
            zero_division=0,
        )
    )

    HISTORY.write_text(
        pd.DataFrame(
            history
        ).to_csv(index=False),
        encoding="utf-8",
    )

    manifest = {
        "base_checkpoint": str(
            BASE_CHECKPOINT.resolve()
        ),
        "training_dataset": str(
            TRAIN_CSV.resolve()
        ),
        "vocab": str(
            VOCAB_PATH.resolve()
        ),
        "labels": str(
            LABELS_PATH.resolve()
        ),
        "architecture": {
            "vocab_size": VOCAB_SIZE,
            "embed_dim": EMBED_DIM,
            "heads": HEADS,
            "layers": LAYERS,
            "ff_dim": FF_DIM,
            "max_len": MAX_LEN,
            "num_classes": NUM_CLASSES,
            "classifier": "64 -> 64 -> 57",
        },
        "training": {
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "epochs_requested": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": 0.03,
            "val_size": VAL_SIZE,
            "patience": PATIENCE,
        },
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_acc,
        "best_validation_macro_f1": best_f1,
        "e5_runtime": False,
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

    print(
        "\nSaved checkpoint:"
    )
    print(
        BEST_CHECKPOINT
    )

    print(
        "\nSaved manifest:"
    )
    print(
        MANIFEST
    )

    print(
        "\nSaved history:"
    )
    print(
        HISTORY
    )

    print(
        "\nIMPORTANT:"
    )
    print(
        "595-row unseen set was NOT used."
    )
    print(
        "ONNX export: NO"
    )
    print(
        "INT8: NO"
    )


if __name__ == "__main__":
    main()
