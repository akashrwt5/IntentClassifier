#!/usr/bin/env python3
"""
TARGETED V2 FROM CLEAN SCRATCH BASE V3 — 57 INTENTS

Pipeline:
    reference_train.csv
          |
          v
    CLEAN BASE V3 checkpoint
          |
          v
    training-only error mining
          |
          v
    targeted V2 dataset
          |
          v
    fine-tune BASE V3 -> V2

IMPORTANT:
- locked_test_57intent.csv is NEVER read.
- The old full-data student_v3_57intent_fp32.pt is NEVER loaded.
- The old V2 checkpoint is NEVER loaded.
- Error mining uses ONLY the BASE V3 validation split.
- No synthetic text is generated.
- No labels are changed.
- Targeted rows are duplicated only from existing training examples that
  were misclassified by BASE V3.
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

BASE_CHECKPOINT = (
    ROOT
    / "v3_57intent_base_scratch_model"
    / "base_v3_scratch_best_fp32.pt"
)

VOCAB_PATH = ROOT / "vocab.json"
LABELS_PATH = ROOT / "labels.json"

OUT_DIR = ROOT / "v3_57intent_v2_from_scratch_base"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETED_CSV = OUT_DIR / "train_v2_targeted.csv"
HARD_ERRORS_CSV = OUT_DIR / "hard_error_rows.csv"
CHECKPOINT = OUT_DIR / "student_v3_57intent_v2_best_fp32.pt"
MANIFEST = OUT_DIR / "training_manifest.json"
HISTORY = OUT_DIR / "training_history.csv"

SEED = 42

# Same model contract.
VOCAB_SIZE = 895
EMBED_DIM = 64
HEADS = 4
LAYERS = 2
FF_DIM = 128
MAX_LEN = 24
NUM_CLASSES = 57
DROPOUT = 0.10

# BASE V3 validation split must be reproduced exactly.
VAL_SIZE = 0.15

# Targeted augmentation:
# each hard validation error contributes extra copies of the SAME original
# training example from the same intent.
MAX_ERROR_COPIES_PER_INTENT = 4
MIN_ERROR_COPIES_PER_INTENT = 1

# V2 fine-tuning recipe.
BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-5
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.03
GRAD_CLIP = 1.0
PATIENCE = 4


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

    weighted_f1 = f1_score(
        ys,
        ps,
        average="weighted",
        zero_division=0,
    )

    return accuracy, macro_f1, weighted_f1, ys, ps


def build_targeted_dataset(
    train_df,
    val_df,
    base_model,
    vocab,
    label_to_id,
    labels,
    device,
):
    """
    Mine BASE V3 errors on validation ONLY.

    Then select original training examples from the same true intent.
    The text itself is never changed.
    """

    val_ds = TextDataset(
        val_df,
        vocab,
        label_to_id,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    base_model.eval()

    error_records = []

    row_offset = 0

    with torch.no_grad():

        for x, y in val_loader:

            x_device = x.to(device)

            logits = base_model(x_device)

            pred = torch.argmax(
                logits,
                dim=1,
            ).cpu().numpy()

            true = y.numpy()

            for i in range(len(true)):

                if int(pred[i]) != int(true[i]):

                    actual_idx = row_offset + i

                    row = val_df.iloc[actual_idx]

                    error_records.append(
                        {
                            "text": str(row["text"]),
                            "intent": str(row["intent"]),
                            "predicted_intent": labels[int(pred[i])],
                        }
                    )

            row_offset += len(true)

    errors = pd.DataFrame(error_records)

    if errors.empty:
        raise RuntimeError(
            "BASE V3 produced zero validation errors. "
            "Targeted V2 has nothing to mine."
        )

    # Save exact error evidence.
    errors.to_csv(
        HARD_ERRORS_CSV,
        index=False,
    )

    # Count errors by TRUE intent.
    error_counts = (
        errors["intent"]
        .value_counts()
        .to_dict()
    )

    # Build a deterministic map of original training examples by intent.
    train_by_intent = {
        intent: group.copy()
        for intent, group in train_df.groupby("intent")
    }

    additions = []

    rng = np.random.default_rng(SEED)

    for intent in labels:

        count = int(error_counts.get(intent, 0))

        if count <= 0:
            continue

        # Cap amplification to avoid letting one large class dominate.
        copies = min(
            MAX_ERROR_COPIES_PER_INTENT,
            max(MIN_ERROR_COPIES_PER_INTENT, count),
        )

        source = train_by_intent.get(intent)

        if source is None or len(source) == 0:
            continue

        # Add examples from the original TRAIN split only.
        # No validation text is inserted into the training set.
        replace = len(source) < copies

        indices = rng.choice(
            len(source),
            size=copies,
            replace=replace,
        )

        for idx in indices:
            r = source.iloc[int(idx)]

            additions.append(
                {
                    "text": str(r["text"]),
                    "intent": str(r["intent"]),
                    "source": "training_error_targeted_copy",
                }
            )

    base_rows = train_df.copy()
    base_rows["source"] = "reference_train"

    targeted = pd.concat(
        [
            base_rows[["text", "intent", "source"]],
            pd.DataFrame(additions),
        ],
        ignore_index=True,
    )

    targeted.to_csv(
        TARGETED_CSV,
        index=False,
    )

    return errors, targeted


def main():

    seed_everything(SEED)

    # Existence checks. The locked test is intentionally not opened/read.
    for path in (
        REFERENCE_CSV,
        LOCKED_TEST_CSV,
        BASE_CHECKPOINT,
        VOCAB_PATH,
        LABELS_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing:\n{path}"
            )

    vocab = load_vocab(VOCAB_PATH)
    labels = load_labels(LABELS_PATH)

    label_to_id = {
        label: i
        for i, label in enumerate(labels)
    }

    df = pd.read_csv(REFERENCE_CSV)

    if "text" not in df.columns or "intent" not in df.columns:
        raise RuntimeError(
            "reference_train.csv must contain columns: text,intent"
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
    print("TARGETED V2 FROM CLEAN BASE V3 — 57 INTENTS")
    print("=" * 78)

    print(f"Reference CSV : {REFERENCE_CSV}")
    print(f"Base checkpoint: {BASE_CHECKPOINT}")
    print("Locked test   : NOT READ")
    print("Old V3        : NOT LOADED")
    print("Old V2        : NOT LOADED")

    train_df, val_df = train_test_split(
        df,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=df["intent"],
    )

    print(f"\nReference rows : {len(df)}")
    print(f"Train rows     : {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    print("\nDevice:", device)

    # Load ONLY clean BASE V3.
    model = V3Student57()

    state = torch.load(
        BASE_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.to(device)

    print("\nBASE V3 checkpoint loaded EXACTLY.")
    print("Checkpoint source:")
    print(BASE_CHECKPOINT)

    # Mine training-only errors.
    print("\n--- MINING BASE V3 VALIDATION ERRORS ---")

    errors, targeted_df = build_targeted_dataset(
        train_df=train_df,
        val_df=val_df,
        base_model=model,
        vocab=vocab,
        label_to_id=label_to_id,
        labels=labels,
        device=device,
    )

    print(f"Hard validation errors : {len(errors)}")
    print(f"Original training rows : {len(train_df)}")
    print(f"Final targeted rows    : {len(targeted_df)}")

    print("\nTop hard-error intents:")

    print(
        errors["intent"]
        .value_counts()
        .head(20)
        .to_string()
    )

    # Targeted V2 fine-tuning.
    train_ds = TextDataset(
        targeted_df,
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING,
    )

    best_f1 = -1.0
    best_epoch = 0
    patience_count = 0

    history = []

    print("\n--- TARGETED V2 FINE-TUNING ---")

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
                "train_loss": train_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
                "val_weighted_f1": val_weighted_f1,
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

    # Final best validation evaluation.
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
    print("TARGETED V2 REFERENCE VALIDATION")
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
        "model": "TARGETED_V2_FROM_CLEAN_BASE_V3",
        "reference_csv": str(REFERENCE_CSV.resolve()),
        "locked_test_csv": str(LOCKED_TEST_CSV.resolve()),
        "locked_test_read": False,
        "old_full_data_v3_loaded": False,
        "old_v2_loaded": False,
        "base_checkpoint": str(BASE_CHECKPOINT.resolve()),
        "vocab": str(VOCAB_PATH.resolve()),
        "labels": str(LABELS_PATH.resolve()),
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
        "targeted_dataset": {
            "original_reference_rows": int(len(df)),
            "training_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "hard_validation_errors": int(len(errors)),
            "final_training_rows": int(len(targeted_df)),
            "synthetic_text_generated": False,
            "labels_changed": False,
            "validation_examples_added_to_training": False,
            "max_error_copies_per_intent": MAX_ERROR_COPIES_PER_INTENT,
        },
        "training": {
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "gradient_clip": GRAD_CLIP,
            "early_stopping_patience": PATIENCE,
            "selection_metric": "validation_macro_f1",
        },
        "best_epoch": int(best_epoch),
        "validation_accuracy": float(val_acc),
        "validation_macro_f1": float(val_f1),
        "validation_weighted_f1": float(val_weighted_f1),
        "checkpoint": str(CHECKPOINT.resolve()),
        "targeted_dataset_csv": str(TARGETED_CSV.resolve()),
        "hard_errors_csv": str(HARD_ERRORS_CSV.resolve()),
        "onnx": False,
        "int8": False,
    }

    MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSaved targeted dataset:")
    print(TARGETED_CSV)

    print("\nSaved hard errors:")
    print(HARD_ERRORS_CSV)

    print("\nSaved checkpoint:")
    print(CHECKPOINT)

    print("\nSaved manifest:")
    print(MANIFEST)

    print("\nSaved history:")
    print(HISTORY)

    print("\nSTATUS:")
    print("TARGETED V2 TRAINING COMPLETE")
    print("Locked test was NOT read.")
    print("No synthetic text generated.")
    print("No labels changed.")
    print("ONNX: NO")
    print("INT8: NO")


if __name__ == "__main__":
    main()
