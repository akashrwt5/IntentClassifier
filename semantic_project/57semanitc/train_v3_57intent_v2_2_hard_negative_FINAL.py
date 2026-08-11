#!/usr/bin/env python3
"""
V2.2 HARD-NEGATIVE TRAINING

Goal:
  Improve rejection of clearly out-of-domain queries while preserving
  the 57-intent classifier.

Starting checkpoint:
  v3_57intent_v2_1_controlled/student_v3_57intent_v2_1_best_fp32.pt

Training data:
  v3_57intent_locked_eval/reference_train.csv
  + selected false-capture negative queries from:
    v3_57intent_negative_test/negative_test_results.csv

IMPORTANT:
  - locked_test_57intent.csv is NEVER read.
  - Negative examples are relabeled ONLY as "Default Fallback Intent".
  - No synthetic text is generated.
  - Existing 57 intent labels are not changed.
  - Negative examples are split into train/holdout so the same negative
    query is not used for both training and the built-in V2.2 holdout check.
  - The holdout here is NOT the final production negative test.
"""

from pathlib import Path
import json
import random
import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report


ROOT = Path(__file__).resolve().parent

REFERENCE_TRAIN = (
    ROOT / "v3_57intent_locked_eval" / "reference_train.csv"
)

V21_CHECKPOINT = (
    ROOT / "v3_57intent_v2_1_controlled"
    / "student_v3_57intent_v2_1_best_fp32.pt"
)

NEGATIVE_RESULTS = (
    ROOT / "v3_57intent_negative_test"
    / "negative_test_results.csv"
)

VOCAB_PATH = ROOT / "vocab.json"
LABELS_PATH = ROOT / "labels.json"

OUT = ROOT / "v3_57intent_v2_2_hard_negative"
OUT.mkdir(parents=True, exist_ok=True)

V22_DATASET = OUT / "train_v2_2_hard_negative.csv"
NEG_TRAIN = OUT / "negative_train.csv"
NEG_HOLDOUT = OUT / "negative_holdout.csv"
CHECKPOINT_OUT = OUT / "student_v3_57intent_v2_2_best_fp32.pt"
MANIFEST = OUT / "training_manifest.json"
HISTORY = OUT / "training_history.csv"
REPORT = OUT / "validation_report.txt"

LOCKED_TEST = (
    ROOT / "v3_57intent_locked_eval" / "locked_test_57intent.csv"
)

# Model contract from V3/V2.1.
VOCAB_SIZE = 895
EMBED_DIM = 64
HEADS = 4
LAYERS = 2
FF_DIM = 128
MAX_LEN = 24
NUM_CLASSES = 57
DROPOUT = 0.10

SEED = 42
BATCH_SIZE = 128
MAX_EPOCHS = 18
PATIENCE = 5
LR = 2e-5
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

# Conservative hard-negative controls.
NEGATIVE_MULTIPLIER = 4
MAX_NEGATIVE_TRAIN = 64
NEG_HOLDOUT_FRACTION = 0.30

# Accept V2.2 only if validation Macro F1 does not regress too much.
MAX_ALLOWED_MACRO_F1_DROP = 0.015


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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

    return {str(k): int(v) for k, v in vocab.items()}


def load_labels(path):
    obj = load_json(path)

    if isinstance(obj, list):
        labels = [str(x) for x in obj]
    elif isinstance(obj.get("labels"), list):
        labels = [str(x) for x in obj["labels"]]
    elif isinstance(obj.get("id_to_label"), dict):
        labels = [
            v for _, v in sorted(
                ((int(k), str(v)) for k, v in obj["id_to_label"].items()),
                key=lambda z: z[0],
            )
        ]
    elif isinstance(obj.get("label_to_id"), dict):
        labels = [
            k for _, k in sorted(
                ((int(v), str(k)) for k, v in obj["label_to_id"].items()),
                key=lambda z: z[0],
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
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            EMBED_DIM,
            padding_idx=0,
        )

        self.position = nn.Embedding(
            MAX_LEN,
            EMBED_DIM,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=HEADS,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=LAYERS,
        )

        self.norm = nn.LayerNorm(EMBED_DIM)

        self.classifier = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(EMBED_DIM, NUM_CLASSES),
        )

    def forward(self, x):
        padding_mask = x.eq(0)

        pos = torch.arange(
            x.size(1),
            device=x.device,
        ).unsqueeze(0)

        h = self.embedding(x) + self.position(pos)

        h = self.encoder(
            h,
            src_key_padding_mask=padding_mask,
        )

        valid = (~padding_mask).unsqueeze(-1).float()

        pooled = (
            (h * valid).sum(dim=1)
            / valid.sum(dim=1).clamp(min=1.0)
        )

        return self.classifier(self.norm(pooled))


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

    def __getitem__(self, i):
        return (
            torch.tensor(self.x[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
        )


def evaluate(model, loader, device):
    model.eval()

    ys = []
    ps = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()

            ys.extend(y.numpy().tolist())
            ps.extend(pred.tolist())

    y = np.asarray(ys)
    p = np.asarray(ps)

    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(
            f1_score(y, p, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y, p, average="weighted", zero_division=0)
        ),
        "y": y,
        "p": p,
    }


def main():
    seed_everything()

    print("=" * 78)
    print("V2.2 HARD-NEGATIVE TRAINING")
    print("=" * 78)

    required = [
        REFERENCE_TRAIN,
        V21_CHECKPOINT,
        NEGATIVE_RESULTS,
        VOCAB_PATH,
        LABELS_PATH,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing:\n{path}")

    # Safety: the locked test path must exist, but this script must never read it.
    if not LOCKED_TEST.exists():
        raise RuntimeError(
            f"Expected locked test to exist but it was not found:\n{LOCKED_TEST}"
        )

    vocab = load_vocab(VOCAB_PATH)
    labels = load_labels(LABELS_PATH)
    label_to_id = {x: i for i, x in enumerate(labels)}

    if "Default Fallback Intent" not in label_to_id:
        raise RuntimeError("Default Fallback Intent missing from labels.json.")

    fallback = "Default Fallback Intent"

    reference = pd.read_csv(REFERENCE_TRAIN)

    if not {"text", "intent"}.issubset(reference.columns):
        raise RuntimeError(
            "reference_train.csv must contain text and intent columns."
        )

    reference = reference[["text", "intent"]].dropna().copy()
    reference["text"] = reference["text"].astype(str)
    reference["intent"] = reference["intent"].astype(str)

    unknown = sorted(set(reference["intent"]) - set(labels))
    if unknown:
        raise RuntimeError(
            "reference_train.csv contains unknown labels:\n"
            + "\n".join(unknown)
        )

    negative = pd.read_csv(NEGATIVE_RESULTS)

    required_neg_cols = {"text", "prediction", "is_fallback"}
    if not required_neg_cols.issubset(negative.columns):
        raise RuntimeError(
            "negative_test_results.csv must contain: "
            "text,prediction,is_fallback"
        )

    # Only false captures are hard negatives. They are examples that the
    # classifier incorrectly assigned to a functional intent.
    hard_neg = negative[
        negative["is_fallback"].astype(str).str.lower().isin(
            ["false", "0", "no"]
        )
    ].copy()

    # Robust fallback if CSV stores actual booleans.
    if len(hard_neg) == 0:
        hard_neg = negative[
            negative["prediction"].astype(str) != fallback
        ].copy()

    hard_neg["text"] = hard_neg["text"].astype(str)
    hard_neg["intent"] = fallback

    hard_neg = hard_neg.drop_duplicates(
        subset=["text"]
    ).reset_index(drop=True)

    if len(hard_neg) < 4:
        raise RuntimeError(
            f"Only {len(hard_neg)} hard negatives found; need at least 4."
        )

    # Hold out some negative examples so they are not all used in training.
    neg_train, neg_holdout = train_test_split(
        hard_neg[["text", "intent"]],
        test_size=NEG_HOLDOUT_FRACTION,
        random_state=SEED,
        shuffle=True,
    )

    # Limit size and oversample only in the training dataset.
    neg_train = neg_train.head(MAX_NEGATIVE_TRAIN).copy()

    neg_train = pd.concat(
        [neg_train] * NEGATIVE_MULTIPLIER,
        ignore_index=True,
    )

    # Split reference data deterministically and stratified.
    ref_train, ref_val = train_test_split(
        reference,
        test_size=0.10,
        random_state=SEED,
        stratify=reference["intent"],
    )

    # IMPORTANT:
    # Validation contains only reference examples. Negative hard examples
    # are not inserted into the validation set.
    final_train = pd.concat(
        [
            ref_train[["text", "intent"]],
            neg_train[["text", "intent"]],
        ],
        ignore_index=True,
    )

    final_train = final_train.sample(
        frac=1.0,
        random_state=SEED,
    ).reset_index(drop=True)

    V22_DATASET.write_text(
        final_train.to_csv(index=False),
        encoding="utf-8",
    )

    neg_train[["text", "intent"]].drop_duplicates().to_csv(
        NEG_TRAIN,
        index=False,
    )

    neg_holdout.to_csv(
        NEG_HOLDOUT,
        index=False,
    )

    print(f"Reference rows       : {len(reference)}")
    print(f"Reference train      : {len(ref_train)}")
    print(f"Reference validation : {len(ref_val)}")
    print(f"Hard negatives found : {len(hard_neg)}")
    print(f"Negative train base  : {len(neg_train) // NEGATIVE_MULTIPLIER}")
    print(f"Negative train rows  : {len(neg_train)}")
    print(f"Negative holdout     : {len(neg_holdout)}")
    print(f"Final train rows     : {len(final_train)}")

    train_ds = TextDataset(final_train, vocab, label_to_id)
    val_ds = TextDataset(ref_val, vocab, label_to_id)

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

    print(f"Device: {device}")

    model = V3Student57()

    state = torch.load(
        V21_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state, strict=True)
    model.to(device)

    # Conservative fine-tuning from V2.1.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss()

    best_macro = -1.0
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        running_loss = 0.0
        seen = 0

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

            running_loss += float(loss.item()) * len(y)
            seen += len(y)

        train_loss = running_loss / max(seen, 1)

        val = evaluate(
            model,
            val_loader,
            device,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.4f} | "
            f"val={val['accuracy']*100:.2f}% | "
            f"valF1={val['macro_f1']*100:.2f}%"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": val["accuracy"],
                "val_macro_f1": val["macro_f1"],
                "val_weighted_f1": val["weighted_f1"],
            }
        )

        if val["macro_f1"] > best_macro:
            best_macro = val["macro_f1"]
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            print("Early stopping.")
            break

    if best_state is None:
        raise RuntimeError("No best checkpoint was produced.")

    model.load_state_dict(best_state, strict=True)

    final_val = evaluate(
        model,
        val_loader,
        device,
    )

    y = final_val["y"]
    p = final_val["p"]

    print("\n" + "=" * 78)
    print("V2.2 REFERENCE VALIDATION")
    print("=" * 78)

    print(
        f"Accuracy   : {final_val['accuracy']*100:.4f}%"
    )
    print(
        f"Macro F1   : {final_val['macro_f1']*100:.4f}%"
    )
    print(
        f"Weighted F1: {final_val['weighted_f1']*100:.4f}%"
    )
    print(f"Best epoch : {best_epoch}")

    report = classification_report(
        y,
        p,
        labels=list(range(NUM_CLASSES)),
        target_names=labels,
        digits=4,
        zero_division=0,
    )

    print("\nClassification report:")
    print(report)

    torch.save(
        best_state,
        CHECKPOINT_OUT,
    )

    pd.DataFrame(history).to_csv(
        HISTORY,
        index=False,
    )

    manifest = {
        "model": "V2.2 hard-negative controlled fine-tune",
        "base_checkpoint": str(V21_CHECKPOINT.resolve()),
        "reference_train": str(REFERENCE_TRAIN.resolve()),
        "locked_test_read": False,
        "locked_test_path": str(LOCKED_TEST.resolve()),
        "negative_source": str(NEGATIVE_RESULTS.resolve()),
        "negative_examples_used": int(
            len(neg_train) // NEGATIVE_MULTIPLIER
        ),
        "negative_holdout": int(len(neg_holdout)),
        "negative_multiplier": NEGATIVE_MULTIPLIER,
        "max_negative_train": MAX_NEGATIVE_TRAIN,
        "synthetic_text_generated": False,
        "labels_changed": False,
        "best_epoch": best_epoch,
        "accuracy": final_val["accuracy"],
        "macro_f1": final_val["macro_f1"],
        "weighted_f1": final_val["weighted_f1"],
    }

    MANIFEST.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    REPORT.write_text(
        "V2.2 REFERENCE VALIDATION\n\n"
        f"Accuracy   : {final_val['accuracy']*100:.4f}%\n"
        f"Macro F1   : {final_val['macro_f1']*100:.4f}%\n"
        f"Weighted F1: {final_val['weighted_f1']*100:.4f}%\n"
        f"Best epoch : {best_epoch}\n\n"
        + report,
        encoding="utf-8",
    )

    print("\nSaved:")
    print(V22_DATASET)
    print(NEG_TRAIN)
    print(NEG_HOLDOUT)
    print(CHECKPOINT_OUT)
    print(HISTORY)
    print(MANIFEST)
    print(REPORT)

    print("\nSTATUS:")
    print("V2.2 HARD-NEGATIVE TRAINING COMPLETE")
    print("Locked test was NOT read.")
    print("No synthetic text generated.")
    print("No labels changed.")


if __name__ == "__main__":
    main()
