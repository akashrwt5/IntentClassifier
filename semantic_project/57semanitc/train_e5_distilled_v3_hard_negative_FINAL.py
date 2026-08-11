#!/usr/bin/env python3
"""
E5 Distilled V3 — controlled hard-negative fine-tuning.

Starting checkpoint:
  v3_57intent_e5_distilled_v2_FINAL/student_e5_distilled_v2_best_fp32.pt

Training source:
  train.csv + NEW hard-negative examples labeled
  "Default Fallback Intent"

IMPORTANT:
- Canonical locked 1686-row test is NEVER read.
- No locked-test examples are added.
- No labels are changed.
- No quantization.
- No ONNX.
- No synthetic generation.
- V2 checkpoint is preserved.
"""

from pathlib import Path
import json
import re
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report


PROJECT = Path(
    "/Users/shuklam/IntentClassifier/semantic_project/57semanitc"
)

TRAIN_CSV = PROJECT / "train.csv"

V2_DIR = PROJECT / "v3_57intent_e5_distilled_v2_FINAL"

CHECKPOINT = V2_DIR / "student_e5_distilled_v2_best_fp32.pt"
VOCAB_JSON = V2_DIR / "vocab.json"
LABEL_MAP_JSON = V2_DIR / "label_map.json"

OUT_DIR = PROJECT / "v3_57intent_e5_distilled_v3_hard_negative"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CHECKPOINT = OUT_DIR / "student_e5_distilled_v3_best_fp32.pt"
HARD_NEGATIVE_CSV = OUT_DIR / "hard_negative_examples.csv"
TRAINING_CSV = OUT_DIR / "training_v3_hard_negative.csv"
HISTORY_CSV = OUT_DIR / "training_history.csv"
VALIDATION_REPORT = OUT_DIR / "validation_report.txt"
MANIFEST_JSON = OUT_DIR / "training_manifest.json"

SEED = 42
MAX_LEN = 24
PAD_ID = 0
UNK_ID = 1

EMBED_DIM = 64
NHEAD = 4
FF_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.10

# Conservative fine-tuning to avoid damaging the 98.75% baseline.
LR = 2e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 20
PATIENCE = 4
BATCH_SIZE = 128

VAL_SIZE = 0.15

# Repeat hard negatives in the training mix, but keep the
# original dataset dominant.
HARD_NEGATIVE_REPEATS = 2


HARD_NEGATIVES = [
    # Messaging / email OOD
    ("send an email", "Default Fallback Intent"),
    ("email my friend", "Default Fallback Intent"),
    ("send an email to John", "Default Fallback Intent"),
    ("write an email", "Default Fallback Intent"),
    ("send an email message", "Default Fallback Intent"),
    ("compose an email", "Default Fallback Intent"),
    ("send this by email", "Default Fallback Intent"),
    ("email this to my friend", "Default Fallback Intent"),
    ("can you send an email", "Default Fallback Intent"),
    ("send an email for me", "Default Fallback Intent"),

    # Bluetooth / Wi-Fi OOD
    ("turn on bluetooth", "Default Fallback Intent"),
    ("turn bluetooth on", "Default Fallback Intent"),
    ("enable bluetooth", "Default Fallback Intent"),
    ("connect to bluetooth", "Default Fallback Intent"),
    ("connect my phone to bluetooth", "Default Fallback Intent"),
    ("turn off bluetooth", "Default Fallback Intent"),
    ("connect to wifi", "Default Fallback Intent"),
    ("connect to wi fi", "Default Fallback Intent"),
    ("connect my phone to wifi", "Default Fallback Intent"),
    ("turn on wifi", "Default Fallback Intent"),
    ("enable wifi", "Default Fallback Intent"),
    ("disconnect from wifi", "Default Fallback Intent"),

    # Music / movie / media OOD
    ("play some music", "Default Fallback Intent"),
    ("play music", "Default Fallback Intent"),
    ("play a song", "Default Fallback Intent"),
    ("play a movie", "Default Fallback Intent"),
    ("show me a funny video", "Default Fallback Intent"),
    ("play a funny video", "Default Fallback Intent"),
    ("what movie should I watch", "Default Fallback Intent"),
    ("tell me a song", "Default Fallback Intent"),
    ("pause the movie", "Default Fallback Intent"),
    ("stop the movie", "Default Fallback Intent"),
    ("play something", "Default Fallback Intent"),

    # Generic conversational OOD
    ("hello", "Default Fallback Intent"),
    ("hi", "Default Fallback Intent"),
    ("how are you", "Default Fallback Intent"),
    ("what are you doing", "Default Fallback Intent"),
    ("what's up", "Default Fallback Intent"),
    ("thank you", "Default Fallback Intent"),
    ("thanks", "Default Fallback Intent"),
    ("I don't know what I want", "Default Fallback Intent"),
    ("I don't know what to do", "Default Fallback Intent"),
    ("what should I do", "Default Fallback Intent"),
    ("what is going on", "Default Fallback Intent"),
    ("something happened yesterday", "Default Fallback Intent"),

    # General knowledge / web-like queries
    ("what is the weather today", "Default Fallback Intent"),
    ("will it rain tomorrow", "Default Fallback Intent"),
    ("what is the temperature", "Default Fallback Intent"),
    ("is it hot outside", "Default Fallback Intent"),
    ("what is the capital of India", "Default Fallback Intent"),
    ("who invented the telephone", "Default Fallback Intent"),
    ("what is artificial intelligence", "Default Fallback Intent"),
    ("explain quantum physics", "Default Fallback Intent"),
    ("how do I cook rice", "Default Fallback Intent"),
    ("give me a pizza recipe", "Default Fallback Intent"),

    # Navigation / social apps / generic calls
    ("take me to the airport", "Default Fallback Intent"),
    ("open google maps", "Default Fallback Intent"),
    ("open instagram", "Default Fallback Intent"),
    ("call my friend", "Default Fallback Intent"),
    ("call my brother", "Default Fallback Intent"),

    # Ambiguous one-word / generic commands
    ("stop", "Default Fallback Intent"),
    ("go", "Default Fallback Intent"),
    ("yes", "Default Fallback Intent"),
    ("no", "Default Fallback Intent"),
    ("maybe", "Default Fallback Intent"),
    ("sure", "Default Fallback Intent"),
    ("later", "Default Fallback Intent"),

    # Garbage / OOD
    ("banana elephant blue", "Default Fallback Intent"),
    ("asdfghjkl", "Default Fallback Intent"),
    ("blah blah blah", "Default Fallback Intent"),
    ("12345", "Default Fallback Intent"),
]


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def basic_tokens(text):
    return re.findall(
        r"[a-z0-9]+(?:'[a-z0-9]+)?",
        str(text).lower().strip(),
    )


def encode_text(text, vocab):
    tokens = basic_tokens(text)[:MAX_LEN]
    ids = [vocab.get(t, UNK_ID) for t in tokens]
    ids += [PAD_ID] * (MAX_LEN - len(ids))
    return ids


def encode_texts(texts, vocab):
    return np.asarray(
        [encode_text(x, vocab) for x in texts],
        dtype=np.int64,
    )


class TinyIntentClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes):
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

        self.norm = nn.LayerNorm(EMBED_DIM)

        self.classifier = nn.Linear(
            EMBED_DIM,
            num_classes,
        )

    def forward(self, input_ids):
        x = self.embedding(input_ids)

        padding_mask = input_ids.eq(PAD_ID)

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        valid = (~padding_mask).unsqueeze(-1).float()

        denom = valid.sum(dim=1).clamp_min(1.0)

        x = (x * valid).sum(dim=1) / denom

        x = self.norm(x)

        return self.classifier(x)


def evaluate(model, X, y):
    model.eval()

    preds = []

    with torch.no_grad():
        for start in range(0, len(X), 256):
            xb = torch.from_numpy(
                X[start:start + 256]
            )

            logits = model(xb)

            preds.append(
                logits.argmax(dim=1).numpy()
            )

    pred = np.concatenate(preds)

    acc = accuracy_score(y, pred)

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

    return acc, macro, weighted, pred


def main():
    seed_everything()

    print("=" * 72)
    print("E5 DISTILLED V3 — CONTROLLED HARD-NEGATIVE TRAINING")
    print("=" * 72)

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(TRAIN_CSV)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    if not VOCAB_JSON.exists():
        raise FileNotFoundError(VOCAB_JSON)

    if not LABEL_MAP_JSON.exists():
        raise FileNotFoundError(LABEL_MAP_JSON)

    # --------------------------------------------------------
    # Load original training data
    # --------------------------------------------------------

    df = pd.read_csv(TRAIN_CSV)

    required = {"text", "intent"}

    if not required.issubset(df.columns):
        raise RuntimeError(
            "train.csv must contain text and intent columns."
        )

    df = df.dropna(
        subset=["text", "intent"]
    ).reset_index(drop=True)

    if df["intent"].nunique() != 57:
        raise RuntimeError(
            f"Expected 57 intents, found {df['intent'].nunique()}."
        )

    labels = sorted(
        df["intent"].unique().tolist()
    )

    if "Default Fallback Intent" not in labels:
        raise RuntimeError(
            "Default Fallback Intent missing from training labels."
        )

    label_to_id = {
        label: i for i, label in enumerate(labels)
    }

    # --------------------------------------------------------
    # Hard negatives
    # --------------------------------------------------------

    hard_df = pd.DataFrame(
        HARD_NEGATIVES,
        columns=["text", "intent"],
    )

    # Remove accidental duplicates against original train data.
    original_pairs = set(
        zip(
            df["text"].astype(str),
            df["intent"].astype(str),
        )
    )

    hard_df = hard_df[
        ~hard_df.apply(
            lambda r: (
                str(r["text"]),
                str(r["intent"])
            ) in original_pairs,
            axis=1,
        )
    ].reset_index(drop=True)

    hard_df.to_csv(
        HARD_NEGATIVE_CSV,
        index=False,
    )

    print()
    print(f"Original training rows : {len(df)}")
    print(f"Hard negatives added   : {len(hard_df)}")
    print(
        "Hard-negative target   : Default Fallback Intent"
    )

    # --------------------------------------------------------
    # Validation split from ORIGINAL data only.
    # Hard negatives are not used for model-selection validation.
    # --------------------------------------------------------

    train_part, val_part = train_test_split(
        df,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=df["intent"],
    )

    # Add hard negatives ONLY to training.
    hard_repeat = pd.concat(
        [hard_df] * HARD_NEGATIVE_REPEATS,
        ignore_index=True,
    )

    train_mix = pd.concat(
        [
            train_part[
                ["text", "intent"]
            ],
            hard_repeat[
                ["text", "intent"]
            ],
        ],
        ignore_index=True,
    )

    train_mix.to_csv(
        TRAINING_CSV,
        index=False,
    )

    print(
        f"Training rows after negatives : {len(train_mix)}"
    )
    print(
        f"Validation rows               : {len(val_part)}"
    )

    # --------------------------------------------------------
    # Load vocab and labels
    # --------------------------------------------------------

    vocab = json.loads(
        VOCAB_JSON.read_text(
            encoding="utf-8"
        )
    )

    label_map = json.loads(
        LABEL_MAP_JSON.read_text(
            encoding="utf-8"
        )
    )

    if all(
        str(k).isdigit()
        for k in label_map.keys()
    ):
        checkpoint_labels = [
            label_map[str(i)]
            for i in range(len(label_map))
        ]
    else:
        checkpoint_labels = [
            k
            for k, _ in sorted(
                label_map.items(),
                key=lambda kv: int(kv[1]),
            )
        ]

    if checkpoint_labels != labels:
        raise RuntimeError(
            "Training labels do not exactly match V2 label map."
        )

    # --------------------------------------------------------
    # Encode
    # --------------------------------------------------------

    X_train = encode_texts(
        train_mix["text"].astype(str),
        vocab,
    )

    y_train = np.asarray(
        [
            label_to_id[x]
            for x in train_mix["intent"]
        ],
        dtype=np.int64,
    )

    X_val = encode_texts(
        val_part["text"].astype(str),
        vocab,
    )

    y_val = np.asarray(
        [
            label_to_id[x]
            for x in val_part["intent"]
        ],
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Load V2 checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
    )

    state = checkpoint.get(
        "model_state_dict",
        checkpoint.get("state_dict"),
    )

    if state is None:
        raise RuntimeError(
            "Could not find model_state_dict in V2 checkpoint."
        )

    model = TinyIntentClassifier(
        vocab_size=len(vocab),
        num_classes=57,
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    # Baseline validation before fine-tuning.
    base_acc, base_macro, base_weighted, _ = evaluate(
        model,
        X_val,
        y_val,
    )

    print()
    print("--- V2 BASELINE ON SAME VALIDATION SPLIT ---")
    print(f"Accuracy   : {base_acc * 100:.4f}%")
    print(f"Macro F1   : {base_macro * 100:.4f}%")
    print(f"Weighted F1: {base_weighted * 100:.4f}%")

    # --------------------------------------------------------
    # Fine-tune
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_macro = -1.0
    best_state = None
    best_epoch = -1
    patience = 0

    history = []

    indices = np.arange(len(X_train))

    for epoch in range(1, EPOCHS + 1):

        model.train()
        np.random.shuffle(indices)

        losses = []

        for start in range(
            0,
            len(indices),
            BATCH_SIZE,
        ):

            idx = indices[
                start:start + BATCH_SIZE
            ]

            xb = torch.from_numpy(
                X_train[idx]
            )

            yb = torch.from_numpy(
                y_train[idx]
            )

            optimizer.zero_grad()

            logits = model(xb)

            loss = nn.functional.cross_entropy(
                logits,
                yb,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            losses.append(
                float(loss.item())
            )

        (
            val_acc,
            val_macro,
            val_weighted,
            _,
        ) = evaluate(
            model,
            X_val,
            y_val,
        )

        avg_loss = float(
            np.mean(losses)
        )

        history.append(
            {
                "epoch": epoch,
                "loss": avg_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_macro,
                "val_weighted_f1": val_weighted,
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"loss={avg_loss:.4f} | "
            f"val={val_acc * 100:.2f}% | "
            f"valF1={val_macro * 100:.2f}%"
        )

        if val_macro > best_macro:

            best_macro = val_macro
            best_epoch = epoch
            patience = 0

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        else:
            patience += 1

        if patience >= PATIENCE:
            print("Early stopping.")
            break

    if best_state is None:
        raise RuntimeError(
            "No best checkpoint was produced."
        )

    model.load_state_dict(
        best_state,
        strict=True,
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    (
        val_acc,
        val_macro,
        val_weighted,
        val_pred,
    ) = evaluate(
        model,
        X_val,
        y_val,
    )

    report = classification_report(
        y_val,
        val_pred,
        labels=list(range(57)),
        target_names=labels,
        digits=4,
        zero_division=0,
    )

    print()
    print("--- V3 HARD-NEGATIVE VALIDATION ---")
    print(f"Accuracy   : {val_acc * 100:.4f}%")
    print(f"Macro F1   : {val_macro * 100:.4f}%")
    print(f"Weighted F1: {val_weighted * 100:.4f}%")
    print(f"Best epoch : {best_epoch}")

    print()
    print(report)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    torch.save(
        {
            "model_state_dict": model.state_dict(),
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
            "base_validation_accuracy": base_acc,
            "base_validation_macro_f1": base_macro,
            "base_validation_weighted_f1": base_weighted,
            "validation_accuracy": val_acc,
            "validation_macro_f1": val_macro,
            "validation_weighted_f1": val_weighted,
            "source_checkpoint": str(CHECKPOINT),
            "hard_negative_count": len(hard_df),
            "hard_negative_repeats": HARD_NEGATIVE_REPEATS,
            "quantization": False,
            "onnx": False,
            "locked_test_used": False,
        },
        BEST_CHECKPOINT,
    )

    pd.DataFrame(history).to_csv(
        HISTORY_CSV,
        index=False,
    )

    VALIDATION_REPORT.write_text(
        (
            "# E5 DISTILLED V3 HARD-NEGATIVE VALIDATION\n\n"
            f"V2 baseline accuracy   : {base_acc * 100:.4f}%\n"
            f"V2 baseline macro F1   : {base_macro * 100:.4f}%\n"
            f"V2 baseline weighted F1: {base_weighted * 100:.4f}%\n\n"
            f"V3 accuracy             : {val_acc * 100:.4f}%\n"
            f"V3 macro F1             : {val_macro * 100:.4f}%\n"
            f"V3 weighted F1          : {val_weighted * 100:.4f}%\n"
            f"Best epoch              : {best_epoch}\n\n"
            + report
        ),
        encoding="utf-8",
    )

    manifest = {
        "source_checkpoint": str(CHECKPOINT),
        "train_csv": str(TRAIN_CSV),
        "hard_negative_csv": str(HARD_NEGATIVE_CSV),
        "training_csv": str(TRAINING_CSV),
        "hard_negative_count": int(len(hard_df)),
        "hard_negative_repeats": HARD_NEGATIVE_REPEATS,
        "validation_rows": int(len(val_part)),
        "validation_accuracy": float(val_acc),
        "validation_macro_f1": float(val_macro),
        "validation_weighted_f1": float(val_weighted),
        "base_validation_accuracy": float(base_acc),
        "base_validation_macro_f1": float(base_macro),
        "base_validation_weighted_f1": float(base_weighted),
        "best_epoch": int(best_epoch),
        "locked_test_used": False,
        "quantization": False,
        "onnx": False,
        "synthetic_text": False,
        "labels_changed": False,
    }

    MANIFEST_JSON.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Saved:")
    print(BEST_CHECKPOINT)
    print(HARD_NEGATIVE_CSV)
    print(TRAINING_CSV)
    print(HISTORY_CSV)
    print(VALIDATION_REPORT)
    print(MANIFEST_JSON)

    print()
    print("STATUS:")
    print("V3 CONTROLLED HARD-NEGATIVE TRAINING COMPLETE")
    print("LOCKED TEST WAS NOT READ.")
    print("V2 CHECKPOINT WAS PRESERVED.")
    print("NO QUANTIZATION.")
    print("NO ONNX.")
    print("NO SYNTHETIC TEXT.")
    print("NO LABELS CHANGED.")


if __name__ == "__main__":
    main()
