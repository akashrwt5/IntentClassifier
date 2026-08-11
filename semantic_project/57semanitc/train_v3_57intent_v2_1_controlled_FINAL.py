#!/usr/bin/env python3
"""
V2.1 CONTROLLED TARGETED TRAINING

Goal:
    Improve Base V3 Scratch without the broad regression seen in V2.

Rules:
    - Uses reference_train.csv only.
    - NEVER reads locked_test_57intent.csv.
    - Starts from the clean Base V3 Scratch checkpoint.
    - Mines hard errors ONLY from the training partition.
    - Keeps validation partition untouched.
    - Does not generate synthetic text.
    - Does not change labels.
    - Limits hard-error oversampling.
    - Uses class-weighted loss with a conservative weight cap.

Expected files:
    v3_57intent_locked_eval/reference_train.csv
    v3_57intent_base_scratch_model/base_v3_scratch_best_fp32.pt
    vocab.json
    labels.json

Output:
    v3_57intent_v2_1_controlled/
        student_v3_57intent_v2_1_best_fp32.pt
        training_history.csv
        training_manifest.json
        hard_training_errors.csv
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

REFERENCE = ROOT / "v3_57intent_locked_eval" / "reference_train.csv"
BASE_CKPT = ROOT / "v3_57intent_base_scratch_model" / "base_v3_scratch_best_fp32.pt"
VOCAB = ROOT / "vocab.json"
LABELS = ROOT / "labels.json"

OUT = ROOT / "v3_57intent_v2_1_controlled"
OUT.mkdir(parents=True, exist_ok=True)

OUT_CKPT = OUT / "student_v3_57intent_v2_1_best_fp32.pt"
OUT_HISTORY = OUT / "training_history.csv"
OUT_MANIFEST = OUT / "training_manifest.json"
OUT_ERRORS = OUT / "hard_training_errors.csv"

SEED = 42
MAX_LEN = 24
VOCAB_SIZE = 895
EMBED_DIM = 64
HEADS = 4
LAYERS = 2
FF_DIM = 128
NUM_CLASSES = 57
DROPOUT = 0.10

BATCH_SIZE = 128

# Conservative controls. The point is correction, not aggressive memorization.
ERROR_COPY_CAP = 2
MAX_HARD_ROWS_PER_CLASS = 80
HARD_LOSS_WEIGHT = 1.35
CLASS_WEIGHT_CAP = 1.50

LR = 1.0e-5
WEIGHT_DECAY = 1.0e-4
EPOCHS = 20
PATIENCE = 5
VAL_SIZE = 0.12


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_vocab(path):
    obj = load_json(path)

    if "token_to_id" in obj:
        v = obj["token_to_id"]
    elif "vocab" in obj and isinstance(obj["vocab"], dict):
        v = obj["vocab"]
    elif isinstance(obj, dict) and all(isinstance(x, int) for x in obj.values()):
        v = obj
    else:
        raise RuntimeError("Unsupported vocab.json format.")

    v = {str(k): int(x) for k, x in v.items()}

    if len(v) != VOCAB_SIZE:
        raise RuntimeError(f"Expected vocab size {VOCAB_SIZE}, got {len(v)}")

    return v


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
            k for k, _ in sorted(
                ((str(k), int(v)) for k, v in obj["label_to_id"].items()),
                key=lambda z: z[1],
            )
        ]
    else:
        raise RuntimeError("Unsupported labels.json format.")

    if len(labels) != NUM_CLASSES:
        raise RuntimeError(f"Expected {NUM_CLASSES} labels, got {len(labels)}")

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
            VOCAB_SIZE, EMBED_DIM, padding_idx=0
        )
        self.position = nn.Embedding(MAX_LEN, EMBED_DIM)

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
    def __init__(self, df, vocab, label_to_id, sample_weights=None):
        self.df = df.reset_index(drop=True).copy()

        self.x = np.asarray(
            [tokenize(t, vocab) for t in self.df["text"]],
            dtype=np.int64,
        )

        self.y = np.asarray(
            [label_to_id[t] for t in self.df["intent"]],
            dtype=np.int64,
        )

        if sample_weights is None:
            self.weights = np.ones(len(self.df), dtype=np.float32)
        else:
            self.weights = np.asarray(
                sample_weights,
                dtype=np.float32,
            )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.long),
            torch.tensor(self.weights[idx], dtype=torch.float32),
        )


def predict(model, dataset, device):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model.eval()
    ys = []
    ps = []
    cs = []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

            ys.extend(y.numpy().tolist())
            ps.extend(pred.cpu().numpy().tolist())
            cs.extend(conf.cpu().numpy().tolist())

    return np.asarray(ys), np.asarray(ps), np.asarray(cs)


def score(y, p):
    return {
        "accuracy": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, p, average="weighted", zero_division=0),
    }


def mine_training_errors(model, train_df, vocab, label_to_id, labels, device):
    """
    IMPORTANT:
    These are errors on the TRAINING partition only.
    Validation remains completely untouched.
    """

    base_ds = TextDataset(
        train_df,
        vocab,
        label_to_id,
    )

    y, p, conf = predict(
        model,
        base_ds,
        device,
    )

    hard = []

    for i in range(len(train_df)):
        if y[i] != p[i]:
            row = train_df.iloc[i].copy()

            row["true_id"] = int(y[i])
            row["pred_id"] = int(p[i])
            row["true_intent"] = labels[int(y[i])]
            row["pred_intent"] = labels[int(p[i])]
            row["confidence"] = float(conf[i])

            hard.append(row)

    if not hard:
        return pd.DataFrame(
            columns=list(train_df.columns)
            + [
                "true_id",
                "pred_id",
                "true_intent",
                "pred_intent",
                "confidence",
            ]
        )

    hard_df = pd.DataFrame(hard)

    # Prevent one class from dominating the targeted set.
    selected = []

    for intent, group in hard_df.groupby("true_intent"):
        group = group.sort_values(
            "confidence",
            ascending=True,
        ).head(MAX_HARD_ROWS_PER_CLASS)

        selected.append(group)

    hard_df = pd.concat(
        selected,
        ignore_index=True,
    )

    return hard_df


def build_targeted_training(train_df, hard_df):
    """
    Add at most ERROR_COPY_CAP copies per hard example.
    Since the hard examples are from training only, no validation leakage occurs.
    """

    if hard_df.empty:
        result = train_df.copy()
        result["_sample_weight"] = 1.0
        return result

    base = train_df.copy()
    base["_sample_weight"] = 1.0

    hard_rows = []

    for _, row in hard_df.iterrows():
        clean = {
            "text": row["text"],
            "intent": row["intent"],
            "_sample_weight": HARD_LOSS_WEIGHT,
        }

        for _ in range(ERROR_COPY_CAP):
            hard_rows.append(clean.copy())

    hard_extra = pd.DataFrame(hard_rows)

    result = pd.concat(
        [base, hard_extra],
        ignore_index=True,
    )

    return result


def make_class_weights(df, label_to_id):
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)

    for x in df["intent"]:
        counts[label_to_id[x]] += 1.0

    # Mild inverse-frequency correction, capped.
    median = np.median(counts[counts > 0])

    weights = np.ones(NUM_CLASSES, dtype=np.float32)

    for i in range(NUM_CLASSES):
        if counts[i] > 0:
            w = np.sqrt(median / counts[i])
            weights[i] = float(
                np.clip(w, 1.0 / CLASS_WEIGHT_CAP, CLASS_WEIGHT_CAP)
            )

    return torch.tensor(weights, dtype=torch.float32)


def main():
    seed_everything(SEED)

    required = [
        REFERENCE,
        BASE_CKPT,
        VOCAB,
        LABELS,
    ]

    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing:\n{p}")

    if (ROOT / "v3_57intent_locked_eval" / "locked_test_57intent.csv").exists():
        print("Locked test exists, but this script will NOT read it.")

    vocab = load_vocab(VOCAB)
    labels = load_labels(LABELS)

    label_to_id = {
        label: i for i, label in enumerate(labels)
    }

    df = pd.read_csv(REFERENCE)

    if "text" not in df.columns or "intent" not in df.columns:
        raise RuntimeError(
            "reference_train.csv must contain text,intent columns."
        )

    df = df[["text", "intent"]].dropna().copy()
    df["text"] = df["text"].astype(str)
    df["intent"] = df["intent"].astype(str)

    unknown = sorted(
        set(df["intent"]) - set(labels)
    )
    if unknown:
        raise RuntimeError(
            "Unknown labels in reference_train.csv:\n"
            + "\n".join(unknown)
        )

    # Fixed split, stratified, and NEVER modified after this point.
    train_df, val_df = train_test_split(
        df,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=df["intent"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    print("=" * 78)
    print("V2.1 CONTROLLED TARGETED TRAINING")
    print("=" * 78)
    print(f"Device              : {device}")
    print(f"Reference rows      : {len(df)}")
    print(f"Training rows       : {len(train_df)}")
    print(f"Validation rows     : {len(val_df)}")
    print(f"Base checkpoint     : {BASE_CKPT}")
    print("Locked test         : NOT READ")
    print("Synthetic text      : NO")
    print("Labels changed     : NO")

    # Load clean Base V3.
    model = V3Student57()

    state = torch.load(
        BASE_CKPT,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.to(device)

    print("\nBase V3 checkpoint loaded EXACTLY.")

    # Baseline validation BEFORE V2.1.
    val_ds = TextDataset(
        val_df,
        vocab,
        label_to_id,
    )

    yv0, pv0, _ = predict(
        model,
        val_ds,
        device,
    )

    base_val = score(yv0, pv0)

    print("\n--- BASE V3 ON HELD-OUT REFERENCE VALIDATION ---")
    print(f"Accuracy   : {base_val['accuracy']*100:.4f}%")
    print(f"Macro F1   : {base_val['macro_f1']*100:.4f}%")
    print(f"Weighted F1: {base_val['weighted_f1']*100:.4f}%")

    # Mine ONLY training errors.
    hard_df = mine_training_errors(
        model,
        train_df,
        vocab,
        label_to_id,
        labels,
        device,
    )

    hard_df.to_csv(
        OUT_ERRORS,
        index=False,
    )

    print("\n--- TRAINING-ONLY ERROR MINING ---")
    print(f"Hard training errors : {len(hard_df)}")

    if not hard_df.empty:
        print("\nTop hard-error true classes:")
        print(
            hard_df["true_intent"]
            .value_counts()
            .head(15)
            .to_string()
        )

    targeted_df = build_targeted_training(
        train_df,
        hard_df,
    )

    print(
        f"\nOriginal training rows : {len(train_df)}"
    )
    print(
        f"Targeted training rows : {len(targeted_df)}"
    )

    train_ds = TextDataset(
        targeted_df,
        vocab,
        label_to_id,
        sample_weights=targeted_df["_sample_weight"].values,
    )

    class_weights = make_class_weights(
        train_df,
        label_to_id,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        reduction="none",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    patience_count = 0

    history = []

    loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    print("\n--- V2.1 CONTROLLED FINE-TUNING ---")

    for epoch in range(1, EPOCHS + 1):

        model.train()

        losses = []

        for x, y, sample_w in loader:

            x = x.to(device)
            y = y.to(device)
            sample_w = sample_w.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)

            loss_each = criterion(
                logits,
                y,
            )

            loss = (
                loss_each * sample_w
            ).mean()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            losses.append(
                float(loss.detach().cpu())
            )

        yv, pv, _ = predict(
            model,
            val_ds,
            device,
        )

        val_score = score(
            yv,
            pv,
        )

        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_accuracy": val_score["accuracy"],
            "val_macro_f1": val_score["macro_f1"],
            "val_weighted_f1": val_score["weighted_f1"],
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"loss={row['loss']:.4f} | "
            f"val={row['val_accuracy']*100:.2f}% | "
            f"valF1={row['val_macro_f1']*100:.2f}%"
        )

        if val_score["macro_f1"] > best_f1:
            best_f1 = val_score["macro_f1"]
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print("Early stopping.")
            break

    if best_state is None:
        raise RuntimeError("No best model was produced.")

    torch.save(
        best_state,
        OUT_CKPT,
    )

    history_df = pd.DataFrame(history)
    history_df.to_csv(
        OUT_HISTORY,
        index=False,
    )

    # Final validation using best checkpoint.
    model.load_state_dict(
        best_state,
        strict=True,
    )

    yvf, pvf, _ = predict(
        model,
        val_ds,
        device,
    )

    final = score(
        yvf,
        pvf,
    )

    print("\n" + "=" * 78)
    print("V2.1 FINAL REFERENCE VALIDATION")
    print("=" * 78)

    print(
        f"Accuracy   : {final['accuracy']*100:.4f}%"
    )
    print(
        f"Macro F1   : {final['macro_f1']*100:.4f}%"
    )
    print(
        f"Weighted F1: {final['weighted_f1']*100:.4f}%"
    )
    print(
        f"Best epoch : {best_epoch}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            yvf,
            pvf,
            target_names=labels,
            digits=4,
            zero_division=0,
        )
    )

    delta = {
        "accuracy": final["accuracy"] - base_val["accuracy"],
        "macro_f1": final["macro_f1"] - base_val["macro_f1"],
        "weighted_f1": final["weighted_f1"] - base_val["weighted_f1"],
    }

    print("--- V2.1 - BASE VALIDATION DELTA ---")
    print(
        f"Accuracy   : {delta['accuracy']*100:+.4f} pp"
    )
    print(
        f"Macro F1   : {delta['macro_f1']*100:+.4f} pp"
    )
    print(
        f"Weighted F1: {delta['weighted_f1']*100:+.4f} pp"
    )

    manifest = {
        "reference_csv": str(REFERENCE.resolve()),
        "base_checkpoint": str(BASE_CKPT.resolve()),
        "locked_test_read": False,
        "synthetic_text_generated": False,
        "labels_changed": False,
        "seed": SEED,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "hard_training_errors": int(len(hard_df)),
        "targeted_training_rows": int(len(targeted_df)),
        "error_copy_cap": ERROR_COPY_CAP,
        "max_hard_rows_per_class": MAX_HARD_ROWS_PER_CLASS,
        "hard_loss_weight": HARD_LOSS_WEIGHT,
        "class_weight_cap": CLASS_WEIGHT_CAP,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "base_validation": base_val,
        "v2_1_validation": final,
        "delta_v2_1_minus_base": delta,
        "checkpoint": str(OUT_CKPT.resolve()),
    }

    OUT_MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSaved:")
    print(OUT_ERRORS)
    print(OUT_CKPT)
    print(OUT_HISTORY)
    print(OUT_MANIFEST)

    print("\nSTATUS:")
    print("V2.1 CONTROLLED TRAINING COMPLETE")
    print("Locked test was NOT read.")
    print("Training-only errors were used.")
    print("No synthetic text generated.")
    print("No labels changed.")


if __name__ == "__main__":
    main()
