#!/usr/bin/env python3
"""
FAIR LOCKED 57-INTENT BENCHMARK

Compares ONLY:
  1) Clean Base V3 Scratch
  2) Targeted V2 trained from Clean Base V3

against the SAME locked test set.

IMPORTANT:
- locked_test_57intent.csv is read ONLY here for final evaluation.
- No training occurs.
- No checkpoint is modified.
- Old full-data V3/V2 checkpoints are NOT used.
- The locked test is never used to select epochs or train.
"""

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parent

LOCKED_TEST = (
    ROOT
    / "v3_57intent_locked_eval"
    / "locked_test_57intent.csv"
)

BASE_CKPT = (
    ROOT
    / "v3_57intent_base_scratch_model"
    / "base_v3_scratch_best_fp32.pt"
)

V2_CKPT = (
    ROOT
    / "v3_57intent_v2_from_scratch_base"
    / "student_v3_57intent_v2_best_fp32.pt"
)

VOCAB_PATH = ROOT / "vocab.json"
LABELS_PATH = ROOT / "labels.json"

OUT_DIR = ROOT / "v3_57intent_locked_final_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_PRED = OUT_DIR / "base_v3_locked_predictions.csv"
V2_PRED = OUT_DIR / "v2_locked_predictions.csv"
CONFUSION = OUT_DIR / "confusion_matrix.csv"
PAIRWISE = OUT_DIR / "v2_vs_base_error_changes.csv"
SUMMARY = OUT_DIR / "benchmark_summary.json"
REPORT = OUT_DIR / "benchmark_report.txt"

SEED = 42

VOCAB_SIZE = 895
EMBED_DIM = 64
HEADS = 4
LAYERS = 2
FF_DIM = 128
MAX_LEN = 24
NUM_CLASSES = 57
DROPOUT = 0.10

BATCH_SIZE = 128


def seed_everything(seed):
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
    elif isinstance(obj, dict) and all(
        isinstance(v, int) for v in obj.values()
    ):
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
            v
            for _, v in sorted(
                (int(k), str(v))
                for k, v in obj["id_to_label"].items()
            )
        ]
    elif isinstance(obj.get("label_to_id"), dict):
        labels = [
            v
            for _, v in sorted(
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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=HEADS,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
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


class LockedDataset(Dataset):

    def __init__(self, df, vocab, label_to_id):
        self.text = df["text"].astype(str).tolist()

        self.x = np.asarray(
            [tokenize(x, vocab) for x in self.text],
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


def predict(model, loader, device):

    model.eval()

    y_true = []
    y_pred = []
    confidences = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)

            logits = model(x)

            probs = torch.softmax(logits, dim=1)

            conf, pred = torch.max(
                probs,
                dim=1,
            )

            y_true.extend(
                y.numpy().tolist()
            )

            y_pred.extend(
                pred.cpu().numpy().tolist()
            )

            confidences.extend(
                conf.cpu().numpy().tolist()
            )

    return (
        np.asarray(y_true),
        np.asarray(y_pred),
        np.asarray(confidences),
    )


def metrics(y_true, y_pred):

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def run_model(path, labels, dataset, device):

    model = V3Student57()

    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    y_true, y_pred, conf = predict(
        model,
        loader,
        device,
    )

    return y_true, y_pred, conf


def main():

    seed_everything(SEED)

    for p in (
        LOCKED_TEST,
        BASE_CKPT,
        V2_CKPT,
        VOCAB_PATH,
        LABELS_PATH,
    ):
        if not p.exists():
            raise FileNotFoundError(f"Missing:\n{p}")

    vocab = load_vocab(VOCAB_PATH)
    labels = load_labels(LABELS_PATH)

    label_to_id = {
        label: i
        for i, label in enumerate(labels)
    }

    # This is the ONLY point in the whole workflow where the locked test
    # is intentionally read.
    df = pd.read_csv(LOCKED_TEST)

    if "text" not in df.columns or "intent" not in df.columns:
        raise RuntimeError(
            "Locked CSV must contain columns: text,intent"
        )

    df = df[["text", "intent"]].copy()

    df["text"] = df["text"].astype(str)
    df["intent"] = df["intent"].astype(str)

    unknown = sorted(
        set(df["intent"]) - set(labels)
    )

    if unknown:
        raise RuntimeError(
            "Locked CSV contains labels absent from labels.json:\n"
            + "\n".join(unknown)
        )

    if df["intent"].nunique() != NUM_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_CLASSES} intents, "
            f"got {df['intent'].nunique()}"
        )

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    dataset = LockedDataset(
        df,
        vocab,
        label_to_id,
    )

    print("=" * 78)
    print("FINAL LOCKED 57-INTENT BENCHMARK")
    print("=" * 78)

    print(f"Locked test rows: {len(df)}")
    print(f"Device          : {device}")

    print("\nBASE checkpoint:")
    print(BASE_CKPT)

    print("\nV2 checkpoint:")
    print(V2_CKPT)

    print("\n--- BASE V3 ---")

    base_true, base_pred, base_conf = run_model(
        BASE_CKPT,
        labels,
        dataset,
        device,
    )

    base_m = metrics(
        base_true,
        base_pred,
    )

    print(
        f"Accuracy   : {base_m['accuracy']*100:.4f}%"
    )
    print(
        f"Macro F1   : {base_m['macro_f1']*100:.4f}%"
    )
    print(
        f"Weighted F1: {base_m['weighted_f1']*100:.4f}%"
    )

    print("\n--- TARGETED V2 ---")

    v2_true, v2_pred, v2_conf = run_model(
        V2_CKPT,
        labels,
        dataset,
        device,
    )

    if not np.array_equal(base_true, v2_true):
        raise RuntimeError(
            "Model evaluation produced different true-label arrays."
        )

    v2_m = metrics(
        v2_true,
        v2_pred,
    )

    print(
        f"Accuracy   : {v2_m['accuracy']*100:.4f}%"
    )
    print(
        f"Macro F1   : {v2_m['macro_f1']*100:.4f}%"
    )
    print(
        f"Weighted F1: {v2_m['weighted_f1']*100:.4f}%"
    )

    print("\n--- DELTA: V2 - BASE ---")

    delta = {
        key: v2_m[key] - base_m[key]
        for key in base_m
    }

    print(
        f"Accuracy   : {delta['accuracy']*100:+.4f} pp"
    )
    print(
        f"Macro F1   : {delta['macro_f1']*100:+.4f} pp"
    )
    print(
        f"Weighted F1: {delta['weighted_f1']*100:+.4f} pp"
    )

    print("\n--- BASE CLASSIFICATION REPORT ---")

    base_report = classification_report(
        base_true,
        base_pred,
        target_names=labels,
        digits=4,
        zero_division=0,
    )

    print(base_report)

    print("\n--- V2 CLASSIFICATION REPORT ---")

    v2_report = classification_report(
        v2_true,
        v2_pred,
        target_names=labels,
        digits=4,
        zero_division=0,
    )

    print(v2_report)

    # Per-intent comparison.
    base_report_dict = classification_report(
        base_true,
        base_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    v2_report_dict = classification_report(
        v2_true,
        v2_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    per_intent = []

    for label in labels:

        b = base_report_dict[label]
        v = v2_report_dict[label]

        per_intent.append(
            {
                "intent": label,
                "base_precision": b["precision"],
                "base_recall": b["recall"],
                "base_f1": b["f1-score"],
                "v2_precision": v["precision"],
                "v2_recall": v["recall"],
                "v2_f1": v["f1-score"],
                "delta_f1": v["f1-score"] - b["f1-score"],
                "support": int(b["support"]),
            }
        )

    per_intent_df = pd.DataFrame(
        per_intent
    ).sort_values(
        "delta_f1"
    )

    per_intent_path = (
        OUT_DIR / "per_intent_comparison.csv"
    )

    per_intent_df.to_csv(
        per_intent_path,
        index=False,
    )

    print("\n--- BIGGEST V2 IMPROVEMENTS ---")

    print(
        per_intent_df
        .sort_values("delta_f1", ascending=False)
        .head(10)
        [
            [
                "intent",
                "base_f1",
                "v2_f1",
                "delta_f1",
            ]
        ]
        .to_string(index=False)
    )

    print("\n--- BIGGEST V2 REGRESSIONS ---")

    print(
        per_intent_df
        .sort_values("delta_f1")
        .head(10)
        [
            [
                "intent",
                "base_f1",
                "v2_f1",
                "delta_f1",
            ]
        ]
        .to_string(index=False)
    )

    # Confusion matrix for V2.
    cm = confusion_matrix(
        v2_true,
        v2_pred,
        labels=list(range(NUM_CLASSES)),
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    )

    cm_df.to_csv(
        CONFUSION,
    )

    # Row-level changes.
    rows = []

    for i in range(len(df)):

        true_label = labels[int(base_true[i])]
        base_label = labels[int(base_pred[i])]
        v2_label = labels[int(v2_pred[i])]

        base_correct = (
            base_pred[i] == base_true[i]
        )

        v2_correct = (
            v2_pred[i] == v2_true[i]
        )

        if base_correct and not v2_correct:
            change = "BASE_CORRECT_V2_WRONG"
        elif not base_correct and v2_correct:
            change = "BASE_WRONG_V2_CORRECT"
        elif not base_correct and not v2_correct:
            change = "BOTH_WRONG"
        else:
            change = "BOTH_CORRECT"

        rows.append(
            {
                "text": df.iloc[i]["text"],
                "true_intent": true_label,
                "base_prediction": base_label,
                "v2_prediction": v2_label,
                "base_confidence": float(base_conf[i]),
                "v2_confidence": float(v2_conf[i]),
                "change": change,
            }
        )

    changes_df = pd.DataFrame(rows)

    changes_df.to_csv(
        PAIRWISE,
        index=False,
    )

    # Save predictions.
    base_out = df.copy()
    base_out["prediction"] = [
        labels[int(x)]
        for x in base_pred
    ]
    base_out["confidence"] = base_conf
    base_out["correct"] = (
        base_pred == base_true
    )

    base_out.to_csv(
        BASE_PRED,
        index=False,
    )

    v2_out = df.copy()
    v2_out["prediction"] = [
        labels[int(x)]
        for x in v2_pred
    ]
    v2_out["confidence"] = v2_conf
    v2_out["correct"] = (
        v2_pred == v2_true
    )

    v2_out.to_csv(
        V2_PRED,
        index=False,
    )

    summary = {
        "test": {
            "path": str(LOCKED_TEST.resolve()),
            "rows": int(len(df)),
            "locked_test_used_for_training": False,
        },
        "base_v3": {
            "checkpoint": str(BASE_CKPT.resolve()),
            **base_m,
            "mean_confidence": float(base_conf.mean()),
        },
        "targeted_v2": {
            "checkpoint": str(V2_CKPT.resolve()),
            **v2_m,
            "mean_confidence": float(v2_conf.mean()),
        },
        "delta_v2_minus_base": delta,
        "selection": {
            "recommended_model": (
                "TARGETED_V2"
                if (
                    v2_m["macro_f1"]
                    > base_m["macro_f1"]
                )
                else "BASE_V3"
            ),
            "criterion": "locked_test_macro_f1",
        },
        "files": {
            "base_predictions": str(BASE_PRED.resolve()),
            "v2_predictions": str(V2_PRED.resolve()),
            "per_intent_comparison": str(
                per_intent_path.resolve()
            ),
            "confusion_matrix": str(
                CONFUSION.resolve()
            ),
            "error_changes": str(
                PAIRWISE.resolve()
            ),
        },
    }

    SUMMARY.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_text = (
        "=" * 78
        + "\nFINAL LOCKED 57-INTENT BENCHMARK\n"
        + "=" * 78
        + "\n\n"
        + f"Locked rows: {len(df)}\n\n"
        + "BASE V3\n"
        + f"Accuracy   : {base_m['accuracy']*100:.4f}%\n"
        + f"Macro F1   : {base_m['macro_f1']*100:.4f}%\n"
        + f"Weighted F1: {base_m['weighted_f1']*100:.4f}%\n\n"
        + "TARGETED V2\n"
        + f"Accuracy   : {v2_m['accuracy']*100:.4f}%\n"
        + f"Macro F1   : {v2_m['macro_f1']*100:.4f}%\n"
        + f"Weighted F1: {v2_m['weighted_f1']*100:.4f}%\n\n"
        + "DELTA V2 - BASE\n"
        + f"Accuracy   : {delta['accuracy']*100:+.4f} pp\n"
        + f"Macro F1   : {delta['macro_f1']*100:+.4f} pp\n"
        + f"Weighted F1: {delta['weighted_f1']*100:+.4f} pp\n\n"
        + "RECOMMENDED BY LOCKED MACRO F1: "
        + summary["selection"]["recommended_model"]
        + "\n\n"
        + "--- BASE REPORT ---\n"
        + base_report
        + "\n--- V2 REPORT ---\n"
        + v2_report
    )

    REPORT.write_text(
        report_text,
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("FINAL LOCKED BENCHMARK COMPLETE")
    print("=" * 78)

    print("\nSaved:")
    print(BASE_PRED)
    print(V2_PRED)
    print(per_intent_path)
    print(CONFUSION)
    print(PAIRWISE)
    print(SUMMARY)
    print(REPORT)


if __name__ == "__main__":
    main()
