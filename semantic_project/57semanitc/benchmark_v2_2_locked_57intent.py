#!/usr/bin/env python3
"""
V2.2 LOCKED 57-INTENT BENCHMARK

Evaluates ONLY:
  v3_57intent_v2_2_hard_negative/student_v3_57intent_v2_2_best_fp32.pt

against ONLY:
  v3_57intent_locked_eval/locked_test_57intent.csv

This script:
- DOES read the locked test because this is the final unseen evaluation.
- DOES NOT train.
- DOES NOT modify labels.
- DOES NOT use the negative-test CSV.
- Reports Accuracy / Macro F1 / Weighted F1 / per-intent metrics.
- Saves predictions, report, confusion matrix and summary.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parent

CHECKPOINT = (
    ROOT
    / "v3_57intent_v2_2_hard_negative"
    / "student_v3_57intent_v2_2_best_fp32.pt"
)

LOCKED_TEST = (
    ROOT
    / "v3_57intent_locked_eval"
    / "locked_test_57intent.csv"
)

VOCAB_PATH = ROOT / "vocab.json"
LABELS_PATH = ROOT / "labels.json"

OUT = ROOT / "v3_57intent_v2_2_locked_benchmark"
OUT.mkdir(parents=True, exist_ok=True)

PREDICTIONS = OUT / "locked_predictions_v2_2.csv"
REPORT = OUT / "classification_report_v2_2.txt"
CONFUSION = OUT / "confusion_matrix_v2_2.csv"
SUMMARY = OUT / "benchmark_summary_v2_2.json"

VOCAB_SIZE = 895
EMBED_DIM = 64
HEADS = 4
LAYERS = 2
FF_DIM = 128
MAX_LEN = 24
NUM_CLASSES = 57
DROPOUT = 0.10


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_vocab(path):
    obj = load_json(path)

    if "token_to_id" in obj:
        obj = obj["token_to_id"]
    elif "vocab" in obj and isinstance(obj["vocab"], dict):
        obj = obj["vocab"]

    if not isinstance(obj, dict):
        raise RuntimeError("Unsupported vocab.json format.")

    return {str(k): int(v) for k, v in obj.items()}


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
            f"Expected {NUM_CLASSES} labels, found {len(labels)}."
        )

    return labels


def tokenize(text, vocab):
    ids = []

    for token in str(text).lower().split():
        token = token.strip(".,!?;:\"'()[]{}")
        if token:
            ids.append(vocab.get(token, 1))

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


def find_columns(df):
    text_candidates = ["text", "utterance", "query", "sentence"]
    label_candidates = ["intent", "label", "target"]

    text_col = next((c for c in text_candidates if c in df.columns), None)
    label_col = next((c for c in label_candidates if c in df.columns), None)

    if text_col is None or label_col is None:
        raise RuntimeError(
            "Locked CSV must contain text + intent columns. "
            f"Found columns: {list(df.columns)}"
        )

    return text_col, label_col


def main():
    print("=" * 78)
    print("V2.2 LOCKED 57-INTENT BENCHMARK")
    print("=" * 78)

    for path in [CHECKPOINT, LOCKED_TEST, VOCAB_PATH, LABELS_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing:\n{path}")

    vocab = load_vocab(VOCAB_PATH)
    labels = load_labels(LABELS_PATH)
    label_to_id = {label: i for i, label in enumerate(labels)}

    df = pd.read_csv(LOCKED_TEST)

    text_col, label_col = find_columns(df)

    df = df[[text_col, label_col]].dropna().copy()
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(str)

    unknown = sorted(set(df[label_col]) - set(labels))
    if unknown:
        raise RuntimeError(
            "Locked test contains labels absent from labels.json:\n"
            + "\n".join(unknown)
        )

    x_np = np.asarray(
        [tokenize(x, vocab) for x in df[text_col]],
        dtype=np.int64,
    )

    y = np.asarray(
        [label_to_id[x] for x in df[label_col]],
        dtype=np.int64,
    )

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    print(f"Checkpoint : {CHECKPOINT}")
    print(f"Locked CSV : {LOCKED_TEST}")
    print(f"Rows       : {len(df)}")
    print(f"Device     : {device}")

    model = V3Student57()

    state = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    x = torch.tensor(
        x_np,
        dtype=torch.long,
        device=device,
    )

    # Warm-up outside measured inference timing.
    with torch.no_grad():
        _ = model(x[: min(32, len(x))])

    if device.type == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        confidence, pred = probs.max(dim=1)

    if device.type == "mps":
        torch.mps.synchronize()

    elapsed = time.perf_counter() - start

    pred = pred.cpu().numpy()
    confidence = confidence.cpu().numpy()

    accuracy = accuracy_score(y, pred)
    macro_f1 = f1_score(
        y,
        pred,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y,
        pred,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y,
        pred,
        labels=list(range(NUM_CLASSES)),
        target_names=labels,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(
        y,
        pred,
        labels=list(range(NUM_CLASSES)),
    )

    result = df.copy()
    result["prediction"] = [labels[i] for i in pred]
    result["confidence"] = confidence
    result["correct"] = pred == y

    result.to_csv(PREDICTIONS, index=False)

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    )
    cm_df.to_csv(CONFUSION)

    REPORT.write_text(
        "# V2.2 LOCKED 57-INTENT BENCHMARK\n\n"
        f"Rows       : {len(df)}\n"
        f"Accuracy   : {accuracy * 100:.4f}%\n"
        f"Macro F1   : {macro_f1 * 100:.4f}%\n"
        f"Weighted F1: {weighted_f1 * 100:.4f}%\n"
        f"Total time : {elapsed:.4f} sec\n"
        f"Rows/sec   : {len(df) / elapsed:.2f}\n"
        f"ms/row     : {elapsed / len(df) * 1000:.4f}\n\n"
        "Classification report:\n"
        + report,
        encoding="utf-8",
    )

    summary = {
        "model": "V2.2 hard-negative",
        "checkpoint": str(CHECKPOINT.resolve()),
        "locked_test": str(LOCKED_TEST.resolve()),
        "locked_test_used": True,
        "training_performed": False,
        "rows": int(len(df)),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "total_time_sec": float(elapsed),
        "rows_per_sec": float(len(df) / elapsed),
        "ms_per_row": float(elapsed / len(df) * 1000),
        "mean_confidence": float(confidence.mean()),
        "median_confidence": float(np.median(confidence)),
    }

    SUMMARY.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n--- V2.2 LOCKED TEST RESULT ---")
    print(f"Accuracy   : {accuracy * 100:.4f}%")
    print(f"Macro F1   : {macro_f1 * 100:.4f}%")
    print(f"Weighted F1: {weighted_f1 * 100:.4f}%")

    print("\nClassification report:")
    print(report)

    print("\n--- INFERENCE SPEED ---")
    print(f"Total rows : {len(df)}")
    print(f"Total time : {elapsed:.4f} sec")
    print(f"Rows/sec   : {len(df) / elapsed:.2f}")
    print(f"ms/row     : {elapsed / len(df) * 1000:.4f}")

    print("\nSaved:")
    print(PREDICTIONS)
    print(REPORT)
    print(CONFUSION)
    print(SUMMARY)

    print("\nSTATUS:")
    print("V2.2 LOCKED 57-INTENT BENCHMARK COMPLETE")


if __name__ == "__main__":
    main()
