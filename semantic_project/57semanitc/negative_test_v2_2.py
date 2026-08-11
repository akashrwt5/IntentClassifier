#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent

# Run this script from the 57semanitc folder, or edit ROOT below if needed.
PROJECT = ROOT

CHECKPOINT = (
    PROJECT
    / "v3_57intent_v2_2_hard_negative"
    / "student_v3_57intent_v2_2_best_fp32.pt"
)

NEGATIVE_INPUT = (
    PROJECT
    / "v3_57intent_negative_test"
    / "negative_test_results.csv"
)

VOCAB_PATH = PROJECT / "vocab.json"
LABELS_PATH = PROJECT / "labels.json"

OUT_DIR = PROJECT / "v3_57intent_v2_2_negative_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "negative_test_v2_2_results.csv"
OUT_JSON = OUT_DIR / "negative_test_v2_2_summary.json"

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
    return {str(k): int(v) for k, v in obj.items()}


def load_labels(path):
    obj = load_json(path)

    if isinstance(obj, list):
        return [str(x) for x in obj]
    if isinstance(obj.get("labels"), list):
        return [str(x) for x in obj["labels"]]
    if isinstance(obj.get("id_to_label"), dict):
        return [
            v for _, v in sorted(
                ((int(k), str(v)) for k, v in obj["id_to_label"].items()),
                key=lambda z: z[0]
            )
        ]
    if isinstance(obj.get("label_to_id"), dict):
        return [
            k for _, k in sorted(
                ((int(v), str(k)) for k, v in obj["label_to_id"].items()),
                key=lambda z: z[0]
            )
        ]
    raise RuntimeError("Unsupported labels.json format")


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
        padding_mask = x.eq(0)

        pos = torch.arange(
            x.size(1),
            device=x.device
        ).unsqueeze(0)

        h = self.embedding(x) + self.position(pos)

        h = self.encoder(
            h,
            src_key_padding_mask=padding_mask
        )

        valid = (~padding_mask).unsqueeze(-1).float()

        pooled = (
            (h * valid).sum(dim=1)
            / valid.sum(dim=1).clamp(min=1.0)
        )

        return self.classifier(self.norm(pooled))


def main():
    required = [
        CHECKPOINT,
        NEGATIVE_INPUT,
        VOCAB_PATH,
        LABELS_PATH,
    ]

    for x in required:
        if not x.exists():
            raise FileNotFoundError(f"Missing:\n{x}")

    vocab = load_vocab(VOCAB_PATH)
    labels = load_labels(LABELS_PATH)

    if len(labels) != NUM_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_CLASSES} labels, found {len(labels)}"
        )

    df = pd.read_csv(NEGATIVE_INPUT)

    if "text" not in df.columns:
        raise RuntimeError(
            "negative_test_results.csv must contain a 'text' column."
        )

    # Use the exact same 60 negative queries.
    df = df[["text"]].copy()
    df["text"] = df["text"].astype(str)

    # Preserve exact input order and remove accidental duplicate rows only.
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    model = V3Student57()

    state = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    x = torch.tensor(
        np.asarray(
            [tokenize(t, vocab) for t in df["text"]],
            dtype=np.int64
        ),
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

    pred_ids = pred.cpu().numpy()
    confs = conf.cpu().numpy()

    df["prediction"] = [labels[i] for i in pred_ids]
    df["confidence"] = confs
    df["is_fallback"] = (
        df["prediction"] == "Default Fallback Intent"
    )

    fallback_count = int(df["is_fallback"].sum())
    functional_count = int(len(df) - fallback_count)

    summary = {
        "checkpoint": str(CHECKPOINT.resolve()),
        "input": str(NEGATIVE_INPUT.resolve()),
        "total_queries": int(len(df)),
        "fallback_predicted": fallback_count,
        "fallback_rate": fallback_count / len(df),
        "functional_predicted": functional_count,
        "functional_rate": functional_count / len(df),
        "mean_confidence": float(df["confidence"].mean()),
        "median_confidence": float(df["confidence"].median()),
        "max_confidence": float(df["confidence"].max()),
        "min_confidence": float(df["confidence"].min()),
        "locked_test_read": False,
    }

    df.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\n# V2.2 NEGATIVE TEST")
    print("=" * 70)
    print(f"Total queries       : {len(df)}")
    print(
        f"Fallback predicted  : {fallback_count} "
        f"({fallback_count / len(df) * 100:.2f}%)"
    )
    print(
        f"Functional predicted: {functional_count} "
        f"({functional_count / len(df) * 100:.2f}%)"
    )
    print(f"Mean confidence     : {summary['mean_confidence']:.4f}")
    print(f"Median confidence   : {summary['median_confidence']:.4f}")
    print(f"Max confidence      : {summary['max_confidence']:.4f}")
    print(f"Min confidence      : {summary['min_confidence']:.4f}")

    print("\n--- PREDICTIONS ---")
    for _, row in df.iterrows():
        print(
            f"{row['confidence']:.4f} | "
            f"{row['prediction']:<42} | "
            f"{row['text']}"
        )

    print("\n--- PREDICTION COUNTS ---")
    print(df["prediction"].value_counts().to_string())

    print("\nSaved:")
    print(OUT_CSV)
    print(OUT_JSON)
    print("\nSTATUS: V2.2 NEGATIVE TEST COMPLETE")


if __name__ == "__main__":
    main()
