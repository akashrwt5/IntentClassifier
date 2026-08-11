#!/usr/bin/env python3
"""
FINAL NEGATIVE TEST FOR CONTROLLED V2.1

Purpose:
  Verify that clearly out-of-domain / unsupported queries are rejected
  as "Default Fallback Intent" rather than being forced into one of the
  56 functional intents.

This script DOES NOT train.
This script DOES NOT modify the checkpoint.
This script DOES NOT read the locked test set.

Checkpoint:
  v3_57intent_v2_1_controlled/student_v3_57intent_v2_1_best_fp32.pt

Input:
  A built-in negative test set is used by default.

You can also provide your own CSV:
  python3 negative_test_v2_1_FINAL.py my_negative_tests.csv

CSV format:
  text
"""

from pathlib import Path
import sys
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent

CHECKPOINT = (
    ROOT
    / "v3_57intent_v2_1_controlled"
    / "student_v3_57intent_v2_1_best_fp32.pt"
)

VOCAB_PATH = ROOT / "vocab.json"
LABELS_PATH = ROOT / "labels.json"

OUT_DIR = ROOT / "v3_57intent_negative_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = OUT_DIR / "negative_test_results.csv"
SUMMARY_JSON = OUT_DIR / "negative_test_summary.json"

VOCAB_SIZE = 895
EMBED_DIM = 64
HEADS = 4
LAYERS = 2
FF_DIM = 128
MAX_LEN = 24
NUM_CLASSES = 57
DROPOUT = 0.10

DEFAULT_FALLBACK = "Default Fallback Intent"

SEED = 42


# ---------------------------------------------------------------------
# NEGATIVE TEST SET
# ---------------------------------------------------------------------

NEGATIVE_QUERIES = [
    # General conversation
    "hello",
    "hi",
    "good morning",
    "good afternoon",
    "how are you",
    "what are you doing",
    "thank you",
    "thanks",
    "okay",
    "what's up",

    # General knowledge
    "what is the capital of India",
    "who is the president of the United States",
    "what is the tallest mountain",
    "tell me a joke",
    "who invented the telephone",
    "what is artificial intelligence",
    "explain quantum physics",

    # Food / cooking
    "how do I cook rice",
    "how do I make coffee",
    "how do I make chicken",
    "what should I eat for dinner",
    "give me a pizza recipe",
    "I want to order pizza",
    "how do I make lasagna",

    # Weather
    "what is the weather today",
    "will it rain tomorrow",
    "is it hot outside",
    "what is the temperature",
    "is there a storm coming",

    # Time / date
    "what time is it",
    "what day is today",
    "what is today's date",
    "set the clock",
    "what time is it in London",

    # Entertainment
    "play some music",
    "play a movie",
    "show me a funny video",
    "what movie should I watch",
    "tell me a song",

    # Navigation / unrelated device tasks
    "take me to the airport",
    "open google maps",
    "send an email",
    "call my friend",
    "open instagram",
    "turn on bluetooth",
    "connect to wifi",

    # Random / noisy
    "banana elephant blue",
    "asdfghjkl",
    "blah blah blah",
    "12345",
    "the quick brown fox jumps over the lazy dog",
    "I don't know what I want",
    "something happened yesterday",

    # Ambiguous short utterances
    "yes",
    "no",
    "maybe",
    "sure",
    "later",
    "stop",
    "go",
]


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_vocab(path):
    obj = load_json(path)

    if isinstance(obj, dict):
        if "token_to_id" in obj:
            vocab = obj["token_to_id"]
        elif "vocab" in obj and isinstance(obj["vocab"], dict):
            vocab = obj["vocab"]
        elif all(isinstance(v, int) for v in obj.values()):
            vocab = obj
        else:
            raise RuntimeError("Unsupported vocab.json format.")
    else:
        raise RuntimeError("Unsupported vocab.json format.")

    vocab = {str(k): int(v) for k, v in vocab.items()}

    print(f"Vocabulary size: {len(vocab)}")

    return vocab


def load_labels(path):
    obj = load_json(path)

    if isinstance(obj, list):
        labels = [str(x) for x in obj]

    elif isinstance(obj, dict) and isinstance(obj.get("labels"), list):
        labels = [str(x) for x in obj["labels"]]

    elif isinstance(obj, dict) and isinstance(obj.get("id_to_label"), dict):
        labels = [
            value
            for _, value in sorted(
                ((int(k), str(v)) for k, v in obj["id_to_label"].items()),
                key=lambda x: x[0],
            )
        ]

    elif isinstance(obj, dict) and isinstance(obj.get("label_to_id"), dict):
        labels = [
            label
            for _, label in sorted(
                ((int(v), str(k)) for k, v in obj["label_to_id"].items()),
                key=lambda x: x[0],
            )
        ]

    else:
        raise RuntimeError("Unsupported labels.json format.")

    print(f"Label count: {len(labels)}")

    return labels


def tokenize(text, vocab):
    tokens = str(text).lower().split()

    ids = []

    for token in tokens:
        token = token.strip(".,!?;:\"'()[]{}")

        if token:
            # Keep the same simple vocabulary lookup contract used
            # by the benchmark/export pipeline.
            ids.append(int(vocab.get(token, 1)))

    ids = ids[:MAX_LEN]

    while len(ids) < MAX_LEN:
        ids.append(0)

    return ids


# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------

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

        positions = torch.arange(
            x.size(1),
            device=x.device,
        ).unsqueeze(0)

        h = (
            self.embedding(x)
            + self.position(positions)
        )

        h = self.encoder(
            h,
            src_key_padding_mask=padding_mask,
        )

        valid = (~padding_mask).unsqueeze(-1).float()

        pooled = (
            (h * valid).sum(dim=1)
            / valid.sum(dim=1).clamp(min=1.0)
        )

        pooled = self.norm(pooled)

        return self.classifier(pooled)


# ---------------------------------------------------------------------
# TEST
# ---------------------------------------------------------------------

def main():

    seed_everything()

    print("=" * 78)
    print("V2.1 NEGATIVE TEST")
    print("=" * 78)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )

    if not VOCAB_PATH.exists():
        raise FileNotFoundError(
            f"vocab.json not found:\n{VOCAB_PATH}"
        )

    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"labels.json not found:\n{LABELS_PATH}"
        )

    vocab = load_vocab(VOCAB_PATH)
    labels = load_labels(LABELS_PATH)

    if len(labels) != NUM_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_CLASSES} labels, got {len(labels)}"
        )

    fallback_id = labels.index(DEFAULT_FALLBACK)

    # Optional custom CSV.
    if len(sys.argv) > 1:
        input_csv = Path(sys.argv[1])

        if not input_csv.exists():
            raise FileNotFoundError(
                f"Input CSV not found:\n{input_csv}"
            )

        df = pd.read_csv(input_csv)

        if "text" not in df.columns:
            raise RuntimeError(
                "Custom CSV must contain a 'text' column."
            )

        texts = df["text"].fillna("").astype(str).tolist()

    else:
        texts = NEGATIVE_QUERIES

    print(f"Negative queries: {len(texts)}")
    print(f"Checkpoint: {CHECKPOINT}")

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    print(f"Device: {device}")

    model = V3Student57()

    state = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.to(device)
    model.eval()

    rows = []

    fallback_count = 0
    functional_count = 0

    with torch.no_grad():

        for i, text in enumerate(texts):

            input_ids = torch.tensor(
                [tokenize(text, vocab)],
                dtype=torch.long,
                device=device,
            )

            logits = model(input_ids)

            probs = torch.softmax(
                logits,
                dim=1,
            )[0]

            confidence, prediction = torch.max(
                probs,
                dim=0,
            )

            pred_id = int(prediction.item())
            pred_label = labels[pred_id]
            conf = float(confidence.item())

            is_fallback = (
                pred_label == DEFAULT_FALLBACK
            )

            if is_fallback:
                fallback_count += 1
            else:
                functional_count += 1

            rows.append(
                {
                    "index": i,
                    "text": text,
                    "prediction": pred_label,
                    "confidence": conf,
                    "is_fallback": is_fallback,
                }
            )

            print(
                f"{i + 1:03d} | "
                f"{conf:.4f} | "
                f"{pred_label:<40} | "
                f"{text}"
            )

    result_df = pd.DataFrame(rows)

    result_df.to_csv(
        RESULTS_CSV,
        index=False,
    )

    total = len(result_df)

    fallback_rate = (
        fallback_count / total
        if total
        else 0.0
    )

    summary = {
        "checkpoint": str(CHECKPOINT.resolve()),
        "total_negative_queries": total,
        "fallback_count": fallback_count,
        "functional_intent_count": functional_count,
        "fallback_rate": fallback_rate,
        "functional_capture_rate": (
            functional_count / total
            if total
            else 0.0
        ),
        "mean_confidence": float(
            result_df["confidence"].mean()
        ),
        "median_confidence": float(
            result_df["confidence"].median()
        ),
        "status": (
            "PASS"
            if fallback_rate >= 0.80
            else "REVIEW"
        ),
        "note": (
            "This is a negative-set diagnostic, not a proof of "
            "production safety. The negative queries are expected "
            "to map to Default Fallback Intent."
        ),
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("NEGATIVE TEST SUMMARY")
    print("=" * 78)

    print(
        f"Total queries       : {total}"
    )

    print(
        f"Fallback predicted  : "
        f"{fallback_count} "
        f"({fallback_rate * 100:.2f}%)"
    )

    print(
        f"Functional predicted: "
        f"{functional_count} "
        f"({(1 - fallback_rate) * 100:.2f}%)"
    )

    print(
        f"Mean confidence     : "
        f"{summary['mean_confidence']:.4f}"
    )

    print(
        f"Median confidence   : "
        f"{summary['median_confidence']:.4f}"
    )

    print(
        f"\nSTATUS: {summary['status']}"
    )

    # Show all false captures.
    false_captures = result_df[
        ~result_df["is_fallback"]
    ].copy()

    if len(false_captures):

        print("\n" + "=" * 78)
        print("NEGATIVE QUERIES CAPTURED BY FUNCTIONAL INTENTS")
        print("=" * 78)

        for _, row in false_captures.iterrows():
            print(
                f"{row['confidence']:.4f} | "
                f"{row['prediction']:<40} | "
                f"{row['text']}"
            )

    else:

        print("\nNo negative query was captured by a functional intent.")

    # Breakdown by predicted intent.
    print("\n" + "=" * 78)
    print("PREDICTION BREAKDOWN")
    print("=" * 78)

    breakdown = (
        result_df["prediction"]
        .value_counts()
        .rename_axis("prediction")
        .reset_index(name="count")
    )

    print(
        breakdown.to_string(index=False)
    )

    print("\nSaved:")
    print(RESULTS_CSV)
    print(SUMMARY_JSON)

    print("\nSTATUS:")
    print("V2.1 NEGATIVE TEST COMPLETE")


if __name__ == "__main__":
    main()
