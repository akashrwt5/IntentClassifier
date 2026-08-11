#!/usr/bin/env python3
"""
E5 Distilled V3 — SAME 60-query negative/OOD regression test.

Uses the V3 checkpoint produced by:
v3_57intent_e5_distilled_v3_hard_negative/

Purpose:
- Compare V3 against the earlier V2 negative test.
- Detect functional false positives.
- Pay special attention to high-confidence OOD predictions.

This script does NOT read the locked 1686-row test.
No quantization.
No ONNX.
"""

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


PROJECT = Path(
    "/Users/shuklam/IntentClassifier/semantic_project/57semanitc"
)

MODEL_DIR = (
    PROJECT
    / "v3_57intent_e5_distilled_v3_hard_negative"
)

CHECKPOINT = (
    MODEL_DIR
    / "student_e5_distilled_v3_best_fp32.pt"
)

VOCAB_JSON = (
    PROJECT
    / "v3_57intent_e5_distilled_v2_FINAL"
    / "vocab.json"
)

LABEL_MAP_JSON = (
    PROJECT
    / "v3_57intent_e5_distilled_v2_FINAL"
    / "label_map.json"
)

OUT_DIR = (
    PROJECT
    / "v3_57intent_e5_distilled_v3_negative_test"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_CSV = (
    OUT_DIR
    / "negative_test_v3_results.csv"
)

SUMMARY_JSON = (
    OUT_DIR
    / "negative_test_v3_summary.json"
)


MAX_LEN = 24
PAD_ID = 0
UNK_ID = 1

EMBED_DIM = 64
NHEAD = 4
FF_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.10


# EXACT SAME 60-QUERY SET USED FOR V2 NEGATIVE TEST
NEGATIVE_QUERIES = [
    "hello",
    "hi",
    "good morning",
    "good afternoon",
    "how are you",
    "what are you doing",
    "thank you",
    "thanks",
    "what's up",

    "tell me a joke",
    "what is the capital of India",
    "who is the president of the United States",
    "what is the tallest mountain",
    "who invented the telephone",
    "what is artificial intelligence",
    "explain quantum physics",

    "how do I cook rice",
    "how do I make coffee",
    "how do I make chicken",
    "what should I eat for dinner",
    "give me a pizza recipe",
    "I want to order pizza",
    "how do I make lasagna",

    "what is the weather today",
    "will it rain tomorrow",
    "is it hot outside",
    "what is the temperature",
    "is there a storm coming",
    "what time is it",
    "what day is today",
    "what is today's date",
    "what time is it in London",

    "play some music",
    "play a movie",
    "show me a funny video",
    "what movie should I watch",
    "tell me a song",

    "take me to the airport",
    "open google maps",
    "send an email",
    "call my friend",
    "open instagram",

    "turn on bluetooth",
    "connect to wifi",

    "banana elephant blue",
    "asdfghjkl",
    "blah blah blah",
    "12345",
    "the quick brown fox jumps over the lazy dog",
    "I don't know what I want",

    "something happened yesterday",
    "yes",
    "no",
    "maybe",
    "sure",
    "later",
    "stop",
    "go",
]


class TinyIntentClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        num_classes,
    ):
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

        self.norm = nn.LayerNorm(
            EMBED_DIM
        )

        self.classifier = nn.Linear(
            EMBED_DIM,
            num_classes,
        )

    def forward(self, input_ids):

        x = self.embedding(
            input_ids
        )

        padding_mask = input_ids.eq(
            PAD_ID
        )

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        valid = (
            ~padding_mask
        ).unsqueeze(-1).float()

        denom = valid.sum(
            dim=1
        ).clamp_min(1.0)

        x = (
            x * valid
        ).sum(dim=1) / denom

        x = self.norm(x)

        return self.classifier(x)


def tokenize(text):
    return re.findall(
        r"[a-z0-9]+(?:'[a-z0-9]+)?",
        str(text).lower().strip(),
    )


def encode_text(text, vocab):
    tokens = tokenize(text)[:MAX_LEN]

    ids = [
        int(vocab.get(
            token,
            UNK_ID,
        ))
        for token in tokens
    ]

    ids += [
        PAD_ID
    ] * (
        MAX_LEN - len(ids)
    )

    return ids


def load_labels():
    obj = json.loads(
        LABEL_MAP_JSON.read_text(
            encoding="utf-8"
        )
    )

    if all(
        str(k).isdigit()
        for k in obj.keys()
    ):
        return [
            obj[str(i)]
            for i in range(len(obj))
        ]

    return [
        k
        for k, _ in sorted(
            obj.items(),
            key=lambda kv: int(kv[1]),
        )
    ]


def main():

    print("=" * 72)
    print(
        "E5 DISTILLED V3 - "
        "SAME 60 QUERY NEGATIVE/OOD TEST"
    )
    print("=" * 72)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )

    if not VOCAB_JSON.exists():
        raise FileNotFoundError(
            f"Vocab not found:\n{VOCAB_JSON}"
        )

    if not LABEL_MAP_JSON.exists():
        raise FileNotFoundError(
            f"Label map not found:\n{LABEL_MAP_JSON}"
        )

    vocab = json.loads(
        VOCAB_JSON.read_text(
            encoding="utf-8"
        )
    )

    labels = load_labels()

    if len(labels) != 57:
        raise RuntimeError(
            f"Expected 57 labels, found {len(labels)}."
        )

    fallback = "Default Fallback Intent"

    if fallback not in labels:
        raise RuntimeError(
            f"{fallback!r} not found in label map."
        )

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
            "No model_state_dict/state_dict in checkpoint."
        )

    model = TinyIntentClassifier(
        vocab_size=len(vocab),
        num_classes=57,
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.eval()

    X = np.asarray(
        [
            encode_text(
                text,
                vocab,
            )
            for text in NEGATIVE_QUERIES
        ],
        dtype=np.int64,
    )

    with torch.no_grad():

        logits = model(
            torch.from_numpy(X)
        ).numpy()

    shifted = (
        logits
        - logits.max(
            axis=1,
            keepdims=True,
        )
    )

    probs = np.exp(
        shifted
    )

    probs /= probs.sum(
        axis=1,
        keepdims=True,
    )

    rows = []

    for i, text in enumerate(
        NEGATIVE_QUERIES
    ):

        order = np.argsort(
            probs[i]
        )[::-1]

        top3 = order[:3]

        pred_id = int(
            top3[0]
        )

        pred_label = labels[
            pred_id
        ]

        confidence = float(
            probs[
                i,
                pred_id,
            ]
        )

        rows.append(
            {
                "text": text,
                "prediction": pred_label,
                "confidence": confidence,
                "is_fallback": (
                    pred_label == fallback
                ),
                "top2_prediction": labels[
                    int(top3[1])
                ],
                "top2_confidence": float(
                    probs[
                        i,
                        top3[1],
                    ]
                ),
                "top3_prediction": labels[
                    int(top3[2])
                ],
                "top3_confidence": float(
                    probs[
                        i,
                        top3[2],
                    ]
                ),
            }
        )

    df = pd.DataFrame(rows)

    fallback_count = int(
        df["is_fallback"].sum()
    )

    functional_count = (
        len(df)
        - fallback_count
    )

    fallback_rate = (
        fallback_count
        / len(df)
    )

    functional_rate = (
        functional_count
        / len(df)
    )

    print()
    print("--- V3 NEGATIVE TEST SUMMARY ---")

    print(
        f"Total queries       : {len(df)}"
    )

    print(
        f"Fallback predicted  : "
        f"{fallback_count} "
        f"({fallback_rate * 100:.2f}%)"
    )

    print(
        f"Functional predicted: "
        f"{functional_count} "
        f"({functional_rate * 100:.2f}%)"
    )

    print(
        f"Mean confidence     : "
        f"{df['confidence'].mean():.4f}"
    )

    print(
        f"Median confidence   : "
        f"{df['confidence'].median():.4f}"
    )

    print(
        f"Max confidence      : "
        f"{df['confidence'].max():.4f}"
    )

    print(
        f"Min confidence      : "
        f"{df['confidence'].min():.4f}"
    )

    print()
    print("--- NON-FALLBACK PREDICTIONS ---")

    non_fallback = df[
        ~df["is_fallback"]
    ]

    if len(non_fallback) == 0:
        print(
            "NONE — all 60 queries "
            "predicted Default Fallback Intent."
        )
    else:
        for _, row in non_fallback.iterrows():

            print(
                f"{row['confidence']:.4f} | "
                f"{row['prediction']:<42} | "
                f"{row['text']}"
            )

    print()
    print("--- PREDICTION COUNTS ---")

    print(
        df["prediction"].value_counts()
    )

    df.to_csv(
        RESULTS_CSV,
        index=False,
    )

    summary = {
        "model": str(CHECKPOINT),
        "total_queries": int(len(df)),
        "fallback_predicted": fallback_count,
        "functional_predicted": functional_count,
        "fallback_rate": float(fallback_rate),
        "functional_rate": float(functional_rate),
        "mean_confidence": float(
            df["confidence"].mean()
        ),
        "median_confidence": float(
            df["confidence"].median()
        ),
        "max_confidence": float(
            df["confidence"].max()
        ),
        "min_confidence": float(
            df["confidence"].min()
        ),
        "locked_test_used": False,
        "quantization": False,
        "onnx": False,
        "query_set": "same_60_queries_as_v2",
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Saved:")
    print(RESULTS_CSV)
    print(SUMMARY_JSON)

    print()
    print(
        "STATUS: "
        "V3 NEGATIVE/OOD TEST COMPLETE"
    )


if __name__ == "__main__":
    main()
