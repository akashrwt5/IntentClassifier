#!/usr/bin/env python3

from pathlib import Path
import json
import re
import string

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# EXACT MODEL PATHS
# ============================================================

PROJECT = Path(
    "/Users/shuklam/IntentClassifier/semantic_project/57semanitc"
)

MODEL_DIR = (
    PROJECT
    / "v3_57intent_e5_distilled_v2_FINAL"
)

CHECKPOINT = (
    MODEL_DIR
    / "student_e5_distilled_v2_best_fp32.pt"
)

VOCAB_JSON = (
    MODEL_DIR
    / "vocab.json"
)

LABEL_MAP_JSON = (
    MODEL_DIR
    / "label_map.json"
)

OUTPUT_DIR = (
    PROJECT
    / "v3_57intent_e5_distilled_v2_negative_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_CSV = (
    OUTPUT_DIR
    / "negative_test_results.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "negative_test_summary.json"
)


# ============================================================
# MODEL CONFIG — MUST MATCH TRAINING
# ============================================================

MAX_LEN = 24
PAD_ID = 0
UNK_ID = 1

EMBED_DIM = 64
NHEAD = 4
FF_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.10


# ============================================================
# NEGATIVE / OOD QUERIES
# Expected behavior for ALL of these:
#     Default Fallback Intent
# ============================================================

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


# ============================================================
# TOKENIZER
# ============================================================

def basic_tokens(text):
    text = str(text).lower().strip()

    return re.findall(
        r"[a-z0-9]+(?:'[a-z0-9]+)?",
        text,
    )


def encode_text(text, vocab):
    tokens = basic_tokens(text)[:MAX_LEN]

    ids = [
        vocab.get(token, UNK_ID)
        for token in tokens
    ]

    ids += [
        PAD_ID
    ] * (MAX_LEN - len(ids))

    return ids


# ============================================================
# MODEL
# ============================================================

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


# ============================================================
# LOADERS
# ============================================================

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_vocab():
    obj = load_json(
        VOCAB_JSON
    )

    if (
        isinstance(obj, dict)
        and "vocab" in obj
    ):
        obj = obj["vocab"]

    return {
        str(k): int(v)
        for k, v in obj.items()
    }


def load_labels(checkpoint):
    if LABEL_MAP_JSON.exists():

        obj = load_json(
            LABEL_MAP_JSON
        )

        if isinstance(obj, dict):

            # {"0": "label", ...}
            if all(
                str(k).isdigit()
                for k in obj.keys()
            ):
                return [
                    obj[str(i)]
                    for i in range(len(obj))
                ]

            # {"label": 0, ...}
            if all(
                isinstance(v, (int, float))
                for v in obj.values()
            ):
                return [
                    k
                    for k, _ in sorted(
                        obj.items(),
                        key=lambda kv: int(kv[1]),
                    )
                ]

            for key in (
                "labels",
                "classes",
                "label_names",
            ):
                if key in obj:
                    return list(obj[key])

    if isinstance(checkpoint, dict):
        for key in (
            "labels",
            "classes",
            "label_names",
        ):
            if key in checkpoint:
                return list(checkpoint[key])

    raise RuntimeError(
        "Could not load 57 intent labels."
    )


def get_state_dict(checkpoint):
    if isinstance(checkpoint, dict):

        for key in (
            "model_state_dict",
            "state_dict",
            "model",
        ):
            if isinstance(
                checkpoint.get(key),
                dict,
            ):
                return checkpoint[key]

        if any(
            isinstance(v, torch.Tensor)
            for v in checkpoint.values()
        ):
            return checkpoint

    raise RuntimeError(
        "Could not find model_state_dict."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "E5 DISTILLED V2 - NEGATIVE / OOD TEST"
    )
    print("=" * 72)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"\nCheckpoint not found:\n{CHECKPOINT}"
        )

    vocab = load_vocab()

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
    )

    labels = load_labels(
        checkpoint
    )

    if len(labels) != 57:
        raise RuntimeError(
            f"Expected 57 labels, got {len(labels)}."
        )

    fallback_name = "Default Fallback Intent"

    if fallback_name not in labels:
        raise RuntimeError(
            f"'{fallback_name}' is not present "
            "in the model label map."
        )

    fallback_id = labels.index(
        fallback_name
    )

    model = TinyIntentClassifier(
        vocab_size=len(vocab),
        num_classes=len(labels),
    )

    state = get_state_dict(
        checkpoint
    )

    clean_state = {}

    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        clean_state[key] = value

    missing, unexpected = (
        model.load_state_dict(
            clean_state,
            strict=False,
        )
    )

    if missing:
        raise RuntimeError(
            "Checkpoint/model mismatch. "
            "Missing keys:\n"
            + "\n".join(missing)
        )

    if unexpected:
        print(
            "WARNING: unexpected checkpoint keys:"
        )
        for key in unexpected:
            print(" ", key)

    model.eval()

    # --------------------------------------------------------
    # ENCODE
    # --------------------------------------------------------

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

    inputs = torch.from_numpy(X)

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    with torch.no_grad():
        logits = model(
            inputs
        ).cpu().numpy()

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

    pred_ids = logits.argmax(
        axis=1
    )

    confidences = probs.max(
        axis=1
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    rows = []

    for i, text in enumerate(
        NEGATIVE_QUERIES
    ):

        pred_id = int(
            pred_ids[i]
        )

        pred_label = labels[
            pred_id
        ]

        top3_ids = np.argsort(
            probs[i]
        )[::-1][:3]

        rows.append(
            {
                "text": text,
                "prediction": pred_label,
                "confidence": float(
                    confidences[i]
                ),
                "is_fallback": (
                    pred_id == fallback_id
                ),
                "top1": labels[
                    int(top3_ids[0])
                ],
                "top1_confidence": float(
                    probs[
                        i,
                        top3_ids[0],
                    ]
                ),
                "top2": labels[
                    int(top3_ids[1])
                ],
                "top2_confidence": float(
                    probs[
                        i,
                        top3_ids[1],
                    ]
                ),
                "top3": labels[
                    int(top3_ids[2])
                ],
                "top3_confidence": float(
                    probs[
                        i,
                        top3_ids[2],
                    ]
                ),
            }
        )

    result_df = pd.DataFrame(
        rows
    )

    fallback_count = int(
        result_df["is_fallback"].sum()
    )

    functional_count = (
        len(result_df)
        - fallback_count
    )

    fallback_rate = (
        fallback_count
        / len(result_df)
    )

    functional_rate = (
        functional_count
        / len(result_df)
    )

    mean_conf = float(
        result_df[
            "confidence"
        ].mean()
    )

    median_conf = float(
        result_df[
            "confidence"
        ].median()
    )

    max_conf = float(
        result_df[
            "confidence"
        ].max()
    )

    min_conf = float(
        result_df[
            "confidence"
        ].min()
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print()
    print(
        "--- NEGATIVE TEST SUMMARY ---"
    )

    print(
        f"Total queries       : "
        f"{len(result_df)}"
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
        f"{mean_conf:.4f}"
    )

    print(
        f"Median confidence   : "
        f"{median_conf:.4f}"
    )

    print(
        f"Max confidence      : "
        f"{max_conf:.4f}"
    )

    print(
        f"Min confidence      : "
        f"{min_conf:.4f}"
    )

    print()
    print(
        "--- PREDICTIONS ---"
    )

    for _, row in result_df.iterrows():

        print(
            f"{row['confidence']:.4f} | "
            f"{row['prediction']:<42} | "
            f"{row['text']}"
        )

    print()
    print(
        "--- NON-FALLBACK PREDICTIONS ---"
    )

    non_fallback = result_df[
        ~result_df["is_fallback"]
    ]

    if len(non_fallback) == 0:
        print(
            "NONE — all negative queries "
            "were rejected as fallback."
        )
    else:
        for _, row in non_fallback.iterrows():
            print(
                f"{row['confidence']:.4f} | "
                f"{row['prediction']:<42} | "
                f"{row['text']}"
            )

    print()
    print(
        "--- PREDICTION COUNTS ---"
    )

    print(
        result_df[
            "prediction"
        ].value_counts()
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    result_df.to_csv(
        RESULTS_CSV,
        index=False,
    )

    summary = {
        "model": str(
            CHECKPOINT
        ),
        "total_queries": len(
            result_df
        ),
        "fallback_predicted": fallback_count,
        "functional_predicted": functional_count,
        "fallback_rate": fallback_rate,
        "functional_rate": functional_rate,
        "mean_confidence": mean_conf,
        "median_confidence": median_conf,
        "max_confidence": max_conf,
        "min_confidence": min_conf,
        "expected_label": fallback_name,
        "quantization": False,
        "onnx": False,
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Saved:"
    )

    print(
        RESULTS_CSV
    )

    print(
        SUMMARY_JSON
    )

    print()
    print(
        "STATUS: "
        "E5 DISTILLED V2 NEGATIVE TEST COMPLETE"
    )


if __name__ == "__main__":
    main()
