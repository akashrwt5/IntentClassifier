#!/usr/bin/env python3
"""
E5 Distilled V3 — Real-world / unseen utterance test.

This test does NOT read the locked 1686-row benchmark.
It is intended to validate natural user phrasing before STT integration.

Modes:
  1) Curated test set
  2) Interactive microphone-independent text input

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

CHECKPOINT = (
    PROJECT
    / "v3_57intent_e5_distilled_v3_hard_negative"
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
    / "v3_57intent_e5_distilled_v3_real_world_test"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = OUT_DIR / "real_world_predictions.csv"

MAX_LEN = 24
PAD_ID = 0
UNK_ID = 1

EMBED_DIM = 64
NHEAD = 4
FF_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.10


# Curated unseen-style hearing-aid/device utterances.
# Expected labels are only for analysis; the model never receives them.
CASES = [
    ("make it a little louder", "Cmd.VolumeIncrease"),
    ("can you turn the volume up", "Cmd.VolumeIncrease"),
    ("it's too quiet", "Cmd.VolumeIncrease"),
    ("increase the sound please", "Cmd.VolumeIncrease"),
    ("make this softer", "Cmd.VolumeDecrease"),
    ("it's too loud for me", "Cmd.VolumeDecrease"),
    ("turn the volume down a bit", "Cmd.VolumeDecrease"),
    ("lower the sound", "Cmd.VolumeDecrease"),
    ("mute my hearing aids", "Cmd.VolumeMute"),
    ("please mute the sound", "Cmd.VolumeMute"),
    ("unmute the hearing aids", "Cmd.VolumeUnmute"),
    ("turn the sound back on", "Cmd.VolumeUnmute"),

    ("switch to another memory", "Cmd.MemoryChange"),
    ("change my hearing program", "Cmd.MemoryChange"),
    ("move to the next memory", "Cmd.MemoryChange"),
    ("can I change the memory", "Cmd.MemoryChange"),

    ("where is my phone", "Cmd.FindMyPhone"),
    ("help me find my phone", "Cmd.FindMyPhone"),
    ("locate my phone please", "Cmd.FindMyPhone"),

    ("start streaming", "Cmd.StreamingStart"),
    ("begin streaming now", "Cmd.StreamingStart"),
    ("stream audio from my phone", "Cmd.StreamingStart"),
    ("stop streaming", "Cmd.StreamingStop"),
    ("end the audio stream", "Cmd.StreamingStop"),

    ("what is my battery level", "Cmd.BatteryLevel"),
    ("how much battery is left", "Cmd.BatteryLevel"),
    ("check the battery", "Cmd.BatteryLevel"),

    ("send a message", "Cmd.SendMessage"),
    ("message my friend", "Cmd.SendMessage"),
    ("send this message", "Cmd.SendMessage"),

    ("start transcribing", "Cmd.TranscribeStart"),
    ("begin transcription", "Cmd.TranscribeStart"),
    ("start translating", "Cmd.TranslationStart"),
    ("begin translation", "Cmd.TranslationStart"),

    ("add a reminder", "reminders.add"),
    ("remind me tomorrow", "reminders.add"),
    ("create a reminder for me", "reminders.add"),
    ("complete my reminder", "reminders.complete"),
    ("mark the reminder as done", "reminders.complete"),

    # Ambiguous / likely fallback.
    ("send an email", "Default Fallback Intent"),
    ("turn on bluetooth", "Default Fallback Intent"),
    ("connect to wifi", "Default Fallback Intent"),
    ("play some music", "Default Fallback Intent"),
    ("what is the weather", "Default Fallback Intent"),
    ("tell me a joke", "Default Fallback Intent"),
    ("open instagram", "Default Fallback Intent"),
    ("call my friend", "Default Fallback Intent"),
    ("what should I eat tonight", "Default Fallback Intent"),
    ("I don't know what I want", "Default Fallback Intent"),
]


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


def tokenize(text):
    return re.findall(
        r"[a-z0-9]+(?:'[a-z0-9]+)?",
        str(text).lower().strip(),
    )


def encode(text, vocab):
    tokens = tokenize(text)[:MAX_LEN]

    ids = [
        int(vocab.get(token, UNK_ID))
        for token in tokens
    ]

    ids += [PAD_ID] * (MAX_LEN - len(ids))

    return ids


def load_labels():
    obj = json.loads(
        LABEL_MAP_JSON.read_text(encoding="utf-8")
    )

    if all(str(k).isdigit() for k in obj.keys()):
        return [obj[str(i)] for i in range(len(obj))]

    return [
        k for k, _ in sorted(
            obj.items(),
            key=lambda kv: int(kv[1]),
        )
    ]


def predict(model, vocab, labels, text):
    x = torch.tensor(
        [encode(text, vocab)],
        dtype=torch.long,
    )

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].numpy()

    order = np.argsort(probs)[::-1]

    return [
        (labels[int(i)], float(probs[int(i)]))
        for i in order[:3]
    ]


def main():
    print("=" * 72)
    print("E5 DISTILLED V3 — REAL-WORLD UNSEEN UTTERANCE TEST")
    print("=" * 72)

    for p in [CHECKPOINT, VOCAB_JSON, LABEL_MAP_JSON]:
        if not p.exists():
            raise FileNotFoundError(p)

    vocab = json.loads(
        VOCAB_JSON.read_text(encoding="utf-8")
    )
    labels = load_labels()

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
            "Checkpoint does not contain model_state_dict/state_dict."
        )

    model = TinyIntentClassifier(
        vocab_size=len(vocab),
        num_classes=57,
    )
    model.load_state_dict(state, strict=True)
    model.eval()

    rows = []

    print()
    print("Running curated unseen utterances...")
    print()

    for text, expected in CASES:
        top3 = predict(
            model,
            vocab,
            labels,
            text,
        )

        pred, conf = top3[0]

        correct = pred == expected

        print(f"You: {text}")
        print(f"Prediction : {pred}")
        print(f"Expected   : {expected}")
        print(f"Confidence : {conf:.4f}")
        print("Top-3:")

        for label, p in top3:
            print(f"  {p:.4f} | {label}")

        print(
            f"Result     : "
            f"{'PASS' if correct else 'FAIL'}"
        )
        print("-" * 72)

        rows.append({
            "text": text,
            "expected": expected,
            "prediction": pred,
            "confidence": conf,
            "correct": correct,
            "top2": top3[1][0],
            "top2_confidence": top3[1][1],
            "top3": top3[2][0],
            "top3_confidence": top3[2][1],
        })

    result_df = pd.DataFrame(rows)

    accuracy = float(
        result_df["correct"].mean()
    )

    fallback_cases = result_df[
        result_df["expected"]
        == "Default Fallback Intent"
    ]

    fallback_correct = float(
        fallback_cases["correct"].mean()
    )

    functional_cases = result_df[
        result_df["expected"]
        != "Default Fallback Intent"
    ]

    functional_correct = float(
        functional_cases["correct"].mean()
    )

    print()
    print("=" * 72)
    print("REAL-WORLD TEST SUMMARY")
    print("=" * 72)
    print(f"Total cases              : {len(result_df)}")
    print(f"Overall accuracy         : {accuracy * 100:.2f}%")
    print(
        f"Functional intent accuracy: "
        f"{functional_correct * 100:.2f}%"
    )
    print(
        f"Fallback/OOD accuracy    : "
        f"{fallback_correct * 100:.2f}%"
    )

    print()
    print("--- FAILURES ---")

    failures = result_df[
        ~result_df["correct"]
    ]

    if len(failures) == 0:
        print("NONE")
    else:
        for _, row in failures.iterrows():
            print(
                f"{row['confidence']:.4f} | "
                f"{row['prediction']:<40} | "
                f"expected={row['expected']} | "
                f"{row['text']}"
            )

    result_df.to_csv(
        RESULTS_CSV,
        index=False,
    )

    print()
    print("Saved:")
    print(RESULTS_CSV)

    print()
    print(
        "STATUS: "
        "V3 REAL-WORLD TEXT TEST COMPLETE"
    )

    print()
    print("Interactive mode:")
    print("Type a sentence and press Enter.")
    print("Type 'exit' to stop.")

    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if text.lower() in {"exit", "quit"}:
            break

        if not text:
            continue

        top3 = predict(
            model,
            vocab,
            labels,
            text,
        )

        print(f"\nPrediction : {top3[0][0]}")
        print(f"Confidence : {top3[0][1]:.4f}")
        print("Top-3:")

        for label, p in top3:
            print(f"  {p:.4f} | {label}")


if __name__ == "__main__":
    main()
