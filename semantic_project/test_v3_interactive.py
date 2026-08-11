#!/usr/bin/env python3

"""
Interactive V3 ONNX intent tester.

Run:
    cd /Users/shuklam/IntentClassifier/semantic_project
    python3 test_v3_interactive.py

Type an utterance and press Enter.
Type 'exit' or 'quit' to stop.

This script:
- Uses the locked V3 ONNX model.
- Uses the same 895-token vocabulary.
- Produces top-K intents and confidence.
- Does NOT train or modify the model.
- Uses one valid [1, 24] ONNX input per utterance.
"""

from pathlib import Path
import json
import re

import numpy as np
import onnxruntime as ort


ROOT = Path("/Users/shuklam/IntentClassifier/semantic_project")

MODEL_PATH = (
    ROOT
    / "tiny_semantic_student_v3_fp32"
    / "v3_semantic_student_fp32.onnx"
)

VOCAB_CANDIDATES = [
    ROOT / "tiny_semantic_student_v2_balanced" / "vocab.json",
    ROOT / "tiny_semantic_student_v3_balanced" / "vocab.json",
    ROOT / "tiny_semantic_student_v3_fp32" / "vocab.json",
]

MAX_LEN = 24

LABELS = [
    "device.memory.change",
    "device.volume.decrease",
    "device.volume.increase",
    "device.volume.mute",
    "device.volume.unmute",
    "find.phone.locate",
    "help.reminder.show",
    "reminders.task.complete",
    "reminders.task.create",
    "streaming.session.start",
    "streaming.session.stop",
]


def load_vocab():
    candidates = []

    for path in VOCAB_CANDIDATES:
        if not path.exists():
            continue

        with path.open("r", encoding="utf-8") as f:
            vocab = json.load(f)

        if isinstance(vocab, dict) and "vocab" in vocab:
            vocab = vocab["vocab"]

        if isinstance(vocab, dict) and len(vocab) == 895:
            candidates.append((path, vocab))

    if not candidates:
        raise FileNotFoundError(
            "No 895-token vocab.json found."
        )

    # This is the vocabulary previously resolved for V3.
    preferred = [
        item for item in candidates
        if "tiny_semantic_student_v2_balanced" in str(item[0])
    ]

    return preferred[0] if preferred else candidates[0]


def find_special(vocab, names, default):
    for name in names:
        if name in vocab:
            return int(vocab[name])
    return default


def tokenize(text, vocab):
    pad_id = find_special(
        vocab,
        ["<pad>", "[PAD]"],
        0,
    )

    unk_id = find_special(
        vocab,
        ["<unk>", "[UNK]"],
        1,
    )

    cls_id = None
    for name in ["<cls>", "[CLS]"]:
        if name in vocab:
            cls_id = int(vocab[name])
            break

    sep_id = None
    for name in ["<sep>", "[SEP]"]:
        if name in vocab:
            sep_id = int(vocab[name])
            break

    text = re.sub(
        r"\s+",
        " ",
        str(text).strip().lower(),
    )

    ids = []

    if cls_id is not None:
        ids.append(cls_id)

    for token in text.split():
        ids.append(
            int(vocab.get(token, unk_id))
        )

    if sep_id is not None:
        ids.append(sep_id)

    ids = ids[:MAX_LEN]

    if len(ids) < MAX_LEN:
        ids.extend(
            [pad_id] * (MAX_LEN - len(ids))
        )

    return np.asarray(
        ids,
        dtype=np.int64,
    )


def softmax(logits):
    x = logits - np.max(logits)
    exp = np.exp(x)
    return exp / np.sum(exp)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"V3 model not found:\n{MODEL_PATH}"
        )

    vocab_path, vocab = load_vocab()

    print("=" * 70)
    print("V3 INTERACTIVE ONNX INTENT TEST")
    print("=" * 70)

    print(f"Model : {MODEL_PATH}")
    print(f"Vocab : {vocab_path}")
    print(f"Vocab size: {len(vocab)}")

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]

    print(f"Input : {input_meta.name} {input_meta.shape}")
    print(f"Output: {output_meta.name} {output_meta.shape}")

    if input_meta.shape != [1, 24]:
        raise RuntimeError(
            f"Unexpected V3 input shape: "
            f"{input_meta.shape}"
        )

    if output_meta.shape != [1, 11]:
        raise RuntimeError(
            f"Unexpected V3 output shape: "
            f"{output_meta.shape}"
        )

    print("\nType an utterance.")
    print("Commands: exit / quit")
    print("-" * 70)

    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not text:
            continue

        if text.lower() in {
            "exit",
            "quit",
        }:
            print("Exiting.")
            break

        ids = tokenize(
            text,
            vocab,
        )

        logits = session.run(
            [output_meta.name],
            {
                input_meta.name:
                    ids.reshape(1, MAX_LEN)
            },
        )[0][0]

        probs = softmax(logits)

        order = np.argsort(
            probs
        )[::-1]

        print("\nV3 RESULT")
        print("-" * 70)

        top_idx = int(order[0])

        print(
            f"Intent     : {LABELS[top_idx]}"
        )
        print(
            f"Confidence : {probs[top_idx] * 100:.2f}%"
        )

        print("\nTop 3:")
        for rank, idx in enumerate(
            order[:3],
            start=1,
        ):
            idx = int(idx)
            print(
                f"{rank}. "
                f"{LABELS[idx]:35s} "
                f"{probs[idx] * 100:7.2f}%"
            )

        print(
            "\nToken IDs:",
            ids.tolist(),
        )


if __name__ == "__main__":
    main()
