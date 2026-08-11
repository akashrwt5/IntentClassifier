#!/usr/bin/env python3

from pathlib import Path
import json
import re
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent

MODEL = ROOT / "tiny_semantic_student_v2_int8" / "v2_semantic_student_int8.onnx"
VOCAB = ROOT / "tiny_semantic_student_v1" / "vocab.json"
LABELS = ROOT / "tiny_semantic_student_v1" / "intent_labels.txt"
CONFIG = ROOT / "tiny_semantic_student_v1" / "config.json"

if not MODEL.exists():
    raise FileNotFoundError(f"Model not found: {MODEL}")
if not VOCAB.exists():
    raise FileNotFoundError(f"Vocabulary not found: {VOCAB}")
if not LABELS.exists():
    raise FileNotFoundError(f"Labels not found: {LABELS}")

with open(VOCAB, "r", encoding="utf-8") as f:
    vocab = json.load(f)

with open(LABELS, "r", encoding="utf-8") as f:
    labels = [x.strip() for x in f if x.strip()]

if CONFIG.exists():
    with open(CONFIG, "r", encoding="utf-8") as f:
        config = json.load(f)
else:
    config = {}

MAX_LEN = int(
    config.get(
        "max_len",
        config.get("max_length", 24)
    )
)

PAD = int(vocab.get("<PAD>", vocab.get("[PAD]", 0)))
UNK = int(vocab.get("<UNK>", vocab.get("[UNK]", 1)))


def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def encode(text):
    tokens = clean_text(text).split()

    ids = [
        int(vocab.get(token, UNK))
        for token in tokens
    ][:MAX_LEN]

    ids += [PAD] * (MAX_LEN - len(ids))
    return ids


session = ort.InferenceSession(
    str(MODEL),
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("=" * 65)
print("V2 INT8 INTERACTIVE TEST")
print("=" * 65)
print(f"Model      : {MODEL}")
print(f"Model size : {MODEL.stat().st_size / 1024 / 1024:.3f} MB")
print(f"Input      : {input_name}")
print(f"Output     : {output_name}")
print()
print("Type a sentence and press Enter.")
print("Type 'exit' or 'quit' to stop.")
print("=" * 65)


while True:
    try:
        text = input("\nYou: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting...")
        break

    if not text:
        continue

    if text.lower() in {"exit", "quit"}:
        print("Exiting...")
        break

    x = np.asarray(
        [encode(text)],
        dtype=np.int64
    )

    logits = session.run(
        [output_name],
        {input_name: x}
    )[0][0].astype(np.float64)

    logits -= np.max(logits)

    probs = np.exp(logits)
    probs /= np.sum(probs)

    order = np.argsort(probs)[::-1]

    best = int(order[0])

    print("\nPrediction:")
    print(f"Intent     : {labels[best]}")
    print(f"Confidence : {probs[best] * 100:.2f}%")

    print("\nTop 3:")
    for rank, idx in enumerate(order[:3], 1):
        print(
            f"{rank}. {labels[int(idx)]:35s}"
            f" {probs[int(idx)] * 100:.2f}%"
        )
