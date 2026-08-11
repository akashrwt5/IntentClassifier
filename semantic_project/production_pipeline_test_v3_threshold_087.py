#!/usr/bin/env python3
"""
V3 FP32 ONNX — END-TO-END PYTHON PRODUCTION PIPELINE TEST

Modes:
  t = type text
  m = microphone -> offline STT -> classifier
  q = quit

Pipeline:
  Audio / Text
      ↓
  Offline STT (microphone mode)
      ↓
  Text normalization
      ↓
  BPE tokenizer
      ↓
  V3 FP32 ONNX
      ↓
  11 intent logits
      ↓
  confidence + margin + entropy safety gate
      ↓
  FINAL INTENT / NO_INTENT

IMPORTANT:
- V3 FP32 ONNX is not modified.
- This script does not train or quantize anything.
- Microphone mode requires Vosk and a compatible offline Vosk model.
- If Vosk is unavailable, text mode still works.
"""

from pathlib import Path
import json
import re
import sys
import math
import queue
import argparse

import numpy as np
import onnxruntime as ort

ROOT = Path("/Users/shuklam/IntentClassifier/semantic_project")

MODEL = ROOT / "tiny_semantic_student_v3_fp32" / "v3_semantic_student_fp32.onnx"
VOCAB_FILE = ROOT / "tiny_semantic_student_v2_balanced" / "vocab.json"

# Optional: set this to your installed Vosk model directory.
# Example:
# VOSK_MODEL = ROOT / "vosk-model-small-en-us-0.15"
VOSK_MODEL = ROOT / "vosk-model-small-en-us-0.15"

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

MAX_LEN = 24

# ---------------------------------------------------------------------
# Safety gate
#
# IMPORTANT:
# These values are intentionally exposed as configuration.
# Do NOT treat them as final production calibration until they are
# validated against representative real-world OOD data.
# ---------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.87
MARGIN_THRESHOLD = 0.00
ENTROPY_THRESHOLD = 1.00

# If you want a conservative first test, set:
# CONFIDENCE_THRESHOLD = 0.87


def load_vocab():
    with open(VOCAB_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if (
        isinstance(raw.get("model"), dict)
        and isinstance(raw["model"].get("vocab"), dict)
    ):
        vocab = raw["model"]["vocab"]
    elif isinstance(raw.get("vocab"), dict):
        vocab = raw["vocab"]
    else:
        vocab = raw

    vocab = {str(k): int(v) for k, v in vocab.items()}

    pad = int(vocab.get("<pad>", vocab.get("[PAD]", 0)))
    unk = int(vocab.get("<unk>", vocab.get("[UNK]", 1)))
    cls = vocab.get("<cls>", vocab.get("[CLS]", None))
    sep = vocab.get("<sep>", vocab.get("[SEP]", None))

    return vocab, pad, unk, cls, sep


VOCAB, PAD, UNK, CLS, SEP = load_vocab()


def normalize_text(text):
    """
    Keep this conservative.
    Do not remove meaningful words such as:
      don't
      not
      off
      back on
      louder
      quieter
    """
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text):
    """
    Same compact vocabulary/subword strategy used by the exported model
    benchmark.
    """
    text = normalize_text(text)
    text = re.sub(r"([.!?,;:()'])", r" \1 ", text)

    ids = []

    if CLS is not None:
        ids.append(int(CLS))

    for word in text.split():

        if word in VOCAB:
            ids.append(int(VOCAB[word]))
            continue

        pos = 0
        matched = False

        while pos < len(word):

            best_id = None
            best_len = 0

            for end in range(len(word), pos, -1):
                piece = word[pos:end]

                for candidate in (piece, "##" + piece):
                    if candidate in VOCAB:
                        best_id = int(VOCAB[candidate])
                        best_len = len(piece)
                        break

                if best_id is not None:
                    break

            if best_id is None:
                ids.append(UNK)
                matched = True
                break

            ids.append(best_id)
            pos += best_len
            matched = True

        if not matched:
            ids.append(UNK)

    if SEP is not None:
        ids.append(int(SEP))

    ids = ids[:MAX_LEN]
    ids += [PAD] * (MAX_LEN - len(ids))

    return ids


def softmax(logits):
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / np.sum(exp)


def entropy(probs):
    p = probs[probs > 0]
    return float(-np.sum(p * np.log(p)))


def classify(text, session, input_name, output_name):
    normalized = normalize_text(text)
    ids = np.asarray([tokenize(normalized)], dtype=np.int64)

    logits = session.run(
        [output_name],
        {input_name: ids},
    )[0][0]

    probs = softmax(logits)
    order = np.argsort(probs)[::-1]

    top1 = int(order[0])
    top2 = int(order[1])

    confidence = float(probs[top1])
    second_confidence = float(probs[top2])
    margin = confidence - second_confidence
    ent = entropy(probs)

    accepted = (
        confidence >= CONFIDENCE_THRESHOLD
        and margin >= MARGIN_THRESHOLD
        and ent <= ENTROPY_THRESHOLD
    )

    return {
        "normalized": normalized,
        "token_ids": ids[0].tolist(),
        "probs": probs,
        "order": order,
        "top1": LABELS[top1],
        "confidence": confidence,
        "top2": LABELS[top2],
        "top2_confidence": second_confidence,
        "margin": margin,
        "entropy": ent,
        "accepted": accepted,
    }


def print_result(result):
    print()
    print("-" * 78)

    print("Normalized text:")
    print(result["normalized"])

    print()
    print("Top 3:")
    for rank, idx in enumerate(result["order"][:3], 1):
        print(
            f"{rank}. "
            f"{LABELS[int(idx)]:34s} "
            f"{float(result['probs'][idx]) * 100:7.2f}%"
        )

    print()
    print(f"Confidence : {result['confidence'] * 100:.2f}%")
    print(f"Margin     : {result['margin'] * 100:.2f} pp")
    print(f"Entropy    : {result['entropy']:.4f}")

    print()
    print("Safety Gate:")
    print(
        f"  confidence >= {CONFIDENCE_THRESHOLD:.2f} : "
        f"{'PASS' if result['confidence'] >= CONFIDENCE_THRESHOLD else 'FAIL'}"
    )
    print(
        f"  margin >= {MARGIN_THRESHOLD:.2f} : "
        f"{'PASS' if result['margin'] >= MARGIN_THRESHOLD else 'FAIL'}"
    )
    print(
        f"  entropy <= {ENTROPY_THRESHOLD:.2f} : "
        f"{'PASS' if result['entropy'] <= ENTROPY_THRESHOLD else 'FAIL'}"
    )

    print()

    if result["accepted"]:
        print("FINAL DECISION")
        print(f"  ACCEPT → {result['top1']}")
    else:
        print("FINAL DECISION")
        print("  REJECT → NO_INTENT / defaultFallbackIntent")

    print("-" * 78)


# ---------------------------------------------------------------------
# Optional Vosk microphone mode
# ---------------------------------------------------------------------
def microphone_once():
    try:
        import sounddevice as sd
        from vosk import Model, KaldiRecognizer
    except ImportError:
        print()
        print("Microphone mode requires:")
        print("  pip install vosk sounddevice")
        print()
        return None

    if not VOSK_MODEL.exists():
        print()
        print("Vosk model not found:")
        print(f"  {VOSK_MODEL}")
        print()
        print("Set VOSK_MODEL at the top of this script to your")
        print("offline Vosk model directory.")
        print()
        return None

    sample_rate = 16000
    audio_queue = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        audio_queue.put(bytes(indata))

    print()
    print("Loading offline Vosk model...")
    model = Model(str(VOSK_MODEL))
    recognizer = KaldiRecognizer(model, sample_rate)

    print("Speak now. Press Ctrl+C to stop.")
    print()

    try:
        with sd.RawInputStream(
            samplerate=sample_rate,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=callback,
        ):
            while True:
                data = audio_queue.get()

                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()

                    if text:
                        return text

    except KeyboardInterrupt:
        final = json.loads(recognizer.FinalResult())
        return final.get("text", "").strip()


def main():
    if not MODEL.exists():
        raise FileNotFoundError(f"V3 ONNX not found: {MODEL}")

    print("=" * 78)
    print("V3 FP32 ONNX — END-TO-END PRODUCTION PIPELINE TEST")
    print("=" * 78)
    print(f"Model : {MODEL}")
    print(f"Size  : {MODEL.stat().st_size / 1024 / 1024:.3f} MB")
    print(f"Vocab : {len(VOCAB)}")
    print(f"MaxLen: {MAX_LEN}")
    print()
    print("Pipeline:")
    print("  Audio/Text → normalization → BPE → V3 ONNX")
    print("            → confidence + margin + entropy")
    print("            → ACCEPT intent / NO_INTENT")
    print()
    print("Safety gate:")
    print(f"  confidence >= {CONFIDENCE_THRESHOLD}")
    print(f"  margin     >= {MARGIN_THRESHOLD}")
    print(f"  entropy    <= {ENTROPY_THRESHOLD}")
    print()
    print("NOTE: 0.87 is a calibration candidate, NOT the final production threshold.")
    print("Use microphone/OOD testing before locking it for release.")
    print()
    print("Modes:")
    print("  t = text input")
    print("  m = microphone + offline Vosk")
    print("  q = quit")
    print("=" * 78)

    session = ort.InferenceSession(
        str(MODEL),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print()
    print("ONNX Runtime:")
    print(f"  input  = {input_name}")
    print(f"  output = {output_name}")

    while True:
        print()
        mode = input("[t]ext / [m]icrophone / [q]uit: ").strip().lower()

        if mode in ("q", "quit", "exit"):
            break

        if mode in ("t", "text"):
            text = input("You: ").strip()

            if not text:
                continue

        elif mode in ("m", "mic", "microphone"):
            text = microphone_once()

            if not text:
                print("No speech recognized.")
                continue

            print(f"STT text: {text}")

        else:
            print("Please choose t, m, or q.")
            continue

        result = classify(
            text,
            session,
            input_name,
            output_name,
        )

        print_result(result)


if __name__ == "__main__":
    main()
