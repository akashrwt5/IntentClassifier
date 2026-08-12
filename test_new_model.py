#!/usr/bin/env python3
import sys
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort

# Add project root to sys.path so we can import from new_semantic
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from new_semantic.scripts.common import load_vocab, encode


def main():
    model_dir = ROOT / "new_semantic" / "models_new_csv" / "en"
    onnx_path = model_dir / "student_new_csv_int8.onnx"

    # Fallback to FP32 if INT8 doesn't exist
    if not onnx_path.exists():
        onnx_path = model_dir / "student_new_csv.onnx"

    if not onnx_path.exists():
        print(f"Error: Model not found at {onnx_path}.")
        print("Please wait for the training pipeline to finish!")
        return

    vocab_path = model_dir / "vocab_new_csv.json"
    labels_path = model_dir / "labels_new_csv.json"
    meta_path = ROOT / "new_semantic" / "reports_new_csv" / "train_new_csv_summary.json"

    print("Loading vocab and labels...")
    vocab, tok_mode = load_vocab(vocab_path)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))

    # Get max_len from summary
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        max_len = meta.get("max_len", 24)
        tok_mode = meta.get("tokenizer", tok_mode)
    except:
        max_len = 24

    print(f"Loading ONNX model: {onnx_path.name}")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    print("\nModel is ready! Type your utterance and press Enter (or type 'quit' to exit).")
    print("-" * 60)

    while True:
        try:
            text = input("\n> ")
            if not text.strip() or text.lower() in ("quit", "exit"):
                break

            # Tokenize and encode the text
            ids, _ = encode(text, vocab, max_len=max_len, mode=tok_mode)
            X = np.array([ids], dtype=np.int64)
            M = X != 0  # PAD_ID is 0

            # Run inference
            logits = sess.run(None, {"input_ids": X, "attention_mask": M})[0][0]

            # Convert to probabilities using softmax
            z = logits - np.max(logits)
            probs = np.exp(z) / np.sum(np.exp(z))

            best_idx = np.argmax(probs)
            confidence = probs[best_idx]

            # The production engine uses a confidence threshold (usually 0.40).
            # If the best match is below the threshold, it is considered Out-Of-Domain.
            THRESHOLD = 0.40
            if confidence < THRESHOLD:
                predicted_intent = "Default Fallback Intent"
            else:
                predicted_intent = labels[best_idx]

            print(f"Intent     : {predicted_intent}")
            print(f"Confidence : {confidence:.3f}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
