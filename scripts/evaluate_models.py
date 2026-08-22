#!/usr/bin/env python3
"""Zero-shot bake-off: TF-IDF vs the student semantic head vs MiniLM vs the
distilled bge encoder, on datasets/semantic_benchmark_250.csv.

The bge encoder is loaded through scripts/semantic_compression/artifact.py
rather than by hand. This file used to carry its own copy of three decisions
that the compression directory had already got wrong once each -- which
tokenizer, which pooling, and what to do about out-of-range token ids -- and a
second copy is a second place for them to drift:

  * pooling was hardcoded to the CLS token. That is correct for bge and wrong
    for every mean-pooled export, and reading the wrong token scores a working
    encoder at 0.005.
  * `input_ids[input_ids >= 10000] = 100` turned an incompatible tokenizer into
    a silent 7% corruption instead of an error.
  * the classifier head was read from output_models/classifier_head.pkl, one
    level above the model. Two exports resolved to that same file and
    overwrote each other; the head now lives beside the model it belongs to.

Those decisions are now made once, by the artifact itself.
"""
import sys
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages" / "runtime"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "semantic_compression"))

from nlu_engine.engine import NLUEngine
from nlu_engine.semantic import SemanticFallback

from artifact import ArtifactContractError, encode, head_path, load_encoder


def main():
    csv_path = REPO_ROOT / "datasets" / "semantic_benchmark_250.csv"
    if not csv_path.exists():
        print(f"Error: Dataset {csv_path} does not exist.")
        sys.exit(1)

    print("Loading Models (TF-IDF, Student Semantic, MiniLM, and BGE Distilled)...")

    # 1 & 2. TF-IDF & Student Semantic
    try:
        engine = NLUEngine(language="en")
    except Exception as e:
        print(f"Failed to load NLUEngine: {e}")
        sys.exit(1)

    # 3. MiniLM
    try:
        minilm = SemanticFallback(threshold=0.0)
    except Exception as e:
        print(f"Error loading MiniLM: {e}")
        sys.exit(1)

    # 4. BGE Distilled -- tokenizer, pooling, id range and head all come from
    #    the artifact's own directory; nothing about it is decided here.
    bge_model_path = (
        REPO_ROOT
        / "scripts"
        / "semantic_compression"
        / "output_models"
        / "stage2_contrastive_bge_small_onnx"
        / "model_quantized.onnx"
    )
    bge_clf_path = head_path(bge_model_path)

    try:
        bge_encoder = load_encoder(bge_model_path)
        with open(bge_clf_path, "rb") as f:
            bge_classifier = pickle.load(f)
        encode(bge_encoder, ["warmup"])  # force ORT to allocate before timing
    except ArtifactContractError as exc:
        print(f"BGE artifact contract not satisfied:\n  {exc}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading BGE Distilled model: {e}")
        sys.exit(1)

    print(f"  BGE: {bge_encoder.summary()}")
    print(f"  head: {bge_clf_path.relative_to(REPO_ROOT)}")

    print(f"Reading dataset: {csv_path.name}")
    df = pd.read_csv(csv_path)

    tf_correct = sem_correct = mini_correct = bge_correct = 0
    total = len(df)

    tf_times, sem_times, mini_times, bge_times = [], [], [], []
    tf_errors, sem_errors, mini_errors, bge_errors = [], [], [], []

    print(f"\nRunning evaluation on {total} utterances. Please wait...\n")

    for idx, row in df.iterrows():
        text = str(row["utterance"])
        expected = str(row["expected_intent"])

        # Test TF-IDF (Stage 2)
        t0 = time.perf_counter()
        tf_intent, tf_conf = engine.classifier.classify(text)
        t1 = time.perf_counter()
        tf_times.append((t1 - t0) * 1000)

        # Test Student Semantic (Stage 3)
        t2 = time.perf_counter()
        sem_intent, sem_conf = engine.semantic.classify(text)
        t3 = time.perf_counter()
        sem_times.append((t3 - t2) * 1000)

        # Test MiniLM (Original Semantic)
        t4 = time.perf_counter()
        mini_intent, mini_conf = minilm.classify(text)
        t5 = time.perf_counter()
        mini_times.append((t5 - t4) * 1000)

        # Test BGE Distilled Model
        t6 = time.perf_counter()
        bge_vec = encode(bge_encoder, [text])[0]
        bge_probs = bge_classifier.predict_proba([bge_vec])[0]
        top_idx = np.argmax(bge_probs)
        bge_intent = bge_classifier.classes_[top_idx]
        bge_conf = bge_probs[top_idx]
        t7 = time.perf_counter()
        bge_times.append((t7 - t6) * 1000)

        # Calculate hits and errors
        if tf_intent == expected:
            tf_correct += 1
        else:
            tf_errors.append({"Model": "TF-IDF", "Utterance": text, "Expected": expected, "Predicted": tf_intent, "Confidence": f"{tf_conf:.2f}"})

        if sem_intent == expected:
            sem_correct += 1
        else:
            sem_errors.append({"Model": "Student", "Utterance": text, "Expected": expected, "Predicted": sem_intent, "Confidence": f"{sem_conf:.2f}"})

        if mini_intent == expected:
            mini_correct += 1
        else:
            mini_errors.append({"Model": "MiniLM", "Utterance": text, "Expected": expected, "Predicted": mini_intent, "Confidence": f"{mini_conf:.2f}"})

        if bge_intent == expected:
            bge_correct += 1
        else:
            bge_errors.append({"Model": "BGE Distilled", "Utterance": text, "Expected": expected, "Predicted": bge_intent, "Confidence": f"{bge_conf:.2f}"})

    tf_acc = (tf_correct / total) * 100
    sem_acc = (sem_correct / total) * 100
    mini_acc = (mini_correct / total) * 100
    bge_acc = (bge_correct / total) * 100

    avg_tf_time = sum(tf_times) / total
    avg_sem_time = sum(sem_times) / total
    avg_mini_time = sum(mini_times) / total
    avg_bge_time = sum(bge_times) / total

    print("=" * 105)
    print(" 📊 EVALUATION REPORT: TF-IDF vs Student vs MiniLM vs BGE Distilled (Zero-Shot)")
    print("=" * 105)
    print(f"Dataset Size: {total} utterances")
    print("-" * 105)
    print(f"{'Metric':<20} | {'TF-IDF':<15} | {'Student (Old)':<15} | {'MiniLM (Heavy)':<15} | {'BGE (New Distilled)'}")
    print("-" * 105)
    print(f"{'Accuracy':<20} | {tf_acc:>5.2f}% ({tf_correct:<3})  | {sem_acc:>5.2f}% ({sem_correct:<3})  | {mini_acc:>5.2f}% ({mini_correct:<3})  | {bge_acc:>5.2f}% ({bge_correct:<3})")
    print(f"{'Avg Latency (ms)':<20} | {avg_tf_time:>8.3f} ms   | {avg_sem_time:>8.3f} ms   | {avg_mini_time:>8.3f} ms   | {avg_bge_time:>8.3f} ms")
    print("-" * 105)

    # Save errors to CSV
    all_errors = tf_errors + sem_errors + mini_errors + bge_errors
    if all_errors:
        error_df = pd.DataFrame(all_errors)
        error_csv_path = REPO_ROOT / "datasets" / "evaluation_errors_zero_shot.csv"
        error_df.to_csv(error_csv_path, index=False)
        print(f"\n⚠️  Found {len(all_errors)} total mispredictions across all 4 models.")
        print(f"📁 Detailed error report saved to: {error_csv_path.name}")

if __name__ == "__main__":
    main()
