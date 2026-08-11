#!/usr/bin/env python3

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer


# ============================================================
# V6 FROZEN PRODUCTION BUNDLE
# ============================================================

ROOT = Path(
    "/Users/shuklam/IntentClassifier/semantic_project/57semanitc"
)

ENCODER_PATH = ROOT / "v6_production_frozen" / "encoder"
CLASSIFIER_PATH = ROOT / "v6_production_frozen" / "intent_classifier.joblib"
LABEL_MAP_PATH = ROOT / "v6_production_frozen" / "label_map.json"

OUTPUT_DIR = ROOT / "v6_python_inference_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [
    # Volume
    ("make it louder", "Cmd.VolumeIncrease"),
    ("turn the volume up", "Cmd.VolumeIncrease"),
    ("it's too loud", "Cmd.VolumeDecrease"),
    ("make it quieter", "Cmd.VolumeDecrease"),
    ("mute everything", "Cmd.VolumeMute"),
    ("cut all the sound", "Cmd.VolumeMute"),
    ("turn the sound back on", "Cmd.VolumeUnmute"),
    ("turn on my hearing aids", "Cmd.VolumeUnmute"),

    # Battery
    ("how much battery do I have", "Cmd.BatteryLevel"),
    ("are my hearing aids getting low", "Cmd.BatteryLevel"),
    ("will they last through the movie", "Cmd.BatteryLevel"),
    ("should I charge them now", "Cmd.BatteryLevel"),

    # Memory
    ("change my hearing aid program", "Cmd.MemoryChange"),
    ("load my normal configuration", "Cmd.MemoryChange"),
    ("what is memory", "Help_ChangingMemories"),
    ("how do different memories work", "Help_ChangingMemories"),
    ("how do I access my hearing aid programs", "Help_MemoryOptions"),
    ("can I set a favourite program", "Help_MemoryOptions"),

    # Find phone / hearing aids
    ("where is my phone", "Cmd.FindMyPhone"),
    ("buzz my phone so I can hear it", "Cmd.FindMyPhone"),
    ("where are my hearing aids", "Help_FindMyHearingAids"),
    ("how can I find my left hearing aid", "Help_FindMyHearingAids"),

    # Reminders
    ("set a reminder", "reminders.add"),
    ("remind me to take my medicine", "reminders.add"),
    ("I took my pills, clear the reminder", "reminders.complete"),
    ("I did it, cross it off", "reminders.complete"),
    ("that's handled, cross it off", "reminders.complete"),

    # Streaming
    ("start streaming", "Cmd.StreamingStart"),
    ("start my audio stream", "Cmd.StreamingStart"),
    ("stop streaming", "Cmd.StreamingStop"),
    ("stop the audio stream", "Cmd.StreamingStop"),

    # Messaging
    ("send a text", "Cmd.SendMessage"),
    ("text a contact with my news", "Cmd.SendMessage"),

    # Activity
    ("log the miles I just covered", "Cmd.ActivityRun"),
    ("how many steps did I take", "Cmd.ActivityStep"),
    ("how long have I been standing", "Cmd.ActivityStand"),
    ("how far did I walk", "Cmd.ActivityWalk"),

    # Self check
    ("run a self check", "Help_SelfCheck"),
    ("how do I know if my microphone is working", "Help_SelfCheck"),

    # Device settings
    ("where are my device settings", "Help_DeviceSettings"),
    ("how do I change my device settings", "Help_DeviceSettings"),

    # Home / What's new
    ("how do I navigate the app", "Help_Home"),
    ("where do I go to do things", "Help_Home"),
    ("show me the quick start guide", "Help_WhatsNew"),

    # Care / insertion
    ("how should I store my hearing aids overnight", "Help_CleanCare"),
    ("how do I wear my hearing aid correctly", "Help_InsertDevice"),

    # OOD / fallback
    ("what is the weather today", "Default Fallback Intent"),
    ("tell me a joke", "Default Fallback Intent"),
    ("what is the capital of France", "Default Fallback Intent"),
    ("play some music", "Default Fallback Intent"),
]


# ============================================================
# LOAD LABEL MAP
# ============================================================

def load_label_map():
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        int(k): str(v)
        for k, v in data.items()
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("V6 PYTHON INFERENCE TEST")
    print("=" * 100)

    print("\nEncoder   :", ENCODER_PATH)
    print("Classifier:", CLASSIFIER_PATH)
    print("Label map :", LABEL_MAP_PATH)
    print("Test cases:", len(TEST_CASES))

    # --------------------------------------------------------
    # Load label map
    # --------------------------------------------------------

    print("\nLoading label map...")

    index_to_label = load_label_map()

    print(
        "Label-map entries:",
        len(index_to_label)
    )

    # --------------------------------------------------------
    # Load classifier
    # --------------------------------------------------------

    print("\nLoading V6 classifier...")

    classifier = joblib.load(CLASSIFIER_PATH)

    print(
        "Classifier classes:",
        len(classifier.classes_)
    )

    print(
        "Classifier dtype:",
        classifier.classes_.dtype
    )

    if len(index_to_label) != len(classifier.classes_):
        raise RuntimeError(
            "Classifier and label-map class count mismatch."
        )

    # --------------------------------------------------------
    # Load encoder
    # --------------------------------------------------------

    print("\nLoading V6 E5 encoder...")

    encoder = SentenceTransformer(
        str(ENCODER_PATH)
    )

    encoder.eval()

    # --------------------------------------------------------
    # Prepare inputs
    # --------------------------------------------------------

    texts = [x[0] for x in TEST_CASES]
    expected = [x[1] for x in TEST_CASES]

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    print("\nGenerating embeddings...")

    t0 = time.perf_counter()

    embeddings = encoder.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    embedding_time = time.perf_counter() - t0

    print(
        "Embedding shape:",
        embeddings.shape
    )

    print(
        "Embedding time:",
        f"{embedding_time:.4f} sec"
    )

    # --------------------------------------------------------
    # Classifier
    # --------------------------------------------------------

    print("\nRunning V6 classifier...")

    t1 = time.perf_counter()

    probabilities = classifier.predict_proba(
        embeddings
    )

    raw_predictions = classifier.predict(
        embeddings
    )

    classifier_time = time.perf_counter() - t1

    # --------------------------------------------------------
    # Convert class IDs -> labels
    # --------------------------------------------------------

    predictions = [
        index_to_label[int(x)]
        for x in raw_predictions
    ]

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    correct = 0
    high_conf_wrong = 0
    low_confidence = 0

    print("\n")
    print("=" * 100)
    print("TEST RESULTS")
    print("=" * 100)

    for i, (
        text,
        expected_label,
        prediction,
        probs
    ) in enumerate(
        zip(
            texts,
            expected,
            predictions,
            probabilities
        ),
        start=1
    ):

        top_indices = np.argsort(probs)[::-1][:3]

        top3 = []

        for idx in top_indices:
            label = index_to_label[int(idx)]
            score = float(probs[idx])

            top3.append(
                f"{label}={score:.4f}"
            )

        confidence = float(
            np.max(probs)
        )

        passed = (
            prediction ==
            expected_label
        )

        if passed:
            correct += 1
        else:
            if confidence >= 0.80:
                high_conf_wrong += 1

        if confidence < 0.50:
            low_confidence += 1

        status = "PASS" if passed else "FAIL"

        print()
        print(
            f"[{i:02d}] {status}"
        )
        print(
            "Text       :",
            text
        )
        print(
            "Expected   :",
            expected_label
        )
        print(
            "Prediction :",
            prediction
        )
        print(
            "Confidence :",
            f"{confidence * 100:.2f}%"
        )
        print(
            "Top-3      :",
            " | ".join(top3)
        )

        results.append({
            "text": text,
            "expected": expected_label,
            "prediction": prediction,
            "confidence": confidence,
            "correct": passed,
            "top1": top3[0],
            "top2": top3[1],
            "top3": top3[2],
        })

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total = len(results)

    accuracy = (
        correct / total
        if total
        else 0.0
    )

    total_time = (
        embedding_time +
        classifier_time
    )

    rows_per_sec = (
        total / total_time
        if total_time > 0
        else 0
    )

    print("\n")
    print("=" * 100)
    print("V6 PYTHON INFERENCE SUMMARY")
    print("=" * 100)

    print()
    print(
        "Total test cases       :",
        total
    )

    print(
        "Correct                :",
        f"{correct}/{total}"
    )

    print(
        "Accuracy               :",
        f"{accuracy * 100:.2f}%"
    )

    print(
        "Wrong                  :",
        total - correct
    )

    print(
        "High-confidence wrong :",
        high_conf_wrong
    )

    print(
        "Low-confidence (<50%) :",
        low_confidence
    )

    print()
    print(
        "Embedding time         :",
        f"{embedding_time:.4f} sec"
    )

    print(
        "Classifier time        :",
        f"{classifier_time:.4f} sec"
    )

    print(
        "Total inference time   :",
        f"{total_time:.4f} sec"
    )

    print(
        "Rows/sec               :",
        f"{rows_per_sec:.2f}"
    )

    # --------------------------------------------------------
    # Failures
    # --------------------------------------------------------

    failures = [
        x for x in results
        if not x["correct"]
    ]

    if failures:

        print("\n")
        print("=" * 100)
        print("FAILURES")
        print("=" * 100)

        for x in failures:

            print()
            print(
                "Text      :",
                x["text"]
            )
            print(
                "Expected  :",
                x["expected"]
            )
            print(
                "Predicted :",
                x["prediction"]
            )
            print(
                "Confidence:",
                f"{x['confidence'] * 100:.2f}%"
            )
            print(
                "Top-3     :",
                x["top1"],
                "|",
                x["top2"],
                "|",
                x["top3"]
            )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    result_csv = (
        OUTPUT_DIR /
        "v6_python_inference_results.csv"
    )

    pd.DataFrame(results).to_csv(
        result_csv,
        index=False
    )

    # --------------------------------------------------------
    # Save JSON summary
    # --------------------------------------------------------

    summary = {
        "model": "V6 frozen production",
        "total_cases": total,
        "correct": correct,
        "wrong": total - correct,
        "accuracy": accuracy,
        "high_confidence_wrong": high_conf_wrong,
        "low_confidence_below_0_50": low_confidence,
        "embedding_time_sec": embedding_time,
        "classifier_time_sec": classifier_time,
        "total_inference_time_sec": total_time,
        "rows_per_second": rows_per_sec,
        "num_classes": len(classifier.classes_),
        "embedding_dimension": int(
            embeddings.shape[1]
        ),
    }

    result_json = (
        OUTPUT_DIR /
        "v6_python_inference_summary.json"
    )

    with open(
        result_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n")
    print("=" * 100)
    print("SAVED")
    print("=" * 100)

    print()
    print(result_csv)
    print(result_json)

    print()
    print(
        "STATUS: V6 PYTHON INFERENCE TEST COMPLETE"
    )


if __name__ == "__main__":
    main()
