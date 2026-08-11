#!/usr/bin/env python3
"""
V2.1 PYTHON MODEL TEST SCRIPT
--------------------------------
No quantization.
No retraining.
No changes to the locked test.

Tests:
1. Verified V2.1 FP32 PyTorch checkpoint on locked 57-intent test
2. Interactive manual sentence testing
3. Top-3 predictions + confidence
4. Batch custom testing from a text file
5. Inference speed
6. Saves manual/batch results

Model:
v3_57intent_v2_1_controlled/student_v3_57intent_v2_1_best_fp32.pt
"""

from pathlib import Path
import json
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


ROOT = Path("/Users/shuklam/IntentClassifier/semantic_project/57semanitc")

CHECKPOINT = (
    ROOT
    / "v3_57intent_v2_1_controlled"
    / "student_v3_57intent_v2_1_best_fp32.pt"
)

LOCKED_TEST = ROOT / "v3_57intent_locked_eval" / "locked_test_57intent.csv"
VOCAB = ROOT / "vocab.json"
LABELS = ROOT / "labels.json"

OUT = ROOT / "v3_57intent_python_test"
OUT.mkdir(parents=True, exist_ok=True)

MANUAL_RESULTS = OUT / "manual_test_results.csv"
CUSTOM_RESULTS = OUT / "custom_test_results.csv"
LOCKED_REPORT = OUT / "locked_test_report.txt"
SUMMARY = OUT / "python_test_summary.json"

MAX_LEN = 24
NUM_CLASSES = 57


# ---------------------------------------------------------------------
# Model architecture
# This matches the verified V2.1 checkpoint architecture used by the
# exact PyTorch <-> ONNX parity audit.
# ---------------------------------------------------------------------

class StudentModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes,
        d_model=64,
        max_len=24,
        num_layers=2,
        nhead=4,
        dim_feedforward=128,
        dropout=0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(max_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, input_ids):
        b, seq_len = input_ids.shape

        positions = torch.arange(
            seq_len,
            device=input_ids.device,
        ).unsqueeze(0)

        x = self.embedding(input_ids) + self.position(positions)
        x = self.encoder(x)

        # Mean pooling
        x = x.mean(dim=1)
        x = self.norm(x)

        return self.classifier(x)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_vocab(path):
    obj = load_json(path)

    if isinstance(obj, list):
        return {str(v): i for i, v in enumerate(obj)}

    if isinstance(obj, dict):
        for key in ("stoi", "vocab"):
            if isinstance(obj.get(key), dict):
                return obj[key]

        if all(isinstance(v, int) for v in obj.values()):
            return obj

    raise RuntimeError(f"Unsupported vocab format: {path}")


def load_labels(path):
    obj = load_json(path)

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        if isinstance(obj.get("labels"), list):
            return obj["labels"]

        if all(str(k).isdigit() for k in obj):
            return [
                v for k, v in sorted(obj.items(), key=lambda x: int(x[0]))
            ]

        if all(isinstance(v, int) for v in obj.values()):
            labels = [None] * (max(obj.values()) + 1)
            for label, idx in obj.items():
                labels[idx] = label
            return labels

    raise RuntimeError(f"Unsupported labels format: {path}")


def encode(text, vocab):
    tokens = re.findall(
        r"\w+|[^\w\s]",
        str(text).lower().strip(),
        flags=re.UNICODE,
    )

    unk = vocab.get(
        "<unk>",
        vocab.get("[UNK]", vocab.get("UNK", 1)),
    )

    pad = vocab.get(
        "<pad>",
        vocab.get("[PAD]", vocab.get("PAD", 0)),
    )

    ids = [
        int(vocab.get(token, unk))
        for token in tokens[:MAX_LEN]
    ]

    ids += [int(pad)] * (MAX_LEN - len(ids))

    return np.asarray(ids[:MAX_LEN], dtype=np.int64)


def find_column(df, candidates):
    lookup = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for name in candidates:
        if name in lookup:
            return lookup[name]

    return None


def load_model(vocab_size, labels):
    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if not isinstance(checkpoint, dict):
        raise RuntimeError("Unsupported checkpoint format")

    model = StudentModel(
        vocab_size=vocab_size,
        num_classes=len(labels),
        d_model=64,
        max_len=MAX_LEN,
        num_layers=2,
        nhead=4,
        dim_feedforward=128,
        dropout=0.1,
    )

    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    return model


def predict(model, vocab, labels, text):
    x = encode(text, vocab)

    with torch.no_grad():
        logits = model(
            torch.from_numpy(x).unsqueeze(0)
        )

        probs = torch.softmax(logits, dim=-1)[0]

    top_values, top_indices = torch.topk(
        probs,
        k=min(3, len(labels)),
    )

    top = []

    for score, idx in zip(
        top_values.tolist(),
        top_indices.tolist(),
    ):
        top.append(
            {
                "intent": labels[idx],
                "confidence": float(score),
            }
        )

    best_idx = int(top_indices[0])

    return {
        "prediction": labels[best_idx],
        "confidence": float(top_values[0]),
        "top3": top,
    }


# ---------------------------------------------------------------------
# Locked 57-intent test
# ---------------------------------------------------------------------

def run_locked_test(model, vocab, labels):
    print("\n" + "=" * 78)
    print("LOCKED 57-INTENT PYTHON TEST")
    print("=" * 78)

    df = pd.read_csv(LOCKED_TEST)

    text_col = find_column(
        df,
        ["text", "utterance", "phrase", "query", "sentence", "input"],
    )

    label_col = find_column(
        df,
        ["label", "intent", "target", "class"],
    )

    if text_col is None or label_col is None:
        raise RuntimeError(
            f"Could not identify text/label columns: {list(df.columns)}"
        )

    texts = df[text_col].fillna("").astype(str).tolist()
    true_labels = df[label_col].astype(str).tolist()

    label_to_id = {
        label: i
        for i, label in enumerate(labels)
    }

    unknown = sorted(
        set(true_labels) - set(label_to_id)
    )

    if unknown:
        raise RuntimeError(
            f"Locked CSV has labels absent from labels.json: {unknown}"
        )

    X = np.stack([
        encode(text, vocab)
        for text in texts
    ])

    y = np.asarray([
        label_to_id[label]
        for label in true_labels
    ])

    start = time.perf_counter()

    with torch.no_grad():
        logits = model(
            torch.from_numpy(X)
        )

        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(logits, dim=-1).numpy()

    elapsed = time.perf_counter() - start

    accuracy = accuracy_score(y, pred)

    report = classification_report(
        y,
        pred,
        target_names=labels,
        zero_division=0,
    )

    cm = confusion_matrix(y, pred)

    print(f"\nAccuracy       : {accuracy * 100:.4f}%")
    print(f"Rows           : {len(X)}")
    print(f"Total time     : {elapsed:.4f} sec")
    print(f"Rows/sec       : {len(X) / elapsed:.2f}")
    print(f"ms/row         : {(elapsed / len(X)) * 1000:.4f}")

    print("\nClassification report:")
    print(report)

    with open(LOCKED_REPORT, "w", encoding="utf-8") as f:
        f.write("V2.1 PYTHON LOCKED TEST\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Accuracy: {accuracy * 100:.4f}%\n")
        f.write(f"Rows: {len(X)}\n")
        f.write(f"Total time: {elapsed:.6f} sec\n")
        f.write(f"Rows/sec: {len(X) / elapsed:.2f}\n")
        f.write(f"ms/row: {(elapsed / len(X)) * 1000:.6f}\n\n")
        f.write(report)

    pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    ).to_csv(
        OUT / "locked_confusion_matrix.csv"
    )

    return {
        "accuracy": float(accuracy),
        "rows": len(X),
        "time_sec": float(elapsed),
        "rows_per_sec": float(len(X) / elapsed),
        "ms_per_row": float((elapsed / len(X)) * 1000),
    }


# ---------------------------------------------------------------------
# Interactive test
# ---------------------------------------------------------------------

def interactive_test(model, vocab, labels):
    print("\n" + "=" * 78)
    print("INTERACTIVE MODEL TEST")
    print("=" * 78)
    print("Type a sentence and press Enter.")
    print("Type 'exit' to stop.")
    print()

    rows = []

    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if text.lower() in {"exit", "quit", "q"}:
            break

        if not text:
            continue

        result = predict(
            model,
            vocab,
            labels,
            text,
        )

        print(f"\nPrediction : {result['prediction']}")
        print(f"Confidence : {result['confidence']:.4f}")

        print("Top-3:")
        for item in result["top3"]:
            print(
                f"  {item['confidence']:.4f} | "
                f"{item['intent']}"
            )

        print()

        rows.append({
            "text": text,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "top1": result["top3"][0]["intent"],
            "top1_confidence": result["top3"][0]["confidence"],
            "top2": result["top3"][1]["intent"],
            "top2_confidence": result["top3"][1]["confidence"],
            "top3": result["top3"][2]["intent"],
            "top3_confidence": result["top3"][2]["confidence"],
        })

    if rows:
        pd.DataFrame(rows).to_csv(
            MANUAL_RESULTS,
            index=False,
        )

        print(f"Saved: {MANUAL_RESULTS}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("=" * 78)
    print("V2.1 FP32 PYTHON MODEL TEST")
    print("=" * 78)

    for path in [
        CHECKPOINT,
        LOCKED_TEST,
        VOCAB,
        LABELS,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    vocab = load_vocab(VOCAB)
    labels = load_labels(LABELS)

    if len(labels) != NUM_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_CLASSES} labels, got {len(labels)}"
        )

    print(f"\nCheckpoint : {CHECKPOINT}")
    print(f"Vocab size : {len(vocab)}")
    print(f"Classes    : {len(labels)}")
    print("Quantization: NO")
    print("Retraining  : NO")

    model = load_model(
        len(vocab),
        labels,
    )

    locked_summary = run_locked_test(
        model,
        vocab,
        labels,
    )

    summary = {
        "status": "V2.1 PYTHON TEST COMPLETE",
        "checkpoint": str(CHECKPOINT),
        "quantization": False,
        "retraining": False,
        "locked_test_used": True,
        "locked_accuracy": locked_summary["accuracy"],
        "locked_rows": locked_summary["rows"],
        "inference_time_sec": locked_summary["time_sec"],
        "rows_per_sec": locked_summary["rows_per_sec"],
        "ms_per_row": locked_summary["ms_per_row"],
    }

    SUMMARY.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:")
    print(LOCKED_REPORT)
    print(OUT / "locked_confusion_matrix.csv")
    print(SUMMARY)

    # Interactive mode.
    interactive_test(
        model,
        vocab,
        labels,
    )


if __name__ == "__main__":
    main()
