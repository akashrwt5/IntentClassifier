#!/usr/bin/env python3
"""
E5 Distilled V3 — EXACT CANONICAL LOCKED 57-INTENT BENCHMARK

Uses:
  V3 checkpoint:
    v3_57intent_e5_distilled_v3_hard_negative/
      student_e5_distilled_v3_best_fp32.pt

  Canonical locked test:
    v3_57intent_locked_eval/
      locked_test_57intent.csv

IMPORTANT:
- Uses ONLY the canonical locked test.
- Does NOT use train.csv.
- Does NOT modify the locked CSV.
- No quantization.
- No ONNX.
- Reports accuracy, macro F1, weighted F1,
  per-intent report and inference speed.
"""

from pathlib import Path
import json
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


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

LOCKED_CSV = (
    PROJECT
    / "v3_57intent_locked_eval"
    / "locked_test_57intent.csv"
)

OUT_DIR = (
    PROJECT
    / "v3_57intent_e5_distilled_v3_locked_benchmark"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREDICTIONS_CSV = (
    OUT_DIR
    / "locked_predictions_v3.csv"
)

REPORT_TXT = (
    OUT_DIR
    / "locked_test_report_v3.txt"
)

CONFUSION_CSV = (
    OUT_DIR
    / "confusion_matrix_v3.csv"
)

SUMMARY_JSON = (
    OUT_DIR
    / "benchmark_summary_v3.json"
)

MAX_LEN = 24
PAD_ID = 0
UNK_ID = 1

EMBED_DIM = 64
NHEAD = 4
FF_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.10

BATCH_SIZE = 256


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
        "EXACT CANONICAL LOCKED 57-INTENT BENCHMARK"
    )
    print("=" * 72)

    for path in [
        CHECKPOINT,
        VOCAB_JSON,
        LABEL_MAP_JSON,
        LOCKED_CSV,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    print()
    print(f"Checkpoint : {CHECKPOINT}")
    print(f"Locked CSV : {LOCKED_CSV}")

    df = pd.read_csv(
        LOCKED_CSV
    )

    if "text" not in df.columns:
        raise RuntimeError(
            "Locked CSV must contain a 'text' column."
        )

    if "intent" not in df.columns:
        raise RuntimeError(
            "Locked CSV must contain an 'intent' column."
        )

    if len(df) != 1686:
        raise RuntimeError(
            f"Expected exactly 1686 locked rows, "
            f"found {len(df)}."
        )

    labels = load_labels()

    if len(labels) != 57:
        raise RuntimeError(
            f"Expected 57 labels, found {len(labels)}."
        )

    locked_labels = set(
        df["intent"].astype(str).unique()
    )

    unknown = sorted(
        locked_labels - set(labels)
    )

    if unknown:
        raise RuntimeError(
            "Locked test contains labels missing "
            f"from label map: {unknown}"
        )

    vocab = json.loads(
        VOCAB_JSON.read_text(
            encoding="utf-8"
        )
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
            "No model_state_dict/state_dict "
            "found in checkpoint."
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
            for text in df["text"].astype(str)
        ],
        dtype=np.int64,
    )

    label_to_id = {
        label: i
        for i, label in enumerate(labels)
    }

    y_true = np.asarray(
        [
            label_to_id[str(x)]
            for x in df["intent"]
        ],
        dtype=np.int64,
    )

    print()
    print(
        "Running V3 inference on "
        f"{len(X)} locked rows..."
    )

    predictions = []
    confidences = []

    start_time = time.perf_counter()

    with torch.no_grad():

        for start in range(
            0,
            len(X),
            BATCH_SIZE,
        ):

            xb = torch.from_numpy(
                X[
                    start:
                    start + BATCH_SIZE
                ]
            )

            logits = model(xb)

            probs = torch.softmax(
                logits,
                dim=1,
            )

            conf, pred = probs.max(
                dim=1
            )

            predictions.extend(
                pred.numpy().tolist()
            )

            confidences.extend(
                conf.numpy().tolist()
            )

    total_time = (
        time.perf_counter()
        - start_time
    )

    y_pred = np.asarray(
        predictions,
        dtype=np.int64,
    )

    confidences = np.asarray(
        confidences,
        dtype=np.float32,
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(57)),
        target_names=labels,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(57)),
    )

    print()
    print("--- V3 LOCKED 57-INTENT TEST RESULT ---")
    print(
        f"Accuracy   : {accuracy * 100:.4f}%"
    )
    print(
        f"Macro F1   : {macro_f1 * 100:.4f}%"
    )
    print(
        f"Weighted F1: {weighted_f1 * 100:.4f}%"
    )

    print()
    print(report)

    rows = []

    for i in range(len(df)):

        rows.append(
            {
                "row_index": i,
                "text": df.iloc[i]["text"],
                "true_intent": labels[
                    y_true[i]
                ],
                "predicted_intent": labels[
                    y_pred[i]
                ],
                "confidence": float(
                    confidences[i]
                ),
                "correct": bool(
                    y_true[i] == y_pred[i]
                ),
            }
        )

    pred_df = pd.DataFrame(
        rows
    )

    pred_df.to_csv(
        PREDICTIONS_CSV,
        index=False,
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    )

    cm_df.to_csv(
        CONFUSION_CSV
    )

    rows_per_sec = (
        len(X)
        / total_time
    )

    ms_per_row = (
        total_time
        / len(X)
        * 1000
    )

    report_text = (
        "# E5 DISTILLED V3 "
        "LOCKED 57-INTENT TEST\n\n"
        f"Rows       : {len(X)}\n"
        f"Accuracy   : {accuracy * 100:.4f}%\n"
        f"Macro F1   : {macro_f1 * 100:.4f}%\n"
        f"Weighted F1: {weighted_f1 * 100:.4f}%\n\n"
        "Classification report:\n"
        f"{report}\n\n"
        "--- INFERENCE SPEED ---\n"
        f"Total rows : {len(X)}\n"
        f"Total time : {total_time:.4f} sec\n"
        f"Rows/sec   : {rows_per_sec:.2f}\n"
        f"ms/row     : {ms_per_row:.4f}\n"
    )

    REPORT_TXT.write_text(
        report_text,
        encoding="utf-8",
    )

    summary = {
        "model": str(CHECKPOINT),
        "locked_csv": str(LOCKED_CSV),
        "rows": int(len(X)),
        "num_intents": 57,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "total_time_sec": float(total_time),
        "rows_per_sec": float(rows_per_sec),
        "ms_per_row": float(ms_per_row),
        "mean_confidence": float(
            confidences.mean()
        ),
        "median_confidence": float(
            np.median(confidences)
        ),
        "quantization": False,
        "onnx": False,
        "locked_test_used": True,
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("--- INFERENCE SPEED ---")
    print(
        f"Total rows : {len(X)}"
    )
    print(
        f"Total time : {total_time:.4f} sec"
    )
    print(
        f"Rows/sec   : {rows_per_sec:.2f}"
    )
    print(
        f"ms/row     : {ms_per_row:.4f}"
    )

    print()
    print("Saved:")
    print(PREDICTIONS_CSV)
    print(REPORT_TXT)
    print(CONFUSION_CSV)
    print(SUMMARY_JSON)

    print()
    print(
        "STATUS: "
        "E5 DISTILLED V3 LOCKED 57-INTENT "
        "BENCHMARK COMPLETE"
    )


if __name__ == "__main__":
    main()
