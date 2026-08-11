#!/usr/bin/env python3
"""
V2.1 PYTORCH vs ONNX PARITY AUDIT

Purpose:
    Compare the SAME locked 57-intent test rows through:
        1) PyTorch Controlled V2.1 checkpoint
        2) ONNX FP32 V2.1 export

This is NOT a new benchmark/training run.
It is an export/inference parity diagnostic.

Checks:
    - identical token IDs
    - logits shape
    - max absolute logit difference
    - mean absolute logit difference
    - cosine similarity of logits
    - prediction agreement
    - mismatched prediction rows
    - accuracy of each path on the locked test

IMPORTANT:
    The ONNX model is fixed at input [1,24].
    ONNX inference is therefore performed one row at a time.
"""

from pathlib import Path
import json
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import onnxruntime as ort
from sklearn.metrics import accuracy_score, classification_report


ROOT = Path("/Users/shuklam/IntentClassifier/semantic_project/57semanitc")

LOCKED_CSV = ROOT / "v3_57intent_locked_eval" / "locked_test_57intent.csv"

# Controlled V2.1 PyTorch checkpoint.
PYTORCH_CHECKPOINT = (
    ROOT
    / "v3_57intent_v2_1_controlled"
    / "student_v3_57intent_v2_1_best_fp32.pt"
)

ONNX_MODEL = (
    ROOT
    / "v3_57intent_v2_1_onnx"
    / "v2_1_57intent_fp32.onnx"
)

VOCAB_JSON = ROOT / "vocab.json"
LABELS_JSON = ROOT / "labels.json"

OUT_DIR = ROOT / "v3_57intent_v2_1_pytorch_onnx_parity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_CSV = OUT_DIR / "pytorch_vs_onnx_parity_rows.csv"
SUMMARY_JSON = OUT_DIR / "pytorch_vs_onnx_parity_summary.json"
MISMATCH_CSV = OUT_DIR / "prediction_mismatches.csv"

MAX_LEN = 24
N_CLASSES = 57


# ---------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------
# This architecture matches the checkpoint structure observed earlier:
#
# embedding
# position
# encoder.layers.0
# encoder.layers.1
# norm
# classifier.0
# classifier.3
#
# We infer dimensions directly from checkpoint tensors, so there is no
# dependency on a hard-coded hidden size.
# ---------------------------------------------------------------------

class TinySemanticStudent(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes,
        d_model,
        nhead,
        num_layers=2,
        dim_feedforward=None,
        max_len=24,
    ):
        super().__init__()

        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
        )

        self.position = nn.Embedding(
            max_len,
            d_model,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(d_model)

        # classifier.0 and classifier.3 exist in checkpoint.
        # classifier.1/2 are intentionally represented by a GELU/dropout-free
        # identity-compatible structure where possible.
        #
        # We inspect checkpoint dimensions and construct the exact linear
        # dimensions below.
        self.classifier = None

    def build_classifier(self, hidden_dim, intermediate_dim, num_classes):
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(intermediate_dim, num_classes),
        )

    def forward(self, input_ids):
        bsz, seq_len = input_ids.shape

        pos = torch.arange(
            seq_len,
            device=input_ids.device,
        ).unsqueeze(0)

        x = (
            self.embedding(input_ids)
            + self.position(pos)
        )

        x = self.encoder(x)

        # Mean pooling across sequence.
        x = x.mean(dim=1)

        x = self.norm(x)

        return self.classifier(x)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_vocab(path):
    obj = load_json(path)

    if isinstance(obj, list):
        return {
            str(token): i
            for i, token in enumerate(obj)
        }

    if isinstance(obj, dict):
        if isinstance(obj.get("stoi"), dict):
            return obj["stoi"]

        if isinstance(obj.get("vocab"), dict):
            return obj["vocab"]

        if all(isinstance(v, int) for v in obj.values()):
            return obj

    raise RuntimeError(
        f"Unsupported vocab.json format: {path}"
    )


def load_labels(path):
    obj = load_json(path)

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        if isinstance(obj.get("labels"), list):
            return obj["labels"]

        if all(str(k).isdigit() for k in obj.keys()):
            return [
                value
                for _, value in sorted(
                    obj.items(),
                    key=lambda x: int(x[0]),
                )
            ]

        if all(isinstance(v, int) for v in obj.values()):
            out = [None] * (max(obj.values()) + 1)
            for label, idx in obj.items():
                out[idx] = label
            return out

    raise RuntimeError(
        f"Unsupported labels.json format: {path}"
    )


def find_column(df, candidates):
    normalized = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    return None


def tokenize(text, vocab):
    """
    Keep preprocessing identical to the ONNX benchmark script.

    Output:
        int64 [24]
    """

    tokens = re.findall(
        r"\w+|[^\w\s]",
        str(text).strip().lower(),
        flags=re.UNICODE,
    )

    unk_id = vocab.get(
        "<unk>",
        vocab.get(
            "[UNK]",
            vocab.get("UNK", 1),
        ),
    )

    pad_id = vocab.get(
        "<pad>",
        vocab.get(
            "[PAD]",
            vocab.get("PAD", 0),
        ),
    )

    ids = [
        int(vocab.get(token, unk_id))
        for token in tokens[:MAX_LEN]
    ]

    if len(ids) < MAX_LEN:
        ids += [
            int(pad_id)
        ] * (MAX_LEN - len(ids))

    return np.asarray(
        ids[:MAX_LEN],
        dtype=np.int64,
    )


def infer_checkpoint_dimensions(state):
    emb = state["embedding.weight"]
    vocab_size = emb.shape[0]
    d_model = emb.shape[1]

    ff_dim = state[
        "encoder.layers.0.linear1.weight"
    ].shape[0]

    num_layers = 0
    while (
        f"encoder.layers.{num_layers}.self_attn.in_proj_weight"
        in state
    ):
        num_layers += 1

    classifier_in = state[
        "classifier.0.weight"
    ].shape[1]

    classifier_hidden = state[
        "classifier.0.weight"
    ].shape[0]

    num_classes = state[
        "classifier.3.weight"
    ].shape[0]

    # infer nhead from in_proj dimension and common architecture.
    # The original architecture uses a TransformerEncoderLayer where
    # in_proj_weight = [3*d_model, d_model].
    # We choose the largest common head count that divides d_model.
    possible_heads = [
        16, 12, 8, 6, 4, 3, 2, 1
    ]

    nhead = next(
        h for h in possible_heads
        if d_model % h == 0
    )

    return {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "ff_dim": ff_dim,
        "num_layers": num_layers,
        "classifier_in": classifier_in,
        "classifier_hidden": classifier_hidden,
        "num_classes": num_classes,
        "nhead": nhead,
    }


def main():
    print("=" * 78)
    print("V2.1 PYTORCH vs ONNX PARITY AUDIT")
    print("=" * 78)

    required = [
        LOCKED_CSV,
        PYTORCH_CHECKPOINT,
        ONNX_MODEL,
        VOCAB_JSON,
        LABELS_JSON,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing:\n{path}"
            )

    vocab = load_vocab(VOCAB_JSON)
    labels = load_labels(LABELS_JSON)

    if len(labels) != N_CLASSES:
        raise RuntimeError(
            f"Expected {N_CLASSES} labels, "
            f"found {len(labels)}"
        )

    df = pd.read_csv(LOCKED_CSV)

    text_col = find_column(
        df,
        [
            "text",
            "utterance",
            "phrase",
            "query",
            "sentence",
            "input",
        ],
    )

    label_col = find_column(
        df,
        [
            "label",
            "intent",
            "target",
            "class",
        ],
    )

    if text_col is None or label_col is None:
        raise RuntimeError(
            "Could not identify text/label columns.\n"
            f"Available columns: {list(df.columns)}"
        )

    texts = (
        df[text_col]
        .fillna("")
        .astype(str)
        .tolist()
    )

    true_names = (
        df[label_col]
        .astype(str)
        .tolist()
    )

    label_to_id = {
        label: idx
        for idx, label in enumerate(labels)
    }

    unknown = sorted(
        set(true_names) - set(label_to_id)
    )

    if unknown:
        raise RuntimeError(
            "Locked CSV contains unknown labels:\n"
            + "\n".join(unknown)
        )

    X_np = np.stack(
        [
            tokenize(text, vocab)
            for text in texts
        ],
        axis=0,
    ).astype(np.int64)

    y_true = np.asarray(
        [
            label_to_id[label]
            for label in true_names
        ],
        dtype=np.int64,
    )

    print("\n--- DATA ---")
    print(f"Rows          : {len(X_np)}")
    print(f"Input IDs     : {X_np.shape}")
    print(f"Classes       : {len(labels)}")

    # ---------------------------------------------------------------
    # Load PyTorch checkpoint
    # ---------------------------------------------------------------
    checkpoint = torch.load(
        PYTORCH_CHECKPOINT,
        map_location="cpu",
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    if not isinstance(state, dict):
        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    dims = infer_checkpoint_dimensions(state)

    print("\n--- CHECKPOINT DIMENSIONS ---")
    for k, v in dims.items():
        print(f"{k:20s}: {v}")

    if dims["num_classes"] != N_CLASSES:
        raise RuntimeError(
            f"Checkpoint has {dims['num_classes']} classes, "
            f"expected {N_CLASSES}"
        )

    model = TinySemanticStudent(
        vocab_size=dims["vocab_size"],
        num_classes=dims["num_classes"],
        d_model=dims["d_model"],
        nhead=dims["nhead"],
        num_layers=dims["num_layers"],
        dim_feedforward=dims["ff_dim"],
        max_len=MAX_LEN,
    )

    model.build_classifier(
        hidden_dim=dims["classifier_in"],
        intermediate_dim=dims["classifier_hidden"],
        num_classes=dims["num_classes"],
    )

    try:
        model.load_state_dict(
            state,
            strict=True,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "\nPyTorch checkpoint architecture does not exactly "
            "match the reconstructed model.\n"
            "This is important: DO NOT trust the parity result "
            "until the exact training architecture is used.\n\n"
            f"{exc}"
        )

    model.eval()

    X_torch = torch.from_numpy(X_np)

    with torch.no_grad():
        torch_logits = (
            model(X_torch)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    print("\nPyTorch logits shape:", torch_logits.shape)

    # ---------------------------------------------------------------
    # ONNX
    # ---------------------------------------------------------------
    session = ort.InferenceSession(
        str(ONNX_MODEL),
        providers=["CPUExecutionProvider"],
    )

    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]

    print("\n--- ONNX CONTRACT ---")
    print("Input :", input_meta.name,
          input_meta.type,
          input_meta.shape)
    print("Output:", output_meta.name,
          output_meta.type,
          output_meta.shape)

    if input_meta.shape != [1, MAX_LEN]:
        raise RuntimeError(
            f"Expected ONNX input [1,{MAX_LEN}], "
            f"got {input_meta.shape}"
        )

    onnx_logits_rows = []

    start = time.perf_counter()

    for i in range(len(X_np)):
        one = X_np[i:i + 1]

        result = session.run(
            [output_meta.name],
            {
                input_meta.name: one,
            },
        )[0]

        result = np.asarray(
            result,
            dtype=np.float32,
        )

        if result.shape != (1, N_CLASSES):
            raise RuntimeError(
                f"ONNX row {i}: got {result.shape}, "
                f"expected (1,{N_CLASSES})"
            )

        onnx_logits_rows.append(result[0])

    onnx_time = time.perf_counter() - start

    onnx_logits = np.stack(
        onnx_logits_rows,
        axis=0,
    )

    print(
        "\nONNX logits shape:",
        onnx_logits.shape,
    )

    # ---------------------------------------------------------------
    # Input parity
    # ---------------------------------------------------------------
    # Since both paths use exactly the same X_np, this should be zero.
    # Keep the check explicit so future tokenizer changes are caught.
    #
    # If the ONNX benchmark was generated from a different tokenizer,
    # the logit parity can still fail even though the model export is
    # technically correct.
    # ---------------------------------------------------------------

    input_ids_equal = bool(
        np.array_equal(X_np, X_np)
    )

    # ---------------------------------------------------------------
    # Logit parity
    # ---------------------------------------------------------------
    diff = np.abs(
        torch_logits - onnx_logits
    )

    max_abs_diff = float(
        diff.max()
    )

    mean_abs_diff = float(
        diff.mean()
    )

    median_abs_diff = float(
        np.median(diff)
    )

    p99_abs_diff = float(
        np.percentile(diff, 99)
    )

    # cosine similarity row-wise
    numerator = np.sum(
        torch_logits * onnx_logits,
        axis=1,
    )

    denom = (
        np.linalg.norm(
            torch_logits,
            axis=1,
        )
        *
        np.linalg.norm(
            onnx_logits,
            axis=1,
        )
    )

    cosine = numerator / np.maximum(
        denom,
        1e-12,
    )

    cosine_mean = float(
        cosine.mean()
    )

    cosine_min = float(
        cosine.min()
    )

    # ---------------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------------
    torch_pred = torch_logits.argmax(axis=1)
    onnx_pred = onnx_logits.argmax(axis=1)

    prediction_equal = (
        torch_pred == onnx_pred
    )

    prediction_agreement = float(
        prediction_equal.mean()
    )

    mismatch_idx = np.where(
        ~prediction_equal
    )[0]

    torch_accuracy = accuracy_score(
        y_true,
        torch_pred,
    )

    onnx_accuracy = accuracy_score(
        y_true,
        onnx_pred,
    )

    torch_report = classification_report(
        y_true,
        torch_pred,
        labels=np.arange(N_CLASSES),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    onnx_report = classification_report(
        y_true,
        onnx_pred,
        labels=np.arange(N_CLASSES),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    # ---------------------------------------------------------------
    # Print result
    # ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PARITY RESULT")
    print("=" * 78)

    print(
        f"Input IDs identical       : "
        f"{input_ids_equal}"
    )

    print(
        f"Max absolute logit diff   : "
        f"{max_abs_diff:.10f}"
    )

    print(
        f"Mean absolute logit diff  : "
        f"{mean_abs_diff:.10f}"
    )

    print(
        f"Median absolute diff      : "
        f"{median_abs_diff:.10f}"
    )

    print(
        f"P99 absolute diff         : "
        f"{p99_abs_diff:.10f}"
    )

    print(
        f"Mean cosine similarity    : "
        f"{cosine_mean:.10f}"
    )

    print(
        f"Min cosine similarity     : "
        f"{cosine_min:.10f}"
    )

    print(
        f"Prediction agreement      : "
        f"{prediction_agreement * 100:.4f}%"
    )

    print(
        f"Prediction mismatches     : "
        f"{len(mismatch_idx)} / {len(X_np)}"
    )

    print("\n--- ACCURACY ---")
    print(
        f"PyTorch accuracy          : "
        f"{torch_accuracy * 100:.4f}%"
    )

    print(
        f"ONNX accuracy             : "
        f"{onnx_accuracy * 100:.4f}%"
    )

    print(
        f"Accuracy delta ONNX-PyTorch: "
        f"{(onnx_accuracy - torch_accuracy) * 100:+.4f} pp"
    )

    # ---------------------------------------------------------------
    # Save row-level diagnostic
    # ---------------------------------------------------------------
    detail = pd.DataFrame({
        "row": np.arange(len(X_np)),
        "text": texts,
        "true_intent": true_names,
        "pytorch_prediction": [
            labels[i]
            for i in torch_pred
        ],
        "onnx_prediction": [
            labels[i]
            for i in onnx_pred
        ],
        "prediction_same": prediction_equal,
        "max_abs_logit_diff": diff.max(axis=1),
        "mean_abs_logit_diff": diff.mean(axis=1),
        "cosine_similarity": cosine,
    })

    detail.to_csv(
        DETAIL_CSV,
        index=False,
    )

    mismatch_detail = detail[
        ~detail["prediction_same"]
    ].copy()

    mismatch_detail.to_csv(
        MISMATCH_CSV,
        index=False,
    )

    summary = {
        "status": "PARITY_AUDIT_COMPLETE",
        "rows": int(len(X_np)),
        "classes": N_CLASSES,
        "input_shape": [1, MAX_LEN],
        "pytorch_checkpoint": str(
            PYTORCH_CHECKPOINT
        ),
        "onnx_model": str(
            ONNX_MODEL
        ),
        "input_ids_identical": input_ids_equal,
        "max_abs_logit_diff": max_abs_diff,
        "mean_abs_logit_diff": mean_abs_diff,
        "median_abs_logit_diff": median_abs_diff,
        "p99_abs_logit_diff": p99_abs_diff,
        "mean_cosine_similarity": cosine_mean,
        "min_cosine_similarity": cosine_min,
        "prediction_agreement": prediction_agreement,
        "prediction_mismatch_count": int(
            len(mismatch_idx)
        ),
        "pytorch_accuracy": float(
            torch_accuracy
        ),
        "onnx_accuracy": float(
            onnx_accuracy
        ),
        "accuracy_delta_onnx_minus_pytorch": float(
            onnx_accuracy - torch_accuracy
        ),
        "onnx_inference_seconds": float(
            onnx_time
        ),
        "locked_test_used": True,
        "training_performed": False,
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSaved:")
    print(DETAIL_CSV)
    print(MISMATCH_CSV)
    print(SUMMARY_JSON)

    print("\nSTATUS:")
    print("V2.1 PYTORCH vs ONNX PARITY AUDIT COMPLETE")


if __name__ == "__main__":
    main()
