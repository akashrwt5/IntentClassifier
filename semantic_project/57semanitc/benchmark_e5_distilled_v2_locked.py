#!/usr/bin/env python3
"""
V2 DISTILLED STUDENT - LOCKED 57-INTENT TEST

IMPORTANT:
- Reads ONLY the locked test CSV for evaluation.
- Does NOT train.
- Does NOT modify the model.
- Does NOT quantize.
- Uses the distilled Tiny checkpoint.
- Uses the SAME tokenizer/vocab contract as the distilled training model.

Before running, update LOCKED_CSV if your locked-test CSV has a different path.
"""

from pathlib import Path
import json
import time
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ============================================================
# CONFIG
# ============================================================

PROJECT = Path(
    "/Users/shuklam/IntentClassifier/semantic_project/57semanitc"
)

DISTILLED_DIR = PROJECT / "v3_57intent_e5_distilled_v2"

CHECKPOINT = DISTILLED_DIR / "student_e5_distilled_v2_best_fp32.pt"

# IMPORTANT:
# Change this if your locked test CSV has another filename/location.
LOCKED_CSV = PROJECT / "locked_test.csv"

# If your existing project already has a locked-test CSV, you can
# replace the line above with its exact path.

VOCAB_JSON = DISTILLED_DIR / "vocab.json"
LABELS_JSON = DISTILLED_DIR / "label_map.json"

MAX_LEN = 24
PAD_ID = 0
UNK_ID = 1

EMBED_DIM = 64
NHEAD = 4
FF_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.10


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text, vocab):
    text = str(text).lower().strip()

    # Same simple whitespace/punctuation-normalization style
    # used by the Tiny student contract.
    import string
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    tokens = text.split()

    ids = [
        int(vocab.get(tok, UNK_ID))
        for tok in tokens[:MAX_LEN]
    ]

    if len(ids) < MAX_LEN:
        ids += [PAD_ID] * (MAX_LEN - len(ids))

    return ids


# ============================================================
# MODEL
# ============================================================

class TinyIntentClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes,
        pad_id=PAD_ID,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            EMBED_DIM,
            padding_idx=pad_id,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NHEAD,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=NUM_LAYERS,
        )

        self.norm = nn.LayerNorm(EMBED_DIM)
        self.classifier = nn.Linear(
            EMBED_DIM,
            num_classes,
        )

    def forward(self, input_ids):
        x = self.embedding(input_ids)

        mask = input_ids.eq(PAD_ID)

        x = self.encoder(
            x,
            src_key_padding_mask=mask,
        )

        valid = (~mask).unsqueeze(-1).float()

        denom = valid.sum(
            dim=1
        ).clamp_min(1.0)

        x = (x * valid).sum(dim=1) / denom

        x = self.norm(x)

        return self.classifier(x)


# ============================================================
# LOAD JSON HELPERS
# ============================================================

def load_vocab(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Vocabulary not found:\n{path}\n\n"
            "Update VOCAB_JSON to the vocab used during "
            "distilled training."
        )

    obj = json.loads(
        path.read_text(encoding="utf-8")
    )

    # Accept either:
    # {"word": id}
    # or {"vocab": {"word": id}}
    if isinstance(obj, dict) and "vocab" in obj:
        obj = obj["vocab"]

    return {
        str(k): int(v)
        for k, v in obj.items()
    }


def load_labels(path, checkpoint=None):
    if path.exists():
        obj = json.loads(
            path.read_text(encoding="utf-8")
        )

        if isinstance(obj, dict):
            # Common forms:
            # {"0": "label", ...}
            # {"label": 0, ...}
            if all(str(k).isdigit() for k in obj.keys()):
                return [
                    obj[str(i)]
                    for i in range(len(obj))
                ]

            if all(
                isinstance(v, (int, float))
                for v in obj.values()
            ):
                inv = sorted(
                    obj.items(),
                    key=lambda kv: int(kv[1]),
                )
                return [k for k, _ in inv]

            for key in (
                "labels",
                "classes",
                "label_names",
            ):
                if key in obj:
                    return list(obj[key])

    # Fallback: checkpoint may contain labels.
    if checkpoint is not None:
        for key in (
            "labels",
            "label_names",
            "classes",
        ):
            if key in checkpoint:
                return list(checkpoint[key])

    raise FileNotFoundError(
        f"Could not load label map from:\n{path}"
    )


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{path}\n\n"
            "Update CHECKPOINT to the distilled V2 checkpoint."
        )

    ckpt = torch.load(
        path,
        map_location="cpu",
    )

    return ckpt


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in (
            "model_state_dict",
            "state_dict",
            "model",
        ):
            value = ckpt.get(key)
            if isinstance(value, dict):
                return value

    if isinstance(ckpt, dict):
        # Raw state_dict.
        if any(
            isinstance(v, torch.Tensor)
            for v in ckpt.values()
        ):
            return ckpt

    raise RuntimeError(
        "Could not find model state_dict in checkpoint."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("V2 DISTILLED STUDENT - LOCKED 57-INTENT TEST")
    print("=" * 72)

    if not LOCKED_CSV.exists():
        raise FileNotFoundError(
            f"""
Locked test CSV not found:

{LOCKED_CSV}

Find your existing locked-test CSV and update:

LOCKED_CSV = Path("...")

Do NOT use train.csv or validation data here.
"""
        )

    print(f"Checkpoint : {CHECKPOINT}")
    print(f"Locked CSV : {LOCKED_CSV}")
    print()

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(LOCKED_CSV)

    required = {"text", "intent"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns {required}. "
            f"Found: {list(df.columns)}"
        )

    df = df.dropna(
        subset=["text", "intent"]
    ).reset_index(drop=True)

    print(f"Locked rows : {len(df)}")
    print(
        f"Unique intents : {df['intent'].nunique()}"
    )

    # --------------------------------------------------------
    # LOAD CHECKPOINT / VOCAB / LABELS
    # --------------------------------------------------------

    ckpt = load_checkpoint(CHECKPOINT)

    vocab_path = VOCAB_JSON
    labels_path = LABELS_JSON

    # Try checkpoint-provided paths if local defaults don't exist.
    if not vocab_path.exists() and isinstance(ckpt, dict):
        vp = ckpt.get("vocab_path")
        if vp:
            vocab_path = Path(vp)

    if not labels_path.exists() and isinstance(ckpt, dict):
        lp = ckpt.get("label_map_path")
        if lp:
            labels_path = Path(lp)

    vocab = load_vocab(vocab_path)
    labels = load_labels(
        labels_path,
        ckpt,
    )

    if len(labels) != 57:
        raise RuntimeError(
            f"Expected 57 labels, got {len(labels)}"
        )

    label_to_id = {
        label: i
        for i, label in enumerate(labels)
    }

    missing_labels = sorted(
        set(df["intent"]) - set(label_to_id)
    )

    if missing_labels:
        raise RuntimeError(
            "Locked test contains labels not present in "
            f"the model label map:\n{missing_labels}"
        )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    state = extract_state_dict(ckpt)

    # Infer vocab/classes from checkpoint where possible.
    vocab_size = len(vocab)
    num_classes = len(labels)

    model = TinyIntentClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
    )

    # Handle checkpoints saved from DataParallel/prefixes.
    clean_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        clean_state[k] = v

    missing, unexpected = model.load_state_dict(
        clean_state,
        strict=False,
    )

    if missing:
        raise RuntimeError(
            "Checkpoint/model mismatch. Missing keys:\n"
            + "\n".join(missing)
        )

    if unexpected:
        print(
            "WARNING: unexpected checkpoint keys:",
            unexpected,
        )

    model.eval()

    # --------------------------------------------------------
    # TOKENIZE
    # --------------------------------------------------------

    X = np.asarray(
        [
            tokenize(text, vocab)
            for text in df["text"].tolist()
        ],
        dtype=np.int64,
    )

    y_true = np.asarray(
        [
            label_to_id[x]
            for x in df["intent"].tolist()
        ],
        dtype=np.int64,
    )

    input_ids = torch.from_numpy(X)

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    print()
    print("Running locked-test inference...")

    all_logits = []

    start = time.perf_counter()

    with torch.no_grad():
        for start_i in range(
            0,
            len(input_ids),
            256,
        ):
            batch = input_ids[
                start_i:start_i + 256
            ]

            logits = model(batch)

            all_logits.append(
                logits.cpu().numpy()
            )

    elapsed = (
        time.perf_counter()
        - start
    )

    logits = np.concatenate(
        all_logits,
        axis=0,
    )

    y_pred = np.argmax(
        logits,
        axis=1,
    )

    # Softmax confidence.
    logits_shift = (
        logits
        - logits.max(
            axis=1,
            keepdims=True,
        )
    )

    probs = np.exp(logits_shift)
    probs /= probs.sum(
        axis=1,
        keepdims=True,
    )

    confidence = probs.max(
        axis=1
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    acc = accuracy_score(
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
        labels=list(range(len(labels))),
        target_names=labels,
        zero_division=0,
        digits=4,
    )

    print()
    print("=" * 72)
    print("--- V2 DISTILLED LOCKED TEST RESULT ---")
    print("=" * 72)
    print(
        f"Accuracy   : {acc * 100:.4f}%"
    )
    print(
        f"Macro F1   : {macro_f1 * 100:.4f}%"
    )
    print(
        f"Weighted F1: {weighted_f1 * 100:.4f}%"
    )
    print()
    print(report)

    print("--- INFERENCE SPEED ---")
    print(
        f"Total rows : {len(df)}"
    )
    print(
        f"Total time : {elapsed:.4f} sec"
    )
    print(
        f"Rows/sec   : {len(df) / elapsed:.2f}"
    )
    print(
        f"ms/row     : {elapsed / len(df) * 1000:.4f}"
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    out_dir = (
        PROJECT
        / "v3_57intent_e5_distilled_v2_locked_benchmark"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pred_df = df.copy()

    pred_df["true_id"] = y_true
    pred_df["predicted_id"] = y_pred
    pred_df["prediction"] = [
        labels[i]
        for i in y_pred
    ]
    pred_df["confidence"] = confidence

    pred_df.to_csv(
        out_dir / "locked_predictions_v2.csv",
        index=False,
    )

    report_path = (
        out_dir
        / "classification_report_v2.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    )

    cm_df.to_csv(
        out_dir / "confusion_matrix_v2.csv"
    )

    summary = {
        "model": str(CHECKPOINT),
        "locked_csv": str(LOCKED_CSV),
        "rows": int(len(df)),
        "num_intents": int(len(labels)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "inference_seconds": float(elapsed),
        "rows_per_second": float(
            len(df) / elapsed
        ),
        "ms_per_row": float(
            elapsed / len(df) * 1000
        ),
        "quantization": False,
        "onnx": False,
        "training_performed": False,
    }

    (
        out_dir / "benchmark_summary_v2.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Saved:")
    print(
        out_dir
        / "locked_predictions_v2.csv"
    )
    print(
        out_dir
        / "classification_report_v2.txt"
    )
    print(
        out_dir
        / "confusion_matrix_v2.csv"
    )
    print(
        out_dir
        / "benchmark_summary_v2.json"
    )

    print()
    print(
        "STATUS: V2 DISTILLED LOCKED 57-INTENT "
        "BENCHMARK COMPLETE"
    )


if __name__ == "__main__":
    main()
