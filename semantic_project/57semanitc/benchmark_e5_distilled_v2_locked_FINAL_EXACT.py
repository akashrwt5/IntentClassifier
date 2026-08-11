#!/usr/bin/env python3

from pathlib import Path
import json
import time
import string

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


# ============================================================
# EXACT PROJECT PATHS
# ============================================================

PROJECT = Path(
    "/Users/shuklam/IntentClassifier/semantic_project/57semanitc"
)

# CANONICAL 57-INTENT LOCKED TEST SET
LOCKED_CSV = Path(
    "/Users/shuklam/IntentClassifier/semantic_project/57semanitc/"
    "v3_57intent_locked_eval/locked_test_57intent.csv"
)

# CORRECTED E5 DISTILLED V2 CHECKPOINT
CHECKPOINT = Path(
    "/Users/shuklam/IntentClassifier/semantic_project/57semanitc/"
    "v3_57intent_e5_distilled_v2/"
    "student_e5_distilled_v2_best_fp32.pt"
)

MODEL_DIR = CHECKPOINT.parent

# These are expected beside the distilled checkpoint.
VOCAB_JSON = MODEL_DIR / "vocab.json"
LABELS_JSON = MODEL_DIR / "label_map.json"

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
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    tokens = text.split()[:MAX_LEN]

    ids = [
        int(vocab.get(token, UNK_ID))
        for token in tokens
    ]

    ids += [PAD_ID] * (MAX_LEN - len(ids))

    return ids


# ============================================================
# MODEL
# ============================================================

class TinyIntentClassifier(nn.Module):

    def __init__(self, vocab_size, num_classes):
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

        self.norm = nn.LayerNorm(EMBED_DIM)

        self.classifier = nn.Linear(
            EMBED_DIM,
            num_classes,
        )

    def forward(self, input_ids):

        x = self.embedding(input_ids)

        padding_mask = input_ids.eq(PAD_ID)

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


# ============================================================
# LOADERS
# ============================================================

def load_vocab(path):

    if not path.exists():
        raise FileNotFoundError(
            f"\nVocabulary file not found:\n{path}\n\n"
            "The distilled checkpoint directory must contain "
            "the exact vocabulary used during training."
        )

    obj = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if (
        isinstance(obj, dict)
        and "vocab" in obj
    ):
        obj = obj["vocab"]

    return {
        str(k): int(v)
        for k, v in obj.items()
    }


def load_labels(path, checkpoint):

    if path.exists():

        obj = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(obj, dict):

            # {"0": "label", ...}
            if all(
                str(k).isdigit()
                for k in obj.keys()
            ):
                return [
                    obj[str(i)]
                    for i in range(len(obj))
                ]

            # {"label": 0, ...}
            if all(
                isinstance(v, (int, float))
                for v in obj.values()
            ):
                pairs = sorted(
                    obj.items(),
                    key=lambda kv: int(kv[1]),
                )

                return [
                    label
                    for label, _ in pairs
                ]

            for key in (
                "labels",
                "classes",
                "label_names",
            ):
                if key in obj:
                    return list(obj[key])

    if isinstance(checkpoint, dict):

        for key in (
            "labels",
            "classes",
            "label_names",
        ):
            if key in checkpoint:
                return list(checkpoint[key])

    raise FileNotFoundError(
        f"\nCould not load label map:\n{path}"
    )


def load_checkpoint(path):

    if not path.exists():
        raise FileNotFoundError(
            f"\nDistilled checkpoint not found:\n{path}\n\n"
            "Make sure the corrected E5 distilled V2 "
            "training has completed."
        )

    return torch.load(
        path,
        map_location="cpu",
    )


def get_state_dict(checkpoint):

    if isinstance(checkpoint, dict):

        for key in (
            "model_state_dict",
            "state_dict",
            "model",
        ):
            value = checkpoint.get(key)

            if isinstance(value, dict):
                return value

        # Raw state_dict
        if any(
            isinstance(v, torch.Tensor)
            for v in checkpoint.values()
        ):
            return checkpoint

    raise RuntimeError(
        "Could not find model state_dict in checkpoint."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "E5 DISTILLED V2 - EXACT LOCKED 57-INTENT TEST"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # PATH VALIDATION
    # --------------------------------------------------------

    print(
        f"Checkpoint : {CHECKPOINT}"
    )

    print(
        f"Locked CSV : {LOCKED_CSV}"
    )

    print()

    if not LOCKED_CSV.exists():
        raise FileNotFoundError(
            f"\nCANONICAL LOCKED TEST CSV NOT FOUND:\n"
            f"{LOCKED_CSV}"
        )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"\nDISTILLED CHECKPOINT NOT FOUND:\n"
            f"{CHECKPOINT}\n\n"
            "Run the corrected E5 distillation training first."
        )

    # --------------------------------------------------------
    # LOAD LOCKED DATA
    # --------------------------------------------------------

    df = pd.read_csv(
        LOCKED_CSV
    )

    required_columns = {
        "text",
        "intent",
    }

    if not required_columns.issubset(
        df.columns
    ):
        raise ValueError(
            "Locked CSV must contain "
            "'text' and 'intent' columns.\n"
            f"Found: {list(df.columns)}"
        )

    df = df.dropna(
        subset=[
            "text",
            "intent",
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"Locked rows    : {len(df)}"
    )

    print(
        f"Unique intents : "
        f"{df['intent'].nunique()}"
    )

    if len(df) != 1686:
        raise RuntimeError(
            f"Expected the canonical 1686-row locked "
            f"test set, but found {len(df)} rows."
        )

    if df["intent"].nunique() != 57:
        raise RuntimeError(
            "Expected 57 intents in the canonical "
            "locked test set, but found "
            f"{df['intent'].nunique()}."
        )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    checkpoint = load_checkpoint(
        CHECKPOINT
    )

    vocab = load_vocab(
        VOCAB_JSON
    )

    labels = load_labels(
        LABELS_JSON,
        checkpoint,
    )

    if len(labels) != 57:
        raise RuntimeError(
            f"Expected 57 model labels, "
            f"got {len(labels)}."
        )

    label_to_id = {
        label: i
        for i, label in enumerate(labels)
    }

    unknown_labels = sorted(
        set(df["intent"])
        - set(label_to_id)
    )

    if unknown_labels:
        raise RuntimeError(
            "Locked test contains labels absent "
            "from the model label map:\n"
            + "\n".join(
                unknown_labels
            )
        )

    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    model = TinyIntentClassifier(
        vocab_size=len(vocab),
        num_classes=len(labels),
    )

    state = get_state_dict(
        checkpoint
    )

    clean_state = {}

    for key, value in state.items():

        if key.startswith(
            "module."
        ):
            key = key[
                len("module.") :
            ]

        clean_state[key] = value

    missing, unexpected = (
        model.load_state_dict(
            clean_state,
            strict=False,
        )
    )

    if missing:
        raise RuntimeError(
            "Checkpoint/model mismatch.\n"
            "Missing keys:\n"
            + "\n".join(
                missing
            )
        )

    if unexpected:
        print(
            "WARNING: unexpected checkpoint keys:"
        )

        for key in unexpected:
            print(
                " ",
                key,
            )

    model.eval()

    # --------------------------------------------------------
    # TOKENIZE
    # --------------------------------------------------------

    X = np.asarray(
        [
            tokenize(
                text,
                vocab,
            )
            for text in df["text"]
        ],
        dtype=np.int64,
    )

    y_true = np.asarray(
        [
            label_to_id[intent]
            for intent in df["intent"]
        ],
        dtype=np.int64,
    )

    input_ids = torch.from_numpy(
        X
    )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    print()
    print(
        "Running exact locked-test inference..."
    )

    outputs = []

    start_time = (
        time.perf_counter()
    )

    with torch.no_grad():

        for start in range(
            0,
            len(input_ids),
            256,
        ):

            batch = input_ids[
                start:start + 256
            ]

            logits = model(
                batch
            )

            outputs.append(
                logits.cpu().numpy()
            )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    logits = np.concatenate(
        outputs,
        axis=0,
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    y_pred = np.argmax(
        logits,
        axis=1,
    )

    shifted = (
        logits
        - logits.max(
            axis=1,
            keepdims=True,
        )
    )

    probabilities = np.exp(
        shifted
    )

    probabilities /= (
        probabilities.sum(
            axis=1,
            keepdims=True,
        )
    )

    confidence = (
        probabilities.max(
            axis=1
        )
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

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
        labels=list(
            range(57)
        ),
        target_names=labels,
        digits=4,
        zero_division=0,
    )

    # --------------------------------------------------------
    # PRINT RESULT
    # --------------------------------------------------------

    print()
    print(
        "--- E5 DISTILLED V2 LOCKED TEST RESULT ---"
    )

    print(
        f"Accuracy   : "
        f"{accuracy * 100:.4f}%"
    )

    print(
        f"Macro F1   : "
        f"{macro_f1 * 100:.4f}%"
    )

    print(
        f"Weighted F1: "
        f"{weighted_f1 * 100:.4f}%"
    )

    print()
    print(
        "Classification report:"
    )

    print(
        report
    )

    # --------------------------------------------------------
    # SPEED
    # --------------------------------------------------------

    rows_per_second = (
        len(df)
        / elapsed
    )

    ms_per_row = (
        elapsed
        / len(df)
        * 1000
    )

    print(
        "--- INFERENCE SPEED ---"
    )

    print(
        f"Total rows : {len(df)}"
    )

    print(
        f"Total time : "
        f"{elapsed:.4f} sec"
    )

    print(
        f"Rows/sec   : "
        f"{rows_per_second:.2f}"
    )

    print(
        f"ms/row     : "
        f"{ms_per_row:.4f}"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_dir = (
        PROJECT
        / "v3_57intent_e5_distilled_v2_locked_benchmark"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_df = (
        df.copy()
    )

    prediction_df[
        "true_id"
    ] = y_true

    prediction_df[
        "predicted_id"
    ] = y_pred

    prediction_df[
        "prediction"
    ] = [
        labels[i]
        for i in y_pred
    ]

    prediction_df[
        "confidence"
    ] = confidence

    predictions_path = (
        output_dir
        / "locked_predictions_e5_distilled_v2.csv"
    )

    prediction_df.to_csv(
        predictions_path,
        index=False,
    )

    report_path = (
        output_dir
        / "classification_report_e5_distilled_v2.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    confusion = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(57)
        ),
    )

    confusion_path = (
        output_dir
        / "confusion_matrix_e5_distilled_v2.csv"
    )

    pd.DataFrame(
        confusion,
        index=labels,
        columns=labels,
    ).to_csv(
        confusion_path
    )

    summary = {
        "model": str(
            CHECKPOINT
        ),
        "locked_test": str(
            LOCKED_CSV
        ),
        "rows": int(
            len(df)
        ),
        "num_intents": 57,
        "accuracy": float(
            accuracy
        ),
        "macro_f1": float(
            macro_f1
        ),
        "weighted_f1": float(
            weighted_f1
        ),
        "inference_seconds": float(
            elapsed
        ),
        "rows_per_second": float(
            rows_per_second
        ),
        "ms_per_row": float(
            ms_per_row
        ),
        "quantization": False,
        "onnx": False,
        "training": False,
        "locked_test_modified": False,
    }

    summary_path = (
        output_dir
        / "benchmark_summary_e5_distilled_v2.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Saved:"
    )

    print(
        predictions_path
    )

    print(
        report_path
    )

    print(
        confusion_path
    )

    print(
        summary_path
    )

    print()
    print(
        "STATUS: "
        "E5 DISTILLED V2 LOCKED "
        "57-INTENT BENCHMARK COMPLETE"
    )


if __name__ == "__main__":
    main()
