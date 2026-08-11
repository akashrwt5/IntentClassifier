#!/usr/bin/env python3
"""
BENCHMARK: V3 FP32 vs V3+E5 HYBRID FP32
========================================

Purpose:
    Compare the original locked V3 model against the new
    V3+E5 hybrid checkpoint on the untouched 595-row unseen set.

IMPORTANT:
    - No training
    - No threshold fitting
    - No modification of V3
    - No modification of Hybrid
    - 595-row unseen set is evaluation-only
    - No ONNX export
    - No INT8 quantization

V3:
    tiny_semantic_student_v3_fp32/v3_semantic_student_fp32.onnx

Hybrid:
    v3_e5_hybrid/v3_e5_hybrid_fp32.pt

The V3 ONNX input is fixed [1, 24], so V3 inference is
performed one row at a time.

Run:
    python3 benchmark_v3_e5_hybrid.py
"""

from pathlib import Path
import copy
import json
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/Users/shuklam/IntentClassifier/semantic_project"
)

V3_ONNX = (
    ROOT
    / "tiny_semantic_student_v3_fp32"
    / "v3_semantic_student_fp32.onnx"
)

HYBRID_CHECKPOINT = (
    ROOT
    / "v3_e5_hybrid"
    / "v3_e5_hybrid_fp32.pt"
)

UNSEEN_CSV = (
    ROOT
    / "unseen_semantic_stress_test.csv"
)

OOD_CSV = (
    ROOT
    / "production_calibration_v2"
    / "production_ood_calibration.csv"
)

OUTPUT_DIR = (
    ROOT
    / "v3_vs_e5_hybrid_benchmark"
)


# ============================================================
# LOCKED V3 ARCHITECTURE
# ============================================================

VOCAB_SIZE = 895
EMBED_DIM = 64
LAYERS = 2
HEADS = 4
FFN = 128
MAX_LEN = 24
NUM_CLASSES = 11


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


# Critical commands are evaluated separately.
CRITICAL_PHRASES = [
    "make it louder",
    "make it quieter",
    "mute it",
    "unmute it",
    "turn off",
    "turn the sound back on",
    "completely silent",
    "keep it on",
]


# ============================================================
# HELPERS
# ============================================================

def normalize_text(x):
    return re.sub(
        r"\s+",
        " ",
        str(x).strip(),
    )


def find_column(df, candidates):
    lower = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]

    return None


def load_vocab():
    candidates = [
        ROOT
        / "tiny_semantic_student_v2_balanced"
        / "vocab.json",

        ROOT
        / "tiny_semantic_student_v3_balanced"
        / "vocab.json",

        ROOT
        / "tiny_semantic_student_v3_fp32"
        / "vocab.json",
    ]

    for path in candidates:
        if not path.exists():
            continue

        obj = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(obj, dict) and "vocab" in obj:
            obj = obj["vocab"]

        if not isinstance(obj, dict):
            continue

        if len(obj) != VOCAB_SIZE:
            continue

        print("\nVocabulary:")
        print(path)
        print("Vocabulary size:", len(obj))

        return obj

    raise FileNotFoundError(
        "895-token vocab.json not found."
    )


def get_token_id(
    vocab,
    names,
    default,
):
    for name in names:
        if name in vocab:
            return int(vocab[name])

    return default


def tokenize(
    text,
    vocab,
):
    pad_id = get_token_id(
        vocab,
        ["<pad>", "[PAD]"],
        0,
    )

    unk_id = get_token_id(
        vocab,
        ["<unk>", "[UNK]"],
        1,
    )

    cls_id = None
    sep_id = None

    for name in [
        "<cls>",
        "[CLS]",
    ]:
        if name in vocab:
            cls_id = int(vocab[name])
            break

    for name in [
        "<sep>",
        "[SEP]",
    ]:
        if name in vocab:
            sep_id = int(vocab[name])
            break

    ids = []

    if cls_id is not None:
        ids.append(cls_id)

    for token in normalize_text(text).lower().split():
        ids.append(
            int(vocab.get(token, unk_id))
        )

    if sep_id is not None:
        ids.append(sep_id)

    ids = ids[:MAX_LEN]

    if len(ids) < MAX_LEN:
        ids.extend(
            [pad_id] * (
                MAX_LEN - len(ids)
            )
        )

    return np.asarray(
        ids,
        dtype=np.int64,
    )


def extract_state(checkpoint):
    if isinstance(checkpoint, dict):
        return checkpoint.get(
            "state_dict",
            checkpoint.get(
                "model_state_dict",
                checkpoint,
            ),
        )

    return checkpoint


# ============================================================
# EXACT V3 / HYBRID ARCHITECTURE
# ============================================================

class V3Student(nn.Module):

    def __init__(
        self,
        pad_id,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            EMBED_DIM,
            padding_idx=pad_id,
        )

        self.position = nn.Embedding(
            MAX_LEN,
            EMBED_DIM,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=HEADS,
            dim_feedforward=FFN,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=LAYERS,
        )

        self.norm = nn.LayerNorm(
            EMBED_DIM
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                EMBED_DIM,
                EMBED_DIM,
            ),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(
                EMBED_DIM,
                NUM_CLASSES,
            ),
        )

    def encode(
        self,
        input_ids,
        attention_mask,
    ):
        seq_len = input_ids.shape[1]

        positions = torch.arange(
            seq_len,
            device=input_ids.device,
        ).unsqueeze(0)

        x = (
            self.embedding(input_ids)
            + self.position(positions)
        )

        padding_mask = (
            attention_mask == 0
        )

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        mask = (
            attention_mask
            .unsqueeze(-1)
            .float()
        )

        x = (
            (x * mask).sum(1)
            /
            mask.sum(1).clamp(
                min=1e-8
            )
        )

        return self.norm(x)

    def forward(
        self,
        input_ids,
        attention_mask,
    ):
        emb = self.encode(
            input_ids,
            attention_mask,
        )

        logits = self.classifier(
            emb
        )

        return emb, logits


# ============================================================
# LOAD V3 ONNX
# ============================================================

def load_v3():
    if not V3_ONNX.exists():
        raise FileNotFoundError(
            f"V3 ONNX not found:\n{V3_ONNX}"
        )

    session = ort.InferenceSession(
        str(V3_ONNX),
        providers=[
            "CPUExecutionProvider"
        ],
    )

    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]

    print("\nV3 ONNX:")
    print(V3_ONNX)
    print(
        "V3 input :",
        inp.name,
        inp.shape,
    )
    print(
        "V3 output:",
        out.name,
        out.shape,
    )

    if list(inp.shape) != [
        1,
        MAX_LEN,
    ]:
        raise RuntimeError(
            f"V3 ONNX must have fixed input "
            f"[1,{MAX_LEN}], got {inp.shape}"
        )

    if list(out.shape) != [
        1,
        NUM_CLASSES,
    ]:
        raise RuntimeError(
            f"V3 ONNX must output "
            f"[1,{NUM_CLASSES}], got {out.shape}"
        )

    return (
        session,
        inp.name,
        out.name,
    )


# ============================================================
# LOAD HYBRID
# ============================================================

def load_hybrid(
    vocab,
    device,
):
    if not HYBRID_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Hybrid checkpoint not found:\n"
            f"{HYBRID_CHECKPOINT}"
        )

    pad_id = get_token_id(
        vocab,
        ["<pad>", "[PAD]"],
        0,
    )

    checkpoint = torch.load(
        HYBRID_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    state = extract_state(
        checkpoint
    )

    model = V3Student(
        pad_id
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.to(device)
    model.eval()

    print("\nHybrid checkpoint: PASS")
    print(HYBRID_CHECKPOINT)
    print(
        "Hybrid runtime architecture:",
        "same V3 64 -> 64 -> 11",
    )

    return model


# ============================================================
# LOAD LOCKED UNSEEN
# ============================================================

def load_unseen():
    if not UNSEEN_CSV.exists():
        raise FileNotFoundError(
            f"Locked unseen file not found:\n"
            f"{UNSEEN_CSV}"
        )

    df = pd.read_csv(
        UNSEEN_CSV
    )

    text_col = find_column(
        df,
        [
            "text",
            "utterance",
            "query",
            "sentence",
        ],
    )

    intent_col = find_column(
        df,
        [
            "intent",
            "label",
            "target",
            "expected_intent",
        ],
    )

    if text_col is None or intent_col is None:
        raise RuntimeError(
            "595-row unseen CSV must contain "
            "text + intent columns."
        )

    out = pd.DataFrame({
        "text": df[text_col].map(
            normalize_text
        ),
        "intent": df[intent_col]
        .astype(str)
        .str.strip(),
    })

    out = out[
        out["intent"].isin(LABELS)
    ].reset_index(
        drop=True
    )

    if len(out) != 595:
        raise RuntimeError(
            f"Expected exactly 595 locked unseen "
            f"rows, got {len(out)}."
        )

    print(
        "\nLOCKED UNSEEN ROWS:",
        len(out),
    )

    return out


# ============================================================
# PREDICT V3
# ============================================================

def predict_v3(
    session,
    input_name,
    output_name,
    ids,
):
    """
    V3 ONNX has fixed [1,24] input.
    Therefore every row is sent independently.
    """

    logits_list = []

    total = len(ids)

    for i in range(total):

        single_ids = np.asarray(
            ids[i:i + 1],
            dtype=np.int64,
        )

        if single_ids.shape != (
            1,
            MAX_LEN,
        ):
            raise RuntimeError(
                f"V3 input shape error: "
                f"{single_ids.shape}"
            )

        logits = session.run(
            [output_name],
            {
                input_name:
                single_ids,
            },
        )[0]

        logits = np.asarray(
            logits,
            dtype=np.float32,
        )

        if logits.shape != (
            1,
            NUM_CLASSES,
        ):
            raise RuntimeError(
                f"V3 output shape error: "
                f"{logits.shape}"
            )

        logits_list.append(
            logits[0]
        )

        if (
            (i + 1) % 100 == 0
            or i + 1 == total
        ):
            print(
                f"V3 progress: "
                f"{i + 1}/{total}",
                end="\r",
            )

    print()

    logits = np.stack(
        logits_list,
        axis=0,
    )

    shifted = (
        logits
        -
        logits.max(
            axis=1,
            keepdims=True,
        )
    )

    exp = np.exp(
        shifted
    )

    probs = (
        exp
        /
        exp.sum(
            axis=1,
            keepdims=True,
        )
    )

    pred = probs.argmax(
        axis=1
    )

    confidence = probs.max(
        axis=1
    )

    return (
        pred,
        confidence,
        probs,
    )


# ============================================================
# PREDICT HYBRID
# ============================================================

@torch.no_grad()
def predict_hybrid(
    model,
    ids,
    masks,
    device,
    batch_size=128,
):
    predictions = []
    confidences = []
    probabilities = []

    for start in range(
        0,
        len(ids),
        batch_size,
    ):
        end = min(
            start + batch_size,
            len(ids),
        )

        ids_b = ids[
            start:end
        ].to(device)

        masks_b = masks[
            start:end
        ].to(device)

        _, logits = model(
            ids_b,
            masks_b,
        )

        probs = torch.softmax(
            logits,
            dim=-1,
        )

        conf, pred = probs.max(
            dim=-1
        )

        predictions.append(
            pred.cpu().numpy()
        )

        confidences.append(
            conf.cpu().numpy()
        )

        probabilities.append(
            probs.cpu().numpy()
        )

    return (
        np.concatenate(predictions),
        np.concatenate(confidences),
        np.concatenate(probabilities),
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    truth,
    pred,
):
    return {
        "accuracy":
            accuracy_score(
                truth,
                pred,
            ),
        "macro_f1":
            f1_score(
                truth,
                pred,
                average="macro",
                zero_division=0,
            ),
    }


def print_report(
    name,
    truth,
    pred,
):
    m = calculate_metrics(
        truth,
        pred,
    )

    print(
        f"\n--- {name} ---"
    )

    print(
        f"Accuracy : "
        f"{m['accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro F1 : "
        f"{m['macro_f1'] * 100:.2f}%"
    )

    print(
        classification_report(
            truth,
            pred,
            labels=list(
                range(NUM_CLASSES)
            ),
            target_names=LABELS,
            digits=4,
            zero_division=0,
        )
    )

    return m


def critical_mask(df):
    return np.asarray(
        [
            any(
                phrase
                in normalize_text(text).lower()
                for phrase in CRITICAL_PHRASES
            )
            for text in df["text"]
        ],
        dtype=bool,
    )


# ============================================================
# OOD
# ============================================================

def load_ood():
    if not OOD_CSV.exists():
        return None

    df = pd.read_csv(
        OOD_CSV
    )

    text_col = find_column(
        df,
        [
            "text",
            "utterance",
            "query",
            "sentence",
        ],
    )

    if text_col is None:
        return None

    out = pd.DataFrame({
        "text": df[text_col].map(
            normalize_text
        )
    })

    return out


def rejection_rate(
    probs,
    threshold,
):
    confidence = probs.max(
        axis=1
    )

    return float(
        (confidence < threshold).mean()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "V3 FP32 vs V3+E5 HYBRID FP32"
    )
    print("=" * 78)

    print("\nSAFETY:")
    print("Training             : NO")
    print("Threshold fitting    : NO")
    print("V3 modification      : NO")
    print("Hybrid modification  : NO")
    print("595 unseen           : EVALUATION ONLY")
    print("ONNX export          : NO")
    print("INT8                  : NO")

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    print(
        "\nDevice:",
        device,
    )

    vocab = load_vocab()

    unseen = load_unseen()

    label_to_id = {
        label: i
        for i, label in enumerate(
            LABELS
        )
    }

    truth = np.asarray(
        [
            label_to_id[x]
            for x in unseen["intent"]
        ],
        dtype=np.int64,
    )

    ids_np = np.stack(
        [
            tokenize(
                text,
                vocab,
            )
            for text in unseen["text"]
        ]
    )

    pad_id = get_token_id(
        vocab,
        ["<pad>", "[PAD]"],
        0,
    )

    masks_np = (
        ids_np != pad_id
    ).astype(np.int64)

    ids_t = torch.tensor(
        ids_np,
        dtype=torch.long,
    )

    masks_t = torch.tensor(
        masks_np,
        dtype=torch.long,
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    v3, v3_input, v3_output = load_v3()

    hybrid = load_hybrid(
        vocab,
        device,
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    print(
        "\nRunning Hybrid on locked unseen..."
    )

    (
        hybrid_pred,
        hybrid_conf,
        hybrid_probs,
    ) = predict_hybrid(
        hybrid,
        ids_t,
        masks_t,
        device,
    )

    print(
        "Running V3 on locked unseen..."
    )

    (
        v3_pred,
        v3_conf,
        v3_probs,
    ) = predict_v3(
        v3,
        v3_input,
        v3_output,
        ids_np,
    )

    # --------------------------------------------------------
    # Main benchmark
    # --------------------------------------------------------

    v3_metrics = print_report(
        "UNSEEN_595 — V3",
        truth,
        v3_pred,
    )

    hybrid_metrics = print_report(
        "UNSEEN_595 — V3+E5 HYBRID",
        truth,
        hybrid_pred,
    )

    delta_accuracy = (
        hybrid_metrics["accuracy"]
        -
        v3_metrics["accuracy"]
    )

    delta_f1 = (
        hybrid_metrics["macro_f1"]
        -
        v3_metrics["macro_f1"]
    )

    print(
        "\n--- DELTA HYBRID vs V3 ---"
    )

    print(
        f"Accuracy delta : "
        f"{delta_accuracy * 100:+.2f} pp"
    )

    print(
        f"Macro F1 delta : "
        f"{delta_f1 * 100:+.2f} pp"
    )

    # --------------------------------------------------------
    # Critical
    # --------------------------------------------------------

    critical = critical_mask(
        unseen
    )

    print(
        "\n--- CRITICAL ---"
    )

    if critical.any():

        v3_critical = accuracy_score(
            truth[critical],
            v3_pred[critical],
        )

        hybrid_critical = accuracy_score(
            truth[critical],
            hybrid_pred[critical],
        )

        print(
            "Rows       :",
            int(critical.sum()),
        )

        print(
            f"V3 accuracy: "
            f"{v3_critical * 100:.2f}%"
        )

        print(
            f"Hybrid     : "
            f"{hybrid_critical * 100:.2f}%"
        )

        print(
            f"Delta      : "
            f"{(hybrid_critical-v3_critical)*100:+.2f} pp"
        )

    else:
        v3_critical = None
        hybrid_critical = None

        print(
            "No critical rows found."
        )

    # --------------------------------------------------------
    # Regression / improvement
    # --------------------------------------------------------

    v3_correct = (
        v3_pred == truth
    )

    hybrid_correct = (
        hybrid_pred == truth
    )

    regressions = (
        v3_correct
        &
        (~hybrid_correct)
    )

    improvements = (
        (~v3_correct)
        &
        hybrid_correct
    )

    print(
        "\n--- REGRESSIONS ---"
    )

    print(
        "V3 correct -> Hybrid wrong:",
        int(regressions.sum()),
    )

    print(
        "V3 wrong -> Hybrid correct:",
        int(improvements.sum()),
    )

    # --------------------------------------------------------
    # Per intent
    # --------------------------------------------------------

    rows = []

    for idx, label in enumerate(
        LABELS
    ):

        mask = truth == idx

        v3_acc = (
            (v3_pred[mask] == idx).mean()
            if mask.any()
            else 0.0
        )

        hybrid_acc = (
            (hybrid_pred[mask] == idx).mean()
            if mask.any()
            else 0.0
        )

        rows.append({
            "intent": label,
            "support": int(mask.sum()),
            "v3_accuracy": v3_acc,
            "hybrid_accuracy": hybrid_acc,
            "delta_pp":
                (
                    hybrid_acc
                    - v3_acc
                ) * 100,
        })

    per_intent = pd.DataFrame(
        rows
    )

    print(
        "\n--- PER INTENT DELTA ---"
    )

    print(
        per_intent.to_string(
            index=False,
            formatters={
                "v3_accuracy":
                    lambda x:
                    f"{x*100:.2f}%",
                "hybrid_accuracy":
                    lambda x:
                    f"{x*100:.2f}%",
                "delta_pp":
                    lambda x:
                    f"{x:+.2f}",
            },
        )
    )

    # --------------------------------------------------------
    # Regression details
    # --------------------------------------------------------

    regression_rows = []

    for i in np.where(
        regressions
    )[0]:

        regression_rows.append({
            "row": int(i),
            "text":
                unseen.iloc[i]["text"],
            "true_intent":
                LABELS[truth[i]],
            "v3_prediction":
                LABELS[v3_pred[i]],
            "hybrid_prediction":
                LABELS[hybrid_pred[i]],
            "v3_confidence":
                float(v3_conf[i]),
            "hybrid_confidence":
                float(hybrid_conf[i]),
            "critical":
                bool(critical[i]),
        })

    regression_df = pd.DataFrame(
        regression_rows
    )

    # --------------------------------------------------------
    # OOD
    # --------------------------------------------------------

    ood = load_ood()

    ood_results = {}

    if ood is not None and len(ood):

        ood_ids_np = np.stack(
            [
                tokenize(
                    text,
                    vocab,
                )
                for text in ood["text"]
            ]
        )

        ood_masks_np = (
            ood_ids_np != pad_id
        ).astype(np.int64)

        ood_ids = torch.tensor(
            ood_ids_np,
            dtype=torch.long,
        )

        ood_masks = torch.tensor(
            ood_masks_np,
            dtype=torch.long,
        )

        print(
            "\nOOD calibration rows:",
            len(ood),
        )

        print(
            "Running Hybrid on OOD..."
        )

        (
            _,
            _,
            hybrid_ood_probs,
        ) = predict_hybrid(
            hybrid,
            ood_ids,
            ood_masks,
            device,
        )

        print(
            "Running V3 on OOD..."
        )

        (
            _,
            _,
            v3_ood_probs,
        ) = predict_v3(
            v3,
            v3_input,
            v3_output,
            ood_ids_np,
        )

        print(
            "\n--- OOD REJECTION ---"
        )

        for threshold in [
            0.50,
            0.60,
            0.70,
            0.80,
            0.87,
            0.90,
            0.95,
            0.97,
        ]:

            v3_reject = rejection_rate(
                v3_ood_probs,
                threshold,
            )

            hybrid_reject = rejection_rate(
                hybrid_ood_probs,
                threshold,
            )

            print(
                f"threshold {threshold:.2f} | "
                f"V3 reject "
                f"{v3_reject*100:6.2f}% | "
                f"Hybrid reject "
                f"{hybrid_reject*100:6.2f}%"
            )

            ood_results[
                f"{threshold:.2f}"
            ] = {
                "v3_rejection":
                    v3_reject,
                "hybrid_rejection":
                    hybrid_reject,
            }

    else:
        print(
            "\nOOD calibration file not found; skipped."
        )

    # --------------------------------------------------------
    # QUALITY GATES
    # --------------------------------------------------------

    gate_accuracy = (
        hybrid_metrics["accuracy"]
        >=
        v3_metrics["accuracy"]
    )

    gate_f1 = (
        hybrid_metrics["macro_f1"]
        >=
        v3_metrics["macro_f1"]
    )

    critical_regression = False

    if critical.any():
        critical_regression = (
            hybrid_critical
            <
            v3_critical
        )

    gate_critical = (
        not critical_regression
    )

    candidate = (
        gate_accuracy
        and gate_f1
        and gate_critical
    )

    print(
        "\n" + "=" * 78
    )
    print("QUALITY GATES")
    print("=" * 78)

    print(
        "Unseen accuracy >= V3 :",
        "PASS"
        if gate_accuracy
        else "FAIL",
    )

    print(
        "Macro F1 >= V3       :",
        "PASS"
        if gate_f1
        else "FAIL",
    )

    print(
        "Critical regression  :",
        "PASS"
        if gate_critical
        else "FAIL",
    )

    print(
        "Regression count      :",
        int(regressions.sum()),
    )

    print(
        "\nSTATUS:"
    )

    if candidate:
        print(
            "HYBRID PASSES PRE-ONNX QUALITY GATE"
        )
        print(
            "Hybrid is eligible for ONNX export."
        )
    else:
        print(
            "HYBRID FAILS PRE-ONNX QUALITY GATE"
        )
        print(
            "Keep V3 as production baseline."
        )
        print(
            "DO NOT export Hybrid to ONNX yet."
        )

    # --------------------------------------------------------
    # Save details
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    details = unseen.copy()

    details["truth"] = [
        LABELS[x]
        for x in truth
    ]

    details["v3_prediction"] = [
        LABELS[x]
        for x in v3_pred
    ]

    details["hybrid_prediction"] = [
        LABELS[x]
        for x in hybrid_pred
    ]

    details["v3_confidence"] = v3_conf
    details["hybrid_confidence"] = hybrid_conf
    details["critical"] = critical

    details["v3_correct"] = v3_correct
    details["hybrid_correct"] = hybrid_correct

    details["hybrid_regression"] = regressions
    details["hybrid_improvement"] = improvements

    details_path = (
        OUTPUT_DIR
        / "unseen_595_details.csv"
    )

    details.to_csv(
        details_path,
        index=False,
    )

    per_intent_path = (
        OUTPUT_DIR
        / "per_intent_comparison.csv"
    )

    per_intent.to_csv(
        per_intent_path,
        index=False,
    )

    regression_path = (
        OUTPUT_DIR
        / "hybrid_regressions_vs_v3.csv"
    )

    regression_df.to_csv(
        regression_path,
        index=False,
    )

    summary = {
        "v3_onnx": str(V3_ONNX),
        "hybrid_checkpoint":
            str(HYBRID_CHECKPOINT),
        "unseen_csv":
            str(UNSEEN_CSV),
        "unseen_rows":
            len(unseen),

        "v3_accuracy":
            v3_metrics["accuracy"],

        "hybrid_accuracy":
            hybrid_metrics["accuracy"],

        "accuracy_delta_pp":
            delta_accuracy * 100,

        "v3_macro_f1":
            v3_metrics["macro_f1"],

        "hybrid_macro_f1":
            hybrid_metrics["macro_f1"],

        "macro_f1_delta_pp":
            delta_f1 * 100,

        "regression_count":
            int(regressions.sum()),

        "improvement_count":
            int(improvements.sum()),

        "critical_rows":
            int(critical.sum()),

        "critical_regression":
            bool(critical_regression),

        "gates": {
            "unseen_accuracy":
                bool(gate_accuracy),
            "macro_f1":
                bool(gate_f1),
            "critical":
                bool(gate_critical),
        },

        "production_candidate":
            bool(candidate),

        "ood":
            ood_results,

        "training_occurred":
            False,

        "threshold_fitting_occurred":
            False,

        "v3_modified":
            False,

        "hybrid_modified":
            False,

        "onnx_exported":
            False,

        "int8_exported":
            False,

        "unseen_used_for_training":
            False,
    }

    summary_path = (
        OUTPUT_DIR
        / "v3_vs_e5_hybrid_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSaved:")
    print(details_path)
    print(per_intent_path)
    print(regression_path)
    print(summary_path)

    print("\nIMPORTANT:")
    print("V3 was NOT modified.")
    print("Hybrid was NOT modified.")
    print("No training occurred.")
    print("No threshold fitting occurred.")
    print("595-row unseen set was evaluation-only.")
    print("Hybrid was NOT exported to ONNX.")
    print("Hybrid was NOT quantized.")


if __name__ == "__main__":
    main()
