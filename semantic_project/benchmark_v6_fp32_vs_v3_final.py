#!/usr/bin/env python3
"""
BENCHMARK V6 FP32 vs LOCKED V3 FP32
===================================

Compares the newly trained V6 FP32 checkpoint against the locked V3 ONNX
model on the untouched 595-row unseen set.

Gates:
  1. V6 unseen accuracy >= V3
  2. V6 macro F1 >= V3
  3. No critical intent regression
  4. Regression count is reported
  5. Optional OOD comparison if the calibration file exists

SAFETY:
  - No training
  - No threshold fitting
  - No modification of V3
  - 595-row unseen set is evaluation-only
  - V6 is not exported or quantized

Run:
  python3 benchmark_v6_fp32_vs_v3.py
"""

from pathlib import Path
import json
import re
import sys

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

V6_CHECKPOINT = (
    ROOT
    / "v6_final_e5"
    / "v6_final_e5_fp32.pt"
)

UNSEEN_CSV = (
    ROOT
    / "unseen_semantic_stress_test.csv"
)

OOD_CANDIDATES = [
    ROOT
    / "production_calibration_v2"
    / "production_ood_calibration.csv",
    ROOT
    / "production_hardening_v2"
    / "ood_predictions.csv",
]

OUTPUT_DIR = (
    ROOT
    / "v6_vs_v3_benchmark"
)


# ============================================================
# LOCKED MODEL CONFIG
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
        str(c).lower(): c
        for c in df.columns
    }

    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]

    return None


def load_vocab():
    candidates = [
        ROOT / "tiny_semantic_student_v2_balanced" / "vocab.json",
        ROOT / "tiny_semantic_student_v3_balanced" / "vocab.json",
        ROOT / "tiny_semantic_student_v3_fp32" / "vocab.json",
    ]

    for path in candidates:
        if path.exists():
            obj = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            if isinstance(obj, dict) and "vocab" in obj:
                obj = obj["vocab"]

            if not isinstance(obj, dict):
                raise RuntimeError(
                    f"Unsupported vocab format: {path}"
                )

            if len(obj) != VOCAB_SIZE:
                raise RuntimeError(
                    f"Expected vocab size {VOCAB_SIZE}, "
                    f"got {len(obj)} from {path}"
                )

            print("\nVocabulary:")
            print(path)
            print("Vocabulary size:", len(obj))

            return obj

    raise FileNotFoundError(
        "Could not find the 895-token vocab.json."
    )


def get_token_id(vocab, names, default):
    for name in names:
        if name in vocab:
            return int(vocab[name])
    return default


def tokenize(text, vocab):
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

    for name in ["<cls>", "[CLS]"]:
        if name in vocab:
            cls_id = int(vocab[name])
            break

    for name in ["<sep>", "[SEP]"]:
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
            [pad_id] * (MAX_LEN - len(ids))
        )

    return np.asarray(
        ids,
        dtype=np.int64,
    )


def is_critical(text):
    text = normalize_text(text).lower()
    return any(
        phrase in text
        for phrase in CRITICAL_PHRASES
    )


# ============================================================
# EXACT V6 ARCHITECTURE
# ============================================================

class V6Model(nn.Module):

    def __init__(self, pad_token_id):
        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            EMBED_DIM,
            padding_idx=pad_token_id,
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

        # EXACT V3 classifier:
        # 64 -> 64 -> 11
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

        # V6-only E5 adapter.
        self.e5_adapter = nn.Sequential(
            nn.Linear(384, 128),
            nn.GELU(),
            nn.Linear(128, EMBED_DIM),
            nn.LayerNorm(EMBED_DIM),
        )

    def encode(
        self,
        input_ids,
        attention_mask,
    ):
        seq_len = input_ids.shape[1]

        pos = torch.arange(
            seq_len,
            device=input_ids.device,
        ).unsqueeze(0)

        x = (
            self.embedding(input_ids)
            + self.position(pos)
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
                min=1e-9
            )
        )

        x = self.norm(x)

        return F.normalize(
            x,
            p=2,
            dim=-1,
        )

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
# LOAD V6
# ============================================================

def load_v6(vocab, device):
    if not V6_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"V6 checkpoint not found:\n{V6_CHECKPOINT}"
        )

    pad_id = get_token_id(
        vocab,
        ["<pad>", "[PAD]"],
        0,
    )

    checkpoint = torch.load(
        V6_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        state = checkpoint.get(
            "state_dict",
            checkpoint.get(
                "model_state_dict",
                checkpoint,
            ),
        )
    else:
        state = checkpoint

    if not isinstance(state, dict):
        raise RuntimeError(
            "V6 checkpoint does not contain a state_dict."
        )

    model = V6Model(
        pad_token_id=pad_id
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.to(device)
    model.eval()

    print("\nV6 checkpoint: PASS")
    print(V6_CHECKPOINT)

    return model


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
        providers=["CPUExecutionProvider"],
    )

    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]

    print("\nV3 ONNX: PASS")
    print(V3_ONNX)
    print("V3 input :", inp.name, inp.shape)
    print("V3 output:", out.name, out.shape)

    if list(inp.shape) != [1, MAX_LEN]:
        print(
            "WARNING: V3 input shape is",
            inp.shape,
        )

    if list(out.shape) != [1, NUM_CLASSES]:
        print(
            "WARNING: V3 output shape is",
            out.shape,
        )

    return session, inp.name, out.name


# ============================================================
# LOAD UNSEEN
# ============================================================

def load_unseen():
    if not UNSEEN_CSV.exists():
        raise FileNotFoundError(
            f"Locked unseen CSV not found:\n{UNSEEN_CSV}"
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
            "Unseen CSV must contain text + intent columns."
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
    ].reset_index(drop=True)

    print("\nLocked unseen rows:", len(out))

    if len(out) != 595:
        raise RuntimeError(
            f"Expected exactly 595 unseen rows, "
            f"got {len(out)}"
        )

    return out


# ============================================================
# PREDICTION
# ============================================================

@torch.no_grad()
def predict_v6(
    model,
    ids,
    mask,
):
    _, logits = model(
        ids,
        mask,
    )

    probs = torch.softmax(
        logits,
        dim=-1,
    )

    conf, pred = probs.max(
        dim=-1
    )

    return (
        pred.cpu().numpy(),
        conf.cpu().numpy(),
        probs.cpu().numpy(),
    )


def predict_v3(
    session,
    input_name,
    output_name,
    ids,
):
    """
    V3 ONNX was exported with a fixed input shape [1, 24].
    Therefore every utterance MUST be sent separately.

    Never pass the complete [595, 24] matrix to ONNX Runtime.
    """
    all_logits = []

    total = len(ids)

    for i in range(total):
        single_ids = np.asarray(
            ids[i:i + 1],
            dtype=np.int64,
        )

        # Defensive shape check: V3 requires exactly [1, 24].
        if single_ids.shape != (1, MAX_LEN):
            raise RuntimeError(
                f"V3 single-input shape error: "
                f"got {single_ids.shape}, "
                f"expected (1, {MAX_LEN})"
            )

        result = session.run(
            [output_name],
            {
                input_name: single_ids,
            },
        )

        logits = np.asarray(
            result[0],
            dtype=np.float32,
        )

        if logits.shape != (1, NUM_CLASSES):
            raise RuntimeError(
                f"V3 output shape error: "
                f"got {logits.shape}, "
                f"expected (1, {NUM_CLASSES})"
            )

        all_logits.append(
            logits[0]
        )

        if (
            (i + 1) % 100 == 0
            or i + 1 == total
        ):
            print(
                f"  V3 progress: {i + 1}/{total}",
                end="\r",
            )

    print()

    logits = np.stack(
        all_logits,
        axis=0,
    )

    shifted = (
        logits
        - logits.max(
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

    conf = probs.max(
        axis=1
    )

    return (
        pred,
        conf,
        probs,
    )


# ============================================================
# METRICS
# ============================================================

def metrics(
    truth,
    pred,
):
    return {
        "accuracy": accuracy_score(
            truth,
            pred,
        ),
        "macro_f1": f1_score(
            truth,
            pred,
            average="macro",
            zero_division=0,
        ),
    }


def critical_indices(df):
    return np.asarray(
        [
            is_critical(x)
            for x in df["text"]
        ],
        dtype=bool,
    )


def print_report(
    name,
    truth,
    pred,
):
    m = metrics(
        truth,
        pred,
    )

    print(
        f"\n--- {name} ---"
    )

    print(
        f"Accuracy : {m['accuracy']*100:.2f}%"
    )

    print(
        f"Macro F1 : {m['macro_f1']*100:.2f}%"
    )

    print(
        classification_report(
            truth,
            pred,
            labels=list(range(NUM_CLASSES)),
            target_names=LABELS,
            digits=4,
            zero_division=0,
        )
    )

    return m


# ============================================================
# OOD
# ============================================================

def load_ood():
    for path in OOD_CANDIDATES:
        if path.exists():
            df = pd.read_csv(path)

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
                continue

            out = pd.DataFrame({
                "text": df[text_col].map(
                    normalize_text
                )
            })

            print(
                "\nOOD calibration rows:",
                len(out),
            )

            return out

    return None


def ood_rejection(
    probs,
    threshold,
):
    conf = probs.max(
        axis=1
    )

    return float(
        (conf < threshold).mean()
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 78)
    print("V6 FP32 vs LOCKED V3 FP32")
    print("=" * 78)

    print("\nSAFETY")
    print("Training              : NO")
    print("Threshold fitting     : NO")
    print("V3 modification       : NO")
    print("V6 modification       : NO")
    print("595-row unseen        : EVALUATION ONLY")
    print("ONNX export           : NO")
    print("INT8 quantization     : NO")

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    print("\nDevice:", device)

    vocab = load_vocab()

    df = load_unseen()

    label_to_id = {
        label: i
        for i, label in enumerate(LABELS)
    }

    truth = np.asarray(
        [
            label_to_id[x]
            for x in df["intent"]
        ],
        dtype=np.int64,
    )

    ids_np = np.stack(
        [
            tokenize(
                text,
                vocab,
            )
            for text in df["text"]
        ]
    )

    pad_id = get_token_id(
        vocab,
        ["<pad>", "[PAD]"],
        0,
    )

    mask_np = (
        ids_np != pad_id
    ).astype(np.int64)

    ids_t = torch.tensor(
        ids_np,
        dtype=torch.long,
        device=device,
    )

    mask_t = torch.tensor(
        mask_np,
        dtype=torch.long,
        device=device,
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    v6 = load_v6(
        vocab,
        device,
    )

    v3, v3_input, v3_output = load_v3()

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    print(
        "\nRunning V6 on locked unseen..."
    )

    v6_pred, v6_conf, v6_probs = predict_v6(
        v6,
        ids_t,
        mask_t,
    )

    print(
        "Running V3 on locked unseen..."
    )

    v3_pred, v3_conf, v3_probs = predict_v3(
        v3,
        v3_input,
        v3_output,
        ids_np,
    )

    # --------------------------------------------------------
    # Main metrics
    # --------------------------------------------------------

    v3_metrics = print_report(
        "UNSEEN_595 — V3",
        truth,
        v3_pred,
    )

    v6_metrics = print_report(
        "UNSEEN_595 — V6",
        truth,
        v6_pred,
    )

    delta_acc = (
        v6_metrics["accuracy"]
        -
        v3_metrics["accuracy"]
    )

    delta_f1 = (
        v6_metrics["macro_f1"]
        -
        v3_metrics["macro_f1"]
    )

    print("\n--- DELTA V6 vs V3 ---")
    print(
        f"Accuracy delta : {delta_acc*100:+.2f} pp"
    )
    print(
        f"Macro F1 delta : {delta_f1*100:+.2f} pp"
    )

    # --------------------------------------------------------
    # Critical
    # --------------------------------------------------------

    crit = critical_indices(
        df
    )

    print(
        "\n--- CRITICAL ---"
    )

    if crit.any():
        v3_crit = accuracy_score(
            truth[crit],
            v3_pred[crit],
        )

        v6_crit = accuracy_score(
            truth[crit],
            v6_pred[crit],
        )

        print(
            f"Rows       : {int(crit.sum())}"
        )

        print(
            f"V3 accuracy: {v3_crit*100:.2f}%"
        )

        print(
            f"V6 accuracy: {v6_crit*100:.2f}%"
        )

        print(
            f"Delta      : {(v6_crit-v3_crit)*100:+.2f} pp"
        )

    else:
        v3_crit = None
        v6_crit = None
        print("No critical rows found.")

    # --------------------------------------------------------
    # Regression analysis
    # --------------------------------------------------------

    v3_correct = (
        v3_pred == truth
    )

    v6_correct = (
        v6_pred == truth
    )

    v6_regressions = (
        v3_correct
        &
        (~v6_correct)
    )

    v6_improvements = (
        (~v3_correct)
        &
        v6_correct
    )

    print(
        "\n--- REGRESSIONS ---"
    )

    print(
        "V3 correct -> V6 wrong:",
        int(v6_regressions.sum()),
    )

    print(
        "V3 wrong -> V6 correct:",
        int(v6_improvements.sum()),
    )

    # --------------------------------------------------------
    # Per-intent comparison
    # --------------------------------------------------------

    rows = []

    for i, label in enumerate(
        LABELS
    ):
        mask = truth == i

        v3_acc = (
            (v3_pred[mask] == i).mean()
            if mask.any()
            else 0.0
        )

        v6_acc = (
            (v6_pred[mask] == i).mean()
            if mask.any()
            else 0.0
        )

        rows.append({
            "intent": label,
            "support": int(mask.sum()),
            "v3_accuracy": v3_acc,
            "v6_accuracy": v6_acc,
            "delta_pp": (
                v6_acc - v3_acc
            ) * 100.0,
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
                    lambda x: f"{x*100:.2f}%",
                "v6_accuracy":
                    lambda x: f"{x*100:.2f}%",
                "delta_pp":
                    lambda x: f"{x:+.2f}",
            },
        )
    )

    # --------------------------------------------------------
    # Detailed regressions
    # --------------------------------------------------------

    regression_rows = []

    for i in np.where(
        v6_regressions
    )[0]:
        regression_rows.append({
            "row": int(i),
            "text": df.iloc[i]["text"],
            "true_intent": LABELS[truth[i]],
            "v3_prediction": LABELS[v3_pred[i]],
            "v6_prediction": LABELS[v6_pred[i]],
            "v3_confidence": float(v3_conf[i]),
            "v6_confidence": float(v6_conf[i]),
            "critical": bool(crit[i]),
        })

    regression_df = pd.DataFrame(
        regression_rows
    )

    # --------------------------------------------------------
    # OOD comparison
    # --------------------------------------------------------

    ood_df = load_ood()

    ood_results = {}

    if ood_df is not None and len(ood_df):
        ood_ids = np.stack(
            [
                tokenize(
                    text,
                    vocab,
                )
                for text in ood_df["text"]
            ]
        )

        ood_mask = (
            ood_ids != pad_id
        ).astype(np.int64)

        ood_ids_t = torch.tensor(
            ood_ids,
            dtype=torch.long,
            device=device,
        )

        ood_mask_t = torch.tensor(
            ood_mask,
            dtype=torch.long,
            device=device,
        )

        _, _, v6_ood_probs = predict_v6(
            v6,
            ood_ids_t,
            ood_mask_t,
        )

        _, _, v3_ood_probs = predict_v3(
            v3,
            v3_input,
            v3_output,
            ood_ids,
        )

        print(
            "\n--- OOD REJECTION ---"
        )

        for threshold in [
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            0.95,
            0.97,
        ]:
            v3_rej = ood_rejection(
                v3_ood_probs,
                threshold,
            )

            v6_rej = ood_rejection(
                v6_ood_probs,
                threshold,
            )

            print(
                f"threshold {threshold:.2f} | "
                f"V3 reject {v3_rej*100:6.2f}% | "
                f"V6 reject {v6_rej*100:6.2f}%"
            )

            ood_results[
                f"{threshold:.2f}"
            ] = {
                "v3_rejection":
                    v3_rej,
                "v6_rejection":
                    v6_rej,
            }

    else:
        print(
            "\nOOD dataset not found; skipped."
        )

    # --------------------------------------------------------
    # Gates
    # --------------------------------------------------------

    gate_unseen = (
        v6_metrics["accuracy"]
        >=
        v3_metrics["accuracy"]
    )

    gate_f1 = (
        v6_metrics["macro_f1"]
        >=
        v3_metrics["macro_f1"]
    )

    critical_regression = False

    if crit.any():
        critical_regression = (
            v6_crit < v3_crit
        )

    gate_critical = (
        not critical_regression
    )

    gate_regression = (
        len(regression_rows) == 0
    )

    print(
        "\n" + "=" * 78
    )
    print("GATES")
    print("=" * 78)

    print(
        "Unseen accuracy >= V3 :",
        "PASS" if gate_unseen else "FAIL",
    )

    print(
        "Macro F1 >= V3       :",
        "PASS" if gate_f1 else "FAIL",
    )

    print(
        "Critical regression  :",
        "PASS" if gate_critical else "FAIL",
    )

    print(
        "Regression count      :",
        "PASS" if gate_regression else "INFO",
        f"({len(regression_rows)})",
    )

    production_candidate = (
        gate_unseen
        and gate_f1
        and gate_critical
    )

    print(
        "\nSTATUS:"
    )

    if production_candidate:
        print(
            "V6 FP32 PASSES PRE-ONNX QUALITY GATE"
        )
        print(
            "V6 is eligible for ONNX export."
        )
    else:
        print(
            "V6 FP32 FAILS PRE-ONNX QUALITY GATE"
        )
        print(
            "DO NOT export V6 to ONNX yet."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    details = df.copy()

    details["truth"] = [
        LABELS[x]
        for x in truth
    ]

    details["v3_prediction"] = [
        LABELS[x]
        for x in v3_pred
    ]

    details["v6_prediction"] = [
        LABELS[x]
        for x in v6_pred
    ]

    details["v3_confidence"] = v3_conf
    details["v6_confidence"] = v6_conf
    details["critical"] = crit
    details["v3_correct"] = v3_correct
    details["v6_correct"] = v6_correct
    details["v6_regression"] = v6_regressions
    details["v6_improvement"] = v6_improvements

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
        / "v6_regressions_vs_v3.csv"
    )

    regression_df.to_csv(
        regression_path,
        index=False,
    )

    summary = {
        "v3_onnx": str(V3_ONNX),
        "v6_checkpoint": str(V6_CHECKPOINT),
        "unseen_csv": str(UNSEEN_CSV),
        "unseen_rows": len(df),
        "v3_accuracy": v3_metrics["accuracy"],
        "v6_accuracy": v6_metrics["accuracy"],
        "accuracy_delta_pp":
            delta_acc * 100.0,
        "v3_macro_f1": v3_metrics["macro_f1"],
        "v6_macro_f1": v6_metrics["macro_f1"],
        "macro_f1_delta_pp":
            delta_f1 * 100.0,
        "regression_count":
            int(len(regression_rows)),
        "improvement_count":
            int(v6_improvements.sum()),
        "critical_rows":
            int(crit.sum()),
        "critical_regression":
            bool(critical_regression),
        "gates": {
            "unseen_accuracy":
                bool(gate_unseen),
            "macro_f1":
                bool(gate_f1),
            "critical":
                bool(gate_critical),
            "zero_regressions":
                bool(gate_regression),
        },
        "production_candidate":
            bool(production_candidate),
        "ood": ood_results,
        "training_occurred": False,
        "threshold_fitting_occurred": False,
        "v3_modified": False,
        "v6_modified": False,
        "onnx_exported": False,
        "int8_exported": False,
        "unseen_used_for_training": False,
    }

    summary_path = (
        OUTPUT_DIR
        / "v6_vs_v3_summary.json"
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
    print("V6 was NOT modified.")
    print("No training occurred.")
    print("No threshold fitting occurred.")
    print("595-row unseen set was evaluation-only.")
    print("V6 was NOT exported to ONNX.")
    print("V6 was NOT quantized.")


if __name__ == "__main__":
    main()
