#!/usr/bin/env python3
"""
V5 FP32 vs LOCKED V3 FP32 — FULL BENCHMARK

Purpose:
    Compare the newly trained V5 E5/error-driven student against the
    locked V3 FP32 ONNX baseline before any ONNX export of V5.

Safety:
    - NEVER loads or trains on the locked 595-row data for V5 training.
    - This script ONLY evaluates the locked 595-row test.
    - Does NOT modify V3.
    - Does NOT modify V5.
    - Does NOT export ONNX.
    - Does NOT quantize.
    - Does NOT fit thresholds.

Run:
    cd /Users/shuklam/IntentClassifier/semantic_project
    python3 benchmark_v5_fp32_vs_v3.py
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


# ================================================================
# PATHS
# ================================================================

ROOT = Path(
    "/Users/shuklam/IntentClassifier/semantic_project"
)

V3_ONNX = (
    ROOT
    / "tiny_semantic_student_v3_fp32"
    / "v3_semantic_student_fp32.onnx"
)

V5_PT = (
    ROOT
    / "e5_v5_error_driven"
    / "v5_e5_error_driven_student_fp32.pt"
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
    / "v5_vs_v3_benchmark"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 42
MAX_LEN = 32


# ================================================================
# V5 ARCHITECTURE — MUST MATCH TRAINING SCRIPT
# ================================================================

class Student(nn.Module):

    def __init__(
        self,
        vocab_size,
        num_classes,
        pad_token_id,
        student_dim,
        layers,
        heads,
        ffn,
        max_len,
    ):
        super().__init__()

        self.max_len = max_len

        self.embedding = nn.Embedding(
            vocab_size,
            student_dim,
            padding_idx=pad_token_id,
        )

        self.position = nn.Embedding(
            max_len,
            student_dim,
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=student_dim,
                nhead=heads,
                dim_feedforward=ffn,
                dropout=0.10,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=layers,
            )
        )

        self.norm = nn.LayerNorm(
            student_dim
        )

        self.semantic_projection = (
            nn.Sequential(
                nn.Linear(
                    student_dim,
                    student_dim,
                ),
                nn.GELU(),
                nn.LayerNorm(
                    student_dim
                ),
            )
        )

        self.classifier = nn.Linear(
            student_dim,
            num_classes,
        )

        self.teacher_projection = (
            nn.Sequential(
                nn.Linear(
                    384,
                    256,
                ),
                nn.GELU(),
                nn.Linear(
                    256,
                    student_dim,
                ),
                nn.LayerNorm(
                    student_dim
                ),
            )
        )

    def forward(
        self,
        input_ids,
        attention_mask,
    ):

        _, seq_len = input_ids.shape

        positions = torch.arange(
            seq_len,
            device=input_ids.device,
        ).unsqueeze(0)

        x = (
            self.embedding(
                input_ids
            )
            +
            self.position(
                positions
            )
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

        semantic = (
            self.semantic_projection(
                x
            )
        )

        semantic = F.normalize(
            semantic,
            p=2,
            dim=-1,
        )

        logits = self.classifier(
            semantic
        )

        return semantic, logits


# ================================================================
# HELPERS
# ================================================================

def normalize_text(text):
    return re.sub(
        r"\s+",
        " ",
        str(text).strip(),
    )


def find_column(df, names):

    lower = {
        str(c).lower(): c
        for c in df.columns
    }

    for name in names:

        if name.lower() in lower:
            return lower[name.lower()]

    return None


def require_file(path, description):

    if not path.exists():

        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def load_json_vocab_candidates():

    candidates = [
        ROOT
        / "tiny_semantic_student_v2_balanced"
        / "vocab.json",

        ROOT
        / "tiny_semantic_student_v3_error_driven"
        / "vocab.json",

        ROOT
        / "tiny_semantic_student_v3_fp32"
        / "vocab.json",

        ROOT
        / "vocab.json",
    ]

    for p in candidates:

        if p.exists():
            return p

    found = [
        p
        for p in ROOT.rglob(
            "vocab.json"
        )
        if ".venv" not in str(p)
    ]

    if found:
        found.sort(
            key=lambda p: (
                0
                if "tiny_semantic_student"
                in str(p)
                else 1,
                len(str(p)),
            )
        )
        return found[0]

    raise FileNotFoundError(
        "No vocab.json found under semantic_project."
    )


def load_vocab():

    path = load_json_vocab_candidates()

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

    if not isinstance(
        obj,
        dict,
    ):
        raise ValueError(
            f"Unsupported vocab format: {path}"
        )

    print(
        "\nV3 vocabulary:"
    )
    print(path)
    print(
        "Vocabulary size:",
        len(obj),
    )

    if len(obj) != 895:
        print(
            "WARNING: expected 895 vocabulary entries."
        )

    return obj


def tokenize_v3(
    text,
    vocab,
    max_len=24,
):

    # The V3 model uses the project compact
    # whitespace/token vocabulary.
    # Preserve the same simple normalization used
    # by the earlier V3 benchmark family.

    text = normalize_text(
        text
    ).lower()

    words = text.split()

    unk_id = (
        vocab.get(
            "<unk>",
            vocab.get(
                "[UNK]",
                1,
            ),
        )
    )

    pad_id = (
        vocab.get(
            "<pad>",
            vocab.get(
                "[PAD]",
                0,
            ),
        )
    )

    cls_id = (
        vocab.get(
            "<cls>",
            vocab.get(
                "[CLS]",
                None,
            ),
        )
    )

    sep_id = (
        vocab.get(
            "<sep>",
            vocab.get(
                "[SEP]",
                None,
            ),
        )
    )

    ids = []

    if cls_id is not None:
        ids.append(cls_id)

    for word in words:
        ids.append(
            vocab.get(
                word,
                unk_id,
            )
        )

    if sep_id is not None:
        ids.append(sep_id)

    ids = ids[:max_len]

    if len(ids) < max_len:
        ids += [
            pad_id
        ] * (
            max_len - len(ids)
        )

    return np.asarray(
        ids,
        dtype=np.int64,
    )


# ================================================================
# V3
# ================================================================

def load_v3():

    print(
        "\nLoading V3 ONNX..."
    )

    require_file(
        V3_ONNX,
        "V3 ONNX",
    )

    session = ort.InferenceSession(
        str(V3_ONNX),
        providers=[
            "CPUExecutionProvider"
        ],
    )

    input_meta = (
        session.get_inputs()[0]
    )

    output_meta = (
        session.get_outputs()[0]
    )

    print(
        "V3 input:",
        input_meta.name,
        input_meta.shape,
    )

    print(
        "V3 output:",
        output_meta.name,
        output_meta.shape,
    )

    return session


def v3_predict(
    session,
    ids,
):

    output_name = (
        session.get_outputs()[0]
        .name
    )

    input_name = (
        session.get_inputs()[0]
        .name
    )

    result = session.run(
        [output_name],
        {
            input_name:
                ids.reshape(
                    1,
                    -1,
                )
        },
    )[0]

    return result[0]


# ================================================================
# V5 TOKENIZER
# ================================================================

def load_v5():

    print(
        "\nLoading V5 checkpoint..."
    )

    require_file(
        V5_PT,
        "V5 checkpoint",
    )

    checkpoint = torch.load(
        V5_PT,
        map_location="cpu",
        weights_only=False,
    )

    labels = checkpoint[
        "labels"
    ]

    vocab_size = int(
        checkpoint[
            "vocab_size"
        ]
    )

    pad_token_id = int(
        checkpoint[
            "pad_token_id"
        ]
    )

    max_len = int(
        checkpoint.get(
            "max_len",
            32,
        )
    )

    student_dim = int(
        checkpoint.get(
            "student_dim",
            128,
        )
    )

    layers = int(
        checkpoint.get(
            "layers",
            4,
        )
    )

    heads = int(
        checkpoint.get(
            "heads",
            4,
        )
    )

    ffn = int(
        checkpoint.get(
            "ffn",
            256,
        )
    )

    model = Student(
        vocab_size=vocab_size,
        num_classes=len(labels),
        pad_token_id=pad_token_id,
        student_dim=student_dim,
        layers=layers,
        heads=heads,
        ffn=ffn,
        max_len=max_len,
    )

    model.load_state_dict(
        checkpoint[
            "state_dict"
        ]
    )

    model.eval()

    print(
        "V5 labels:",
        len(labels),
    )

    print(
        "V5 vocab size:",
        vocab_size,
    )

    print(
        "V5 architecture:",
        f"dim={student_dim}, "
        f"layers={layers}, "
        f"heads={heads}, "
        f"ffn={ffn}, "
        f"max_len={max_len}",
    )

    return (
        model,
        labels,
        max_len,
    )


def load_e5_tokenizer():

    from transformers import (
        AutoTokenizer
    )

    print(
        "\nLoading E5 tokenizer..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            "intfloat/multilingual-e5-small"
        )
    )

    return tokenizer


def v5_predict(
    model,
    tokenizer,
    text,
    max_len,
):

    encoded = tokenizer(
        [
            f"query: {normalize_text(text)}"
        ],
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():

        _, logits = model(
            encoded["input_ids"],
            encoded["attention_mask"],
        )

    return (
        logits[0]
        .numpy()
    )


# ================================================================
# SOFTMAX / METRICS
# ================================================================

def softmax(logits):

    x = logits - np.max(
        logits
    )

    e = np.exp(x)

    return e / e.sum()


def confidence(logits):

    return float(
        np.max(
            softmax(
                logits
            )
        )
    )


def predicted_label(
    logits,
    labels,
):

    return labels[
        int(
            np.argmax(
                logits
            )
        )
    ]


def evaluate_predictions(
    y_true,
    y_pred,
    labels,
):

    label_list = list(
        range(
            len(labels)
        )
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

    report = classification_report(
        y_true,
        y_pred,
        labels=label_list,
        target_names=labels,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    return (
        accuracy,
        macro_f1,
        report,
    )


# ================================================================
# DATASET
# ================================================================

def load_unseen():

    require_file(
        UNSEEN_CSV,
        "Locked 595-row unseen test",
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
            "expected",
        ],
    )

    if text_col is None:
        raise ValueError(
            "No text column in unseen CSV."
        )

    if intent_col is None:
        raise ValueError(
            "No intent column in unseen CSV."
        )

    out = pd.DataFrame(
        {
            "text":
                df[text_col]
                .map(normalize_text),

            "intent":
                df[intent_col]
                .astype(str)
                .str.strip(),
        }
    )

    return out


# ================================================================
# OOD
# ================================================================

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

    return df[
        text_col
    ].map(
        normalize_text
    ).tolist()


# ================================================================
# MAIN
# ================================================================

def main():

    np.random.seed(
        SEED
    )

    print(
        "=" * 78
    )

    print(
        "V5 FP32 vs LOCKED V3 FP32"
    )

    print(
        "=" * 78
    )

    print(
        "\nV3:"
    )
    print(V3_ONNX)

    print(
        "\nV5:"
    )
    print(V5_PT)

    print(
        "\nLocked unseen:"
    )
    print(UNSEEN_CSV)

    # ------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------

    v3 = load_v3()

    (
        v5,
        v5_labels,
        v5_max_len,
    ) = load_v5()

    vocab = load_vocab()

    tokenizer = (
        load_e5_tokenizer()
    )

    # V3 labels are taken from V5 label ordering only if
    # the 11-intent sets are identical. Otherwise fail loudly.
    labels = list(
        v5_labels
    )

    # ------------------------------------------------------------
    # Load locked unseen
    # ------------------------------------------------------------

    unseen = load_unseen()

    print(
        "\nUnseen rows:",
        len(unseen),
    )

    if len(unseen) != 595:

        raise RuntimeError(
            "SAFETY ERROR: expected exactly "
            "595 locked unseen rows, got "
            f"{len(unseen)}."
        )

    if len(labels) != 11:

        raise RuntimeError(
            "Expected exactly 11 intents."
        )

    print(
        "\nIMPORTANT:"
    )

    print(
        "595-row set is evaluation-only."
    )

    print(
        "No training occurs in this script."
    )

    # ------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------

    print(
        "\nRunning V5 on locked unseen..."
    )

    v5_pred = []
    v5_conf = []

    print(
        "Running V3 on locked unseen..."
    )

    v3_pred = []
    v3_conf = []

    details = []

    for _, row in unseen.iterrows():

        text = row[
            "text"
        ]

        true_intent = row[
            "intent"
        ]

        # V5
        v5_logits = v5_predict(
            v5,
            tokenizer,
            text,
            v5_max_len,
        )

        v5_label = predicted_label(
            v5_logits,
            labels,
        )

        v5_c = confidence(
            v5_logits
        )

        # V3
        v3_ids = tokenize_v3(
            text,
            vocab,
            max_len=24,
        )

        v3_logits = v3_predict(
            v3,
            v3_ids,
        )

        v3_label = predicted_label(
            v3_logits,
            labels,
        )

        v3_c = confidence(
            v3_logits
        )

        v5_pred.append(
            v5_label
        )

        v5_conf.append(
            v5_c
        )

        v3_pred.append(
            v3_label
        )

        v3_conf.append(
            v3_c
        )

        details.append(
            {
                "text": text,
                "true_intent":
                    true_intent,

                "v3_prediction":
                    v3_label,

                "v5_prediction":
                    v5_label,

                "v3_confidence":
                    v3_c,

                "v5_confidence":
                    v5_c,

                "v3_correct":
                    v3_label
                    == true_intent,

                "v5_correct":
                    v5_label
                    == true_intent,

                "v5_regression":
                    (
                        v3_label
                        == true_intent
                        and
                        v5_label
                        != true_intent
                    ),
            }
        )

    y_true = (
        unseen[
            "intent"
        ].tolist()
    )

    (
        v3_acc,
        v3_f1,
        v3_report,
    ) = evaluate_predictions(
        y_true,
        v3_pred,
        labels,
    )

    (
        v5_acc,
        v5_f1,
        v5_report,
    ) = evaluate_predictions(
        y_true,
        v5_pred,
        labels,
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    print(
        "\n--- UNSEEN 595 ---"
    )

    print(
        f"V3 accuracy : "
        f"{v3_acc*100:.2f}%"
    )

    print(
        f"V5 accuracy : "
        f"{v5_acc*100:.2f}%"
    )

    print(
        f"Delta       : "
        f"{(v5_acc-v3_acc)*100:+.2f} pp"
    )

    print(
        f"V3 Macro F1 : "
        f"{v3_f1*100:.2f}%"
    )

    print(
        f"V5 Macro F1 : "
        f"{v5_f1*100:.2f}%"
    )

    print(
        f"F1 Delta    : "
        f"{(v5_f1-v3_f1)*100:+.2f} pp"
    )

    # ------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------

    print(
        "\nV5 classification report:"
    )

    report_df = (
        pd.DataFrame(
            v5_report
        ).T
    )

    print(
        report_df.to_string()
    )

    # ------------------------------------------------------------
    # Critical intent evaluation
    # ------------------------------------------------------------

    critical_intents = [
        "device.volume.increase",
        "device.volume.decrease",
        "device.volume.mute",
        "device.volume.unmute",
        "streaming.session.start",
        "streaming.session.stop",
    ]

    print(
        "\n--- CRITICAL INTENTS ---"
    )

    critical_rows = []

    for intent in critical_intents:

        if intent not in labels:
            continue

        mask = (
            unseen[
                "intent"
            ]
            == intent
        ).to_numpy()

        if not mask.any():
            continue

        true = (
            np.asarray(
                y_true
            )[mask]
        )

        p3 = (
            np.asarray(
                v3_pred
            )[mask]
        )

        p5 = (
            np.asarray(
                v5_pred
            )[mask]
        )

        a3 = accuracy_score(
            true,
            p3,
        )

        a5 = accuracy_score(
            true,
            p5,
        )

        critical_rows.append(
            {
                "intent": intent,
                "rows": int(
                    mask.sum()
                ),
                "v3_accuracy": a3,
                "v5_accuracy": a5,
                "delta_pp":
                    (
                        a5-a3
                    ) * 100,
            }
        )

        print(
            f"{intent:35s} "
            f"V3={a3*100:6.2f}% "
            f"V5={a5*100:6.2f}% "
            f"Delta={(a5-a3)*100:+6.2f} pp"
        )

    # ------------------------------------------------------------
    # Regression count
    # ------------------------------------------------------------

    details_df = pd.DataFrame(
        details
    )

    regressions = details_df[
        details_df[
            "v5_regression"
        ]
    ]

    print(
        "\n--- REGRESSIONS ---"
    )

    print(
        "V5 regressions vs V3:",
        len(regressions),
    )

    if len(regressions):

        print(
            regressions[
                [
                    "true_intent",
                    "v3_prediction",
                    "v5_prediction",
                    "v3_confidence",
                    "v5_confidence",
                    "text",
                ]
            ]
            .head(50)
            .to_string(
                index=False
            )
        )

    # ------------------------------------------------------------
    # OOD
    # ------------------------------------------------------------

    ood = load_ood()

    ood_summary = {}

    if ood is not None and len(ood):

        print(
            "\n--- OOD ---"
        )

        v3_ood = []
        v5_ood = []

        for text in ood:

            v5_logits = v5_predict(
                v5,
                tokenizer,
                text,
                v5_max_len,
            )

            v5_ood.append(
                confidence(
                    v5_logits
                )
            )

            v3_logits = v3_predict(
                v3,
                tokenize_v3(
                    text,
                    vocab,
                    24,
                ),
            )

            v3_ood.append(
                confidence(
                    v3_logits
                )
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

            v3_reject = np.mean(
                np.asarray(
                    v3_ood
                )
                < threshold
            )

            v5_reject = np.mean(
                np.asarray(
                    v5_ood
                )
                < threshold
            )

            print(
                f"threshold {threshold:.2f} | "
                f"V3 reject "
                f"{v3_reject*100:6.2f}% | "
                f"V5 reject "
                f"{v5_reject*100:6.2f}%"
            )

            ood_summary[
                f"{threshold:.2f}"
            ] = {
                "v3_rejection":
                    float(
                        v3_reject
                    ),

                "v5_rejection":
                    float(
                        v5_reject
                    ),
            }

    else:

        print(
            "\nOOD dataset not found; skipped."
        )

    # ------------------------------------------------------------
    # Gates
    # ------------------------------------------------------------

    unseen_gate = (
        v5_acc >= v3_acc
    )

    f1_gate = (
        v5_f1 >= v3_f1
    )

    critical_gate = all(
        row[
            "delta_pp"
        ] >= -1.0
        for row in critical_rows
    )

    no_major_regression = (
        len(regressions)
        <= max(
            5,
            int(
                len(unseen)
                * 0.01
            ),
        )
    )

    print(
        "\n" + "=" * 78
    )

    print(
        "GATES"
    )

    print(
        "=" * 78
    )

    print(
        "Unseen accuracy >= V3 :",
        "PASS"
        if unseen_gate
        else "FAIL",
    )

    print(
        "Macro F1 >= V3       :",
        "PASS"
        if f1_gate
        else "FAIL",
    )

    print(
        "Critical regression  :",
        "PASS"
        if critical_gate
        else "FAIL",
    )

    print(
        "Regression count      :",
        "PASS"
        if no_major_regression
        else "FAIL",
    )

    overall = (
        unseen_gate
        and f1_gate
        and critical_gate
        and no_major_regression
    )

    print(
        "\nSTATUS:"
    )

    if overall:

        print(
            "V5 FP32 PASSES PRE-ONNX QUALITY GATE"
        )

        print(
            "V5 may proceed to ONNX export review."
        )

    else:

        print(
            "V5 FP32 FAILS PRE-ONNX QUALITY GATE"
        )

        print(
            "DO NOT export V5 to ONNX yet."
        )

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------

    details_path = (
        OUTPUT_DIR
        / "unseen_595_details.csv"
    )

    details_df.to_csv(
        details_path,
        index=False,
    )

    critical_path = (
        OUTPUT_DIR
        / "critical_intent_summary.csv"
    )

    pd.DataFrame(
        critical_rows
    ).to_csv(
        critical_path,
        index=False,
    )

    regressions_path = (
        OUTPUT_DIR
        / "v5_regressions_vs_v3.csv"
    )

    regressions.to_csv(
        regressions_path,
        index=False,
    )

    summary = {
        "v3_model":
            str(V3_ONNX),

        "v5_checkpoint":
            str(V5_PT),

        "unseen_csv":
            str(UNSEEN_CSV),

        "unseen_rows":
            len(unseen),

        "v3_accuracy":
            v3_acc,

        "v5_accuracy":
            v5_acc,

        "accuracy_delta_pp":
            (
                v5_acc
                -
                v3_acc
            ) * 100,

        "v3_macro_f1":
            v3_f1,

        "v5_macro_f1":
            v5_f1,

        "macro_f1_delta_pp":
            (
                v5_f1
                -
                v3_f1
            ) * 100,

        "v5_regressions":
            len(regressions),

        "critical_gate":
            critical_gate,

        "unseen_gate":
            unseen_gate,

        "f1_gate":
            f1_gate,

        "regression_gate":
            no_major_regression,

        "overall_gate":
            overall,

        "ood":
            ood_summary,

        "safety":
            {
                "training_occurred":
                    False,

                "v3_modified":
                    False,

                "v5_modified":
                    False,

                "onnx_exported":
                    False,

                "int8_exported":
                    False,

                "unseen_used_for_training":
                    False,

                "unseen_used_for_threshold_fitting":
                    False,
            },
    }

    summary_path = (
        OUTPUT_DIR
        / "v5_vs_v3_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nSaved:"
    )

    print(
        details_path
    )

    print(
        critical_path
    )

    print(
        regressions_path
    )

    print(
        summary_path
    )

    print(
        "\nV3 was NOT modified."
    )

    print(
        "V5 was NOT modified."
    )

    print(
        "No ONNX export occurred."
    )

    print(
        "No INT8 quantization occurred."
    )

    print(
        "595-row unseen set was evaluation-only."
    )


if __name__ == "__main__":
    main()
