#!/usr/bin/env python3
"""
BENCHMARK V4 E5-DISTILLED FP32 vs LOCKED V3 FP32

Purpose:
  Compare the new E5-distilled V4 student against the locked V3 FP32
  on the same evaluation sets before any ONNX export.

IMPORTANT:
  - No training.
  - No threshold fitting.
  - 595-row unseen set is evaluation-only.
  - V3 is never modified.
  - V4 is not exported.
  - Do not use this result as a production gate until real microphone/OOD
    data has also been evaluated.

Run:
  python3 benchmark_v4_fp32_vs_v3.py

If a dataset path differs, edit the DATA_FILES section below.
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

from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer


# ================================================================
# PATHS
# ================================================================

ROOT = Path("/Users/shuklam/IntentClassifier/semantic_project")

V4_CHECKPOINT = (
    ROOT
    / "e5_v4_student_final"
    / "v4_e5_distilled_student_fp32.pt"
)

V3_ONNX = (
    ROOT
    / "tiny_semantic_student_v3_fp32"
    / "v3_semantic_student_fp32.onnx"
)

V3_VOCAB = (
    ROOT
    / "tiny_semantic_student_v3_balanced"
    / "vocab.json"
)

# Known locked evaluation set.
UNSEEN = ROOT / "unseen_semantic_stress_test.csv"

# The benchmark searches these names for optional evaluation sets.
DATA_FILES = {
    "contextual": [
        ROOT / "contextual_stress_test.csv",
        ROOT / "contextual_test.csv",
        ROOT / "contextual_queries.csv",
    ],
    "targeted": [
        ROOT / "targeted_critical_test.csv",
        ROOT / "targeted_critical.csv",
        ROOT / "critical_commands.csv",
    ],
    "ood": [
        ROOT / "ood_stress_test.csv",
        ROOT / "ood_test.csv",
        ROOT / "production_ood_calibration.csv",
        ROOT / "production_calibration_v2" / "production_ood_calibration.csv",
    ],
}


# ================================================================
# V4 ARCHITECTURE
# ================================================================

MAX_LEN = 32
STUDENT_DIM = 128
LAYERS = 4
HEADS = 4
FFN = 256


class V4Student(nn.Module):

    def __init__(self, vocab_size, num_classes, pad_token_id):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            STUDENT_DIM,
            padding_idx=pad_token_id,
        )

        self.position = nn.Embedding(
            MAX_LEN,
            STUDENT_DIM,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=STUDENT_DIM,
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

        self.norm = nn.LayerNorm(STUDENT_DIM)

        self.semantic_projection = nn.Sequential(
            nn.Linear(STUDENT_DIM, STUDENT_DIM),
            nn.GELU(),
            nn.LayerNorm(STUDENT_DIM),
        )

        self.classifier = nn.Linear(
            STUDENT_DIM,
            num_classes,
        )

        self.teacher_projection = nn.Sequential(
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Linear(256, STUDENT_DIM),
            nn.LayerNorm(STUDENT_DIM),
        )

    def forward(self, input_ids, attention_mask):

        batch, seq_len = input_ids.shape

        positions = torch.arange(
            seq_len,
            device=input_ids.device,
        ).unsqueeze(0)

        x = (
            self.embedding(input_ids)
            + self.position(positions)
        )

        padding_mask = attention_mask == 0

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        mask = attention_mask.unsqueeze(-1).float()

        x = (x * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        x = self.norm(x)

        x = self.semantic_projection(x)

        x = F.normalize(x, p=2, dim=-1)

        return x, self.classifier(x)


# ================================================================
# HELPERS
# ================================================================

def normalize_text(x):
    return re.sub(r"\s+", " ", str(x).strip())


def find_first(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def load_csv(path):
    df = pd.read_csv(path)

    # Common aliases.
    if "text" not in df.columns:
        for candidate in ["utterance", "query", "sentence", "input"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "text"})
                break

    if "intent" not in df.columns:
        for candidate in ["label", "intent_name", "target"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "intent"})
                break

    if "text" not in df.columns:
        raise ValueError(
            f"{path} does not contain a text/utterance/query column."
        )

    df["text"] = df["text"].map(normalize_text)

    if "intent" in df.columns:
        df["intent"] = df["intent"].astype(str).str.strip()

    return df


def load_v4():

    if not V4_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"V4 checkpoint not found:\n{V4_CHECKPOINT}"
        )

    checkpoint = torch.load(
        V4_CHECKPOINT,
        map_location="cpu",
    )

    labels = checkpoint["labels"]

    tokenizer_name = checkpoint.get(
        "tokenizer",
        checkpoint.get(
            "teacher",
            "intfloat/multilingual-e5-small",
        ),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name
    )

    model = V4Student(
        vocab_size=checkpoint["vocab_size"],
        num_classes=len(labels),
        pad_token_id=checkpoint["pad_token_id"],
    )

    model.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
    )

    model.eval()

    return model, tokenizer, labels


def tokenize_v4(tokenizer, texts):

    return tokenizer(
        [f"query: {x}" for x in texts],
        max_length=MAX_LEN,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )


def predict_v4(model, tokenizer, texts):

    all_logits = []

    with torch.no_grad():

        for start in range(0, len(texts), 32):

            batch = texts[start:start + 32]

            tokens = tokenize_v4(
                tokenizer,
                batch,
            )

            _, logits = model(
                tokens["input_ids"],
                tokens["attention_mask"],
            )

            all_logits.append(
                logits.cpu().numpy()
            )

    return np.concatenate(
        all_logits,
        axis=0,
    )


# ================================================================
# V3 TOKENIZER
# ================================================================

def load_vocab():

    if not V3_VOCAB.exists():
        raise FileNotFoundError(
            f"V3 vocab not found:\n{V3_VOCAB}"
        )

    obj = json.loads(
        V3_VOCAB.read_text(
            encoding="utf-8"
        )
    )

    # Support common vocab.json formats.
    if isinstance(obj, dict) and "vocab" in obj:
        obj = obj["vocab"]

    if not isinstance(obj, dict):
        raise ValueError(
            "Unsupported V3 vocab.json format."
        )

    return obj


def v3_tokenize_text(text, vocab, max_len=24):

    # This tokenizer mirrors the simple whitespace/subword style used
    # by the compact V3 project. It deliberately stays local/offline.
    text = normalize_text(text).lower()

    words = text.split()

    ids = []

    unk = (
        vocab.get("<unk>")
        if "<unk>" in vocab
        else vocab.get("[UNK]", 0)
    )

    pad = (
        vocab.get("<pad>")
        if "<pad>" in vocab
        else vocab.get("[PAD]", 0)
    )

    cls = (
        vocab.get("<cls>")
        if "<cls>" in vocab
        else vocab.get("[CLS]")
    )

    sep = (
        vocab.get("<sep>")
        if "<sep>" in vocab
        else vocab.get("[SEP]")
    )

    if cls is not None:
        ids.append(cls)

    for word in words:

        if word in vocab:
            ids.append(vocab[word])
            continue

        # Try character pieces if the vocabulary contains them.
        pieces = []
        remaining = word

        while remaining:

            found = None

            for n in range(
                min(len(remaining), 8),
                0,
                -1,
            ):
                piece = remaining[:n]

                candidates = [
                    piece,
                    "##" + piece,
                    "▁" + piece,
                ]

                for candidate in candidates:
                    if candidate in vocab:
                        found = candidate
                        break

                if found is not None:
                    break

            if found is None:
                pieces = None
                break

            pieces.append(
                vocab[found]
            )

            if found.startswith("##"):
                remaining = remaining[len(found) - 2:]
            elif found.startswith("▁"):
                remaining = remaining[len(found) - 1:]
            else:
                remaining = remaining[len(found):]

        if pieces:
            ids.extend(pieces)
        else:
            ids.append(unk)

    if sep is not None:
        ids.append(sep)

    ids = ids[:max_len]

    attention = [1] * len(ids)

    while len(ids) < max_len:
        ids.append(pad)
        attention.append(0)

    return ids, attention


def predict_v3(session, vocab, texts):

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    ids = []
    masks = []

    for text in texts:
        i, m = v3_tokenize_text(
            text,
            vocab,
            max_len=24,
        )
        ids.append(i)
        masks.append(m)

    x = np.asarray(
        ids,
        dtype=np.int64,
    )

    # Some V3 exports accept [N,24]; if the model exposes dynamic
    # batch this works directly. If it is fixed at batch=1, run one
    # sample at a time.
    shape = session.get_inputs()[0].shape

    if len(shape) == 2 and shape[0] == 1:

        outputs = []

        for row in x:

            out = session.run(
                [output_name],
                {
                    input_name:
                        row[None, :],
                },
            )[0]

            outputs.append(
                out[0]
            )

        return np.asarray(outputs)

    out = session.run(
        [output_name],
        {
            input_name: x,
        },
    )[0]

    return np.asarray(out)


# ================================================================
# METRICS
# ================================================================

def softmax(logits):

    z = logits - logits.max(
        axis=1,
        keepdims=True,
    )

    e = np.exp(z)

    return e / e.sum(
        axis=1,
        keepdims=True,
    )


def evaluate_classification(
    name,
    df,
    v4_logits,
    v3_logits,
    labels,
):

    if "intent" not in df.columns:

        print(
            f"\n--- {name.upper()} ---"
        )

        print(
            "No intent column; classification "
            "accuracy/F1 skipped."
        )

        return {}

    label_to_id = {
        label: i
        for i, label in enumerate(labels)
    }

    usable = df["intent"].isin(
        label_to_id
    )

    if not usable.all():

        unknown = sorted(
            set(
                df.loc[
                    ~usable,
                    "intent",
                ]
            )
        )

        print(
            f"WARNING {name}: "
            f"{len(unknown)} unknown labels: {unknown}"
        )

    df2 = df.loc[usable].reset_index(
        drop=True
    )

    if len(df2) == 0:
        return {}

    y = np.asarray(
        [
            label_to_id[x]
            for x in df2["intent"]
        ]
    )

    v4_pred = v4_logits[
        usable.to_numpy()
    ].argmax(1)

    v3_pred = v3_logits[
        usable.to_numpy()
    ].argmax(1)

    v4_acc = accuracy_score(
        y,
        v4_pred,
    )

    v3_acc = accuracy_score(
        y,
        v3_pred,
    )

    v4_f1 = f1_score(
        y,
        v4_pred,
        average="macro",
    )

    v3_f1 = f1_score(
        y,
        v3_pred,
        average="macro",
    )

    print(
        f"\n--- {name.upper()} ---"
    )

    print(
        f"V3 accuracy : {v3_acc*100:.2f}%"
    )

    print(
        f"V4 accuracy : {v4_acc*100:.2f}%"
    )

    print(
        f"Delta       : {(v4_acc-v3_acc)*100:+.2f} pp"
    )

    print(
        f"V3 Macro F1 : {v3_f1*100:.2f}%"
    )

    print(
        f"V4 Macro F1 : {v4_f1*100:.2f}%"
    )

    print(
        "\nV4 classification report:"
    )

    print(
        classification_report(
            y,
            v4_pred,
            labels=list(range(len(labels))),
            target_names=labels,
            digits=4,
            zero_division=0,
        )
    )

    return {
        "v3_accuracy": v3_acc,
        "v4_accuracy": v4_acc,
        "v3_macro_f1": v3_f1,
        "v4_macro_f1": v4_f1,
    }


def evaluate_ood(
    name,
    df,
    v3_logits,
    v4_logits,
):

    p3 = softmax(v3_logits)
    p4 = softmax(v4_logits)

    v3_conf = p3.max(1)
    v4_conf = p4.max(1)

    print(
        f"\n--- {name.upper()} OOD ---"
    )

    result = {}

    for threshold in [
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.95,
        0.97,
    ]:

        v3_reject = (
            v3_conf < threshold
        ).mean()

        v4_reject = (
            v4_conf < threshold
        ).mean()

        print(
            f"threshold {threshold:.2f} | "
            f"V3 reject {v3_reject*100:6.2f}% | "
            f"V4 reject {v4_reject*100:6.2f}%"
        )

        result[str(threshold)] = {
            "v3_rejection":
                float(v3_reject),
            "v4_rejection":
                float(v4_reject),
        }

    return result


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 78)
    print("V4 E5-DISTILLED FP32 vs LOCKED V3 FP32")
    print("=" * 78)

    print("\nV3:")
    print(V3_ONNX)

    print("\nV4:")
    print(V4_CHECKPOINT)

    print("\nLocked unseen:")
    print(UNSEEN)

    if not UNSEEN.exists():
        raise FileNotFoundError(
            "The locked 595-row unseen CSV was not found."
        )

    # ------------------------------------------------------------
    # LOAD V4
    # ------------------------------------------------------------

    print(
        "\nLoading V4 FP32..."
    )

    v4, v4_tokenizer, labels = load_v4()

    print(
        "V4 labels:",
        len(labels),
    )

    # ------------------------------------------------------------
    # LOAD V3
    # ------------------------------------------------------------

    print(
        "\nLoading V3 ONNX..."
    )

    if not V3_ONNX.exists():
        raise FileNotFoundError(
            f"V3 ONNX not found:\n{V3_ONNX}"
        )

    v3_session = ort.InferenceSession(
        str(V3_ONNX),
        providers=["CPUExecutionProvider"],
    )

    print(
        "V3 input:",
        v3_session.get_inputs()[0].name,
        v3_session.get_inputs()[0].shape,
    )

    print(
        "V3 output:",
        v3_session.get_outputs()[0].name,
        v3_session.get_outputs()[0].shape,
    )

    v3_vocab = load_vocab()

    # ------------------------------------------------------------
    # LOAD UNSEEN
    # ------------------------------------------------------------

    unseen = load_csv(
        UNSEEN
    )

    print(
        "\nUnseen rows:",
        len(unseen),
    )

    if len(unseen) != 595:

        raise RuntimeError(
            "CRITICAL: locked unseen set is not 595 rows. "
            f"Found {len(unseen)}."
        )

    texts = unseen[
        "text"
    ].tolist()

    print(
        "\nRunning V4 on locked unseen..."
    )

    v4_unseen = predict_v4(
        v4,
        v4_tokenizer,
        texts,
    )

    print(
        "Running V3 on locked unseen..."
    )

    v3_unseen = predict_v3(
        v3_session,
        v3_vocab,
        texts,
    )

    if v3_unseen.shape[1] != len(labels):

        raise RuntimeError(
            "V3/V4 class count mismatch: "
            f"V3={v3_unseen.shape}, "
            f"V4={v4_unseen.shape}"
        )

    # ------------------------------------------------------------
    # UNSEEN
    # ------------------------------------------------------------

    results = {}

    results["unseen"] = evaluate_classification(
        "unseen_595",
        unseen,
        v4_unseen,
        v3_unseen,
        labels,
    )

    # ------------------------------------------------------------
    # OPTIONAL DATASETS
    # ------------------------------------------------------------

    for name, candidates in DATA_FILES.items():

        path = find_first(candidates)

        if path is None:

            print(
                f"\n{name.upper()}: no dataset found; skipped."
            )

            continue

        print(
            f"\n{name.upper()} file:"
        )

        print(path)

        df = load_csv(path)

        print(
            "Rows:",
            len(df),
        )

        texts = df[
            "text"
        ].tolist()

        v4_logits = predict_v4(
            v4,
            v4_tokenizer,
            texts,
        )

        v3_logits = predict_v3(
            v3_session,
            v3_vocab,
            texts,
        )

        if name == "ood":

            results[name] = evaluate_ood(
                name,
                df,
                v3_logits,
                v4_logits,
            )

        else:

            results[name] = evaluate_classification(
                name,
                df,
                v4_logits,
                v3_logits,
                labels,
            )

    # ------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------

    print(
        "\n" + "=" * 78
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 78
    )

    for name in [
        "unseen",
        "contextual",
        "targeted",
    ]:

        if name not in results:
            continue

        r = results[name]

        if not r:
            continue

        print(
            f"{name:12s} | "
            f"V3 {r['v3_accuracy']*100:6.2f}% | "
            f"V4 {r['v4_accuracy']*100:6.2f}% | "
            f"Delta {(r['v4_accuracy']-r['v3_accuracy'])*100:+6.2f} pp"
        )

    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------

    out_dir = (
        ROOT
        / "v4_vs_v3_benchmark"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        out_dir
        / "v4_vs_v3_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            results,
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    # Save unseen per-row predictions for regression analysis.
    v3_probs = softmax(
        v3_unseen
    )

    v4_probs = softmax(
        v4_unseen
    )

    detail = unseen.copy()

    detail["v3_pred"] = [
        labels[i]
        for i in v3_probs.argmax(1)
    ]

    detail["v4_pred"] = [
        labels[i]
        for i in v4_probs.argmax(1)
    ]

    detail["v3_confidence"] = v3_probs.max(1)
    detail["v4_confidence"] = v4_probs.max(1)

    detail["v3_correct"] = (
        detail["v3_pred"]
        ==
        detail["intent"]
    )

    detail["v4_correct"] = (
        detail["v4_pred"]
        ==
        detail["intent"]
    )

    detail["regression"] = np.select(
        [
            detail["v3_correct"]
            &
            ~detail["v4_correct"],

            ~detail["v3_correct"]
            &
            detail["v4_correct"],
        ],
        [
            "V3_ONLY",
            "V4_ONLY",
        ],
        default="SAME",
    )

    detail.to_csv(
        out_dir
        / "unseen_595_details.csv",
        index=False,
    )

    print(
        "\nSaved:"
    )

    print(
        summary_path
    )

    print(
        out_dir
        / "unseen_595_details.csv"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "V3 was NOT modified."
    )

    print(
        "V4 was NOT exported."
    )

    print(
        "No training occurred."
    )

    print(
        "No threshold fitting occurred."
    )

    print(
        "The locked 595-row unseen set was evaluation-only."
    )

    print(
        "\nNEXT:"
    )

    print(
        "Review V4 regressions before ONNX export."
    )


if __name__ == "__main__":
    main()
