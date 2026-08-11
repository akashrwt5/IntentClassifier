#!/usr/bin/env python3
"""
V3 REPRODUCIBILITY AUDIT
========================

Purpose:
    Lock down the exact V3 evaluation inputs and independently reproduce
    the current 595-row result.

This audit does NOT:
    - train anything
    - modify V3
    - modify the dataset
    - fit thresholds
    - export ONNX
    - quantize
    - use the 595 rows for training

It records SHA256 hashes for:
    1. V3 ONNX
    2. vocab.json
    3. locked 595-row CSV

It also records:
    - ONNX input/output metadata
    - exact tokenizer behavior
    - prediction hash
    - accuracy
    - macro F1
    - per-intent metrics
    - confidence statistics

Run:
    cd /Users/shuklam/IntentClassifier/semantic_project
    python3 audit_v3_reproducibility.py
"""

from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
import onnxruntime as ort

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)


ROOT = Path("/Users/shuklam/IntentClassifier/semantic_project")

V3_ONNX = ROOT / "tiny_semantic_student_v3_fp32" / "v3_semantic_student_fp32.onnx"

UNSEEN_CSV = ROOT / "unseen_semantic_stress_test.csv"

VOCAB_CANDIDATES = [
    ROOT / "tiny_semantic_student_v2_balanced" / "vocab.json",
    ROOT / "tiny_semantic_student_v3_balanced" / "vocab.json",
    ROOT / "tiny_semantic_student_v3_fp32" / "vocab.json",
]

OUTPUT_DIR = ROOT / "v3_reproducibility_audit"

MAX_LEN = 24
VOCAB_SIZE = 895
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


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def normalize_text(x):
    return re.sub(
        r"\s+",
        " ",
        str(x).strip(),
    )


def find_vocab():
    valid = []

    for path in VOCAB_CANDIDATES:
        if not path.exists():
            continue

        try:
            obj = json.loads(path.read_text(encoding="utf-8"))

            if isinstance(obj, dict) and "vocab" in obj:
                obj = obj["vocab"]

            if isinstance(obj, dict):
                valid.append((path, obj))
        except Exception:
            pass

    if not valid:
        raise FileNotFoundError("No vocab.json found.")

    exact = [x for x in valid if len(x[1]) == VOCAB_SIZE]

    if not exact:
        raise RuntimeError("Found vocab files, but none has " f"exactly {VOCAB_SIZE} tokens.")

    # IMPORTANT:
    # The benchmark history showed that V3 was resolved to
    # tiny_semantic_student_v2_balanced/vocab.json.
    # Prefer that path when it exists, because this audit is
    # checking reproducibility of the already-run benchmark.
    preferred = [
        x
        for x in exact
        if x[0].name == "vocab.json" and "tiny_semantic_student_v2_balanced" in str(x[0])
    ]

    if preferred:
        return preferred[0]

    return exact[0]


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
        ids.append(int(vocab.get(token, unk_id)))

    if sep_id is not None:
        ids.append(sep_id)

    ids = ids[:MAX_LEN]

    if len(ids) < MAX_LEN:
        ids.extend([pad_id] * (MAX_LEN - len(ids)))

    return np.asarray(
        ids,
        dtype=np.int64,
    )


def load_unseen():
    if not UNSEEN_CSV.exists():
        raise FileNotFoundError(f"Locked unseen CSV not found:\n{UNSEEN_CSV}")

    df = pd.read_csv(UNSEEN_CSV)

    cols = {str(c).strip().lower(): c for c in df.columns}

    text_col = next(
        (
            cols[x]
            for x in [
                "text",
                "utterance",
                "query",
                "sentence",
            ]
            if x in cols
        ),
        None,
    )

    label_col = next(
        (
            cols[x]
            for x in [
                "intent",
                "label",
                "target",
                "expected_intent",
            ]
            if x in cols
        ),
        None,
    )

    if text_col is None or label_col is None:
        raise RuntimeError("Could not identify text/intent columns.")

    out = pd.DataFrame(
        {
            "text": df[text_col].map(normalize_text),
            "intent": df[label_col].astype(str).str.strip(),
        }
    )

    if len(out) != 595:
        raise RuntimeError(f"Expected exactly 595 rows, got {len(out)}.")

    if not out["intent"].isin(LABELS).all():
        bad = sorted(set(out["intent"]) - set(LABELS))

        raise RuntimeError(f"Unexpected labels in locked set: {bad}")

    return out


def softmax(logits):
    shifted = logits - logits.max(
        axis=1,
        keepdims=True,
    )

    exp = np.exp(shifted)

    return exp / exp.sum(
        axis=1,
        keepdims=True,
    )


def main():
    print("=" * 78)
    print("V3 REPRODUCIBILITY AUDIT")
    print("=" * 78)

    print("\nNo training.")
    print("No threshold fitting.")
    print("No model modification.")
    print("595 rows are evaluation-only.")

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    if not V3_ONNX.exists():
        raise FileNotFoundError(f"V3 ONNX not found:\n{V3_ONNX}")

    if not UNSEEN_CSV.exists():
        raise FileNotFoundError(f"595 CSV not found:\n{UNSEEN_CSV}")

    vocab_path, vocab = find_vocab()

    print("\n--- FILE IDENTITY ---")

    print("V3 ONNX:")
    print(V3_ONNX)

    v3_hash = sha256_file(V3_ONNX)

    print("V3 SHA256:")
    print(v3_hash)

    print("\nVocab:")
    print(vocab_path)

    vocab_hash = sha256_file(vocab_path)

    print("Vocab SHA256:")
    print(vocab_hash)

    print("\nLocked 595 CSV:")
    print(UNSEEN_CSV)

    csv_hash = sha256_file(UNSEEN_CSV)

    print("CSV SHA256:")
    print(csv_hash)

    print("\nVocabulary size:", len(vocab))

    # --------------------------------------------------------
    # ONNX metadata
    # --------------------------------------------------------

    session = ort.InferenceSession(
        str(V3_ONNX),
        providers=["CPUExecutionProvider"],
    )

    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]

    print("\n--- ONNX METADATA ---")

    print(
        "Input name :",
        inp.name,
    )

    print(
        "Input shape:",
        inp.shape,
    )

    print(
        "Input type :",
        inp.type,
    )

    print(
        "Output name:",
        out.name,
    )

    print(
        "Output shape:",
        out.shape,
    )

    print(
        "Output type :",
        out.type,
    )

    if inp.shape != [
        1,
        MAX_LEN,
    ]:
        raise RuntimeError(f"Unexpected V3 input shape: {inp.shape}")

    if out.shape != [
        1,
        NUM_CLASSES,
    ]:
        raise RuntimeError(f"Unexpected V3 output shape: {out.shape}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    df = load_unseen()

    label_to_id = {x: i for i, x in enumerate(LABELS)}

    truth = np.asarray(
        [label_to_id[x] for x in df["intent"]],
        dtype=np.int64,
    )

    ids = np.stack(
        [
            tokenize(
                text,
                vocab,
            )
            for text in df["text"]
        ]
    )

    print("\n--- DATASET ---")
    print("Rows:", len(df))
    print(
        "Input tensor shape:",
        ids.shape,
    )

    # --------------------------------------------------------
    # Tokenization fingerprints
    # --------------------------------------------------------

    token_hash = hashlib.sha256(ids.tobytes()).hexdigest()

    print(
        "Token tensor SHA256:",
        token_hash,
    )

    print("\nFirst 10 tokenized rows:")

    for i in range(min(10, len(ids))):
        print(
            f"{i:03d}:",
            ids[i].tolist(),
        )

    # --------------------------------------------------------
    # Inference ONE ROW AT A TIME
    # --------------------------------------------------------

    logits_list = []

    print("\n--- INFERENCE ---")

    for i in range(len(ids)):
        result = session.run(
            [out.name],
            {inp.name: ids[i : i + 1]},
        )[0]

        result = np.asarray(
            result,
            dtype=np.float32,
        )

        if result.shape != (
            1,
            NUM_CLASSES,
        ):
            raise RuntimeError(f"Bad output at row {i}: " f"{result.shape}")

        logits_list.append(result[0])

        if (i + 1) % 100 == 0 or i + 1 == len(ids):
            print(
                f"Progress: {i+1}/{len(ids)}",
                end="\r",
            )

    print()

    logits = np.stack(logits_list)

    probs = softmax(logits)

    pred = probs.argmax(axis=1)

    confidence = probs.max(axis=1)

    prediction_hash = hashlib.sha256(pred.astype(np.int64).tobytes()).hexdigest()

    logits_hash = hashlib.sha256(logits.astype(np.float32).tobytes()).hexdigest()

    print(
        "Prediction SHA256:",
        prediction_hash,
    )

    print(
        "Logits SHA256:",
        logits_hash,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        truth,
        pred,
    )

    macro_f1 = f1_score(
        truth,
        pred,
        average="macro",
        zero_division=0,
    )

    print("\n--- REPRODUCED RESULT ---")

    print(f"Accuracy : {accuracy * 100:.4f}%")

    print(f"Macro F1 : {macro_f1 * 100:.4f}%")

    print("\nClassification report:")

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

    # --------------------------------------------------------
    # Per-row details
    # --------------------------------------------------------

    details = df.copy()

    details["row"] = np.arange(len(df))

    details["prediction"] = [LABELS[x] for x in pred]

    details["confidence"] = confidence

    details["correct"] = pred == truth

    # --------------------------------------------------------
    # Per-intent
    # --------------------------------------------------------

    print("\n--- PER INTENT ---")

    per_intent = []

    for i, label in enumerate(LABELS):
        mask = truth == i

        intent_accuracy = (pred[mask] == i).mean() if mask.any() else 0.0

        per_intent.append(
            {
                "intent": label,
                "support": int(mask.sum()),
                "accuracy": intent_accuracy,
            }
        )

        print(f"{label:35s} " f"{intent_accuracy * 100:7.2f}% " f"support={int(mask.sum())}")

    # --------------------------------------------------------
    # Save audit
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    details_path = OUTPUT_DIR / "reproduced_595_details.csv"

    details.to_csv(
        details_path,
        index=False,
    )

    summary = {
        "v3_onnx": str(V3_ONNX),
        "v3_sha256": v3_hash,
        "vocab": str(vocab_path),
        "vocab_sha256": vocab_hash,
        "locked_595_csv": str(UNSEEN_CSV),
        "locked_595_sha256": csv_hash,
        "vocab_size": len(vocab),
        "onnx_input_name": inp.name,
        "onnx_input_shape": inp.shape,
        "onnx_output_name": out.name,
        "onnx_output_shape": out.shape,
        "rows": len(df),
        "token_tensor_sha256": token_hash,
        "prediction_sha256": prediction_hash,
        "logits_sha256": logits_hash,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "training_occurred": False,
        "threshold_fitting": False,
        "model_modified": False,
        "unseen_used_for_training": False,
    }

    summary_path = OUTPUT_DIR / "v3_reproducibility_summary.json"

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSaved:")
    print(details_path)
    print(summary_path)

    print("\n" + "=" * 78)

    print("AUDIT COMPLETE")

    print(f"Reproduced accuracy : " f"{accuracy * 100:.4f}%")

    print(f"Reproduced Macro F1 : " f"{macro_f1 * 100:.4f}%")

    print("\nUse this exact SHA256 + prediction hash " "as the canonical V3 benchmark identity.")


if __name__ == "__main__":
    main()
