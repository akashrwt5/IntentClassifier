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


PROJECT = Path("/Users/shuklam/IntentClassifier/semantic_project/57semanitc")

# Put the exact path here if auto-discovery finds more than one candidate.
LOCKED_CSV = None

CHECKPOINT = PROJECT / "v3_57intent_e5_distilled_v2" / "student_e5_distilled_v2_best_fp32.pt"
VOCAB_JSON = PROJECT / "v3_57intent_e5_distilled_v2" / "vocab.json"
LABELS_JSON = PROJECT / "v3_57intent_e5_distilled_v2" / "label_map.json"

MAX_LEN = 24
PAD_ID = 0
UNK_ID = 1
EMBED_DIM = 64
NHEAD = 4
FF_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.10


def tokenize(text, vocab):
    text = str(text).lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    ids = [int(vocab.get(tok, UNK_ID)) for tok in text.split()[:MAX_LEN]]
    ids += [PAD_ID] * (MAX_LEN - len(ids))
    return ids


class TinyIntentClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, EMBED_DIM, padding_idx=PAD_ID
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
            layer, num_layers=NUM_LAYERS
        )
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.classifier = nn.Linear(EMBED_DIM, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        pad_mask = input_ids.eq(PAD_ID)
        x = self.encoder(x, src_key_padding_mask=pad_mask)

        valid = (~pad_mask).unsqueeze(-1).float()
        denom = valid.sum(dim=1).clamp_min(1.0)
        x = (x * valid).sum(dim=1) / denom

        return self.classifier(self.norm(x))


def load_vocab(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Vocabulary not found:\n{path}\n"
            "Update VOCAB_JSON to the vocabulary used by the distilled model."
        )

    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "vocab" in obj:
        obj = obj["vocab"]

    return {str(k): int(v) for k, v in obj.items()}


def load_labels(path, checkpoint):
    if path.exists():
        obj = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(obj, dict):
            if all(str(k).isdigit() for k in obj.keys()):
                return [obj[str(i)] for i in range(len(obj))]

            if all(isinstance(v, (int, float)) for v in obj.values()):
                pairs = sorted(obj.items(), key=lambda kv: int(kv[1]))
                return [k for k, _ in pairs]

            for key in ("labels", "classes", "label_names"):
                if key in obj:
                    return list(obj[key])

    if isinstance(checkpoint, dict):
        for key in ("labels", "classes", "label_names"):
            if key in checkpoint:
                return list(checkpoint[key])

    raise FileNotFoundError(
        f"Could not load label map:\n{path}"
    )


def load_checkpoint(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{path}\n"
            "Update CHECKPOINT to the distilled V2 checkpoint."
        )
    return torch.load(path, map_location="cpu")


def get_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value

        if any(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint

    raise RuntimeError("Could not find model state_dict in checkpoint.")


def discover_locked_csv():
    candidates = []

    for p in PROJECT.rglob("*.csv"):
        name = p.name.lower()
        full = str(p).lower()

        if not any(x in name for x in (
            "locked", "lock_test", "test_locked", "final_test"
        )):
            if not any(x in full for x in (
                "locked_benchmark",
                "locked_test",
                "final_3model_locked",
            )):
                continue

        if any(x in full for x in (
            "train", "validation", "negative_test", "hard_error"
        )):
            continue

        try:
            probe = pd.read_csv(p, usecols=["text", "intent"])
            rows = len(probe)
            intents = probe["intent"].nunique()

            if rows >= 1000 and intents >= 50:
                candidates.append((p, rows, intents))
        except Exception:
            continue

    if not candidates:
        raise FileNotFoundError(
            "Could not auto-discover the locked 57-intent CSV under:\n"
            f"{PROJECT}\n\n"
            "Run this command and send me the output:\n\n"
            f'find "{PROJECT}" -type f -name "*.csv" | sort'
        )

    exact = [x for x in candidates if x[1] == 1686 and x[2] >= 57]
    if len(exact) == 1:
        return exact[0][0]

    if len(candidates) == 1:
        return candidates[0][0]

    print("Multiple possible locked-test CSVs found:")
    for i, (p, rows, intents) in enumerate(candidates, 1):
        print(f"  [{i}] {p} | rows={rows} | intents={intents}")

    raise RuntimeError(
        "Multiple locked-test CSVs found. "
        "Set LOCKED_CSV to the exact path shown above."
    )


def main():
    print("=" * 72)
    print("V2 DISTILLED STUDENT - LOCKED 57-INTENT TEST")
    print("=" * 72)

    global LOCKED_CSV

    if LOCKED_CSV is None:
        LOCKED_CSV = discover_locked_csv()

    LOCKED_CSV = Path(LOCKED_CSV)

    if not LOCKED_CSV.exists():
        raise FileNotFoundError(
            f"Locked test CSV not found:\n{LOCKED_CSV}"
        )

    print(f"Checkpoint : {CHECKPOINT}")
    print(f"Locked CSV : {LOCKED_CSV}")
    print()

    df = pd.read_csv(LOCKED_CSV)

    if not {"text", "intent"}.issubset(df.columns):
        raise ValueError(
            f"CSV must contain text and intent columns. "
            f"Found: {list(df.columns)}"
        )

    df = df.dropna(subset=["text", "intent"]).reset_index(drop=True)

    print(f"Locked rows   : {len(df)}")
    print(f"Unique intents: {df['intent'].nunique()}")

    checkpoint = load_checkpoint(CHECKPOINT)
    vocab = load_vocab(VOCAB_JSON)
    labels = load_labels(LABELS_JSON, checkpoint)

    if len(labels) != 57:
        raise RuntimeError(
            f"Expected 57 labels, got {len(labels)}"
        )

    label_to_id = {label: i for i, label in enumerate(labels)}

    unknown = sorted(set(df["intent"]) - set(label_to_id))
    if unknown:
        raise RuntimeError(
            "Locked test contains labels absent from model label map:\n"
            + "\n".join(unknown)
        )

    model = TinyIntentClassifier(
        vocab_size=len(vocab),
        num_classes=len(labels),
    )

    state = get_state_dict(checkpoint)
    clean_state = {}

    for key, value in state.items():
        clean_key = key[7:] if key.startswith("module.") else key
        clean_state[clean_key] = value

    missing, unexpected = model.load_state_dict(
        clean_state, strict=False
    )

    if missing:
        raise RuntimeError(
            "Checkpoint/model mismatch. Missing keys:\n"
            + "\n".join(missing)
        )

    if unexpected:
        print("WARNING: unexpected checkpoint keys:")
        for key in unexpected:
            print(" ", key)

    model.eval()

    X = np.asarray(
        [tokenize(x, vocab) for x in df["text"]],
        dtype=np.int64,
    )

    y_true = np.asarray(
        [label_to_id[x] for x in df["intent"]],
        dtype=np.int64,
    )

    inputs = torch.from_numpy(X)

    print()
    print("Running locked-test inference...")

    outputs = []
    start = time.perf_counter()

    with torch.no_grad():
        for i in range(0, len(inputs), 256):
            outputs.append(
                model(inputs[i:i + 256]).cpu().numpy()
            )

    elapsed = time.perf_counter() - start
    logits = np.concatenate(outputs, axis=0)

    y_pred = logits.argmax(axis=1)

    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    confidence = probs.max(axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(57)),
        target_names=labels,
        zero_division=0,
        digits=4,
    )

    print()
    print("--- V2 DISTILLED LOCKED TEST RESULT ---")
    print(f"Accuracy   : {accuracy * 100:.4f}%")
    print(f"Macro F1   : {macro_f1 * 100:.4f}%")
    print(f"Weighted F1: {weighted_f1 * 100:.4f}%")
    print()
    print(report)

    print("--- INFERENCE SPEED ---")
    print(f"Total rows : {len(df)}")
    print(f"Total time : {elapsed:.4f} sec")
    print(f"Rows/sec   : {len(df) / elapsed:.2f}")
    print(f"ms/row     : {elapsed / len(df) * 1000:.4f}")

    out_dir = (
        PROJECT / "v3_57intent_e5_distilled_v2_locked_benchmark"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    result = df.copy()
    result["true_id"] = y_true
    result["predicted_id"] = y_pred
    result["prediction"] = [labels[i] for i in y_pred]
    result["confidence"] = confidence

    result.to_csv(
        out_dir / "locked_predictions_v2.csv",
        index=False,
    )

    (out_dir / "classification_report_v2.txt").write_text(
        report, encoding="utf-8"
    )

    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(57))
    )

    pd.DataFrame(
        cm, index=labels, columns=labels
    ).to_csv(
        out_dir / "confusion_matrix_v2.csv"
    )

    summary = {
        "checkpoint": str(CHECKPOINT),
        "locked_csv": str(LOCKED_CSV),
        "rows": int(len(df)),
        "num_intents": int(len(labels)),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "inference_seconds": float(elapsed),
        "rows_per_second": float(len(df) / elapsed),
        "ms_per_row": float(elapsed / len(df) * 1000),
        "quantization": False,
        "onnx": False,
        "training_performed": False,
    }

    (out_dir / "benchmark_summary_v2.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("Saved:")
    print(out_dir / "locked_predictions_v2.csv")
    print(out_dir / "classification_report_v2.txt")
    print(out_dir / "confusion_matrix_v2.csv")
    print(out_dir / "benchmark_summary_v2.json")
    print()
    print("STATUS: V2 DISTILLED LOCKED 57-INTENT BENCHMARK COMPLETE")


if __name__ == "__main__":
    main()
