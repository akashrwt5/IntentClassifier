#!/usr/bin/env python3
"""
FINAL FAIR LOCKED BENCHMARK: BASE V3 vs V2 vs V2.1

Models:
  Base V3 Scratch
  Targeted V2
  Controlled V2.1

Evaluation:
  Same locked_test_57intent.csv for all three.

No training.
No checkpoint modification.
No locked-test tuning.
"""

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent

LOCKED_TEST = ROOT / "v3_57intent_locked_eval" / "locked_test_57intent.csv"
VOCAB_PATH = ROOT / "vocab.json"
LABELS_PATH = ROOT / "labels.json"

BASE_CKPT = ROOT / "v3_57intent_base_scratch_model" / "base_v3_scratch_best_fp32.pt"
V2_CKPT = ROOT / "v3_57intent_v2_from_scratch_base" / "student_v3_57intent_v2_best_fp32.pt"
V21_CKPT = ROOT / "v3_57intent_v2_1_controlled" / "student_v3_57intent_v2_1_best_fp32.pt"

OUT = ROOT / "v3_57intent_final_3model_locked_benchmark"
OUT.mkdir(parents=True, exist_ok=True)

SUMMARY = OUT / "final_summary.json"
REPORT = OUT / "final_report.txt"
PER_INTENT = OUT / "per_intent_3model_comparison.csv"
CONFUSIONS = OUT / "v2_1_confusion_matrix.csv"
CHANGES = OUT / "model_error_changes.csv"

SEED = 42
VOCAB_SIZE = 895
EMBED_DIM = 64
HEADS = 4
LAYERS = 2
FF_DIM = 128
MAX_LEN = 24
NUM_CLASSES = 57
DROPOUT = 0.10
BATCH_SIZE = 128


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_vocab(path):
    obj = load_json(path)
    if "token_to_id" in obj:
        v = obj["token_to_id"]
    elif "vocab" in obj and isinstance(obj["vocab"], dict):
        v = obj["vocab"]
    elif isinstance(obj, dict) and all(isinstance(x, int) for x in obj.values()):
        v = obj
    else:
        raise RuntimeError("Unsupported vocab.json format.")
    v = {str(k): int(x) for k, x in v.items()}
    if len(v) != VOCAB_SIZE:
        raise RuntimeError(f"Expected vocab size {VOCAB_SIZE}, got {len(v)}")
    return v


def load_labels(path):
    obj = load_json(path)

    if isinstance(obj, list):
        labels = [str(x) for x in obj]
    elif isinstance(obj.get("labels"), list):
        labels = [str(x) for x in obj["labels"]]
    elif isinstance(obj.get("id_to_label"), dict):
        labels = [v for _, v in sorted(
            ((int(k), str(v)) for k, v in obj["id_to_label"].items()),
            key=lambda z: z[0]
        )]
    elif isinstance(obj.get("label_to_id"), dict):
        labels = [k for _, k in sorted(
            ((int(v), str(k)) for k, v in obj["label_to_id"].items()),
            key=lambda z: z[0]
        )]
    else:
        raise RuntimeError("Unsupported labels.json format.")

    if len(labels) != NUM_CLASSES:
        raise RuntimeError(f"Expected {NUM_CLASSES} labels, got {len(labels)}")
    return labels


def tokenize(text, vocab):
    ids = []
    for token in str(text).lower().split():
        token = token.strip(".,!?;:\"'()[]{}")
        if token:
            ids.append(int(vocab.get(token, 1)))
    ids = ids[:MAX_LEN]
    ids += [0] * (MAX_LEN - len(ids))
    return ids


class V3Student57(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.position = nn.Embedding(MAX_LEN, EMBED_DIM)

        layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=HEADS,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=LAYERS)
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.classifier = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(EMBED_DIM, NUM_CLASSES),
        )

    def forward(self, x):
        pad = x.eq(0)
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.position(pos)
        h = self.encoder(h, src_key_padding_mask=pad)

        valid = (~pad).unsqueeze(-1).float()
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return self.classifier(self.norm(pooled))


class LockedDataset(Dataset):
    def __init__(self, df, vocab, label_to_id):
        self.df = df.reset_index(drop=True)
        self.x = np.asarray(
            [tokenize(t, vocab) for t in self.df["text"]],
            dtype=np.int64,
        )
        self.y = np.asarray(
            [label_to_id[t] for t in self.df["intent"]],
            dtype=np.int64,
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


def evaluate(checkpoint, dataset, device):
    model = V3Student57()

    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    ys, ps, cs = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

            ys.extend(y.numpy().tolist())
            ps.extend(pred.cpu().numpy().tolist())
            cs.extend(conf.cpu().numpy().tolist())

    return np.asarray(ys), np.asarray(ps), np.asarray(cs)


def metrics(y, p):
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, p, average="weighted", zero_division=0)),
    }


def main():
    seed_everything(SEED)

    required = [
        LOCKED_TEST,
        VOCAB_PATH,
        LABELS_PATH,
        BASE_CKPT,
        V2_CKPT,
        V21_CKPT,
    ]

    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing:\n{p}")

    vocab = load_vocab(VOCAB_PATH)
    labels = load_labels(LABELS_PATH)
    label_to_id = {x: i for i, x in enumerate(labels)}

    # The locked test is read ONLY for this final evaluation.
    df = pd.read_csv(LOCKED_TEST)

    if "text" not in df.columns or "intent" not in df.columns:
        raise RuntimeError("Locked CSV must contain text,intent.")

    df = df[["text", "intent"]].dropna().copy()
    df["text"] = df["text"].astype(str)
    df["intent"] = df["intent"].astype(str)

    unknown = sorted(set(df["intent"]) - set(labels))
    if unknown:
        raise RuntimeError("Unknown labels:\n" + "\n".join(unknown))

    if df["intent"].nunique() != NUM_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_CLASSES} intents, got {df['intent'].nunique()}"
        )

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    dataset = LockedDataset(df, vocab, label_to_id)

    print("=" * 78)
    print("FINAL 3-MODEL LOCKED 57-INTENT BENCHMARK")
    print("=" * 78)
    print(f"Rows   : {len(df)}")
    print(f"Device : {device}")
    print("\nLOCKED TEST IS EVALUATION ONLY.")
    print("NO TRAINING / NO TUNING / NO CHECKPOINT MODIFICATION.")

    models = {
        "Base V3": BASE_CKPT,
        "Targeted V2": V2_CKPT,
        "Controlled V2.1": V21_CKPT,
    }

    results = {}
    predictions = {}

    true_ref = None

    for name, ckpt in models.items():
        print(f"\n--- {name} ---")
        print(ckpt)

        y, p, c = evaluate(ckpt, dataset, device)

        if true_ref is None:
            true_ref = y
        elif not np.array_equal(true_ref, y):
            raise RuntimeError("True labels differ between evaluations.")

        m = metrics(y, p)
        results[name] = {
            **m,
            "mean_confidence": float(c.mean()),
        }
        predictions[name] = {
            "y": y,
            "p": p,
            "c": c,
        }

        print(f"Accuracy   : {m['accuracy']*100:.4f}%")
        print(f"Macro F1   : {m['macro_f1']*100:.4f}%")
        print(f"Weighted F1: {m['weighted_f1']*100:.4f}%")

    print("\n" + "=" * 78)
    print("FINAL MODEL COMPARISON")
    print("=" * 78)

    table = pd.DataFrame(results).T
    table = table.sort_values("macro_f1", ascending=False)

    print(table.to_string(float_format=lambda x: f"{x:.6f}"))

    winner = table.index[0]

    print(f"\nWINNER BY LOCKED MACRO F1: {winner}")

    # Pairwise deltas against Base.
    base = results["Base V3"]

    print("\n--- DELTA VS BASE V3 ---")

    deltas = {}

    for name, m in results.items():
        if name == "Base V3":
            continue

        d = {
            k: m[k] - base[k]
            for k in ["accuracy", "macro_f1", "weighted_f1"]
        }
        deltas[name] = d

        print(f"\n{name}")
        print(f"Accuracy   : {d['accuracy']*100:+.4f} pp")
        print(f"Macro F1   : {d['macro_f1']*100:+.4f} pp")
        print(f"Weighted F1: {d['weighted_f1']*100:+.4f} pp")

    # Per-intent metrics for all three.
    per_rows = []

    for i, label in enumerate(labels):
        row = {
            "intent": label,
        }

        for name in models:
            y = predictions[name]["y"]
            p = predictions[name]["p"]

            # one-vs-rest metrics
            true_bin = (y == i).astype(int)
            pred_bin = (p == i).astype(int)

            f1 = f1_score(
                true_bin,
                pred_bin,
                zero_division=0,
            )

            row[f"{name}_f1"] = f1

        row["V2_1_minus_Base"] = (
            row["Controlled V2.1_f1"]
            - row["Base V3_f1"]
        )

        row["V2_minus_Base"] = (
            row["Targeted V2_f1"]
            - row["Base V3_f1"]
        )

        per_rows.append(row)

    per_df = pd.DataFrame(per_rows)
    per_df.to_csv(PER_INTENT, index=False)

    print("\n--- V2.1 BIGGEST IMPROVEMENTS VS BASE ---")
    print(
        per_df.sort_values(
            "V2_1_minus_Base",
            ascending=False,
        ).head(10).to_string(index=False)
    )

    print("\n--- V2.1 BIGGEST REGRESSIONS VS BASE ---")
    print(
        per_df.sort_values(
            "V2_1_minus_Base",
        ).head(10).to_string(index=False)
    )

    # V2.1 classification report.
    v21 = predictions["Controlled V2.1"]

    v21_report = classification_report(
        v21["y"],
        v21["p"],
        target_names=labels,
        digits=4,
        zero_division=0,
    )

    print("\n--- CONTROLLED V2.1 CLASSIFICATION REPORT ---")
    print(v21_report)

    # V2.1 confusion matrix.
    cm = confusion_matrix(
        v21["y"],
        v21["p"],
        labels=list(range(NUM_CLASSES)),
    )

    pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    ).to_csv(CONFUSIONS)

    # Row-level correctness changes for all models.
    rows = []

    for i in range(len(df)):
        true_label = labels[int(true_ref[i])]

        row = {
            "text": df.iloc[i]["text"],
            "true_intent": true_label,
        }

        for name in models:
            pred = predictions[name]["p"][i]
            conf = predictions[name]["c"][i]
            row[f"{name}_prediction"] = labels[int(pred)]
            row[f"{name}_confidence"] = float(conf)
            row[f"{name}_correct"] = bool(pred == true_ref[i])

        rows.append(row)

    pd.DataFrame(rows).to_csv(CHANGES, index=False)

    summary = {
        "locked_test": {
            "path": str(LOCKED_TEST.resolve()),
            "rows": int(len(df)),
            "training_used": False,
            "tuning_used": False,
        },
        "models": {
            name: {
                "checkpoint": str(ckpt.resolve()),
                **results[name],
            }
            for name, ckpt in models.items()
        },
        "delta_vs_base": deltas,
        "winner_by_locked_macro_f1": winner,
        "selection_rule": "highest locked-test Macro F1",
        "next_step": (
            "Run negative testing and realistic hearing-aid command testing "
            "on the winner before ONNX/INT8."
        ),
    }

    SUMMARY.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    report = (
        "=" * 78
        + "\nFINAL 3-MODEL LOCKED 57-INTENT BENCHMARK\n"
        + "=" * 78
        + "\n\n"
        + table.to_string()
        + "\n\nWINNER BY LOCKED MACRO F1: "
        + winner
        + "\n\n--- V2.1 CLASSIFICATION REPORT ---\n"
        + v21_report
    )

    REPORT.write_text(
        report,
        encoding="utf-8",
    )

    print("\nSaved:")
    print(SUMMARY)
    print(REPORT)
    print(PER_INTENT)
    print(CONFUSIONS)
    print(CHANGES)

    print("\nSTATUS:")
    print("FINAL 3-MODEL LOCKED BENCHMARK COMPLETE")
    print(f"WINNER: {winner}")


if __name__ == "__main__":
    main()
