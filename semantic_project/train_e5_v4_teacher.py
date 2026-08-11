#!/usr/bin/env python3
"""
V4 E5 SEMANTIC TEACHER — STRONG MULTILINGUAL BASELINE

Uses intfloat/multilingual-e5-small as the semantic encoder/teacher.
The E5 model is NOT intended to be shipped to the mobile app; it is a
training-time teacher. The later student will be distilled from it.

Pipeline:
  text
    -> multilingual-e5-small
    -> normalized 384-d embedding
    -> supervised intent head

Two stages:
  1) Frozen E5 + MLP head: fast, establishes a strong semantic baseline.
  2) Optional last-layer fine-tuning: enabled with --finetune.

Important:
  - V3 remains untouched.
  - The 595-row unseen set is NEVER used for training.
  - This script does not export a deployment model.
  - If your dataset has a `language` column, it is reported so we can
    verify multilingual coverage. It does not invent translations.

Install:
  pip install torch transformers sentencepiece scikit-learn pandas numpy tqdm

Run:
  python3 train_e5_v4_teacher.py

For fine-tuning:
  python3 train_e5_v4_teacher.py --finetune --epochs 3

Expected source:
  v4_semantic_training/v4_training_data.csv
  v4_semantic_training/v4_dev_data.csv
"""

from pathlib import Path
import argparse
import json
import math
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

ROOT = Path("/Users/shuklam/IntentClassifier/semantic_project")
DATA = ROOT / "v4_semantic_training"
OUT = ROOT / "e5_v4_teacher"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "intfloat/multilingual-e5-small"
LOCKED_UNSEEN = ROOT / "unseen_semantic_stress_test.csv"

SEED = 42
MAX_LEN = 64
BATCH = 32
EMB = 384

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def norm_text(x):
    return re.sub(r"\s+", " ", str(x).strip())

def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = torch.sum(last_hidden * mask, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom

def encode_batch(model, tokenizer, texts, device, normalize=True):
    batch = tokenizer(
        list(texts),
        max_length=MAX_LEN,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        out = model(**batch)
        emb = mean_pool(out.last_hidden_state, batch["attention_mask"])
        if normalize:
            emb = F.normalize(emb, p=2, dim=1)
    return emb

class IntentHead(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)

def load_data():
    train_path = DATA / "v4_training_data.csv"
    dev_path = DATA / "v4_dev_data.csv"

    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError(
            "Run build_v4_semantic_data.py first."
        )

    train = pd.read_csv(train_path)
    dev = pd.read_csv(dev_path)

    for df in (train, dev):
        if "text" not in df.columns or "intent" not in df.columns:
            raise ValueError("Dataset must contain text and intent columns.")
        df["text"] = df["text"].map(norm_text)
        df["intent"] = df["intent"].astype(str).str.strip()

    return train, dev

def check_unseen_not_used(train, dev):
    if not LOCKED_UNSEEN.exists():
        print("WARNING: locked unseen file not found; cannot verify overlap.")
        return

    unseen = pd.read_csv(LOCKED_UNSEEN)
    if "text" not in unseen.columns:
        print("WARNING: unseen file has no text column; overlap check skipped.")
        return

    unseen_text = set(unseen["text"].map(norm_text))
    tr_overlap = set(train["text"]) & unseen_text
    dv_overlap = set(dev["text"]) & unseen_text

    print(f"Locked unseen overlap in training: {len(tr_overlap)}")
    print(f"Locked unseen overlap in dev     : {len(dv_overlap)}")

    if tr_overlap or dv_overlap:
        raise RuntimeError(
            "CRITICAL: locked unseen examples overlap V4 train/dev. "
            "Remove the overlap before training."
        )

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def build_embeddings(train, dev, tokenizer, encoder, device):
    encoder.eval()

    def run(df, name):
        chunks = []
        texts = df["text"].tolist()
        for i in tqdm(range(0, len(texts), BATCH), desc=f"Embedding {name}"):
            chunks.append(
                encode_batch(
                    encoder, tokenizer,
                    texts[i:i+BATCH],
                    device
                ).cpu().numpy()
            )
        return np.concatenate(chunks, axis=0)

    return run(train, "train"), run(dev, "dev")

def train_head(train_x, train_y, dev_x, dev_y, n_classes, device, epochs=20):
    model = IntentHead(EMB, n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.03)

    tx = torch.tensor(train_x, dtype=torch.float32)
    ty = torch.tensor(train_y, dtype=torch.long)
    dx = torch.tensor(dev_x, dtype=torch.float32, device=device)

    best_state = None
    best_acc = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(tx))
        total = 0.0

        for s in range(0, len(tx), 128):
            idx = perm[s:s+128]
            xb = tx[idx].to(device)
            yb = ty[idx].to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())

        model.eval()
        with torch.no_grad():
            pred = model(dx).argmax(1).cpu().numpy()

        acc = accuracy_score(dev_y, pred)
        f1 = f1_score(dev_y, pred, average="macro")
        print(
            f"HEAD epoch {epoch:02d} | "
            f"loss={total:.4f} | val={acc*100:.2f}% | valF1={f1*100:.2f}%"
        )

        if acc > best_acc:
            best_acc = acc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(dx).argmax(1).cpu().numpy()

    return model, pred

def run_finetune(train, dev, tokenizer, encoder, le, device, epochs):
    """
    Fine-tune E5 + classifier jointly.

    We deliberately use a conservative learning rate and only unfreeze
    the last 4 encoder blocks. This protects the pretrained multilingual
    semantic space while adapting it to the 11 intents.
    """
    for p in encoder.parameters():
        p.requires_grad = False

    # E5-small is BERT-compatible; unfreeze last four transformer blocks.
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        layers = encoder.encoder.layer
        for layer in layers[-4:]:
            for p in layer.parameters():
                p.requires_grad = True
    else:
        raise RuntimeError("Unexpected E5 encoder architecture.")

    head = IntentHead(EMB, len(le.classes__)).to(device)
    encoder.to(device)
    encoder.train()
    head.train()

    params = [
        {"params": [p for p in encoder.parameters() if p.requires_grad], "lr": 1e-5},
        {"params": head.parameters(), "lr": 8e-4},
    ]
    opt = torch.optim.AdamW(params, weight_decay=1e-4)

    steps = math.ceil(len(train) / BATCH) * epochs
    sched = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=max(1, steps // 10),
        num_training_steps=steps,
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.03)

    best = None
    best_acc = -1.0

    train_text = train["text"].tolist()
    train_y = le.transform(train["intent"]).astype(np.int64)
    dev_text = dev["text"].tolist()
    dev_y = le.transform(dev["intent"]).astype(np.int64)

    for epoch in range(1, epochs + 1):
        order = np.random.permutation(len(train))
        total = 0.0

        for s in range(0, len(order), BATCH):
            idx = order[s:s+BATCH]
            texts = [train_text[i] for i in idx]
            y = torch.tensor(train_y[idx], dtype=torch.long, device=device)

            tok = tokenizer(
                texts,
                max_length=MAX_LEN,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            tok = {k: v.to(device) for k, v in tok.items()}

            opt.zero_grad()
            out = encoder(**tok)
            emb = mean_pool(out.last_hidden_state, tok["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
            logits = head(emb)

            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(head.parameters()), 1.0
            )
            opt.step()
            sched.step()

            total += float(loss.item())

        # Dev
        encoder.eval()
        head.eval()
        preds = []

        with torch.no_grad():
            for s in range(0, len(dev_text), BATCH):
                texts = dev_text[s:s+BATCH]
                tok = tokenizer(
                    texts,
                    max_length=MAX_LEN,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                tok = {k: v.to(device) for k, v in tok.items()}
                out = encoder(**tok)
                emb = F.normalize(
                    mean_pool(out.last_hidden_state, tok["attention_mask"]),
                    p=2, dim=1
                )
                preds.extend(head(emb).argmax(1).cpu().numpy().tolist())

        acc = accuracy_score(dev_y, preds)
        f1 = f1_score(dev_y, preds, average="macro")
        print(
            f"FINETUNE epoch {epoch:02d} | "
            f"loss={total:.4f} | val={acc*100:.2f}% | valF1={f1*100:.2f}%"
        )

        if acc > best_acc:
            best_acc = acc
            best = {
                "encoder": {
                    k: v.detach().cpu().clone()
                    for k, v in encoder.state_dict().items()
                },
                "head": {
                    k: v.detach().cpu().clone()
                    for k, v in head.state_dict().items()
                },
            }

        encoder.train()
        head.train()

    encoder.load_state_dict(best["encoder"])
    head.load_state_dict(best["head"])
    return encoder, head, best_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    seed_everything()
    device = get_device()
    print("Device:", device)
    print("Teacher:", MODEL_NAME)

    train, dev = load_data()
    check_unseen_not_used(train, dev)

    labels = sorted(train["intent"].unique().tolist())
    if len(labels) != 11:
        raise RuntimeError(f"Expected 11 intents, found {len(labels)}: {labels}")

    le = LabelEncoder()
    le.fit(labels)

    print("Train rows:", len(train))
    print("Dev rows  :", len(dev))
    print("Intents   :", len(labels))

    if "language" in train.columns:
        print("Languages in train:", train["language"].value_counts().to_dict())
    else:
        print(
            "Language column: NOT PRESENT. "
            "This run establishes the English-domain E5 baseline; "
            "do not claim 14-language training coverage yet."
        )

    print("\nLoading multilingual-e5-small...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = AutoModel.from_pretrained(MODEL_NAME).to(device)

    # Stage 1: frozen E5 embeddings + supervised head.
    print("\n" + "=" * 78)
    print("STAGE 1 — FROZEN E5 + INTENT HEAD")
    print("=" * 78)

    train_x, dev_x = build_embeddings(
        train, dev, tokenizer, encoder, device
    )
    train_y = le.transform(train["intent"])
    dev_y = le.transform(dev["intent"])

    head, pred = train_head(
        train_x, train_y, dev_x, dev_y,
        len(labels), device
    )

    acc = accuracy_score(dev_y, pred)
    f1 = f1_score(dev_y, pred, average="macro")

    print("\nE5 frozen validation accuracy :", f"{acc*100:.2f}%")
    print("E5 frozen validation Macro F1 :", f"{f1*100:.2f}%")
    print(
        classification_report(
            dev_y, pred, target_names=le.classes_, digits=4
        )
    )

    # Save embeddings/head for reproducibility.
    np.save(OUT / "train_embeddings.npy", train_x)
    np.save(OUT / "dev_embeddings.npy", dev_x)
    torch.save(head.state_dict(), OUT / "e5_frozen_intent_head.pt")

    if args.finetune:
        print("\n" + "=" * 78)
        print("STAGE 2 — LAST 4 E5 LAYERS + INTENT HEAD FINE-TUNING")
        print("=" * 78)

        encoder, ft_head, ft_acc = run_finetune(
            train, dev, tokenizer, encoder, le, device, args.epochs
        )

        torch.save(
            {
                "model_name": MODEL_NAME,
                "encoder_state_dict": encoder.state_dict(),
                "head_state_dict": ft_head.state_dict(),
                "labels": le.classes_.tolist(),
                "max_len": MAX_LEN,
                "embedding_dim": EMB,
            },
            OUT / "e5_v4_finetuned_checkpoint.pt",
        )

        print(
            f"\nE5 fine-tuned validation accuracy: {ft_acc*100:.2f}%"
        )

    manifest = {
        "teacher": MODEL_NAME,
        "device": str(device),
        "max_len": MAX_LEN,
        "embedding_dim": EMB,
        "train_rows": int(len(train)),
        "dev_rows": int(len(dev)),
        "labels": le.classes_.tolist(),
        "frozen_e5_accuracy": float(acc),
        "frozen_e5_macro_f1": float(f1),
        "finetuned": bool(args.finetune),
        "locked_unseen_used": False,
        "v3_modified": False,
        "deployment_exported": False,
        "note": (
            "E5 is training-time teacher. The next phase is to distill "
            "this semantic teacher into a compact mobile student and "
            "benchmark that student against locked V3."
        ),
    }

    (OUT / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:", OUT)
    print("V3 modified: NO")
    print("595-row unseen used for training: NO")
    print("Next: use the best E5 teacher to train/distill a mobile student.")

if __name__ == "__main__":
    main()
