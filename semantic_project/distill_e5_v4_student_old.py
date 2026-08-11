#!/usr/bin/env python3
"""
V4 MULTILINGUAL E5 -> COMPACT STUDENT DISTILLATION

Teacher:
    intfloat/multilingual-e5-small

Student:
    compact Transformer encoder
    multilingual E5 tokenizer
    128-d embedding
    4 layers / 4 heads / FFN 256
    max length 32

Training:
    - supervised intent classification
    - E5 embedding distillation
    - teacher soft-logit distillation from class prototypes
    - hard-negative weighting
    - multilingual input support when language-labelled data is available

IMPORTANT:
    * V3 remains untouched.
    * The 595-row unseen set is NEVER used.
    * This script does not export ONNX.
    * It first creates a FP32 student checkpoint.
    * Do not call this model production-ready until the locked benchmark passes.

Expected files:
    v4_semantic_training/v4_training_data.csv
    v4_semantic_training/v4_dev_data.csv

Run:
    python3 distill_e5_v4_student.py

Optional:
    python3 distill_e5_v4_student.py --epochs 12
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
from transformers import AutoTokenizer, AutoModel

ROOT = Path("/Users/shuklam/IntentClassifier/semantic_project")
DATA = ROOT / "v4_semantic_training"
OUT = ROOT / "e5_v4_student"
OUT.mkdir(parents=True, exist_ok=True)

TEACHER_NAME = "intfloat/multilingual-e5-small"
MAX_LEN = 32
STUDENT_DIM = 128
LAYERS = 4
HEADS = 4
FFN = 256
BATCH = 32
SEED = 42

LOCKED_UNSEEN = ROOT / "unseen_semantic_stress_test.csv"


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def norm(x):
    return re.sub(r"\s+", " ", str(x).strip())


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mean_pool(x, mask):
    mask = mask.unsqueeze(-1).float()
    return (x * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def load_data():
    tr = DATA / "v4_training_data.csv"
    dv = DATA / "v4_dev_data.csv"
    if not tr.exists() or not dv.exists():
        raise FileNotFoundError("Run build_v4_semantic_data.py first.")
    train = pd.read_csv(tr)
    dev = pd.read_csv(dv)
    for df in (train, dev):
        if "text" not in df.columns or "intent" not in df.columns:
            raise ValueError("CSV needs text and intent columns.")
        df["text"] = df["text"].map(norm)
        df["intent"] = df["intent"].astype(str).str.strip()
    return train, dev


def check_unseen(train, dev):
    if not LOCKED_UNSEEN.exists():
        print("WARNING: locked unseen file not found; overlap check skipped.")
        return
    u = pd.read_csv(LOCKED_UNSEEN)
    if "text" not in u.columns:
        print("WARNING: unseen file has no text column; overlap check skipped.")
        return
    unseen = set(u["text"].map(norm))
    a = set(train.text) & unseen
    b = set(dev.text) & unseen
    print("Locked unseen overlap in train:", len(a))
    print("Locked unseen overlap in dev  :", len(b))
    if a or b:
        raise RuntimeError("CRITICAL: locked 595-row unseen data overlaps V4 train/dev.")


class Student(nn.Module):
    def __init__(self, vocab_size, n_classes, pad_id):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, STUDENT_DIM, padding_idx=pad_id)
        self.pos = nn.Embedding(MAX_LEN, STUDENT_DIM)

        layer = nn.TransformerEncoderLayer(
            d_model=STUDENT_DIM,
            nhead=HEADS,
            dim_feedforward=FFN,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=LAYERS)
        self.norm = nn.LayerNorm(STUDENT_DIM)

        self.proj = nn.Sequential(
            nn.Linear(STUDENT_DIM, STUDENT_DIM),
            nn.GELU(),
            nn.LayerNorm(STUDENT_DIM),
        )
        self.classifier = nn.Linear(STUDENT_DIM, n_classes)

    def forward(self, input_ids, attention_mask):
        b, n = input_ids.shape
        pos = torch.arange(n, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos(pos)

        pad_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=pad_mask)

        x = mean_pool(x, attention_mask)
        x = self.norm(x)
        emb = F.normalize(self.proj(x), p=2, dim=-1)
        logits = self.classifier(emb)
        return emb, logits


def tokenize(tokenizer, texts, dev):
    t = tokenizer(
        [f"query: {x}" for x in texts],
        max_length=MAX_LEN,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    return {k: v.to(dev) for k, v in t.items()}


@torch.no_grad()
def teacher_embeddings(teacher, tokenizer, texts, dev):
    t = tokenizer(
        [f"query: {x}" for x in texts],
        max_length=MAX_LEN,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    t = {k: v.to(dev) for k, v in t.items()}
    o = teacher(**t)
    e = F.normalize(mean_pool(o.last_hidden_state, t["attention_mask"]), p=2, dim=-1)
    return e


def build_teacher_prototypes(teacher, tokenizer, train, labels, dev):
    protos = []
    for lab in labels:
        texts = train.loc[train.intent == lab, "text"].tolist()
        # Limit prototype computation to keep training setup fast.
        texts = texts[:256]
        e = teacher_embeddings(teacher, tokenizer, texts, dev)
        p = F.normalize(e.mean(0, keepdim=True), p=2, dim=-1)
        protos.append(p.squeeze(0))
    return torch.stack(protos)


def estimate_weight(source):
    s = str(source).lower()
    if "hard" in s or "typo" in s or "short" in s:
        return 1.35
    if "context" in s:
        return 1.20
    return 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    seed_all()
    dev = device()
    print("Device:", dev)
    print("Teacher:", TEACHER_NAME)

    train, val = load_data()
    check_unseen(train, val)

    labels = sorted(train.intent.unique())
    if len(labels) != 11:
        raise RuntimeError(f"Expected 11 intents, got {len(labels)}")

    le = LabelEncoder()
    le.fit(labels)

    print("Train:", len(train))
    print("Validation:", len(val))
    print("Intents:", len(labels))

    if "language" in train.columns:
        print("Language distribution:")
        print(train["language"].value_counts().to_string())
    else:
        print("WARNING: no language column. " "This is not yet a 14-language training run.")

    print("\nLoading E5 teacher...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)
    teacher = AutoModel.from_pretrained(TEACHER_NAME).to(dev)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = Student(
        tokenizer.vocab_size,
        len(labels),
        tokenizer.pad_token_id,
    ).to(dev)

    prototypes = build_teacher_prototypes(teacher, tokenizer, train, labels, dev)

    # Cache teacher embeddings so the expensive E5 model is used once.
    print("\nCaching teacher embeddings...")
    train_teacher = []
    val_teacher = []

    texts = train.text.tolist()
    for i in tqdm(range(0, len(texts), BATCH), desc="Teacher train"):
        train_teacher.append(
            teacher_embeddings(teacher, tokenizer, texts[i : i + BATCH], dev).cpu()
        )
    train_teacher = torch.cat(train_teacher)

    texts = val.text.tolist()
    for i in tqdm(range(0, len(texts), BATCH), desc="Teacher validation"):
        val_teacher.append(teacher_embeddings(teacher, tokenizer, texts[i : i + BATCH], dev).cpu())
    val_teacher = torch.cat(val_teacher)

    train_y = torch.tensor(le.transform(train.intent), dtype=torch.long)
    val_y = torch.tensor(le.transform(val.intent), dtype=torch.long)

    weights = torch.tensor(
        [estimate_weight(x) for x in train.get("source", ["base"] * len(train))],
        dtype=torch.float32,
    )

    opt = torch.optim.AdamW(
        student.parameters(),
        lr=2e-3,
        weight_decay=1e-4,
    )

    best_acc = -1
    best_state = None

    train_text = train.text.tolist()
    val_text = val.text.tolist()

    for epoch in range(1, args.epochs + 1):
        student.train()
        order = np.random.permutation(len(train))
        losses = []

        for s in range(0, len(order), BATCH):
            idx = order[s : s + BATCH]
            texts = [train_text[i] for i in idx]
            y = train_y[idx].to(dev)
            tw = train_teacher[idx].to(dev)
            sw = weights[idx].to(dev)

            tok = tokenize(tokenizer, texts, dev)

            opt.zero_grad()
            se, logits = student(tok["input_ids"], tok["attention_mask"])

            # 1) supervised classification
            ce_each = F.cross_entropy(logits, y, reduction="none")
            ce = (ce_each * sw).mean()

            # 2) semantic embedding distillation
            kd_emb = (1.0 - (se * tw).sum(-1)).mean()

            # 3) teacher prototype matching
            tp = prototypes.to(dev)
            teacher_scores = tw @ tp.T
            student_scores = se @ tp.T
            proto_loss = F.mse_loss(student_scores, teacher_scores)

            loss = ce + 1.5 * kd_emb + 0.35 * proto_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()

            losses.append(float(loss.item()))

        # Validation
        student.eval()
        pred = []

        with torch.no_grad():
            for s in range(0, len(val_text), BATCH):
                tok = tokenize(tokenizer, val_text[s : s + BATCH], dev)
                _, logits = student(tok["input_ids"], tok["attention_mask"])
                pred.extend(logits.argmax(1).cpu().tolist())

        acc = accuracy_score(val_y.numpy(), pred)
        f1 = f1_score(val_y.numpy(), pred, average="macro")

        print(
            f"Epoch {epoch:02d} | "
            f"loss={np.mean(losses):.4f} | "
            f"val={acc*100:.2f}% | "
            f"valF1={f1*100:.2f}%"
        )

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

    student.load_state_dict(best_state)
    student.eval()

    pred = []
    with torch.no_grad():
        for s in range(0, len(val_text), BATCH):
            tok = tokenize(tokenizer, val_text[s : s + BATCH], dev)
            _, logits = student(tok["input_ids"], tok["attention_mask"])
            pred.extend(logits.argmax(1).cpu().tolist())

    final_acc = accuracy_score(val_y.numpy(), pred)
    final_f1 = f1_score(val_y.numpy(), pred, average="macro")

    print("\n" + "=" * 78)
    print("V4 E5 DISTILLED STUDENT")
    print("=" * 78)
    print("Validation accuracy:", f"{final_acc*100:.2f}%")
    print("Validation Macro F1:", f"{final_f1*100:.2f}%")
    print(classification_report(val_y.numpy(), pred, target_names=le.classes_, digits=4))

    ckpt = OUT / "v4_e5_distilled_student_fp32.pt"
    torch.save(
        {
            "state_dict": student.state_dict(),
            "labels": le.classes_.tolist(),
            "vocab_size": tokenizer.vocab_size,
            "pad_token_id": tokenizer.pad_token_id,
            "max_len": MAX_LEN,
            "student_dim": STUDENT_DIM,
            "layers": LAYERS,
            "heads": HEADS,
            "ffn": FFN,
            "teacher": TEACHER_NAME,
            "tokenizer_name": TEACHER_NAME,
        },
        ckpt,
    )

    manifest = {
        "teacher": TEACHER_NAME,
        "student": {
            "embedding": STUDENT_DIM,
            "layers": LAYERS,
            "heads": HEADS,
            "ffn": FFN,
            "max_len": MAX_LEN,
        },
        "train_rows": len(train),
        "validation_rows": len(val),
        "validation_accuracy": float(final_acc),
        "validation_macro_f1": float(final_f1),
        "locked_595_used": False,
        "v3_modified": False,
        "onnx_exported": False,
        "deployment_ready": False,
        "note": "FP32 student only. Benchmark against locked V3 before export.",
    }

    (OUT / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nSaved:", ckpt)
    print("V3 modified: NO")
    print("595-row unseen used: NO")
    print("ONNX exported: NO")
    print("Next: benchmark this FP32 student against locked V3.")


if __name__ == "__main__":
    main()
