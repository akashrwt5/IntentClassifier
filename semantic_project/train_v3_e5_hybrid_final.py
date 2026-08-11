#!/usr/bin/env python3
"""
V3 + E5-SMALL HYBRID SEMANTIC TRAINING
======================================

Goal
----
Improve the existing V3 model using multilingual-e5-small as an
AUXILIARY TEACHER SIGNAL while preserving the V3 runtime architecture.

IMPORTANT DESIGN:
    Runtime = V3 only
    E5      = training-time teacher only

So the final model remains:
    vocab 895
    embedding 64
    transformer 2 layers / 4 heads
    FFN 128
    classifier 64 -> 64 -> 11

E5 is NOT embedded in the mobile runtime.

Why this approach?
------------------
Previous E5 V4/V5/V6 models changed the student representation too much
and lost performance on the locked 595-row unseen set.

This script instead:
  1. Starts from the locked V3 student checkpoint.
  2. Keeps the V3 architecture unchanged.
  3. Uses normal supervised CE loss as the dominant signal.
  4. Uses E5-small only as a semantic teacher.
  5. Uses E5 class prototypes to create teacher soft targets.
  6. Uses KL loss only as a small auxiliary term.
  7. Anchors the student logits to the original V3 logits.
  8. Does NOT load or evaluate the locked 595-row unseen set during training.

This is NOT V7 yet. It is a controlled hybrid experiment.
Only promote it if it beats V3 on the untouched 595-row benchmark.

Example:
    python3 train_v3_e5_hybrid.py \
        --data /path/to/train.csv \
        --v3 /Users/shuklam/IntentClassifier/semantic_project/tiny_semantic_student_v3_error_driven/student_v3_fp32.pt

The script tries to auto-discover a training CSV if --data is omitted.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel


# ============================================================
# PATHS / LABELS
# ============================================================

ROOT = Path("/Users/shuklam/IntentClassifier/semantic_project")

DEFAULT_V3 = (
    ROOT
    / "tiny_semantic_student_v3_error_driven"
    / "student_v3_fp32.pt"
)

OUT_DIR = ROOT / "v3_e5_hybrid"

E5_NAME = "intfloat/multilingual-e5-small"

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

NUM_CLASSES = 11
VOCAB_SIZE = 895
EMBED_DIM = 64
LAYERS = 2
HEADS = 4
FFN = 128
MAX_LEN = 24


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# TEXT / VOCAB
# ============================================================

def norm_text(x):
    return re.sub(r"\s+", " ", str(x).strip())


def find_vocab():
    candidates = [
        ROOT / "tiny_semantic_student_v2_balanced" / "vocab.json",
        ROOT / "tiny_semantic_student_v3_balanced" / "vocab.json",
        ROOT / "tiny_semantic_student_v3_fp32" / "vocab.json",
    ]

    for p in candidates:
        if p.exists():
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, dict) and "vocab" in obj:
                obj = obj["vocab"]
            if isinstance(obj, dict) and len(obj) == VOCAB_SIZE:
                print("Vocab:", p)
                print("Vocabulary size:", len(obj))
                return obj

    raise FileNotFoundError(
        "Could not find the 895-token vocab.json."
    )


def token_id(vocab, names, default):
    for n in names:
        if n in vocab:
            return int(vocab[n])
    return default


def tokenize_v3(text, vocab):
    pad_id = token_id(vocab, ["<pad>", "[PAD]"], 0)
    unk_id = token_id(vocab, ["<unk>", "[UNK]"], 1)

    cls_id = None
    sep_id = None

    for n in ["<cls>", "[CLS]"]:
        if n in vocab:
            cls_id = int(vocab[n])
            break

    for n in ["<sep>", "[SEP]"]:
        if n in vocab:
            sep_id = int(vocab[n])
            break

    ids = []

    if cls_id is not None:
        ids.append(cls_id)

    for tok in norm_text(text).lower().split():
        ids.append(int(vocab.get(tok, unk_id)))

    if sep_id is not None:
        ids.append(sep_id)

    ids = ids[:MAX_LEN]

    if len(ids) < MAX_LEN:
        ids += [pad_id] * (MAX_LEN - len(ids))

    return np.asarray(ids, dtype=np.int64)


# ============================================================
# DATA
# ============================================================

def discover_csv():
    """
    Discover the training CSV by usable rows, NOT by modification time.

    The previous implementation could select a CSV with valid-looking
    text/label columns but labels outside our 11 intents, resulting in
    0 usable rows. This version scores every candidate by the number of
    rows whose labels belong to LABELS and selects the strongest dataset.

    The locked 595-row unseen file is always excluded.
    """
    excluded_names = {
        "unseen_semantic_stress_test.csv",
        "production_ood_calibration.csv",
        "unseen_595_details.csv",
        "v6_regressions_vs_v3.csv",
        "v5_regressions_vs_v3.csv",
    }

    candidates = []

    for p in ROOT.rglob("*.csv"):
        if p.name in excluded_names:
            continue

        try:
            df = pd.read_csv(p)
        except Exception:
            continue

        cols = {
            str(c).strip().lower(): c
            for c in df.columns
        }

        text_col = next(
            (
                cols[x]
                for x in (
                    "text",
                    "utterance",
                    "query",
                    "sentence",
                )
                if x in cols
            ),
            None,
        )

        label_col = next(
            (
                cols[x]
                for x in (
                    "intent",
                    "label",
                    "target",
                    "expected_intent",
                )
                if x in cols
            ),
            None,
        )

        if text_col is None or label_col is None:
            continue

        labels = (
            df[label_col]
            .astype(str)
            .str.strip()
        )

        usable = labels.isin(LABELS)

        usable_count = int(usable.sum())
        unique_intents = int(
            labels[usable].nunique()
        )

        if usable_count >= 500 and unique_intents >= 8:
            candidates.append(
                (
                    usable_count,
                    unique_intents,
                    p.stat().st_mtime,
                    p,
                )
            )

    if not candidates:
        raise FileNotFoundError(
            "No suitable 11-intent training CSV was found. "
            "Run with --data /absolute/path/to/your/training.csv"
        )

    candidates.sort(
        key=lambda x: (
            x[0],
            x[1],
            x[2],
        ),
        reverse=True,
    )

    print("\nTraining CSV candidates:")
    for usable_count, unique_intents, _, p in candidates[:10]:
        print(
            f"  {p}"
            f" | usable_rows={usable_count}"
            f" | intents={unique_intents}"
        )

    selected = candidates[0][3]

    print("\nSelected training CSV:")
    print(selected)

    return selected


def load_data(path: Path):
    df = pd.read_csv(path)

    text_col = next(
        (
            c for c in df.columns
            if str(c).lower() in
            {"text", "utterance", "query", "sentence"}
        ),
        None,
    )

    label_col = next(
        (
            c for c in df.columns
            if str(c).lower() in
            {"intent", "label", "target", "expected_intent"}
        ),
        None,
    )

    if text_col is None or label_col is None:
        raise RuntimeError(
            f"Could not identify text/intent columns in {path}. "
            f"Columns: {list(df.columns)}"
        )

    raw_labels = (
        df[label_col]
        .astype(str)
        .str.strip()
    )

    print("\nDetected columns:")
    print("  text  :", text_col)
    print("  label :", label_col)

    print("\nLabel distribution before filtering:")
    print(raw_labels.value_counts().head(30).to_string())

    usable_mask = raw_labels.isin(LABELS)

    df = pd.DataFrame({
        "text": df[text_col].map(norm_text),
        "intent": raw_labels,
    })

    df = df[usable_mask].copy()
    df = df.drop_duplicates(
        subset=["text", "intent"]
    ).reset_index(drop=True)

    if len(df) < 500:
        raise RuntimeError(
            f"Only {len(df)} usable rows found after filtering to "
            f"the exact 11 intents.\n"
            f"Expected labels:\n{LABELS}\n\n"
            "Use --data with the actual 11-intent training dataset."
        )

    print("\nTraining data:", path)
    print("Rows:", len(df))
    print("Intents:", df["intent"].nunique())

    counts = df["intent"].value_counts()
    print("\nClass distribution:")
    print(counts.to_string())

    return df


# ============================================================
# EXACT V3 STUDENT ARCHITECTURE
# ============================================================

class V3Student(nn.Module):
    def __init__(self, pad_id):
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

        enc_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=HEADS,
            dim_feedforward=FFN,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=LAYERS,
        )

        self.norm = nn.LayerNorm(EMBED_DIM)

        self.classifier = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(EMBED_DIM, NUM_CLASSES),
        )

    def encode(self, input_ids, attention_mask):
        n = input_ids.shape[1]

        pos = torch.arange(
            n,
            device=input_ids.device,
        ).unsqueeze(0)

        x = (
            self.embedding(input_ids)
            + self.position(pos)
        )

        pad_mask = attention_mask == 0

        x = self.encoder(
            x,
            src_key_padding_mask=pad_mask,
        )

        m = attention_mask.unsqueeze(-1).float()

        x = (
            (x * m).sum(1)
            /
            m.sum(1).clamp(min=1e-8)
        )

        return self.norm(x)

    def forward(self, input_ids, attention_mask):
        emb = self.encode(
            input_ids,
            attention_mask,
        )
        logits = self.classifier(emb)
        return emb, logits


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


def load_v3(path, vocab, device):
    pad_id = token_id(
        vocab,
        ["<pad>", "[PAD]"],
        0,
    )

    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    state = extract_state(ckpt)

    model = V3Student(
        pad_id=pad_id
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.to(device)

    print("\nV3 checkpoint loaded EXACTLY.")
    print("V3 architecture:")
    print("  vocab:", VOCAB_SIZE)
    print("  embedding:", EMBED_DIM)
    print("  layers:", LAYERS)
    print("  heads:", HEADS)
    print("  FFN:", FFN)
    print("  classifier: 64 -> 64 -> 11")

    return model


# ============================================================
# E5 TEACHER
# ============================================================

@torch.no_grad()
def e5_embed(
    tokenizer,
    model,
    texts,
    device,
    batch_size=32,
):
    out = []

    for start in range(
        0,
        len(texts),
        batch_size,
    ):
        batch = texts[
            start:start + batch_size
        ]

        # E5 recommends query: prefix for retrieval-like
        # semantic matching. We use it consistently for both
        # training examples and class prototypes.
        batch = [
            "query: " + norm_text(x)
            for x in batch
        ]

        tok = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        tok = {
            k: v.to(device)
            for k, v in tok.items()
        }

        outputs = model(**tok)

        emb = outputs.last_hidden_state

        mask = tok["attention_mask"].unsqueeze(-1)

        emb = (
            (emb * mask).sum(1)
            /
            mask.sum(1).clamp(min=1)
        )

        emb = F.normalize(
            emb,
            p=2,
            dim=-1,
        )

        out.append(
            emb.cpu()
        )

    return torch.cat(
        out,
        dim=0,
    )


def build_teacher_prototypes(
    teacher_embeddings,
    labels,
):
    prototypes = []

    for label in LABELS:
        mask = torch.tensor(
            [
                x == label
                for x in labels
            ],
            dtype=torch.bool,
        )

        proto = teacher_embeddings[mask].mean(0)

        proto = F.normalize(
            proto,
            p=2,
            dim=-1,
        )

        prototypes.append(proto)

    return torch.stack(
        prototypes,
        dim=0,
    )


# ============================================================
# TRAINING
# ============================================================

def evaluate(
    model,
    ids,
    masks,
    y,
    device,
):
    model.eval()

    with torch.no_grad():
        _, logits = model(
            ids.to(device),
            masks.to(device),
        )

        pred = logits.argmax(
            dim=-1
        ).cpu().numpy()

    return (
        accuracy_score(y, pred),
        f1_score(
            y,
            pred,
            average="macro",
            zero_division=0,
        ),
        pred,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--v3",
        type=str,
        default=str(DEFAULT_V3),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    seed_everything(args.seed)

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    print("Device:", device)
    print("Teacher:", E5_NAME)

    vocab = find_vocab()

    data_path = (
        Path(args.data)
        if args.data
        else discover_csv()
    )

    df = load_data(
        data_path
    )

    # --------------------------------------------------------
    # IMPORTANT: explicitly reject the locked 595 test.
    # --------------------------------------------------------

    if data_path.name == "unseen_semantic_stress_test.csv":
        raise RuntimeError(
            "REFUSING to train on the locked 595-row unseen set."
        )

    # --------------------------------------------------------
    # Split train/dev
    # --------------------------------------------------------

    y_all = np.asarray(
        [
            LABELS.index(x)
            for x in df["intent"]
        ],
        dtype=np.int64,
    )

    train_idx, dev_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.10,
        random_state=args.seed,
        stratify=y_all,
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    dev_df = df.iloc[dev_idx].reset_index(drop=True)

    # --------------------------------------------------------
    # V3 tokenization
    # --------------------------------------------------------

    train_ids_np = np.stack(
        [
            tokenize_v3(x, vocab)
            for x in train_df["text"]
        ]
    )

    dev_ids_np = np.stack(
        [
            tokenize_v3(x, vocab)
            for x in dev_df["text"]
        ]
    )

    pad_id = token_id(
        vocab,
        ["<pad>", "[PAD]"],
        0,
    )

    train_mask_np = (
        train_ids_np != pad_id
    ).astype(np.int64)

    dev_mask_np = (
        dev_ids_np != pad_id
    ).astype(np.int64)

    train_ids = torch.tensor(
        train_ids_np,
        dtype=torch.long,
    )

    train_mask = torch.tensor(
        train_mask_np,
        dtype=torch.long,
    )

    dev_ids = torch.tensor(
        dev_ids_np,
        dtype=torch.long,
    )

    dev_mask = torch.tensor(
        dev_mask_np,
        dtype=torch.long,
    )

    y_train = np.asarray(
        [
            LABELS.index(x)
            for x in train_df["intent"]
        ],
        dtype=np.int64,
    )

    y_dev = np.asarray(
        [
            LABELS.index(x)
            for x in dev_df["intent"]
        ],
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Load V3
    # --------------------------------------------------------

    v3 = load_v3(
        Path(args.v3),
        vocab,
        device,
    )

    # Frozen copy = original V3 anchor.
    anchor = copy.deepcopy(v3).eval()

    for p in anchor.parameters():
        p.requires_grad = False

    # --------------------------------------------------------
    # Load E5 teacher
    # --------------------------------------------------------

    print("\nLoading multilingual-e5-small...")

    tokenizer = AutoTokenizer.from_pretrained(
        E5_NAME
    )

    teacher = AutoModel.from_pretrained(
        E5_NAME
    ).to(device)

    teacher.eval()

    for p in teacher.parameters():
        p.requires_grad = False

    # --------------------------------------------------------
    # Cache E5 embeddings
    # --------------------------------------------------------

    print("\nCaching E5 teacher embeddings...")

    train_e5 = e5_embed(
        tokenizer,
        teacher,
        train_df["text"].tolist(),
        device,
    )

    dev_e5 = e5_embed(
        tokenizer,
        teacher,
        dev_df["text"].tolist(),
        device,
    )

    prototypes = build_teacher_prototypes(
        train_e5,
        train_df["intent"].tolist(),
    )

    # Teacher semantic logits:
    # cosine similarity between utterance and intent prototype.
    train_teacher_logits = (
        train_e5 @ prototypes.T
    )

    dev_teacher_logits = (
        dev_e5 @ prototypes.T
    )

    # Temperature controls softness.
    teacher_temperature = 0.07

    train_teacher_probs = F.softmax(
        train_teacher_logits
        /
        teacher_temperature,
        dim=-1,
    )

    dev_teacher_probs = F.softmax(
        dev_teacher_logits
        /
        teacher_temperature,
        dim=-1,
    )

    # --------------------------------------------------------
    # Training setup
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        v3.parameters(),
        lr=2e-5,
        weight_decay=0.01,
    )

    # Conservative weights:
    # supervised CE dominates.
    # E5 is auxiliary only.
    #
    # This is intentionally much smaller than a normal
    # distillation loss to prevent V3 semantic boundaries
    # from being overwritten.
    alpha_e5 = 0.08
    alpha_anchor = 0.12

    batch_size = 64

    best_state = None
    best_score = -1.0
    best_epoch = -1

    print("\nHYBRID TRAINING")
    print("CE weight       : 1.00")
    print("E5 auxiliary    :", alpha_e5)
    print("V3 anchor       :", alpha_anchor)
    print("Learning rate   : 2e-5")
    print("E5 runtime      : NO")
    print("V3 architecture : PRESERVED")

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        v3.train()

        perm = torch.randperm(
            len(train_ids)
        )

        total_loss = 0.0
        total_n = 0

        for start in range(
            0,
            len(perm),
            batch_size,
        ):
            idx = perm[
                start:start + batch_size
            ]

            ids = train_ids[idx].to(device)
            mask = train_mask[idx].to(device)

            target = torch.tensor(
                y_train[idx.cpu().numpy()],
                dtype=torch.long,
                device=device,
            )

            teacher_probs = (
                train_teacher_probs[idx]
                .to(device)
            )

            # Original V3 logits.
            with torch.no_grad():
                _, anchor_logits = anchor(
                    ids,
                    mask,
                )

            _, student_logits = v3(
                ids,
                mask,
            )

            # 1. Primary supervised objective.
            ce = F.cross_entropy(
                student_logits,
                target,
            )

            # 2. E5 auxiliary semantic objective.
            student_log_probs = F.log_softmax(
                student_logits
                /
                1.0,
                dim=-1,
            )

            e5_soft = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction="batchmean",
            )

            # 3. V3 anchor objective.
            #
            # Prevents the hybrid training from moving too
            # far away from the already-strong V3 decision
            # boundaries.
            anchor_probs = F.softmax(
                anchor_logits.detach(),
                dim=-1,
            )

            anchor_loss = F.kl_div(
                F.log_softmax(
                    student_logits,
                    dim=-1,
                ),
                anchor_probs,
                reduction="batchmean",
            )

            loss = (
                ce
                +
                alpha_e5 * e5_soft
                +
                alpha_anchor * anchor_loss
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                v3.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_loss += (
                float(loss.item())
                *
                len(idx)
            )

            total_n += len(idx)

        train_loss = (
            total_loss
            /
            max(total_n, 1)
        )

        dev_acc, dev_f1, _ = evaluate(
            v3,
            dev_ids,
            dev_mask,
            y_dev,
            device,
        )

        # Score F1 more strongly than accuracy.
        score = (
            0.5 * dev_acc
            +
            0.5 * dev_f1
        )

        print(
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.4f} | "
            f"val={dev_acc*100:.2f}% | "
            f"valF1={dev_f1*100:.2f}%"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(
                v3.state_dict()
            )

    # --------------------------------------------------------
    # Restore best
    # --------------------------------------------------------

    if best_state is None:
        raise RuntimeError(
            "No best checkpoint was produced."
        )

    v3.load_state_dict(
        best_state,
        strict=True,
    )

    dev_acc, dev_f1, dev_pred = evaluate(
        v3,
        dev_ids,
        dev_mask,
        y_dev,
        device,
    )

    print(
        "\n" + "=" * 70
    )
    print("FINAL HYBRID VALIDATION")
    print("=" * 70)

    print(
        f"Validation accuracy : {dev_acc*100:.2f}%"
    )

    print(
        f"Validation Macro F1 : {dev_f1*100:.2f}%"
    )

    print(
        classification_report(
            y_dev,
            dev_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=LABELS,
            digits=4,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_ckpt = (
        OUT_DIR
        / "v3_e5_hybrid_fp32.pt"
    )

    torch.save(
        v3.state_dict(),
        out_ckpt,
    )

    manifest = {
        "model": "V3 + E5-small auxiliary hybrid",
        "runtime_model": "V3 architecture only",
        "e5_runtime": False,
        "e5_teacher": E5_NAME,
        "v3_checkpoint": str(args.v3),
        "training_data": str(data_path),
        "locked_unseen_used": False,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "alpha_e5": alpha_e5,
        "alpha_v3_anchor": alpha_anchor,
        "learning_rate": 2e-5,
        "validation_accuracy": dev_acc,
        "validation_macro_f1": dev_f1,
        "onnx_exported": False,
        "int8_exported": False,
    }

    (
        OUT_DIR
        / "v3_e5_hybrid_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSaved checkpoint:")
    print(out_ckpt)

    print("\nIMPORTANT:")
    print("V3 architecture preserved.")
    print("E5 is training-time only.")
    print("E5 is NOT part of runtime inference.")
    print("595-row unseen set was NOT used.")
    print("ONNX export: NO")
    print("INT8: NO")

    print(
        "\nNEXT: benchmark this checkpoint "
        "against the locked V3 on the untouched 595 rows."
    )


if __name__ == "__main__":
    main()
