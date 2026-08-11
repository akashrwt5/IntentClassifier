#!/usr/bin/env python3
"""
E5-small-v2 TEACHER -> Tiny 57-INTENT STUDENT DISTILLATION
===========================================================

Goal:
    Use the already-trained E5-small-v2 classifier as a teacher and
    distill its knowledge into the existing TinySemanticStudent.

IMPORTANT:
    - Locked 57-intent test is NEVER read.
    - No quantization.
    - No ONNX.
    - No synthetic text.
    - No labels are changed.
    - Student architecture is preserved.
    - Existing V2.1 student checkpoint is used as initialization.
    - Validation is created only from train.csv.
    - Early stopping uses validation Macro F1.

Teacher:
    intfloat/e5-small-v2 + saved Logistic Regression classifier

Student:
    Existing V3/V2.1 TinySemanticStudent
    vocab=895
    embedding=64
    layers=2
    heads=4
    FFN=128
    max_len=24
    classes=57

Loss:
    hard CE + temperature-scaled KL distillation

Recommended first run:
    alpha=0.70 KL
    beta=0.30 CE
    temperature=2.0
"""

from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import SentenceTransformer
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/Users/shuklam/IntentClassifier/semantic_project/57semanitc"
)

TRAIN_CSV = ROOT / "train.csv"

TEACHER_DIR = (
    ROOT / "v3_57intent_e5_small_v2"
)

TEACHER_CLASSIFIER = (
    TEACHER_DIR / "e5_logistic_classifier.joblib"
)

TEACHER_LABEL_MAP = (
    TEACHER_DIR / "label_map.json"
)

VOCAB_PATH = ROOT / "vocab.json"
LABELS_PATH = ROOT / "labels.json"

# Use the strongest tiny student already selected in the project.
# Change only if your local path differs.
STUDENT_INIT = (
    ROOT
    / "v3_57intent_v2_1_controlled"
    / "student_v3_57intent_v2_1_best_fp32.pt"
)

OUT_DIR = (
    ROOT / "v3_57intent_e5_distilled_v1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BEST_CHECKPOINT = (
    OUT_DIR / "student_e5_distilled_v1_best_fp32.pt"
)

HISTORY_PATH = (
    OUT_DIR / "training_history.csv"
)

REPORT_PATH = (
    OUT_DIR / "validation_report.txt"
)

MANIFEST_PATH = (
    OUT_DIR / "training_manifest.json"
)

TEACHER_CACHE = (
    OUT_DIR / "teacher_probabilities.npy"
)


# ============================================================
# CONTRACT
# ============================================================

SEED = 42

BATCH_SIZE = 64
EPOCHS = 30
PATIENCE = 6

LR = 2e-5
WEIGHT_DECAY = 0.01
DROPOUT = 0.10

VAL_SIZE = 0.15

TEMPERATURE = 2.0

# Total loss = ALPHA * KD + BETA * CE
ALPHA = 0.70
BETA = 0.30

VOCAB_SIZE = 895
EMBED_DIM = 64
HEADS = 4
LAYERS = 2
FF_DIM = 128
MAX_LEN = 24
NUM_CLASSES = 57


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# VOCAB / LABELS
# ============================================================

def load_vocab(path):

    obj = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if "token_to_id" in obj:
        return obj["token_to_id"]

    if (
        "vocab" in obj
        and isinstance(obj["vocab"], dict)
    ):
        return obj["vocab"]

    if all(
        isinstance(v, int)
        for v in obj.values()
    ):
        return obj

    raise RuntimeError(
        "Unsupported vocab.json format."
    )


def load_labels(path):

    obj = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if isinstance(obj, list):
        return [str(x) for x in obj]

    if (
        "labels" in obj
        and isinstance(obj["labels"], list)
    ):
        return [str(x) for x in obj["labels"]]

    if (
        "id_to_label" in obj
        and isinstance(obj["id_to_label"], dict)
    ):
        pairs = sorted(
            (
                int(k),
                str(v)
            )
            for k, v
            in obj["id_to_label"].items()
        )

        return [
            v for _, v in pairs
        ]

    if (
        "label_to_id" in obj
        and isinstance(obj["label_to_id"], dict)
    ):
        pairs = sorted(
            (
                int(v),
                str(k)
            )
            for k, v
            in obj["label_to_id"].items()
        )

        return [
            v for _, v in pairs
        ]

    raise RuntimeError(
        "Unsupported labels.json format."
    )


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(
    text,
    vocab,
):

    ids = []

    for token in (
        str(text)
        .lower()
        .split()
    ):

        token = token.strip(
            ".,!?;:\"'()[]{}"
        )

        if token:

            ids.append(
                int(
                    vocab.get(
                        token,
                        1,
                    )
                )
            )

    ids = ids[:MAX_LEN]

    if len(ids) < MAX_LEN:

        ids += [
            0
        ] * (
            MAX_LEN - len(ids)
        )

    return ids


# ============================================================
# STUDENT MODEL
# ============================================================

class V3Student57(nn.Module):

    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
        heads=HEADS,
        layers=LAYERS,
        ff_dim=FF_DIM,
        max_len=MAX_LEN,
        dropout=DROPOUT,
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0,
        )

        self.position = nn.Embedding(
            max_len,
            embed_dim,
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
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
            embed_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                embed_dim,
                embed_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                embed_dim,
                num_classes,
            ),
        )

    def forward(self, x):

        padding_mask = x.eq(0)

        pos = torch.arange(
            x.size(1),
            device=x.device,
        ).unsqueeze(0)

        h = (
            self.embedding(x)
            + self.position(pos)
        )

        h = self.encoder(
            h,
            src_key_padding_mask=padding_mask,
        )

        valid = (
            (~padding_mask)
            .unsqueeze(-1)
            .float()
        )

        pooled = (
            (h * valid).sum(dim=1)
            /
            valid.sum(dim=1).clamp(
                min=1.0
            )
        )

        return self.classifier(
            self.norm(pooled)
        )


# ============================================================
# DATASET
# ============================================================

class DistillDataset(Dataset):

    def __init__(
        self,
        input_ids,
        labels,
        teacher_probs,
    ):

        self.input_ids = torch.tensor(
            input_ids,
            dtype=torch.long,
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.long,
        )

        self.teacher_probs = torch.tensor(
            teacher_probs,
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        return (
            self.input_ids[idx],
            self.labels[idx],
            self.teacher_probs[idx],
        )


# ============================================================
# CHECKPOINT LOADER
# ============================================================

def load_checkpoint_state(path):

    obj = torch.load(
        path,
        map_location="cpu",
    )

    if isinstance(obj, dict):

        if "model_state_dict" in obj:
            obj = obj["model_state_dict"]

        elif "state_dict" in obj:
            obj = obj["state_dict"]

    if not isinstance(obj, dict):

        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    cleaned = {}

    for key, value in obj.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned[key] = value

    return cleaned


# ============================================================
# MAIN
# ============================================================

def main():

    seed_everything(SEED)

    print("=" * 78)
    print("E5 TEACHER -> TINY 57-INTENT STUDENT DISTILLATION")
    print("=" * 78)

    # --------------------------------------------------------
    # Verify files
    # --------------------------------------------------------

    required = [
        TRAIN_CSV,
        TEACHER_CLASSIFIER,
        TEACHER_LABEL_MAP,
        VOCAB_PATH,
        LABELS_PATH,
        STUDENT_INIT,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"\nRequired file not found:\n{path}\n"
            )

    # --------------------------------------------------------
    # Load labels/vocab
    # --------------------------------------------------------

    vocab = load_vocab(
        VOCAB_PATH
    )

    labels = load_labels(
        LABELS_PATH
    )

    if len(vocab) != VOCAB_SIZE:

        raise RuntimeError(
            f"Expected vocab size {VOCAB_SIZE}, "
            f"got {len(vocab)}"
        )

    if len(labels) != NUM_CLASSES:

        raise RuntimeError(
            f"Expected {NUM_CLASSES} labels, "
            f"got {len(labels)}"
        )

    label_to_id = {
        label: i
        for i, label
        in enumerate(labels)
    }

    print()
    print(
        f"Vocab size : {len(vocab)}"
    )

    print(
        f"Classes    : {len(labels)}"
    )

    # --------------------------------------------------------
    # Load train.csv
    # --------------------------------------------------------

    df = pd.read_csv(
        TRAIN_CSV
    )

    if (
        "text" not in df.columns
        or "intent" not in df.columns
    ):

        raise RuntimeError(
            f"train.csv must contain "
            f"text + intent columns.\n"
            f"Found: {list(df.columns)}"
        )

    df = df[
        ["text", "intent"]
    ].dropna().copy()

    df["text"] = (
        df["text"]
        .astype(str)
        .str.strip()
    )

    df["intent"] = (
        df["intent"]
        .astype(str)
        .str.strip()
    )

    df = df[
        (df["text"] != "")
        &
        (df["intent"] != "")
    ].reset_index(drop=True)

    unknown = sorted(
        set(df["intent"])
        -
        set(labels)
    )

    if unknown:

        raise RuntimeError(
            "train.csv contains labels not "
            "present in labels.json:\n"
            + "\n".join(unknown)
        )

    print()
    print(
        f"Total train.csv rows: {len(df)}"
    )

    # --------------------------------------------------------
    # Train/validation split
    # --------------------------------------------------------

    train_df, val_df = (
        train_test_split(
            df,
            test_size=VAL_SIZE,
            random_state=SEED,
            stratify=df["intent"],
        )
    )

    # IMPORTANT:
    # Preserve original dataframe indices so teacher probabilities,
    # tokenized inputs, and labels remain aligned with the exact text.
    train_indices = train_df.index.to_numpy()
    val_indices = val_df.index.to_numpy()

    train_df = train_df.reset_index(
        drop=True
    )

    val_df = val_df.reset_index(
        drop=True
    )

    print(
        f"Train rows: {len(train_df)}"
    )

    print(
        f"Val rows  : {len(val_df)}"
    )

    # --------------------------------------------------------
    # E5 teacher
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("LOADING E5 TEACHER")
    print("=" * 78)

    teacher_encoder = (
        SentenceTransformer(
            "intfloat/e5-small-v2"
        )
    )

    teacher_classifier = joblib.load(
        TEACHER_CLASSIFIER
    )

    # --------------------------------------------------------
    # Teacher probabilities
    # --------------------------------------------------------

    all_texts = df["text"].tolist()

    print()
    print(
        "Generating E5 teacher probabilities..."
    )

    start = time.perf_counter()

    e5_texts = [
        "query: " + x
        for x in all_texts
    ]

    embeddings = (
        teacher_encoder.encode(
            e5_texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    )

    teacher_probs = (
        teacher_classifier.predict_proba(
            embeddings
        )
    )

    teacher_time = (
        time.perf_counter() - start
    )

    if teacher_probs.shape[1] != NUM_CLASSES:

        raise RuntimeError(
            f"Teacher returned "
            f"{teacher_probs.shape[1]} classes, "
            f"expected {NUM_CLASSES}"
        )

    print(
        f"Teacher generation time: "
        f"{teacher_time:.3f} sec"
    )

    # Save cache for reproducibility.
    np.save(
        TEACHER_CACHE,
        teacher_probs.astype(
            np.float32
        ),
    )

    # --------------------------------------------------------
    # Tokenize student inputs
    # --------------------------------------------------------

    print()
    print(
        "Tokenizing Tiny Student inputs..."
    )

    X = np.asarray(
        [
            tokenize(
                x,
                vocab,
            )
            for x in df["text"]
        ],
        dtype=np.int64,
    )

    y = np.asarray(
        [
            label_to_id[x]
            for x in df["intent"]
        ],
        dtype=np.int64,
    )

    X_train = X[train_indices]
    X_val = X[val_indices]

    y_train = y[train_indices]
    y_val = y[val_indices]

    p_train = teacher_probs[
        train_indices
    ]

    p_val = teacher_probs[
        val_indices
    ]

    # Hard guards: every class must be represented in validation
    # for a valid 57-intent macro-F1 benchmark.
    missing_val_labels = sorted(
        set(labels)
        - set(val_df["intent"].tolist())
    )

    if missing_val_labels:
        raise RuntimeError(
            "Validation split is missing intents: "
            + ", ".join(missing_val_labels)
        )

    if len(X_train) != len(p_train) or len(X_val) != len(p_val):
        raise RuntimeError(
            "Teacher/student split alignment failure."
        )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_ds = DistillDataset(
        X_train,
        y_train,
        p_train,
    )

    val_ds = DistillDataset(
        X_val,
        y_val,
        p_val,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # --------------------------------------------------------
    # Student
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("LOADING TINY STUDENT")
    print("=" * 78)

    student = V3Student57()

    state = load_checkpoint_state(
        STUDENT_INIT
    )

    student.load_state_dict(
        state,
        strict=True,
    )

    print(
        "Initial student checkpoint: PASS"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    student.to(device)

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Optimizer / loss
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    ce_loss = nn.CrossEntropyLoss()

    best_macro_f1 = -1.0
    best_epoch = -1
    patience_count = 0

    history = []

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("DISTILLATION TRAINING")
    print("=" * 78)

    print(
        f"Temperature : {TEMPERATURE}"
    )

    print(
        f"KD weight   : {ALPHA}"
    )

    print(
        f"CE weight   : {BETA}"
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        student.train()

        train_loss_sum = 0.0
        train_count = 0

        for (
            input_ids,
            hard_labels,
            teacher_p,
        ) in train_loader:

            input_ids = (
                input_ids.to(device)
            )

            hard_labels = (
                hard_labels.to(device)
            )

            teacher_p = (
                teacher_p.to(device)
            )

            optimizer.zero_grad()

            student_logits = student(
                input_ids
            )

            # Hard-label CE.
            loss_ce = ce_loss(
                student_logits,
                hard_labels,
            )

            # Teacher distribution.
            teacher_log_p = torch.log(
                teacher_p.clamp(
                    min=1e-8
                )
            )

            # Student softened distribution.
            student_log_p = (
                torch.log_softmax(
                    student_logits
                    / TEMPERATURE,
                    dim=1,
                )
            )

            # KL(student || teacher)
            # F.kl_div expects input log-probs
            # and target probabilities.
            loss_kd = nn.functional.kl_div(
                student_log_p,
                teacher_p,
                reduction="batchmean",
            ) * (
                TEMPERATURE ** 2
            )

            loss = (
                ALPHA * loss_kd
                +
                BETA * loss_ce
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                student.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            batch_n = (
                input_ids.size(0)
            )

            train_loss_sum += (
                float(loss.item())
                * batch_n
            )

            train_count += batch_n

        train_loss = (
            train_loss_sum
            /
            max(train_count, 1)
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        student.eval()

        val_preds = []
        val_true = []

        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():

            for (
                input_ids,
                hard_labels,
                teacher_p,
            ) in val_loader:

                input_ids = (
                    input_ids.to(device)
                )

                hard_labels = (
                    hard_labels.to(device)
                )

                teacher_p = (
                    teacher_p.to(device)
                )

                logits = student(
                    input_ids
                )

                loss_ce = ce_loss(
                    logits,
                    hard_labels,
                )

                student_log_p = (
                    torch.log_softmax(
                        logits
                        / TEMPERATURE,
                        dim=1,
                    )
                )

                loss_kd = nn.functional.kl_div(
                    student_log_p,
                    teacher_p,
                    reduction="batchmean",
                ) * (
                    TEMPERATURE ** 2
                )

                loss = (
                    ALPHA * loss_kd
                    +
                    BETA * loss_ce
                )

                batch_n = (
                    input_ids.size(0)
                )

                val_loss_sum += (
                    float(loss.item())
                    * batch_n
                )

                val_count += batch_n

                pred = (
                    logits.argmax(
                        dim=1
                    )
                    .cpu()
                    .numpy()
                )

                true = (
                    hard_labels
                    .cpu()
                    .numpy()
                )

                val_preds.extend(
                    pred.tolist()
                )

                val_true.extend(
                    true.tolist()
                )

        val_loss = (
            val_loss_sum
            /
            max(val_count, 1)
        )

        val_acc = accuracy_score(
            val_true,
            val_preds,
        )

        val_f1 = f1_score(
            val_true,
            val_preds,
            average="macro",
            zero_division=0,
        )

        val_weighted_f1 = f1_score(
            val_true,
            val_preds,
            average="weighted",
            zero_division=0,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.4f} | "
            f"val={val_acc * 100:.2f}% | "
            f"valF1={val_f1 * 100:.2f}%"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
                "val_weighted_f1":
                    val_weighted_f1,
            }
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if val_f1 > best_macro_f1:

            best_macro_f1 = val_f1
            best_epoch = epoch
            patience_count = 0

            torch.save(
                student.state_dict(),
                BEST_CHECKPOINT,
            )

            print(
                "  -> BEST CHECKPOINT SAVED"
            )

        else:

            patience_count += 1

            if (
                patience_count
                >= PATIENCE
            ):

                print(
                    "Early stopping."
                )

                break

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        HISTORY_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Load best student
    # --------------------------------------------------------

    best_state = torch.load(
        BEST_CHECKPOINT,
        map_location="cpu",
    )

    student.load_state_dict(
        best_state,
        strict=True,
    )

    student.to(device)
    student.eval()

    # --------------------------------------------------------
    # Final validation report
    # --------------------------------------------------------

    final_preds = []
    final_true = []

    with torch.no_grad():

        for (
            input_ids,
            hard_labels,
            _teacher_p,
        ) in val_loader:

            input_ids = (
                input_ids.to(device)
            )

            logits = student(
                input_ids
            )

            pred = (
                logits.argmax(
                    dim=1
                )
                .cpu()
                .numpy()
            )

            final_preds.extend(
                pred.tolist()
            )

            final_true.extend(
                hard_labels.numpy().tolist()
            )

    final_acc = accuracy_score(
        final_true,
        final_preds,
    )

    final_macro_f1 = f1_score(
        final_true,
        final_preds,
        average="macro",
        zero_division=0,
    )

    final_weighted_f1 = f1_score(
        final_true,
        final_preds,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        final_true,
        final_preds,
        labels=list(
            range(NUM_CLASSES)
        ),
        target_names=labels,
        zero_division=0,
    )

    print()
    print("=" * 78)
    print("E5 DISTILLED STUDENT VALIDATION")
    print("=" * 78)

    print(
        f"Accuracy   : "
        f"{final_acc * 100:.4f}%"
    )

    print(
        f"Macro F1   : "
        f"{final_macro_f1 * 100:.4f}%"
    )

    print(
        f"Weighted F1: "
        f"{final_weighted_f1 * 100:.4f}%"
    )

    print(
        f"Best epoch : {best_epoch}"
    )

    print()
    print(report)

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    REPORT_PATH.write_text(
        (
            "E5 DISTILLED STUDENT "
            "VALIDATION\n"
            + "=" * 78
            + "\n\n"
            + f"Accuracy: "
              f"{final_acc * 100:.4f}%\n"
            + f"Macro F1: "
              f"{final_macro_f1 * 100:.4f}%\n"
            + f"Weighted F1: "
              f"{final_weighted_f1 * 100:.4f}%\n"
            + f"Best epoch: "
              f"{best_epoch}\n\n"
            + "Classification report:\n\n"
            + report
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = {
        "status":
            "E5 TEACHER DISTILLATION COMPLETE",

        "teacher":
            "intfloat/e5-small-v2",

        "teacher_classifier":
            str(TEACHER_CLASSIFIER),

        "student_initialization":
            str(STUDENT_INIT),

        "student_architecture": {
            "vocab_size": VOCAB_SIZE,
            "embed_dim": EMBED_DIM,
            "heads": HEADS,
            "layers": LAYERS,
            "ff_dim": FF_DIM,
            "max_len": MAX_LEN,
            "classes": NUM_CLASSES,
        },

        "dataset":
            str(TRAIN_CSV),

        "total_rows":
            int(len(df)),

        "train_rows":
            int(len(train_df)),

        "validation_rows":
            int(len(val_df)),

        "split_alignment_fix":
            "original dataframe indices preserved",

        "temperature":
            TEMPERATURE,

        "kd_weight":
            ALPHA,

        "ce_weight":
            BETA,

        "learning_rate":
            LR,

        "best_epoch":
            best_epoch,

        "validation_accuracy":
            float(final_acc),

        "validation_macro_f1":
            float(final_macro_f1),

        "validation_weighted_f1":
            float(final_weighted_f1),

        "locked_test_used":
            False,

        "quantization":
            False,

        "onnx":
            False,

        "synthetic_text":
            False,

        "labels_changed":
            False,
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("STATUS: E5 TEACHER DISTILLATION COMPLETE")
    print("=" * 78)

    print()
    print("Saved checkpoint:")
    print(BEST_CHECKPOINT)

    print()
    print("Saved history:")
    print(HISTORY_PATH)

    print()
    print("Saved report:")
    print(REPORT_PATH)

    print()
    print("Saved manifest:")
    print(MANIFEST_PATH)

    print()
    print("IMPORTANT:")
    print("Locked 57-intent test: NOT READ")
    print("Quantization: NO")
    print("ONNX: NO")
    print("Synthetic text: NO")
    print("Labels changed: NO")


if __name__ == "__main__":
    main()
