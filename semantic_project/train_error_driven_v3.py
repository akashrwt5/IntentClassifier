#!/usr/bin/env python3
"""
train_error_driven_v3.py

V3 error-driven fine-tuning starting from the LOCKED V2 FP32 candidate.

Purpose:
  - Improve critical intent boundaries using the 210 hard-negative examples.
  - Preserve the V2 representation and architecture.
  - Do NOT modify the frozen V2 INT8 model.
  - Do NOT use the 595-row unseen test for training.

Expected:
  tiny_semantic_student_v2_balanced/
      student_fp32.pt
      vocab.json
      labels.json   (or labels.json elsewhere in the directory)
  production_calibration_v2/
      production_hard_negative.csv
      production_contrastive_pairs.csv
  Existing labeled training dataset CSV with text + intent columns.

Output:
  tiny_semantic_student_v3_error_driven/
      student_v3_fp32.pt
      labels.json
      training_manifest.json
      hard_negative_results.csv

IMPORTANT:
This script is intentionally conservative:
  - starts from V2 FP32
  - mixes original data with hard negatives
  - uses a lower learning rate
  - uses class-balanced sampling
  - evaluates targeted hard negatives
  - does not export ONNX/INT8 automatically
"""

from pathlib import Path
import json
import random
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "tiny_semantic_student_v2_balanced"
HARD = ROOT / "production_calibration_v2" / "production_hard_negative.csv"
CONTRAST = ROOT / "production_calibration_v2" / "production_contrastive_pairs.csv"
OUT = ROOT / "tiny_semantic_student_v3_error_driven"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260809
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

# V2 architecture from the locked candidate.
ED = 64
NH = 4
FF = 128
NL = 2
ML = 24
DROPOUT = 0.10
BATCH = 64
EPOCHS = 8
LR = 2e-4
WEIGHT_DECAY = 1e-4

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
L2I = {x: i for i, x in enumerate(LABELS)}
I2L = {i: x for x, i in L2I.items()}

def clean(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())

def locate_json(name):
    candidates = [
        BASE / name,
        ROOT / name,
        ROOT / "tiny_semantic_student_v2_int8" / name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def load_vocab():
    p = locate_json("vocab.json")
    if p is None:
        raise FileNotFoundError("vocab.json not found for V2 tokenizer.")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw.get("model"), dict) and isinstance(raw["model"].get("vocab"), dict):
        vocab = raw["model"]["vocab"]
    elif isinstance(raw.get("vocab"), dict):
        vocab = raw["vocab"]
    else:
        vocab = raw
    return {str(k): int(v) for k, v in vocab.items()}, p

def load_labels():
    p = locate_json("labels.json")
    if p is None:
        return LABELS, None
    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        labels = raw
    elif isinstance(raw, dict):
        labels = raw.get("labels", raw.get("classes", LABELS))
    else:
        labels = LABELS
    if set(labels) != set(LABELS):
        raise ValueError("V2 labels.json does not match the locked 11-intent set.")
    return labels, p

vocab, vocab_path = load_vocab()
labels, labels_path = load_labels()

PAD = int(vocab.get("<pad>", vocab.get("[PAD]", 0)))
UNK = int(vocab.get("<unk>", vocab.get("[UNK]", 1)))
CLS = vocab.get("<cls>", vocab.get("[CLS]", None))
SEP = vocab.get("<sep>", vocab.get("[SEP]", None))

def tokenize(text):
    text = clean(text)
    text = re.sub(r"([.!?,;:()'])", r" \1 ", text)
    words = text.split()
    ids = []

    if CLS is not None:
        ids.append(int(CLS))

    for word in words:
        if word in vocab:
            ids.append(int(vocab[word]))
            continue

        pos = 0
        pieces = []
        while pos < len(word):
            best = None
            for end in range(len(word), pos, -1):
                candidate = word[pos:end]
                for c in (candidate, "##" + candidate):
                    if c in vocab:
                        best = c
                        break
                if best is not None:
                    break

            if best is None:
                pieces = [UNK]
                break

            pieces.append(int(vocab[best]))
            pos += len(best.replace("##", ""))

        ids.extend(pieces)

    if SEP is not None:
        ids.append(int(SEP))

    ids = ids[:ML]
    ids += [PAD] * (ML - len(ids))
    return ids

class TextDataset(Dataset):
    def __init__(self, df):
        self.x = torch.tensor(
            [tokenize(x) for x in df.text],
            dtype=torch.long
        )
        self.y = torch.tensor(
            [L2I[x] for x in df.intent],
            dtype=torch.long
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(
            len(vocab), ED, padding_idx=PAD
        )
        self.position = nn.Embedding(ML, ED)

        layer = nn.TransformerEncoderLayer(
            ED,
            NH,
            FF,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, NL)
        self.norm = nn.LayerNorm(ED)
        self.classifier = nn.Sequential(
            nn.Linear(ED, ED),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(ED, len(LABELS)),
        )

    def forward(self, x):
        mask = x.eq(PAD)
        pos = torch.arange(
            x.size(1),
            device=x.device
        ).unsqueeze(0)

        h = self.encoder(
            self.embedding(x) + self.position(pos),
            src_key_padding_mask=mask,
        )

        valid = (~mask).unsqueeze(-1).float()
        h = (h * valid).sum(1) / valid.sum(1).clamp(min=1)

        return self.classifier(self.norm(h))

# ---------------------------------------------------------------------
# Find original labeled data.
# ---------------------------------------------------------------------

CSV_CANDIDATES = [
    ROOT / "balanced_dataset.csv",
    ROOT / "semantic_dataset.csv",
    ROOT / "dataset.csv",
    ROOT / "fine_tuned_test_predictions.csv",
]

source = None
source_path = None

for p in CSV_CANDIDATES:
    if not p.exists():
        continue
    try:
        df = pd.read_csv(p)
    except Exception:
        continue

    if "text" not in df.columns:
        continue

    col = next(
        (
            c for c in
            ["intent", "label", "expected_intent", "true_intent"]
            if c in df.columns
        ),
        None
    )

    if col:
        source = df.rename(columns={col: "intent"})[
            ["text", "intent"]
        ].dropna()
        source_path = p
        break

if source is None:
    raise FileNotFoundError(
        "Could not locate original labeled training dataset."
    )

source["text"] = source["text"].map(clean)
source = source[source.intent.isin(LABELS)]
source = source.drop_duplicates("text").reset_index(drop=True)

# IMPORTANT:
# Do not use the 595-row unseen stress test.
stress = ROOT / "unseen_semantic_stress_test.csv"
if stress.exists():
    try:
        stress_df = pd.read_csv(stress)
        stress_texts = set(stress_df["text"].astype(str).map(clean))
        source = source[~source.text.isin(stress_texts)]
    except Exception:
        pass

# ---------------------------------------------------------------------
# Load hard negatives.
# ---------------------------------------------------------------------

if not HARD.exists():
    raise FileNotFoundError(
        f"Hard-negative dataset missing: {HARD}"
    )

hard = pd.read_csv(HARD).dropna(subset=["text", "intent"])
hard["text"] = hard["text"].map(clean)
hard = hard[hard.intent.isin(LABELS)]
hard = hard.drop_duplicates("text").reset_index(drop=True)

# Give hard negatives higher sampling probability.
hard["is_hard"] = True
source["is_hard"] = False

# Remove exact duplicates.
source = source[~source.text.isin(set(hard.text))].copy()

# ---------------------------------------------------------------------
# Build fine-tuning mixture.
#
# Hard negatives are repeated 5x in the training mixture. This is not
# duplication in the stored output; it is intentional sampling emphasis.
# ---------------------------------------------------------------------

hard_repeat = pd.concat([hard] * 5, ignore_index=True)

train_df = pd.concat(
    [
        source[["text", "intent", "is_hard"]],
        hard_repeat[["text", "intent", "is_hard"]],
    ],
    ignore_index=True,
)

# Split only the normal source data. Hard negatives remain available in
# training, but a fixed 20% hard-negative holdout is reserved for targeted
# evaluation.
hard_eval = hard.sample(
    frac=0.20,
    random_state=SEED
).reset_index(drop=True)

hard_train = hard[~hard.text.isin(set(hard_eval.text))].copy()

normal_train, normal_val = train_test_split(
    source,
    test_size=0.10,
    random_state=SEED,
    stratify=source.intent,
)

train_df = pd.concat(
    [
        normal_train[["text", "intent", "is_hard"]],
        pd.concat([hard_train] * 5, ignore_index=True)[
            ["text", "intent", "is_hard"]
        ],
    ],
    ignore_index=True,
)

val_df = normal_val[["text", "intent", "is_hard"]].copy()

print("=" * 78)
print("ERROR-DRIVEN V3 FINE-TUNING")
print("=" * 78)
print("Device:", DEVICE)
print("V2 FP32 baseline:", BASE)
print("Vocab:", len(vocab))
print("Architecture: layers=2 heads=4 embedding=64 FF=128 max_len=24")
print("Original source:", source_path)
print("Normal train rows:", len(normal_train))
print("Normal validation rows:", len(normal_val))
print("Hard-negative total:", len(hard))
print("Hard-negative train:", len(hard_train))
print("Hard-negative eval:", len(hard_eval))
print("Fine-tune rows:", len(train_df))
print()

# ---------------------------------------------------------------------
# Load V2 FP32 weights.
# ---------------------------------------------------------------------

BASE_PT = BASE / "student_fp32.pt"
if not BASE_PT.exists():
    # Search a little more broadly.
    pts = list(BASE.glob("*.pt"))
    if not pts:
        raise FileNotFoundError(
            f"V2 FP32 weights not found under {BASE}"
        )
    BASE_PT = pts[0]

state = torch.load(BASE_PT, map_location="cpu")

model = Model()

# Handle checkpoints that wrap state_dict.
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]

model.load_state_dict(state, strict=True)
model.to(DEVICE)

# ---------------------------------------------------------------------
# Loss and optimizer.
# ---------------------------------------------------------------------

train_ds = TextDataset(train_df)
val_ds = TextDataset(val_df)
hard_ds = TextDataset(hard_eval)

# Balanced weights from normal training distribution.
counts = np.bincount(
    [L2I[x] for x in normal_train.intent],
    minlength=len(LABELS)
)
weights = np.array([
    1.0 / max(1, counts[L2I[x]])
    for x in train_df.intent
], dtype=np.float64)

# Hard negatives get additional sampling weight.
weights *= np.where(
    train_df.is_hard.to_numpy(),
    5.0,
    1.0
)

sampler = WeightedRandomSampler(
    torch.as_tensor(weights, dtype=torch.double),
    num_samples=len(train_df),
    replacement=True,
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH,
    sampler=sampler,
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH,
    shuffle=False,
)
hard_loader = DataLoader(
    hard_ds,
    batch_size=BATCH,
    shuffle=False,
)

loss_fn = nn.CrossEntropyLoss(
    label_smoothing=0.03
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

best_state = None
best_score = -1.0
history = []

def evaluate(loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            p = logits.argmax(-1).cpu().numpy()
            ys.extend(y.numpy().tolist())
            ps.extend(p.tolist())
    acc = accuracy_score(ys, ps)
    f1 = f1_score(
        ys,
        ps,
        average="macro",
        zero_division=0,
    )
    return acc, f1, ys, ps

def targeted_accuracy():
    acc, f1, ys, ps = evaluate(hard_loader)
    return acc, f1

# ---------------------------------------------------------------------
# Fine-tune.
# ---------------------------------------------------------------------

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    steps = 0

    for x, y in train_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )
        optimizer.step()

        total_loss += float(loss.item())
        steps += 1

    val_acc, val_f1, _, _ = evaluate(val_loader)
    hard_acc, hard_f1 = targeted_accuracy()

    # Prioritize targeted safety while retaining normal accuracy.
    score = (
        0.55 * hard_acc
        + 0.30 * val_acc
        + 0.15 * val_f1
    )

    history.append({
        "epoch": epoch,
        "loss": total_loss / max(1, steps),
        "val_accuracy": val_acc,
        "val_macro_f1": val_f1,
        "hard_accuracy": hard_acc,
        "hard_macro_f1": hard_f1,
        "score": score,
    })

    print(
        f"Epoch {epoch:02d} | "
        f"loss={total_loss/max(1,steps):.4f} | "
        f"val={val_acc*100:.2f}% | "
        f"valF1={val_f1*100:.2f}% | "
        f"hard={hard_acc*100:.2f}%"
    )

    if score > best_score:
        best_score = score
        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

if best_state is None:
    raise RuntimeError("No V3 checkpoint was produced.")

model.load_state_dict(best_state)
model.to(DEVICE)

# ---------------------------------------------------------------------
# Final targeted report.
# ---------------------------------------------------------------------

model.eval()
rows = []

with torch.no_grad():
    for _, r in hard_eval.iterrows():
        x = torch.tensor(
            [tokenize(r.text)],
            dtype=torch.long,
            device=DEVICE,
        )
        probs = torch.softmax(model(x), dim=-1)[0]
        order = torch.argsort(probs, descending=True)
        pred_id = int(order[0])
        conf = float(probs[pred_id])
        rows.append({
            "text": r.text,
            "expected": r.intent,
            "predicted": I2L[pred_id],
            "confidence": conf,
            "top2": I2L[int(order[1])],
            "top2_confidence": float(probs[int(order[1])]),
            "correct": I2L[pred_id] == r.intent,
        })

hard_results = pd.DataFrame(rows)
hard_results.to_csv(
    OUT / "hard_negative_results.csv",
    index=False
)

# Normal validation classification report.
val_acc, val_f1, ys, ps = evaluate(val_loader)

print()
print("=" * 78)
print("V3 RESULT")
print("=" * 78)
print("Normal validation accuracy : %.2f%%" % (val_acc * 100))
print("Normal validation Macro F1 : %.2f%%" % (val_f1 * 100))
print("Hard-negative accuracy     : %.2f%%" %
      (hard_results.correct.mean() * 100))
print()
print(classification_report(
    ys,
    ps,
    labels=list(range(len(LABELS))),
    target_names=LABELS,
    digits=4,
    zero_division=0,
))

# ---------------------------------------------------------------------
# Save checkpoint.
# ---------------------------------------------------------------------

out_pt = OUT / "student_v3_fp32.pt"
torch.save(model.state_dict(), out_pt)

(OUT / "labels.json").write_text(
    json.dumps(LABELS, indent=2),
    encoding="utf-8"
)

manifest = {
    "version": "v3_error_driven",
    "base_model": str(BASE_PT),
    "base_model_was_modified": False,
    "device": str(DEVICE),
    "vocab_size": len(vocab),
    "embedding_dim": ED,
    "attention_heads": NH,
    "feed_forward": FF,
    "transformer_layers": NL,
    "max_length": ML,
    "normal_train_rows": len(normal_train),
    "normal_validation_rows": len(normal_val),
    "hard_negative_total": len(hard),
    "hard_negative_train": len(hard_train),
    "hard_negative_eval": len(hard_eval),
    "learning_rate": LR,
    "epochs": EPOCHS,
    "history": history,
    "final_normal_validation_accuracy": val_acc,
    "final_normal_validation_macro_f1": val_f1,
    "final_hard_negative_accuracy": float(hard_results.correct.mean()),
    "unseen_test_used": False,
    "v2_int8_modified": False,
    "next_step": (
        "Benchmark V3 FP32 against locked V2 INT8 on the full "
        "595-sample unseen/contextual/targeted/OOD suite before ONNX export."
    ),
}

(OUT / "training_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8"
)

print("Saved:")
print(out_pt)
print(OUT / "hard_negative_results.csv")
print(OUT / "training_manifest.json")
print()
print("IMPORTANT:")
print("V2 INT8 was NOT modified.")
print("595-row unseen test was NOT used for training.")
print("Do NOT export V3 to ONNX/INT8 yet.")
print("Next: full benchmark V3 FP32 vs locked V2 INT8.")
