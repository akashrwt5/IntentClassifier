#!/usr/bin/env python3
"""
V2 - Current INT8 baseline improvement experiment.

IMPORTANT:
- The CURRENT INT8 model remains the deployment baseline.
- Training starts from the ORIGINAL FP32 checkpoint that produced it.
- V1 is NOT used as the model/weight base.
- V1 is used only as a source of hard/error examples.
- The script never overwrites the current INT8 model.

Goal:
Improve the current baseline while preventing the V1 contextual regression.

Expected gate:
    unseen      >= 94.29%
    contextual  >  90.62%
    OOD         >= 34.38%
    targeted    >= 95%

The script uses the same tokenizer and architecture as tiny_semantic_student_v1.
"""

from pathlib import Path
import json
import random
import re
import shutil

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent

BASE = ROOT / "tiny_semantic_student_v1"
V1_DIR = ROOT / "tiny_semantic_student_error_driven_v1"
OUT = ROOT / "tiny_semantic_student_v2_balanced"

SEED = 42
BATCH_SIZE = 64
EPOCHS = 8

# Small LR: this is a conservative fine-tune of the baseline.
LR = 7e-5
WEIGHT_DECAY = 1e-4

# Hard examples are repeated, but deliberately capped.
HARD_REPEAT = 2

# Promotion reference = locked Current INT8 benchmark.
BASELINE_UNSEEN = 94.29
BASELINE_CONTEXTUAL = 90.62
BASELINE_OOD = 34.38

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("=" * 78)
print("V2 BALANCED CURRENT-BASELINE IMPROVEMENT")
print("=" * 78)
print("Device:", DEVICE)
print("Baseline FP32:", BASE)
print("V1 hard-example source:", V1_DIR)
print("Output:", OUT)


# ============================================================
# REQUIRED BASELINE FILES
# ============================================================

for name in [
    "student_fp32.pt",
    "vocab.json",
    "config.json",
    "intent_labels.txt",
]:
    p = BASE / name
    if not p.exists():
        raise FileNotFoundError(
            f"Missing required baseline file:\n{p}\n"
            "The Current INT8 baseline's original FP32 checkpoint "
            "is required for training."
        )


# ============================================================
# LOAD BASELINE ARTIFACTS
# ============================================================

with open(BASE / "config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

with open(BASE / "vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

with open(BASE / "intent_labels.txt", "r", encoding="utf-8") as f:
    labels = [x.strip() for x in f if x.strip()]

label_to_id = {x: i for i, x in enumerate(labels)}


def get_cfg(*keys, default=None):
    for k in keys:
        if k in config:
            return config[k]
    return default


ED = int(get_cfg("embed_dim", "embedding_dim", default=64))
NH = int(get_cfg("num_heads", "nhead", default=4))
NL = int(get_cfg("num_layers", "layers", default=2))
FF = int(get_cfg("ff_dim", "feedforward_dim", default=128))
ML = int(get_cfg("max_len", "max_length", "sequence_length", default=24))

PAD = int(vocab.get("<PAD>", vocab.get("[PAD]", 0)))
UNK = int(vocab.get("<UNK>", vocab.get("[UNK]", 1)))

print("\nArchitecture:")
print("  vocab:", len(vocab))
print("  embedding:", ED)
print("  layers:", NL)
print("  heads:", NH)
print("  FFN:", FF)
print("  max_len:", ML)
print("  intents:", len(labels))


# ============================================================
# SAME TOKENIZER
# ============================================================

def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def encode(text):
    tokens = clean_text(text).split()

    ids = [
        int(vocab.get(tok, UNK))
        for tok in tokens
    ][:ML]

    ids += [PAD] * (ML - len(ids))
    return ids


# ============================================================
# SAME BASELINE ARCHITECTURE
# ============================================================

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            len(vocab),
            ED,
            padding_idx=PAD
        )

        self.position = nn.Embedding(
            ML,
            ED
        )

        layer = nn.TransformerEncoderLayer(
            d_model=ED,
            nhead=NH,
            dim_feedforward=FF,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            NL
        )

        self.norm = nn.LayerNorm(ED)

        self.classifier = nn.Sequential(
            nn.Linear(ED, ED),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(ED, len(labels))
        )

    def forward(self, x):
        mask = x.eq(PAD)

        pos = torch.arange(
            x.size(1),
            device=x.device
        ).unsqueeze(0)

        h = self.embedding(x) + self.position(pos)

        h = self.encoder(
            h,
            src_key_padding_mask=mask
        )

        valid = (~mask).unsqueeze(-1).float()

        h = (
            h * valid
        ).sum(1) / valid.sum(1).clamp(min=1.0)

        return self.classifier(
            self.norm(h)
        )


model = Model()

state = torch.load(
    BASE / "student_fp32.pt",
    map_location="cpu"
)

# Handle either a raw state_dict or a checkpoint wrapper.
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]

model.load_state_dict(state)
model.to(DEVICE)

print("\nLoaded CURRENT BASELINE FP32 weights.")


# ============================================================
# FIND ORIGINAL TRAINING DATA
# ============================================================

dataset_candidates = [
    ROOT / "semantic_training_v3_hard_negatives.xlsx",
    ROOT / "semantic_training_v3.xlsx",
    ROOT / "semantic_training.xlsx",
    ROOT / "training_data.csv",
    ROOT / "semantic_training.csv",
]

dataset_path = next(
    (p for p in dataset_candidates if p.exists()),
    None
)

if dataset_path is None:
    raise FileNotFoundError(
        "\nOriginal training dataset was not found.\n"
        "V2 intentionally refuses to train only on hard examples because "
        "V1 demonstrated contextual regression.\n"
        "Put the same dataset used to train tiny_semantic_student_v1 "
        "in the semantic_project directory and rerun."
    )

print("Training dataset:", dataset_path)


if dataset_path.suffix.lower() == ".xlsx":
    book = pd.ExcelFile(dataset_path)

    if "dataset" in book.sheet_names:
        df = pd.read_excel(
            dataset_path,
            sheet_name="dataset"
        )
    else:
        df = pd.read_excel(dataset_path)
else:
    df = pd.read_csv(dataset_path)

text_col = "text" if "text" in df.columns else df.columns[0]
intent_col = "intent" if "intent" in df.columns else df.columns[1]

df = df[
    [text_col, intent_col]
].rename(
    columns={
        text_col: "text",
        intent_col: "intent"
    }
)

df["text"] = df["text"].astype(str).str.strip()
df["intent"] = df["intent"].astype(str).str.strip()

df = df[
    (df["text"] != "")
    & df["intent"].isin(labels)
].drop_duplicates(
    ["text", "intent"]
).reset_index(drop=True)

print("Original rows:", len(df))


# ============================================================
# HARD EXAMPLES
# These come from the failures / successful corrections seen
# during the current baseline improvement work.
# ============================================================

HARD = [
    # increase
    ("it's quieter can you make it a little louder",
     "device.volume.increase"),
    ("the sound is too quiet make it louder",
     "device.volume.increase"),
    ("i can barely hear it turn the volume up",
     "device.volume.increase"),
    ("the audio is weak please raise the volume",
     "device.volume.increase"),
    ("it got quieter here can you increase the sound",
     "device.volume.increase"),
    ("make the audio stronger it is difficult to hear",
     "device.volume.increase"),
    ("turn it up a little because i cannot hear",
     "device.volume.increase"),

    # decrease
    ("it's a bit loud here can you make it quieter",
     "device.volume.decrease"),
    ("it's bit loudy here can you make it quieter",
     "device.volume.decrease"),
    ("it's bit loudy here can you make it quietr",
     "device.volume.decrease"),
    ("the sound is too loud turn it down",
     "device.volume.decrease"),
    ("the audio is louder than i need please reduce it",
     "device.volume.decrease"),
    ("turn the sound down a little",
     "device.volume.decrease"),
    ("i want less audio because it is loud",
     "device.volume.decrease"),

    # mute
    ("turn off",
     "device.volume.mute"),
    ("i can still hear it make it completely silent",
     "device.volume.mute"),
    ("i can still hear some sound please mute it",
     "device.volume.mute"),
    ("make everything completely silent",
     "device.volume.mute"),
    ("i don't want any sound at all",
     "device.volume.mute"),
    ("silence the hearing aids completely",
     "device.volume.mute"),

    # unmute
    ("turn the sound back on",
     "device.volume.unmute"),
    ("restore the audio",
     "device.volume.unmute"),
    ("i want to hear again",
     "device.volume.unmute"),
    ("bring the sound back",
     "device.volume.unmute"),
    ("unmute the hearing aids",
     "device.volume.unmute"),

    # reminder / memory boundary
    ("i need to go to airport tomorrow",
     "reminders.task.create"),
    ("i need to go to airport tommorow",
     "reminders.task.create"),
    ("i need to go to airport tomorow",
     "reminders.task.create"),
    ("remind me that i need to go to the airport tomorrow",
     "reminders.task.create"),
    ("set a reminder for tomorrow to go to the airport",
     "reminders.task.create"),

    # streaming boundaries
    ("start streaming",
     "streaming.session.start"),
    ("begin streaming",
     "streaming.session.start"),
    ("stop streaming",
     "streaming.session.stop"),
    ("end the streaming session",
     "streaming.session.stop"),

    # phone
    ("where is my phone",
     "find.phone.locate"),
    ("locate my phone",
     "find.phone.locate"),
    ("find my phone",
     "find.phone.locate"),

    # reminders
    ("show my reminders",
     "help.reminder.show"),
    ("what reminders do i have",
     "help.reminder.show"),
    ("complete my reminder",
     "reminders.task.complete"),
    ("mark the reminder complete",
     "reminders.task.complete"),
]


# ============================================================
# TYPO AUGMENTATION
# ============================================================

TYPO = {
    "louder": ["loudr", "loudar"],
    "quieter": ["quiter", "quietr", "quietter"],
    "tomorrow": ["tommorow", "tomorow", "tomorroe"],
    "volume": ["volum", "volme"],
    "mute": ["mut", "mutt"],
    "unmute": ["unmut", "unmte"],
}


def make_typos(text, intent):
    out = []

    for word, variants in TYPO.items():
        if word not in text.lower():
            continue

        for variant in variants:
            mutated = re.sub(
                re.escape(word),
                variant,
                text,
                count=1,
                flags=re.IGNORECASE
            )
            out.append((mutated, intent))

    return out


hard_rows = list(HARD)

for text, intent in HARD:
    hard_rows.extend(
        make_typos(text, intent)
    )

hard_df = pd.DataFrame(
    hard_rows,
    columns=["text", "intent"]
).drop_duplicates(
    ["text", "intent"]
)

hard_df = hard_df[
    hard_df["intent"].isin(labels)
].reset_index(drop=True)

print("Hard rows:", len(hard_df))


# ============================================================
# BALANCED MIX
#
# V1 used a broad hard-example fine-tune that hurt contextual
# performance. V2 therefore:
# - keeps ALL original training data
# - adds hard rows only twice
# - does not discard original examples
# ============================================================

hard_repeat_df = pd.concat(
    [hard_df] * HARD_REPEAT,
    ignore_index=True
)

train_df = pd.concat(
    [df, hard_repeat_df],
    ignore_index=True
).drop_duplicates(
    ["text", "intent"]
).reset_index(drop=True)

print("Final training rows:", len(train_df))
print(
    "Hard-example fraction:",
    f"{len(hard_repeat_df) / len(train_df) * 100:.2f}%"
)


# ============================================================
# DATASET
# ============================================================

class IntentDataset(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]

        x = torch.tensor(
            encode(row["text"]),
            dtype=torch.long
        )

        y = torch.tensor(
            label_to_id[row["intent"]],
            dtype=torch.long
        )

        return x, y


loader = DataLoader(
    IntentDataset(train_df),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# ============================================================
# TRAIN
# ============================================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.02
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

print("\nFine-tuning CURRENT BASELINE...")
print("This is NOT training from scratch.")
print("V1 weights are NOT loaded.")

for epoch in range(EPOCHS):

    model.train()
    total = 0.0

    for x, y in loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total += float(loss.item())

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Loss: {total / max(1, len(loader)):.4f}"
    )


# ============================================================
# SAVE CANDIDATE
# ============================================================

OUT.mkdir(
    parents=True,
    exist_ok=True
)

candidate = OUT / "student_v2_balanced_fp32.pt"

torch.save(
    model.state_dict(),
    candidate
)

for filename in [
    "vocab.json",
    "config.json",
    "intent_labels.txt",
]:
    shutil.copy2(
        BASE / filename,
        OUT / filename
    )

print("\nCandidate saved:")
print(candidate)


# ============================================================
# PREDICT
# ============================================================

def predict(texts):
    model.eval()

    x = torch.tensor(
        [encode(t) for t in texts],
        dtype=torch.long
    )

    preds = []
    confs = []

    with torch.no_grad():

        for start in range(
            0,
            len(x),
            BATCH_SIZE
        ):

            batch = x[
                start:start + BATCH_SIZE
            ].to(DEVICE)

            probs = torch.softmax(
                model(batch),
                dim=1
            )

            conf, idx = probs.max(dim=1)

            preds.extend(
                labels[int(i)]
                for i in idx.cpu().numpy()
            )

            confs.extend(
                conf.cpu().numpy().tolist()
            )

    return preds, confs


# ============================================================
# TARGETED TEST
# ============================================================

REG = [
    ("it's quieter can you make it a little louder",
     "device.volume.increase"),

    ("i can still hear it make it completely silent",
     "device.volume.mute"),

    ("turn off",
     "device.volume.mute"),

    ("i need to go to airport tomorrow",
     "reminders.task.create"),

    ("i need to go to airport tommorow",
     "reminders.task.create"),

    ("it's bit loudy here can you make it quietr",
     "device.volume.decrease"),

    ("the sound is too loud turn it down",
     "device.volume.decrease"),

    ("the sound is too quiet make it louder",
     "device.volume.increase"),

    ("turn the sound back on",
     "device.volume.unmute"),

    ("where is my phone",
     "find.phone.locate"),

    ("show my reminders",
     "help.reminder.show"),

    ("complete my reminder",
     "reminders.task.complete"),

    ("start streaming",
     "streaming.session.start"),

    ("stop streaming",
     "streaming.session.stop"),
]

reg_text = [x[0] for x in REG]
reg_expected = [x[1] for x in REG]

reg_pred, reg_conf = predict(reg_text)

reg_acc = accuracy_score(
    reg_expected,
    reg_pred
)

print("\n")
print("=" * 78)
print("TARGETED REGRESSION")
print("=" * 78)

for t, e, p, c in zip(
    reg_text,
    reg_expected,
    reg_pred,
    reg_conf
):
    print(
        f"{'OK' if e == p else 'WRONG':5} | "
        f"{c * 100:6.2f}% | "
        f"{e:32} | {t}"
    )

print(
    "\nTargeted accuracy:",
    f"{reg_acc * 100:.2f}%"
)

pd.DataFrame({
    "text": reg_text,
    "expected": reg_expected,
    "predicted": reg_pred,
    "confidence": reg_conf,
    "correct": [
        e == p
        for e, p in zip(reg_expected, reg_pred)
    ]
}).to_csv(
    OUT / "targeted_results.csv",
    index=False
)


# ============================================================
# UNSEEN TEST
# ============================================================

stress_candidates = [
    ROOT / "unseen_semantic_stress_test.csv",
    ROOT / "v3_unseen_semantic_stress_predictions.csv",
    ROOT / "v2_unseen_semantic_stress_predictions.csv",
]

stress_path = next(
    (p for p in stress_candidates if p.exists()),
    None
)

if stress_path is not None:

    stress = pd.read_csv(stress_path)

    if "text" not in stress.columns:
        raise ValueError(
            "Stress CSV must contain 'text'."
        )

    if "intent" not in stress.columns:

        if "expected_intent" in stress.columns:
            stress["intent"] = stress["expected_intent"]

        elif "true_intent" in stress.columns:
            stress["intent"] = stress["true_intent"]

        else:
            raise ValueError(
                "Stress CSV needs intent/expected_intent/true_intent."
            )

    stress = stress.dropna(
        subset=["text", "intent"]
    )

    unseen_pred, unseen_conf = predict(
        stress["text"].tolist()
    )

    unseen_acc = accuracy_score(
        stress["intent"],
        unseen_pred
    )

    unseen_f1 = f1_score(
        stress["intent"],
        unseen_pred,
        average="macro"
    )

    print("\n")
    print("=" * 78)
    print("UNSEEN SEMANTIC STRESS TEST")
    print("=" * 78)

    print(
        f"Baseline:  {BASELINE_UNSEEN:.2f}%"
    )

    print(
        f"Candidate: {unseen_acc * 100:.2f}%"
    )

    print(
        f"Delta:     "
        f"{unseen_acc * 100 - BASELINE_UNSEEN:+.2f} pp"
    )

    print(
        f"Macro F1:  {unseen_f1 * 100:.2f}%"
    )

    print(
        classification_report(
            stress["intent"],
            unseen_pred,
            digits=4
        )
    )

    out_stress = stress.copy()

    out_stress["predicted_intent"] = unseen_pred
    out_stress["confidence"] = unseen_conf
    out_stress["correct"] = (
        out_stress["intent"]
        == out_stress["predicted_intent"]
    )

    out_stress.to_csv(
        OUT / "unseen_results.csv",
        index=False
    )

    out_stress[
        ~out_stress["correct"]
    ].to_csv(
        OUT / "unseen_errors.csv",
        index=False
    )

else:
    unseen_acc = None
    unseen_f1 = None

    print(
        "\nWARNING: unseen stress CSV not found."
    )


# ============================================================
# CONTEXTUAL TEST
# ============================================================

CONTEXTUAL = [
    ("it's quieter can you make it a little louder",
     "device.volume.increase"),

    ("it's a little loud can you make it quieter",
     "device.volume.decrease"),

    ("i can still hear it make it completely silent",
     "device.volume.mute"),

    ("turn the sound back on",
     "device.volume.unmute"),

    ("the audio is quiet but don't mute it make it louder",
     "device.volume.increase"),

    ("the audio is loud but keep it on just lower it",
     "device.volume.decrease"),

    ("i need to go to airport tomorrow",
     "reminders.task.create"),

    ("i need to go to airport tommorow",
     "reminders.task.create"),

    ("where can i find my phone",
     "find.phone.locate"),

    ("please show the reminders i have",
     "help.reminder.show"),

    ("mark that reminder as completed",
     "reminders.task.complete"),

    ("please start the streaming session",
     "streaming.session.start"),

    ("please stop the streaming session",
     "streaming.session.stop"),
]

ctx_text = [x[0] for x in CONTEXTUAL]
ctx_expected = [x[1] for x in CONTEXTUAL]

ctx_pred, ctx_conf = predict(ctx_text)

ctx_acc = accuracy_score(
    ctx_expected,
    ctx_pred
)

ctx_f1 = f1_score(
    ctx_expected,
    ctx_pred,
    average="macro"
)

print("\n")
print("=" * 78)
print("CONTEXTUAL TEST")
print("=" * 78)

for t, e, p, c in zip(
    ctx_text,
    ctx_expected,
    ctx_pred,
    ctx_conf
):
    print(
        f"{'OK' if e == p else 'WRONG':5} | "
        f"{c * 100:6.2f}% | "
        f"{e:32} | {t}"
    )

print(
    "\nContextual accuracy:",
    f"{ctx_acc * 100:.2f}%"
)

print(
    "Contextual macro F1:",
    f"{ctx_f1 * 100:.2f}%"
)

pd.DataFrame({
    "text": ctx_text,
    "expected": ctx_expected,
    "predicted": ctx_pred,
    "confidence": ctx_conf,
    "correct": [
        e == p
        for e, p in zip(ctx_expected, ctx_pred)
    ]
}).to_csv(
    OUT / "contextual_results.csv",
    index=False
)


# ============================================================
# OOD SANITY TEST
# ============================================================

OOD = [
    "what is the weather today",
    "i want to order pizza",
    "how do i cook pasta",
    "what time is the train",
    "tell me a joke",
    "i am planning a holiday",
    "what is the capital of france",
    "my car needs an oil change",
    "please open my browser",
    "i need a hotel tonight",
    "what is the stock market doing",
    "how do i charge my laptop",
]

ood_pred, ood_conf = predict(OOD)

OOD_THRESHOLD = 0.70

ood_reject = [
    c < OOD_THRESHOLD
    for c in ood_conf
]

ood_rate = (
    sum(ood_reject)
    / len(ood_reject)
)

print("\n")
print("=" * 78)
print("OOD SANITY TEST")
print("=" * 78)

for t, p, c, r in zip(
    OOD,
    ood_pred,
    ood_conf,
    ood_reject
):
    print(
        f"{'REJECT' if r else 'ACCEPT':6} | "
        f"{c * 100:6.2f}% | "
        f"{p:32} | {t}"
    )

print(
    "\nOOD rejection @ 0.70:",
    f"{ood_rate * 100:.2f}%"
)

pd.DataFrame({
    "text": OOD,
    "predicted": ood_pred,
    "confidence": ood_conf,
    "rejected": ood_reject
}).to_csv(
    OUT / "ood_results.csv",
    index=False
)


# ============================================================
# FINAL GATE
# ============================================================

print("\n")
print("=" * 78)
print("CURRENT INT8 BASELINE vs V2")
print("=" * 78)

print(
    f"Unseen:      {BASELINE_UNSEEN:.2f}% -> "
    f"{unseen_acc * 100:.2f}%"
    if unseen_acc is not None
    else "Unseen:      NOT AVAILABLE"
)

print(
    f"Contextual:  {BASELINE_CONTEXTUAL:.2f}% -> "
    f"{ctx_acc * 100:.2f}%"
)

print(
    f"OOD @ .70:   {BASELINE_OOD:.2f}% -> "
    f"{ood_rate * 100:.2f}%"
)

print(
    f"Targeted:    --- -> "
    f"{reg_acc * 100:.2f}%"
)

if unseen_acc is not None:
    unseen_pass = (
        unseen_acc * 100 >= BASELINE_UNSEEN
    )
else:
    unseen_pass = False

context_pass = (
    ctx_acc * 100 > BASELINE_CONTEXTUAL
)

ood_pass = (
    ood_rate * 100 >= BASELINE_OOD
)

target_pass = (
    reg_acc * 100 >= 95.0
)

print("\nGATES:")
print("  Unseen     :", "PASS" if unseen_pass else "FAIL")
print("  Contextual :", "PASS" if context_pass else "FAIL")
print("  OOD        :", "PASS" if ood_pass else "FAIL")
print("  Targeted   :", "PASS" if target_pass else "FAIL")

if (
    unseen_pass
    and context_pass
    and ood_pass
    and target_pass
):
    print("\nSTATUS: V2 PASSES THE BASELINE GATE")
    print(
        "Next step: export V2 FP32 -> ONNX -> INT8, "
        "then rerun the same benchmark."
    )
else:
    print("\nSTATUS: DO NOT PROMOTE V2")
    print(
        "Keep the CURRENT INT8 baseline."
    )
    print(
        "Inspect the saved *_errors/results.csv files."
    )

print("\nCandidate:")
print(OUT)

print(
    "\nCURRENT INT8 baseline was NOT modified."
)
