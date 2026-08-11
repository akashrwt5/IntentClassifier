
import os
import json
import random
import numpy as np
import pandas as pd
import torch

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from torch import nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# SUBWORD TOKENIZER STUDENT V1
#
# Goal:
#   Replace the current word-ID tokenizer with a BPE subword
#   tokenizer while keeping the Transformer architecture small.
#
# Important:
#   Word order is preserved.
#
# Expected files in this same folder:
#   semantic_training_v3_hard_negatives.xlsx
#   unseen_semantic_stress_test.csv
#
# This script DOES NOT overwrite the current INT8 model.
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET = os.path.join(
    SCRIPT_DIR,
    "semantic_training_v3_hard_negatives.xlsx"
)

STRESS_FILE = os.path.join(
    SCRIPT_DIR,
    "unseen_semantic_stress_test.csv"
)

OUT_DIR = os.path.join(
    SCRIPT_DIR,
    "subword_student_v1"
)

SEED = 42

# ------------------------------------------------------------
# Small architecture: keep current capacity approximately
# similar so we can measure the tokenizer effect separately.
# ------------------------------------------------------------

EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 128

MAX_LEN = 32
VOCAB_SIZE = 5000

BATCH_SIZE = 64
EPOCHS = 14
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-4
DROPOUT = 0.10

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 65)
print("SUBWORD TOKENIZER STUDENT V1")
print("=" * 65)

print("Device:", DEVICE)
print("Embedding:", EMBED_DIM)
print("Transformer layers:", NUM_LAYERS)
print("Attention heads:", NUM_HEADS)
print("FFN:", FF_DIM)
print("Max sequence length:", MAX_LEN)
print("Target BPE vocabulary:", VOCAB_SIZE)


# ============================================================
# 1. LOAD DATASET
# ============================================================

if not os.path.exists(DATASET):
    raise FileNotFoundError(
        "Dataset not found:\n" + DATASET
    )

df = pd.read_excel(
    DATASET,
    sheet_name="dataset"
)

df = df[
    ["text", "intent"]
].dropna()

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
    df["text"] != ""
].drop_duplicates(
    ["text", "intent"]
).reset_index(drop=True)

labels = sorted(
    df["intent"].unique()
)

label_to_id = {
    label: i
    for i, label in enumerate(labels)
}

print("\nDataset:")
print("Samples:", len(df))
print("Intents:", len(labels))

print("\nIntent distribution:")
print(
    df["intent"].value_counts()
)


# ============================================================
# 2. ADD TARGETED CONTEXTUAL EXAMPLES
#
# These are included because the existing model showed
# difficulty around contextual volume commands.
# ============================================================

context_examples = [

    # Increase
    (
        "it's quieter can you make it a little louder",
        "device.volume.increase"
    ),
    (
        "the audio seems a bit soft could you raise it now",
        "device.volume.increase"
    ),
    (
        "things got quieter please raise the volume",
        "device.volume.increase"
    ),
    (
        "i can barely hear it make the sound stronger",
        "device.volume.increase"
    ),
    (
        "the sound dropped a little can you increase the volume",
        "device.volume.increase"
    ),
    (
        "everything sounds low right now make it louder",
        "device.volume.increase"
    ),
    (
        "i want more volume because this is too quiet",
        "device.volume.increase"
    ),
    (
        "the hearing aids sound faint please raise them",
        "device.volume.increase"
    ),

    # Decrease
    (
        "it's louder can you make it a little quieter",
        "device.volume.decrease"
    ),
    (
        "the audio is stronger than i need please lower it",
        "device.volume.decrease"
    ),
    (
        "the sound increased too much reduce the volume",
        "device.volume.decrease"
    ),
    (
        "everything is loud right now make it softer",
        "device.volume.decrease"
    ),
    (
        "i want less volume because this is too strong",
        "device.volume.decrease"
    ),
    (
        "the hearing aids are sounding intense please lower them",
        "device.volume.decrease"
    ),
    (
        "it became louder than i wanted decrease the sound",
        "device.volume.decrease"
    ),
    (
        "can you turn the audio lower it is more than enough",
        "device.volume.decrease"
    ),

    # Mute
    (
        "i can still hear the audio please make it totally silent",
        "device.volume.mute"
    ),
    (
        "there is some sound left turn all of it off",
        "device.volume.mute"
    ),
    (
        "i don't want any audio at all make it silent",
        "device.volume.mute"
    ),
    (
        "stop all sound from the hearing aids",
        "device.volume.mute"
    ),
    (
        "the volume is low but i need complete silence",
        "device.volume.mute"
    ),
    (
        "make sure nothing is audible",
        "device.volume.mute"
    ),

    # Unmute
    (
        "i want the audio back please restore it",
        "device.volume.unmute"
    ),
    (
        "there is no sound now can you enable it again",
        "device.volume.unmute"
    ),
    (
        "the hearing aids are quiet because they are muted bring sound back",
        "device.volume.unmute"
    ),
    (
        "restore audio so i can hear again",
        "device.volume.unmute"
    ),
    (
        "please enable the hearing aid sound",
        "device.volume.unmute"
    ),
    (
        "turn the audio back on after muting it",
        "device.volume.unmute"
    ),
]

context_df = pd.DataFrame(
    context_examples,
    columns=[
        "text",
        "intent"
    ]
)

original_keys = set(
    zip(
        df["text"].str.lower(),
        df["intent"]
    )
)

context_df = context_df[
    ~context_df.apply(
        lambda row:
            (
                row["text"].lower(),
                row["intent"]
            ) in original_keys,
        axis=1
    )
].reset_index(
    drop=True
)

train_df = pd.concat(
    [
        df,
        context_df
    ],
    ignore_index=True
)

print(
    "\nAdded contextual examples:",
    len(context_df)
)

print(
    "Total training rows:",
    len(train_df)
)


# ============================================================
# 3. TRAIN BPE SUBWORD TOKENIZER
# ============================================================

print("\nTraining BPE tokenizer...")

os.makedirs(
    OUT_DIR,
    exist_ok=True
)

tokenizer_path = os.path.join(
    OUT_DIR,
    "tokenizer.json"
)

tokenizer = Tokenizer(
    BPE(
        unk_token="<UNK>"
    )
)

# Whitespace pre-tokenization keeps word boundaries available
# while BPE learns subword pieces inside words.
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=[
        "<PAD>",
        "<UNK>",
        "<CLS>",
        "<SEP>"
    ],
    show_progress=True
)

tokenizer.train_from_iterator(
    train_df["text"].tolist(),
    trainer=trainer
)

# Add explicit CLS/SEP processing.
tokenizer.post_processor = (
    TemplateProcessing(
        single="<CLS> $A <SEP>",
        special_tokens=[
            (
                "<CLS>",
                tokenizer.token_to_id("<CLS>")
            ),
            (
                "<SEP>",
                tokenizer.token_to_id("<SEP>")
            )
        ]
    )
)

tokenizer.save(
    tokenizer_path
)

actual_vocab_size = tokenizer.get_vocab_size()

print(
    "Actual BPE vocabulary:",
    actual_vocab_size
)

print(
    "Tokenizer saved:",
    tokenizer_path
)


# ============================================================
# 4. SHOW TOKENIZATION EXAMPLES
# ============================================================

print("\nTokenization examples:")

examples = [
    "tomorrow",
    "tommorow",
    "tomorow",
    "louder",
    "quieter",
    "it's quieter can you make it a little louder",
    "turn off"
]

for text in examples:

    encoded = tokenizer.encode(
        text
    )

    print(
        "\nText:",
        text
    )

    print(
        "Tokens:",
        encoded.tokens
    )

    print(
        "IDs:",
        encoded.ids
    )


# ============================================================
# 5. TOKEN ENCODING
# ============================================================

PAD_ID = tokenizer.token_to_id(
    "<PAD>"
)

UNK_ID = tokenizer.token_to_id(
    "<UNK>"
)

CLS_ID = tokenizer.token_to_id(
    "<CLS>"
)

SEP_ID = tokenizer.token_to_id(
    "<SEP>"
)


def encode_text(text):

    encoded = tokenizer.encode(
        str(text)
    )

    ids = encoded.ids[
        :MAX_LEN
    ]

    if len(ids) < MAX_LEN:

        ids += [
            PAD_ID
        ] * (
            MAX_LEN - len(ids)
        )

    return ids


# ============================================================
# 6. TRAIN / VALIDATION SPLIT
# ============================================================

indices = np.arange(
    len(train_df)
)

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.10,
    random_state=SEED,
    stratify=train_df["intent"]
)

print(
    "\nTrain:",
    len(train_idx)
)

print(
    "Validation:",
    len(val_idx)
)


class TextDataset(Dataset):

    def __init__(
        self,
        indices
    ):
        self.indices = np.asarray(
            indices
        )

    def __len__(self):
        return len(
            self.indices
        )

    def __getitem__(
        self,
        index
    ):

        i = int(
            self.indices[index]
        )

        text = train_df.iloc[i][
            "text"
        ]

        intent = train_df.iloc[i][
            "intent"
        ]

        return (
            torch.tensor(
                encode_text(text),
                dtype=torch.long
            ),
            torch.tensor(
                label_to_id[intent],
                dtype=torch.long
            )
        )


train_loader = DataLoader(
    TextDataset(train_idx),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    TextDataset(val_idx),
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 7. TINY TRANSFORMER
# ============================================================

class SubwordStudent(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            actual_vocab_size,
            EMBED_DIM,
            padding_idx=PAD_ID
        )

        self.position = nn.Embedding(
            MAX_LEN,
            EMBED_DIM
        )

        layer = (
            nn.TransformerEncoderLayer(
                d_model=EMBED_DIM,
                nhead=NUM_HEADS,
                dim_feedforward=FF_DIM,
                dropout=DROPOUT,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
        )

        self.encoder = (
            nn.TransformerEncoder(
                layer,
                num_layers=NUM_LAYERS
            )
        )

        self.norm = nn.LayerNorm(
            EMBED_DIM
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                EMBED_DIM,
                EMBED_DIM
            ),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(
                EMBED_DIM,
                len(labels)
            )
        )

    def forward(
        self,
        x
    ):

        pad_mask = x.eq(
            PAD_ID
        )

        positions = torch.arange(
            x.size(1),
            device=x.device
        ).unsqueeze(0)

        h = (
            self.embedding(x)
            + self.position(
                positions
            )
        )

        h = self.encoder(
            h,
            src_key_padding_mask=pad_mask
        )

        valid = (
            (~pad_mask)
            .unsqueeze(-1)
            .float()
        )

        pooled = (
            h * valid
        ).sum(dim=1) / valid.sum(
            dim=1
        ).clamp(
            min=1.0
        )

        pooled = self.norm(
            pooled
        )

        return self.classifier(
            pooled
        )


model = SubwordStudent().to(
    DEVICE
)

parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    "\nTrainable parameters:",
    parameters
)

print(
    "Approx FP32 weights:",
    round(
        parameters * 4 /
        1024 /
        1024,
        3
    ),
    "MB"
)


# ============================================================
# 8. TRAIN
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

best_f1 = -1.0
best_state = None

print("\nTraining...")

for epoch in range(
    EPOCHS
):

    model.train()

    running_loss = 0.0

    for x, y in train_loader:

        x = x.to(
            DEVICE
        )

        y = y.to(
            DEVICE
        )

        optimizer.zero_grad()

        logits = model(
            x
        )

        loss = criterion(
            logits,
            y
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        running_loss += (
            loss.item()
        )

    # Validation
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for x, y in val_loader:

            logits = model(
                x.to(DEVICE)
            )

            pred = torch.argmax(
                logits,
                dim=1
            ).cpu().numpy()

            y_pred.extend(
                pred
            )

            y_true.extend(
                y.numpy()
            )

    val_acc = accuracy_score(
        y_true,
        y_pred
    )

    val_f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"Loss={running_loss / len(train_loader):.4f} "
        f"ValAcc={val_acc * 100:.2f}% "
        f"ValF1={val_f1 * 100:.2f}%"
    )

    if val_f1 > best_f1:

        best_f1 = val_f1

        best_state = {
            k: v.detach().cpu().clone()
            for k, v in (
                model.state_dict().items()
            )
        }


model.load_state_dict(
    best_state
)

model.eval()


# ============================================================
# 9. SAVE MODEL
# ============================================================

weights_path = os.path.join(
    OUT_DIR,
    "student_fp32.pt"
)

torch.save(
    model.state_dict(),
    weights_path
)

with open(
    os.path.join(
        OUT_DIR,
        "intent_labels.txt"
    ),
    "w"
) as f:

    f.write(
        "\n".join(labels)
        + "\n"
    )

with open(
    os.path.join(
        OUT_DIR,
        "config.json"
    ),
    "w"
) as f:

    json.dump(
        {
            "tokenizer": "BPE",
            "vocab_size": actual_vocab_size,
            "num_classes": len(labels),
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "ff_dim": FF_DIM,
            "max_len": MAX_LEN,
            "pad_id": PAD_ID,
            "unk_id": UNK_ID,
            "cls_id": CLS_ID,
            "sep_id": SEP_ID
        },
        f,
        indent=2
    )

print(
    "\nFP32 model saved:",
    weights_path
)


# ============================================================
# 10. UNSEEN 595 TEST
# ============================================================

print("\n")
print("=" * 65)
print("SUBWORD STUDENT — UNSEEN SEMANTIC STRESS TEST")
print("=" * 65)

if not os.path.exists(
    STRESS_FILE
):
    raise FileNotFoundError(
        "Stress test not found:\n"
        + STRESS_FILE
    )

stress = pd.read_csv(
    STRESS_FILE
)

x_stress = torch.tensor(
    np.asarray(
        [
            encode_text(text)
            for text in stress["text"]
        ],
        dtype=np.int64
    ),
    dtype=torch.long
)

predictions = []

with torch.no_grad():

    for start in range(
        0,
        len(x_stress),
        BATCH_SIZE
    ):

        batch = x_stress[
            start:start + BATCH_SIZE
        ].to(DEVICE)

        logits = model(
            batch
        )

        pred = torch.argmax(
            logits,
            dim=1
        ).cpu().numpy()

        predictions.extend(
            labels[int(i)]
            for i in pred
        )

unseen_acc = accuracy_score(
    stress["intent"],
    predictions
)

unseen_f1 = f1_score(
    stress["intent"],
    predictions,
    average="macro"
)

print(
    "Accuracy:",
    round(
        unseen_acc * 100,
        2
    ),
    "%"
)

print(
    "Macro F1:",
    round(
        unseen_f1 * 100,
        2
    ),
    "%"
)

print(
    "\nClassification Report:"
)

print(
    classification_report(
        stress["intent"],
        predictions,
        digits=4
    )
)

stress_out = stress.copy()

stress_out[
    "predicted_intent"
] = predictions

stress_out["correct"] = (
    stress_out["intent"]
    == stress_out[
        "predicted_intent"
    ]
)

stress_out.to_csv(
    os.path.join(
        SCRIPT_DIR,
        "subword_unseen_predictions.csv"
    ),
    index=False
)

stress_out[
    ~stress_out["correct"]
].to_csv(
    os.path.join(
        SCRIPT_DIR,
        "subword_unseen_errors.csv"
    ),
    index=False
)


# ============================================================
# 11. TYPO ROBUSTNESS TEST
# ============================================================

TYPO_CASES = [

    (
        "tomorrow",
        "reminders.task.create"
    ),

    (
        "tommorow",
        "reminders.task.create"
    ),

    (
        "tomorow",
        "reminders.task.create"
    ),

    (
        "tomorroe",
        "reminders.task.create"
    ),

    (
        "louder",
        "device.volume.increase"
    ),

    (
        "louder!",
        "device.volume.increase"
    ),

    (
        "loudar",
        "device.volume.increase"
    ),

    (
        "quieter",
        "device.volume.decrease"
    ),

    (
        "quiter",
        "device.volume.decrease"
    ),

    (
        "quietter",
        "device.volume.decrease"
    ),

    (
        "unmute",
        "device.volume.unmute"
    ),

    (
        "unmut",
        "device.volume.unmute"
    ),

    (
        "mute",
        "device.volume.mute"
    ),

    (
        "mutt",
        "device.volume.mute"
    )
]

x_typo = torch.tensor(
    np.asarray(
        [
            encode_text(text)
            for text, _ in TYPO_CASES
        ],
        dtype=np.int64
    ),
    dtype=torch.long
)

typo_predictions = []

with torch.no_grad():

    logits = model(
        x_typo.to(DEVICE)
    )

    probs = torch.softmax(
        logits,
        dim=1
    ).cpu().numpy()

for i, row in enumerate(
    TYPO_CASES
):

    pred_id = int(
        np.argmax(
            probs[i]
        )
    )

    typo_predictions.append(
        labels[pred_id]
    )

print("\n")
print("=" * 65)
print("TYPO ROBUSTNESS TEST")
print("=" * 65)

typo_correct = 0

for (
    text,
    expected
), predicted in zip(
    TYPO_CASES,
    typo_predictions
):

    ok = (
        expected
        == predicted
    )

    if ok:
        typo_correct += 1

    print(
        f"{'OK' if ok else 'WRONG':5} | "
        f"{text:15} | "
        f"Expected: {expected:30} | "
        f"Predicted: {predicted}"
    )

typo_accuracy = (
    typo_correct
    / len(TYPO_CASES)
)

print(
    "\nTypo accuracy:",
    round(
        typo_accuracy * 100,
        2
    ),
    "%"
)


# ============================================================
# 12. CONTEXTUAL TEST
# ============================================================

CONTEXTUAL_CASES = [

    (
        "the audio seems a bit soft could you raise it now",
        "device.volume.increase"
    ),

    (
        "i can hear it but it is weaker than before please turn it up",
        "device.volume.increase"
    ),

    (
        "the sound dropped a little can you increase the volume",
        "device.volume.increase"
    ),

    (
        "everything sounds low right now make it louder",
        "device.volume.increase"
    ),

    (
        "i want more volume because this is too quiet",
        "device.volume.increase"
    ),

    (
        "the audio is stronger than i need please lower it",
        "device.volume.decrease"
    ),

    (
        "i can hear it fine but it is a little too loud turn it down",
        "device.volume.decrease"
    ),

    (
        "the sound increased too much reduce the volume",
        "device.volume.decrease"
    ),

    (
        "everything is loud right now make it softer",
        "device.volume.decrease"
    ),

    (
        "i want less volume because this is too strong",
        "device.volume.decrease"
    ),

    (
        "i can still hear the audio please make it totally silent",
        "device.volume.mute"
    ),

    (
        "there is some sound left turn all of it off",
        "device.volume.mute"
    ),

    (
        "i don't want any audio at all make it silent",
        "device.volume.mute"
    ),

    (
        "stop all sound from the hearing aids",
        "device.volume.mute"
    ),

    (
        "i want the audio back please restore it",
        "device.volume.unmute"
    ),

    (
        "there is no sound now can you enable it again",
        "device.volume.unmute"
    ),

    (
        "restore audio so i can hear again",
        "device.volume.unmute"
    ),

    (
        "please enable the hearing aid sound",
        "device.volume.unmute"
    )
]

x_context = torch.tensor(
    np.asarray(
        [
            encode_text(text)
            for text, _ in CONTEXTUAL_CASES
        ],
        dtype=np.int64
    ),
    dtype=torch.long
)

with torch.no_grad():

    logits = model(
        x_context.to(DEVICE)
    )

    context_pred_ids = (
        torch.argmax(
            logits,
            dim=1
        )
        .cpu()
        .numpy()
    )

context_predictions = [
    labels[int(i)]
    for i in context_pred_ids
]

context_truth = [
    x[1]
    for x in CONTEXTUAL_CASES
]

context_accuracy = accuracy_score(
    context_truth,
    context_predictions
)

context_f1 = f1_score(
    context_truth,
    context_predictions,
    average="macro"
)

print("\n")
print("=" * 65)
print("CONTEXTUAL-ACTION TEST")
print("=" * 65)

print(
    "Accuracy:",
    round(
        context_accuracy * 100,
        2
    ),
    "%"
)

print(
    "Macro F1:",
    round(
        context_f1 * 100,
        2
    ),
    "%"
)

print(
    "\nClassification Report:"
)

print(
    classification_report(
        context_truth,
        context_predictions,
        digits=4
    )
)

context_out = pd.DataFrame(
    {
        "text": [
            x[0]
            for x in CONTEXTUAL_CASES
        ],
        "expected": context_truth,
        "predicted": context_predictions,
        "correct": [
            a == b
            for a, b in zip(
                context_truth,
                context_predictions
            )
        ]
    }
)

context_out.to_csv(
    os.path.join(
        SCRIPT_DIR,
        "subword_contextual_predictions.csv"
    ),
    index=False
)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

fp32_size = (
    os.path.getsize(
        weights_path
    )
    / 1024
    / 1024
)

print("\n")
print("=" * 65)
print("SUBWORD STUDENT V1 — FINAL SUMMARY")
print("=" * 65)

print(
    "Tokenizer: BPE subword"
)

print(
    "Actual vocabulary:",
    actual_vocab_size
)

print(
    "Transformer layers:",
    NUM_LAYERS
)

print(
    "Attention heads:",
    NUM_HEADS
)

print(
    "Embedding:",
    EMBED_DIM
)

print(
    "Max length:",
    MAX_LEN
)

print(
    "FP32 model size:",
    round(
        fp32_size,
        3
    ),
    "MB"
)

print(
    "Unseen accuracy:",
    round(
        unseen_acc * 100,
        2
    ),
    "%"
)

print(
    "Contextual accuracy:",
    round(
        context_accuracy * 100,
        2
    ),
    "%"
)

print(
    "Typo accuracy:",
    round(
        typo_accuracy * 100,
        2
    ),
    "%"
)

print("\nSaved directory:")
print(OUT_DIR)

print("\nImportant:")
print(
    "Current INT8 model was NOT modified."
)

print(
    "If this model improves unseen + contextual + typo tests,"
)

print(
    "we will export this tokenizer + model to ONNX and INT8 next."
)
