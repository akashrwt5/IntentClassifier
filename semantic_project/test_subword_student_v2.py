
import os
import json
import numpy as np
import torch
from torch import nn
from tokenizers import Tokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(
    SCRIPT_DIR,
    "subword_student_v2_distilled"
)

CONFIG_FILE = os.path.join(MODEL_DIR, "config.json")
TOKENIZER_FILE = os.path.join(MODEL_DIR, "tokenizer.json")
WEIGHTS_FILE = os.path.join(MODEL_DIR, "student_v2_fp32.pt")
LABELS_FILE = os.path.join(MODEL_DIR, "intent_labels.txt")


# ============================================================
# SUBWORD STUDENT V2 — INTERACTIVE TEST
# ============================================================

print("=" * 70)
print("SUBWORD STUDENT V2 — INTERACTIVE TEST")
print("=" * 70)

for path in [
    CONFIG_FILE,
    TOKENIZER_FILE,
    WEIGHTS_FILE,
    LABELS_FILE
]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found:\n{path}\n\n"
            "Make sure train_subword_student_v2_distilled_fixed.py "
            "completed successfully first."
        )


# ============================================================
# LOAD CONFIG / TOKENIZER / LABELS
# ============================================================

with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

with open(LABELS_FILE, "r") as f:
    labels = [
        x.strip()
        for x in f
        if x.strip()
    ]

tokenizer = Tokenizer.from_file(
    TOKENIZER_FILE
)

PAD_ID = config["pad_id"]
MAX_LEN = config["max_len"]


# ============================================================
# MODEL
# ============================================================

class SubwordStudent(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            config["vocab_size"],
            config["embed_dim"],
            padding_idx=PAD_ID
        )

        self.position = nn.Embedding(
            MAX_LEN,
            config["embed_dim"]
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["embed_dim"],
            nhead=config["num_heads"],
            dim_feedforward=config["ff_dim"],
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config["num_layers"]
        )

        self.norm = nn.LayerNorm(
            config["embed_dim"]
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                config["embed_dim"],
                config["embed_dim"]
            ),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(
                config["embed_dim"],
                config["num_classes"]
            )
        )

    def forward(self, x):

        pad_mask = x.eq(PAD_ID)

        positions = torch.arange(
            x.size(1),
            device=x.device
        ).unsqueeze(0)

        h = (
            self.embedding(x)
            + self.position(positions)
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
        ).clamp(min=1.0)

        pooled = self.norm(
            pooled
        )

        return self.classifier(
            pooled
        )


model = SubwordStudent()

model.load_state_dict(
    torch.load(
        WEIGHTS_FILE,
        map_location="cpu"
    )
)

model.eval()


# ============================================================
# ENCODE + PREDICT
# ============================================================

def encode_text(text):

    encoded = tokenizer.encode(
        text
    )

    ids = encoded.ids[:MAX_LEN]

    if len(ids) < MAX_LEN:
        ids += [
            PAD_ID
        ] * (
            MAX_LEN - len(ids)
        )

    return torch.tensor(
        [ids],
        dtype=torch.long
    )


def predict(text, show_tokens=True):

    x = encode_text(text)

    with torch.no_grad():

        logits = model(x)

        probs = torch.softmax(
            logits,
            dim=1
        )[0].numpy()

    order = np.argsort(
        probs
    )[::-1]

    predicted_id = int(
        order[0]
    )

    if show_tokens:

        print("\nTokens:")
        print(
            tokenizer.encode(text).tokens
        )

    print("\nPrediction:")
    print(
        "Intent     :",
        labels[predicted_id]
    )

    print(
        "Confidence :",
        f"{probs[predicted_id] * 100:.2f}%"
    )

    print("\nTop 3:")

    for rank, idx in enumerate(
        order[:3],
        1
    ):
        print(
            f"{rank}. "
            f"{labels[int(idx)]:32} "
            f"{probs[idx] * 100:.2f}%"
        )

    return (
        labels[predicted_id],
        float(probs[predicted_id])
    )


# ============================================================
# REQUIRED PROBLEM CASES
# ============================================================

TEST_CASES = [
    (
        "it's quieter can you make it a little louder",
        "device.volume.increase"
    ),
    (
        "i can still hear it make it completely silent",
        "device.volume.mute"
    ),
    (
        "turn off",
        "device.volume.mute"
    ),
    (
        "i need to go to airport tomorrow",
        "reminders.task.create"
    ),
    (
        "i need to go to airport tommorow",
        "reminders.task.create"
    ),
    (
        "the sound seems low please turn it up",
        "device.volume.increase"
    ),
    (
        "the sound is too loud turn it down",
        "device.volume.decrease"
    ),
    (
        "turn the sound back on",
        "device.volume.unmute"
    ),
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
    )
]


print("\n")
print("=" * 70)
print("V2 — REQUIRED TEST CASES")
print("=" * 70)

correct = 0

for text, expected in TEST_CASES:

    print("\n" + "-" * 70)
    print("Text:", text)
    print("Expected:", expected)

    predicted, confidence = predict(
        text,
        show_tokens=True
    )

    if predicted == expected:
        correct += 1
        print("RESULT     : CORRECT")
    else:
        print("RESULT     : WRONG")


print("\n")
print("=" * 70)
print("REQUIRED TEST SUMMARY")
print("=" * 70)

print(
    "Correct:",
    correct,
    "/",
    len(TEST_CASES)
)

print(
    "Accuracy:",
    f"{correct / len(TEST_CASES) * 100:.2f}%"
)


# ============================================================
# INTERACTIVE MODE
# ============================================================

print("\n")
print("=" * 70)
print("INTERACTIVE MODE")
print("=" * 70)

print(
    "Type your own sentence."
)

print(
    "Type 'exit' to stop."
)

while True:

    text = input("\nYou: ").strip()

    if text.lower() == "exit":
        break

    if not text:
        continue

    predict(text)
