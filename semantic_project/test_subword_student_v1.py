
import os
import json
import numpy as np
import torch
from torch import nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "subword_student_v1")

# -----------------------------
# Load config/tokenizer/model
# -----------------------------
from tokenizers import Tokenizer

with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
    config = json.load(f)

with open(os.path.join(MODEL_DIR, "intent_labels.txt"), "r") as f:
    labels = [x.strip() for x in f if x.strip()]

tokenizer = Tokenizer.from_file(
    os.path.join(MODEL_DIR, "tokenizer.json")
)

PAD_ID = config["pad_id"]
MAX_LEN = config["max_len"]


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

        layer = nn.TransformerEncoderLayer(
            d_model=config["embed_dim"],
            nhead=config["num_heads"],
            dim_feedforward=config["ff_dim"],
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            layer,
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

        pos = torch.arange(
            x.size(1),
            device=x.device
        ).unsqueeze(0)

        h = (
            self.embedding(x)
            + self.position(pos)
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

        return self.classifier(
            self.norm(pooled)
        )


model = SubwordStudent()

model.load_state_dict(
    torch.load(
        os.path.join(
            MODEL_DIR,
            "student_fp32.pt"
        ),
        map_location="cpu"
    )
)

model.eval()


def encode(text):

    encoded = tokenizer.encode(text)

    ids = encoded.ids[:MAX_LEN]

    ids += [PAD_ID] * (
        MAX_LEN - len(ids)
    )

    return torch.tensor(
        [ids],
        dtype=torch.long
    )


def predict(text):

    x = encode(text)

    with torch.no_grad():

        logits = model(x)

        probs = torch.softmax(
            logits,
            dim=1
        )[0].numpy()

    order = np.argsort(
        probs
    )[::-1]

    print("\nYou:", text)

    print(
        "\nTokens:"
    )

    print(
        tokenizer.encode(text).tokens
    )

    print(
        "\nPrediction:"
    )

    print(
        "Intent     :",
        labels[order[0]]
    )

    print(
        "Confidence :",
        f"{probs[order[0]] * 100:.2f}%"
    )

    print("\nTop 3:")

    for rank, idx in enumerate(
        order[:3],
        1
    ):
        print(
            f"{rank}. "
            f"{labels[idx]:32} "
            f"{probs[idx] * 100:.2f}%"
        )


print("=" * 65)
print("SUBWORD STUDENT V1 — INTERACTIVE TEST")
print("=" * 65)

print("\nType a sentence.")
print("Type 'exit' to stop.")

print("\nSuggested tests:")
print("  it's quieter can you make it a little louder")
print("  the sound seems low please turn it up")
print("  the sound is too loud turn it down")
print("  turn off")
print("  i can still hear it make it completely silent")
print("  turn the sound back on")
print("  tomorrow")
print("  tommorow")
print("  tomorow")
print("  i need to go to airport tomorrow")
print()

while True:

    text = input("You: ").strip()

    if text.lower() == "exit":
        break

    if not text:
        continue

    predict(text)
