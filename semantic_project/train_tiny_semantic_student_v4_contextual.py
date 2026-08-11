
import os
import json
import random
import numpy as np
import pandas as pd
import joblib
import torch

from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# V4 CONTEXTUAL-ACTION STUDENT
# Fixes cases where a sentence describes the current state
# and then asks for an opposite action.
#
# Example:
# "it's quieter, can you make it a little louder"
#              -> device.volume.increase
#
# Existing 595-sample unseen test remains untouched.
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET = os.path.join(
    SCRIPT_DIR, "semantic_training_v3_hard_negatives.xlsx"
)
STRESS_FILE = os.path.join(
    SCRIPT_DIR, "unseen_semantic_stress_test.csv"
)
TEACHER_TFIDF = os.path.join(
    SCRIPT_DIR, "v3_hybrid_tfidf.joblib"
)
TEACHER_CLASSIFIER = os.path.join(
    SCRIPT_DIR, "v3_hybrid_classifier.joblib"
)

SEED = 42
BATCH_SIZE = 64
EPOCHS = 14
LR = 1.5e-3
WEIGHT_DECAY = 1e-4

TEMPERATURE = 2.0
ALPHA = 0.35       # teacher soft target weight
BETA = 0.65        # hard label weight

MAX_LEN = 24
VOCAB_SIZE = 8000
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 128
DROPOUT = 0.10

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Device:", DEVICE)

# ============================================================
# 1. LOAD ORIGINAL V3 DATA
# ============================================================

df = pd.read_excel(DATASET, sheet_name="dataset")
df = df[["text", "intent"]].dropna()
df["text"] = df["text"].astype(str).str.strip()
df["intent"] = df["intent"].astype(str).str.strip()
df = df[df["text"] != ""].drop_duplicates(
    ["text", "intent"]
).reset_index(drop=True)

labels = sorted(df["intent"].unique())
label_to_id = {
    label: i for i, label in enumerate(labels)
}

print("Original samples:", len(df))
print("Intents:", len(labels))

# ============================================================
# 2. TARGETED CONTEXTUAL-ACTION HARD NEGATIVES
# ============================================================

context_examples = [

    # ---------------- VOLUME INCREASE ----------------
    ("it's quieter can you make it a little louder",
     "device.volume.increase"),
    ("it is quieter can you make it louder",
     "device.volume.increase"),
    ("it's quiet please make it louder",
     "device.volume.increase"),
    ("the sound is quieter now turn it up",
     "device.volume.increase"),
    ("the sound got quieter make it louder",
     "device.volume.increase"),
    ("the volume is lower can you turn it up",
     "device.volume.increase"),
    ("the volume is low please increase it",
     "device.volume.increase"),
    ("it sounds too quiet please turn it up",
     "device.volume.increase"),
    ("it became quieter I want it louder",
     "device.volume.increase"),
    ("things sound quiet make them louder",
     "device.volume.increase"),
    ("the sound is too soft increase the volume",
     "device.volume.increase"),
    ("it is not loud enough turn it up",
     "device.volume.increase"),
    ("the audio got lower make it louder",
     "device.volume.increase"),
    ("it sounds low please make it a bit louder",
     "device.volume.increase"),
    ("the hearing aids sound quiet turn them up",
     "device.volume.increase"),
    ("can you make it louder because it sounds quiet",
     "device.volume.increase"),

    # ---------------- VOLUME DECREASE ----------------
    ("it's louder can you make it a little quieter",
     "device.volume.decrease"),
    ("it is louder can you make it quieter",
     "device.volume.decrease"),
    ("it's loud please make it quieter",
     "device.volume.decrease"),
    ("the sound is louder now turn it down",
     "device.volume.decrease"),
    ("the sound got louder make it quieter",
     "device.volume.decrease"),
    ("the volume is higher can you turn it down",
     "device.volume.decrease"),
    ("the volume is high please lower it",
     "device.volume.decrease"),
    ("it sounds too loud please turn it down",
     "device.volume.decrease"),
    ("it became louder I want it quieter",
     "device.volume.decrease"),
    ("things sound loud make them quieter",
     "device.volume.decrease"),
    ("the sound is too strong lower the volume",
     "device.volume.decrease"),
    ("it is too loud turn it down a little",
     "device.volume.decrease"),
    ("the audio got higher make it quieter",
     "device.volume.decrease"),
    ("it sounds loud please make it a bit quieter",
     "device.volume.decrease"),
    ("the hearing aids sound loud turn them down",
     "device.volume.decrease"),
    ("can you make it quieter because it sounds loud",
     "device.volume.decrease"),

    # ---------------- MUTE ----------------
    ("it's still audible make it completely silent",
     "device.volume.mute"),
    ("the sound is quiet but I want it completely off",
     "device.volume.mute"),
    ("it is still making sound turn everything off",
     "device.volume.mute"),
    ("lower the sound all the way until it is silent",
     "device.volume.mute"),
    ("I can still hear it please mute it",
     "device.volume.mute"),
    ("the volume is low but make it completely silent",
     "device.volume.mute"),
    ("make the hearing aids completely silent",
     "device.volume.mute"),
    ("turn all sound off even though it is already quiet",
     "device.volume.mute"),

    # ---------------- UNMUTE ----------------
    ("it's silent turn the sound back on",
     "device.volume.unmute"),
    ("it is muted can you restore the sound",
     "device.volume.unmute"),
    ("there is no sound please turn it back on",
     "device.volume.unmute"),
    ("the hearing aids are silent bring the sound back",
     "device.volume.unmute"),
    ("it is completely quiet unmute it",
     "device.volume.unmute"),
    ("the sound is off please restore it",
     "device.volume.unmute"),
    ("nothing is audible turn the sound back on",
     "device.volume.unmute"),
    ("I cannot hear any sound please unmute",
     "device.volume.unmute"),
]

context_df = pd.DataFrame(
    context_examples,
    columns=["text", "intent"]
)

# Remove accidental duplicates against original data.
original_keys = set(
    zip(df["text"].str.lower(), df["intent"])
)

context_df = context_df[
    ~context_df.apply(
        lambda r: (
            r["text"].lower(),
            r["intent"]
        ) in original_keys,
        axis=1
    )
].reset_index(drop=True)

print("New contextual examples:", len(context_df))

# Combine.
train_df = pd.concat(
    [df, context_df],
    ignore_index=True
)

# ============================================================
# 3. BUILD VOCABULARY FROM ORIGINAL + V4 EXAMPLES
# ============================================================

word_counts = {}

for text in train_df["text"]:
    for token in text.lower().split():
        token = token.strip(".,!?;:\"'()[]{}")
        if token:
            word_counts[token] = (
                word_counts.get(token, 0) + 1
            )

special = ["<PAD>", "<UNK>"]

sorted_words = sorted(
    word_counts.items(),
    key=lambda x: (-x[1], x[0])
)[:VOCAB_SIZE - len(special)]

vocab = {
    "<PAD>": 0,
    "<UNK>": 1
}

for token, _ in sorted_words:
    if token not in vocab:
        vocab[token] = len(vocab)

print("Vocabulary:", len(vocab))

def tokenize(text):
    tokens = []

    for token in str(text).lower().split():
        token = token.strip(".,!?;:\"'()[]{}")
        if token:
            tokens.append(
                vocab.get(token, 1)
            )

    tokens = tokens[:MAX_LEN]

    if len(tokens) < MAX_LEN:
        tokens += [0] * (
            MAX_LEN - len(tokens)
        )

    return tokens

# ============================================================
# 4. LOAD V3 TEACHER
# ============================================================

print("\nLoading V3 teacher...")

tfidf = joblib.load(TEACHER_TFIDF)
teacher_clf = joblib.load(TEACHER_CLASSIFIER)

teacher_encoder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

SEMANTIC_WEIGHT = 2.0

print("Creating teacher embeddings...")

X_tfidf = tfidf.transform(
    train_df["text"]
)

embeddings = teacher_encoder.encode(
    train_df["text"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

X_sem = csr_matrix(embeddings)

X_teacher = hstack([
    X_tfidf,
    X_sem * SEMANTIC_WEIGHT
]).tocsr()

teacher_probs_raw = teacher_clf.predict_proba(
    X_teacher
)

teacher_probs = np.zeros(
    (
        len(train_df),
        len(labels)
    ),
    dtype=np.float32
)

for j, label in enumerate(
    teacher_clf.classes_
):
    teacher_probs[
        :,
        label_to_id[label]
    ] = teacher_probs_raw[:, j]

# For explicitly authored contextual examples,
# make the hard label authoritative instead of
# allowing a possibly-confused teacher prediction
# to dominate.
context_start = len(df)

for i in range(
    context_start,
    len(train_df)
):
    true_id = label_to_id[
        train_df.iloc[i]["intent"]
    ]

    teacher_probs[i] *= 0.20
    teacher_probs[i, true_id] += 0.80

    teacher_probs[i] /= (
        teacher_probs[i].sum()
    )

# ============================================================
# 5. TRAIN/VALIDATION SPLIT
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

print("Train:", len(train_idx))
print("Validation:", len(val_idx))

class StudentDataset(Dataset):

    def __init__(self, indices):
        self.indices = np.asarray(
            indices
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):

        idx = int(
            self.indices[i]
        )

        return (
            torch.tensor(
                tokenize(
                    train_df.iloc[idx]["text"]
                ),
                dtype=torch.long
            ),
            torch.tensor(
                label_to_id[
                    train_df.iloc[idx]["intent"]
                ],
                dtype=torch.long
            ),
            torch.tensor(
                teacher_probs[idx],
                dtype=torch.float32
            )
        )

train_loader = DataLoader(
    StudentDataset(train_idx),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    StudentDataset(val_idx),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# 6. MODEL
# ============================================================

class TinySemanticStudent(nn.Module):

    def __init__(
        self,
        vocab_size,
        num_classes,
        embed_dim=64,
        heads=4,
        layers=2,
        ff_dim=128,
        max_len=24,
        dropout=0.1
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        self.position = nn.Embedding(
            max_len,
            embed_dim
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
        )

        self.encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=layers
            )
        )

        self.norm = nn.LayerNorm(
            embed_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                64,
                num_classes
            )
        )

    def forward(self, x):

        mask = x.eq(0)

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
            src_key_padding_mask=mask
        )

        valid = (
            (~mask)
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

student = TinySemanticStudent(
    vocab_size=len(vocab),
    num_classes=len(labels),
    embed_dim=EMBED_DIM,
    heads=NUM_HEADS,
    layers=NUM_LAYERS,
    ff_dim=FF_DIM,
    max_len=MAX_LEN,
    dropout=DROPOUT
).to(DEVICE)

parameters = sum(
    p.numel()
    for p in student.parameters()
    if p.requires_grad
)

print(
    "Student parameters:",
    parameters
)

print(
    "Approx FP32 weights:",
    round(
        parameters * 4 / 1024 / 1024,
        3
    ),
    "MB"
)

# ============================================================
# 7. DISTILLATION LOSS
# ============================================================

hard_ce = nn.CrossEntropyLoss()

def loss_fn(
    logits,
    hard,
    teacher_prob
):

    teacher_soft = torch.softmax(
        torch.log(
            teacher_prob.clamp(
                min=1e-8
            )
        ) / TEMPERATURE,
        dim=-1
    )

    student_log = torch.log_softmax(
        logits / TEMPERATURE,
        dim=-1
    )

    soft_loss = (
        nn.functional.kl_div(
            student_log,
            teacher_soft,
            reduction="batchmean"
        )
        * (TEMPERATURE ** 2)
    )

    hard_loss = hard_ce(
        logits,
        hard
    )

    return (
        ALPHA * soft_loss
        + BETA * hard_loss
    )

optimizer = torch.optim.AdamW(
    student.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

# ============================================================
# 8. TRAIN
# ============================================================

best_f1 = -1.0
best_state = None

for epoch in range(EPOCHS):

    student.train()

    total_loss = 0.0

    for x, hard, soft in train_loader:

        x = x.to(DEVICE)
        hard = hard.to(DEVICE)
        soft = soft.to(DEVICE)

        optimizer.zero_grad()

        logits = student(x)

        loss = loss_fn(
            logits,
            hard,
            soft
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            student.parameters(),
            1.0
        )

        optimizer.step()

        total_loss += loss.item()

    student.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for x, hard, _ in val_loader:

            logits = student(
                x.to(DEVICE)
            )

            pred = torch.argmax(
                logits,
                dim=1
            ).cpu().numpy()

            y_pred.extend(pred)
            y_true.extend(
                hard.numpy()
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
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"Loss={total_loss / len(train_loader):.4f} "
        f"ValAcc={val_acc * 100:.2f}% "
        f"ValF1={val_f1 * 100:.2f}%"
    )

    if val_f1 > best_f1:

        best_f1 = val_f1

        best_state = {
            k: v.detach().cpu().clone()
            for k, v in (
                student.state_dict()
                .items()
            )
        }

student.load_state_dict(
    best_state
)

student.eval()

# ============================================================
# 9. SAVE V4
# ============================================================

OUT_DIR = os.path.join(
    SCRIPT_DIR,
    "tiny_semantic_student_v4_contextual"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)

torch.save(
    student.state_dict(),
    os.path.join(
        OUT_DIR,
        "student_v4_fp32.pt"
    )
)

with open(
    os.path.join(
        OUT_DIR,
        "vocab.json"
    ),
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        vocab,
        f,
        ensure_ascii=False
    )

with open(
    os.path.join(
        OUT_DIR,
        "intent_labels.txt"
    ),
    "w"
) as f:
    for label in labels:
        f.write(
            label + "\n"
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
            "vocab_size": len(vocab),
            "num_classes": len(labels),
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "ff_dim": FF_DIM,
            "max_len": MAX_LEN
        },
        f,
        indent=2
    )

# ============================================================
# 10. EXISTING 595-SAMPLE UNSEEN TEST
# ============================================================

print("\n================================")
print("V4 UNSEEN SEMANTIC STRESS TEST")
print("================================")

stress = pd.read_csv(
    STRESS_FILE
)

x_stress = torch.tensor(
    np.array([
        tokenize(text)
        for text in stress["text"]
    ]),
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

        logits = student(
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

acc = accuracy_score(
    stress["intent"],
    predictions
)

macro_f1 = f1_score(
    stress["intent"],
    predictions,
    average="macro"
)

print(
    "Accuracy:",
    round(acc * 100, 2),
    "%"
)

print(
    "Macro F1:",
    round(
        macro_f1 * 100,
        2
    ),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        stress["intent"],
        predictions,
        digits=4
    )
)

results = stress.copy()

results["predicted_intent"] = (
    predictions
)

results["correct"] = (
    results["intent"]
    == results["predicted_intent"]
)

results.to_csv(
    os.path.join(
        SCRIPT_DIR,
        "v4_unseen_semantic_predictions.csv"
    ),
    index=False
)

results[
    ~results["correct"]
].to_csv(
    os.path.join(
        SCRIPT_DIR,
        "v4_unseen_semantic_errors.csv"
    ),
    index=False
)

# ============================================================
# 11. DEDICATED CONTEXTUAL-ACTION TEST
# ============================================================

context_test = pd.DataFrame(
    context_examples,
    columns=[
        "text",
        "intent"
    ]
)

x_context = torch.tensor(
    np.array([
        tokenize(text)
        for text in context_test["text"]
    ]),
    dtype=torch.long
)

context_predictions = []

with torch.no_grad():

    for start in range(
        0,
        len(x_context),
        BATCH_SIZE
    ):

        batch = x_context[
            start:start + BATCH_SIZE
        ].to(DEVICE)

        logits = student(
            batch
        )

        pred = torch.argmax(
            logits,
            dim=1
        ).cpu().numpy()

        context_predictions.extend(
            labels[int(i)]
            for i in pred
        )

context_acc = accuracy_score(
    context_test["intent"],
    context_predictions
)

context_f1 = f1_score(
    context_test["intent"],
    context_predictions,
    average="macro"
)

print("\n================================")
print("V4 CONTEXTUAL-ACTION TEST")
print("================================")

print(
    "Accuracy:",
    round(
        context_acc * 100,
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

print("\nClassification Report:")

print(
    classification_report(
        context_test["intent"],
        context_predictions,
        digits=4
    )
)

context_results = context_test.copy()

context_results[
    "predicted_intent"
] = context_predictions

context_results["correct"] = (
    context_results["intent"]
    == context_results[
        "predicted_intent"
    ]
)

context_results.to_csv(
    os.path.join(
        SCRIPT_DIR,
        "v4_contextual_action_predictions.csv"
    ),
    index=False
)

context_results[
    ~context_results["correct"]
].to_csv(
    os.path.join(
        SCRIPT_DIR,
        "v4_contextual_action_errors.csv"
    ),
    index=False
)

# Print the exact problem sentence.
target = (
    "it's quieter can you make it a little louder"
)

match = context_results[
    context_results["text"] == target
]

if len(match) > 0:

    print("\nTarget sentence check:")

    print(
        "Text:",
        target
    )

    print(
        "Expected:",
        match.iloc[0]["intent"]
    )

    print(
        "Predicted:",
        match.iloc[0][
            "predicted_intent"
        ]
    )

# ============================================================
# FINAL
# ============================================================

print("\n================================")
print("V4 COMPLETE")
print("================================")

print(
    "Existing unseen accuracy:",
    round(acc * 100, 2),
    "%"
)

print(
    "Contextual-action accuracy:",
    round(
        context_acc * 100,
        2
    ),
    "%"
)

print(
    "\nV4 FP32 model saved to:"
)

print(
    os.path.join(
        OUT_DIR,
        "student_v4_fp32.pt"
    )
)

print(
    "\nDo NOT replace the current INT8 model yet."
)

print(
    "Only promote V4 if it improves "
    "the contextual test without an "
    "unacceptable regression on the "
    "595-sample unseen test."
)
