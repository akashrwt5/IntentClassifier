
import os
import json
import random
import re
import numpy as np
import pandas as pd
import joblib
import torch

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from sentence_transformers import SentenceTransformer
from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from torch import nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# SUBWORD STUDENT V2
#
# BPE tokenizer
# + V3 hybrid teacher distillation
# + contextual hard negatives
# + typo augmentation
#
# Goal:
#   Improve the V1 BPE model while preserving word order.
#
# Existing INT8 model is NOT modified.
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET = os.path.join(
    SCRIPT_DIR,
    "semantic_training_v3_hard_negatives.xlsx"
)

STRESS_FILE = os.path.join(
    SCRIPT_DIR,
    "unseen_semantic_stress_test.csv"
)

TEACHER_TFIDF = os.path.join(
    SCRIPT_DIR,
    "v3_hybrid_tfidf.joblib"
)

TEACHER_CLASSIFIER = os.path.join(
    SCRIPT_DIR,
    "v3_hybrid_classifier.joblib"
)

OUT_DIR = os.path.join(
    SCRIPT_DIR,
    "subword_student_v2_distilled"
)

SEED = 42

# ------------------------------------------------------------
# Student architecture
# Keep architecture close to V1 so we isolate the effect
# of better training/data/tokenizer.
# ------------------------------------------------------------

EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 128

MAX_LEN = 32
VOCAB_SIZE = 5000

BATCH_SIZE = 64
EPOCHS = 16

LR = 8e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.10

# ------------------------------------------------------------
# Distillation
# ------------------------------------------------------------

TEMPERATURE = 2.0

# Hard label is dominant.
# Teacher provides semantic guidance.
ALPHA = 0.30
BETA = 0.70

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 70)
print("SUBWORD STUDENT V2 — DISTILLED + TYPO + HARD NEGATIVES")
print("=" * 70)

print("Device:", DEVICE)
print("Embedding:", EMBED_DIM)
print("Transformer layers:", NUM_LAYERS)
print("Attention heads:", NUM_HEADS)
print("FFN:", FF_DIM)
print("Max sequence length:", MAX_LEN)
print("Target BPE vocabulary:", VOCAB_SIZE)
print("Teacher alpha:", ALPHA)
print("Hard-label beta:", BETA)


# ============================================================
# 1. LOAD ORIGINAL DATA
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

print("\nOriginal samples:", len(df))
print("Intents:", len(labels))


# ============================================================
# 2. TARGETED CONTEXTUAL HARD NEGATIVES
#
# Focus:
#   increase vs decrease
#   mute vs increase
#   mute vs decrease
#   unmute vs mute
#   streaming.stop vs volume.mute
#
# These phrases explicitly encode state + requested action.
# ============================================================

context_examples = [

    # --------------------------------------------------------
    # INCREASE
    # --------------------------------------------------------

    (
        "it's quieter can you make it a little louder",
        "device.volume.increase"
    ),
    (
        "the sound seems low could you raise it now",
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
    (
        "the audio is too soft turn it up",
        "device.volume.increase"
    ),
    (
        "i can hear it but the sound is weak make it louder",
        "device.volume.increase"
    ),
    (
        "the volume is lower than before increase it",
        "device.volume.increase"
    ),
    (
        "please increase the sound because it is difficult to hear",
        "device.volume.increase"
    ),
    (
        "make the audio louder it has become quiet",
        "device.volume.increase"
    ),
    (
        "turn the volume up because the sound is faint",
        "device.volume.increase"
    ),
    (
        "can you raise the hearing aid volume it is too soft",
        "device.volume.increase"
    ),
    (
        "the audio needs to be louder please increase it",
        "device.volume.increase"
    ),

    # --------------------------------------------------------
    # DECREASE
    # --------------------------------------------------------

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
    (
        "the sound is too loud please reduce it",
        "device.volume.decrease"
    ),
    (
        "i can hear it but it is too strong turn it down",
        "device.volume.decrease"
    ),
    (
        "the volume is higher than before lower it",
        "device.volume.decrease"
    ),
    (
        "please decrease the sound because it is too loud",
        "device.volume.decrease"
    ),
    (
        "make the audio quieter it has become loud",
        "device.volume.decrease"
    ),
    (
        "turn the volume down because the sound is intense",
        "device.volume.decrease"
    ),
    (
        "can you lower the hearing aid volume it is too loud",
        "device.volume.decrease"
    ),
    (
        "the audio needs to be quieter please decrease it",
        "device.volume.decrease"
    ),

    # --------------------------------------------------------
    # MUTE
    # --------------------------------------------------------

    (
        "i can still hear it make it completely silent",
        "device.volume.mute"
    ),
    (
        "i can still hear some sound please mute it",
        "device.volume.mute"
    ),
    (
        "the sound is still audible make everything silent",
        "device.volume.mute"
    ),
    (
        "there is still audio turn all sound off",
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
    (
        "i hear something please make the hearing aids completely silent",
        "device.volume.mute"
    ),
    (
        "there is some sound left shut it off completely",
        "device.volume.mute"
    ),
    (
        "the audio is quiet but i want zero sound",
        "device.volume.mute"
    ),
    (
        "turn off all hearing aid audio",
        "device.volume.mute"
    ),
    (
        "make the sound completely disappear",
        "device.volume.mute"
    ),
    (
        "i still hear audio please turn it fully off",
        "device.volume.mute"
    ),
    (
        "silence the hearing aids completely",
        "device.volume.mute"
    ),
    (
        "no sound at all please mute the audio",
        "device.volume.mute"
    ),

    # --------------------------------------------------------
    # UNMUTE
    # --------------------------------------------------------

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
    (
        "i need the sound restored",
        "device.volume.unmute"
    ),
    (
        "unmute the hearing aids so audio returns",
        "device.volume.unmute"
    ),
    (
        "the audio is silent bring it back",
        "device.volume.unmute"
    ),
    (
        "sound is off please turn it back on",
        "device.volume.unmute"
    ),
    (
        "there is no audio restore the sound",
        "device.volume.unmute"
    ),
    (
        "enable audio again so i can hear",
        "device.volume.unmute"
    ),
]


# ============================================================
# 3. TYPO AUGMENTATION
#
# We create realistic spelling/noise variants.
# IMPORTANT:
# Only small, controlled changes are applied.
# ============================================================

def typo_variants(word):

    word = str(word)

    if len(word) < 4:
        return []

    variants = set()

    # Missing character
    for i in range(len(word)):
        variants.add(
            word[:i] + word[i + 1:]
        )

    # Duplicate character
    for i in range(len(word)):
        variants.add(
            word[:i]
            + word[i]
            + word[i:]
        )

    # Swap adjacent characters
    for i in range(len(word) - 1):
        chars = list(word)
        chars[i], chars[i + 1] = (
            chars[i + 1],
            chars[i]
        )
        variants.add(
            "".join(chars)
        )

    # Limit per word
    return list(variants)[:8]


TYPO_SEED_WORDS = [
    "tomorrow",
    "louder",
    "loud",
    "quieter",
    "quiet",
    "volume",
    "increase",
    "decrease",
    "mute",
    "unmute",
    "silent",
    "sound",
    "hear",
    "hearing",
    "reminder",
    "streaming",
    "memory"
]


def apply_one_typo(text):

    words = text.split()

    candidates = [
        i
        for i, word in enumerate(words)
        if len(word) >= 4
    ]

    if not candidates:
        return text

    i = random.choice(
        candidates
    )

    word = words[i]

    variants = typo_variants(
        word
    )

    if not variants:
        return text

    words[i] = random.choice(
        variants
    )

    return " ".join(words)


# Generate typo examples from:
# 1) contextual examples
# 2) selected original examples containing useful words
typo_source = []

for example in context_examples:
    # context_examples already contains (text, intent) tuples.
    typo_source.append(example)

selected_original = df[
    df["text"].str.lower().str.contains(
        "tomorrow|louder|quieter|mute|unmute|volume|sound|silent",
        regex=True,
        na=False
    )
]

for _, row in selected_original.iterrows():
    typo_source.append(
        (
            row["text"],
            row["intent"]
        )
    )

typo_rows = []

for text, intent in typo_source:

    for _ in range(2):

        typo_text = apply_one_typo(
            text
        )

        if typo_text != text:
            typo_rows.append(
                (
                    typo_text,
                    intent
                )
            )

# Explicit important variants.
explicit_typos = [
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
        "loudar",
        "device.volume.increase"
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
        "mutt",
        "device.volume.mute"
    ),
    (
        "unmut",
        "device.volume.unmute"
    )
]

typo_rows.extend(
    explicit_typos
)

typo_df = pd.DataFrame(
    typo_rows,
    columns=[
        "text",
        "intent"
    ]
).drop_duplicates()

print(
    "\nGenerated typo examples:",
    len(typo_df)
)


# ============================================================
# 4. COMBINE DATA
# ============================================================

original_keys = set(
    zip(
        df["text"].str.lower(),
        df["intent"]
    )
)

context_df = pd.DataFrame(
    context_examples,
    columns=[
        "text",
        "intent"
    ]
)

context_df = context_df[
    ~context_df.apply(
        lambda r:
            (
                r["text"].lower(),
                r["intent"]
            ) in original_keys,
        axis=1
    )
].reset_index(drop=True)

train_df = pd.concat(
    [
        df,
        context_df,
        typo_df
    ],
    ignore_index=True
)

train_df = train_df[
    ["text", "intent"]
].drop_duplicates(
    ["text", "intent"]
).reset_index(drop=True)

print(
    "Contextual examples:",
    len(context_df)
)

print(
    "Total training rows:",
    len(train_df)
)


# ============================================================
# 5. TRAIN BPE TOKENIZER
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

tokenizer.post_processor = (
    TemplateProcessing(
        single="<CLS> $A <SEP>",
        special_tokens=[
            (
                "<CLS>",
                tokenizer.token_to_id(
                    "<CLS>"
                )
            ),
            (
                "<SEP>",
                tokenizer.token_to_id(
                    "<SEP>"
                )
            )
        ]
    )
)

tokenizer.save(
    tokenizer_path
)

actual_vocab_size = (
    tokenizer.get_vocab_size()
)

print(
    "Actual BPE vocabulary:",
    actual_vocab_size
)


# ============================================================
# 6. TOKENIZER DEBUG
# ============================================================

print("\nTokenizer debug examples:")

debug_examples = [
    "tomorrow",
    "tommorow",
    "tomorow",
    "louder",
    "loudar",
    "quieter",
    "quiter",
    "i can still hear it make it completely silent",
    "it's quieter can you make it a little louder",
    "turn off"
]

for text in debug_examples:

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
# 7. ENCODING
# ============================================================

PAD_ID = tokenizer.token_to_id(
    "<PAD>"
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
# 8. LOAD V3 HYBRID TEACHER
# ============================================================

print("\nLoading V3 hybrid teacher...")

if not os.path.exists(
    TEACHER_TFIDF
):
    raise FileNotFoundError(
        "Teacher TF-IDF not found:\n"
        + TEACHER_TFIDF
    )

if not os.path.exists(
    TEACHER_CLASSIFIER
):
    raise FileNotFoundError(
        "Teacher classifier not found:\n"
        + TEACHER_CLASSIFIER
    )

tfidf = joblib.load(
    TEACHER_TFIDF
)

teacher_clf = joblib.load(
    TEACHER_CLASSIFIER
)

teacher_encoder = (
    SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
)

print(
    "Creating teacher embeddings..."
)

X_tfidf = tfidf.transform(
    train_df["text"]
)

embeddings = teacher_encoder.encode(
    train_df["text"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

X_semantic = csr_matrix(
    embeddings
)

X_teacher = hstack(
    [
        X_tfidf,
        X_semantic * 2.0
    ]
).tocsr()

teacher_raw = (
    teacher_clf.predict_proba(
        X_teacher
    )
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
    ] = teacher_raw[:, j]


# ============================================================
# 9. OVERRIDE TEACHER ON EXPLICIT HARD EXAMPLES
#
# For contextual and typo examples, hard label should dominate.
# ============================================================

context_start = len(df)

context_end = (
    context_start
    + len(context_df)
)

# Contextual examples
for i in range(
    context_start,
    context_end
):

    true_id = label_to_id[
        train_df.iloc[i]["intent"]
    ]

    teacher_probs[i] *= 0.15

    teacher_probs[
        i,
        true_id
    ] += 0.85

    teacher_probs[i] /= (
        teacher_probs[i].sum()
    )


# Typo examples
for i in range(
    context_end,
    len(train_df)
):

    true_id = label_to_id[
        train_df.iloc[i]["intent"]
    ]

    teacher_probs[i] *= 0.20

    teacher_probs[
        i,
        true_id
    ] += 0.80

    teacher_probs[i] /= (
        teacher_probs[i].sum()
    )


# ============================================================
# 10. TRAIN / VALIDATION SPLIT
# ============================================================

indices = np.arange(
    len(train_df)
)

train_idx, val_idx = (
    train_test_split(
        indices,
        test_size=0.10,
        random_state=SEED,
        stratify=train_df["intent"]
    )
)


class TextDataset(
    Dataset
):

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
        k
    ):

        i = int(
            self.indices[k]
        )

        return (
            torch.tensor(
                encode_text(
                    train_df.iloc[i][
                        "text"
                    ]
                ),
                dtype=torch.long
            ),
            torch.tensor(
                label_to_id[
                    train_df.iloc[i][
                        "intent"
                    ]
                ],
                dtype=torch.long
            ),
            torch.tensor(
                teacher_probs[i],
                dtype=torch.float32
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

print(
    "\nTrain:",
    len(train_idx)
)

print(
    "Validation:",
    len(val_idx)
)


# ============================================================
# 11. MODEL
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

        encoder_layer = (
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
                encoder_layer,
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
        ).clamp(
            min=1.0
        )

        return self.classifier(
            self.norm(pooled)
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
# 12. DISTILLATION LOSS
# ============================================================

hard_ce = nn.CrossEntropyLoss()


def distillation_loss(
    logits,
    hard_labels,
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
        * (
            TEMPERATURE
            ** 2
        )
    )

    hard_loss = hard_ce(
        logits,
        hard_labels
    )

    return (
        ALPHA * soft_loss
        + BETA * hard_loss
    )


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# 13. TRAIN
# ============================================================

best_f1 = -1.0
best_state = None

print("\nTraining...")

for epoch in range(
    EPOCHS
):

    model.train()

    running_loss = 0.0

    for (
        x,
        y,
        teacher
    ) in train_loader:

        x = x.to(
            DEVICE
        )

        y = y.to(
            DEVICE
        )

        teacher = teacher.to(
            DEVICE
        )

        optimizer.zero_grad()

        logits = model(
            x
        )

        loss = distillation_loss(
            logits,
            y,
            teacher
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

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for (
            x,
            y,
            _
        ) in val_loader:

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
# 14. SAVE MODEL
# ============================================================

weights_path = os.path.join(
    OUT_DIR,
    "student_v2_fp32.pt"
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
            "temperature": TEMPERATURE,
            "alpha": ALPHA,
            "beta": BETA
        },
        f,
        indent=2
    )

print(
    "\nSaved FP32 model:",
    weights_path
)


# ============================================================
# 15. EVALUATION HELPER
# ============================================================

def predict_texts(
    texts
):

    x = torch.tensor(
        np.asarray(
            [
                encode_text(text)
                for text in texts
            ],
            dtype=np.int64
        ),
        dtype=torch.long
    )

    predictions = []
    confidences = []

    with torch.no_grad():

        for start in range(
            0,
            len(x),
            BATCH_SIZE
        ):

            logits = model(
                x[
                    start:
                    start + BATCH_SIZE
                ].to(DEVICE)
            )

            probs = torch.softmax(
                logits,
                dim=1
            )

            conf, ids = torch.max(
                probs,
                dim=1
            )

            predictions.extend(
                labels[int(i)]
                for i in ids.cpu().numpy()
            )

            confidences.extend(
                conf.cpu().numpy()
            )

    return (
        predictions,
        confidences
    )


# ============================================================
# 16. EXISTING 595 UNSEEN TEST
# ============================================================

print("\n")
print("=" * 70)
print("V2 — UNSEEN SEMANTIC STRESS TEST")
print("=" * 70)

stress = pd.read_csv(
    STRESS_FILE
)

stress_preds, stress_conf = (
    predict_texts(
        stress["text"].tolist()
    )
)

unseen_acc = accuracy_score(
    stress["intent"],
    stress_preds
)

unseen_f1 = f1_score(
    stress["intent"],
    stress_preds,
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
        stress_preds,
        digits=4
    )
)

stress_out = stress.copy()

stress_out[
    "predicted_intent"
] = stress_preds

stress_out[
    "confidence"
] = stress_conf

stress_out["correct"] = (
    stress_out["intent"]
    == stress_out[
        "predicted_intent"
    ]
)

stress_out.to_csv(
    os.path.join(
        SCRIPT_DIR,
        "subword_v2_unseen_predictions.csv"
    ),
    index=False
)

stress_out[
    ~stress_out["correct"]
].to_csv(
    os.path.join(
        SCRIPT_DIR,
        "subword_v2_unseen_errors.csv"
    ),
    index=False
)


# ============================================================
# 17. DEDICATED CONTEXTUAL TEST
# ============================================================

CONTEXT_TEST = [
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

context_texts = [
    x[0]
    for x in CONTEXT_TEST
]

context_truth = [
    x[1]
    for x in CONTEXT_TEST
]

context_preds, context_conf = (
    predict_texts(
        context_texts
    )
)

context_acc = accuracy_score(
    context_truth,
    context_preds
)

context_f1 = f1_score(
    context_truth,
    context_preds,
    average="macro"
)

print("\n")
print("=" * 70)
print("V2 — NEW CONTEXTUAL TEST")
print("=" * 70)

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

print(
    classification_report(
        context_truth,
        context_preds,
        digits=4
    )
)

context_out = pd.DataFrame(
    {
        "text": context_texts,
        "expected": context_truth,
        "predicted": context_preds,
        "confidence": context_conf,
        "correct": [
            a == b
            for a, b in zip(
                context_truth,
                context_preds
            )
        ]
    }
)

context_out.to_csv(
    os.path.join(
        SCRIPT_DIR,
        "subword_v2_contextual_predictions.csv"
    ),
    index=False
)


# ============================================================
# 18. TYPO TEST
# ============================================================

TYPO_TEST = [
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
        "mute",
        "device.volume.mute"
    ),
    (
        "mutt",
        "device.volume.mute"
    ),
    (
        "unmute",
        "device.volume.unmute"
    ),
    (
        "unmut",
        "device.volume.unmute"
    ),
]

typo_texts = [
    x[0]
    for x in TYPO_TEST
]

typo_truth = [
    x[1]
    for x in TYPO_TEST
]

typo_preds, typo_conf = (
    predict_texts(
        typo_texts
    )
)

typo_acc = accuracy_score(
    typo_truth,
    typo_preds
)

print("\n")
print("=" * 70)
print("V2 — TYPO ROBUSTNESS TEST")
print("=" * 70)

print(
    "Accuracy:",
    round(
        typo_acc * 100,
        2
    ),
    "%"
)

for (
    text,
    expected,
    predicted,
    confidence
) in zip(
    typo_texts,
    typo_truth,
    typo_preds,
    typo_conf
):

    print(
        f"{'OK' if expected == predicted else 'WRONG':5} | "
        f"{text:12} | "
        f"Expected: {expected:30} | "
        f"Predicted: {predicted:30} | "
        f"{confidence * 100:.2f}%"
    )

typo_out = pd.DataFrame(
    {
        "text": typo_texts,
        "expected": typo_truth,
        "predicted": typo_preds,
        "confidence": typo_conf,
        "correct": [
            a == b
            for a, b in zip(
                typo_truth,
                typo_preds
            )
        ]
    }
)

typo_out.to_csv(
    os.path.join(
        SCRIPT_DIR,
        "subword_v2_typo_predictions.csv"
    ),
    index=False
)


# ============================================================
# 19. EXACT PROBLEMATIC SENTENCES
# ============================================================

EXACT_TEST = [
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
    )
]

exact_texts = [
    x[0]
    for x in EXACT_TEST
]

exact_truth = [
    x[1]
    for x in EXACT_TEST
]

exact_preds, exact_conf = (
    predict_texts(
        exact_texts
    )
)

print("\n")
print("=" * 70)
print("V2 — EXACT PROBLEM CASES")
print("=" * 70)

for (
    text,
    expected,
    predicted,
    confidence
) in zip(
    exact_texts,
    exact_truth,
    exact_preds,
    exact_conf
):

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
        "Expected:",
        expected
    )

    print(
        "Predicted:",
        predicted
    )

    print(
        "Confidence:",
        f"{confidence * 100:.2f}%"
    )

# ============================================================
# 20. FINAL SUMMARY
# ============================================================

fp32_size = (
    os.path.getsize(
        weights_path
    )
    / 1024
    / 1024
)

print("\n")
print("=" * 70)
print("SUBWORD STUDENT V2 — FINAL SUMMARY")
print("=" * 70)

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
        context_acc * 100,
        2
    ),
    "%"
)

print(
    "Typo accuracy:",
    round(
        typo_acc * 100,
        2
    ),
    "%"
)

print(
    "\nSaved directory:"
)

print(
    OUT_DIR
)

print(
    "\nCurrent INT8 model was NOT modified."
)

print(
    "Do NOT export to ONNX/INT8 yet."
)

print(
    "First compare V2 against the current INT8 baseline."
)
