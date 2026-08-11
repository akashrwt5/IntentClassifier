
import os
import random
import json
import numpy as np
import pandas as pd
import joblib
import torch

from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# LIGHTWEIGHT CUSTOM STUDENT
# No bert-tiny / Hugging Face tokenizer required.
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def local(name):
    p = os.path.join(SCRIPT_DIR, name)
    return p if os.path.exists(p) else name

DATASET = local("semantic_training_v3_hard_negatives.xlsx")
STRESS_FILE = local("unseen_semantic_stress_test.csv")
TEACHER_TFIDF = local("v3_hybrid_tfidf.joblib")
TEACHER_CLASSIFIER = local("v3_hybrid_classifier.joblib")

SEED = 42
BATCH_SIZE = 64
EPOCHS = 12
LR = 2e-3
WEIGHT_DECAY = 1e-4
TEMPERATURE = 2.0
ALPHA = 0.70
BETA = 0.30

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
# 1. LOAD V3 DATASET
# ============================================================

df = pd.read_excel(DATASET, sheet_name="dataset")
df = df[["text", "intent"]].dropna()
df["text"] = df["text"].astype(str).str.strip()
df["intent"] = df["intent"].astype(str).str.strip()
df = df[df["text"] != ""].drop_duplicates(["text", "intent"]).reset_index(drop=True)

labels = sorted(df["intent"].unique())
label_to_id = {x: i for i, x in enumerate(labels)}

print("Training samples:", len(df))
print("Intents:", len(labels))

# ============================================================
# 2. BUILD A SMALL WORD VOCABULARY
# ============================================================

# Use a simple deterministic word vocabulary. This avoids a large
# external tokenizer and makes ONNX/mobile deployment straightforward.
word_counts = {}

for text in df["text"]:
    for token in text.lower().split():
        token = token.strip(".,!?;:\"'()[]{}")
        if token:
            word_counts[token] = word_counts.get(token, 0) + 1

special = ["<PAD>", "<UNK>"]
sorted_words = sorted(
    word_counts.items(),
    key=lambda x: (-x[1], x[0])
)[:VOCAB_SIZE - len(special)]

vocab = {tok: i for i, tok in enumerate(special)}
for tok, _ in sorted_words:
    if tok not in vocab:
        vocab[tok] = len(vocab)

print("Vocabulary:", len(vocab))

def tokenize(text):
    tokens = []
    for token in text.lower().split():
        token = token.strip(".,!?;:\"'()[]{}")
        if token:
            tokens.append(vocab.get(token, 1))
    tokens = tokens[:MAX_LEN]
    if len(tokens) < MAX_LEN:
        tokens += [0] * (MAX_LEN - len(tokens))
    return tokens

# ============================================================
# 3. LOAD V3 TEACHER AND CREATE SOFT TARGETS
# ============================================================

print("\nLoading V3 teacher...")

tfidf = joblib.load(TEACHER_TFIDF)
teacher_clf = joblib.load(TEACHER_CLASSIFIER)

teacher_encoder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

SEMANTIC_WEIGHT = 2.0

print("Creating teacher embeddings...")
X_tfidf = tfidf.transform(df["text"])

embeddings = teacher_encoder.encode(
    df["text"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

X_sem = csr_matrix(embeddings)

X_teacher = hstack([
    X_tfidf,
    X_sem * SEMANTIC_WEIGHT
]).tocsr()

teacher_probs_raw = teacher_clf.predict_proba(X_teacher)

teacher_probs = np.zeros(
    (len(df), len(labels)),
    dtype=np.float32
)

for j, label in enumerate(teacher_clf.classes_):
    teacher_probs[:, label_to_id[label]] = teacher_probs_raw[:, j]

# ============================================================
# 4. TRAIN/VALIDATION SPLIT
# ============================================================

indices = np.arange(len(df))

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.10,
    random_state=SEED,
    stratify=df["intent"]
)

print("Train:", len(train_idx))
print("Validation:", len(val_idx))

# ============================================================
# 5. DATASET
# ============================================================

class StudentDataset(Dataset):
    def __init__(self, indices):
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])

        return (
            torch.tensor(tokenize(df.iloc[idx]["text"]), dtype=torch.long),
            torch.tensor(label_to_id[df.iloc[idx]["intent"]], dtype=torch.long),
            torch.tensor(teacher_probs[idx], dtype=torch.float32)
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
# 6. SMALL SEMANTIC STUDENT
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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        mask = x.eq(0)

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

        pooled = (
            h * valid
        ).sum(dim=1) / valid.sum(
            dim=1
        ).clamp(min=1.0)

        pooled = self.norm(pooled)

        return self.classifier(pooled)


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

print("Student parameters:", parameters)
print(
    "Approx FP32 weights:",
    round(parameters * 4 / 1024 / 1024, 2),
    "MB"
)

# ============================================================
# 7. DISTILLATION LOSS
# ============================================================

ce = nn.CrossEntropyLoss()

def distill_loss(logits, hard, teacher_prob):

    teacher_soft = torch.softmax(
        torch.log(
            teacher_prob.clamp(min=1e-8)
        ) / TEMPERATURE,
        dim=-1
    )

    student_log = torch.log_softmax(
        logits / TEMPERATURE,
        dim=-1
    )

    soft = nn.functional.kl_div(
        student_log,
        teacher_soft,
        reduction="batchmean"
    ) * (TEMPERATURE ** 2)

    hard_loss = ce(logits, hard)

    return ALPHA * soft + BETA * hard_loss

optimizer = torch.optim.AdamW(
    student.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

# ============================================================
# 8. TRAIN WITH VALIDATION
# ============================================================

best_f1 = -1.0
best_state = None

for epoch in range(EPOCHS):

    student.train()
    total = 0.0

    for x, hard, soft in train_loader:

        x = x.to(DEVICE)
        hard = hard.to(DEVICE)
        soft = soft.to(DEVICE)

        optimizer.zero_grad()

        logits = student(x)

        loss = distill_loss(
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

        total += loss.item()

    # validation
    student.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, hard, _ in val_loader:
            logits = student(x.to(DEVICE))
            pred = torch.argmax(
                logits,
                dim=1
            ).cpu().numpy()

            y_pred.extend(pred)
            y_true.extend(hard.numpy())

    val_acc = accuracy_score(y_true, y_pred)
    val_f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"Loss={total / len(train_loader):.4f} "
        f"ValAcc={val_acc * 100:.2f}% "
        f"ValF1={val_f1 * 100:.2f}%"
    )

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = {
            k: v.detach().cpu().clone()
            for k, v in student.state_dict().items()
        }

# restore best
student.load_state_dict(best_state)
student.eval()

# ============================================================
# 9. SAVE STUDENT + VOCAB
# ============================================================

OUT_DIR = os.path.join(
    SCRIPT_DIR,
    "tiny_semantic_student_v1"
)

os.makedirs(OUT_DIR, exist_ok=True)

torch.save(
    student.state_dict(),
    os.path.join(
        OUT_DIR,
        "student_fp32.pt"
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
        f.write(label + "\n")

# Save architecture config.
config = {
    "vocab_size": len(vocab),
    "num_classes": len(labels),
    "embed_dim": EMBED_DIM,
    "num_heads": NUM_HEADS,
    "num_layers": NUM_LAYERS,
    "ff_dim": FF_DIM,
    "max_len": MAX_LEN
}

with open(
    os.path.join(
        OUT_DIR,
        "config.json"
    ),
    "w"
) as f:
    json.dump(config, f, indent=2)

# ============================================================
# 10. CLEAN UNSEEN SEMANTIC TEST
# ============================================================

print("\n================================")
print("STUDENT UNSEEN SEMANTIC TEST")
print("================================")

stress = pd.read_csv(STRESS_FILE)

x_stress = torch.tensor(
    np.array([
        tokenize(t)
        for t in stress["text"]
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

        logits = student(batch)

        pred = torch.argmax(
            logits,
            dim=1
        ).cpu().numpy()

        predictions.extend(
            labels[i]
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
    round(macro_f1 * 100, 2),
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
results["predicted_intent"] = predictions
results["correct"] = (
    results["intent"] ==
    results["predicted_intent"]
)

results.to_csv(
    os.path.join(
        SCRIPT_DIR,
        "tiny_student_unseen_predictions.csv"
    ),
    index=False
)

results[
    ~results["correct"]
].to_csv(
    os.path.join(
        SCRIPT_DIR,
        "tiny_student_unseen_errors.csv"
    ),
    index=False
)

# ============================================================
# 11. SIZE
# ============================================================

def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)

model_path = os.path.join(
    OUT_DIR,
    "student_fp32.pt"
)

print(
    "\nFP32 student weights:",
    round(file_size_mb(model_path), 3),
    "MB"
)

print(
    "Student directory:",
    OUT_DIR
)

print("\nGenerated:")
print("tiny_semantic_student_v1/student_fp32.pt")
print("tiny_semantic_student_v1/vocab.json")
print("tiny_semantic_student_v1/config.json")
print("tiny_semantic_student_v1/intent_labels.txt")
print("tiny_student_unseen_predictions.csv")
print("tiny_student_unseen_errors.csv")

print(
    "\nNext step if accuracy is acceptable: "
    "export this student directly to ONNX and INT8."
)
