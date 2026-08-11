
import os, json, random
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
# LARGE STUDENT TEMPLATE
# This script trains a compact Transformer student by
# distilling the V3 hybrid teacher while preserving the
# contextual-action hard-negative examples.
#
# Expected files in this same folder:
#   semantic_training_v3_hard_negatives.xlsx
#   unseen_semantic_stress_test.csv
#   v3_hybrid_tfidf.joblib
#   v3_hybrid_classifier.joblib
#
# Run one script for Model B or Model C.
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET = os.path.join(SCRIPT_DIR, "semantic_training_v3_hard_negatives.xlsx")
STRESS_FILE = os.path.join(SCRIPT_DIR, "unseen_semantic_stress_test.csv")
TEACHER_TFIDF = os.path.join(SCRIPT_DIR, "v3_hybrid_tfidf.joblib")
TEACHER_CLASSIFIER = os.path.join(SCRIPT_DIR, "v3_hybrid_classifier.joblib")

SEED = 42
BATCH_SIZE = 64
EPOCHS = 14
LR = 1.0e-3
WEIGHT_DECAY = 1e-4
TEMPERATURE = 2.0
ALPHA = 0.35
BETA = 0.65

# ------------------------------------------------------------
# MODEL CONFIGURATION
# These four values are replaced by Model B / Model C.
# ------------------------------------------------------------
EMBED_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 4
FF_DIM = 256

MAX_LEN = 32
VOCAB_LIMIT = 12000
DROPOUT = 0.10

MODEL_NAME = "MODEL B — MEDIUM STUDENT"
OUT_DIR = os.path.join(SCRIPT_DIR, "tiny_semantic_student_model_b")

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 60)
print(MODEL_NAME)
print("=" * 60)
print("Device:", DEVICE)
print("Embedding:", EMBED_DIM)
print("Layers:", NUM_LAYERS)
print("Heads:", NUM_HEADS)
print("FFN:", FF_DIM)
print("Max length:", MAX_LEN)
print("Vocabulary limit:", VOCAB_LIMIT)

# ------------------------------------------------------------
# Load V3 training data
# ------------------------------------------------------------
df = pd.read_excel(DATASET, sheet_name="dataset")
df = df[["text", "intent"]].dropna()
df["text"] = df["text"].astype(str).str.strip()
df["intent"] = df["intent"].astype(str).str.strip()
df = df[df["text"] != ""].drop_duplicates(["text", "intent"]).reset_index(drop=True)

labels = sorted(df["intent"].unique())
label_to_id = {x: i for i, x in enumerate(labels)}

print("\nOriginal samples:", len(df))
print("Intents:", len(labels))

# ------------------------------------------------------------
# Contextual-action hard negatives
# These are deliberately different from the original stress
# set and are used only for training.
# ------------------------------------------------------------
context_examples = [
("it's quieter can you make it a little louder","device.volume.increase"),
("the sound feels low could you turn it up slightly","device.volume.increase"),
("things got quieter please raise the volume","device.volume.increase"),
("i can barely hear it make the sound stronger","device.volume.increase"),
("the audio is too soft turn it up a little","device.volume.increase"),
("it sounds low compared with before increase it","device.volume.increase"),
("the volume dropped can you increase it","device.volume.increase"),
("the hearing aids sound quiet please turn them up","device.volume.increase"),
("it's louder than before bring the volume down","device.volume.decrease"),
("the sound feels high could you lower it slightly","device.volume.decrease"),
("things got louder please reduce the volume","device.volume.decrease"),
("the audio is too strong turn it down a little","device.volume.decrease"),
("it sounds loud compared with before decrease it","device.volume.decrease"),
("the volume increased can you lower it","device.volume.decrease"),
("the hearing aids sound loud please turn them down","device.volume.decrease"),
("i can still hear sound make it completely silent","device.volume.mute"),
("the sound is already low but i want total silence","device.volume.mute"),
("it is still audible turn everything off","device.volume.mute"),
("nothing is muted yet make the hearing aids silent","device.volume.mute"),
("the sound is off bring it back","device.volume.unmute"),
("it is silent restore the sound","device.volume.unmute"),
("there is no audio please turn it back on","device.volume.unmute"),
("the hearing aids are muted bring the sound back","device.volume.unmute"),
]

context_df = pd.DataFrame(context_examples, columns=["text","intent"])

original_keys = set(zip(df["text"].str.lower(), df["intent"]))
context_df = context_df[
    ~context_df.apply(lambda r: (r["text"].lower(), r["intent"]) in original_keys, axis=1)
].reset_index(drop=True)

train_df = pd.concat([df, context_df], ignore_index=True)

print("Contextual training examples:", len(context_df))
print("Total training rows:", len(train_df))

# ------------------------------------------------------------
# Vocabulary
# ------------------------------------------------------------
counts = {}
for text in train_df["text"]:
    for tok in text.lower().split():
        tok = tok.strip(".,!?;:\"'()[]{}")
        if tok:
            counts[tok] = counts.get(tok, 0) + 1

vocab = {"<PAD>": 0, "<UNK>": 1}

for tok, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:VOCAB_LIMIT - 2]:
    if tok not in vocab:
        vocab[tok] = len(vocab)

print("Actual vocabulary size:", len(vocab))

def tokenize(text):
    ids = []
    for tok in str(text).lower().split():
        tok = tok.strip(".,!?;:\"'()[]{}")
        if tok:
            ids.append(vocab.get(tok, 1))
    ids = ids[:MAX_LEN]
    ids += [0] * (MAX_LEN - len(ids))
    return ids

# ------------------------------------------------------------
# Teacher
# ------------------------------------------------------------
print("\nLoading V3 teacher...")
tfidf = joblib.load(TEACHER_TFIDF)
teacher_clf = joblib.load(TEACHER_CLASSIFIER)
teacher_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

X_tfidf = tfidf.transform(train_df["text"])
emb = teacher_encoder.encode(
    train_df["text"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)
X_teacher = hstack([X_tfidf, csr_matrix(emb) * 2.0]).tocsr()

raw = teacher_clf.predict_proba(X_teacher)
teacher_probs = np.zeros((len(train_df), len(labels)), dtype=np.float32)

for j, lab in enumerate(teacher_clf.classes_):
    teacher_probs[:, label_to_id[lab]] = raw[:, j]

# Make explicit contextual hard labels authoritative.
context_start = len(df)
for i in range(context_start, len(train_df)):
    true_id = label_to_id[train_df.iloc[i]["intent"]]
    teacher_probs[i] *= 0.20
    teacher_probs[i, true_id] += 0.80
    teacher_probs[i] /= teacher_probs[i].sum()

# ------------------------------------------------------------
# Split
# ------------------------------------------------------------
idx = np.arange(len(train_df))
train_idx, val_idx = train_test_split(
    idx, test_size=0.10, random_state=SEED,
    stratify=train_df["intent"]
)

class TextDataset(Dataset):
    def __init__(self, indices):
        self.indices = np.asarray(indices)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, k):
        i = int(self.indices[k])
        return (
            torch.tensor(tokenize(train_df.iloc[i]["text"]), dtype=torch.long),
            torch.tensor(label_to_id[train_df.iloc[i]["intent"]], dtype=torch.long),
            torch.tensor(teacher_probs[i], dtype=torch.float32)
        )

train_loader = DataLoader(TextDataset(train_idx), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TextDataset(val_idx), batch_size=BATCH_SIZE, shuffle=False)

# ------------------------------------------------------------
# Student model
# ------------------------------------------------------------
class TinySemanticStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), EMBED_DIM, padding_idx=0)
        self.position = nn.Embedding(MAX_LEN, EMBED_DIM)

        layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=NUM_LAYERS)
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.classifier = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(EMBED_DIM, len(labels))
        )

    def forward(self, x):
        pad = x.eq(0)
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.position(pos)
        h = self.encoder(h, src_key_padding_mask=pad)

        valid = (~pad).unsqueeze(-1).float()
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return self.classifier(self.norm(pooled))

student = TinySemanticStudent().to(DEVICE)

params = sum(p.numel() for p in student.parameters() if p.requires_grad)
print("Trainable parameters:", params)
print("Approx FP32 weights:", round(params * 4 / 1024 / 1024, 3), "MB")

# ------------------------------------------------------------
# Distillation
# ------------------------------------------------------------
ce = nn.CrossEntropyLoss()

def loss_fn(logits, hard, teacher_prob):
    t = torch.softmax(torch.log(teacher_prob.clamp(min=1e-8)) / TEMPERATURE, dim=-1)
    s = torch.log_softmax(logits / TEMPERATURE, dim=-1)
    soft = nn.functional.kl_div(s, t, reduction="batchmean") * (TEMPERATURE ** 2)
    hard_loss = ce(logits, hard)
    return ALPHA * soft + BETA * hard_loss

optimizer = torch.optim.AdamW(
    student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
)

best_f1 = -1.0
best_state = None

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
for epoch in range(EPOCHS):
    student.train()
    running = 0.0

    for x, y, teacher in train_loader:
        x, y, teacher = x.to(DEVICE), y.to(DEVICE), teacher.to(DEVICE)
        optimizer.zero_grad()
        logits = student(x)
        loss = loss_fn(logits, y, teacher)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        running += loss.item()

    student.eval()
    yt, yp = [], []

    with torch.no_grad():
        for x, y, _ in val_loader:
            pred = torch.argmax(student(x.to(DEVICE)), dim=1).cpu().numpy()
            yp.extend(pred)
            yt.extend(y.numpy())

    va = accuracy_score(yt, yp)
    vf = f1_score(yt, yp, average="macro")

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} "
        f"Loss={running/len(train_loader):.4f} "
        f"ValAcc={va*100:.2f}% "
        f"ValF1={vf*100:.2f}%"
    )

    if vf > best_f1:
        best_f1 = vf
        best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

student.load_state_dict(best_state)
student.eval()

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)

torch.save(student.state_dict(), os.path.join(OUT_DIR, "student_fp32.pt"))

with open(os.path.join(OUT_DIR, "vocab.json"), "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False)

with open(os.path.join(OUT_DIR, "intent_labels.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(labels) + "\n")

with open(os.path.join(OUT_DIR, "config.json"), "w", encoding="utf-8") as f:
    json.dump({
        "vocab_size": len(vocab),
        "num_classes": len(labels),
        "embed_dim": EMBED_DIM,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "ff_dim": FF_DIM,
        "max_len": MAX_LEN
    }, f, indent=2)

# ------------------------------------------------------------
# Existing 595-sample unseen test
# ------------------------------------------------------------
stress = pd.read_csv(STRESS_FILE)
X = torch.tensor(np.array([tokenize(x) for x in stress["text"]]), dtype=torch.long)

preds = []
with torch.no_grad():
    for start in range(0, len(X), BATCH_SIZE):
        logits = student(X[start:start+BATCH_SIZE].to(DEVICE))
        p = torch.argmax(logits, dim=1).cpu().numpy()
        preds.extend(labels[int(i)] for i in p)

acc = accuracy_score(stress["intent"], preds)
mf1 = f1_score(stress["intent"], preds, average="macro")

print("\n==============================================")
print(MODEL_NAME + " — UNSEEN SEMANTIC STRESS TEST")
print("==============================================")
print("Accuracy:", round(acc*100, 2), "%")
print("Macro F1:", round(mf1*100, 2), "%")
print(classification_report(stress["intent"], preds, digits=4))

out = stress.copy()
out["predicted_intent"] = preds
out["correct"] = out["intent"] == out["predicted_intent"]
out.to_csv(os.path.join(SCRIPT_DIR, "model_b_unseen_predictions.csv"), index=False)
out[~out["correct"]].to_csv(os.path.join(SCRIPT_DIR, "model_b_unseen_errors.csv"), index=False)

print("\nSaved model:", os.path.join(OUT_DIR, "student_fp32.pt"))
print("Unseen predictions:", "model_b_unseen_predictions.csv")
print("Unseen errors:", "model_b_unseen_errors.csv")
print("\nDo not replace the current INT8 model yet.")
