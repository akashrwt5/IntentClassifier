import os
import random
import numpy as np
import pandas as pd
import joblib
import torch

from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ============================================================
# CONFIG
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_FILE = "semantic_training_v3_hard_negatives.xlsx"
DATASET = os.path.join(SCRIPT_DIR, DATASET_FILE) if os.path.exists(os.path.join(SCRIPT_DIR, DATASET_FILE)) else DATASET_FILE

STRESS_FILENAME = "unseen_semantic_stress_test.csv"
STRESS_FILE = os.path.join(SCRIPT_DIR, STRESS_FILENAME) if os.path.exists(os.path.join(SCRIPT_DIR, STRESS_FILENAME)) else STRESS_FILENAME

TEACHER_TFIDF_FILE = "v3_hybrid_tfidf.joblib"
TEACHER_TFIDF = os.path.join(SCRIPT_DIR, TEACHER_TFIDF_FILE) if os.path.exists(os.path.join(SCRIPT_DIR, TEACHER_TFIDF_FILE)) else TEACHER_TFIDF_FILE

TEACHER_CLASSIFIER_FILE = "v3_hybrid_classifier.joblib"
TEACHER_CLASSIFIER = os.path.join(SCRIPT_DIR, TEACHER_CLASSIFIER_FILE) if os.path.exists(os.path.join(SCRIPT_DIR, TEACHER_CLASSIFIER_FILE)) else TEACHER_CLASSIFIER_FILE

# Small semantic student.
# BERT-tiny is ~4.4M parameters; INT8 should be comfortably below 15 MB.
STUDENT_NAME = "prajjwal1/bert-tiny"

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "student_bert_tiny_v1")

SEED = 42
BATCH_SIZE = 32
EPOCHS = 5
LR = 3e-5

TEMPERATURE = 2.0
ALPHA = 0.70       # soft teacher loss
BETA = 0.30        # hard-label loss

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Device:", DEVICE)


# ============================================================
# 1. LOAD TRAINING DATA
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
# 2. LOAD V3 TEACHER
# ============================================================

print("\nLoading V3 teacher...")

tfidf = joblib.load(TEACHER_TFIDF)
teacher_clf = joblib.load(TEACHER_CLASSIFIER)

teacher_encoder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

SEMANTIC_WEIGHT = 2.0


# ============================================================
# 3. CREATE TEACHER SOFT TARGETS
# ============================================================

print("\nCreating teacher soft targets...")

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

teacher_probs = teacher_clf.predict_proba(X_teacher)

# Ensure class order matches our label mapping.
teacher_class_order = list(teacher_clf.classes_)

teacher_probs_ordered = np.zeros(
    (len(df), len(labels)),
    dtype=np.float32
)

for teacher_index, label in enumerate(teacher_class_order):
    student_index = label_to_id[label]
    teacher_probs_ordered[:, student_index] = teacher_probs[:, teacher_index]

teacher_probs = torch.tensor(
    teacher_probs_ordered,
    dtype=torch.float32
)


# ============================================================
# 4. DATASET
# ============================================================

class DistillDataset(Dataset):

    def __init__(self, dataframe, soft_targets):

        self.texts = dataframe["text"].tolist()

        self.labels = torch.tensor(
            [
                label_to_id[x]
                for x in dataframe["intent"]
            ],
            dtype=torch.long
        )

        self.soft_targets = soft_targets

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):

        return {
            "text": self.texts[index],
            "label": self.labels[index],
            "soft": self.soft_targets[index]
        }


dataset = DistillDataset(
    df,
    teacher_probs
)


# ============================================================
# 5. STUDENT TOKENIZER + MODEL
# ============================================================

print("\nLoading student:", STUDENT_NAME)

tokenizer = BertTokenizer.from_pretrained(
    STUDENT_NAME
)

student = AutoModelForSequenceClassification.from_pretrained(
    STUDENT_NAME,
    num_labels=len(labels),
    ignore_mismatched_sizes=True
)

student.to(DEVICE)


# ============================================================
# 6. COLLATE FUNCTION
# ============================================================

def collate(batch):

    texts = [x["text"] for x in batch]

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )

    labels_batch = torch.stack(
        [x["label"] for x in batch]
    )

    soft_batch = torch.stack(
        [x["soft"] for x in batch]
    )

    return encoded, labels_batch, soft_batch


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate
)


# ============================================================
# 7. DISTILLATION LOSS
# ============================================================

hard_loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    student.parameters(),
    lr=LR,
    weight_decay=0.01
)


def distillation_loss(
    student_logits,
    hard_labels,
    teacher_probabilities
):

    # Teacher probabilities are softened with temperature.
    teacher_soft = torch.softmax(
        torch.log(
            teacher_probabilities.clamp(min=1e-8)
        ) / TEMPERATURE,
        dim=-1
    )

    student_log_soft = torch.log_softmax(
        student_logits / TEMPERATURE,
        dim=-1
    )

    soft_loss = torch.nn.functional.kl_div(
        student_log_soft,
        teacher_soft,
        reduction="batchmean"
    ) * (TEMPERATURE ** 2)

    hard_loss = hard_loss_fn(
        student_logits,
        hard_labels
    )

    return (
        ALPHA * soft_loss +
        BETA * hard_loss
    )


# ============================================================
# 8. TRAIN
# ============================================================

print("\nStarting knowledge distillation...")

student.train()

for epoch in range(EPOCHS):

    total_loss = 0.0

    for step, (encoded, hard_labels, soft_targets) in enumerate(loader):

        encoded = {
            k: v.to(DEVICE)
            for k, v in encoded.items()
        }

        hard_labels = hard_labels.to(DEVICE)
        soft_targets = soft_targets.to(DEVICE)

        optimizer.zero_grad()

        outputs = student(**encoded)

        loss = distillation_loss(
            outputs.logits,
            hard_labels,
            soft_targets
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            student.parameters(),
            1.0
        )

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"Loss: {total_loss / len(loader):.4f}"
    )


# ============================================================
# 9. SAVE STUDENT
# ============================================================

student.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(
    os.path.join(OUTPUT_DIR, "intent_labels.txt"),
    "w"
) as f:
    for label in labels:
        f.write(label + "\n")

print("\nStudent saved:", OUTPUT_DIR)


# ============================================================
# 10. EVALUATE UNSEEN SEMANTIC TEST
# ============================================================

print("\nEvaluating unseen semantic stress test...")

stress = pd.read_csv(STRESS_FILE)

student.eval()

predictions = []

with torch.no_grad():

    for start in range(
        0,
        len(stress),
        BATCH_SIZE
    ):

        texts = stress["text"].iloc[
            start:start + BATCH_SIZE
        ].tolist()

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        encoded = {
            k: v.to(DEVICE)
            for k, v in encoded.items()
        }

        outputs = student(**encoded)

        pred_ids = torch.argmax(
            outputs.logits,
            dim=1
        ).cpu().numpy()

        predictions.extend(
            labels[i]
            for i in pred_ids
        )


accuracy = accuracy_score(
    stress["intent"],
    predictions
)

macro_f1 = f1_score(
    stress["intent"],
    predictions,
    average="macro"
)

print("\n================================")
print("STUDENT UNSEEN SEMANTIC TEST")
print("================================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
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
    os.path.join(SCRIPT_DIR, "student_unseen_semantic_predictions.csv"),
    index=False
)

results[
    ~results["correct"]
].to_csv(
    os.path.join(SCRIPT_DIR, "student_unseen_semantic_errors.csv"),
    index=False
)


# ============================================================
# 11. MODEL SIZE
# ============================================================

def directory_size_mb(directory):

    total = 0

    for root, _, files in os.walk(directory):

        for file in files:

            path = os.path.join(
                root,
                file
            )

            total += os.path.getsize(path)

    return total / (
        1024 * 1024
    )


size_mb = directory_size_mb(
    OUTPUT_DIR
)

print(
    "\nFP32/HuggingFace student directory size:",
    round(size_mb, 2),
    "MB"
)

print(
    "\nNext target: INT8 ONNX < 15 MB"
)

print("\nFiles generated:")
print("student_unseen_semantic_predictions.csv")
print("student_unseen_semantic_errors.csv")
print(OUTPUT_DIR + "/")
