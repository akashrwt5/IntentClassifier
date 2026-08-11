import os
import random
import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

DATASET = "semantic_balanced_dataset.xlsx"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

OUTPUT_DIR = "supervised_hearing_aid_model"

SEED = 42
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-5

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print("Using device:", DEVICE)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_excel(
    DATASET,
    sheet_name="dataset"
)

df = df[["text", "intent"]].dropna()

df["text"] = df["text"].astype(str).str.strip()
df["intent"] = df["intent"].astype(str).str.strip()

print("Total samples:", len(df))
print("Total intents:", df["intent"].nunique())


# ============================================================
# INTENT LABELS
# ============================================================

intent_names = sorted(df["intent"].unique())

intent_to_id = {
    intent: i
    for i, intent in enumerate(intent_names)
}

id_to_intent = {
    i: intent
    for i, intent in enumerate(intent_names)
}

print("\nIntent labels:")

for i, intent in enumerate(intent_names):
    print(i, "->", intent)


# ============================================================
# TRAIN / VALIDATION / TEST
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=SEED,
    stratify=df["intent"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_df["intent"]
)

print("\nDataset split:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# ============================================================
# DATASET CLASS
# ============================================================

class IntentDataset(Dataset):

    def __init__(self, dataframe):

        self.texts = dataframe["text"].tolist()

        self.labels = [
            intent_to_id[x]
            for x in dataframe["intent"]
        ]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):

        return (
            self.texts[index],
            torch.tensor(
                self.labels[index],
                dtype=torch.long
            )
        )


train_dataset = IntentDataset(train_df)
val_dataset = IntentDataset(val_df)
test_dataset = IntentDataset(test_df)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading MiniLM...")

encoder = SentenceTransformer(
    MODEL_NAME
)

embedding_dimension = encoder.get_sentence_embedding_dimension()

print(
    "Embedding dimension:",
    embedding_dimension
)


class SemanticClassifier(nn.Module):

    def __init__(
        self,
        encoder,
        embedding_dimension,
        num_classes
    ):

        super().__init__()

        self.encoder = encoder

        self.classifier = nn.Sequential(

            nn.Linear(
                embedding_dimension,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                256,
                num_classes
            )
        )

    def forward(self, texts):

        embeddings = self.encoder.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=False
        ).clone()

        logits = self.classifier(
            embeddings
        )

        return logits


model = SemanticClassifier(
    encoder,
    embedding_dimension,
    len(intent_names)
)

model = model.to(DEVICE)


# ============================================================
# HARD NEGATIVE PAIR WEIGHTS
# ============================================================

# These are the most difficult intent boundaries
# identified from our previous experiments.

HARD_INTENTS = {
    "device.volume.increase",
    "device.volume.decrease",
    "device.volume.mute",
    "device.volume.unmute",
}

# Give these classes slightly higher importance.
class_weights = torch.ones(
    len(intent_names),
    dtype=torch.float32
)

for intent in HARD_INTENTS:

    if intent in intent_to_id:

        class_weights[
            intent_to_id[intent]
        ] = 1.5

class_weights = class_weights.to(DEVICE)

print("\nClass weights:")

for intent, idx in intent_to_id.items():

    print(
        intent,
        "->",
        float(class_weights[idx])
    )


# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(loader):

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for texts, labels in loader:

            logits = model(texts)

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    return (
        accuracy,
        macro_f1,
        all_labels,
        all_predictions
    )


# ============================================================
# TRAIN
# ============================================================

best_val_f1 = 0.0

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("\nStarting training...")


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for texts, labels in train_loader:

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits = model(texts)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

    avg_loss = (
        total_loss /
        len(train_loader)
    )

    (
        val_accuracy,
        val_f1,
        _,
        _
    ) = evaluate(val_loader)

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print(
        "Loss:",
        round(avg_loss, 4)
    )

    print(
        "Validation Accuracy:",
        round(val_accuracy * 100, 2),
        "%"
    )

    print(
        "Validation Macro F1:",
        round(val_f1 * 100, 2),
        "%"
    )

    # Save best model
    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        torch.save(
            model.state_dict(),
            os.path.join(
                OUTPUT_DIR,
                "best_model.pt"
            )
        )

        print(
            "Best model saved."
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

model.load_state_dict(
    torch.load(
        os.path.join(
            OUTPUT_DIR,
            "best_model.pt"
        ),
        map_location=DEVICE
    )
)


# ============================================================
# FINAL TEST
# ============================================================

(
    test_accuracy,
    test_f1,
    test_labels,
    test_predictions
) = evaluate(test_loader)


print("\n================================")
print("SUPERVISED SEMANTIC RESULTS")
print("================================")

print(
    "Test Accuracy:",
    round(test_accuracy * 100, 2),
    "%"
)

print(
    "Test Macro F1:",
    round(test_f1 * 100, 2),
    "%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

print(
    classification_report(
        test_labels,
        test_predictions,
        labels=list(range(len(intent_names))),
        target_names=intent_names,
        digits=4
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_labels,
    test_predictions
)

cm_df = pd.DataFrame(
    cm,
    index=intent_names,
    columns=intent_names
)

print("\nConfusion Matrix:\n")

print(cm_df)


# ============================================================
# SAVE RESULTS
# ============================================================

test_results = test_df.copy()

test_results["predicted_intent"] = [
    id_to_intent[int(x)]
    for x in test_predictions
]

test_results.to_csv(
    "supervised_test_predictions.csv",
    index=False
)

cm_df.to_csv(
    "supervised_confusion_matrix.csv"
)

print("\nFiles generated:")

print("supervised_test_predictions.csv")
print("supervised_confusion_matrix.csv")
print(
    OUTPUT_DIR +
    "/best_model.pt"
)
