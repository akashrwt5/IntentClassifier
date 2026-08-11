import pandas as pd
import numpy as np
import torch

from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ============================================================
# CONFIG
# ============================================================

DATASET = "semantic_balanced_dataset.xlsx"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

OUTPUT_MODEL = "hearing_aid_semantic_model"

BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5


# ============================================================
# 1. LOAD DATASET
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
# 2. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["intent"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["intent"]
)

print("\nDataset split:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# ============================================================
# 3. LOAD MINI-LM
# ============================================================

print("\nLoading MiniLM...")

model = SentenceTransformer(
    MODEL_NAME
)

print("Model loaded.")


# ============================================================
# 4. CREATE CLASSIFICATION LABELS
# ============================================================

intent_names = sorted(
    train_df["intent"].unique()
)

intent_to_id = {
    intent: i
    for i, intent in enumerate(intent_names)
}

id_to_intent = {
    i: intent
    for intent, i in intent_to_id.items()
}

print("\nIntent mapping:")

for intent, idx in intent_to_id.items():
    print(idx, "->", intent)


# ============================================================
# 5. CREATE TRAINING EXAMPLES
# ============================================================

train_examples = []

for _, row in train_df.iterrows():

    text = row["text"]
    intent = row["intent"]

    train_examples.append(
        InputExample(
            texts=[text],
            label=intent_to_id[intent]
        )
    )


# ============================================================
# 6. DATA LOADER
# ============================================================

train_loader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=BATCH_SIZE
)


# ============================================================
# 7. LOSS
# ============================================================

train_loss = losses.BatchAllTripletLoss(
    model=model
)


# ============================================================
# 8. FINE-TUNE
# ============================================================

print("\nStarting semantic fine-tuning...")

warmup_steps = int(
    len(train_loader) * EPOCHS * 0.1
)

model.fit(
    train_objectives=[
        (train_loader, train_loss)
    ],
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    optimizer_params={
        "lr": LEARNING_RATE
    },
    output_path=OUTPUT_MODEL,
    show_progress_bar=True
)

print("\nFine-tuning complete.")

print(
    "Model saved to:",
    OUTPUT_MODEL
)


# ============================================================
# 9. CREATE INTENT PROTOTYPES
# ============================================================

print("\nCreating intent prototypes...")

train_embeddings = model.encode(
    train_df["text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

prototypes = {}

for intent in intent_names:

    mask = (
        train_df["intent"].values == intent
    )

    vectors = train_embeddings[mask]

    prototype = np.mean(
        vectors,
        axis=0
    )

    prototype = (
        prototype /
        np.linalg.norm(prototype)
    )

    prototypes[intent] = prototype


prototype_matrix = np.vstack(
    [
        prototypes[intent]
        for intent in intent_names
    ]
)

print(
    "Prototype shape:",
    prototype_matrix.shape
)


# ============================================================
# 10. TEST
# ============================================================

print("\nCreating test embeddings...")

test_embeddings = model.encode(
    test_df["text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)


similarities = np.matmul(
    test_embeddings,
    prototype_matrix.T
)

predictions = []

confidences = []

for scores in similarities:

    best_index = np.argmax(scores)

    predictions.append(
        intent_names[best_index]
    )

    confidences.append(
        scores[best_index]
    )


# ============================================================
# 11. METRICS
# ============================================================

accuracy = accuracy_score(
    test_df["intent"],
    predictions
)

macro_f1 = f1_score(
    test_df["intent"],
    predictions,
    average="macro"
)


print("\n================================")
print("FINE-TUNED SEMANTIC RESULTS")
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
        test_df["intent"],
        predictions,
        digits=4
    )
)


# ============================================================
# 12. SAVE TEST RESULTS
# ============================================================

results = test_df.copy()

results["predicted_intent"] = predictions

results["confidence"] = confidences

results.to_csv(
    "fine_tuned_test_predictions.csv",
    index=False
)


# ============================================================
# 13. SAVE PROTOTYPES
# ============================================================

np.save(
    "fine_tuned_intent_prototypes.npy",
    prototype_matrix
)

with open(
    "fine_tuned_intent_labels.txt",
    "w"
) as f:

    for intent in intent_names:
        f.write(intent + "\n")


print("\nFiles generated:")

print("fine_tuned_test_predictions.csv")
print("fine_tuned_intent_prototypes.npy")
print("fine_tuned_intent_labels.txt")
print("hearing_aid_semantic_model/")
