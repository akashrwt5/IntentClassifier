import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import normalize


# ============================================================
# 1. LOAD DATASET
# ============================================================

FILE = "semantic_balanced_dataset.xlsx"

df = pd.read_excel(FILE, sheet_name="dataset")

df = df[["text", "intent"]].dropna()

df["text"] = df["text"].astype(str).str.strip()
df["intent"] = df["intent"].astype(str).str.strip()

print("Total samples:", len(df))
print("Total intents:", df["intent"].nunique())

print("\nIntent distribution:")
print(df["intent"].value_counts())


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
# 3. LOAD SEMANTIC MODEL
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("\nLoading semantic model...")
model = SentenceTransformer(MODEL_NAME)

print("Model loaded.")


# ============================================================
# 4. CREATE EMBEDDINGS
# ============================================================

print("\nCreating train embeddings...")

train_embeddings = model.encode(
    train_df["text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("\nCreating validation embeddings...")

val_embeddings = model.encode(
    val_df["text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("\nCreating test embeddings...")

test_embeddings = model.encode(
    test_df["text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)


# ============================================================
# 5. CREATE INTENT PROTOTYPES
# ============================================================

print("\nCreating intent prototypes...")

intent_names = sorted(train_df["intent"].unique())

prototypes = {}

for intent in intent_names:

    mask = train_df["intent"].values == intent

    intent_vectors = train_embeddings[mask]

    # Mean embedding
    prototype = np.mean(intent_vectors, axis=0)

    # Normalize prototype
    prototype = prototype / np.linalg.norm(prototype)

    prototypes[intent] = prototype


prototype_matrix = np.vstack(
    [prototypes[intent] for intent in intent_names]
)

print("Prototype shape:", prototype_matrix.shape)


# ============================================================
# 6. PREDICTION FUNCTION
# ============================================================

def predict_embeddings(embeddings):

    # Since both embeddings and prototypes are normalized,
    # matrix multiplication = cosine similarity.

    similarity = np.matmul(
        embeddings,
        prototype_matrix.T
    )

    predictions = []

    confidence = []

    for scores in similarity:

        best_index = np.argmax(scores)

        predictions.append(
            intent_names[best_index]
        )

        confidence.append(
            scores[best_index]
        )

    return predictions, confidence


# ============================================================
# 7. VALIDATION
# ============================================================

val_predictions, val_confidence = predict_embeddings(
    val_embeddings
)

val_accuracy = accuracy_score(
    val_df["intent"],
    val_predictions
)

val_f1 = f1_score(
    val_df["intent"],
    val_predictions,
    average="macro"
)

print("\n==============================")
print("VALIDATION RESULTS")
print("==============================")

print("Accuracy:", round(val_accuracy * 100, 2), "%")
print("Macro F1:", round(val_f1 * 100, 2), "%")


# ============================================================
# 8. FINAL TEST
# ============================================================

test_predictions, test_confidence = predict_embeddings(
    test_embeddings
)

test_accuracy = accuracy_score(
    test_df["intent"],
    test_predictions
)

test_f1 = f1_score(
    test_df["intent"],
    test_predictions,
    average="macro"
)

print("\n==============================")
print("TEST RESULTS")
print("==============================")

print("Accuracy:", round(test_accuracy * 100, 2), "%")
print("Macro F1:", round(test_f1 * 100, 2), "%")


# ============================================================
# 9. DETAILED REPORT
# ============================================================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        test_df["intent"],
        test_predictions,
        digits=4
    )
)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_df["intent"],
    test_predictions,
    labels=intent_names
)

cm_df = pd.DataFrame(
    cm,
    index=intent_names,
    columns=intent_names
)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

print(cm_df)


# ============================================================
# 11. SAVE RESULTS
# ============================================================

results = test_df.copy()

results["predicted_intent"] = test_predictions
results["confidence"] = test_confidence

results.to_csv(
    "semantic_test_predictions.csv",
    index=False
)

cm_df.to_csv(
    "semantic_confusion_matrix.csv"
)

print("\nResults saved:")
print("semantic_test_predictions.csv")
print("semantic_confusion_matrix.csv")


# ============================================================
# 12. SAVE PROTOTYPES
# ============================================================

np.save(
    "intent_prototypes.npy",
    prototype_matrix
)

with open(
    "intent_labels.txt",
    "w"
) as f:

    for intent in intent_names:
        f.write(intent + "\n")

print("intent_prototypes.npy saved")
print("intent_labels.txt saved")
