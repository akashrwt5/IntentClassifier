import os
import random
import joblib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_FILE = "semantic_training_v2_hard_negatives.xlsx"
DATASET = os.path.join(SCRIPT_DIR, DATASET_FILE) if os.path.exists(os.path.join(SCRIPT_DIR, DATASET_FILE)) else DATASET_FILE

STRESS_FILENAME = "unseen_semantic_stress_test.csv"
STRESS_FILE = os.path.join(SCRIPT_DIR, STRESS_FILENAME) if os.path.exists(os.path.join(SCRIPT_DIR, STRESS_FILENAME)) else STRESS_FILENAME

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42
SEMANTIC_WEIGHT = 2.0

random.seed(SEED)
np.random.seed(SEED)

# ---------------- Load V2 dataset ----------------
df = pd.read_excel(DATASET, sheet_name="dataset")
df = df[["text", "intent"]].dropna()
df["text"] = df["text"].astype(str).str.strip()
df["intent"] = df["intent"].astype(str).str.strip()
df = df[df["text"] != ""].drop_duplicates(["text", "intent"]).reset_index(drop=True)

print("Total V2 samples:", len(df))
print("Total intents:", df["intent"].nunique())

# ---------------- Same stratified split ----------------
train, temp = train_test_split(
    df, test_size=0.20, random_state=SEED, stratify=df["intent"]
)
val, test = train_test_split(
    temp, test_size=0.50, random_state=SEED, stratify=temp["intent"]
)

print("Train:", len(train))
print("Validation:", len(val))
print("Test:", len(test))

# ---------------- TF-IDF ----------------
print("\nCreating TF-IDF embeddings...")
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=60000,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(train["text"])
X_val_tfidf = tfidf.transform(val["text"])
X_test_tfidf = tfidf.transform(test["text"])

# ---------------- Semantic embeddings ----------------
print("\nLoading MiniLM...")
encoder = SentenceTransformer(MODEL_NAME)

def encode(texts):
    return encoder.encode(
        texts.tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

print("\nEncoding train...")
X_train_sem = csr_matrix(encode(train["text"]))

print("\nEncoding validation...")
X_val_sem = csr_matrix(encode(val["text"]))

print("\nEncoding test...")
X_test_sem = csr_matrix(encode(test["text"]))

# ---------------- Hybrid features ----------------
X_train = hstack([
    X_train_tfidf,
    X_train_sem * SEMANTIC_WEIGHT
]).tocsr()

X_val = hstack([
    X_val_tfidf,
    X_val_sem * SEMANTIC_WEIGHT
]).tocsr()

X_test = hstack([
    X_test_tfidf,
    X_test_sem * SEMANTIC_WEIGHT
]).tocsr()

# ---------------- Classifier ----------------
print("\nTraining V2 hybrid classifier...")

clf = LogisticRegression(
    max_iter=2000,
    C=4.0,
    solver="lbfgs",
    random_state=SEED
)

clf.fit(X_train, train["intent"])

def evaluate(name, X, y):
    pred = clf.predict(X)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro")

    print("\n==============================")
    print(name)
    print("==============================")
    print("Accuracy:", round(acc * 100, 2), "%")
    print("Macro F1:", round(f1 * 100, 2), "%")

    return pred

val_pred = evaluate("V2 Validation", X_val, val["intent"])
test_pred = evaluate("V2 Internal Test", X_test, test["intent"])

print("\nV2 Internal Classification Report:")
print(classification_report(test["intent"], test_pred, digits=4))

pd.DataFrame(
    confusion_matrix(test["intent"], test_pred, labels=clf.classes_),
    index=clf.classes_,
    columns=clf.classes_
).to_csv(os.path.join(SCRIPT_DIR, "v2_hybrid_internal_confusion_matrix.csv"))

# ---------------- Clean unseen semantic stress test ----------------
print("\n================================")
print("EVALUATING CLEAN UNSEEN TEST")
print("================================")

stress = pd.read_csv(STRESS_FILE)
X_stress_tfidf = tfidf.transform(stress["text"])

X_stress_sem = csr_matrix(
    encoder.encode(
        stress["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )
)

X_stress = hstack([
    X_stress_tfidf,
    X_stress_sem * SEMANTIC_WEIGHT
]).tocsr()

stress_pred = clf.predict(X_stress)

stress_acc = accuracy_score(stress["intent"], stress_pred)
stress_f1 = f1_score(stress["intent"], stress_pred, average="macro")

print("\nV2 UNSEEN SEMANTIC STRESS TEST")
print("Accuracy:", round(stress_acc * 100, 2), "%")
print("Macro F1:", round(stress_f1 * 100, 2), "%")

print("\nStress Classification Report:")
print(classification_report(stress["intent"], stress_pred, digits=4))

stress_out = stress.copy()
stress_out["predicted_intent"] = stress_pred
stress_out["correct"] = stress_out["intent"] == stress_out["predicted_intent"]
stress_out.to_csv(os.path.join(SCRIPT_DIR, "v2_unseen_semantic_stress_predictions.csv"), index=False)

pd.DataFrame(
    confusion_matrix(
        stress["intent"],
        stress_pred,
        labels=clf.classes_
    ),
    index=clf.classes_,
    columns=clf.classes_
).to_csv(os.path.join(SCRIPT_DIR, "v2_unseen_semantic_stress_confusion_matrix.csv"))

stress_out[~stress_out["correct"]].to_csv(
    os.path.join(SCRIPT_DIR, "v2_unseen_semantic_stress_errors.csv"),
    index=False
)

# ---------------- Save model components ----------------
joblib.dump(tfidf, os.path.join(SCRIPT_DIR, "v2_hybrid_tfidf.joblib"))
joblib.dump(clf, os.path.join(SCRIPT_DIR, "v2_hybrid_classifier.joblib"))

with open(os.path.join(SCRIPT_DIR, "v2_hybrid_intent_labels.txt"), "w") as f:
    for label in clf.classes_:
        f.write(label + "\n")

print("\nSaved V2 files:")
print("v2_hybrid_tfidf.joblib")
print("v2_hybrid_classifier.joblib")
print("v2_hybrid_intent_labels.txt")
print("v2_hybrid_internal_confusion_matrix.csv")
print("v2_unseen_semantic_stress_predictions.csv")
print("v2_unseen_semantic_stress_confusion_matrix.csv")
print("v2_unseen_semantic_stress_errors.csv")
