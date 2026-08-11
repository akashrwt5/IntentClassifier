
import os
import random
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from scipy.sparse import hstack, csr_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(SCRIPT_DIR, "canonical_semantic_training_20724.xlsx")
if not os.path.exists(DATASET) and os.path.exists("canonical_semantic_training_20724.xlsx"):
    DATASET = "canonical_semantic_training_20724.xlsx"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

df = pd.read_excel(DATASET, sheet_name="dataset")
df = df[["text", "intent"]].dropna()
df["text"] = df["text"].astype(str).str.strip()
df["intent"] = df["intent"].astype(str).str.strip()

train, temp = train_test_split(
    df, test_size=0.20, random_state=SEED, stratify=df["intent"]
)
val, test = train_test_split(
    temp, test_size=0.50, random_state=SEED, stratify=temp["intent"]
)

print("Train:", len(train), "Validation:", len(val), "Test:", len(test))

# ---------------- TF-IDF branch ----------------
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=60000,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(train["text"])
X_val_tfidf = tfidf.transform(val["text"])
X_test_tfidf = tfidf.transform(test["text"])

# ---------------- Semantic branch ----------------
print("\nLoading MiniLM...")
encoder = SentenceTransformer(MODEL_NAME)

def encode(texts):
    return encoder.encode(
        texts.tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

X_train_sem = encode(train["text"])
X_val_sem = encode(val["text"])
X_test_sem = encode(test["text"])

# Convert dense semantic vectors to sparse so they can be fused with TF-IDF.
X_train_sem = csr_matrix(X_train_sem)
X_val_sem = csr_matrix(X_val_sem)
X_test_sem = csr_matrix(X_test_sem)

# Semantic branch gets a moderate weight.
SEMANTIC_WEIGHT = 2.0

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

# Logistic regression remains lightweight and ONNX-friendly.
clf = LogisticRegression(
    max_iter=2000,
    C=4.0,
    solver="lbfgs",
    random_state=SEED
)

print("\nTraining hybrid classifier...")
clf.fit(X_train, train["intent"])

for name, X, y in [
    ("Validation", X_val, val["intent"]),
    ("Test", X_test, test["intent"])
]:
    pred = clf.predict(X)

    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro")

    print("\n==============================")
    print(name)
    print("==============================")
    print("Accuracy:", round(acc * 100, 2), "%")
    print("Macro F1:", round(f1 * 100, 2), "%")

    if name == "Test":
        print("\nClassification Report:\n")
        print(classification_report(y, pred, digits=4))

        cm = pd.DataFrame(
            confusion_matrix(y, pred, labels=clf.classes_),
            index=clf.classes_,
            columns=clf.classes_
        )
        cm.to_csv("hybrid_confusion_matrix.csv")

        out = test.copy()
        out["predicted_intent"] = pred
        out.to_csv("hybrid_test_predictions.csv", index=False)

# Save the sklearn components for later ONNX conversion.
import joblib
joblib.dump(tfidf, "hybrid_tfidf.joblib")
joblib.dump(clf, "hybrid_classifier.joblib")

with open("hybrid_intent_labels.txt", "w") as f:
    for label in clf.classes_:
        f.write(label + "\n")

print("\nSaved:")
print("hybrid_tfidf.joblib")
print("hybrid_classifier.joblib")
print("hybrid_intent_labels.txt")
print("hybrid_test_predictions.csv")
print("hybrid_confusion_matrix.csv")
