
import numpy as np
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

STRESS_FILE = "unseen_semantic_stress_test.csv"
TFIDF_FILE = "hybrid_tfidf.joblib"
CLASSIFIER_FILE = "hybrid_classifier.joblib"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_WEIGHT = 2.0

df = pd.read_csv(STRESS_FILE)
tfidf = joblib.load(TFIDF_FILE)
clf = joblib.load(CLASSIFIER_FILE)

print("Stress samples:", len(df))
print("Loading semantic encoder...")
encoder = SentenceTransformer(MODEL_NAME)

X_tfidf = tfidf.transform(df["text"])

X_sem = encoder.encode(
    df["text"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)
X_sem = csr_matrix(X_sem)

X = hstack([
    X_tfidf,
    X_sem * SEMANTIC_WEIGHT
]).tocsr()

pred = clf.predict(X)
y = df["intent"]

acc = accuracy_score(y, pred)
f1 = f1_score(y, pred, average="macro")

print("\n================================")
print("UNSEEN SEMANTIC STRESS TEST")
print("================================")
print("Accuracy:", round(acc * 100, 2), "%")
print("Macro F1:", round(f1 * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y, pred, digits=4))

out = df.copy()
out["predicted_intent"] = pred
out["correct"] = out["intent"] == out["predicted_intent"]
out.to_csv("unseen_semantic_stress_predictions.csv", index=False)

cm = pd.DataFrame(
    confusion_matrix(y, pred, labels=clf.classes_),
    index=clf.classes_,
    columns=clf.classes_
)
cm.to_csv("unseen_semantic_stress_confusion_matrix.csv")

errors = out[~out["correct"]]
errors.to_csv("unseen_semantic_stress_errors.csv", index=False)

print("\nErrors:", len(errors))
print("Saved:")
print("unseen_semantic_stress_predictions.csv")
print("unseen_semantic_stress_confusion_matrix.csv")
print("unseen_semantic_stress_errors.csv")
