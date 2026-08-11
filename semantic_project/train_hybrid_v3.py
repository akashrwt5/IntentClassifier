import os
import random
import numpy as np
import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_FILE = "semantic_training_v3_hard_negatives.xlsx"
DATASET = os.path.join(SCRIPT_DIR, DATASET_FILE) if os.path.exists(os.path.join(SCRIPT_DIR, DATASET_FILE)) else DATASET_FILE

STRESS_FILENAME = "unseen_semantic_stress_test.csv"
STRESS_FILE = os.path.join(SCRIPT_DIR, STRESS_FILENAME) if os.path.exists(os.path.join(SCRIPT_DIR, STRESS_FILENAME)) else STRESS_FILENAME

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42
SEMANTIC_WEIGHT = 2.0

random.seed(SEED)
np.random.seed(SEED)

df = pd.read_excel(DATASET, sheet_name="dataset").dropna()
df = df[["text", "intent"]]
df["text"] = df["text"].astype(str).str.strip()
df["intent"] = df["intent"].astype(str).str.strip()
df = df.drop_duplicates(["text", "intent"]).reset_index(drop=True)

print("Total V3 samples:", len(df))

train, temp = train_test_split(
    df, test_size=0.20, random_state=SEED, stratify=df["intent"]
)
val, test = train_test_split(
    temp, test_size=0.50, random_state=SEED, stratify=temp["intent"]
)

print("Train:", len(train), "Validation:", len(val), "Test:", len(test))

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=60000,
    sublinear_tf=True
)

Xtr_t = tfidf.fit_transform(train["text"])
Xv_t = tfidf.transform(val["text"])
Xte_t = tfidf.transform(test["text"])

encoder = SentenceTransformer(MODEL_NAME)

def enc(s):
    return encoder.encode(
        s.tolist(), batch_size=64, show_progress_bar=True,
        normalize_embeddings=True
    )

Xtr_s = csr_matrix(enc(train["text"]))
Xv_s = csr_matrix(enc(val["text"]))
Xte_s = csr_matrix(enc(test["text"]))

Xtr = hstack([Xtr_t, Xtr_s * SEMANTIC_WEIGHT]).tocsr()
Xv = hstack([Xv_t, Xv_s * SEMANTIC_WEIGHT]).tocsr()
Xte = hstack([Xte_t, Xte_s * SEMANTIC_WEIGHT]).tocsr()

clf = LogisticRegression(
    max_iter=2000, C=4.0, solver="lbfgs", random_state=SEED
)
clf.fit(Xtr, train["intent"])

internal_pred = clf.predict(Xte)
print("\nV3 INTERNAL TEST")
print("Accuracy:", round(accuracy_score(test["intent"], internal_pred) * 100, 2), "%")
print("Macro F1:", round(f1_score(test["intent"], internal_pred, average="macro") * 100, 2), "%")

pd.DataFrame(
    confusion_matrix(test["intent"], internal_pred, labels=clf.classes_),
    index=clf.classes_, columns=clf.classes_
).to_csv(os.path.join(SCRIPT_DIR, "v3_hybrid_internal_confusion_matrix.csv"))

# Clean 595-sample semantic benchmark: NEVER train on it.
stress = pd.read_csv(STRESS_FILE)
Xs_t = tfidf.transform(stress["text"])
Xs_s = csr_matrix(enc(stress["text"]))
Xs = hstack([Xs_t, Xs_s * SEMANTIC_WEIGHT]).tocsr()

pred = clf.predict(Xs)
acc = accuracy_score(stress["intent"], pred)
f1 = f1_score(stress["intent"], pred, average="macro")

print("\nV3 UNSEEN SEMANTIC STRESS TEST")
print("Accuracy:", round(acc * 100, 2), "%")
print("Macro F1:", round(f1 * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(stress["intent"], pred, digits=4))

out = stress.copy()
out["predicted_intent"] = pred
out["correct"] = out["intent"] == out["predicted_intent"]
out.to_csv(os.path.join(SCRIPT_DIR, "v3_unseen_semantic_stress_predictions.csv"), index=False)
out[~out["correct"]].to_csv(os.path.join(SCRIPT_DIR, "v3_unseen_semantic_stress_errors.csv"), index=False)

pd.DataFrame(
    confusion_matrix(stress["intent"], pred, labels=clf.classes_),
    index=clf.classes_, columns=clf.classes_
).to_csv(os.path.join(SCRIPT_DIR, "v3_unseen_semantic_stress_confusion_matrix.csv"))

joblib.dump(tfidf, os.path.join(SCRIPT_DIR, "v3_hybrid_tfidf.joblib"))
joblib.dump(clf, os.path.join(SCRIPT_DIR, "v3_hybrid_classifier.joblib"))
with open(os.path.join(SCRIPT_DIR, "v3_hybrid_intent_labels.txt"), "w") as f:
    for label in clf.classes_:
        f.write(label + "\n")

print("\nV3 model files saved.")
