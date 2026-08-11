
import os
import json
import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.metrics import accuracy_score, f1_score, classification_report

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ONNX_FILE = os.path.join(
    SCRIPT_DIR, "tiny_semantic_student_v1.onnx"
)
TEST_FILE = os.path.join(
    SCRIPT_DIR, "unseen_semantic_stress_test.csv"
)
VOCAB_FILE = os.path.join(
    SCRIPT_DIR, "tiny_semantic_student_v1", "vocab.json"
)
LABEL_FILE = os.path.join(
    SCRIPT_DIR, "tiny_semantic_student_v1", "intent_labels.txt"
)
CONFIG_FILE = os.path.join(
    SCRIPT_DIR, "tiny_semantic_student_v1", "config.json"
)

with open(VOCAB_FILE, "r", encoding="utf-8") as f:
    vocab = json.load(f)

with open(LABEL_FILE, "r", encoding="utf-8") as f:
    labels = [x.strip() for x in f if x.strip()]

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)

max_len = config["max_len"]

def tokenize(text):
    ids = []

    for token in str(text).lower().split():
        token = token.strip(".,!?;:\"'()[]{}")
        if token:
            ids.append(vocab.get(token, 1))

    ids = ids[:max_len]

    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))

    return ids

df = pd.read_csv(TEST_FILE)

X = np.asarray(
    [tokenize(x) for x in df["text"]],
    dtype=np.int64
)

print("Loading ONNX model...")
session = ort.InferenceSession(
    ONNX_FILE,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("Input:", input_name)
print("Output:", output_name)
print("Samples:", len(df))

logits = session.run(
    [output_name],
    {input_name: X}
)[0]

pred_ids = np.argmax(logits, axis=1)

predictions = [
    labels[int(i)]
    for i in pred_ids
]

accuracy = accuracy_score(
    df["intent"],
    predictions
)

macro_f1 = f1_score(
    df["intent"],
    predictions,
    average="macro"
)

print("\n================================")
print("ONNX VALIDATION")
print("================================")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Macro F1:", round(macro_f1 * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        df["intent"],
        predictions,
        digits=4
    )
)

result = df.copy()
result["predicted_intent"] = predictions
result["correct"] = (
    result["intent"] == result["predicted_intent"]
)

result.to_csv(
    os.path.join(
        SCRIPT_DIR,
        "onnx_unseen_semantic_predictions.csv"
    ),
    index=False
)

result[
    ~result["correct"]
].to_csv(
    os.path.join(
        SCRIPT_DIR,
        "onnx_unseen_semantic_errors.csv"
    ),
    index=False
)

size_mb = os.path.getsize(ONNX_FILE) / (1024 * 1024)

print("\nONNX model size:", round(size_mb, 3), "MB")
print("Saved:")
print("onnx_unseen_semantic_predictions.csv")
print("onnx_unseen_semantic_errors.csv")
