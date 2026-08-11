
import os
import numpy as np
import pandas as pd
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_dynamic,
    QuantType
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FP32_MODEL = os.path.join(
    SCRIPT_DIR, "tiny_semantic_student_v1.onnx"
)

INT8_MODEL = os.path.join(
    SCRIPT_DIR, "tiny_semantic_student_v1_int8.onnx"
)

TEST_FILE = os.path.join(
    SCRIPT_DIR, "unseen_semantic_stress_test.csv"
)

VOCAB_FILE = os.path.join(
    SCRIPT_DIR,
    "tiny_semantic_student_v1",
    "vocab.json"
)

CONFIG_FILE = os.path.join(
    SCRIPT_DIR,
    "tiny_semantic_student_v1",
    "config.json"
)

LABEL_FILE = os.path.join(
    SCRIPT_DIR,
    "tiny_semantic_student_v1",
    "intent_labels.txt"
)

# ------------------------------------------------------------
# Load tokenizer/config
# ------------------------------------------------------------

import json

with open(VOCAB_FILE, "r", encoding="utf-8") as f:
    vocab = json.load(f)

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)

with open(LABEL_FILE, "r", encoding="utf-8") as f:
    labels = [x.strip() for x in f if x.strip()]

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

# ------------------------------------------------------------
# 1. Dynamic INT8 quantization
# ------------------------------------------------------------

print("Loading FP32 ONNX:")
print(FP32_MODEL)

if not os.path.exists(FP32_MODEL):
    raise FileNotFoundError(
        f"Missing model: {FP32_MODEL}"
    )

print("\nQuantizing FP32 ONNX -> INT8...")

quantize_dynamic(
    model_input=FP32_MODEL,
    model_output=INT8_MODEL,
    weight_type=QuantType.QInt8,
    per_channel=True,
    reduce_range=False
)

fp32_size = os.path.getsize(
    FP32_MODEL
) / (1024 * 1024)

int8_size = os.path.getsize(
    INT8_MODEL
) / (1024 * 1024)

print("\nQuantization successful")
print("FP32 size:", round(fp32_size, 3), "MB")
print("INT8 size:", round(int8_size, 3), "MB")
print(
    "Size reduction:",
    round((1 - int8_size / fp32_size) * 100, 2),
    "%"
)

# ------------------------------------------------------------
# 2. Validate INT8 on exact same 595 samples
# ------------------------------------------------------------

df = pd.read_csv(TEST_FILE)

X = np.asarray(
    [tokenize(x) for x in df["text"]],
    dtype=np.int64
)

print("\nLoading INT8 ONNX...")
session = ort.InferenceSession(
    INT8_MODEL,
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

pred_ids = np.argmax(
    logits,
    axis=1
)

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
print("INT8 ONNX VALIDATION")
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

pred_file = os.path.join(
    SCRIPT_DIR,
    "int8_unseen_semantic_predictions.csv"
)

error_file = os.path.join(
    SCRIPT_DIR,
    "int8_unseen_semantic_errors.csv"
)

result.to_csv(
    pred_file,
    index=False
)

result[
    ~result["correct"]
].to_csv(
    error_file,
    index=False
)

print("\nSaved:")
print(INT8_MODEL)
print("int8_unseen_semantic_predictions.csv")
print("int8_unseen_semantic_errors.csv")
