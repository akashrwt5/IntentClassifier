
import os
import json
import numpy as np
import onnxruntime as ort

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(
    SCRIPT_DIR,
    "tiny_semantic_student_v1_int8.onnx"
)

MODEL_DIR = os.path.join(
    SCRIPT_DIR,
    "tiny_semantic_student_v1"
)

VOCAB_FILE = os.path.join(MODEL_DIR, "vocab.json")
CONFIG_FILE = os.path.join(MODEL_DIR, "config.json")
LABEL_FILE = os.path.join(MODEL_DIR, "intent_labels.txt")


# ------------------------------------------------------------
# Load model metadata
# ------------------------------------------------------------

with open(VOCAB_FILE, "r", encoding="utf-8") as f:
    vocab = json.load(f)

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)

with open(LABEL_FILE, "r", encoding="utf-8") as f:
    labels = [x.strip() for x in f if x.strip()]

MAX_LEN = config["max_len"]


# ------------------------------------------------------------
# Tokenizer
# Must exactly match training tokenizer
# ------------------------------------------------------------

def tokenize(text):
    ids = []

    for token in str(text).lower().split():
        token = token.strip(".,!?;:\"'()[]{}")

        if token:
            ids.append(vocab.get(token, 1))  # 1 = <UNK>

    ids = ids[:MAX_LEN]

    if len(ids) < MAX_LEN:
        ids += [0] * (MAX_LEN - len(ids))

    return ids


# ------------------------------------------------------------
# Load ONNX model
# ------------------------------------------------------------

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_FILE}\n"
    )

print("Loading INT8 ONNX model...")

session = ort.InferenceSession(
    MODEL_FILE,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("Model loaded.")
print("Input :", input_name)
print("Output:", output_name)
print("Intents:", len(labels))

print("\n==============================================")
print(" OFFLINE SEMANTIC INTENT TEST")
print("==============================================")
print("Type a sentence and press Enter.")
print("Type 'exit' to quit.")
print("==============================================\n")


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

def predict(text):

    input_ids = np.asarray(
        [tokenize(text)],
        dtype=np.int64
    )

    logits = session.run(
        [output_name],
        {input_name: input_ids}
    )[0][0]

    # Stable softmax
    exp_logits = np.exp(
        logits - np.max(logits)
    )

    probabilities = (
        exp_logits /
        np.sum(exp_logits)
    )

    best_id = int(
        np.argmax(probabilities)
    )

    intent = labels[best_id]
    confidence = float(
        probabilities[best_id]
    )

    # Top 3 predictions
    top_ids = np.argsort(
        probabilities
    )[::-1][:3]

    return intent, confidence, [
        (labels[int(i)], float(probabilities[int(i)]))
        for i in top_ids
    ]


# ------------------------------------------------------------
# Interactive loop
# ------------------------------------------------------------

while True:

    try:
        text = input("You: ").strip()

    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        break

    if not text:
        continue

    if text.lower() in {
        "exit",
        "quit",
        "q"
    }:
        print("Exiting.")
        break

    intent, confidence, top3 = predict(text)

    print("\nPrediction:")
    print("Intent     :", intent)
    print(
        "Confidence :",
        f"{confidence * 100:.2f}%"
    )

    print("\nTop 3:")
    for i, (label, prob) in enumerate(top3, 1):
        print(
            f"{i}. {label:<30} "
            f"{prob * 100:.2f}%"
        )

    print()
