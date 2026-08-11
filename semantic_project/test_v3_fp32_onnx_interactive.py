
import json
import re
import numpy as np
import onnxruntime as ort

ROOT = "/Users/shuklam/IntentClassifier/semantic_project"

MODEL = ROOT + "/tiny_semantic_student_v3_fp32/v3_semantic_student_fp32.onnx"
VOCAB_FILE = ROOT + "/tiny_semantic_student_v2_balanced/vocab.json"

LABELS = [
    "device.memory.change",
    "device.volume.decrease",
    "device.volume.increase",
    "device.volume.mute",
    "device.volume.unmute",
    "find.phone.locate",
    "help.reminder.show",
    "reminders.task.complete",
    "reminders.task.create",
    "streaming.session.start",
    "streaming.session.stop",
]

with open(VOCAB_FILE, "r", encoding="utf-8") as f:
    raw = json.load(f)

if isinstance(raw.get("model"), dict) and "vocab" in raw["model"]:
    vocab = raw["model"]["vocab"]
elif "vocab" in raw:
    vocab = raw["vocab"]
else:
    vocab = raw

vocab = {str(k): int(v) for k, v in vocab.items()}

PAD = int(vocab.get("<pad>", vocab.get("[PAD]", 0)))
UNK = int(vocab.get("<unk>", vocab.get("[UNK]", 1)))
CLS = vocab.get("<cls>", vocab.get("[CLS]", None))
SEP = vocab.get("<sep>", vocab.get("[SEP]", None))

MAX_LEN = 24


def clean(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def tokenize(text):
    text = clean(text)

    # Same punctuation handling used during benchmark
    text = re.sub(r"([.!?,;:()'])", r" \1 ", text)

    ids = []

    if CLS is not None:
        ids.append(int(CLS))

    for word in text.split():

        if word in vocab:
            ids.append(int(vocab[word]))
            continue

        pos = 0
        matched = False

        while pos < len(word):

            best_id = None
            best_len = 0

            for end in range(len(word), pos, -1):

                piece = word[pos:end]

                for candidate in (piece, "##" + piece):

                    if candidate in vocab:
                        best_id = int(vocab[candidate])
                        best_len = len(piece)
                        break

                if best_id is not None:
                    break

            if best_id is None:
                ids.append(UNK)
                matched = True
                break

            ids.append(best_id)
            pos += best_len
            matched = True

        if not matched:
            ids.append(UNK)

    if SEP is not None:
        ids.append(int(SEP))

    ids = ids[:MAX_LEN]

    ids += [PAD] * (MAX_LEN - len(ids))

    return ids


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


print("=" * 70)
print("V3 FP32 ONNX — INTERACTIVE TEST")
print("=" * 70)

session = ort.InferenceSession(
    MODEL,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("Model :", MODEL)
print("Input :", input_name)
print("Output:", output_name)
print()
print("Type a sentence.")
print("Type 'exit' to quit.")
print("=" * 70)


while True:

    try:
        text = input("\nYou: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")
        break

    if not text:
        continue

    if text.lower() in ["exit", "quit", "q"]:
        print("Bye.")
        break

    ids = np.asarray(
        [tokenize(text)],
        dtype=np.int64
    )

    logits = session.run(
        [output_name],
        {input_name: ids}
    )[0][0]

    probs = softmax(logits)

    order = np.argsort(probs)[::-1]

    predicted = int(order[0])
    confidence = float(probs[predicted])

    print()
    print("Tokens:")
    print(ids[0].tolist())

    print()
    print("Prediction:")
    print(f"Intent     : {LABELS[predicted]}")
    print(f"Confidence : {confidence * 100:.2f}%")

    print()
    print("Top 3:")

    for rank, idx in enumerate(order[:3], 1):
        print(
            f"{rank}. "
            f"{LABELS[int(idx)]:34s} "
            f"{float(probs[idx]) * 100:.2f}%"
        )
PY
