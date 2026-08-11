#!/usr/bin/env python3
"""
Benchmark V2 INT8 against the locked Current INT8 baseline.

Tests:
1. 595-sample unseen semantic stress test
2. Contextual test
3. Targeted regression test
4. OOD rejection @ 0.70

No model is modified.

Expected baseline:
    Current INT8:
        size       0.236 MB
        unseen     94.29%
        contextual 90.62%
        OOD        34.38%

V2 FP32 previously passed:
    unseen     95.46%
    contextual 92.31%
    OOD        41.67%
    targeted   100%

Now we verify whether those gains survive INT8 quantization.
"""

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.metrics import accuracy_score, f1_score, classification_report


ROOT = Path(__file__).resolve().parent

# ------------------------------------------------------------
# Model paths
# ------------------------------------------------------------

CURRENT_INT8_CANDIDATES = [
    ROOT / "tiny_semantic_student_v1_int8.onnx",
    ROOT / "tiny_semantic_student_v1" / "student_int8.onnx",
    ROOT / "tiny_semantic_student_v1" / "intent_model_int8.onnx",
]

V2_INT8 = (
    ROOT
    / "tiny_semantic_student_v2_int8"
    / "v2_semantic_student_int8.onnx"
)

# ------------------------------------------------------------
# Supporting files
# ------------------------------------------------------------

BASE = ROOT / "tiny_semantic_student_v1"

VOCAB_PATH = BASE / "vocab.json"
LABELS_PATH = BASE / "intent_labels.txt"

STRESS_CANDIDATES = [
    ROOT / "unseen_semantic_stress_test.csv",
    ROOT / "v3_unseen_semantic_stress_predictions.csv",
    ROOT / "v2_unseen_semantic_stress_predictions.csv",
    ROOT / "onnx_unseen_semantic_predictions.csv",
]

OUT = ROOT / "v2_int8_benchmark"


# ------------------------------------------------------------
# Locked baseline reference
# ------------------------------------------------------------

BASELINE = {
    "size_mb": 0.236,
    "unseen": 94.29,
    "contextual": 90.62,
    "ood": 34.38,
}

OOD_THRESHOLD = 0.70


# ------------------------------------------------------------
# Find current INT8
# ------------------------------------------------------------

current_path = next(
    (p for p in CURRENT_INT8_CANDIDATES if p.exists()),
    None,
)

if current_path is None:
    raise FileNotFoundError(
        "Could not find Current INT8 baseline.\n"
        "Expected one of:\n"
        + "\n".join(str(x) for x in CURRENT_INT8_CANDIDATES)
    )

if not V2_INT8.exists():
    raise FileNotFoundError(
        f"V2 INT8 not found:\n{V2_INT8}\n"
        "Run export_v2_to_onnx_int8_explicit_attention.py first."
    )

if not VOCAB_PATH.exists():
    raise FileNotFoundError(f"Missing vocabulary: {VOCAB_PATH}")

if not LABELS_PATH.exists():
    raise FileNotFoundError(f"Missing labels: {LABELS_PATH}")


with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = [x.strip() for x in f if x.strip()]

label_set = set(labels)

PAD = int(
    vocab.get(
        "<PAD>",
        vocab.get("[PAD]", 0),
    )
)

UNK = int(
    vocab.get(
        "<UNK>",
        vocab.get("[UNK]", 1),
    )
)


# ------------------------------------------------------------
# Read config for max length
# ------------------------------------------------------------

config_path = BASE / "config.json"

if config_path.exists():
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
else:
    config = {}

def cfg(*keys, default=None):
    for key in keys:
        if key in config:
            return config[key]
    return default

MAX_LEN = int(
    cfg(
        "max_len",
        "max_length",
        "sequence_length",
        default=24,
    )
)


# ------------------------------------------------------------
# Tokenizer: SAME baseline tokenizer
# ------------------------------------------------------------

def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(
        r"[^a-z0-9']+",
        " ",
        text,
    )
    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def encode(text):
    tokens = clean_text(text).split()

    ids = [
        int(vocab.get(tok, UNK))
        for tok in tokens
    ][:MAX_LEN]

    ids += [
        PAD
    ] * (
        MAX_LEN - len(ids)
    )

    return ids


# ------------------------------------------------------------
# ONNX model wrapper
# ------------------------------------------------------------

class ONNXIntentModel:

    def __init__(self, path):
        self.path = Path(path)

        self.session = ort.InferenceSession(
            str(self.path),
            providers=["CPUExecutionProvider"],
        )

        self.input_name = (
            self.session
            .get_inputs()[0]
            .name
        )

        self.output_name = (
            self.session
            .get_outputs()[0]
            .name
        )

        self.size_mb = (
            self.path.stat().st_size
            / 1024
            / 1024
        )

    def predict(self, texts):

        x = np.asarray(
            [
                encode(t)
                for t in texts
            ],
            dtype=np.int64,
        )

        logits = self.session.run(
            [self.output_name],
            {
                self.input_name: x
            },
        )[0]

        logits = logits.astype(
            np.float64
        )

        logits -= logits.max(
            axis=1,
            keepdims=True,
        )

        probs = np.exp(logits)

        probs /= probs.sum(
            axis=1,
            keepdims=True,
        )

        indices = np.argmax(
            probs,
            axis=1,
        )

        confidence = np.max(
            probs,
            axis=1,
        )

        predictions = [
            labels[int(i)]
            for i in indices
        ]

        return predictions, confidence.tolist()


print("=" * 80)
print("V2 INT8 vs CURRENT INT8 — FULL BENCHMARK")
print("=" * 80)

print("\nLoading models...")

current = ONNXIntentModel(
    current_path
)

v2 = ONNXIntentModel(
    V2_INT8
)

print(
    f"Current INT8 loaded | "
    f"size: {current.size_mb:.3f} MB"
)

print(
    f"V2 INT8 loaded      | "
    f"size: {v2.size_mb:.3f} MB"
)


# ------------------------------------------------------------
# Test data
# ------------------------------------------------------------

def load_stress_file():

    path = next(
        (
            p
            for p in STRESS_CANDIDATES
            if p.exists()
        ),
        None,
    )

    if path is None:
        raise FileNotFoundError(
            "Could not find unseen semantic stress CSV.\n"
            "Expected one of:\n"
            + "\n".join(
                str(x)
                for x in STRESS_CANDIDATES
            )
        )

    df = pd.read_csv(path)

    if "text" not in df.columns:
        raise ValueError(
            f"{path} does not contain 'text'. "
            f"Columns: {list(df.columns)}"
        )

    intent_col = None

    for col in [
        "intent",
        "expected_intent",
        "true_intent",
        "label",
    ]:
        if col in df.columns:
            intent_col = col
            break

    if intent_col is None:
        raise ValueError(
            f"Could not find true intent column in {path}. "
            f"Columns: {list(df.columns)}"
        )

    df = df[
        ["text", intent_col]
    ].rename(
        columns={
            intent_col: "intent"
        }
    )

    df = df.dropna(
        subset=[
            "text",
            "intent",
        ]
    ).reset_index(drop=True)

    return df, path


stress, stress_path = load_stress_file()

print(
    "\nUnseen stress file:",
    stress_path
)

print(
    "Unseen samples:",
    len(stress)
)


# ------------------------------------------------------------
# Generic benchmark helper
# ------------------------------------------------------------

def evaluate_unseen(model, name):

    pred, conf = model.predict(
        stress["text"].tolist()
    )

    acc = accuracy_score(
        stress["intent"],
        pred,
    )

    macro = f1_score(
        stress["intent"],
        pred,
        average="macro",
    )

    details = stress.copy()

    details["predicted"] = pred
    details["confidence"] = conf

    details["correct"] = (
        details["intent"]
        == details["predicted"]
    )

    details.to_csv(
        OUT
        / f"{name}_unseen.csv",
        index=False,
    )

    details[
        ~details["correct"]
    ].to_csv(
        OUT
        / f"{name}_unseen_errors.csv",
        index=False,
    )

    return acc * 100, macro * 100


# ------------------------------------------------------------
# Contextual test
# SAME test set used for the previous V2 benchmark.
# ------------------------------------------------------------

CONTEXTUAL = [
    (
        "it's quieter can you make it a little louder",
        "device.volume.increase",
    ),
    (
        "it's a little loud can you make it quieter",
        "device.volume.decrease",
    ),
    (
        "i can still hear it make it completely silent",
        "device.volume.mute",
    ),
    (
        "turn the sound back on",
        "device.volume.unmute",
    ),
    (
        "the audio is quiet but don't mute it make it louder",
        "device.volume.increase",
    ),
    (
        "the audio is loud but keep it on just lower it",
        "device.volume.decrease",
    ),
    (
        "i need to go to airport tomorrow",
        "reminders.task.create",
    ),
    (
        "i need to go to airport tommorow",
        "reminders.task.create",
    ),
    (
        "where can i find my phone",
        "find.phone.locate",
    ),
    (
        "please show the reminders i have",
        "help.reminder.show",
    ),
    (
        "mark that reminder as completed",
        "reminders.task.complete",
    ),
    (
        "please start the streaming session",
        "streaming.session.start",
    ),
    (
        "please stop the streaming session",
        "streaming.session.stop",
    ),
]


# ------------------------------------------------------------
# Targeted regression
# ------------------------------------------------------------

TARGETED = [
    (
        "it's quieter can you make it a little louder",
        "device.volume.increase",
    ),
    (
        "i can still hear it make it completely silent",
        "device.volume.mute",
    ),
    (
        "turn off",
        "device.volume.mute",
    ),
    (
        "i need to go to airport tomorrow",
        "reminders.task.create",
    ),
    (
        "i need to go to airport tommorow",
        "reminders.task.create",
    ),
    (
        "it's bit loudy here can you make it quietr",
        "device.volume.decrease",
    ),
    (
        "the sound is too loud turn it down",
        "device.volume.decrease",
    ),
    (
        "the sound is too quiet make it louder",
        "device.volume.increase",
    ),
    (
        "turn the sound back on",
        "device.volume.unmute",
    ),
    (
        "where is my phone",
        "find.phone.locate",
    ),
    (
        "show my reminders",
        "help.reminder.show",
    ),
    (
        "complete my reminder",
        "reminders.task.complete",
    ),
    (
        "start streaming",
        "streaming.session.start",
    ),
    (
        "stop streaming",
        "streaming.session.stop",
    ),
]


# ------------------------------------------------------------
# OOD
# SAME 12-example sanity set used for V2 FP32.
# ------------------------------------------------------------

OOD = [
    "what is the weather today",
    "i want to order pizza",
    "how do i cook pasta",
    "what time is the train",
    "tell me a joke",
    "i am planning a holiday",
    "what is the capital of france",
    "my car needs an oil change",
    "please open my browser",
    "i need a hotel tonight",
    "what is the stock market doing",
    "how do i charge my laptop",
]


def evaluate_contextual(model, name):

    texts = [
        x[0]
        for x in CONTEXTUAL
    ]

    expected = [
        x[1]
        for x in CONTEXTUAL
    ]

    pred, conf = model.predict(
        texts
    )

    acc = accuracy_score(
        expected,
        pred,
    )

    macro = f1_score(
        expected,
        pred,
        average="macro",
    )

    details = pd.DataFrame({
        "text": texts,
        "expected": expected,
        "predicted": pred,
        "confidence": conf,
    })

    details["correct"] = (
        details["expected"]
        == details["predicted"]
    )

    details.to_csv(
        OUT
        / f"{name}_contextual.csv",
        index=False,
    )

    return acc * 100, macro * 100, details


def evaluate_targeted(model, name):

    texts = [
        x[0]
        for x in TARGETED
    ]

    expected = [
        x[1]
        for x in TARGETED
    ]

    pred, conf = model.predict(
        texts
    )

    acc = accuracy_score(
        expected,
        pred,
    )

    details = pd.DataFrame({
        "text": texts,
        "expected": expected,
        "predicted": pred,
        "confidence": conf,
    })

    details["correct"] = (
        details["expected"]
        == details["predicted"]
    )

    details.to_csv(
        OUT
        / f"{name}_targeted.csv",
        index=False,
    )

    return acc * 100, details


def evaluate_ood(model, name):

    pred, conf = model.predict(
        OOD
    )

    rejected = [
        float(c) < OOD_THRESHOLD
        for c in conf
    ]

    rejection_rate = (
        sum(rejected)
        / len(rejected)
        * 100
    )

    details = pd.DataFrame({
        "text": OOD,
        "predicted": pred,
        "confidence": conf,
        "rejected_at_0_70": rejected,
    })

    details.to_csv(
        OUT
        / f"{name}_ood.csv",
        index=False,
    )

    return rejection_rate, details


# ------------------------------------------------------------
# Run all tests
# ------------------------------------------------------------

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

models = [
    ("Current INT8", current),
    ("V2 INT8", v2),
]

results = []

for name, model in models:

    print("\n" + "-" * 80)
    print(name)
    print("-" * 80)

    unseen_acc, unseen_f1 = (
        evaluate_unseen(
            model,
            name.lower().replace(
                " ",
                "_",
            ),
        )
    )

    contextual_acc, contextual_f1, contextual_details = (
        evaluate_contextual(
            model,
            name.lower().replace(
                " ",
                "_",
            ),
        )
    )

    targeted_acc, targeted_details = (
        evaluate_targeted(
            model,
            name.lower().replace(
                " ",
                "_",
            ),
        )
    )

    ood_rate, ood_details = (
        evaluate_ood(
            model,
            name.lower().replace(
                " ",
                "_",
            ),
        )
    )

    print(
        f"Unseen accuracy: {unseen_acc:.2f}%"
    )

    print(
        f"Unseen macro F1: {unseen_f1:.2f}%"
    )

    print(
        f"Contextual accuracy: "
        f"{contextual_acc:.2f}%"
    )

    print(
        f"Contextual macro F1: "
        f"{contextual_f1:.2f}%"
    )

    print(
        f"Targeted accuracy: "
        f"{targeted_acc:.2f}%"
    )

    print(
        f"OOD rejection @ 0.70: "
        f"{ood_rate:.2f}%"
    )

    results.append({
        "Model": name,
        "Size": (
            f"{model.size_mb:.3f} MB"
        ),
        "Unseen": (
            f"{unseen_acc:.2f}%"
        ),
        "Unseen Macro F1": (
            f"{unseen_f1:.2f}%"
        ),
        "Contextual": (
            f"{contextual_acc:.2f}%"
        ),
        "Contextual Macro F1": (
            f"{contextual_f1:.2f}%"
        ),
        "Targeted": (
            f"{targeted_acc:.2f}%"
        ),
        "OOD": (
            f"{ood_rate:.2f}%"
        ),
    })


# ------------------------------------------------------------
# Final table
# ------------------------------------------------------------

comparison = pd.DataFrame(
    results
)

comparison.to_csv(
    OUT / "v2_int8_comparison.csv",
    index=False,
)

print("\n")
print("=" * 80)
print("FINAL COMPARISON")
print("=" * 80)

print(
    comparison.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Extract V2 numbers
# ------------------------------------------------------------

v2row = comparison[
    comparison["Model"]
    == "V2 INT8"
].iloc[0]


def pct(value):
    return float(
        str(value)
        .replace("%", "")
        .strip()
    )


v2_unseen = pct(
    v2row["Unseen"]
)

v2_contextual = pct(
    v2row["Contextual"]
)

v2_targeted = pct(
    v2row["Targeted"]
)

v2_ood = pct(
    v2row["OOD"]
)

size_wins = (
    v2.size_mb
    < current.size_mb
)

unseen_pass = (
    v2_unseen
    >= BASELINE["unseen"]
)

context_pass = (
    v2_contextual
    > BASELINE["contextual"]
)

ood_pass = (
    v2_ood
    >= BASELINE["ood"]
)

target_pass = (
    v2_targeted
    >= 95.0
)


print("\n")
print("=" * 80)
print("V2 INT8 vs LOCKED CURRENT INT8")
print("=" * 80)

print(
    f"Size       : "
    f"{current.size_mb:.3f} MB -> "
    f"{v2.size_mb:.3f} MB "
    f"({'smaller' if size_wins else 'larger'})"
)

print(
    f"Unseen     : "
    f"{BASELINE['unseen']:.2f}% -> "
    f"{v2_unseen:.2f}% "
    f"({v2_unseen - BASELINE['unseen']:+.2f} pp)"
)

print(
    f"Contextual : "
    f"{BASELINE['contextual']:.2f}% -> "
    f"{v2_contextual:.2f}% "
    f"({v2_contextual - BASELINE['contextual']:+.2f} pp)"
)

print(
    f"OOD @ .70  : "
    f"{BASELINE['ood']:.2f}% -> "
    f"{v2_ood:.2f}% "
    f"({v2_ood - BASELINE['ood']:+.2f} pp)"
)

print(
    f"Targeted   : "
    f"--- -> "
    f"{v2_targeted:.2f}%"
)

print("\nGATES:")

print(
    "  Size       :",
    "PASS" if size_wins else "INFO",
)

print(
    "  Unseen     :",
    "PASS" if unseen_pass else "FAIL",
)

print(
    "  Contextual :",
    "PASS" if context_pass else "FAIL",
)

print(
    "  OOD        :",
    "PASS" if ood_pass else "FAIL",
)

print(
    "  Targeted   :",
    "PASS" if target_pass else "FAIL",
)


if (
    unseen_pass
    and context_pass
    and ood_pass
    and target_pass
):
    print("\nSTATUS: V2 INT8 PASSES THE DEPLOYMENT GATE")
    print(
        "V2 INT8 is eligible to become the new "
        "deployment candidate."
    )
else:
    print("\nSTATUS: KEEP CURRENT INT8 BASELINE")
    print(
        "V2 INT8 did not pass all required gates."
    )

print(
    "\nCurrent INT8 baseline was NOT modified."
)

print(
    "\nSaved results:",
    OUT,
)
