
import os
import json
import numpy as np
import pandas as pd
import torch
import onnxruntime as ort

from sklearn.metrics import accuracy_score, f1_score

# ============================================================
# THREE-MODEL COMPARISON
#
# Models:
#   1) Current INT8
#   2) V5 Medium
#   3) V5 Large
#
# Tests:
#   A. Existing 595-sample unseen semantic stress test
#   B. NEW contextual-action test cases
#   C. NEW OOD / unsupported-intent test cases
#
# OOD is measured using a confidence threshold because the
# current 11-class models do not contain a trained
# defaultFallbackIntent class.
#
# IMPORTANT:
#   Contextual and OOD cases below are deliberately separate
#   from the V5 training examples.
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------
# Model paths
# ------------------------------------------------------------

CURRENT_INT8 = os.path.join(
    SCRIPT_DIR,
    "tiny_semantic_student_v1_int8.onnx"
)

CURRENT_MODEL_DIR = os.path.join(
    SCRIPT_DIR,
    "tiny_semantic_student_v1"
)

MODEL_B_DIR = os.path.join(
    SCRIPT_DIR,
    "tiny_semantic_student_model_b"
)

MODEL_C_DIR = os.path.join(
    SCRIPT_DIR,
    "tiny_semantic_student_model_c"
)

STRESS_FILE = os.path.join(
    SCRIPT_DIR,
    "unseen_semantic_stress_test.csv"
)

# ------------------------------------------------------------
# OOD confidence threshold
#
# This is NOT a calibrated production threshold yet.
# It is only used to compare the three models under the
# same rule.
# ------------------------------------------------------------

OOD_THRESHOLD = 0.70

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# MODEL DEFINITIONS
# ============================================================

class TinySemanticStudent(torch.nn.Module):

    def __init__(
        self,
        vocab_size,
        num_classes,
        embed_dim,
        heads,
        layers,
        ff_dim,
        max_len,
        dropout=0.10
    ):
        super().__init__()

        self.embedding = torch.nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        self.position = torch.nn.Embedding(
            max_len,
            embed_dim
        )

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers
        )

        self.norm = torch.nn.LayerNorm(
            embed_dim
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(
                embed_dim,
                embed_dim
            ),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(
                embed_dim,
                num_classes
            )
        )

    def forward(self, x):

        pad = x.eq(0)

        pos = torch.arange(
            x.size(1),
            device=x.device
        ).unsqueeze(0)

        h = (
            self.embedding(x)
            + self.position(pos)
        )

        h = self.encoder(
            h,
            src_key_padding_mask=pad
        )

        valid = (
            (~pad)
            .unsqueeze(-1)
            .float()
        )

        pooled = (
            h * valid
        ).sum(dim=1) / valid.sum(
            dim=1
        ).clamp(min=1.0)

        return self.classifier(
            self.norm(pooled)
        )


# ============================================================
# HELPERS
# ============================================================

def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def load_labels(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return [
            x.strip()
            for x in f
            if x.strip()
        ]


def tokenize(text, vocab, max_len):
    ids = []

    for token in str(text).lower().split():

        token = token.strip(
            ".,!?;:\"'()[]{}"
        )

        if token:
            ids.append(
                vocab.get(token, 1)
            )

    ids = ids[:max_len]

    if len(ids) < max_len:
        ids += [0] * (
            max_len - len(ids)
        )

    return ids


def softmax(logits):
    logits = np.asarray(
        logits,
        dtype=np.float32
    )

    e = np.exp(
        logits - np.max(logits)
    )

    return e / e.sum()


def model_size_mb(path):
    return os.path.getsize(path) / (
        1024 * 1024
    )


# ============================================================
# LOAD PYTORCH MODEL
# ============================================================

def load_pytorch_model(model_dir):

    config = load_json(
        os.path.join(
            model_dir,
            "config.json"
        )
    )

    vocab = load_json(
        os.path.join(
            model_dir,
            "vocab.json"
        )
    )

    labels = load_labels(
        os.path.join(
            model_dir,
            "intent_labels.txt"
        )
    )

    model = TinySemanticStudent(
        vocab_size=config["vocab_size"],
        num_classes=config["num_classes"],
        embed_dim=config["embed_dim"],
        heads=config["num_heads"],
        layers=config["num_layers"],
        ff_dim=config["ff_dim"],
        max_len=config["max_len"]
    )

    weights = os.path.join(
        model_dir,
        "student_fp32.pt"
    )

    state = torch.load(
        weights,
        map_location="cpu"
    )

    model.load_state_dict(
        state
    )

    model.eval()

    return {
        "type": "pytorch",
        "model": model,
        "vocab": vocab,
        "labels": labels,
        "max_len": config["max_len"],
        "path": weights,
        "size_mb": model_size_mb(weights)
    }


# ============================================================
# LOAD CURRENT INT8 ONNX
# ============================================================

def load_onnx_model():

    config = load_json(
        os.path.join(
            CURRENT_MODEL_DIR,
            "config.json"
        )
    )

    vocab = load_json(
        os.path.join(
            CURRENT_MODEL_DIR,
            "vocab.json"
        )
    )

    labels = load_labels(
        os.path.join(
            CURRENT_MODEL_DIR,
            "intent_labels.txt"
        )
    )

    session = ort.InferenceSession(
        CURRENT_INT8,
        providers=[
            "CPUExecutionProvider"
        ]
    )

    return {
        "type": "onnx",
        "session": session,
        "input_name": session.get_inputs()[0].name,
        "output_name": session.get_outputs()[0].name,
        "vocab": vocab,
        "labels": labels,
        "max_len": config["max_len"],
        "path": CURRENT_INT8,
        "size_mb": model_size_mb(
            CURRENT_INT8
        )
    }


# ============================================================
# PREDICT
# ============================================================

def predict_pytorch(info, texts):

    x = np.asarray(
        [
            tokenize(
                t,
                info["vocab"],
                info["max_len"]
            )
            for t in texts
        ],
        dtype=np.int64
    )

    with torch.no_grad():

        logits = info["model"](
            torch.tensor(x)
        ).numpy()

    results = []

    for row in logits:

        probs = softmax(row)

        idx = int(
            np.argmax(probs)
        )

        results.append(
            (
                info["labels"][idx],
                float(probs[idx])
            )
        )

    return results


def predict_onnx(info, texts):

    x = np.asarray(
        [
            tokenize(
                t,
                info["vocab"],
                info["max_len"]
            )
            for t in texts
        ],
        dtype=np.int64
    )

    logits = info["session"].run(
        [info["output_name"]],
        {
            info["input_name"]: x
        }
    )[0]

    results = []

    for row in logits:

        probs = softmax(row)

        idx = int(
            np.argmax(probs)
        )

        results.append(
            (
                info["labels"][idx],
                float(probs[idx])
            )
        )

    return results


def predict(info, texts):

    if info["type"] == "onnx":
        return predict_onnx(
            info,
            texts
        )

    return predict_pytorch(
        info,
        texts
    )


# ============================================================
# TEST A: EXISTING 595-SAMPLE UNSEEN TEST
# ============================================================

def run_unseen_test(info, stress):

    texts = stress["text"].tolist()

    predictions = predict(
        info,
        texts
    )

    pred_labels = [
        x[0]
        for x in predictions
    ]

    acc = accuracy_score(
        stress["intent"],
        pred_labels
    )

    f1 = f1_score(
        stress["intent"],
        pred_labels,
        average="macro"
    )

    return acc, f1


# ============================================================
# TEST B: NEW CONTEXTUAL-ACTION TEST
#
# IMPORTANT:
# These are NEW phrasings, not the contextual examples
# used in the V4/V5 training scripts.
# ============================================================

CONTEXTUAL_CASES = [

    # Increase
    (
        "the audio seems a bit soft could you raise it now",
        "device.volume.increase"
    ),
    (
        "I can hear it but it is weaker than before please turn it up",
        "device.volume.increase"
    ),
    (
        "the sound dropped a little can you increase the volume",
        "device.volume.increase"
    ),
    (
        "everything sounds low right now make it louder",
        "device.volume.increase"
    ),
    (
        "I want more volume because this is too quiet",
        "device.volume.increase"
    ),
    (
        "the hearing aids are sounding faint please raise them",
        "device.volume.increase"
    ),
    (
        "it got softer than I expected increase the sound",
        "device.volume.increase"
    ),
    (
        "can you turn the audio higher it is barely loud enough",
        "device.volume.increase"
    ),

    # Decrease
    (
        "the audio is stronger than I need please lower it",
        "device.volume.decrease"
    ),
    (
        "I can hear it fine but it is a little too loud turn it down",
        "device.volume.decrease"
    ),
    (
        "the sound increased too much reduce the volume",
        "device.volume.decrease"
    ),
    (
        "everything is loud right now make it softer",
        "device.volume.decrease"
    ),
    (
        "I want less volume because this is too strong",
        "device.volume.decrease"
    ),
    (
        "the hearing aids are sounding intense please lower them",
        "device.volume.decrease"
    ),
    (
        "it became louder than I wanted decrease the sound",
        "device.volume.decrease"
    ),
    (
        "can you turn the audio lower it is more than enough",
        "device.volume.decrease"
    ),

    # Mute
    (
        "I can still hear the audio please make it totally silent",
        "device.volume.mute"
    ),
    (
        "there is some sound left turn all of it off",
        "device.volume.mute"
    ),
    (
        "I don't want any audio at all make it silent",
        "device.volume.mute"
    ),
    (
        "stop all sound from the hearing aids",
        "device.volume.mute"
    ),
    (
        "the volume is low but I need complete silence",
        "device.volume.mute"
    ),
    (
        "make sure nothing is audible",
        "device.volume.mute"
    ),
    (
        "shut off the hearing aid sound completely",
        "device.volume.mute"
    ),
    (
        "I want zero sound right now",
        "device.volume.mute"
    ),

    # Unmute
    (
        "I want the audio back please restore it",
        "device.volume.unmute"
    ),
    (
        "there is no sound now can you enable it again",
        "device.volume.unmute"
    ),
    (
        "the hearing aids are quiet because they are muted bring sound back",
        "device.volume.unmute"
    ),
    (
        "restore audio so I can hear again",
        "device.volume.unmute"
    ),
    (
        "please enable the hearing aid sound",
        "device.volume.unmute"
    ),
    (
        "turn the audio back on after muting it",
        "device.volume.unmute"
    ),
    (
        "I need the sound restored",
        "device.volume.unmute"
    ),
    (
        "unmute the hearing aids so audio returns",
        "device.volume.unmute"
    ),
]


def run_contextual_test(info):

    texts = [
        x[0]
        for x in CONTEXTUAL_CASES
    ]

    truth = [
        x[1]
        for x in CONTEXTUAL_CASES
    ]

    predictions = predict(
        info,
        texts
    )

    pred_labels = [
        x[0]
        for x in predictions
    ]

    acc = accuracy_score(
        truth,
        pred_labels
    )

    f1 = f1_score(
        truth,
        pred_labels,
        average="macro"
    )

    return (
        acc,
        f1,
        predictions,
        truth,
        texts
    )


# ============================================================
# TEST C: NEW OOD / UNSUPPORTED INPUTS
#
# These should NOT map to one of the 11 supported intents.
#
# Because the current model has no fallback output class,
# OOD is measured as:
#
#   confidence < OOD_THRESHOLD
#
# This is a comparison metric only.
# ============================================================

OOD_CASES = [
    "I need to go to the airport tomorrow",
    "what is the weather going to be today",
    "call my brother when you get a chance",
    "play some relaxing music",
    "open the camera",
    "what time is it right now",
    "how far away is the airport",
    "tell me a joke",
    "I am going shopping this afternoon",
    "send an email to my manager",
    "what is my battery percentage on the phone",
    "set an alarm for six in the morning",
    "show me the nearest restaurant",
    "translate this sentence into Hindi",
    "what is the capital of France",
    "start navigation to the airport",
    "take a photo",
    "open my calendar",
    "what day is tomorrow",
    "play the latest news",
    "I need directions home",
    "connect to WiFi",
    "turn on Bluetooth",
    "check my email",
    "what is the temperature outside",
    "remind me to buy milk next week",
    "book me a taxi",
    "send a WhatsApp message",
    "open YouTube",
    "I have a meeting at three",
    "tell me today's date",
    "search for a nearby pharmacy",
]


def run_ood_test(info):

    predictions = predict(
        info,
        OOD_CASES
    )

    confidences = np.asarray(
        [x[1] for x in predictions]
    )

    rejected = (
        confidences < OOD_THRESHOLD
    )

    rejection_rate = (
        rejected.mean()
    )

    return (
        rejection_rate,
        confidences,
        predictions
    )


# ============================================================
# LOAD ALL THREE MODELS
# ============================================================

print("\nLoading models...")

models = {}

if not os.path.exists(CURRENT_INT8):
    raise FileNotFoundError(
        "Current INT8 model not found:\n"
        + CURRENT_INT8
    )

models["Current INT8"] = load_onnx_model()

if not os.path.exists(
    os.path.join(
        MODEL_B_DIR,
        "student_fp32.pt"
    )
):
    raise FileNotFoundError(
        "Model B not found:\n"
        + MODEL_B_DIR
    )

models["V5 Medium"] = load_pytorch_model(
    MODEL_B_DIR
)

if not os.path.exists(
    os.path.join(
        MODEL_C_DIR,
        "student_fp32.pt"
    )
):
    raise FileNotFoundError(
        "Model C not found:\n"
        + MODEL_C_DIR
    )

models["V5 Large"] = load_pytorch_model(
    MODEL_C_DIR
)

for name, info in models.items():

    print(
        name,
        "loaded | size:",
        round(
            info["size_mb"],
            3
        ),
        "MB"
    )

# ============================================================
# LOAD 595 UNSEEN TEST
# ============================================================

if not os.path.exists(
    STRESS_FILE
):
    raise FileNotFoundError(
        "Missing:\n"
        + STRESS_FILE
    )

stress = pd.read_csv(
    STRESS_FILE
)

# ============================================================
# RUN COMPARISON
# ============================================================

rows = []

all_context_details = []
all_ood_details = []

for name, info in models.items():

    print(
        "\n========================================"
    )
    print(name)
    print(
        "========================================"
    )

    # --------------------------------------------------------
    # A. Unseen
    # --------------------------------------------------------

    unseen_acc, unseen_f1 = (
        run_unseen_test(
            info,
            stress
        )
    )

    print(
        "Unseen accuracy:",
        round(
            unseen_acc * 100,
            2
        ),
        "%"
    )

    print(
        "Unseen macro F1:",
        round(
            unseen_f1 * 100,
            2
        ),
        "%"
    )

    # --------------------------------------------------------
    # B. Contextual
    # --------------------------------------------------------

    (
        context_acc,
        context_f1,
        context_preds,
        context_truth,
        context_texts
    ) = run_contextual_test(
        info
    )

    print(
        "Contextual accuracy:",
        round(
            context_acc * 100,
            2
        ),
        "%"
    )

    print(
        "Contextual macro F1:",
        round(
            context_f1 * 100,
            2
        ),
        "%"
    )

    # --------------------------------------------------------
    # C. OOD
    # --------------------------------------------------------

    (
        ood_rejection,
        ood_conf,
        ood_preds
    ) = run_ood_test(
        info
    )

    print(
        "OOD rejection @",
        OOD_THRESHOLD,
        ":",
        round(
            ood_rejection * 100,
            2
        ),
        "%"
    )

    rows.append({
        "Model": name,
        "Size": f"{info['size_mb']:.3f} MB",
        "Unseen": f"{unseen_acc * 100:.2f}%",
        "Contextual": f"{context_acc * 100:.2f}%",
        "OOD": f"{ood_rejection * 100:.2f}%"
    })

    # Save detailed contextual results
    for text, truth, pred in zip(
        context_texts,
        context_truth,
        context_preds
    ):
        all_context_details.append({
            "model": name,
            "text": text,
            "expected": truth,
            "predicted": pred[0],
            "confidence": pred[1],
            "correct": pred[0] == truth
        })

    # Save detailed OOD results
    for text, pred in zip(
        OOD_CASES,
        ood_preds
    ):
        all_ood_details.append({
            "model": name,
            "text": text,
            "predicted": pred[0],
            "confidence": pred[1],
            "rejected_as_ood": pred[1] < OOD_THRESHOLD
        })

# ============================================================
# FINAL TABLE
# ============================================================

comparison = pd.DataFrame(
    rows,
    columns=[
        "Model",
        "Size",
        "Unseen",
        "Contextual",
        "OOD"
    ]
)

print("\n\n")
print("=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)

print(
    comparison.to_string(
        index=False
    )
)

print("\nMarkdown table:\n")

print("| Model | Size | Unseen | Contextual | OOD |")
print("|---|---:|---:|---:|---:|")

for _, row in comparison.iterrows():

    print(
        f"| {row['Model']} | "
        f"{row['Size']} | "
        f"{row['Unseen']} | "
        f"{row['Contextual']} | "
        f"{row['OOD']} |"
    )

# ============================================================
# SAVE RESULTS
# ============================================================

comparison.to_csv(
    os.path.join(
        SCRIPT_DIR,
        "three_model_comparison.csv"
    ),
    index=False
)

pd.DataFrame(
    all_context_details
).to_csv(
    os.path.join(
        SCRIPT_DIR,
        "three_model_contextual_details.csv"
    ),
    index=False
)

pd.DataFrame(
    all_ood_details
).to_csv(
    os.path.join(
        SCRIPT_DIR,
        "three_model_ood_details.csv"
    ),
    index=False
)

print("\nSaved:")
print("three_model_comparison.csv")
print("three_model_contextual_details.csv")
print("three_model_ood_details.csv")

print("\n")
print(
    "OOD note: OOD is NOT a trained fallback class."
)
print(
    f"It is rejection rate at confidence < {OOD_THRESHOLD:.2f}."
)
print(
    "The threshold should be calibrated later using a"
)
print(
    "separate validation/OOD calibration set before production."
)
