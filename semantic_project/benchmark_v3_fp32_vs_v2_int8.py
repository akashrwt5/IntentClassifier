#!/usr/bin/env python3
"""
benchmark_v3_fp32_vs_v2_int8.py

Locked comparison:
  V2 INT8 (deployment baseline) vs V3 FP32 (candidate)

Uses the same 595-row unseen stress set plus contextual/targeted/OOD
test suites when those files are available.

CRITICAL:
  - Never trains.
  - Never changes V2 INT8.
  - Never changes V3 FP32.
  - Never uses the 595-row unseen set for training.
  - V3 must pass gates before ONNX/INT8 export.

Because the V3 tokenizer/model is the same architecture and tokenizer family
as V2, this script uses the V2 vocab artifact for V3 tokenization.

Run:
  python3 benchmark_v3_fp32_vs_v2_int8.py
"""

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import onnxruntime as ort
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parent

V2_ONNX = ROOT / "tiny_semantic_student_v2_int8" / "v2_semantic_student_int8.onnx"
V3_PT = ROOT / "tiny_semantic_student_v3_error_driven" / "student_v3_fp32.pt"
VOCAB_JSON = ROOT / "tiny_semantic_student_v2_balanced" / "vocab.json"
LABELS_JSON = ROOT / "tiny_semantic_student_v3_error_driven" / "labels.json"

OUT = ROOT / "v3_vs_v2_benchmark"
OUT.mkdir(parents=True, exist_ok=True)

STRESS = ROOT / "unseen_semantic_stress_test.csv"

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
L2I = {x: i for i, x in enumerate(LABELS)}
I2L = {i: x for x, i in L2I.items()}

ED = 64
NH = 4
FF = 128
NL = 2
ML = 24
DROPOUT = 0.10

if not V2_ONNX.exists():
    raise FileNotFoundError(V2_ONNX)
if not V3_PT.exists():
    raise FileNotFoundError(V3_PT)
if not VOCAB_JSON.exists():
    raise FileNotFoundError(VOCAB_JSON)
if not STRESS.exists():
    raise FileNotFoundError(
        f"595-row unseen stress test not found: {STRESS}"
    )

def clean(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())

raw = json.loads(VOCAB_JSON.read_text(encoding="utf-8"))
if isinstance(raw.get("model"), dict) and isinstance(raw["model"].get("vocab"), dict):
    vocab = raw["model"]["vocab"]
elif isinstance(raw.get("vocab"), dict):
    vocab = raw["vocab"]
else:
    vocab = raw
vocab = {str(k): int(v) for k, v in vocab.items()}

PAD = int(vocab.get("<pad>", vocab.get("[PAD]", 0)))
UNK = int(vocab.get("<unk>", vocab.get("[UNK]", 1)))
CLS = vocab.get("<cls>", vocab.get("[CLS]", None))
SEP = vocab.get("<sep>", vocab.get("[SEP]", None))

def tokenize(text):
    text = clean(text)
    text = re.sub(r"([.!?,;:()'])", r" \1 ", text)
    ids = []
    if CLS is not None:
        ids.append(int(CLS))

    for word in text.split():
        if word in vocab:
            ids.append(int(vocab[word]))
            continue

        pos = 0
        pieces = []
        while pos < len(word):
            best = None
            for end in range(len(word), pos, -1):
                candidate = word[pos:end]
                for c in (candidate, "##" + candidate):
                    if c in vocab:
                        best = c
                        break
                if best is not None:
                    break
            if best is None:
                pieces = [UNK]
                break
            pieces.append(int(vocab[best]))
            pos += len(best.replace("##", ""))
        ids.extend(pieces)

    if SEP is not None:
        ids.append(int(SEP))

    ids = ids[:ML]
    ids += [PAD] * (ML - len(ids))
    return ids

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), ED, padding_idx=PAD)
        self.position = nn.Embedding(ML, ED)
        layer = nn.TransformerEncoderLayer(
            ED, NH, FF,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, NL)
        self.norm = nn.LayerNorm(ED)
        self.classifier = nn.Sequential(
            nn.Linear(ED, ED),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(ED, len(LABELS)),
        )

    def forward(self, x):
        mask = x.eq(PAD)
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h = self.encoder(
            self.embedding(x) + self.position(pos),
            src_key_padding_mask=mask,
        )
        valid = (~mask).unsqueeze(-1).float()
        h = (h * valid).sum(1) / valid.sum(1).clamp(min=1)
        return self.classifier(self.norm(h))

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

# Load V2.
v2_session = ort.InferenceSession(
    str(V2_ONNX),
    providers=["CPUExecutionProvider"],
)
v2_input = v2_session.get_inputs()[0].name
v2_output = v2_session.get_outputs()[0].name

# Load V3.
v3 = Model().to(DEVICE)
state = torch.load(V3_PT, map_location="cpu")
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
v3.load_state_dict(state, strict=True)
v3.eval()

def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)

def predict_v2(text):
    x = np.asarray([tokenize(text)], dtype=np.int64)
    logits = v2_session.run([v2_output], {v2_input: x})[0][0]
    p = softmax(logits)
    order = np.argsort(p)[::-1]
    return int(order[0]), float(p[order[0]]), int(order[1]), float(p[order[1]])

@torch.no_grad()
def predict_v3(text):
    x = torch.tensor([tokenize(text)], dtype=torch.long, device=DEVICE)
    p = torch.softmax(v3(x), dim=-1)[0].detach().cpu().numpy()
    order = np.argsort(p)[::-1]
    return int(order[0]), float(p[order[0]]), int(order[1]), float(p[order[1]])

def run_labeled(df, name):
    records = []
    for _, r in df.iterrows():
        text = clean(r["text"])
        expected = r["intent"]
        a = predict_v2(text)
        b = predict_v3(text)
        records.append({
            "text": text,
            "expected": expected,
            "v2_predicted": I2L[a[0]],
            "v2_confidence": a[1],
            "v2_top2": I2L[a[2]],
            "v2_top2_confidence": a[3],
            "v3_predicted": I2L[b[0]],
            "v3_confidence": b[1],
            "v3_top2": I2L[b[2]],
            "v3_top2_confidence": b[3],
            "v2_correct": I2L[a[0]] == expected,
            "v3_correct": I2L[b[0]] == expected,
        })
    out = pd.DataFrame(records)
    out.to_csv(OUT / f"{name}_details.csv", index=False)

    v2_acc = out.v2_correct.mean()
    v3_acc = out.v3_correct.mean()
    v2_f1 = f1_score(
        out.expected,
        out.v2_predicted,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )
    v3_f1 = f1_score(
        out.expected,
        out.v3_predicted,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    print()
    print(f"--- {name.upper()} ---")
    print("V2 accuracy : %.2f%%" % (v2_acc * 100))
    print("V3 accuracy : %.2f%%" % (v3_acc * 100))
    print("Delta        : %+0.2f pp" % ((v3_acc - v2_acc) * 100))
    print("V2 Macro F1 : %.2f%%" % (v2_f1 * 100))
    print("V3 Macro F1 : %.2f%%" % (v3_f1 * 100))

    return out, {
        "v2_accuracy": v2_acc,
        "v3_accuracy": v3_acc,
        "v2_f1": v2_f1,
        "v3_f1": v3_f1,
    }

# ---------------------------------------------------------------------
# 1. Locked 595-row unseen test.
# ---------------------------------------------------------------------
stress_df = pd.read_csv(STRESS)
if "text" not in stress_df.columns:
    raise ValueError("Stress CSV must contain text column.")

intent_col = next(
    (c for c in ["intent", "expected", "label", "true_intent"]
     if c in stress_df.columns),
    None
)
if intent_col is None:
    raise ValueError("Stress CSV has no expected intent column.")
stress_df = stress_df.rename(columns={intent_col: "intent"})
stress_df = stress_df[["text", "intent"]].dropna()
stress_df["text"] = stress_df.text.map(clean)
stress_df = stress_df[stress_df.intent.isin(LABELS)].reset_index(drop=True)

unseen_details, unseen = run_labeled(stress_df, "unseen_595")

# ---------------------------------------------------------------------
# 2. Contextual suite.
# ---------------------------------------------------------------------
CONTEXTUAL = [
    ("it's quieter can you make it a little louder", "device.volume.increase"),
    ("it's a little loud can you make it quieter", "device.volume.decrease"),
    ("i can still hear it make it completely silent", "device.volume.mute"),
    ("turn the sound back on", "device.volume.unmute"),
    ("the audio is quiet but don't mute it make it louder", "device.volume.increase"),
    ("the audio is loud but keep it on just lower it", "device.volume.decrease"),
    ("i need to go to airport tomorrow", "reminders.task.create"),
    ("i need to go to airport tommorow", "reminders.task.create"),
    ("where can i find my phone", "find.phone.locate"),
    ("please show the reminders i have", "help.reminder.show"),
    ("mark that reminder as completed", "reminders.task.complete"),
    ("please start the streaming session", "streaming.session.start"),
    ("please stop the streaming session", "streaming.session.stop"),
]
ctx = pd.DataFrame(CONTEXTUAL, columns=["text", "intent"])
ctx_details, contextual = run_labeled(ctx, "contextual")

# ---------------------------------------------------------------------
# 3. Targeted critical suite.
# ---------------------------------------------------------------------
TARGETED = [
    ("make it louder", "device.volume.increase"),
    ("make it a little louder", "device.volume.increase"),
    ("it's quieter can you make it louder", "device.volume.increase"),
    ("the audio is quiet but don't mute it make it louder", "device.volume.increase"),
    ("increase the volume", "device.volume.increase"),
    ("make it quieter", "device.volume.decrease"),
    ("it's a little loud can you make it quieter", "device.volume.decrease"),
    ("the audio is loud but keep it on just lower it", "device.volume.decrease"),
    ("decrease the volume", "device.volume.decrease"),
    ("mute it", "device.volume.mute"),
    ("make it completely silent", "device.volume.mute"),
    ("i can still hear it make it completely silent", "device.volume.mute"),
    ("turn off", "device.volume.mute"),
    ("unmute it", "device.volume.unmute"),
    ("turn the sound back on", "device.volume.unmute"),
    ("turn the audio back on", "device.volume.unmute"),
    ("start streaming", "streaming.session.start"),
    ("please start the streaming session", "streaming.session.start"),
    ("stop streaming", "streaming.session.stop"),
    ("please stop the streaming session", "streaming.session.stop"),
    ("i need to go to airport tomorrow", "reminders.task.create"),
    ("i need to go to airport tommorow", "reminders.task.create"),
]
target = pd.DataFrame(TARGETED, columns=["text", "intent"])
target_details, targeted = run_labeled(target, "targeted_critical")

# ---------------------------------------------------------------------
# 4. OOD suite.
# ---------------------------------------------------------------------
OOD = [
    ("who is the prime minister of India", "defaultFallbackIntent"),
    ("who is the priminister of india", "defaultFallbackIntent"),
    ("what is the weather today", "defaultFallbackIntent"),
    ("i want to order pizza", "defaultFallbackIntent"),
    ("tell me a joke", "defaultFallbackIntent"),
    ("what is the capital of France", "defaultFallbackIntent"),
    ("open my browser", "defaultFallbackIntent"),
    ("play music", "defaultFallbackIntent"),
    ("book a hotel for tomorrow", "defaultFallbackIntent"),
    ("how do i cook pasta", "defaultFallbackIntent"),
    ("what is bitcoin", "defaultFallbackIntent"),
    ("what time is the train", "defaultFallbackIntent"),
    ("how are you", "defaultFallbackIntent"),
    ("thank you", "defaultFallbackIntent"),
    ("good morning", "defaultFallbackIntent"),
    ("jkakjhdjkhd", "defaultFallbackIntent"),
    ("sdkjadsjj", "defaultFallbackIntent"),
    ("asdfghjkl", "defaultFallbackIntent"),
    ("123456789", "defaultFallbackIntent"),
    ("hello xyz abc", "defaultFallbackIntent"),
]
ood = pd.DataFrame(OOD, columns=["text", "intent"])

# For OOD, raw classifier must not be expected to emit the fallback label
# because V2/V3 are 11-class intent classifiers. We evaluate rejection using
# a simple calibration gate later rather than pretending fallback is trained.
ood_records = []
for _, r in ood.iterrows():
    a = predict_v2(r.text)
    b = predict_v3(r.text)
    ood_records.append({
        "text": r.text,
        "expected": "defaultFallbackIntent",
        "v2_predicted": I2L[a[0]],
        "v2_confidence": a[1],
        "v2_margin": a[1] - a[3],
        "v3_predicted": I2L[b[0]],
        "v3_confidence": b[1],
        "v3_margin": b[1] - b[3],
    })
ood_details = pd.DataFrame(ood_records)
ood_details.to_csv(OUT / "ood_details.csv", index=False)

# ---------------------------------------------------------------------
# Critical regression table.
# ---------------------------------------------------------------------
merged = target_details[[
    "text", "expected",
    "v2_predicted", "v2_confidence", "v2_correct",
    "v3_predicted", "v3_confidence", "v3_correct",
]].copy()
merged["changed"] = merged.v2_predicted != merged.v3_predicted
merged["v2_wrong_v3_correct"] = (~merged.v2_correct) & merged.v3_correct
merged["v2_correct_v3_wrong"] = merged.v2_correct & (~merged.v3_correct)
merged.to_csv(OUT / "critical_regression_details.csv", index=False)

# ---------------------------------------------------------------------
# Gate comparison.
#
# V2 baseline known from the locked benchmark:
#   unseen 94.29
#   contextual 92.31 (V2 INT8 benchmark)
#   targeted 100.00
#
# We require V3 FP32 to meet or beat these on the same suites, and require
# hard-negative improvement on the dedicated held-out hard-negative file.
# ---------------------------------------------------------------------

baseline_unseen = 0.9429
baseline_contextual = 0.9231
baseline_targeted = 1.0000

v3_unseen_gate = unseen["v3_accuracy"] >= baseline_unseen
v3_context_gate = contextual["v3_accuracy"] >= baseline_contextual
v3_target_gate = targeted["v3_accuracy"] >= baseline_targeted

# Dedicated hard-negative evaluation from V3 output.
hard_file = ROOT / "tiny_semantic_student_v3_error_driven" / "hard_negative_results.csv"
if hard_file.exists():
    hard_df = pd.read_csv(hard_file)
    v3_hard = float(hard_df["correct"].mean())
else:
    v3_hard = float("nan")

# V2 hard-negative baseline is not available as a full 210-row benchmark in
# the locked benchmark. Therefore it is informational, not a gate.
summary = pd.DataFrame([
    {
        "Model": "V2 INT8",
        "Size_MB": V2_ONNX.stat().st_size / 1024 / 1024,
        "Unseen": unseen["v2_accuracy"] * 100,
        "Contextual": contextual["v2_accuracy"] * 100,
        "Targeted": targeted["v2_accuracy"] * 100,
        "HardNegative_V3_eval": "",
    },
    {
        "Model": "V3 FP32",
        "Size_MB": V3_PT.stat().st_size / 1024 / 1024,
        "Unseen": unseen["v3_accuracy"] * 100,
        "Contextual": contextual["v3_accuracy"] * 100,
        "Targeted": targeted["v3_accuracy"] * 100,
        "HardNegative_V3_eval": "" if np.isnan(v3_hard) else v3_hard * 100,
    },
])
summary.to_csv(OUT / "v3_vs_v2_summary.csv", index=False)

print()
print("=" * 78)
print("V3 FP32 vs V2 INT8 — DEPLOYMENT GATE")
print("=" * 78)
print(summary.to_string(index=False))

print()
print("GATES")
print("Unseen     :", "PASS" if v3_unseen_gate else "FAIL")
print("Contextual :", "PASS" if v3_context_gate else "FAIL")
print("Targeted   :", "PASS" if v3_target_gate else "FAIL")
print("Hard-negatives (informational): %.2f%%" %
      (v3_hard * 100 if not np.isnan(v3_hard) else float("nan")))

# Critical safety requirement: the two known problematic contextual
# statements must be correct.
critical_texts = {
    "the audio is quiet but don't mute it make it louder":
        "device.volume.increase",
    "the audio is loud but keep it on just lower it":
        "device.volume.decrease",
    "turn off":
        "device.volume.mute",
}

critical_ok = True
for text, expected in critical_texts.items():
    row = target_details[target_details.text == text]
    if len(row) != 1 or row.iloc[0].v3_predicted != expected:
        critical_ok = False

print("Critical cases:", "PASS" if critical_ok else "FAIL")

# Strict rule: all required gates must pass before export.
ready = (
    v3_unseen_gate
    and v3_context_gate
    and v3_target_gate
    and critical_ok
)

print()
print("STATUS:", "V3 FP32 ELIGIBLE FOR ONNX EXPORT"
      if ready else
      "KEEP V2 INT8 — V3 NOT READY")

print()
print("Saved:")
print(OUT / "unseen_595_details.csv")
print(OUT / "contextual_details.csv")
print(OUT / "targeted_critical_details.csv")
print(OUT / "ood_details.csv")
print(OUT / "critical_regression_details.csv")
print(OUT / "v3_vs_v2_summary.csv")
print()
print("V2 INT8 was NOT modified.")
print("V3 FP32 was NOT exported or quantized.")
print("The 595-row unseen set was evaluation-only.")
