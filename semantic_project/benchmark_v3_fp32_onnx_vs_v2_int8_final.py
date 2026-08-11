#!/usr/bin/env python3
"""
FINAL PRODUCTION BENCHMARK
V3 FP32 ONNX vs LOCKED V2 INT8

This benchmark evaluates the ACTUAL exported V3 ONNX artifact.
No training. No quantization. No model modification.

Required:
  tiny_semantic_student_v3_fp32/v3_semantic_student_fp32.onnx
  tiny_semantic_student_v2_int8/v2_semantic_student_int8.onnx
  tiny_semantic_student_v2_balanced/vocab.json
  unseen_semantic_stress_test.csv

Run:
  python3 benchmark_v3_fp32_onnx_vs_v2_int8_final.py
"""

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parent

V2 = ROOT / "tiny_semantic_student_v2_int8" / "v2_semantic_student_int8.onnx"
V3 = ROOT / "tiny_semantic_student_v3_fp32" / "v3_semantic_student_fp32.onnx"
VOCAB = ROOT / "tiny_semantic_student_v2_balanced" / "vocab.json"
STRESS = ROOT / "unseen_semantic_stress_test.csv"

OUT = ROOT / "final_v3_fp32_onnx_benchmark"
OUT.mkdir(parents=True, exist_ok=True)

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

if not V2.exists():
    raise FileNotFoundError(V2)
if not V3.exists():
    raise FileNotFoundError(V3)
if not VOCAB.exists():
    raise FileNotFoundError(VOCAB)
if not STRESS.exists():
    raise FileNotFoundError(STRESS)

raw = json.loads(VOCAB.read_text(encoding="utf-8"))
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

def clean(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())

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

    ids = ids[:24]
    ids += [PAD] * (24 - len(ids))
    return ids

def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)

class OnnxModel:
    def __init__(self, path):
        self.path = path
        self.session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
        self.input = self.session.get_inputs()[0].name
        self.output = self.session.get_outputs()[0].name

    def predict(self, text):
        x = np.asarray([tokenize(text)], dtype=np.int64)
        logits = self.session.run(
            [self.output],
            {self.input: x},
        )[0][0]
        p = softmax(logits)
        order = np.argsort(p)[::-1]
        return (
            int(order[0]),
            float(p[order[0]]),
            int(order[1]),
            float(p[order[1]]),
        )

    def size_mb(self):
        return self.path.stat().st_size / 1024 / 1024

v2 = OnnxModel(V2)
v3 = OnnxModel(V3)

print("=" * 78)
print("FINAL PRODUCTION BENCHMARK — V3 FP32 ONNX vs V2 INT8")
print("=" * 78)
print(f"V2 INT8 : {V2}")
print(f"V3 FP32 : {V3}")
print(f"V2 size : {v2.size_mb():.3f} MB")
print(f"V3 size : {v3.size_mb():.3f} MB")
print()

# ---------------------------------------------------------------------
# Locked 595-row unseen evaluation.
# ---------------------------------------------------------------------
df = pd.read_csv(STRESS)
intent_col = next(
    (c for c in ["intent", "expected", "label", "true_intent"] if c in df.columns),
    None
)
if "text" not in df.columns or intent_col is None:
    raise ValueError("Stress CSV needs text + intent/expected/label column.")

df = df.rename(columns={intent_col: "intent"})
df = df[["text", "intent"]].dropna()
df["text"] = df["text"].map(clean)
df = df[df["intent"].isin(LABELS)].reset_index(drop=True)

records = []
for _, r in df.iterrows():
    a = v2.predict(r.text)
    b = v3.predict(r.text)

    records.append({
        "text": r.text,
        "expected": r.intent,
        "v2_predicted": LABELS[a[0]],
        "v2_confidence": a[1],
        "v2_top2": LABELS[a[2]],
        "v2_top2_confidence": a[3],
        "v3_predicted": LABELS[b[0]],
        "v3_confidence": b[1],
        "v3_top2": LABELS[b[2]],
        "v3_top2_confidence": b[3],
        "v2_correct": LABELS[a[0]] == r.intent,
        "v3_correct": LABELS[b[0]] == r.intent,
    })

unseen = pd.DataFrame(records)
unseen.to_csv(OUT / "unseen_595_details.csv", index=False)

v2_unseen_acc = unseen.v2_correct.mean()
v3_unseen_acc = unseen.v3_correct.mean()
v2_unseen_f1 = f1_score(
    unseen.expected, unseen.v2_predicted,
    labels=LABELS, average="macro", zero_division=0
)
v3_unseen_f1 = f1_score(
    unseen.expected, unseen.v3_predicted,
    labels=LABELS, average="macro", zero_division=0
)

print("--- UNSEEN 595 ---")
print(f"V2 accuracy : {v2_unseen_acc*100:.2f}%")
print(f"V3 accuracy : {v3_unseen_acc*100:.2f}%")
print(f"Delta       : {(v3_unseen_acc-v2_unseen_acc)*100:+.2f} pp")
print(f"V2 Macro F1 : {v2_unseen_f1*100:.2f}%")
print(f"V3 Macro F1 : {v3_unseen_f1*100:.2f}%")
print()

# ---------------------------------------------------------------------
# Contextual + targeted critical suite.
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

TARGETED = [
    ("make it louder", "device.volume.increase"),
    ("make it a little louder", "device.volume.increase"),
    ("it's quieter can you make it a little louder", "device.volume.increase"),
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

def evaluate_suite(items, filename):
    rows = []

    for text, expected in items:
        a = v2.predict(text)
        b = v3.predict(text)

        rows.append({
            "text": text,
            "expected": expected,
            "v2_predicted": LABELS[a[0]],
            "v2_confidence": a[1],
            "v2_correct": LABELS[a[0]] == expected,
            "v3_predicted": LABELS[b[0]],
            "v3_confidence": b[1],
            "v3_correct": LABELS[b[0]] == expected,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT / filename, index=False)

    return out

ctx = evaluate_suite(CONTEXTUAL, "contextual_details.csv")
target = evaluate_suite(TARGETED, "targeted_critical_details.csv")

v2_ctx = ctx.v2_correct.mean()
v3_ctx = ctx.v3_correct.mean()
v2_tgt = target.v2_correct.mean()
v3_tgt = target.v3_correct.mean()

print("--- CONTEXTUAL ---")
print(f"V2 accuracy : {v2_ctx*100:.2f}%")
print(f"V3 accuracy : {v3_ctx*100:.2f}%")
print(f"Delta       : {(v3_ctx-v2_ctx)*100:+.2f} pp")
print()

print("--- TARGETED CRITICAL ---")
print(f"V2 accuracy : {v2_tgt*100:.2f}%")
print(f"V3 accuracy : {v3_tgt*100:.2f}%")
print(f"Delta       : {(v3_tgt-v2_tgt)*100:+.2f} pp")
print()

# ---------------------------------------------------------------------
# OOD safety suite.
#
# The 11-class model has no trained defaultFallbackIntent output. Therefore
# OOD is measured as rejection by confidence threshold.
#
# We report several thresholds so we can see the tradeoff rather than
# hard-code one number as "production truth".
# ---------------------------------------------------------------------
OOD = [
    "who is the prime minister of India",
    "who is the priminister of india",
    "what is the weather today",
    "i want to order pizza",
    "tell me a joke",
    "what is the capital of France",
    "open my browser",
    "play music",
    "book a hotel for tomorrow",
    "how do i cook pasta",
    "what is bitcoin",
    "what time is the train",
    "how are you",
    "thank you",
    "good morning",
    "jkakjhdjkhd",
    "sdkjadsjj",
    "asdfghjkl",
    "123456789",
    "hello xyz abc",
]

ood_rows = []
for text in OOD:
    a = v2.predict(text)
    b = v3.predict(text)
    ood_rows.append({
        "text": text,
        "v2_predicted": LABELS[a[0]],
        "v2_confidence": a[1],
        "v2_margin": a[1]-a[3],
        "v3_predicted": LABELS[b[0]],
        "v3_confidence": b[1],
        "v3_margin": b[1]-b[3],
    })

ood = pd.DataFrame(ood_rows)
ood.to_csv(OUT / "ood_details.csv", index=False)

print("--- OOD REJECTION ---")
for threshold in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97]:
    v2_rej = (ood.v2_confidence < threshold).mean()
    v3_rej = (ood.v3_confidence < threshold).mean()
    print(
        f"threshold {threshold:.2f} | "
        f"V2 reject {v2_rej*100:6.2f}% | "
        f"V3 reject {v3_rej*100:6.2f}%"
    )
print()

# ---------------------------------------------------------------------
# Critical regressions.
# ---------------------------------------------------------------------
reg = target.copy()
reg["changed"] = reg.v2_predicted != reg.v3_predicted
reg["v2_wrong_v3_correct"] = (~reg.v2_correct) & reg.v3_correct
reg["v2_correct_v3_wrong"] = reg.v2_correct & (~reg.v3_correct)
reg.to_csv(OUT / "critical_regression_details.csv", index=False)

# These are hard safety expectations for this project.
critical_expected = {
    "the audio is quiet but don't mute it make it louder":
        "device.volume.increase",
    "the audio is loud but keep it on just lower it":
        "device.volume.decrease",
    "turn off":
        "device.volume.mute",
    "i can still hear it make it completely silent":
        "device.volume.mute",
    "i need to go to airport tommorow":
        "reminders.task.create",
}

critical_pass = True
for text, expected in critical_expected.items():
    row = target[target.text == text]
    ok = len(row) == 1 and row.iloc[0].v3_predicted == expected
    critical_pass = critical_pass and ok

# ---------------------------------------------------------------------
# Final decision.
#
# For the final release gate we require:
#   - exported V3 ONNX >= V2 on unseen
#   - exported V3 ONNX >= V2 on contextual
#   - targeted critical = 100%
#   - all critical examples correct
#
# OOD is reported for review; threshold must ultimately be calibrated with
# representative real-world OOD data before an unconditional production GO.
# ---------------------------------------------------------------------
unseen_pass = v3_unseen_acc >= v2_unseen_acc
contextual_pass = v3_ctx >= v2_ctx
targeted_pass = v3_tgt >= 1.0

summary = pd.DataFrame([
    {
        "Model": "V2 INT8 (locked)",
        "Size_MB": v2.size_mb(),
        "Unseen": v2_unseen_acc*100,
        "Unseen_Macro_F1": v2_unseen_f1*100,
        "Contextual": v2_ctx*100,
        "Targeted": v2_tgt*100,
    },
    {
        "Model": "V3 FP32 ONNX (candidate)",
        "Size_MB": v3.size_mb(),
        "Unseen": v3_unseen_acc*100,
        "Unseen_Macro_F1": v3_unseen_f1*100,
        "Contextual": v3_ctx*100,
        "Targeted": v3_tgt*100,
    },
])

summary.to_csv(OUT / "final_summary.csv", index=False)

print("=" * 78)
print("FINAL PRODUCTION GATE")
print("=" * 78)
print(summary.to_string(index=False))
print()
print("Unseen     :", "PASS" if unseen_pass else "FAIL")
print("Contextual :", "PASS" if contextual_pass else "FAIL")
print("Targeted   :", "PASS" if targeted_pass else "FAIL")
print("Critical   :", "PASS" if critical_pass else "FAIL")

# We intentionally do NOT call this "unconditional production ready" solely
# from synthetic OOD. Real-world OOD is still required for a robust release.
final_candidate = (
    unseen_pass and
    contextual_pass and
    targeted_pass and
    critical_pass
)

print()
if final_candidate:
    print("STATUS: V3 FP32 ONNX PASSES MODEL QUALITY GATE")
    print("STATUS: READY FOR PRODUCTION INTEGRATION / STAGED RELEASE")
    print()
    print("Important: before a broad release, collect real microphone/user")
    print("OOD examples and calibrate the confidence rejection threshold.")
else:
    print("STATUS: DO NOT PROMOTE V3")
    print("Keep V2 INT8 as the locked rollback baseline.")

print()
print("Saved to:", OUT)
print("V2 INT8 was NOT modified.")
print("V3 FP32 ONNX was NOT modified.")
print("No training occurred.")
print("The 595-row unseen test was evaluation-only.")
