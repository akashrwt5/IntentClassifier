#!/usr/bin/env python3
"""
V3 FP32 ONNX — SAFETY GATE THRESHOLD SWEEP

Purpose:
  Find a practical confidence threshold instead of blindly using 0.97.

Uses:
  1) In-domain calibration CSV from production_calibration_v2
  2) OOD calibration CSV from production_calibration_v2
  3) Critical/representative in-domain commands
  4) Actual V3 FP32 ONNX model for representative checks

No training.
No quantization.
No model modification.

The script reports, for each threshold:
  - in-domain coverage
  - accepted in-domain accuracy
  - valid-command false rejects
  - OOD rejection
  - OOD false accepts
  - combined safety score

IMPORTANT:
The calibration CSVs are only as good as their data. A threshold selected
from synthetic OOD is a calibration candidate, not proof of production safety.
"""

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import onnxruntime as ort

ROOT = Path("/Users/shuklam/IntentClassifier/semantic_project")

MODEL = ROOT / "tiny_semantic_student_v3_fp32" / "v3_semantic_student_fp32.onnx"
VOCAB_FILE = ROOT / "tiny_semantic_student_v2_balanced" / "vocab.json"

CAL_DIR = ROOT / "production_calibration_v2"
INDOMAIN = CAL_DIR / "production_indomain_calibration.csv"
OOD = CAL_DIR / "production_ood_calibration.csv"

OUT = ROOT / "v3_threshold_calibration"
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

MAX_LEN = 24

# Representative production commands.
CRITICAL = [
    ("make it louder", "device.volume.increase"),
    ("it's quieter can you make it a little louder", "device.volume.increase"),
    ("the audio is quiet but don't mute it make it louder", "device.volume.increase"),
    ("make it quieter", "device.volume.decrease"),
    ("the audio is loud but keep it on just lower it", "device.volume.decrease"),
    ("mute it", "device.volume.mute"),
    ("i can still hear it make it completely silent", "device.volume.mute"),
    ("turn off", "device.volume.mute"),
    ("unmute it", "device.volume.unmute"),
    ("turn the sound back on", "device.volume.unmute"),
    ("i need to go to airport tomorrow", "reminders.task.create"),
    ("i need to go to airport tommorow", "reminders.task.create"),
    ("where can i find my phone", "find.phone.locate"),
    ("please show the reminders i have", "help.reminder.show"),
    ("mark that reminder as completed", "reminders.task.complete"),
    ("please start the streaming session", "streaming.session.start"),
    ("please stop the streaming session", "streaming.session.stop"),
]

def load_vocab():
    raw = json.loads(VOCAB_FILE.read_text(encoding="utf-8"))
    if isinstance(raw.get("model"), dict) and isinstance(raw["model"].get("vocab"), dict):
        v = raw["model"]["vocab"]
    elif isinstance(raw.get("vocab"), dict):
        v = raw["vocab"]
    else:
        v = raw
    v = {str(k): int(x) for k, x in v.items()}
    return (
        v,
        int(v.get("<pad>", v.get("[PAD]", 0))),
        int(v.get("<unk>", v.get("[UNK]", 1))),
        v.get("<cls>", v.get("[CLS]", None)),
        v.get("<sep>", v.get("[SEP]", None)),
    )

VOCAB, PAD, UNK, CLS, SEP = load_vocab()

def normalize(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())

def tokenize(text):
    text = normalize(text)
    text = re.sub(r"([.!?,;:()'])", r" \1 ", text)
    ids = []

    if CLS is not None:
        ids.append(int(CLS))

    for word in text.split():
        if word in VOCAB:
            ids.append(int(VOCAB[word]))
            continue

        pos = 0
        matched = False

        while pos < len(word):
            best_id = None
            best_len = 0

            for end in range(len(word), pos, -1):
                piece = word[pos:end]
                for candidate in (piece, "##" + piece):
                    if candidate in VOCAB:
                        best_id = int(VOCAB[candidate])
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

def model_scores(session, input_name, output_name, text):
    x = np.asarray([tokenize(text)], dtype=np.int64)
    logits = session.run([output_name], {input_name: x})[0][0]
    p = softmax(logits)
    order = np.argsort(p)[::-1]
    conf = float(p[order[0]])
    margin = float(p[order[0]] - p[order[1]])
    entropy = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
    return conf, margin, entropy, LABELS[int(order[0])]

def find_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

if not MODEL.exists():
    raise FileNotFoundError(MODEL)
if not INDOMAIN.exists():
    raise FileNotFoundError(INDOMAIN)
if not OOD.exists():
    raise FileNotFoundError(OOD)

ind = pd.read_csv(INDOMAIN)
ood = pd.read_csv(OOD)

# Calibration builder may use different names. Detect common forms.
text_ind = find_col(ind, ["text", "utterance", "sentence"])
label_ind = find_col(ind, ["intent", "expected", "label", "true_intent"])
conf_ind = find_col(ind, ["confidence", "max_probability", "probability", "max_prob"])
correct_ind = find_col(ind, ["correct", "is_correct"])

text_ood = find_col(ood, ["text", "utterance", "sentence"])
conf_ood = find_col(ood, ["confidence", "max_probability", "probability", "max_prob"])

if text_ind is None or label_ind is None:
    raise ValueError(
        f"Could not identify text/intent columns in {INDOMAIN}. "
        f"Columns: {list(ind.columns)}"
    )

if text_ood is None:
    raise ValueError(
        f"Could not identify text column in {OOD}. "
        f"Columns: {list(ood.columns)}"
    )

# If confidence is already present, use it. Otherwise calculate it with ONNX.
session = ort.InferenceSession(
    str(MODEL),
    providers=["CPUExecutionProvider"],
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

if conf_ind is None:
    vals = []
    for text in ind[text_ind]:
        vals.append(model_scores(session, input_name, output_name, text)[0])
    ind["_confidence_v3"] = vals
    conf_ind = "_confidence_v3"

if conf_ood is None:
    vals = []
    for text in ood[text_ood]:
        vals.append(model_scores(session, input_name, output_name, text)[0])
    ood["_confidence_v3"] = vals
    conf_ood = "_confidence_v3"

# Correctness: use existing correctness if present, otherwise infer from
# prediction if calibration contains predicted intent.
if correct_ind is not None:
    ind["_correct"] = ind[correct_ind].astype(bool)
else:
    pred_col = find_col(ind, ["predicted", "prediction", "predicted_intent"])
    if pred_col is not None:
        ind["_correct"] = (
            ind[pred_col].astype(str).str.strip()
            == ind[label_ind].astype(str).str.strip()
        )
    else:
        # Safest fallback: evaluate the actual V3 ONNX.
        vals = []
        for text, expected in zip(ind[text_ind], ind[label_ind]):
            pred = model_scores(
                session, input_name, output_name, text
            )[3]
            vals.append(pred == str(expected))
        ind["_correct"] = vals

ind["_confidence"] = pd.to_numeric(ind[conf_ind], errors="coerce")
ood["_confidence"] = pd.to_numeric(ood[conf_ood], errors="coerce")

ind = ind.dropna(subset=["_confidence"]).reset_index(drop=True)
ood = ood.dropna(subset=["_confidence"]).reset_index(drop=True)

# Threshold sweep.
thresholds = [round(x, 3) for x in np.arange(0.80, 0.991, 0.005)]

rows = []

for t in thresholds:
    accepted = ind["_confidence"] >= t
    ood_rejected = ood["_confidence"] < t

    coverage = float(accepted.mean()) if len(ind) else 0.0

    if accepted.any():
        accepted_accuracy = float(ind.loc[accepted, "_correct"].mean())
    else:
        accepted_accuracy = 0.0

    false_reject = float((~accepted).mean()) if len(ind) else 0.0
    ood_rejection = float(ood_rejected.mean()) if len(ood) else 0.0
    ood_false_accept = float((~ood_rejected).mean()) if len(ood) else 0.0

    # Conservative combined score:
    # prioritize valid accepted correctness and OOD rejection.
    safety_score = (
        accepted_accuracy * coverage * ood_rejection
    )

    rows.append({
        "threshold": t,
        "indomain_coverage": coverage,
        "indomain_accepted_accuracy": accepted_accuracy,
        "indomain_false_reject_rate": false_reject,
        "ood_rejection_rate": ood_rejection,
        "ood_false_accept_rate": ood_false_accept,
        "safety_score": safety_score,
    })

sweep = pd.DataFrame(rows)

# Representative critical suite using actual V3 ONNX.
critical_rows = []
for text, expected in CRITICAL:
    conf, margin, ent, pred = model_scores(
        session, input_name, output_name, text
    )
    critical_rows.append({
        "text": text,
        "expected": expected,
        "predicted": pred,
        "confidence": conf,
        "margin": margin,
        "entropy": ent,
        "correct": pred == expected,
    })

critical_df = pd.DataFrame(critical_rows)
critical_df.to_csv(OUT / "critical_scores.csv", index=False)

# Candidate selection:
# First require ALL critical commands to be accepted and correct.
# Among those thresholds, prefer high OOD rejection, then low false reject.
eligible = sweep.copy()

min_critical_conf = float(critical_df.confidence.min())
eligible = eligible[eligible.threshold <= min_critical_conf].copy()

if len(eligible):
    best = eligible.sort_values(
        ["ood_rejection_rate", "indomain_coverage", "safety_score"],
        ascending=[False, False, False],
    ).iloc[0]
else:
    best = sweep.sort_values(
        ["safety_score", "ood_rejection_rate"],
        ascending=[False, False],
    ).iloc[0]

# Show the useful neighborhood.
best_t = float(best.threshold)
neighborhood = sweep[
    (sweep.threshold >= best_t - 0.03)
    & (sweep.threshold <= best_t + 0.03)
].copy()

sweep.to_csv(OUT / "threshold_sweep.csv", index=False)
neighborhood.to_csv(OUT / "threshold_neighborhood.csv", index=False)

manifest = {
    "model": str(MODEL),
    "indomain_calibration": str(INDOMAIN),
    "ood_calibration": str(OOD),
    "indomain_rows": int(len(ind)),
    "ood_rows": int(len(ood)),
    "critical_rows": int(len(critical_df)),
    "selected_candidate_threshold": best_t,
    "selection_rule": (
        "threshold <= minimum critical confidence, then maximize OOD "
        "rejection, then in-domain coverage"
    ),
    "warning": (
        "Synthetic/derived OOD calibration is not sufficient by itself "
        "for broad production release."
    ),
}
(OUT / "calibration_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print("=" * 78)
print("V3 FP32 ONNX — SAFETY GATE THRESHOLD SWEEP")
print("=" * 78)
print(f"Model: {MODEL}")
print(f"In-domain calibration rows: {len(ind)}")
print(f"OOD calibration rows      : {len(ood)}")
print(f"Critical rows              : {len(critical_df)}")
print()

print("CRITICAL COMMANDS")
print("-" * 78)
for _, r in critical_df.iterrows():
    print(
        f"{'PASS' if r.correct else 'FAIL'} | "
        f"{r.confidence*100:6.2f}% | "
        f"{r.predicted:34s} | {r.text}"
    )

print()
print("=" * 78)
print("BEST CANDIDATE THRESHOLD")
print("=" * 78)
print(f"Candidate threshold : {best_t:.3f}")
print(f"Min critical conf   : {min_critical_conf:.4f}")
print(f"In-domain coverage  : {best.indomain_coverage*100:.2f}%")
print(f"Accepted accuracy   : {best.indomain_accepted_accuracy*100:.2f}%")
print(f"False reject rate   : {best.indomain_false_reject_rate*100:.2f}%")
print(f"OOD rejection       : {best.ood_rejection_rate*100:.2f}%")
print(f"OOD false accepts    : {best.ood_false_accept_rate*100:.2f}%")
print(f"Safety score        : {best.safety_score:.6f}")

print()
print("THRESHOLD NEIGHBORHOOD")
print("-" * 78)
print(
    neighborhood[
        [
            "threshold",
            "indomain_coverage",
            "indomain_accepted_accuracy",
            "indomain_false_reject_rate",
            "ood_rejection_rate",
            "ood_false_accept_rate",
        ]
    ].to_string(index=False)
)

print()
print("=" * 78)
print("RECOMMENDATION")
print("=" * 78)
print(
    f"Use {best_t:.3f} as the NEXT CALIBRATION CANDIDATE for Python testing."
)
print(
    "Do not call this the final production threshold until real-world "
    "microphone/user OOD data is included."
)
print()
print(f"Saved: {OUT}")
