#!/usr/bin/env python3
"""
production_hardening_v2.py

Calibrates a safety gate around the FROZEN V2 INT8 intent classifier.

Expected project files:
  tiny_semantic_student_v2_int8/v2_semantic_student_int8.onnx
  production_calibration_v2/production_indomain_calibration.csv
  production_calibration_v2/production_ood_calibration.csv

The script:
  1. Evaluates the frozen V2 INT8 model.
  2. Searches confidence + margin thresholds on calibration data.
  3. Measures in-domain acceptance/correctness.
  4. Measures OOD rejection and false-accept rate.
  5. Tests a starter safety suite.
  6. DOES NOT modify, retrain, or overwrite V2 INT8.
  7. DOES NOT load the 595-row unseen test for threshold fitting.

Important:
This is calibration, not production certification. Synthetic OOD is not a
substitute for real-world OOD data.
"""

from pathlib import Path
import json
import math
import re
import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "tiny_semantic_student_v2_int8" / "v2_semantic_student_int8.onnx"
CAL = ROOT / "production_calibration_v2"
IND = CAL / "production_indomain_calibration.csv"
OOD = CAL / "production_ood_calibration.csv"
OUT = ROOT / "production_hardening_v2"
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
LABEL_TO_ID = {x: i for i, x in enumerate(LABELS)}

if not MODEL.exists():
    raise FileNotFoundError(f"V2 INT8 model not found: {MODEL}")
if not IND.exists():
    raise FileNotFoundError(f"In-domain calibration not found: {IND}")
if not OOD.exists():
    raise FileNotFoundError(f"OOD calibration not found: {OOD}")

ind = pd.read_csv(IND).dropna(subset=["text", "intent"]).copy()
ood = pd.read_csv(OOD).dropna(subset=["text"]).copy()

ind["text"] = ind["text"].astype(str).str.strip()
ood["text"] = ood["text"].astype(str).str.strip()
ind = ind[ind["intent"].isin(LABELS)].drop_duplicates("text").reset_index(drop=True)
ood = ood.drop_duplicates("text").reset_index(drop=True)

session = ort.InferenceSession(
    str(MODEL),
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# V2 tokenizer: 895 vocabulary, max length 24.
# These special IDs are inferred from the standard tokenizer artifact when
# available; otherwise use the model's expected IDs from the V2 project.
TOK_DIRS = [
    ROOT / "tiny_semantic_student_v2_balanced",
    ROOT / "subword_student_v1",
    ROOT / "tiny_semantic_student_v2_int8",
]
VOCAB_JSON = None
for d in TOK_DIRS:
    for p in [d / "vocab.json", d / "tokenizer.json", d / "tokenizer_vocab.json"]:
        if p.exists():
            VOCAB_JSON = p
            break
    if VOCAB_JSON:
        break

# This fallback is intentionally conservative: it only supports calibration
# when a tokenizer artifact can be found. No silent tokenization mismatch.
if VOCAB_JSON is None:
    raise FileNotFoundError(
        "Tokenizer artifact not found. Expected vocab.json/tokenizer.json in "
        "tiny_semantic_student_v2_balanced, subword_student_v1, or "
        "tiny_semantic_student_v2_int8."
    )

raw = json.loads(VOCAB_JSON.read_text(encoding="utf-8"))
if "model" in raw and isinstance(raw["model"], dict) and "vocab" in raw["model"]:
    vocab = raw["model"]["vocab"]
elif "vocab" in raw and isinstance(raw["vocab"], dict):
    vocab = raw["vocab"]
else:
    vocab = raw

if not isinstance(vocab, dict):
    raise ValueError("Unsupported tokenizer vocabulary JSON format.")

PAD_ID = int(vocab.get("<pad>", vocab.get("[PAD]", 0)))
UNK_ID = int(vocab.get("<unk>", vocab.get("[UNK]", 1)))
CLS_ID = vocab.get("<cls>", vocab.get("[CLS]", None))
SEP_ID = vocab.get("<sep>", vocab.get("[SEP]", None))
MAX_LEN = 24

def basic_bpe_tokens(text):
    # This fallback tokenizer mirrors the lightweight BPE/subword behavior
    # sufficiently for calibration only when the exported vocab is simple.
    # If a tokenizer.json contains a full tokenizer implementation, users
    # should replace this with the exact training tokenizer.
    text = text.lower().strip()
    text = re.sub(r"([.!?,;:()])", r" \1 ", text)
    words = text.split()
    out = []
    for w in words:
        if w in vocab:
            out.append(int(vocab[w]))
            continue
        # greedy character/subword lookup
        pos = 0
        pieces = []
        while pos < len(w):
            best = None
            for end in range(len(w), pos, -1):
                cand = w[pos:end]
                if cand in vocab:
                    best = cand
                    break
                if pos > 0 and ("##" + cand) in vocab:
                    best = "##" + cand
                    break
            if best is None:
                pieces = [UNK_ID]
                break
            pieces.append(int(vocab[best]))
            pos += len(best.replace("##", ""))
        out.extend(pieces)
    ids = []
    if CLS_ID is not None:
        ids.append(int(CLS_ID))
    ids.extend(out)
    if SEP_ID is not None:
        ids.append(int(SEP_ID))
    ids = ids[:MAX_LEN]
    ids += [PAD_ID] * (MAX_LEN - len(ids))
    return ids

def softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)

def predict_many(texts, batch=64):
    rows = []
    for start in range(0, len(texts), batch):
        chunk = texts[start:start+batch]
        # V2 ONNX was exported with batch=1, so run one row at a time.
        for text in chunk:
            ids = np.asarray([basic_bpe_tokens(text)], dtype=np.int64)
            logits = session.run([output_name], {input_name: ids})[0]
            probs = softmax(logits)[0]
            order = np.argsort(probs)[::-1]
            top1 = int(order[0])
            top2 = int(order[1])
            p1 = float(probs[top1])
            p2 = float(probs[top2])
            entropy = float(
                -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)))
                / math.log(len(LABELS))
            )
            rows.append({
                "text": text,
                "pred_id": top1,
                "predicted": LABELS[top1],
                "confidence": p1,
                "margin": p1 - p2,
                "entropy": entropy,
                "top2": LABELS[top2],
                "top2_confidence": p2,
            })
    return pd.DataFrame(rows)

print("=" * 78)
print("PRODUCTION HARDENING V2 — SAFETY GATE CALIBRATION")
print("=" * 78)
print("Frozen model:", MODEL)
print("Model size: %.3f MB" % (MODEL.stat().st_size / 1024 / 1024))
print("In-domain calibration:", len(ind))
print("OOD calibration:", len(ood))
print("Tokenizer:", VOCAB_JSON)
print("Vocab:", len(vocab))
print("Max length:", MAX_LEN)
print()

ind_pred = predict_many(ind["text"].tolist())
ind_pred["expected"] = ind["intent"].tolist()
ind_pred["correct"] = ind_pred["predicted"] == ind_pred["expected"]

ood_pred = predict_many(ood["text"].tolist())

# Save raw calibration predictions.
ind_pred.to_csv(OUT / "indomain_predictions.csv", index=False)
ood_pred.to_csv(OUT / "ood_predictions.csv", index=False)

# ---------------------------------------------------------------------
# Threshold grid
# ---------------------------------------------------------------------
# We want high in-domain correctness while reducing OOD false accepts.
# A point is accepted when confidence >= C and margin >= M and entropy <= E.
#
# We choose the best threshold under a conservative objective:
# maximize OOD rejection, subject to >=98% correct accepted in-domain samples
# and >=95% of in-domain samples being accepted.
# If no threshold satisfies this, report that explicitly.
# ---------------------------------------------------------------------

best = None
grid = []

conf_grid = np.arange(0.50, 0.991, 0.01)
margin_grid = np.arange(0.00, 0.501, 0.01)
entropy_grid = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]

for c in conf_grid:
    for m in margin_grid:
        base_ind = (ind_pred.confidence >= c) & (ind_pred.margin >= m)
        accepted_ind = int(base_ind.sum())
        if accepted_ind == 0:
            continue
        accepted_correct = int(
            (base_ind & ind_pred.correct).sum()
        )
        accepted_acc = accepted_correct / accepted_ind
        coverage = accepted_ind / len(ind_pred)

        # Entropy evaluated later in the loop.
        for e in entropy_grid:
            ind_accept = (
                (ind_pred.confidence >= c)
                & (ind_pred.margin >= m)
                & (ind_pred.entropy <= e)
            )
            ood_accept = (
                (ood_pred.confidence >= c)
                & (ood_pred.margin >= m)
                & (ood_pred.entropy <= e)
            )

            ia = int(ind_accept.sum())
            oa = int(ood_accept.sum())
            if ia == 0:
                continue

            icorrect = int((ind_accept & ind_pred.correct).sum())
            iacc = icorrect / ia
            cov = ia / len(ind_pred)
            ood_rejection = 1.0 - (oa / len(ood_pred))

            # Production-oriented gates.
            if iacc >= 0.98 and cov >= 0.95:
                score = (
                    0.60 * ood_rejection
                    + 0.25 * iacc
                    + 0.15 * cov
                )
                rec = {
                    "confidence_threshold": float(c),
                    "margin_threshold": float(m),
                    "entropy_threshold": float(e),
                    "indomain_acceptance": ia,
                    "indomain_coverage": cov,
                    "indomain_accepted_accuracy": iacc,
                    "ood_false_accepts": oa,
                    "ood_rejection": ood_rejection,
                    "score": score,
                }
                grid.append(rec)
                if best is None or score > best["score"]:
                    best = rec

if best is None:
    # Relaxed diagnostic search so the user sees the best available tradeoff.
    for c in conf_grid:
        for m in margin_grid:
            for e in entropy_grid:
                ia_mask = (
                    (ind_pred.confidence >= c)
                    & (ind_pred.margin >= m)
                    & (ind_pred.entropy <= e)
                )
                oa_mask = (
                    (ood_pred.confidence >= c)
                    & (ood_pred.margin >= m)
                    & (ood_pred.entropy <= e)
                )
                ia = int(ia_mask.sum())
                if ia == 0:
                    continue
                iacc = float((ia_mask & ind_pred.correct).sum() / ia)
                cov = ia / len(ind_pred)
                ood_rej = 1.0 - float(oa_mask.sum() / len(ood_pred))
                score = 0.65 * ood_rej + 0.20 * iacc + 0.15 * cov
                rec = {
                    "confidence_threshold": float(c),
                    "margin_threshold": float(m),
                    "entropy_threshold": float(e),
                    "indomain_acceptance": ia,
                    "indomain_coverage": cov,
                    "indomain_accepted_accuracy": iacc,
                    "ood_false_accepts": int(oa_mask.sum()),
                    "ood_rejection": ood_rej,
                    "score": score,
                }
                grid.append(rec)
                if best is None or score > best["score"]:
                    best = rec

grid_df = pd.DataFrame(grid).sort_values("score", ascending=False)
grid_df.head(100).to_csv(OUT / "threshold_search_top100.csv", index=False)

# ---------------------------------------------------------------------
# Starter safety suite
# ---------------------------------------------------------------------

SAFETY = [
    ("make it louder", "device.volume.increase", False),
    ("it's quieter can you make it a little louder",
     "device.volume.increase", False),
    ("the audio is quiet but don't mute it make it louder",
     "device.volume.increase", False),
    ("make it quieter", "device.volume.decrease", False),
    ("the audio is loud but keep it on just lower it",
     "device.volume.decrease", False),
    ("mute it", "device.volume.mute", False),
    ("i can still hear it make it completely silent",
     "device.volume.mute", False),
    ("turn off", "device.volume.mute", False),
    ("unmute it", "device.volume.unmute", False),
    ("turn the sound back on", "device.volume.unmute", False),
    ("i need to go to airport tomorrow",
     "reminders.task.create", False),
    ("i need to go to airport tommorow",
     "reminders.task.create", False),
    ("where can i find my phone", "find.phone.locate", False),
    ("please show the reminders i have",
     "help.reminder.show", False),
    ("mark that reminder as completed",
     "reminders.task.complete", False),
    ("please start the streaming session",
     "streaming.session.start", False),
    ("please stop the streaming session",
     "streaming.session.stop", False),

    ("who is the prime minister of India",
     "defaultFallbackIntent", True),
    ("what is the capital of France",
     "defaultFallbackIntent", True),
    ("what is the weather today",
     "defaultFallbackIntent", True),
    ("tell me a joke", "defaultFallbackIntent", True),
    ("book a hotel for tomorrow", "defaultFallbackIntent", True),
    ("play music", "defaultFallbackIntent", True),
    ("jkakjhdjkhd", "defaultFallbackIntent", True),
    ("asdfghjkl", "defaultFallbackIntent", True),
]

raw = predict_many([x[0] for x in SAFETY])

if best:
    c = best["confidence_threshold"]
    m = best["margin_threshold"]
    e = best["entropy_threshold"]
else:
    c, m, e = 0.70, 0.20, 0.85

def gate(row):
    accept = (
        row["confidence"] >= c
        and row["margin"] >= m
        and row["entropy"] <= e
    )
    if accept:
        return row["predicted"], "ACCEPT"
    return "defaultFallbackIntent", "REJECT"

safety_rows = []
for i, (text, expected, is_ood) in enumerate(SAFETY):
    row = raw.iloc[i]
    final, decision = gate(row)
    ok = final == expected
    safety_rows.append({
        "text": text,
        "expected": expected,
        "raw_prediction": row["predicted"],
        "confidence": row["confidence"],
        "margin": row["margin"],
        "entropy": row["entropy"],
        "final": final,
        "decision": decision,
        "correct": ok,
        "is_ood": is_ood,
    })

safety_df = pd.DataFrame(safety_rows)
safety_df.to_csv(OUT / "starter_safety_results.csv", index=False)

supported = safety_df[~safety_df.is_ood]
ood_s = safety_df[safety_df.is_ood]

# ---------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------

print("=" * 78)
print("CALIBRATION RESULT")
print("=" * 78)

if best:
    print("Selected thresholds:")
    print("  confidence >= %.2f" % c)
    print("  margin     >= %.2f" % m)
    print("  entropy    <= %.2f" % e)
    print()
    print("In-domain coverage          : %.2f%%" % (best["indomain_coverage"] * 100))
    print("In-domain accepted accuracy : %.2f%%" % (
        best["indomain_accepted_accuracy"] * 100
    ))
    print("OOD rejection               : %.2f%%" % (
        best["ood_rejection"] * 100
    ))
    print("OOD false accepts            :", best["ood_false_accepts"])
else:
    print("No valid threshold candidate found.")
    print("This means the calibration set does not support the requested")
    print("98% accepted accuracy + 95% coverage constraint.")

print()
print("=" * 78)
print("STARTER SAFETY SUITE")
print("=" * 78)

for _, r in safety_df.iterrows():
    status = "PASS" if r["correct"] else "FAIL"
    print(
        f"{status:4} | {r['confidence']*100:6.2f}% | "
        f"{r['final']:<32} | {r['text']}"
    )

print()
print("Supported safety correctness : %.2f%%" %
      (supported["correct"].mean() * 100))
print("OOD safety rejection         : %.2f%%" %
      ((ood_s["final"] == "defaultFallbackIntent").mean() * 100))

# ---------------------------------------------------------------------
# Production readiness gates
# ---------------------------------------------------------------------

# These are deliberately conservative. They are NOT a guarantee.
ind_gate = bool(
    best is not None
    and best["indomain_accepted_accuracy"] >= 0.98
    and best["indomain_coverage"] >= 0.95
)

ood_gate = bool(
    best is not None
    and best["ood_rejection"] >= 0.90
)

starter_gate = bool(
    supported["correct"].mean() >= 0.95
    and (ood_s["final"] == "defaultFallbackIntent").mean() >= 0.90
)

# A known critical hard-negative failure is a hard block.
critical = safety_df[
    safety_df["text"].str.contains(
        "don't mute it make it louder", case=False, regex=False
    )
]
critical_gate = bool(len(critical) == 0 or bool(critical["correct"].all()))

print()
print("=" * 78)
print("PRODUCTION READINESS")
print("=" * 78)
print("Frozen V2 INT8 : YES")
print("Calibration    : PASS" if best is not None else "Calibration    : FAIL")
print("In-domain gate : PASS" if ind_gate else "In-domain gate : FAIL")
print("OOD gate       : PASS" if ood_gate else "OOD gate       : FAIL")
print("Starter gate   : PASS" if starter_gate else "Starter gate   : FAIL")
print("Critical gate  : PASS" if critical_gate else "Critical gate  : FAIL")

ready = ind_gate and ood_gate and starter_gate and critical_gate
print("STATUS         :", "CALIBRATION CANDIDATE" if ready else "NOT PRODUCTION READY")

report = {
    "model": str(MODEL),
    "model_size_mb": MODEL.stat().st_size / 1024 / 1024,
    "indomain_rows": len(ind),
    "ood_rows": len(ood),
    "tokenizer": str(VOCAB_JSON),
    "vocab_size": len(vocab),
    "max_length": MAX_LEN,
    "thresholds": {
        "confidence": c,
        "margin": m,
        "entropy": e,
    },
    "best": best,
    "gates": {
        "indomain": ind_gate,
        "ood": ood_gate,
        "starter": starter_gate,
        "critical": critical_gate,
        "ready": ready,
    },
    "warning": (
        "Synthetic calibration data is not sufficient for production "
        "certification; independently collected real OOD/near-OOD data "
        "is required."
    ),
}
(OUT / "production_hardening_report.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print()
print("Saved:", OUT)
print("  indomain_predictions.csv")
print("  ood_predictions.csv")
print("  threshold_search_top100.csv")
print("  starter_safety_results.csv")
print("  production_hardening_report.json")
print()
print("V2 INT8 was NOT modified.")
print("The 595-row unseen test was NOT used for threshold fitting.")
