#!/usr/bin/env python3
"""
Production Hardening V1

Keeps the frozen V2 INT8 classifier unchanged and adds a lightweight
safety/OOD gate based on:
  - classifier confidence
  - Top1 vs Top2 margin
  - known-token ratio
  - normalized entropy

For real calibration provide:
  production_indomain_calibration.csv : text,intent
  production_ood_calibration.csv      : text

Starter sanity tests run if calibration files are absent, but the script
will explicitly report NOT PRODUCTION READY.
"""

from pathlib import Path
import json
import re
import math
import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "tiny_semantic_student_v2_int8" / "v2_semantic_student_int8.onnx"
VOCAB = ROOT / "tiny_semantic_student_v1" / "vocab.json"
LABELS = ROOT / "tiny_semantic_student_v1" / "intent_labels.txt"
CONFIG = ROOT / "tiny_semantic_student_v1" / "config.json"
OUT = ROOT / "production_hardening_v1"

INDOMAIN_CANDIDATES = [
    ROOT / "production_indomain_calibration.csv",
    ROOT / "indomain_calibration.csv",
    ROOT / "calibration_indomain.csv",
]
OOD_CANDIDATES = [
    ROOT / "production_ood_calibration.csv",
    ROOT / "ood_calibration.csv",
    ROOT / "calibration_ood.csv",
    ROOT / "production_ood.csv",
]

DEFAULT_THRESHOLD = {
    "confidence": 0.70,
    "margin": 0.10,
    "known_token_ratio": 0.20,
    "max_normalized_entropy": 0.95,
}

STARTER_SUPPORTED = [
    ("make it louder", "device.volume.increase"),
    ("it's quieter can you make it a little louder", "device.volume.increase"),
    ("the audio is quiet but don't mute it make it louder", "device.volume.increase"),
    ("make it quieter", "device.volume.decrease"),
    ("it's a little loud can you make it quieter", "device.volume.decrease"),
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

STARTER_OOD = [
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
    "what time is the train",
    "what is bitcoin",
    "how are you",
    "thank you",
    "good morning",
    "jkakjhdjkhd",
    "sdkjadsjj",
    "asdfghjkl",
    "123456789",
    "hello xyz abc",
]

for p in (MODEL, VOCAB, LABELS):
    if not p.exists():
        raise FileNotFoundError(f"Missing required artifact: {p}")

with open(VOCAB, "r", encoding="utf-8") as f:
    vocab = json.load(f)
with open(LABELS, "r", encoding="utf-8") as f:
    labels = [x.strip() for x in f if x.strip()]
config = {}
if CONFIG.exists():
    with open(CONFIG, "r", encoding="utf-8") as f:
        config = json.load(f)

MAX_LEN = int(config.get("max_len", config.get("max_length", 24)))
PAD = int(vocab.get("<PAD>", vocab.get("[PAD]", 0)))
UNK = int(vocab.get("<UNK>", vocab.get("[UNK]", 1)))


def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text):
    return clean_text(text).split()


def encode(text):
    ids = [int(vocab.get(t, UNK)) for t in tokens(text)][:MAX_LEN]
    return ids + [PAD] * (MAX_LEN - len(ids))


def known_ratio(text):
    ts = tokens(text)
    return 0.0 if not ts else sum(t in vocab for t in ts) / len(ts)


class FrozenV2:
    def __init__(self, path):
        self.path = path
        self.session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.size_mb = path.stat().st_size / 1024 / 1024

    def predict_one(self, text):
        x = np.asarray([encode(text)], dtype=np.int64)
        logits = self.session.run(
            [self.output_name], {self.input_name: x}
        )[0][0].astype(np.float64)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        order = np.argsort(probs)[::-1]
        i1, i2 = int(order[0]), int(order[1])
        p1, p2 = float(probs[i1]), float(probs[i2])
        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
        norm_entropy = entropy / math.log(max(len(labels), 2))
        return {
            "intent": labels[i1],
            "confidence": p1,
            "top2_intent": labels[i2],
            "top2_confidence": p2,
            "margin": p1 - p2,
            "normalized_entropy": norm_entropy,
            "known_token_ratio": known_ratio(text),
        }


model = FrozenV2(MODEL)
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 76)
print("PRODUCTION HARDENING V1")
print("=" * 76)
print(f"Frozen V2 INT8 : {MODEL}")
print(f"Model size     : {model.size_mb:.3f} MB")
print(f"Vocabulary     : {len(vocab)}")
print(f"Intents        : {len(labels)}")
print(f"Max length     : {MAX_LEN}")


def find_file(candidates):
    return next((p for p in candidates if p.exists()), None)


def load_indomain():
    path = find_file(INDOMAIN_CANDIDATES)
    if path is None:
        return None, None
    df = pd.read_csv(path)
    if "text" not in df.columns:
        raise ValueError(f"{path} needs a text column")
    col = next((c for c in ["intent", "expected_intent", "true_intent", "label"]
                if c in df.columns), None)
    if col is None:
        raise ValueError(f"{path} needs an intent column")
    df = df[["text", col]].rename(columns={col: "intent"}).dropna().reset_index(drop=True)
    bad = sorted(set(df.intent) - set(labels))
    if bad:
        raise ValueError(f"Unknown intents in calibration: {bad}")
    return df, path


def load_ood():
    path = find_file(OOD_CANDIDATES)
    if path is None:
        return None, None
    df = pd.read_csv(path)
    if "text" not in df.columns:
        raise ValueError(f"{path} needs a text column")
    return df[["text"]].dropna().reset_index(drop=True), path


indomain, indomain_path = load_indomain()
ood, ood_path = load_ood()

print(f"In-domain calibration: {indomain_path or 'NOT FOUND'}")
print(f"OOD calibration       : {ood_path or 'NOT FOUND'}")


def score_frame(df):
    rows = []
    for text in df.text.tolist():
        r = model.predict_one(text)
        rows.append({"text": text, **r})
    return pd.DataFrame(rows)


def decision(row, th):
    ok = (
        row.confidence >= th["confidence"]
        and row.margin >= th["margin"]
        and row.known_token_ratio >= th["known_token_ratio"]
        and row.normalized_entropy <= th["max_normalized_entropy"]
    )
    return row.intent if ok else "defaultFallbackIntent"


def calibrate(indf, ooddf):
    ins = score_frame(indf)
    ods = score_frame(ooddf)
    rows = []
    for c in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95]:
        for m in [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
            for k in [0.00, 0.10, 0.20, 0.30, 0.40]:
                for e in [0.75, 0.85, 0.95, 1.00]:
                    th = {
                        "confidence": c,
                        "margin": m,
                        "known_token_ratio": k,
                        "max_normalized_entropy": e,
                    }
                    ia = np.mean([decision(r, th) != "defaultFallbackIntent"
                                  for _, r in ins.iterrows()])
                    orr = np.mean([decision(r, th) == "defaultFallbackIntent"
                                   for _, r in ods.iterrows()])
                    # Strongly penalize OOD false accepts.
                    score = ia - 2.0 * (1.0 - orr)
                    rows.append({**th, "in_acceptance": ia,
                                 "ood_rejection": orr, "score": score})
    grid = pd.DataFrame(rows).sort_values(
        ["score", "ood_rejection", "in_acceptance"],
        ascending=False
    ).reset_index(drop=True)
    grid.to_csv(OUT / "threshold_search.csv", index=False)
    return grid.iloc[0].to_dict(), ins, ods


calibrated = False
threshold = dict(DEFAULT_THRESHOLD)
in_acceptance = None
ood_rejection = None
ood_auc = None
ood_ap = None

if indomain is not None and ood is not None and len(indomain) >= 100 and len(ood) >= 100:
    print("\nCalibrating safety gate...")
    best, ins, ods = calibrate(indomain, ood)
    threshold = {
        "confidence": float(best["confidence"]),
        "margin": float(best["margin"]),
        "known_token_ratio": float(best["known_token_ratio"]),
        "max_normalized_entropy": float(best["max_normalized_entropy"]),
    }
    calibrated = True

    ins["expected"] = indomain.intent.tolist()
    ins["final_decision"] = [decision(r, threshold) for _, r in ins.iterrows()]
    ins["accepted"] = ins.final_decision != "defaultFallbackIntent"
    ins["correct"] = ins.final_decision == ins.expected

    ods["final_decision"] = [decision(r, threshold) for _, r in ods.iterrows()]
    ods["rejected"] = ods.final_decision == "defaultFallbackIntent"

    in_acceptance = float(ins.accepted.mean()) * 100
    ood_rejection = float(ods.rejected.mean()) * 100

    y = np.r_[np.zeros(len(ins)), np.ones(len(ods))]
    s = np.r_[1.0 - ins.confidence.to_numpy(),
              1.0 - ods.confidence.to_numpy()]
    try:
        ood_auc = float(roc_auc_score(y, s))
        ood_ap = float(average_precision_score(y, s))
    except Exception:
        pass

    pd.concat([
        ins.assign(dataset="IN_DOMAIN"),
        ods.assign(dataset="OOD"),
    ], ignore_index=True).to_csv(
        OUT / "calibration_results.csv", index=False
    )

    print("\nCalibrated thresholds:")
    print(json.dumps(threshold, indent=2))
    print(f"In-domain acceptance: {in_acceptance:.2f}%")
    print(f"OOD rejection       : {ood_rejection:.2f}%")
    if ood_auc is not None:
        print(f"OOD AUROC diagnostic : {ood_auc:.4f}")
    if ood_ap is not None:
        print(f"OOD AP diagnostic    : {ood_ap:.4f}")
else:
    print("\nWARNING: representative calibration files not found.")
    print("Starter tests are NOT sufficient for production.")


# Starter sanity
starter = []
for text, expected in STARTER_SUPPORTED:
    r = model.predict_one(text)
    final = decision(pd.Series(r), threshold)
    starter.append({
        "type": "SUPPORTED", "text": text, "expected": expected,
        **{k: r[k] for k in ["intent", "confidence", "top2_intent",
                              "top2_confidence", "margin",
                              "normalized_entropy", "known_token_ratio"]},
        "final_decision": final,
        "correct": final == expected,
    })

for text in STARTER_OOD:
    r = model.predict_one(text)
    final = decision(pd.Series(r), threshold)
    starter.append({
        "type": "OOD", "text": text, "expected": "defaultFallbackIntent",
        **{k: r[k] for k in ["intent", "confidence", "top2_intent",
                              "top2_confidence", "margin",
                              "normalized_entropy", "known_token_ratio"]},
        "final_decision": final,
        "correct": final == "defaultFallbackIntent",
    })

starter_df = pd.DataFrame(starter)
starter_df.to_csv(OUT / "starter_sanity_results.csv", index=False)

print("\n" + "=" * 76)
print("STARTER SANITY")
print("=" * 76)
for _, r in starter_df.iterrows():
    print(
        f"{'PASS' if r.correct else 'FAIL':4s} | "
        f"{r.confidence*100:6.2f}% | "
        f"{r.final_decision:32s} | {r.text}"
    )

starter_supported = starter_df[starter_df.type == "SUPPORTED"]
starter_ood = starter_df[starter_df.type == "OOD"]
print(f"\nStarter supported correctness: {starter_supported.correct.mean()*100:.2f}%")
print(f"Starter OOD rejection        : {starter_ood.correct.mean()*100:.2f}%")


# Production gates.
# These are deliberately conservative starting gates and must be reviewed
# against the product's actual risk tolerance.
if calibrated:
    gate_calibration = True
    gate_in = in_acceptance >= 97.0
    gate_ood = ood_rejection >= 95.0
    production_ready = gate_calibration and gate_in and gate_ood
else:
    gate_calibration = False
    gate_in = False
    gate_ood = False
    production_ready = False

report = {
    "status": "PRODUCTION_CANDIDATE" if production_ready else "NOT_PRODUCTION_READY",
    "frozen_classifier": True,
    "model": str(MODEL),
    "model_size_mb": model.size_mb,
    "calibrated": calibrated,
    "threshold": threshold,
    "in_domain_acceptance_percent": in_acceptance,
    "ood_rejection_percent": ood_rejection,
    "ood_auroc": ood_auc,
    "ood_average_precision": ood_ap,
    "gates": {
        "calibration_available": gate_calibration,
        "in_domain_acceptance_ge_97": gate_in,
        "ood_rejection_ge_95": gate_ood,
    },
    "warning": (
        "Starter sanity tests are not production evidence. "
        "Use representative independently collected calibration/OOD data."
    ),
}

with open(OUT / "production_gate.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

with open(OUT / "production_gate_config.json", "w", encoding="utf-8") as f:
    json.dump({
        "model": str(MODEL),
        "threshold": threshold,
        "calibrated": calibrated,
        "in_domain_file": str(indomain_path) if indomain_path else None,
        "ood_file": str(ood_path) if ood_path else None,
    }, f, indent=2)

print("\n" + "=" * 76)
print("PRODUCTION READINESS")
print("=" * 76)
print(f"Frozen V2 INT8 : YES")
print(f"Calibration    : {'PASS' if gate_calibration else 'FAIL'}")
print(f"In-domain gate : {'PASS' if gate_in else 'FAIL'}")
print(f"OOD gate       : {'PASS' if gate_ood else 'FAIL'}")
print(f"STATUS         : {'PRODUCTION CANDIDATE' if production_ready else 'NOT PRODUCTION READY'}")
print(f"\nSaved to: {OUT}")

if not calibrated:
    print("\nNext required files:")
    print("  production_indomain_calibration.csv")
    print("  production_ood_calibration.csv")
    print("\nThen rerun this script.")

print("\nCurrent INT8 baseline was NOT modified.")
