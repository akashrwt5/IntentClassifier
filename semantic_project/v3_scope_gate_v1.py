#!/usr/bin/env python3
"""
V3 FP32 ONNX + DOMAIN/OOD SCOPE GATE V1

Problem solved:
A closed-set 11-intent classifier always chooses one of its 11 intents.
Therefore high confidence alone cannot distinguish:
  "make it louder"          -> valid
  "tell me a joke"          -> OOD
even when the latter gets 94%+ confidence.

This script adds a separate domain gate BEFORE accepting the V3 prediction.

Pipeline:
  Text
    -> normalization
    -> V3 FP32 ONNX
    -> confidence/margin/entropy
    -> domain/OOD gate
    -> ACCEPT intent OR NO_INTENT

The domain gate is built from:
  - in-domain calibration examples
  - OOD calibration examples

It uses TF-IDF word + character n-grams and a logistic regression scope
classifier. This is intentionally a Python validation prototype first.
It does NOT modify V3 ONNX.

IMPORTANT:
- 595-row unseen test is NOT used.
- V3 ONNX is NOT modified.
- This is a production-safety prototype, not yet the final mobile artifact.
- Real microphone/OOD data is still required before release.
"""

from pathlib import Path
import json
import re
import pickle
import numpy as np
import pandas as pd
import onnxruntime as ort

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path("/Users/shuklam/IntentClassifier/semantic_project")

MODEL = ROOT / "tiny_semantic_student_v3_fp32" / "v3_semantic_student_fp32.onnx"
VOCAB_FILE = ROOT / "tiny_semantic_student_v2_balanced" / "vocab.json"
CAL_DIR = ROOT / "production_calibration_v2"
INDOMAIN = CAL_DIR / "production_indomain_calibration.csv"
OOD = CAL_DIR / "production_ood_calibration.csv"

OUT = ROOT / "v3_scope_gate_v1"
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

# V3 classifier threshold is intentionally lower than the old 0.97 gate.
# The scope gate is now responsible for domain rejection.
V3_CONFIDENCE = 0.87

# Scope probability required to consider text part of the hearing-aid domain.
# We sweep this on calibration data and report a recommended candidate.
SCOPE_THRESHOLD_DEFAULT = 0.80

def normalize(text):
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text

def find_col(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None

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

def v3_predict(session, input_name, output_name, text):
    x = np.asarray([tokenize(text)], dtype=np.int64)
    logits = session.run([output_name], {input_name: x})[0][0]
    probs = softmax(logits)
    order = np.argsort(probs)[::-1]

    top1 = int(order[0])
    top2 = int(order[1])

    confidence = float(probs[top1])
    second = float(probs[top2])
    margin = confidence - second
    p = probs[probs > 0]
    ent = float(-np.sum(p * np.log(p)))

    return {
        "intent": LABELS[top1],
        "confidence": confidence,
        "margin": margin,
        "entropy": ent,
        "top2": LABELS[top2],
        "top2_confidence": second,
    }

if not MODEL.exists():
    raise FileNotFoundError(MODEL)
if not INDOMAIN.exists():
    raise FileNotFoundError(INDOMAIN)
if not OOD.exists():
    raise FileNotFoundError(OOD)

ind = pd.read_csv(INDOMAIN)
ood = pd.read_csv(OOD)

ind_text_col = find_col(ind, ["text", "utterance", "sentence"])
ood_text_col = find_col(ood, ["text", "utterance", "sentence"])

if not ind_text_col:
    raise ValueError(f"No text column in {INDOMAIN}: {list(ind.columns)}")
if not ood_text_col:
    raise ValueError(f"No text column in {OOD}: {list(ood.columns)}")

ind_text = ind[ind_text_col].astype(str).map(normalize)
ood_text = ood[ood_text_col].astype(str).map(normalize)

X_text = pd.concat([ind_text, ood_text], ignore_index=True)
y_scope = np.array(
    ["IN_DOMAIN"] * len(ind_text) +
    ["OOD"] * len(ood_text)
)

# ---------------------------------------------------------------------
# Scope classifier
# ---------------------------------------------------------------------
word = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True,
    max_features=12000,
)

char = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=1,
    sublinear_tf=True,
    max_features=16000,
)

X_word = word.fit_transform(X_text)
X_char = char.fit_transform(X_text)
X = hstack([X_word, X_char]).tocsr()

scope_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42,
)

scope_model.fit(X, y_scope)

with open(OUT / "scope_gate_v1.pkl", "wb") as f:
    pickle.dump(
        {
            "word_vectorizer": word,
            "char_vectorizer": char,
            "scope_model": scope_model,
            "v3_confidence": V3_CONFIDENCE,
        },
        f,
    )

# Calibration evaluation on the same calibration pool is diagnostic only.
scope_prob = scope_model.predict_proba(X)
in_index = list(scope_model.classes_).index("IN_DOMAIN")
scope_in_prob = scope_prob[:, in_index]

cal = pd.DataFrame({
    "text": X_text,
    "expected_scope": y_scope,
    "scope_probability": scope_in_prob,
})

# ---------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------
thresholds = np.arange(0.50, 0.991, 0.01)
rows = []

for t in thresholds:
    accepted = scope_in_prob >= t

    in_mask = y_scope == "IN_DOMAIN"
    ood_mask = y_scope == "OOD"

    in_coverage = float(accepted[in_mask].mean())
    ood_rejection = float((~accepted[ood_mask]).mean())

    in_wrong_reject = float((~accepted[in_mask]).mean())
    ood_false_accept = float(accepted[ood_mask].mean())

    # A simple conservative diagnostic score.
    score = in_coverage * ood_rejection

    rows.append({
        "scope_threshold": float(t),
        "in_domain_coverage": in_coverage,
        "in_domain_false_reject": in_wrong_reject,
        "ood_rejection": ood_rejection,
        "ood_false_accept": ood_false_accept,
        "score": score,
    })

sweep = pd.DataFrame(rows)

# Choose a candidate that keeps >=98% of calibration in-domain examples,
# then maximizes OOD rejection.
eligible = sweep[sweep.in_domain_coverage >= 0.98]

if len(eligible):
    best = eligible.sort_values(
        ["ood_rejection", "in_domain_coverage"],
        ascending=[False, False],
    ).iloc[0]
else:
    best = sweep.sort_values(
        ["score", "ood_rejection"],
        ascending=[False, False],
    ).iloc[0]

scope_threshold = float(best.scope_threshold)

sweep.to_csv(OUT / "scope_threshold_sweep.csv", index=False)
cal.to_csv(OUT / "scope_calibration_scores.csv", index=False)

# ---------------------------------------------------------------------
# Actual V3 + scope pipeline
# ---------------------------------------------------------------------
session = ort.InferenceSession(
    str(MODEL),
    providers=["CPUExecutionProvider"],
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def scope_probability(text):
    z = [normalize(text)]
    a = word.transform(z)
    b = char.transform(z)
    xx = hstack([a, b]).tocsr()
    probs = scope_model.predict_proba(xx)[0]
    return float(probs[in_index])

def pipeline(text):
    text = normalize(text)
    v3 = v3_predict(session, input_name, output_name, text)
    scope = scope_probability(text)

    accepted = (
        v3["confidence"] >= V3_CONFIDENCE
        and scope >= scope_threshold
    )

    return {
        "text": text,
        "intent": v3["intent"],
        "confidence": v3["confidence"],
        "margin": v3["margin"],
        "entropy": v3["entropy"],
        "scope_probability": scope,
        "scope_threshold": scope_threshold,
        "decision": "ACCEPT" if accepted else "NO_INTENT",
    }

TESTS = [
    ("make it louder", "IN_DOMAIN"),
    ("it's quieter can you make it a little louder", "IN_DOMAIN"),
    ("the audio is quiet but don't mute it make it louder", "IN_DOMAIN"),
    ("make it quieter", "IN_DOMAIN"),
    ("the audio is loud but keep it on just lower it", "IN_DOMAIN"),
    ("mute it", "IN_DOMAIN"),
    ("i can still hear it make it completely silent", "IN_DOMAIN"),
    ("turn off", "IN_DOMAIN"),
    ("unmute it", "IN_DOMAIN"),
    ("turn the sound back on", "IN_DOMAIN"),
    ("i need to go to airport tomorrow", "IN_DOMAIN"),
    ("i need to go to airport tommorow", "IN_DOMAIN"),
    ("where can i find my phone", "IN_DOMAIN"),
    ("please show the reminders i have", "IN_DOMAIN"),
    ("mark that reminder as completed", "IN_DOMAIN"),
    ("please start the streaming session", "IN_DOMAIN"),
    ("please stop the streaming session", "IN_DOMAIN"),

    ("tell me a joke", "OOD"),
    ("who is the prime minister of India", "OOD"),
    ("what is the weather today", "OOD"),
    ("what is the capital of France", "OOD"),
    ("play music", "OOD"),
    ("book a hotel for tomorrow", "OOD"),
    ("open my browser", "OOD"),
    ("what is bitcoin", "OOD"),
    ("how are you", "OOD"),
    ("asdfghjkl", "OOD"),
]

test_rows = []
for text, expected in TESTS:
    r = pipeline(text)
    test_rows.append({
        **r,
        "expected_scope": expected,
        "scope_correct": (
            (r["decision"] == "ACCEPT" and expected == "IN_DOMAIN") or
            (r["decision"] == "NO_INTENT" and expected == "OOD")
        ),
    })

tests = pd.DataFrame(test_rows)
tests.to_csv(OUT / "pipeline_scope_test.csv", index=False)

manifest = {
    "model": str(MODEL),
    "scope_model": str(OUT / "scope_gate_v1.pkl"),
    "in_domain_rows": int(len(ind_text)),
    "ood_rows": int(len(ood_text)),
    "v3_confidence_threshold": V3_CONFIDENCE,
    "scope_threshold_candidate": scope_threshold,
    "note": (
        "Scope gate is a Python prototype. It must be evaluated on "
        "real-world microphone/OOD data before production deployment."
    ),
}
(OUT / "scope_gate_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print("=" * 78)
print("V3 FP32 ONNX + DOMAIN/OOD SCOPE GATE V1")
print("=" * 78)
print(f"V3 model       : {MODEL}")
print(f"In-domain rows : {len(ind_text)}")
print(f"OOD rows       : {len(ood_text)}")
print()
print("Candidate scope threshold:", f"{scope_threshold:.2f}")
print(f"V3 confidence threshold  : {V3_CONFIDENCE:.2f}")
print()
print("CALIBRATION")
print(f"In-domain coverage : {best.in_domain_coverage*100:.2f}%")
print(f"OOD rejection      : {best.ood_rejection*100:.2f}%")
print(f"OOD false accepts   : {best.ood_false_accept*100:.2f}%")

print()
print("=" * 78)
print("END-TO-END TESTS")
print("=" * 78)

for _, r in tests.iterrows():
    print(
        f"{'PASS' if r.scope_correct else 'FAIL'} | "
        f"scope={r.scope_probability*100:6.2f}% | "
        f"v3={r.confidence*100:6.2f}% | "
        f"{r.decision:9s} | "
        f"{r.intent:34s} | "
        f"{r.text}"
    )

print()
print("IMPORTANT:")
print("- V3 ONNX was NOT modified.")
print("- 595-row unseen test was NOT used.")
print("- Scope gate is a Python prototype.")
print("- Real microphone/OOD data is required before release.")
print()
print("Saved:", OUT)
