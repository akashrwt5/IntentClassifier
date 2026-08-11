#!/usr/bin/env python3
"""
V4 STRONG SEMANTIC MODEL — TRAINING PLAN / DATA BUILDER

Goal:
  Build a stronger semantic intent model than the locked V3 baseline.
  Size is NOT the optimization target.

This script prepares a V4 training manifest and expanded error-driven
dataset from the existing project data. It intentionally DOES NOT touch:
  - V3 model
  - 595-row unseen evaluation set
  - V3 ONNX
  - production baseline artifacts

Recommended next architecture:
  multilingual/pretrained semantic encoder -> intent classifier
For this first V4 step, we prepare and audit the data before selecting
the exact encoder/checkpoint.

Why data-first:
  V3 already reaches 96.47% unseen / 100% contextual / 100% targeted.
  The remaining problems are generalization, short commands, typos/STT
  variation, confusing intent boundaries, and OOD scope.

Output:
  v4_semantic_training/
    v4_manifest.json
    v4_training_data.csv
    v4_dev_data.csv
    v4_intent_pair_tests.csv
    v4_ood_data.csv
    v4_short_command_data.csv
    v4_typo_data.csv
    v4_hard_negative_data.csv
"""

from pathlib import Path
import json
import re
import pandas as pd
import numpy as np

ROOT = Path("/Users/shuklam/IntentClassifier/semantic_project")
OUT = ROOT / "v4_semantic_training"
OUT.mkdir(parents=True, exist_ok=True)

# Existing balanced semantic dataset is preferred as the normal source.
CANDIDATE_FILES = [
    ROOT / "balanced_dataset.csv",
    ROOT / "semantic_dataset.csv",
    ROOT / "dataset.csv",
    ROOT / "fine_tuned_test_predictions.csv",
]

# Never use this file for V4 training.
LOCKED_UNSEEN = ROOT / "unseen_semantic_stress_test.csv"

INTENTS = [
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

def norm(x):
    return re.sub(r"\s+", " ", str(x).strip().lower())

def find_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def load_existing():
    # Prefer a normal training dataset over prediction outputs.
    for p in CANDIDATE_FILES:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        tc = find_col(df, ["text", "utterance", "sentence"])
        lc = find_col(df, ["intent", "label", "expected", "true_intent"])
        if tc and lc:
            x = pd.DataFrame({
                "text": df[tc].map(norm),
                "intent": df[lc].astype(str).str.strip(),
                "source": p.name,
            })
            x = x[x.intent.isin(INTENTS)]
            if len(x):
                return x, p
    raise FileNotFoundError(
        "Could not find an existing training CSV with text + intent columns. "
        "Please point CANDIDATE_FILES to your balanced dataset."
    )

base, source = load_existing()

# Remove exact duplicates from the training source.
base = base.drop_duplicates(["text", "intent"]).reset_index(drop=True)

# ---------------------------------------------------------------------
# 1. Short natural commands
# ---------------------------------------------------------------------
SHORT = {
    "device.volume.increase": [
        "louder", "turn it up", "turn the sound up", "increase volume",
        "raise the volume", "make it louder", "more volume",
    ],
    "device.volume.decrease": [
        "quieter", "turn it down", "turn the sound down", "decrease volume",
        "lower the volume", "make it quieter", "less volume",
    ],
    "device.volume.mute": [
        "mute", "mute it", "silence it", "make it silent", "turn the sound off",
        "stop the sound",
    ],
    "device.volume.unmute": [
        "unmute", "unmute it", "turn sound back on", "turn it back on",
        "restore the sound",
    ],
    "device.memory.change": [
        "change memory", "change my memory", "update memory",
    ],
    "find.phone.locate": [
        "find my phone", "locate my phone", "where is my phone",
    ],
    "help.reminder.show": [
        "show reminders", "show my reminders", "list reminders",
        "my reminders",
    ],
    "reminders.task.complete": [
        "complete reminder", "finish reminder", "mark reminder complete",
        "complete that reminder",
    ],
    "reminders.task.create": [
        "set a reminder", "create a reminder", "add a reminder",
        "remind me tomorrow", "set reminder",
    ],
    "streaming.session.start": [
        "start streaming", "start the stream", "begin streaming",
        "play streaming",
    ],
    "streaming.session.stop": [
        "stop streaming", "stop the stream", "end streaming",
        "end the stream",
    ],
}

# ---------------------------------------------------------------------
# 2. Controlled STT/typo variants of known production phrases.
# ---------------------------------------------------------------------
TYPO = [
    ("i need to go to airport tommorow", "reminders.task.create"),
    ("i need to go to airport tomorow", "reminders.task.create"),
    ("make it quiter", "device.volume.decrease"),
    ("make it quiteer", "device.volume.decrease"),
    ("turn the sound upp", "device.volume.increase"),
    ("make it loader", "device.volume.increase"),
    ("turn it dowm", "device.volume.decrease"),
    ("unmte it", "device.volume.unmute"),
    ("mutt it", "device.volume.mute"),
    ("find my phne", "find.phone.locate"),
    ("show my remidners", "help.reminder.show"),
    ("start streeming", "streaming.session.start"),
    ("stop streeming", "streaming.session.stop"),
]

# ---------------------------------------------------------------------
# 3. Natural contextual paraphrases.
# ---------------------------------------------------------------------
PARAPHRASES = [
    ("it's really quiet in here, can you make the sound louder",
     "device.volume.increase"),
    ("i can barely hear it, turn the volume up",
     "device.volume.increase"),
    ("the audio is low, please raise it",
     "device.volume.increase"),
    ("it's too loud for me, lower the sound",
     "device.volume.decrease"),
    ("the sound is too strong, can you turn it down",
     "device.volume.decrease"),
    ("i want the audio completely silent",
     "device.volume.mute"),
    ("please stop all sound",
     "device.volume.mute"),
    ("i want the sound back on",
     "device.volume.unmute"),
    ("please restore the audio",
     "device.volume.unmute"),
    ("remind me that i need to go to the airport tomorrow",
     "reminders.task.create"),
    ("what reminders do i currently have",
     "help.reminder.show"),
    ("mark the reminder as done",
     "reminders.task.complete"),
    ("can you help me find my phone",
     "find.phone.locate"),
    ("start the hearing aid streaming session",
     "streaming.session.start"),
    ("please end the streaming session now",
     "streaming.session.stop"),
]

# ---------------------------------------------------------------------
# 4. OOD examples. These are NOT intent labels.
# ---------------------------------------------------------------------
OOD = [
    "tell me a joke",
    "who is the prime minister of india",
    "what is the weather today",
    "what is the capital of france",
    "what is bitcoin",
    "what is the stock market doing",
    "book a hotel for tomorrow",
    "i want to order pizza",
    "open my browser",
    "what time is the train",
    "how do i cook pasta",
    "play music",
    "how are you",
    "thank you",
    "good morning",
    "what happened in the news today",
    "translate this sentence",
    "set an alarm on my phone",
    "send an email",
    "take a photo",
    "call my friend",
    "show me a recipe",
    "what is the temperature",
    "tell me about football",
    "asdfghjkl",
    "jkakjhdjkhd",
    "123456789",
]

# ---------------------------------------------------------------------
# 5. Confusing intent-pair tests.
# These are evaluation/training diagnostics, not unseen-set replacements.
# ---------------------------------------------------------------------
PAIR_TESTS = [
    ("make it louder", "device.volume.increase", "device.volume.decrease"),
    ("make it quieter", "device.volume.decrease", "device.volume.increase"),
    ("turn it up", "device.volume.increase", "device.volume.decrease"),
    ("turn it down", "device.volume.decrease", "device.volume.increase"),
    ("mute it", "device.volume.mute", "device.volume.unmute"),
    ("unmute it", "device.volume.unmute", "device.volume.mute"),
    ("start streaming", "streaming.session.start", "streaming.session.stop"),
    ("stop streaming", "streaming.session.stop", "streaming.session.start"),
    ("create a reminder", "reminders.task.create", "reminders.task.complete"),
    ("complete the reminder", "reminders.task.complete", "reminders.task.create"),
]

def rows_from_dict(d, source_name):
    rows = []
    for intent, texts in d.items():
        for t in texts:
            rows.append({
                "text": norm(t),
                "intent": intent,
                "source": source_name,
            })
    return rows

extra = []
extra.extend(rows_from_dict(SHORT, "v4_short_commands"))
extra.extend(
    {"text": norm(t), "intent": y, "source": "v4_typo_stt"}
    for t, y in TYPO
)
extra.extend(
    {"text": norm(t), "intent": y, "source": "v4_contextual_paraphrase"}
    for t, y in PARAPHRASES
)

extra_df = pd.DataFrame(extra)

# Do not accidentally include the locked unseen set.
if LOCKED_UNSEEN.exists():
    unseen = pd.read_csv(LOCKED_UNSEEN)
    unseen_text_col = find_col(unseen, ["text", "utterance", "sentence"])
    if unseen_text_col:
        locked_texts = set(unseen[unseen_text_col].map(norm))
        extra_df = extra_df[~extra_df.text.isin(locked_texts)]
        base = base[~base.text.isin(locked_texts)]

train = pd.concat([base, extra_df], ignore_index=True)
train = train.drop_duplicates(["text", "intent"]).reset_index(drop=True)

# Deterministic dev split only from the existing training pool.
# Do NOT use the 595-row unseen test.
rng = np.random.default_rng(42)
dev_parts = []
train_parts = []

for intent in INTENTS:
    sub = train[train.intent == intent].sample(
        frac=1.0, random_state=42
    ).reset_index(drop=True)

    n_dev = max(20, int(round(len(sub) * 0.10)))
    dev_parts.append(sub.iloc[:n_dev])
    train_parts.append(sub.iloc[n_dev:])

dev = pd.concat(dev_parts, ignore_index=True)
train_final = pd.concat(train_parts, ignore_index=True)

# Save specialized datasets.
short_df = pd.DataFrame(
    rows_from_dict(SHORT, "v4_short_commands")
).drop_duplicates()

typo_df = pd.DataFrame(
    [{"text": norm(t), "intent": y, "source": "v4_typo_stt"} for t, y in TYPO]
).drop_duplicates()

ood_df = pd.DataFrame(
    [{"text": norm(t), "intent": "OOD", "source": "v4_ood"} for t in OOD]
).drop_duplicates()

pair_df = pd.DataFrame(
    [
        {
            "text": t,
            "positive_intent": pos,
            "confusable_intent": neg,
        }
        for t, pos, neg in PAIR_TESTS
    ]
)

train_final.to_csv(OUT / "v4_training_data.csv", index=False)
dev.to_csv(OUT / "v4_dev_data.csv", index=False)
short_df.to_csv(OUT / "v4_short_command_data.csv", index=False)
typo_df.to_csv(OUT / "v4_typo_data.csv", index=False)
ood_df.to_csv(OUT / "v4_ood_data.csv", index=False)
pair_df.to_csv(OUT / "v4_intent_pair_tests.csv", index=False)

manifest = {
    "base_source": str(source),
    "base_rows_after_dedup": int(len(base)),
    "v4_training_rows": int(len(train_final)),
    "v4_dev_rows": int(len(dev)),
    "short_command_rows": int(len(short_df)),
    "typo_rows": int(len(typo_df)),
    "ood_rows": int(len(ood_df)),
    "intent_pair_test_rows": int(len(pair_df)),
    "intents": INTENTS,
    "unseen_test_locked": str(LOCKED_UNSEEN),
    "unseen_test_used_for_training": False,
    "v3_modified": False,
    "recommendation": (
        "Train V4 using a strong pretrained semantic encoder first; "
        "benchmark against locked V3 before export/deployment."
    ),
}

(OUT / "v4_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print("=" * 78)
print("V4 SEMANTIC DATA BUILD")
print("=" * 78)
print("Base source:", source)
print("Base rows after dedup:", len(base))
print("V4 training rows:", len(train_final))
print("V4 dev rows:", len(dev))
print("Short commands:", len(short_df))
print("Typo/STT variants:", len(typo_df))
print("OOD examples:", len(ood_df))
print("Intent-pair tests:", len(pair_df))
print()
print("595-row unseen test used for training: NO")
print("V3 model modified: NO")
print()
print("NEXT MODEL:")
print("  Strong pretrained semantic encoder")
print("  + 11-intent classification head")
print("  + OOD/scope evaluation")
print("  + locked V3 benchmark")
print()
print("Saved:", OUT)
