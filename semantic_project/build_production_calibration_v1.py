#!/usr/bin/env python3
"""
build_production_calibration_v1.py

Creates production calibration datasets without modifying the frozen V2 INT8
classifier.

Inputs:
  1) Existing labeled dataset CSV: auto-detected from common filenames.
  2) Optional existing OOD CSV: if present, it is merged.
  3) Optional existing unseen/contextual CSVs are also searched.

Outputs:
  production_indomain_calibration.csv
  production_ood_calibration.csv
  production_hard_negative.csv
  production_calibration_manifest.json

IMPORTANT:
- The 2,064 validation rows should be used for IN-DOMAIN calibration.
- The 595 unseen test rows must NOT be used for calibration.
- OOD examples generated here are a starter set, not a final production
  guarantee. Add independently collected OOD/near-OOD examples before release.
"""

from pathlib import Path
import json
import re
import random
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "production_calibration_v1"
OUT.mkdir(parents=True, exist_ok=True)

random.seed(42)

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

# Common project files. The script picks the first matching CSV that contains
# both text and an intent/label column.
CSV_CANDIDATES = [
    ROOT / "dataset.csv",
    ROOT / "balanced_dataset.csv",
    ROOT / "semantic_dataset.csv",
    ROOT / "train.csv",
    ROOT / "data.csv",
]

def clean(s):
    return re.sub(r"\s+", " ", str(s).strip())

def find_labeled_csv():
    for p in CSV_CANDIDATES:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "text" not in df.columns:
            continue
        intent_col = next(
            (c for c in ["intent", "label", "expected_intent", "true_intent"]
             if c in df.columns),
            None
        )
        if intent_col:
            return p, df.rename(columns={intent_col: "intent"})
    # fallback: inspect all CSVs in root
    for p in sorted(ROOT.glob("*.csv")):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "text" not in df.columns:
            continue
        intent_col = next(
            (c for c in ["intent", "label", "expected_intent", "true_intent"]
             if c in df.columns),
            None
        )
        if intent_col:
            return p, df.rename(columns={intent_col: "intent"})
    return None, None

source_path, source = find_labeled_csv()

if source is None:
    raise FileNotFoundError(
        "No labeled dataset CSV found. Put your project dataset CSV in "
        f"{ROOT} and ensure it has text + intent columns."
    )

source = source[["text", "intent"]].dropna()
source["text"] = source["text"].map(clean)
source = source[source["intent"].isin(LABELS)].drop_duplicates("text")

print("=" * 78)
print("BUILD PRODUCTION CALIBRATION V1")
print("=" * 78)
print("Source:", source_path)
print("Rows:", len(source))
print()

# ---------------------------------------------------------------------
# IN-DOMAIN CALIBRATION
# ---------------------------------------------------------------------
# Prefer a validation file if the project already has one. Otherwise create
# a stratified 10% calibration sample from the labeled dataset. We never touch
# the frozen ONNX model.
#
# If a file named validation.csv / val.csv exists, use it.
# ---------------------------------------------------------------------

validation_candidates = [
    ROOT / "validation.csv",
    ROOT / "val.csv",
    ROOT / "validation_dataset.csv",
    ROOT / "semantic_validation.csv",
]

validation = None
validation_path = None

for p in validation_candidates:
    if not p.exists():
        continue
    try:
        df = pd.read_csv(p)
    except Exception:
        continue
    if "text" not in df.columns:
        continue
    intent_col = next(
        (c for c in ["intent", "label", "expected_intent", "true_intent"]
         if c in df.columns),
        None
    )
    if intent_col:
        validation = df.rename(columns={intent_col: "intent"})[
            ["text", "intent"]
        ].dropna()
        validation["text"] = validation["text"].map(clean)
        validation = validation[
            validation["intent"].isin(LABELS)
        ].drop_duplicates("text")
        validation_path = p
        break

if validation is not None and len(validation) >= 500:
    indomain = validation.copy()
    indomain_source = str(validation_path)
else:
    # Stratified fallback. This is only a calibration candidate; if the
    # project already has a known 2,064-row validation split, use that file.
    pieces = []
    per_intent = max(1, min(300, len(source) // len(LABELS)))
    for intent in LABELS:
        part = source[source.intent == intent]
        n = min(per_intent, len(part))
        pieces.append(part.sample(n=n, random_state=42))
    indomain = pd.concat(pieces, ignore_index=True)
    indomain_source = "stratified fallback from labeled dataset"

indomain = indomain.drop_duplicates("text").reset_index(drop=True)
indomain.to_csv(
    OUT / "production_indomain_calibration.csv",
    index=False
)

# ---------------------------------------------------------------------
# HARD NEGATIVES
# ---------------------------------------------------------------------
hard_negative_templates = {
    "device.volume.increase": [
        "the audio is quiet but don't mute it make it louder",
        "it's too quiet, don't silence it, turn it up",
        "I can barely hear it, keep it on and make it louder",
        "don't turn the sound off, increase the volume",
        "keep the sound on but raise the volume",
        "it is quiet here, make it louder without muting",
        "I don't want mute, I want more volume",
        "don't silence it, just make it louder",
        "keep audio enabled and turn it up",
        "the sound is low, increase it but don't mute it",
    ],
    "device.volume.decrease": [
        "the audio is loud but don't mute it make it quieter",
        "it's too loud, keep it on and lower the volume",
        "don't silence it, just turn the volume down",
        "keep sound enabled but reduce the volume",
        "I don't want mute, I want less volume",
        "lower the sound without turning it off",
        "make it quieter but keep audio on",
        "don't turn it off, just reduce the volume",
        "the sound is high, lower it without muting",
        "keep it audible but turn it down",
    ],
    "device.volume.mute": [
        "I want complete silence",
        "silence the sound completely",
        "make it completely silent",
        "mute the audio now",
        "I don't want to hear anything",
        "turn the sound completely off",
    ],
    "device.volume.unmute": [
        "turn the sound back on",
        "restore the sound",
        "unmute the audio",
        "enable sound again",
        "I want audio back",
        "turn audio back on",
    ],
}

hard_rows = []
for intent, texts in hard_negative_templates.items():
    for text in texts:
        hard_rows.append({
            "text": text,
            "intent": intent,
            "source": "production_hard_negative_v1",
        })

hard_df = pd.DataFrame(hard_rows).drop_duplicates("text")
hard_df.to_csv(
    OUT / "production_hard_negative.csv",
    index=False
)

# ---------------------------------------------------------------------
# OOD CALIBRATION
# ---------------------------------------------------------------------
# Broad categories. These are intentionally outside the 11 supported intents.
# This is a starter set and should be expanded with real user-like OOD data.
# ---------------------------------------------------------------------

ood_groups = {
    "general_knowledge": [
        "who is the prime minister of India",
        "who is the priminister of india",
        "what is the capital of France",
        "what is the population of India",
        "who invented the telephone",
        "when did the moon landing happen",
        "what is the tallest mountain",
        "who won the cricket match",
        "what is artificial intelligence",
        "what is machine learning",
        "explain quantum computing",
        "what is the stock market doing",
        "what is bitcoin",
        "what is the exchange rate today",
        "who is the president of the United States",
        "what is the temperature today",
    ],
    "weather": [
        "what is the weather today",
        "will it rain tomorrow",
        "is it going to be hot today",
        "what is the forecast",
        "is there a storm coming",
        "how cold will it be tonight",
        "will it rain this weekend",
        "what is the temperature outside",
    ],
    "food": [
        "I want to order pizza",
        "how do I cook pasta",
        "what should I eat for dinner",
        "find me a restaurant",
        "give me a biryani recipe",
        "I am hungry",
        "where can I get coffee",
        "order some food",
    ],
    "travel": [
        "book a hotel for tomorrow",
        "I need a flight to Mumbai",
        "find a hotel near the airport",
        "what time is my train",
        "how do I get to Delhi",
        "book a taxi",
        "show me flights to London",
        "I am planning a holiday",
    ],
    "web_device_unrelated": [
        "open my browser",
        "open youtube",
        "send an email",
        "call my friend",
        "take a photo",
        "open settings",
        "turn on bluetooth",
        "connect to wifi",
        "play music",
        "pause the video",
        "open maps",
    ],
    "conversation": [
        "how are you",
        "thank you",
        "good morning",
        "good night",
        "hello",
        "what are you doing",
        "tell me a joke",
        "that's great",
        "I am bored",
        "can you help me",
    ],
    "finance": [
        "what is the stock market doing",
        "should I buy shares",
        "what is the price of gold",
        "show me today's market news",
        "how is bitcoin doing",
        "what is the interest rate",
    ],
    "random_garbage": [
        "jkakjhdjkhd",
        "sdkjadsjj",
        "asdfghjkl",
        "qwertyuiop",
        "zzzxxyyqq",
        "123456789",
        "hello xyz abc",
        "qazwsxedc",
        "random words banana airplane",
        "kjhkjhkjh",
    ],
    "near_ood": [
        "make the music more enjoyable",
        "make this sound better",
        "adjust what I am hearing",
        "I want a better listening experience",
        "the audio does not feel right",
        "can you improve the sound quality",
        "make the audio comfortable",
        "fix the sound for me",
        "I am having trouble hearing this",
        "help me with the audio",
        "change the way this sounds",
        "make the listening experience better",
    ],
    "adversarial": [
        "who is the prime minister but make it louder",
        "tell me a joke and turn the volume up",
        "what is the weather and mute it",
        "play music but do not change the hearing aid",
        "book a hotel and make it quieter",
        "what is bitcoin make it louder",
        "open youtube and turn the sound back on",
        "tell me the capital of France",
        "I need a flight tomorrow",
        "please browse the internet",
    ],
}

ood_rows = []
for group, texts in ood_groups.items():
    for text in texts:
        ood_rows.append({
            "text": clean(text),
            "ood_group": group,
        })

# Add simple perturbations for OOD coverage.
base_ood = [r["text"] for r in ood_rows]
perturbations = [
    lambda s: s + " please",
    lambda s: "please " + s,
    lambda s: s + " now",
    lambda s: s.replace("the", "teh"),
    lambda s: s.replace("tomorrow", "tommorow"),
]

for text in base_ood:
    for fn in random.sample(perturbations, k=2):
        new_text = clean(fn(text))
        if new_text != text:
            ood_rows.append({
                "text": new_text,
                "ood_group": "perturbed_ood",
            })

ood_df = pd.DataFrame(ood_rows).drop_duplicates("text")

# Add more synthetic OOD combinations to get a larger starter set.
subjects = [
    "weather", "bitcoin", "pizza", "football", "cricket",
    "hotel", "flight", "train", "recipe", "stock market",
    "browser", "youtube", "email", "tax", "news",
]
actions = [
    "what is", "tell me about", "show me", "explain",
    "how do I", "where can I find", "give me information about",
]
objects = [
    "the weather", "bitcoin", "pizza", "the cricket score",
    "a hotel", "a flight", "the train schedule", "a pasta recipe",
    "the stock market", "my browser", "youtube", "email",
    "today's news", "tax rules",
]

synthetic = []
for _ in range(2500):
    a = random.choice(actions)
    o = random.choice(objects)
    s = clean(f"{a} {o}")
    synthetic.append({
        "text": s,
        "ood_group": "synthetic_general_ood",
    })

ood_df = pd.concat(
    [ood_df, pd.DataFrame(synthetic)],
    ignore_index=True
).drop_duplicates("text").reset_index(drop=True)

ood_df.to_csv(
    OUT / "production_ood_calibration.csv",
    index=False
)

# ---------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------

manifest = {
    "source_labeled_dataset": str(source_path),
    "indomain_source": indomain_source,
    "indomain_rows": int(len(indomain)),
    "ood_rows": int(len(ood_df)),
    "hard_negative_rows": int(len(hard_df)),
    "intents": LABELS,
    "do_not_use_for_calibration": [
        "unseen_semantic_stress_test.csv",
        "595-sample final test set",
    ],
    "notes": [
        "OOD data generated here is a starter calibration set.",
        "Production release requires independently collected OOD data.",
        "Near-OOD and adversarial examples are intentionally included.",
        "V2 INT8 classifier is not modified by this script.",
    ],
}

with open(
    OUT / "production_calibration_manifest.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(manifest, f, indent=2)

print("\n" + "=" * 78)
print("CALIBRATION DATASET CREATED")
print("=" * 78)
print(f"In-domain calibration : {len(indomain)} rows")
print(f"OOD calibration       : {len(ood_df)} rows")
print(f"Hard negatives        : {len(hard_df)} rows")
print(f"Output directory      : {OUT}")
print()
print("Files:")
print("  production_indomain_calibration.csv")
print("  production_ood_calibration.csv")
print("  production_hard_negative.csv")
print("  production_calibration_manifest.json")
print()
print("IMPORTANT:")
print("- The frozen V2 INT8 model was NOT changed.")
print("- The 595-row unseen test must remain untouched.")
print("- Synthetic OOD is only a calibration starter; add real OOD data before release.")
