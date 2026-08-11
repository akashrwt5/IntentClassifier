#!/usr/bin/env python3
"""
build_production_calibration_v2.py

Production calibration dataset builder for the frozen V2 INT8 intent model.

Creates:
  production_calibration_v2/
    production_indomain_calibration.csv
    production_ood_calibration.csv
    production_hard_negative.csv
    production_calibration_manifest.json

Goals:
  - preserve the existing in-domain calibration
  - expand OOD to >5,000 diverse examples
  - create 200+ targeted hard negatives
  - keep the 595-row unseen test OUT of calibration
  - keep V2 INT8 model untouched

IMPORTANT:
Synthetic OOD is calibration data, not proof of production safety.
Before release, add independently collected real OOD/near-OOD data.
"""

from pathlib import Path
import json
import random
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent
V1 = ROOT / "production_calibration_v1"
OUT = ROOT / "production_calibration_v2"
OUT.mkdir(parents=True, exist_ok=True)

random.seed(20260809)

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

def clean(s):
    return re.sub(r"\s+", " ", str(s).strip())

def load_existing():
    ind = V1 / "production_indomain_calibration.csv"
    ood = V1 / "production_ood_calibration.csv"
    hard = V1 / "production_hard_negative.csv"

    if not ind.exists():
        raise FileNotFoundError(
            f"Missing {ind}. Run build_production_calibration_v1.py first."
        )

    ind_df = pd.read_csv(ind)
    ood_df = pd.read_csv(ood) if ood.exists() else pd.DataFrame(columns=["text"])
    hard_df = pd.read_csv(hard) if hard.exists() else pd.DataFrame(
        columns=["text", "intent"]
    )

    return ind_df, ood_df, hard_df

indomain, old_ood, old_hard = load_existing()

# ---------------------------------------------------------------------
# 1) OOD phrase banks
# ---------------------------------------------------------------------

OOD_BANKS = {
    "general_knowledge": [
        "who is the prime minister of India",
        "who is the president of India",
        "what is the capital of France",
        "what is the population of India",
        "who invented the telephone",
        "when was the internet invented",
        "what is artificial intelligence",
        "what is machine learning",
        "explain quantum computing",
        "what is the tallest mountain",
        "who wrote this book",
        "what happened in history today",
        "what is the meaning of this word",
        "tell me about space",
        "how does gravity work",
        "what is photosynthesis",
        "who discovered electricity",
        "what is the largest ocean",
    ],
    "weather": [
        "what is the weather today",
        "what will the weather be tomorrow",
        "will it rain today",
        "is it going to rain tomorrow",
        "what is the temperature outside",
        "is it hot today",
        "is it cold tonight",
        "what is the weather forecast",
        "will there be a storm",
        "is it windy outside",
        "will it snow tomorrow",
        "what is the humidity",
    ],
    "food": [
        "I want to order pizza",
        "how do I cook pasta",
        "give me a pasta recipe",
        "find a restaurant",
        "where can I get coffee",
        "what should I eat tonight",
        "I am hungry",
        "order some food",
        "find me a good restaurant",
        "how do I make rice",
        "give me a biryani recipe",
        "what is a healthy breakfast",
    ],
    "travel": [
        "book a hotel for tomorrow",
        "find a hotel near the airport",
        "book a flight to Mumbai",
        "I need a flight to Delhi",
        "what time is the train",
        "book a taxi",
        "how do I get to the airport",
        "show me flights to London",
        "I am planning a holiday",
        "find a hotel for tonight",
        "what is the train schedule",
        "book a rental car",
    ],
    "finance": [
        "what is the stock market doing",
        "what is bitcoin worth",
        "should I buy shares",
        "what is the price of gold",
        "show me market news",
        "what is the interest rate",
        "how are stocks doing today",
        "what is the exchange rate",
        "tell me about cryptocurrency",
        "should I invest in stocks",
    ],
    "browser_phone": [
        "open my browser",
        "open youtube",
        "open maps",
        "send an email",
        "call my friend",
        "take a photo",
        "open settings",
        "turn on bluetooth",
        "connect to wifi",
        "play a video",
        "open my camera",
        "send a message",
        "check my calendar",
    ],
    "entertainment": [
        "play music",
        "play a song",
        "stop the movie",
        "pause the video",
        "play my playlist",
        "find a movie",
        "recommend a song",
        "play some relaxing music",
        "start a podcast",
        "what movie should I watch",
    ],
    "conversation": [
        "how are you",
        "thank you",
        "good morning",
        "good night",
        "hello",
        "what are you doing",
        "tell me a joke",
        "I am bored",
        "that is great",
        "nice to meet you",
        "can you help me",
        "what do you think",
    ],
    "shopping": [
        "buy a new phone",
        "find cheap headphones",
        "where can I buy shoes",
        "order a laptop",
        "show me the best headphones",
        "find a gift for my friend",
        "compare these phones",
        "where can I buy a charger",
    ],
    "health_general": [
        "I have a headache",
        "what should I eat when sick",
        "how can I sleep better",
        "what is a healthy diet",
        "why am I tired",
        "how much water should I drink",
    ],
    "work_school": [
        "write an email to my manager",
        "help me with my homework",
        "summarize this document",
        "create a presentation",
        "write a report",
        "explain this equation",
        "help me prepare for an interview",
    ],
    "near_ood_audio": [
        "make the listening experience better",
        "make this sound better",
        "adjust what I am hearing",
        "improve the audio quality",
        "make the sound more comfortable",
        "fix the sound for me",
        "help me hear this better",
        "I am having trouble hearing this",
        "change the way this sounds",
        "make the audio comfortable",
        "improve my listening experience",
        "the sound does not feel right",
        "can you optimize the sound",
        "make everything sound better",
        "help with the audio quality",
        "adjust my listening experience",
    ],
    "random_garbage": [
        "jkakjhdjkhd",
        "sdkjadsjj",
        "asdfghjkl",
        "qwertyuiop",
        "zzzxxyyqq",
        "qazwsxedc",
        "kjhkjhkjh",
        "random words banana airplane",
        "hello xyz abc",
        "123456789",
        "aaaa bbbb cccc",
        "zxcasdqwe",
    ],
    "adversarial": [
        "who is the prime minister but make it louder",
        "tell me a joke and turn the volume up",
        "what is the weather and mute it",
        "book a hotel and make it quieter",
        "what is bitcoin make it louder",
        "open youtube and turn the sound back on",
        "tell me the capital of France and lower it",
        "play music and mute the hearing aid",
        "find a restaurant and increase the volume",
        "what is the train time make it louder",
        "send an email and turn it down",
        "tell me a joke and mute it",
    ],
}

# ---------------------------------------------------------------------
# 2) Generate OOD variations
# ---------------------------------------------------------------------

prefixes = [
    "",
    "please ",
    "can you ",
    "could you ",
    "I want to know ",
    "I need to know ",
    "tell me ",
    "can you tell me ",
]

suffixes = [
    "",
    " please",
    " now",
    " today",
    " for me",
    " right now",
]

ood_rows = []

for category, phrases in OOD_BANKS.items():
    for phrase in phrases:
        phrase = clean(phrase)
        ood_rows.append({"text": phrase, "ood_group": category})

        # Natural request variants.
        for _ in range(8):
            p = random.choice(prefixes)
            s = random.choice(suffixes)
            variant = clean(p + phrase + s)
            if variant:
                ood_rows.append({
                    "text": variant,
                    "ood_group": category,
                })

# Add structured general OOD combinations.
subjects = [
    "weather", "bitcoin", "pizza", "cricket", "football",
    "the stock market", "a hotel", "a flight", "the train schedule",
    "a pasta recipe", "today's news", "my browser", "youtube",
    "email", "tax rules", "a movie", "a song", "the exchange rate",
]
verbs = [
    "what is", "tell me about", "show me", "explain",
    "how do I", "where can I find", "can you explain",
]
for _ in range(3500):
    text = clean(
        f"{random.choice(verbs)} {random.choice(subjects)}"
    )
    ood_rows.append({
        "text": text,
        "ood_group": "synthetic_general_ood",
    })

# More conversational/random OOD.
random_templates = [
    "I was wondering about {}",
    "Can you help me understand {}",
    "I want information about {}",
    "Please explain {}",
    "What can you tell me about {}",
]
for _ in range(1200):
    subject = random.choice(subjects)
    template = random.choice(random_templates)
    ood_rows.append({
        "text": clean(template.format(subject)),
        "ood_group": "synthetic_conversational_ood",
    })

new_ood = pd.DataFrame(ood_rows)
new_ood = new_ood.drop_duplicates("text")

# Merge existing V1 OOD.
if len(old_ood):
    old = old_ood.copy()
    if "ood_group" not in old.columns:
        old["ood_group"] = "existing_v1"
    old = old[["text", "ood_group"]]
    new_ood = pd.concat([old, new_ood], ignore_index=True)

new_ood["text"] = new_ood["text"].map(clean)
new_ood = new_ood.drop_duplicates("text").reset_index(drop=True)

# ---------------------------------------------------------------------
# 3) Targeted hard negatives
# ---------------------------------------------------------------------

HARD = {
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
        "please don't mute, I only want it louder",
        "don't make it silent, make it louder",
        "keep audio on and increase volume",
        "I still want sound, just turn it up",
        "do not mute it, increase the sound",
        "I need more volume, not silence",
        "raise the volume without muting",
        "make it louder while keeping sound on",
        "increase volume but leave it unmuted",
        "it is difficult to hear, turn it up",
        "I can hardly hear it, make it louder",
        "the sound is very low, raise it",
        "turn it up but don't turn it off",
        "make the audio louder, keep it playing",
        "I need the audio louder, not muted",
        "increase the sound level, do not silence",
        "make this louder and keep the audio enabled",
        "don't silence the device, increase volume",
        "keep it audible and turn it up",
        "I want higher volume, not mute",
        "make the hearing sound louder without muting",
        "turn the volume up, don't silence anything",
        "increase audio level while keeping it on",
        "the volume is low, raise it and keep it on",
        "make it a little louder, don't mute",
        "make it much louder but leave it on",
        "please raise volume instead of muting",
        "I said louder, not silent",
        "more sound please, don't mute",
        "louder please, keep the audio on",
        "turn it up without turning it off",
        "raise the volume, do not mute",
        "increase volume and keep audio active",
        "don't silence it, make it louder instead",
        "I need more sound, keep it enabled",
        "make the audio stronger without muting",
        "keep it playing and turn the volume up",
        "I can hear it but it is too quiet, raise it",
        "quiet audio, higher volume, no mute",
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
        "please don't mute, just lower the volume",
        "make it quieter, don't silence it",
        "reduce the volume while keeping sound on",
        "turn it down but don't turn it off",
        "I need less volume, not mute",
        "lower audio without muting",
        "decrease volume and keep it playing",
        "the sound is too loud, reduce it",
        "turn down the volume but keep audio enabled",
        "make the audio softer without muting",
        "I want lower volume, not silence",
        "keep sound active and turn it down",
        "reduce the sound level, do not mute",
        "lower the hearing volume while keeping it on",
        "don't silence the device, lower the volume",
        "turn it down without disabling sound",
        "less sound please, keep it on",
        "make it quieter while staying unmuted",
        "lower the audio, don't mute",
        "the volume is high, bring it down",
        "decrease sound but leave it enabled",
        "I said quieter, not silent",
        "lower it but keep the sound",
        "make the audio less loud, don't mute",
        "reduce volume and leave audio on",
        "turn the sound down, don't turn it off",
        "quieter please, keep it playing",
        "lower the sound without silence",
        "decrease volume instead of muting",
        "I need softer audio, not silence",
        "turn it down and keep it audible",
        "reduce audio level but don't mute",
        "make it less loud while staying on",
        "lower volume, no mute",
        "turn it down but leave sound active",
        "reduce the volume, keep audio enabled",
        "softer sound, keep it on",
        "please lower the audio, don't silence",
        "make the sound quieter without turning it off",
    ],
    "device.volume.mute": [
        "make it completely silent",
        "silence the audio completely",
        "mute the device",
        "mute it now",
        "I want complete silence",
        "I don't want to hear anything",
        "turn the sound completely off",
        "make the audio silent",
        "silence everything",
        "stop all sound",
        "I want no sound",
        "turn audio off completely",
        "mute all audio",
        "make it totally silent",
        "please silence the sound",
        "I need complete silence",
        "turn off the audio",
        "quiet it completely",
        "remove all sound",
        "make the hearing aid silent",
    ],
    "device.volume.unmute": [
        "turn the sound back on",
        "restore the sound",
        "unmute the audio",
        "enable sound again",
        "I want audio back",
        "turn audio back on",
        "bring the sound back",
        "restore audio",
        "start the sound again",
        "let me hear it again",
        "turn sound on",
        "enable the audio",
        "unmute it now",
        "I want the sound restored",
        "bring audio back",
        "turn my sound back on",
        "remove mute",
        "stop muting the audio",
        "restore the hearing sound",
        "make the audio audible again",
    ],
    "device.memory.change": [
        "change the memory",
        "switch my memory",
        "change the saved memory",
        "update the device memory",
        "change what is remembered",
        "switch to another memory",
        "modify the memory setting",
        "change the memory configuration",
        "update my memory setting",
        "switch memory settings",
    ],
    "find.phone.locate": [
        "find my phone",
        "where is my phone",
        "locate my phone",
        "help me find my phone",
        "make my phone ring so I can find it",
        "I cannot find my phone",
        "locate the phone",
        "where can I find my phone",
        "help locate my missing phone",
        "find the phone for me",
    ],
    "help.reminder.show": [
        "show my reminders",
        "show the reminders I have",
        "what reminders do I have",
        "list my reminders",
        "display my reminders",
        "open my reminder list",
        "show pending reminders",
        "what are my reminders",
        "show all reminders",
        "let me see my reminders",
    ],
    "reminders.task.complete": [
        "mark the reminder completed",
        "complete that reminder",
        "finish my reminder",
        "mark this reminder as done",
        "complete the task reminder",
        "set the reminder to completed",
        "mark it done",
        "finish that reminder",
        "complete my task reminder",
        "close the reminder task",
    ],
    "reminders.task.create": [
        "create a reminder",
        "remind me tomorrow",
        "set a reminder for tomorrow",
        "add a reminder",
        "make a new reminder",
        "remind me to call someone",
        "create a task reminder",
        "set a new reminder",
        "add something to my reminders",
        "remember this for later",
    ],
    "streaming.session.start": [
        "start streaming",
        "start the streaming session",
        "begin streaming",
        "turn streaming on",
        "start my stream",
        "begin the audio stream",
        "start the session",
        "enable streaming",
        "begin the streaming session",
        "start streaming now",
    ],
    "streaming.session.stop": [
        "stop streaming",
        "stop the streaming session",
        "end streaming",
        "turn streaming off",
        "stop my stream",
        "end the audio stream",
        "stop the session",
        "disable streaming",
        "end the streaming session",
        "stop streaming now",
    ],
}

hard_rows = []
for intent, phrases in HARD.items():
    for text in phrases:
        hard_rows.append({
            "text": clean(text),
            "intent": intent,
            "source": "targeted_hard_negative_v2",
        })

hard_df = pd.DataFrame(hard_rows)
hard_df = pd.concat([old_hard, hard_df], ignore_index=True)
hard_df["text"] = hard_df["text"].map(clean)
hard_df = hard_df.drop_duplicates("text").reset_index(drop=True)

# ---------------------------------------------------------------------
# 4) Add explicit adversarial/contrastive pairs as metadata.
# ---------------------------------------------------------------------

contrastive = [
    ("the audio is quiet but don't mute it make it louder",
     "device.volume.increase"),
    ("the audio is loud but don't mute it make it quieter",
     "device.volume.decrease"),
    ("I said louder, not silent",
     "device.volume.increase"),
    ("I said quieter, not silent",
     "device.volume.decrease"),
    ("turn the sound back on",
     "device.volume.unmute"),
    ("turn the sound completely off",
     "device.volume.mute"),
    ("start streaming",
     "streaming.session.start"),
    ("stop streaming",
     "streaming.session.stop"),
]

contrastive_df = pd.DataFrame(
    contrastive,
    columns=["text", "expected_intent"]
)
contrastive_df.to_csv(
    OUT / "production_contrastive_pairs.csv",
    index=False
)

# ---------------------------------------------------------------------
# 5) Save datasets
# ---------------------------------------------------------------------

indomain = indomain[["text", "intent"]].drop_duplicates("text")
indomain["text"] = indomain["text"].map(clean)

new_ood.to_csv(
    OUT / "production_ood_calibration.csv",
    index=False
)

indomain.to_csv(
    OUT / "production_indomain_calibration.csv",
    index=False
)

hard_df.to_csv(
    OUT / "production_hard_negative.csv",
    index=False
)

manifest = {
    "version": "production_calibration_v2",
    "indomain_rows": int(len(indomain)),
    "ood_rows": int(len(new_ood)),
    "hard_negative_rows": int(len(hard_df)),
    "contrastive_rows": int(len(contrastive_df)),
    "intents": LABELS,
    "source_indomain": str(V1 / "production_indomain_calibration.csv"),
    "source_ood": str(V1 / "production_ood_calibration.csv"),
    "source_hard_negative": str(V1 / "production_hard_negative.csv"),
    "unseen_test_protected": True,
    "unseen_test_note": (
        "595-row unseen test is not loaded or used by this script."
    ),
    "production_warning": (
        "Synthetic OOD is a calibration starter. Before release, "
        "add independently collected real OOD/near-OOD examples."
    ),
}

with open(
    OUT / "production_calibration_manifest.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(manifest, f, indent=2)

print("=" * 78)
print("PRODUCTION CALIBRATION V2 CREATED")
print("=" * 78)
print(f"In-domain calibration : {len(indomain)} rows")
print(f"OOD calibration       : {len(new_ood)} rows")
print(f"Hard negatives        : {len(hard_df)} rows")
print(f"Contrastive pairs     : {len(contrastive_df)} rows")
print(f"Output                : {OUT}")
print()
print("Files:")
for p in sorted(OUT.iterdir()):
    print(" ", p.name)
print()
print("V2 INT8 classifier was NOT modified.")
print("The 595-row unseen test was NOT used.")
print()
print("NEXT:")
print("1) Run production_hardening_v1.py after copying the two")
print("   calibration CSVs into the project root, OR update its")
print("   paths to production_calibration_v2.")
print("2) Review false accepts/rejects.")
print("3) Add real-world OOD data before production release.")
