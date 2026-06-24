#!/usr/bin/env python3
"""
Add curated, genuinely-distinct paraphrases to tail intents in
data/01_source_base_training_data.csv so every intent reaches >=40 rows.

These are hand-written natural paraphrases in the hearing-aid / Thrive
domain — NOT carrier-phrase permutations ("please X" / "can you X"),
which are exactly the near-duplicate spam the capping step removes.

Safety guards:
  * Skips any addition already present (text+intent) in the dataset.
  * Skips any addition whose text appears in the semantic holdout
    (prevents train/holdout leakage).
  * Reports exactly what was added per intent.

Run from repo root:
    python scripts/add_tail_class_data.py            # writes the file
    python scripts/add_tail_class_data.py --dry-run  # preview only
"""

import csv
import sys
from pathlib import Path
from collections import Counter

BASE_DIR     = Path(__file__).parent.parent
DATA_PATH    = BASE_DIR / "data" / "01_source_base_training_data.csv"
HOLDOUT_PATH = BASE_DIR / "data" / "semantic_holdout_100.csv"

# ── Curated additions (genuine paraphrases, not permutations) ──
ADDITIONS = {
    "Help_Battery": [
        "my hearing aid won't charge",
        "why isn't my hearing aid charging",
        "how long does the battery last",
        "how long does a charge last",
        "my battery dies too quickly",
        "battery drains fast",
        "what size battery does my hearing aid use",
        "which battery do i need for my hearing aid",
        "where do i put the battery",
        "how do i open the battery door",
        "is my hearing aid rechargeable",
        "how do i know my hearing aid is charging",
        "how long to fully charge my hearing aid",
        "can i overcharge my hearing aid",
        "my hearing aid battery is low",
        "replace the battery in my hearing aid",
        "how often should i change the battery",
        "where do i buy new batteries",
        "the charger isn't working",
        "how do i use the charging case",
        "do i need to remove the battery at night",
        "battery life is getting shorter",
    ],
    "Help_EdgeMode": [
        "when should i use edge mode",
        "is edge mode good for restaurants",
        "does edge mode help in noisy places",
        "how do i activate edge mode",
        "how do i exit edge mode",
        "can i customize edge mode",
        "what settings does edge mode change",
        "does edge mode improve speech clarity",
        "will edge mode reduce background noise",
        "how long does edge mode stay on",
        "does edge mode drain the battery faster",
        "can i use edge mode with streaming",
        "edge mode isn't working",
        "why would i use edge mode",
        "is edge mode available on my hearing aid",
        "how do i tap for edge mode",
        "edge mode help",
        "tell me about edge mode",
        "benefits of edge mode",
        "when does edge mode turn off",
        "does edge mode work automatically",
    ],
    "Cmd.ActivityAerobics": [
        "start tracking my aerobics",
        "log an aerobics workout",
        "begin aerobics session",
        "how far am i from my aerobics goal",
        "track my aerobic exercise",
        "record my aerobics activity",
        "show my aerobics progress",
        "what's my aerobics count this week",
        "how much aerobics have i done today",
        "did i hit my aerobics target",
        "how many aerobic minutes this week",
        "start an aerobics session",
        "log aerobics",
        "am i close to my aerobics goal",
        "how long until i reach my aerobics goal",
        "show me my aerobics stats",
        "track aerobics workout",
        "begin tracking aerobics",
        "how much more aerobics do i need today",
    ],
    "Cmd.ActivityCalories": [
        "how many calories have i burned this week",
        "show me my calorie burn",
        "what's my total calorie count",
        "how many calories did i burn yesterday",
        "calories burned this month",
        "track my calories",
        "what's my weekly calorie burn",
        "how many calories have i lost this week",
        "show my calorie history",
        "am i close to my calorie goal",
        "what's my calorie goal",
        "how many more calories to reach my goal",
        "calories burned so far today",
        "my calorie total",
        "did i meet my calorie goal",
        "how many calories left to burn",
        "show calories burned this week",
        "weekly calorie summary",
    ],
    "Help_MaskMode": [
        "when should i use mask mode",
        "does mask mode help when wearing a face mask",
        "how does mask mode work",
        "is mask mode useful with masks",
        "can i adjust mask mode",
        "mask mode help",
        "tell me about mask mode",
        "why can't i hear people in masks",
        "people wearing masks are hard to hear",
        "mask mode settings",
        "how do i access mask mode",
        "does mask mode improve clarity with masks",
        "turn mask mode on",
        "turn mask mode off",
        "benefits of mask mode",
        "when does mask mode help",
        "how do i find mask mode",
    ],
    "Help_HearingCareAnywhereConnect": [
        "how do i connect with my hearing professional",
        "can my audiologist adjust my settings remotely",
        "how does telehear work",
        "set up a remote appointment",
        "how do i request a remote adjustment",
        "connect to my hearing care provider",
        "how do i use hearing care anywhere",
        "how do i sync my settings to the cloud",
        "where is my data backed up",
        "restore my settings from the cloud",
        "how do i get remote support",
        "can i get my hearing aids tuned remotely",
        "how do i save my settings online",
        "recover my hearing aid settings",
        "how do i back up my profile",
        "tell me about hearing care anywhere",
    ],
    "Help_Health": [
        "show me my health overview",
        "what's on the health screen",
        "how do i view my activity summary",
        "where can i see my daily goals",
        "how do i set my health goals",
        "track my wellness",
        "show my health dashboard",
        "how do i change my activity goals",
        "can i set monthly health goals",
        "where are my fitness goals",
        "how do i view my health progress",
        "show me my weekly health summary",
        "what health metrics does the app track",
        "how do i reset my health goals",
        "view my body and brain health",
        "where do i find my engagement goals",
    ],
    "Help_HeartRate": [
        "can my hearing aids track my heart rate",
        "show me my heart rate history",
        "is my heart rate normal",
        "what's my resting heart rate",
        "how accurate is the heart rate reading",
        "where do i see my heart rate trends",
        "can i track heart rate during exercise",
        "how often is my heart rate measured",
        "view my heart rate data",
        "does the app record my heart rate",
        "explain the heart rate feature",
        "how do i enable heart rate tracking",
        "what's a healthy heart rate",
        "show my current heart rate",
        "track my heart rate",
    ],
    "Cmd.StreamingStop": [
        "stop the audio stream",
        "end the tv connection",
        "stop casting to my hearing aids",
        "disconnect the streaming",
        "halt streaming",
        "cut off the stream",
        "stop playing through my hearing aids",
        "turn off the audio streaming",
        "stop streaming from my phone",
        "disconnect the remote microphone",
        "end the music stream",
        "stop streaming media",
        "close the stream",
        "stop streaming audio",
        "stop the broadcast to my hearing aids",
    ],
    "Cmd.ActivityExercise": [
        "start tracking my workout",
        "log an exercise session",
        "begin my workout",
        "record my exercise",
        "how's my exercise progress",
        "show my workout stats",
        "track this workout",
        "start an exercise session",
        "how many minutes have i exercised this week",
        "did i reach my workout goal",
        "log my training session",
        "show me my exercise summary",
        "how much exercise have i done today",
        "am i close to my exercise goal",
    ],
    "Help_HeartRateRecovery": [
        "how do i improve my recovery rate",
        "what affects heart rate recovery",
        "is my recovery rate healthy",
        "show my heart rate recovery trend",
        "where can i view heart rate recovery",
        "explain heart rate recovery to me",
        "track my heart rate recovery",
        "what's a good recovery number",
        "how often is recovery measured",
        "view my recovery score",
        "does recovery indicate fitness",
        "help with heart rate recovery",
    ],
    "Help_DemoMode": [
        "show me demo mode",
        "can i try the app without my hearing aids",
        "preview app features without aids",
        "explore the app in demo mode",
        "how do i exit demo mode",
        "turn off demo mode",
        "what can i do in demo mode",
        "is demo mode available",
        "let me try the app features",
        "show app capabilities without hearing aids",
        "demonstrate the app",
        "can i test features without aids connected",
    ],
    "Help_Reminder": [
        "how do reminders work",
        "can i edit a reminder",
        "how do i delete a reminder",
        "where do i manage my reminders",
        "can i set recurring reminders",
        "how do i turn off a reminder",
        "do reminders sync to my hearing aids",
        "how do i see all my reminders",
        "can i snooze a reminder",
    ],
    "Help_CleanCare": [
        "how often should i clean my hearing aids",
        "what's the best way to clean my hearing aids",
        "can i clean the wax guard",
        "how do i replace the wax filter",
        "should i clean my hearing aids daily",
        "how do i dry my hearing aids",
        "can moisture damage my hearing aids",
        "how do i clean the microphone",
        "tips for maintaining my hearing aids",
    ],
    "Cmd.TranscribeStart": [
        "write down what people are saying",
        "caption this conversation",
        "show me a transcript",
        "turn speech into text",
        "transcribe what i hear",
        "enable live captions",
        "start captioning",
        "write out this meeting",
        "give me text of this conversation",
    ],
    "Cmd.ActivityCycle": [
        "start tracking my bike ride",
        "log a cycling session",
        "begin a bike ride",
        "record my cycling",
        "how's my cycling progress",
        "show me my cycling stats",
        "track this ride",
        "start a bike workout",
        "did i reach my cycling goal",
    ],
    "Cmd.ActivityStep": [
        "start counting my steps",
        "track my steps",
        "did i hit my step goal",
        "how am i doing on steps",
        "show my step progress",
        "how many steps until my goal",
        "record my steps",
    ],
    "Cmd.ActivityWalk": [
        "track my walking distance",
        "did i reach my walking goal today",
        "begin tracking my walk",
        "log my walking session",
    ],
    "Help_AppSettings": [
        "how do i reset my app settings",
        "where do i manage notifications",
        "how do i update the app",
        "can i restore default settings",
    ],
    "Help_WiCROS": [
        "how does wicros work",
        "what is wicros used for",
    ],
    "Cmd.ActivityStand": [
        "track my standing time",
    ],
    "Help_Customize": [
        "how do i personalize my sound settings",
    ],
}


def norm(s: str) -> str:
    return s.strip().lower()


def main():
    dry_run = "--dry-run" in sys.argv

    # Load existing data
    with open(DATA_PATH, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        existing = list(reader)

    existing_pairs = {(norm(r["text"]), r["intent"].strip()) for r in existing}
    counts_before = Counter(r["intent"].strip() for r in existing)

    # Load holdout texts (leakage guard)
    holdout_texts = set()
    if HOLDOUT_PATH.exists():
        with open(HOLDOUT_PATH, encoding="utf-8-sig") as f:
            hr = csv.DictReader(f)
            tcol = next((c for c in hr.fieldnames
                         if c.strip().lower() in ("text", "utterance", "query", "sentence", "phrase")),
                        hr.fieldnames[0])
            for row in hr:
                holdout_texts.add(norm(row[tcol]))

    # Apply additions
    new_rows = []
    skipped_dupe = []
    skipped_leak = []
    added_per_intent = Counter()

    for intent, phrases in ADDITIONS.items():
        for phrase in phrases:
            key = (norm(phrase), intent)
            if key in existing_pairs:
                skipped_dupe.append((phrase, intent))
                continue
            if norm(phrase) in holdout_texts:
                skipped_leak.append((phrase, intent))
                continue
            existing_pairs.add(key)
            new_rows.append({"text": phrase, "intent": intent})
            added_per_intent[intent] += 1

    # Report
    print("=" * 64)
    print("TAIL-CLASS DATA ADDITION")
    print("=" * 64)
    print(f"\n{'Intent':38s} {'before':>7} {'+added':>7} {'after':>7}")
    print("-" * 64)
    for intent in ADDITIONS:
        before = counts_before[intent]
        added = added_per_intent[intent]
        after = before + added
        flag = "" if after >= 40 else "  ⚠ still <40"
        print(f"{intent[:38]:38s} {before:>7} {added:>+7} {after:>7}{flag}")

    print(f"\nTotal new rows: {len(new_rows)}")
    if skipped_dupe:
        print(f"Skipped (already in data): {len(skipped_dupe)}")
        for p, i in skipped_dupe:
            print(f"    - [{i}] {p}")
    if skipped_leak:
        print(f"Skipped (in holdout — leakage guard): {len(skipped_leak)}")
        for p, i in skipped_leak:
            print(f"    - [{i}] {p}")

    if dry_run:
        print("\n[dry-run] No file written.")
        return

    # Write augmented file
    all_rows = existing + new_rows
    with open(DATA_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n✅ Wrote {len(all_rows)} rows to {DATA_PATH}")
    print(f"   ({len(existing)} original + {len(new_rows)} added)")


if __name__ == "__main__":
    main()
