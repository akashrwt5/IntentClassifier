#!/usr/bin/env python3
"""
Build semantic_holdout_2.csv — 59-intent coverage.
Keeps all 100 existing rows and adds 5 rows per unmeasured intent.
Deduplication guard against both the existing holdout and intent_data_new_2.csv.
"""
import csv
from pathlib import Path

BASE_DIR    = Path(".")
HOLDOUT_OLD = BASE_DIR / "data" / "semantic_holdout_100.csv"
HOLDOUT_NEW = BASE_DIR / "data" / "semantic_holdout_2.csv"
DATA2_PATH  = BASE_DIR / "data" / "intent_data_new_2.csv"

# ── New rows: 5 per unmeasured intent ──────────────────────────────────────
# Style: natural paraphrases, NOT verbatim training phrases.
# difficulty: EASY = close to training data, HARD = paraphrase/edge case
NEW_ROWS = [
    # Cmd.VolumeMute
    ("the world has gone mercifully quiet",       "Cmd.VolumeMute",        "HARD"),
    ("kill the noise completely",                 "Cmd.VolumeMute",        "HARD"),
    ("i need dead silence right now",             "Cmd.VolumeMute",        "HARD"),
    ("mute everything for me",                    "Cmd.VolumeMute",        "EASY"),
    ("not a peep please",                         "Cmd.VolumeMute",        "HARD"),

    # Cmd.VolumeUnmute
    ("let the sound back in",                     "Cmd.VolumeUnmute",      "HARD"),
    ("unmute my hearing aids",                    "Cmd.VolumeUnmute",      "EASY"),
    ("i want to hear again",                      "Cmd.VolumeUnmute",      "HARD"),
    ("bring the audio back please",               "Cmd.VolumeUnmute",      "HARD"),
    ("take me off mute",                          "Cmd.VolumeUnmute",      "HARD"),

    # Cmd.StreamingStop
    ("cut the tv connection",                     "Cmd.StreamingStop",     "HARD"),
    ("i'm done listening to the stream",          "Cmd.StreamingStop",     "HARD"),
    ("stop sending audio to my hearing aids",     "Cmd.StreamingStop",     "EASY"),
    ("disconnect from what i was streaming",      "Cmd.StreamingStop",     "HARD"),
    ("end the broadcast to my ears",              "Cmd.StreamingStop",     "HARD"),

    # Cmd.FindMyPhone
    ("i've misplaced my phone again",             "Cmd.FindMyPhone",       "HARD"),
    ("ring my phone so i can find it",            "Cmd.FindMyPhone",       "HARD"),
    ("help me track down my handset",             "Cmd.FindMyPhone",       "HARD"),
    ("where did i put my phone",                  "Cmd.FindMyPhone",       "EASY"),
    ("my phone has gone missing",                 "Cmd.FindMyPhone",       "HARD"),

    # Cmd.ActivityRun
    ("i'm about to head out for a jog",           "Cmd.ActivityRun",       "HARD"),
    ("track this run for me",                     "Cmd.ActivityRun",       "EASY"),
    ("log the miles i just covered",              "Cmd.ActivityRun",       "HARD"),
    ("i went for a run this morning",             "Cmd.ActivityRun",       "HARD"),
    ("start counting my running",                 "Cmd.ActivityRun",       "HARD"),

    # Cmd.ActivityWalk
    ("i'm off for a stroll track it please",      "Cmd.ActivityWalk",      "HARD"),
    ("count my steps on this walk",               "Cmd.ActivityWalk",      "HARD"),
    ("log my evening walk",                       "Cmd.ActivityWalk",      "EASY"),
    ("i went for a long walk today",              "Cmd.ActivityWalk",      "HARD"),
    ("record this walk for me",                   "Cmd.ActivityWalk",      "HARD"),

    # Cmd.ActivityStep
    ("have i hit ten thousand steps yet",         "Cmd.ActivityStep",      "HARD"),
    ("how close am i to my step target",         "Cmd.ActivityStep",       "HARD"),
    ("what's my step count for today",            "Cmd.ActivityStep",      "EASY"),
    ("am i near my walking goal",                 "Cmd.ActivityStep",      "HARD"),
    ("how many steps have i clocked today",       "Cmd.ActivityStep",      "HARD"),

    # Cmd.ActivityCycle
    ("i just finished a bike ride log it",        "Cmd.ActivityCycle",     "HARD"),
    ("track my cycling session",                  "Cmd.ActivityCycle",     "EASY"),
    ("how many kilometres have i cycled",         "Cmd.ActivityCycle",     "HARD"),
    ("am i close to my biking goal",              "Cmd.ActivityCycle",     "HARD"),
    ("record this ride please",                   "Cmd.ActivityCycle",     "HARD"),

    # Cmd.ActivityStand
    ("have i stood up enough today",              "Cmd.ActivityStand",     "HARD"),
    ("how long have i been on my feet",           "Cmd.ActivityStand",     "HARD"),
    ("check my standing time",                    "Cmd.ActivityStand",     "EASY"),
    ("am i meeting my standing goal",             "Cmd.ActivityStand",     "HARD"),
    ("how much longer should i stay standing",    "Cmd.ActivityStand",     "HARD"),

    # Cmd.ActivityAerobics
    ("track my aerobics class",                   "Cmd.ActivityAerobics",  "EASY"),
    ("how many aerobic minutes today",            "Cmd.ActivityAerobics",  "HARD"),
    ("did i hit my aerobics target",              "Cmd.ActivityAerobics",  "HARD"),
    ("log the aerobics i just did",               "Cmd.ActivityAerobics",  "HARD"),
    ("show me my aerobic progress",               "Cmd.ActivityAerobics",  "HARD"),

    # Cmd.ActivityCalories
    ("how many calories have i torched today",    "Cmd.ActivityCalories",  "HARD"),
    ("show me my burn for the week",              "Cmd.ActivityCalories",  "HARD"),
    ("what's my calorie count so far",            "Cmd.ActivityCalories",  "EASY"),
    ("did i hit my calorie goal",                 "Cmd.ActivityCalories",  "HARD"),
    ("how many more calories to reach my target", "Cmd.ActivityCalories",  "HARD"),

    # Cmd.ActivityExercise
    ("log my gym session",                        "Cmd.ActivityExercise",  "HARD"),
    ("how long did i work out today",             "Cmd.ActivityExercise",  "EASY"),
    ("did i reach my exercise goal",              "Cmd.ActivityExercise",  "HARD"),
    ("track this training session",               "Cmd.ActivityExercise",  "HARD"),
    ("record my workout just now",                "Cmd.ActivityExercise",  "HARD"),

    # Cmd.TranscribeStart
    ("write down everything being said",          "Cmd.TranscribeStart",   "HARD"),
    ("i need a live transcript",                  "Cmd.TranscribeStart",   "HARD"),
    ("caption what people are saying",            "Cmd.TranscribeStart",   "HARD"),
    ("start converting speech to text",           "Cmd.TranscribeStart",   "EASY"),
    ("put the conversation in writing",           "Cmd.TranscribeStart",   "HARD"),

    # Cmd.TranslationStart
    ("start translating for me",                  "Cmd.TranslationStart",  "EASY"),
    ("i need this conversation in english",       "Cmd.TranslationStart",  "HARD"),
    ("switch on translation mode",                "Cmd.TranslationStart",  "HARD"),
    ("help me understand this foreign language",  "Cmd.TranslationStart",  "HARD"),
    ("translate what they're saying",             "Cmd.TranslationStart",  "HARD"),

    # Cmd.ListenMessage
    ("read out my messages",                      "Cmd.ListenMessage",     "HARD"),
    ("play back my latest texts",                 "Cmd.ListenMessage",     "HARD"),
    ("any messages for me",                       "Cmd.ListenMessage",     "HARD"),
    ("let me hear my messages",                   "Cmd.ListenMessage",     "EASY"),
    ("what did people send me",                   "Cmd.ListenMessage",     "HARD"),

    # Cmd.SendMessage - yes
    ("yep go ahead and send it",                  "Cmd.SendMessage - yes", "HARD"),
    ("that's right send it",                      "Cmd.SendMessage - yes", "HARD"),
    ("confirm and send",                          "Cmd.SendMessage - yes", "EASY"),
    ("yes that's what i wanted to say",           "Cmd.SendMessage - yes", "HARD"),
    ("correct please send it now",                "Cmd.SendMessage - yes", "HARD"),

    # Cmd.SendMessage - no
    ("no don't send that",                        "Cmd.SendMessage - no",  "HARD"),
    ("hold on that's not right",                  "Cmd.SendMessage - no",  "HARD"),
    ("cancel the message",                        "Cmd.SendMessage - no",  "EASY"),
    ("i changed my mind don't send it",           "Cmd.SendMessage - no",  "HARD"),
    ("scrap that message",                        "Cmd.SendMessage - no",  "HARD"),

    # Cmd.BatteryLevel (already 10 rows — adding harder edge cases)
    ("roughly how much life do these have left",  "Cmd.BatteryLevel",      "HARD"),
    ("battery check please",                      "Cmd.BatteryLevel",      "EASY"),
    ("are my aids running low",                   "Cmd.BatteryLevel",      "HARD"),
    ("will they last the afternoon",              "Cmd.BatteryLevel",      "HARD"),
    ("let me know the charge level",              "Cmd.BatteryLevel",      "HARD"),

    # reminders.complete
    ("mark that task as done",                    "reminders.complete",    "HARD"),
    ("i finished the reminder",                   "reminders.complete",    "HARD"),
    ("tick off my reminder",                      "reminders.complete",    "HARD"),
    ("complete the reminder for me",              "reminders.complete",    "EASY"),
    ("i took my pills you can clear that",        "reminders.complete",    "HARD"),

    # Help_Tinnitus
    ("my ears won't stop ringing",                "Help_Tinnitus",         "HARD"),
    ("there's a constant buzzing in my head",     "Help_Tinnitus",         "HARD"),
    ("how do i turn on the masking sound",        "Help_Tinnitus",         "HARD"),
    ("tinnitus relief help",                      "Help_Tinnitus",         "EASY"),
    ("the ringing is driving me mad",             "Help_Tinnitus",         "HARD"),

    # Help_Pairing
    ("my hearing aids won't connect",             "Help_Pairing",          "HARD"),
    ("link my aids to the app",                   "Help_Pairing",          "HARD"),
    ("how do i set up the bluetooth connection",  "Help_Pairing",          "HARD"),
    ("my aids have lost their connection",        "Help_Pairing",          "HARD"),
    ("pair my hearing aids",                      "Help_Pairing",          "EASY"),

    # Help_DeviceSettings
    ("the sound is cutting in and out",           "Help_DeviceSettings",   "HARD"),
    ("audio quality has gone downhill",           "Help_DeviceSettings",   "HARD"),
    ("my aids sound distorted",                   "Help_DeviceSettings",   "HARD"),
    ("fix the audio problems with my aids",       "Help_DeviceSettings",   "HARD"),
    ("help with my hearing aid settings",         "Help_DeviceSettings",   "EASY"),

    # Help_Volume
    ("how do i change the volume on one side",    "Help_Volume",           "HARD"),
    ("volume controls explained",                 "Help_Volume",           "HARD"),
    ("where is the volume setting",               "Help_Volume",           "EASY"),
    ("can i set different volumes per ear",       "Help_Volume",           "HARD"),
    ("help with adjusting loudness",              "Help_Volume",           "HARD"),

    # Help_MemoryOptions
    ("how do i create a custom program",          "Help_MemoryOptions",    "HARD"),
    ("what memory presets are available",         "Help_MemoryOptions",    "HARD"),
    ("how do i rename a memory",                  "Help_MemoryOptions",    "HARD"),
    ("memory options help",                       "Help_MemoryOptions",    "EASY"),
    ("can i set a favourite program",             "Help_MemoryOptions",    "HARD"),

    # Help_ChangingMemories
    ("how do i switch between programs",          "Help_ChangingMemories", "HARD"),
    ("change memory explained",                   "Help_ChangingMemories", "HARD"),
    ("i want to know about changing my setup",    "Help_ChangingMemories", "HARD"),
    ("help with switching profiles",              "Help_ChangingMemories", "EASY"),
    ("how do different memories work",            "Help_ChangingMemories", "HARD"),

    # Help_Battery
    ("my hearing aid battery died quickly",       "Help_Battery",          "HARD"),
    ("how do i replace the battery",              "Help_Battery",          "EASY"),
    ("rechargeable or disposable which is better","Help_Battery",          "HARD"),
    ("the charger isn't charging my aid",         "Help_Battery",          "HARD"),
    ("battery tips for hearing aids",             "Help_Battery",          "HARD"),

    # Help_CleanCare
    ("how do i maintain my hearing aids",         "Help_CleanCare",        "HARD"),
    ("cleaning tips for my aids",                 "Help_CleanCare",        "EASY"),
    ("i've got wax stuck in my hearing aid",      "Help_CleanCare",        "HARD"),
    ("is it ok to get my aids wet",               "Help_CleanCare",        "HARD"),
    ("how often should i clean them",             "Help_CleanCare",        "HARD"),

    # Help_InsertDevice
    ("how do i get this in my ear properly",      "Help_InsertDevice",     "HARD"),
    ("the device keeps falling out",              "Help_InsertDevice",     "HARD"),
    ("help putting in my hearing aid",            "Help_InsertDevice",     "EASY"),
    ("which way does this go in my ear",          "Help_InsertDevice",     "HARD"),
    ("how to wear my hearing aid correctly",      "Help_InsertDevice",     "HARD"),

    # Help_Accessories
    ("what accessories can i use with my aids",   "Help_Accessories",      "EASY"),
    ("tell me about the tv streamer",             "Help_Accessories",      "HARD"),
    ("how do i use the remote microphone",        "Help_Accessories",      "HARD"),
    ("what add-ons are compatible",               "Help_Accessories",      "HARD"),
    ("can i stream from a clip-on mic",           "Help_Accessories",      "HARD"),

    # Help_RemoteProgramming
    ("my audiologist can adjust remotely yes",    "Help_RemoteProgramming","HARD"),
    ("how does remote tuning work",               "Help_RemoteProgramming","HARD"),
    ("remote programming help",                   "Help_RemoteProgramming","EASY"),
    ("can my audiologist change settings from afar","Help_RemoteProgramming","HARD"),
    ("how do i request a remote fitting",         "Help_RemoteProgramming","HARD"),

    # Help_HearingCareAnywhereConnect
    ("connect with my hearing care professional", "Help_HearingCareAnywhereConnect","HARD"),
    ("how do i use telehear",                     "Help_HearingCareAnywhereConnect","EASY"),
    ("remote appointment with my audiologist",    "Help_HearingCareAnywhereConnect","HARD"),
    ("back up my settings to the cloud",          "Help_HearingCareAnywhereConnect","HARD"),
    ("restore my profile after reinstalling",     "Help_HearingCareAnywhereConnect","HARD"),

    # Help_SelfCheck
    ("run diagnostics on my hearing aids",        "Help_SelfCheck",        "HARD"),
    ("check if my aids are working properly",     "Help_SelfCheck",        "HARD"),
    ("self check help",                           "Help_SelfCheck",        "EASY"),
    ("test my hearing aids",                      "Help_SelfCheck",        "HARD"),
    ("something seems off with one aid diagnose", "Help_SelfCheck",        "HARD"),

    # Help_FallAlert
    ("set up fall detection",                     "Help_FallAlert",        "EASY"),
    ("how does fall alert work",                  "Help_FallAlert",        "HARD"),
    ("will it detect if i take a tumble",         "Help_FallAlert",        "HARD"),
    ("alert my family if i fall",                 "Help_FallAlert",        "HARD"),
    ("fall alert setup help",                     "Help_FallAlert",        "HARD"),

    # Help_IntelliVoice
    ("how does the smart voice feature work",     "Help_IntelliVoice",     "HARD"),
    ("intellivoice explained",                    "Help_IntelliVoice",     "EASY"),
    ("the voice recognition feature help",        "Help_IntelliVoice",     "HARD"),
    ("what can intelligent voice do",             "Help_IntelliVoice",     "HARD"),
    ("enable the smart speech feature",           "Help_IntelliVoice",     "HARD"),

    # Help_MaskMode
    ("i can't hear people with face masks on",    "Help_MaskMode",         "HARD"),
    ("mask mode help",                            "Help_MaskMode",         "EASY"),
    ("how does mask mode improve speech",         "Help_MaskMode",         "HARD"),
    ("people in masks are hard to understand",    "Help_MaskMode",         "HARD"),
    ("turn on mask mode",                         "Help_MaskMode",         "HARD"),

    # Help_EdgeMode
    ("too much wind noise outside",               "Help_EdgeMode",         "HARD"),
    ("edge mode help",                            "Help_EdgeMode",         "EASY"),
    ("reduce wind interference with edge mode",   "Help_EdgeMode",         "HARD"),
    ("is edge mode good for outdoor use",         "Help_EdgeMode",         "HARD"),
    ("voices sound clearer with what setting",    "Help_EdgeMode",         "HARD"),

    # Help_DemoMode
    ("let me explore the app without aids",       "Help_DemoMode",         "HARD"),
    ("demo mode help",                            "Help_DemoMode",         "EASY"),
    ("try out features without connecting aids",  "Help_DemoMode",         "HARD"),
    ("i want to show someone the app without aids","Help_DemoMode",        "HARD"),
    ("preview the app in demonstration mode",     "Help_DemoMode",         "HARD"),

    # Help_AppSettings
    ("where do i find the app preferences",       "Help_AppSettings",      "HARD"),
    ("switch to advanced mode in the app",        "Help_AppSettings",      "HARD"),
    ("app settings help",                         "Help_AppSettings",      "EASY"),
    ("how do i change to basic mode",             "Help_AppSettings",      "HARD"),
    ("notification settings for the app",         "Help_AppSettings",      "HARD"),

    # Help_Customize
    ("how do i tweak the equalizer",              "Help_Customize",        "HARD"),
    ("customize my hearing aid sound",            "Help_Customize",        "EASY"),
    ("i want less bass in my left aid",           "Help_Customize",        "HARD"),
    ("reduce the background noise level",         "Help_Customize",        "HARD"),
    ("personalise my listening experience",       "Help_Customize",        "HARD"),

    # Help_Transcribe
    ("where is the transcription feature",        "Help_Transcribe",       "HARD"),
    ("transcribe feature help",                   "Help_Transcribe",       "EASY"),
    ("does the app do live captions",             "Help_Transcribe",       "HARD"),
    ("how does speech to text work in thrive",    "Help_Transcribe",       "HARD"),
    ("what is the transcribe function",           "Help_Transcribe",       "HARD"),

    # Help_Translate
    ("what languages can it translate",           "Help_Translate",        "HARD"),
    ("translate feature help",                    "Help_Translate",        "EASY"),
    ("how do i use the translation function",     "Help_Translate",        "HARD"),
    ("is french available for translation",       "Help_Translate",        "HARD"),
    ("how does the real-time translation work",   "Help_Translate",        "HARD"),

    # Help_ThriveScore
    ("what is my wellness score",                 "Help_ThriveScore",      "HARD"),
    ("show my thrive score",                      "Help_ThriveScore",      "EASY"),
    ("how is the thrive score calculated",        "Help_ThriveScore",      "HARD"),
    ("my body and brain score",                   "Help_ThriveScore",      "HARD"),
    ("where do i see my overall health rating",   "Help_ThriveScore",      "HARD"),

    # Help_HeartRate
    ("can these aids measure my pulse",           "Help_HeartRate",        "HARD"),
    ("heart rate help",                           "Help_HeartRate",        "EASY"),
    ("show my heart rate from today",             "Help_HeartRate",        "HARD"),
    ("is my heart rate tracked automatically",    "Help_HeartRate",        "HARD"),
    ("where do i see my bpm",                     "Help_HeartRate",        "HARD"),

    # Help_HeartRateRecovery
    ("how quickly did my heart recover",          "Help_HeartRateRecovery","HARD"),
    ("heart rate recovery help",                  "Help_HeartRateRecovery","EASY"),
    ("explain my recovery score",                 "Help_HeartRateRecovery","HARD"),
    ("what's a good heart rate recovery time",    "Help_HeartRateRecovery","HARD"),
    ("is my recovery getting better",             "Help_HeartRateRecovery","HARD"),

    # Help_Health
    ("show me my overall activity dashboard",     "Help_Health",           "HARD"),
    ("health screen help",                        "Help_Health",           "EASY"),
    ("where do i see all my fitness data",        "Help_Health",           "HARD"),
    ("how do i set my weekly health goals",       "Help_Health",           "HARD"),
    ("i want to see my full health summary",      "Help_Health",           "HARD"),

    # Help_HearShare
    ("how do i share what i'm hearing",           "Help_HearShare",        "HARD"),
    ("can someone else listen with me",           "Help_HearShare",        "HARD"),
    ("hear share feature help",                   "Help_HearShare",        "EASY"),
    ("stream audio to another person",            "Help_HearShare",        "HARD"),
    ("how does shared listening work",            "Help_HearShare",        "HARD"),

    # Help_VoiceAssistant
    ("what voice assistants work with my aids",   "Help_VoiceAssistant",   "HARD"),
    ("voice assistant help",                      "Help_VoiceAssistant",   "EASY"),
    ("can i use siri with my hearing aids",       "Help_VoiceAssistant",   "HARD"),
    ("how do i activate google assistant",        "Help_VoiceAssistant",   "HARD"),
    ("enable the voice assistant feature",        "Help_VoiceAssistant",   "HARD"),

    # Help_WiCROS
    ("help with my cros hearing aids",            "Help_WiCROS",           "EASY"),
    ("what is the cros balance control",          "Help_WiCROS",           "HARD"),
    ("my transmitter isn't working",              "Help_WiCROS",           "HARD"),
    ("how do i adjust cros balance",              "Help_WiCROS",           "HARD"),
    ("wicros feature explained",                  "Help_WiCROS",           "HARD"),

    # Help_Reminder
    ("how do reminders work in this app",         "Help_Reminder",         "HARD"),
    ("reminder feature help",                     "Help_Reminder",         "EASY"),
    ("can i get a daily reminder",                "Help_Reminder",         "HARD"),
    ("how do i delete a reminder",                "Help_Reminder",         "HARD"),
    ("set up recurring reminders",                "Help_Reminder",         "HARD"),

    # Help_WhatsNew
    ("what's been added to the app lately",       "Help_WhatsNew",         "HARD"),
    ("show me what's new",                        "Help_WhatsNew",         "EASY"),
    ("any new features i should know about",      "Help_WhatsNew",         "HARD"),
    ("what changed in the latest update",         "Help_WhatsNew",         "HARD"),
    ("recent updates to thrive",                  "Help_WhatsNew",         "HARD"),

    # Default Fallback (add 5 more hard OOS examples)
    ("what's the weather going to be like",       "Default Fallback Intent","EASY"),
    ("who won last night's match",                "Default Fallback Intent","EASY"),
    ("recommend me a tv show",                    "Default Fallback Intent","EASY"),
    ("how do i make a cheese omelette",           "Default Fallback Intent","EASY"),
    ("what's a good book to read this month",     "Default Fallback Intent","EASY"),
]

# ── Load existing holdout ──
with open(HOLDOUT_OLD, encoding="utf-8-sig") as f:
    old_rows = list(csv.DictReader(f))
existing_utterances = {r["utterance"].strip().lower() for r in old_rows}

# ── Load training data for leakage guard ──
with open(DATA2_PATH, encoding="utf-8-sig") as f:
    train_texts = {r["text"].strip().lower() for r in csv.DictReader(f)}

# ── Filter new rows ──
filtered = []
skipped_dupe = []
skipped_leak = []

for utterance, intent, diff in NEW_ROWS:
    key = utterance.strip().lower()
    if key in existing_utterances:
        skipped_dupe.append(utterance)
        continue
    if key in train_texts:
        skipped_leak.append(utterance)
        continue
    existing_utterances.add(key)
    filtered.append({"utterance": utterance, "expected_intent": intent, "difficulty": diff})

# ── Write combined file ──
all_rows = old_rows + filtered
with open(HOLDOUT_NEW, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["utterance", "expected_intent", "difficulty"])
    writer.writeheader()
    writer.writerows(all_rows)

from collections import Counter
coverage = Counter(r["expected_intent"] for r in all_rows)

print(f"semantic_holdout_100.csv  (original): {len(old_rows)} rows, {len(set(r['expected_intent'] for r in old_rows))} intents  — UNTOUCHED")
print(f"semantic_holdout_2.csv    (new):      {len(all_rows)} rows, {len(coverage)} intents")
print(f"  New rows added:  {len(filtered)}")
if skipped_dupe: print(f"  Skipped (dupe):  {len(skipped_dupe)} — {skipped_dupe}")
if skipped_leak: print(f"  Skipped (leak):  {len(skipped_leak)} — {skipped_leak}")
print(f"\nIntent coverage:")
for intent in sorted(coverage):
    print(f"  {intent:45s}  {coverage[intent]:3d}")
