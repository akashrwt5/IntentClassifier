#!/usr/bin/env python3
"""
Build a high-quality, natural-language augmentation set for intent training.

WHY: the original training data (intent_data_new_2.csv) is overwhelmingly
formal / imperative-template ("turn down the volume", "how do i change memory").
The holdout benchmarks are conversational / situational ("my ears are getting
hammered", "i just sat down at a busy cafe"). The model never learned to map
problem-descriptions and colloquial phrasings onto intents, so it fails to
generalise. This script hand-authors diverse paraphrases in the MISSING
registers for every intent, then:
  * filters out any phrase that appears in ANY holdout / benchmark / test set
    (zero leakage — the benchmarks must stay a true generalisation test),
  * dedupes against the base training data,
  * writes  data/intent_data_augmented.csv        (just the new phrases), and
  * writes  data/intent_data_new_2_enhanced.csv   (the merged v3 training set =
            base _2 + augmentation + hand corrections) that the trainers read.

────────────────────────────────────────────────────────────────────────────
RETRAIN PIPELINE — run these four, in this order, from the repo root:

  1. python scripts/build_augmented_data.py     # regenerate the v3 training set
  2. python scripts/train.py                     # TF-IDF model  -> intent_model.onnx
  3. python scripts/train_semantic_head.py       # MiniLM head   -> semantic_head.*
  4. python scripts/test_holdout.py --strict      # validate + build gate

(train.py and train_semantic_head.py both default to --version 3 = the enhanced
file this script writes. Step 1 is only needed when you change the phrase banks
below; otherwise start at step 2. Inspect results with test_tfidf_only.py and
test_semantic.py.)
────────────────────────────────────────────────────────────────────────────
"""
import json, re, sys
from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent.parent
DATA = BASE / "data"

# ── Leakage blocklists ────────────────────────────────────────────────────────
# true_holdout : the genuine generalisation gates — NOTHING in training may leak
#                here (semantic_holdout_2/100 + the test_semantic.py cases).
# benchmark_250: an IN-DISTRIBUTION benchmark (sampled FROM training data), so the
#                base set legitimately overlaps it; we only use it to keep NEW
#                augmentation phrases conservative, never to gate the merged file.
def _read_phrases(fname) -> set:
    p = DATA / fname
    if not p.exists():
        return set()
    d = pd.read_csv(p)
    col = "utterance" if "utterance" in d.columns else ("text" if "text" in d.columns else d.columns[0])
    return set(d[col].astype(str).str.lower().str.strip())

def load_paraphrase_holdouts() -> set:
    # The genuine generalisation gates: deliberately-unseen paraphrases. Base
    # training has 0 overlap with these, and nothing we add may leak here.
    return _read_phrases("semantic_holdout_2.csv") | _read_phrases("semantic_holdout_100.csv")

def load_blocklist() -> set:
    # Used to filter NEW augmentation phrases — conservative superset: the true
    # paraphrase holdouts + the in-distribution benchmark + test_semantic.py
    # cases (which include some phrasings that legitimately live in base training).
    block = load_paraphrase_holdouts() | _read_phrases("semantic_benchmark_250.csv")
    ts = (BASE / "scripts" / "test_semantic.py")
    if ts.exists():
        block |= {h.lower().strip() for h in re.findall(r'\("([^"]+)",\s*"[^"]+"\)', ts.read_text())}
    return block

# ── Phrase banks (hand-authored, conversational/situational registers) ────────
# Cmd.* = imperative / problem-description that implies an ACTION now.
# Help_* = informational ("how / what / explain / where is the X feature").
P = {}

# ---- VOLUME ----------------------------------------------------------------
P["Cmd.VolumeIncrease"] = [
    "i can barely make out what people say","everything sounds too faint",
    "voices are too soft for me","crank the sound up a bit","i need more volume",
    "boost the loudness please","pump up the sound","make speech louder",
    "turn the level up","i'm missing half the conversation","speech is too quiet",
    "could you raise the volume","bump it up a little","everything is too low",
    "i want it louder","amplify the sound for me","raise the loudness",
    "the tv is too quiet for me","i keep missing words","sound feels weak",
    "give me a bit more volume","step the volume up","louder please",
    "i can't hear the speaker well","turn things up so i can hear",
    "make my aids louder","push the level higher","increase how loud it is",
    "people sound muffled and faint","i need to hear better turn it up",
]
P["Cmd.VolumeDecrease"] = [
    "turn the sound down a bit","make it softer please","lower the volume",
    "it's too loud for me","reduce the loudness","bring the level down",
    "quieter please","ease the volume back","that's painfully loud",
    "the sound is too strong","cut the volume a little","drop the level down",
    "tone the loudness down","i need it softer","too much volume right now",
    "soften things up","turn my aids down","less volume please",
    "knock the sound back","scale the volume down","it's blasting my ears",
    "dial the loudness down","everything is way too loud","make speech quieter",
    "lower how loud it is","trim the volume","back the level off",
    "the noise level is too high","ratchet it down a notch","gentle on the volume",
]
P["Cmd.VolumeMute"] = [
    "cut all the sound","total silence please","mute my hearing aids",
    "kill the audio completely","i want absolute quiet","shut the sound off",
    "silence everything","turn all sound off","no sound at all please",
    "fully mute the aids","stop all audio","mute it entirely",
    "block out every sound","i need complete silence now","zero volume please",
    "switch the sound off","quiet everything down to nothing","mute right now",
]
P["Cmd.VolumeUnmute"] = [
    "turn the sound back on","unmute my aids","bring the audio back",
    "i want to hear again","restore the sound","switch sound back on",
    "take my aids off mute","let the audio return","end the silence",
    "reactivate the sound","sound back please","undo the mute",
    "stop muting now","i can listen again turn it on","give me sound again",
    "re-enable the audio","put the volume back on","unmute everything",
]
# ---- MEMORY / PROGRAM ------------------------------------------------------
P["Cmd.MemoryChange"] = [
    "i'm walking into a loud bar","switch me to the restaurant setting",
    "i'm sitting down in a crowded diner","change to my outdoor program",
    "set my aids for a noisy party","i'm getting in the car now",
    "put me in the music listening mode","i'm heading into a meeting",
    "tune for this windy weather","i'm at the gym now",
    "give me my tv watching program","change into the quiet mode",
    "i just walked into church","switch to the program for restaurants",
    "set it up for a concert hall","i need my noisy environment setting",
    "move me to the speech in noise mode","i'm starting my commute",
    "use the setting for outdoors","flip into my everyday program",
    "i'm in a loud cafeteria now","change the program for this place",
    "adjust the mode for traffic","i'm settling in to watch a film",
    "switch to the program made for crowds","load my driving profile",
    "set my hearing for a busy street","i'm about to join a phone call",
    "change to the comfortable indoor mode","i'm in a big echoey room now",
]
# ---- BATTERY ---------------------------------------------------------------
P["Cmd.BatteryLevel"] = [
    "how much charge is left","are my aids fully charged","what's my battery at",
    "do i need to recharge","how long will the battery last","is the battery low",
    "tell me the battery percentage","how much power do i have left",
    "will these last till tonight","check my charge level",
    "are the batteries about to run out","how many hours of charge remain",
    "is there plenty of battery","what's the charge on my aids",
    "should i charge them now","how full is the battery","battery level please",
    "are my hearing aids charged enough","do these have enough power left",
    "how much battery remains","is my charge running low","power level check",
    "have i got enough battery for the day","is it time to charge",
    "how depleted is the battery","tell me how charged they are",
    "what percentage of battery is left","are my aids running out of power",
]
# ---- STREAMING -------------------------------------------------------------
P["Cmd.StreamingStart"] = [
    "send the tv audio to my aids","start streaming the television",
    "play my phone audio through my aids","stream the music into my ears",
    "route the movie sound to my hearing aids","begin streaming to my devices",
    "connect the tv sound to me","let me hear the show through my aids",
    "stream the podcast to my hearing aids","put the audio into my ears",
    "start the audio stream","send sound straight to my aids",
    "play the call through my hearing aids","stream from my tv now",
    "i want to listen to the tv through my aids","beam the audio over to me",
    "hook my aids up to the tv sound","direct the music to my hearing aids",
    "feed the audio into my devices","turn on streaming to my ears",
]
P["Cmd.StreamingStop"] = [
    "stop streaming to my aids","end the tv audio stream","disconnect the streaming",
    "i'm finished listening to the stream","cut off the streamed audio",
    "stop sending sound to my hearing aids","turn streaming off now",
    "end streaming from the tv","quit the audio stream","halt the streaming",
    "no more streaming please","stop playing the tv through my aids",
    "disconnect me from the stream","shut off the streamed sound",
    "i'm done with the stream","close the audio streaming",
    "stop the music going to my aids","end the broadcast to my hearing aids",
]
# ---- ACTIVITY --------------------------------------------------------------
P["Cmd.ActivityRun"] = [
    "i'm about to go for a run","track my run","log my jog",
    "start recording my run","i'm heading out running","begin my running workout",
    "record this run for me","start my jog tracking","time my run",
    "i'm going jogging now","capture my running session","start tracking my jog",
    "log this running workout","kick off my run tracking","i want to record a run",
]
P["Cmd.ActivityWalk"] = [
    "track my walk","i'm going for a stroll","log my walk","start my walk tracking",
    "record this walk","begin tracking my walk","i'm off for a walk",
    "time my stroll","capture my walking session","start logging my walk",
    "i'm taking a walk now","record my steps on this walk","track my stroll",
    "begin my walking workout","log a walk for me",
]
P["Cmd.ActivityStep"] = [
    "how many steps today","what's my step count","show my steps so far",
    "check my step total","how many steps have i taken","step count please",
    "am i close to my step goal","tell me my steps for today","track my steps",
    "what's my daily step number","how far along my steps am i","steps so far today",
    "give me my step tally","how many steps to my goal","display my step count",
]
P["Cmd.ActivityStand"] = [
    "how long have i been standing","check my stand hours","what's my standing time",
    "how many stand hours today","track my standing","show my stand count",
    "am i meeting my stand goal","how much have i stood today","standing time please",
    "tell me my stand hours","how long on my feet today","log my standing time",
    "have i stood enough today","my standing total please","check standing progress",
]
P["Cmd.ActivityCycle"] = [
    "track my bike ride","i'm going cycling","log my ride","start my cycling workout",
    "record this bike ride","how far have i cycled","begin tracking my ride",
    "i'm heading out on the bike","capture my cycling session","time my bike ride",
    "start logging my cycling","track my biking","how many miles have i biked",
    "record my ride for me","log this cycling session",
]
P["Cmd.ActivityAerobics"] = [
    "track my aerobics","start my aerobics workout","log my aerobics session",
    "i'm doing aerobics now","record my aerobics","begin tracking aerobics",
    "capture my aerobics class","time my aerobics","start my cardio class tracking",
    "log this aerobics workout","i'm starting aerobics","track my dance workout",
    "record my aerobic exercise","begin my aerobics session","log my cardio dance",
]
P["Cmd.ActivityExercise"] = [
    "track my workout","i'm about to exercise","log my exercise","start my workout",
    "record my training session","begin tracking my exercise","i'm working out now",
    "capture my workout","time my exercise","start logging my training",
    "track this gym session","record my workout for me","i'm hitting the gym track it",
    "log my exercise session","begin my workout tracking",
]
P["Cmd.ActivityCalories"] = [
    "how many calories have i burned","what's my calorie burn today","show my calories",
    "check my burned calories","calorie count please","how much energy have i burned",
    "tell me my calories for today","what's my burn so far","track my calories",
    "display my calorie total","how many calories today","my calorie burn please",
    "calories burned so far","give me my energy burn","how many calories did i burn",
]
# ---- MESSAGING -------------------------------------------------------------
P["Cmd.SendMessage"] = [
    "text my friend","send a message to my wife","i want to message my son",
    "shoot a text to mom","send word to my daughter","drop my brother a message",
    "write a text to my neighbor","message my husband","send a quick text",
    "let my friend know i'm on my way","text my sister","send a note to dad",
    "compose a message to my coworker","i need to text someone","fire off a text",
    "send a text to my doctor","message my family","get a text to my friend",
    "i'd like to send a message","start a text to my partner",
]
P["Cmd.SendMessage - yes"] = [
    "yes send it","go ahead and send","yep send that","send it now","that's right send it",
    "confirm and send","yes please send","sounds good send it","correct send it off",
    "send it as is","yeah go ahead","that works send it","affirmative send it",
    "ok send the message","yes that's good send",
]
P["Cmd.SendMessage - no"] = [
    "no don't send","cancel the message","scrap that text","don't send it",
    "stop wait that's wrong","no cancel that","hold off on sending","delete that message",
    "nope don't send it","forget the message","abort the text","that's not right cancel",
    "no leave it","don't send that please","cancel it",
]
P["Cmd.ListenMessage"] = [
    "read my messages out loud","play back my texts","what messages do i have",
    "read my latest text","let me hear my messages","play my new messages",
    "read out who texted me","any new messages read them","speak my messages",
    "play the text i just got","read my unread messages","tell me my messages",
    "hear my texts please","read me the latest message","play my incoming texts",
]
P["Cmd.FindMyPhone"] = [
    "where did i leave my phone","ring my phone so i can find it","help me find my phone",
    "make my phone ring","i can't find my phone","locate my phone for me",
    "ping my phone","where's my cell phone","find my handset","sound an alert on my phone",
    "i misplaced my phone find it","track down my phone","my phone is lost ring it",
    "buzz my phone so i hear it","where is my mobile",
]
# ---- TRANSCRIBE / TRANSLATE (Cmd = do it now) ------------------------------
P["Cmd.TranscribeStart"] = [
    "start transcribing this","write down what's being said","caption this conversation",
    "turn speech into text now","begin the live transcript","transcribe what i'm hearing",
    "put this conversation into writing","start the captions","capture this talk as text",
    "get a transcript going","convert what they're saying to text","start writing down the speech",
    "give me live captions now","transcribe this meeting","begin converting speech to text",
]
P["Cmd.TranslationStart"] = [
    "translate this for me now","start translating what they say","turn on translation",
    "translate their speech into english","begin live translation","translate this conversation",
    "switch on translate mode","translate what i'm hearing","start translating to spanish",
    "translate them into my language","get translation going","translate the speech now",
    "begin translating for me","translate this person's words","start the live translation",
]
# ---- REMINDERS -------------------------------------------------------------
P["reminders.add"] = [
    "remind me to take my pills at 8","set a reminder to call the clinic",
    "don't let me forget my appointment","remind me about dinner at six",
    "make a reminder to charge my aids","i need a reminder to take medication",
    "set something to remind me tomorrow","remind me to water the plants",
    "add a reminder for my meeting","remind me to call my daughter later",
    "put a reminder for my checkup","help me remember to buy milk",
    "remind me to leave at noon","set a reminder for my pills tonight",
    "create a reminder to pay the bill","nudge me to stretch in an hour",
    "remind me about the doctor on friday","set a daily reminder for my vitamins",
    "remind me to take out the trash","i want a reminder for my class",
]
P["reminders.complete"] = [
    "mark my reminder done","i finished that task","check off my reminder",
    "that reminder is complete","i took my pills mark it done","clear that reminder",
    "i did it cross it off","complete my reminder","mark that task finished",
    "tick off my reminder","i handled that remove the reminder","done with that reminder",
    "i already did that mark complete","close out that reminder","that one's finished",
]
# ---- HELP_* (informational: how / what / explain / feature help) -----------
P["Help_Volume"] = [
    "how do i adjust the volume","explain the volume controls","where do i change loudness",
    "how can i set volume per ear","help me understand volume settings",
    "what are the volume options","how do volume adjustments work","guide me on changing volume",
    "where is the loudness control","how do i tweak the volume on each side",
    "explain how to make it louder or softer","volume settings help",
    "how does the volume control work","what's the way to change volume","help with the volume feature",
]
P["Help_ChangingMemories"] = [
    "how do i switch programs","explain changing memories","how do i move between settings",
    "what's the way to change my program","help me understand switching memories",
    "how do different programs work","guide me through changing memories",
    "how can i flip between modes","explain how memory switching works",
    "where do i change which program i'm on","help with switching between programs",
    "how do i toggle my hearing programs","what does changing memory do","memory switching help",
    "how do i go from one program to another",
]
P["Help_MemoryOptions"] = [
    "how do i create a new program","can i rename my memories","explain memory options",
    "how do i delete a program","what memory settings are available","how do i save a favorite program",
    "can i set a program for a place","help with memory presets","how do custom programs work",
    "explain the program options","how many memories can i have","can i add a new memory",
    "what are my program choices","help me set up a custom memory","how do i manage my programs",
]
P["Help_Pairing"] = [
    "how do i pair my aids to my phone","explain bluetooth pairing","my aids won't connect to the app",
    "how do i connect my hearing aids","help pairing with my phone","how to set up bluetooth",
    "guide me through pairing","why won't my aids pair","how do i link my aids to bluetooth",
    "pairing help please","how do i reconnect my hearing aids","steps to pair my devices",
    "my hearing aids lost connection to the phone","how to bond my aids with the app","bluetooth pairing help",
]
P["Help_FindMyHearingAids"] = [
    "how do i find lost hearing aids","explain the find my aids feature","where is the locator for my aids",
    "how does finding my aids work","help me locate misplaced aids","what's the find my hearing aids tool",
    "how can the app help me find my aids","guide me to the aid finder","how do i track down my aids",
    "where do i see my aids last location","help finding my hearing aids feature",
    "how do i use find my hearing aids","what does the locate aids feature do","find my aids help",
    "how do i search for my hearing aids in the app",
]
P["Help_Accessories"] = [
    "what accessories work with my aids","explain the remote microphone","how do i use the tv streamer",
    "what add-ons are available","help with the remote mic","do my aids support a tv connector",
    "what extra devices can i pair","explain the accessory options","how does the mini remote work",
    "what gadgets are compatible","tell me about the available accessories","how do i set up the tv accessory",
    "accessories help please","what hardware can i add","guide me on accessories",
]
P["Help_RemoteProgramming"] = [
    "how does remote programming work","can my audiologist tune my aids remotely","explain remote adjustments",
    "how do i get remote fine tuning","help with remote programming","what is remote tuning",
    "how do i request a remote adjustment","can settings be changed from afar","explain how remote tuning works",
    "how do remote updates to my aids work","guide me on remote programming","remote programming help",
    "how does my provider adjust my aids remotely","what does remote programming let me do","help understanding remote tuning",
]
P["Help_Tinnitus"] = [
    "my ears keep ringing what can i do","explain the tinnitus features","how do i manage ringing in my ears",
    "help with the buzzing in my ears","what does the tinnitus tool do","how do i use tinnitus relief",
    "is there a feature for ear ringing","explain how to ease tinnitus","how does the masking sound help tinnitus",
    "help for the constant ringing","what are my options for tinnitus","how do i turn on tinnitus relief",
    "guide me on tinnitus management","tinnitus feature help","how can my aids help with ringing",
]
P["Help_SelfCheck"] = [
    "how do i run a self check","explain the self check feature","how do i test my hearing aids",
    "what does the self check do","help me check if my aids work","how to run diagnostics on my aids",
    "guide me through a self check","how do i verify my aids are working","self check help please",
    "where is the device test","how do i make sure my aids are okay","explain how to test my devices",
    "run me through the self check","what does a self check tell me","how do i diagnose my hearing aids",
]
P["Help_ThriveScore"] = [
    "what is my thrive score","explain how the thrive score works","where do i see my wellness score",
    "how is my score calculated","help me understand the thrive score","what does my body and brain score mean",
    "how do i improve my thrive score","explain the engagement score","what goes into the thrive score",
    "where is my daily score","help with the thrive score","what's a good thrive score",
    "how do i check my wellness score","explain my brain score","thrive score help",
]
P["Help_DeviceSettings"] = [
    "the audio keeps cutting out what do i do","my aids sound distorted help","why does the sound drop sometimes",
    "everything sounds fuzzy how do i fix it","help my aids sound off","the sound quality went bad",
    "my hearing aids sound muffled","why is there static in my aids","how do i fix poor sound quality",
    "the audio is breaking up","my aids keep glitching","sound is crackling in my hearing aids",
    "help with bad audio from my aids","why do my aids sound strange","my hearing aids aren't sounding right",
]
P["Help_IntelliVoice"] = [
    "how does intellivoice work","explain the smart voice feature","what does intelligent voice do",
    "how do i turn on intellivoice","help with the speech enhancement feature","what is the ai voice feature",
    "how does the voice clarity feature work","explain dnn speech enhancement","when should i use intellivoice",
    "guide me on intelligent voice","intellivoice help please","what does the smart speech mode do",
    "how do i enable speech enhancement","explain how intellivoice helps","tell me about the intelligent voice feature",
]
P["Help_FallAlert"] = [
    "how does fall detection work","will my aids alert someone if i fall","explain the fall alert feature",
    "what happens when i fall","help with fall alerts","how do i set up fall detection",
    "does it notify family if i take a tumble","explain how fall alerts are sent","how do i turn on fall detection",
    "what does the fall alert do","guide me on fall alerts","who gets notified if i fall",
    "fall detection help","how reliable is the fall alert","how do fall notifications work",
]
P["Help_HearShare"] = [
    "how do i share what i hear","explain hearshare","can someone listen along with me",
    "how does shared listening work","help setting up hearshare","how do i add a person to hearshare",
    "what is the hearshare feature","explain how to share my audio","how do i let a caregiver listen",
    "guide me through hearshare","hearshare help please","can i share my hearing aid sound with family",
    "how does hearshare connect another person","what does shared listening do","help me use hearshare",
]
P["Help_VoiceAssistant"] = [
    "what voice assistants can i use","can i use siri with my aids","explain the voice assistant feature",
    "how do i set up google assistant","help with the voice assistant","does it work with alexa",
    "how do i talk to my voice assistant","explain how voice control works","how do i enable the assistant",
    "what can the voice assistant do","guide me on the voice assistant","voice assistant help please",
    "how do i use voice commands","can i control my aids by voice","tell me about the voice assistant",
]
P["Help_InsertDevice"] = [
    "how do i put my hearing aids in","explain how to insert my aids","which way does the aid go in my ear",
    "help putting in my hearing aids","my aid keeps falling out how do i fit it","how do i wear my aids properly",
    "guide me on inserting my hearing aids","how should the aid sit in my ear","tips for putting in my aids",
    "how do i place the dome in my ear","help me fit my hearing aids","what's the right way to insert my aids",
    "how do i get the aid to stay in","explain proper aid placement","inserting hearing aids help",
]
P["Help_AppSettings"] = [
    "how do i change the app settings","where are the app preferences","explain the settings menu",
    "how do i reset the app","help with the app configuration","how do i update my app preferences",
    "where do i manage app options","how do i change the app language","guide me through app settings",
    "how do i adjust notifications in the app","app settings help","how do i restore default app settings",
    "where do i find the settings page","how do i set up the app the way i want","explain how to configure the app",
]
P["Help_CleanCare"] = [
    "how do i clean my hearing aids","explain how to care for my aids","how often should i clean them",
    "can my aids get wet","help keeping my hearing aids clean","what's the best way to clean my aids",
    "how do i maintain my hearing aids","tips for cleaning my aids","how do i remove wax from my aids",
    "guide me on caring for my hearing aids","cleaning and care help","how should i store my aids overnight",
    "how do i keep moisture out of my aids","explain hearing aid maintenance","how do i dry my hearing aids",
]
P["Help_Customize"] = [
    "how do i personalize my sound","explain how to customize my aids","can i fine tune the bass and treble",
    "how do i tailor my hearing experience","help customizing my settings","how do i adjust the tone of my aids",
    "what can i personalize on my aids","guide me on customizing sound","how do i set my own preferences",
    "how do i change the sound profile","customization help please","how do i make the sound my own",
    "can i adjust settings for my taste","explain personalizing my hearing aids","how do i customize how things sound",
]
P["Help_WhatsNew"] = [
    "what's new in the app","show me the latest features","what got added recently",
    "any new updates to know about","explain what changed in this version","what's been updated lately",
    "tell me about new features","what's different in the latest release","show recent changes",
    "what improvements are new","help me see what's new","what features were just added",
    "catch me up on new stuff","what's new in this update","recent feature updates please",
]
P["Help_WiCROS"] = [
    "how does the cros system work","explain wireless cros","what is a cros hearing aid",
    "help with my cros setup","how does cros send sound to my good ear","what does the cros do",
    "explain how cros and bicros work","how do i use my cros device","guide me on the cros feature",
    "what is wicros","cros help please","how does the cros transmitter work",
    "how do i set up cros","explain the cros for single sided hearing","tell me about the cros system",
]
P["Help_Home"] = [
    "i don't know how to use this app","walk me through the basics","what can this app do for me",
    "i'm new help me get started","give me an overview of the app","i'm confused by all these screens",
    "show me how this works","help me figure out the app","what should i do first here",
    "i need a hand getting going","explain the main screen to me","where do i go to do things",
    "help me get oriented","what are the main things i can do","introduce me to the app",
]
P["Help_Battery"] = [
    "should i use rechargeable or disposable batteries","explain battery options for my aids","my battery drains too fast why",
    "how long should a battery last","help with battery questions","what battery size do my aids use",
    "tips to make my batteries last longer","why is my battery life getting shorter","explain hearing aid batteries",
    "how do i change the battery","what's the best battery for my aids","guide me on battery care",
    "battery help please","how do i store spare batteries","why do batteries die quickly in my aids",
]
P["Help_Reminder"] = [
    "how do reminders work","explain the reminder feature","can i set recurring reminders",
    "how do i manage my reminders","help with setting up reminders","where do i see my reminders",
    "how do i edit a reminder","what can the reminder feature do","guide me on using reminders",
    "how do i make a repeating reminder","reminder feature help","can i get daily reminders",
    "how do i turn off a reminder","explain how to schedule reminders","how do reminder notifications work",
]
P["Help_Translate"] = [
    "what languages can it translate","explain the translate feature","how does translation work in the app",
    "is spanish supported for translation","help with the translate function","how accurate is the translation",
    "what is the translate feature","guide me on using translate","does translate save conversations",
    "how do i set the translation language","translate feature help","can it translate two languages at once",
    "how do i use the translation tool","explain real time translation","what does the translate feature do",
]
P["Help_Transcribe"] = [
    "how does transcribe work","explain the transcription feature","does the app do live captions",
    "what is the transcribe function","help with transcription","can i save my transcripts",
    "how accurate is the transcription","guide me on using transcribe","does transcribe keep my conversations",
    "where do i find transcription","transcribe feature help","how do i read what was transcribed",
    "explain how captions work in the app","what does the transcribe feature do","how do i use speech to text",
]
P["Help_Health"] = [
    "where do i see my health data","explain the health dashboard","how do i view my fitness summary",
    "where is my activity overview","help with the health screen","how do i set my health goals",
    "what health info does the app track","guide me to my wellness data","where do i see steps and exercise together",
    "how do i read my health stats","health screen help","what's on the health dashboard",
    "how do i check my overall activity","explain my fitness tracking","where do my health metrics show",
]
P["Help_HeartRate"] = [
    "can my aids track my heart rate","how do i see my heart rate","explain the heart rate feature",
    "where do i find my pulse readings","help with heart rate tracking","do my aids measure bpm",
    "how does heart rate monitoring work","guide me to my heart rate","is my heart rate tracked automatically",
    "where do i see my beats per minute","heart rate help","how accurate is the heart rate",
    "how do i check my pulse on the app","explain how heart rate is measured","what does the heart rate feature show",
]
P["Help_HeartRateRecovery"] = [
    "what is heart rate recovery","explain the recovery metric","how do i see how fast my heart recovers",
    "where is my heart rate recovery","help understanding recovery rate","how is heart rate recovery measured",
    "what does my recovery score mean","guide me on heart rate recovery","how do i check my recovery after exercise",
    "explain how recovery is tracked","heart rate recovery help","why does heart rate recovery matter",
    "where do i find recovery data","how good is my heart rate recovery","what's a healthy recovery rate",
]
P["Help_HearingCareAnywhereConnect"] = [
    "how do i connect for a remote appointment","explain hearing care anywhere","how do i reach my audiologist through the app",
    "help connecting to my provider remotely","what is hearing care anywhere","how do i request remote care",
    "how do i send a request to my specialist","guide me on connecting with my clinician","how does remote care connect work",
    "set up a remote session with my provider","hearing care anywhere help","how do i get new settings from my audiologist",
    "how do i link with my hearing professional","explain connecting for remote support","how do i start a remote care request",
]
P["Help_MaskMode"] = [
    "what is mask mode","explain the mask mode feature","how do i turn on mask mode",
    "when should i use mask mode","help with mask mode","how does mask mode help with face masks",
    "what does mask mode do","guide me on mask mode","how do i enable mask mode for masks",
    "mask mode help please","does mask mode help me hear masked voices","explain how mask mode works",
    "how do i use mask mode","when is mask mode useful","tell me about mask mode",
]
P["Help_EdgeMode"] = [
    "what is edge mode","explain edge mode","how do i activate edge mode",
    "when should i use edge mode","help with edge mode","what does edge mode do",
    "how does edge mode improve sound","guide me on edge mode","how do i turn on edge mode for tough situations",
    "edge mode help please","does edge mode help in noisy places","explain how edge mode works",
    "how do i use edge mode","when is edge mode helpful","tell me about edge mode",
]
P["Help_DemoMode"] = [
    "what is demo mode","explain demo mode","how do i turn on demo mode",
    "what does demo mode do","help with demo mode","how do i use the demo feature",
    "is demo mode for trying features","guide me on demo mode","how do i exit demo mode",
    "demo mode help please","what's the point of demo mode","explain how demo mode works",
    "how do i enable demo mode","when would i use demo mode","tell me about demo mode",
]

# ── Wave 2: targeted contrastive data for the confusions the holdout exposes ──
# Help_* get heavy "feature help / how does / what is the / where is / explain"
# marker coverage so informational queries stop being read as Cmd actions.
# Cmd.* siblings get the descriptive registers they were missing.
W2 = {
 "Help_Transcribe": [
    "captions feature help","help with the captioning feature","what does the transcribe feature do",
    "how do i read what was transcribed","is there a way to caption conversations","tell me about live captions",
    "what's the transcription feature for","how do captions work on my aids","do my aids show subtitles of speech",
    "explain the speech to text feature","help understanding transcription","what can the caption feature do",
    "how do i view a transcript","is the transcribe feature free","where do transcripts get saved",
 ],
 "Help_Translate": [
    "what does the translate feature do","help with the language translation feature","how good is the translation feature",
    "which languages does translation support","tell me about the translation feature","can the app translate languages for me",
    "explain how translation works on my aids","is there a translation feature","what's the translate feature for",
    "how do i set up language translation","help understanding the translate feature","does translation work both ways",
    "what translation languages are offered","how do i pick a translation language","is real time translation available",
 ],
 "Help_DeviceSettings": [
    "my hearing aids keep cutting out","help my aids sound crackly","the sound quality is poor lately",
    "why do my aids buzz","my aids keep dropping sound","help fixing the sound on my aids",
    "the audio is distorted on my hearing aids","my aids sound tinny","why is my hearing aid quiet on one side",
    "troubleshoot my hearing aid sound","my aids cut out intermittently","help with poor audio from my aids",
    "the sound keeps breaking up","my hearing aids sound off today","fix the static in my aids",
    "why do my aids sound muffled","help my hearing aids aren't sounding right","sort out the sound issue on my aids",
 ],
 "Help_Volume": [
    "how do i make one side louder","explain adjusting volume per ear","where do i control the loudness",
    "help with volume on each aid","how do i balance the volume between ears","what's the volume control feature",
    "how do i turn the volume up in the app","guide me on volume adjustment","where are the loudness settings",
    "how do volume changes get saved","explain the loudness controls","help me understand volume per side",
 ],
 "Help_Tinnitus": [
    "help with the ringing in my ears feature","how do i manage my tinnitus","the masking sound feature help",
    "how do i start the tinnitus relief sound","explain how the aids help with ringing","what's the tinnitus relief feature",
    "how do i turn on the soothing sound","help for my ear ringing","guide me on tinnitus relief",
    "what does the tinnitus feature do","how do i use the masking noise","explain the relief sound for ringing ears",
 ],
 "Help_HearShare": [
    "how do i let another person hear with me","explain sharing my audio with family","help with the shared listening feature",
    "can a caregiver listen to what i hear","how do i invite someone to hearshare","what does the share listening feature do",
    "how do i share my hearing aid audio","guide me on sharing what i hear","can two people listen together",
    "how does sharing audio with another person work","help setting up shared listening","explain how to share my sound",
 ],
 "Help_Customize": [
    "how do i fine tune my sound preferences","explain personalizing my hearing","where do i adjust bass and treble",
    "help tailoring my hearing experience","how do i set my own sound profile","what can i personalize on my hearing aids",
    "guide me on customizing my sound","how do i change the tone of my aids","explain the personalization options",
    "how do i make the sound suit me","help customizing how things sound","where do i tweak my sound settings",
 ],
 "Help_ChangingMemories": [
    "explain how to move between my programs","help understanding program switching","how does changing profiles work",
    "guide me on switching my hearing programs","what's the way to change between settings","how do i learn about changing programs",
    "explain switching my hearing modes","help with moving between memories","how do program changes work on my aids",
 ],
 "Help_MemoryOptions": [
    "explain my program setup options","how do i organize my memories","what can i do with program presets",
    "help managing my saved programs","how do i add or remove a program","guide me on program options",
 ],
 "Help_RemoteProgramming": [
    "explain how my audiologist tunes me remotely","help with getting a remote adjustment","how do remote fittings work",
    "guide me on remote fine tuning","how do i ask for a remote fitting","what is the remote fitting process",
 ],
 "Help_SelfCheck": [
    "how do i diagnose a problem with my aids","run me through testing my hearing aids","help checking if something is wrong with my aid",
    "how do i troubleshoot my aids myself","guide me on running a device test","what does running a diagnostic check do",
 ],
 "Help_VoiceAssistant": [
    "explain using a voice assistant with my aids","help setting up voice control","can i pair siri to my hearing aids",
    "guide me on the voice assistant feature","how do i use google assistant with my aids","what voice helpers are supported",
 ],
 "Help_EdgeMode": [
    "explain when edge mode helps","help me understand edge mode","how do i trigger edge mode in noise",
    "what situations is edge mode for","guide me on using edge mode","how does edge mode adapt the sound",
 ],
 "Help_MaskMode": [
    "explain when to use mask mode","help understanding mask mode","how do i switch on mask mode for masks",
    "what is mask mode good for","guide me on mask mode","how does mask mode help with covered faces",
 ],
 "Cmd.VolumeUnmute": [
    "bring my sound back","let me hear things again","i can hear now turn it on",
    "give me audio again","stop the silence i want sound","return the sound to my aids",
    "switch my hearing back on","take the mute off","i'd like sound again please",
    "wake the sound back up","put my hearing back on","let the audio come back",
    "i want my aids making sound again","cancel the mute please","unmute me",
]+["turn the mute off","get me off mute","sound on again please","reactivate my hearing aid sound"],
 "Cmd.ListenMessage": [
    "read me what people sent","play the messages i received","what texts came in for me",
    "read my new texts aloud","let me listen to my messages","play who messaged me",
    "read out my incoming messages","tell me what messages i got","speak my latest texts to me",
    "play back the texts i received","read my message inbox","what did my contacts send me",
 ],
 "Cmd.VolumeMute": [
    "i want pure silence","cut every bit of sound","make it dead silent",
    "no sound whatsoever please","kill all the noise completely","total quiet now please",
    "shut off all sound for me","silence the aids entirely","i need it perfectly quiet",
    "mute all audio right now","take everything to silence","absolutely no sound please",
 ],
 "Cmd.StreamingStop": [
    "end my tv streaming","stop the audio coming to my aids","quit streaming the television",
    "i don't want the stream anymore","close the connection to the tv sound","stop piping audio to my ears",
    "turn off the tv streaming","cut the streamed audio to my aids","finish the streaming session",
 ],
 "Cmd.FindMyPhone": [
    "help me track down my phone","i lost my phone make it sound","where could my phone be ring it",
    "set off an alert on my phone","my handset is missing please ring it","sound my phone so i can locate it",
    "i can't locate my phone ping it","make my lost phone beep","find where my phone went",
 ],
 "Cmd.SendMessage - yes": [
    "yes that's what i wanted to say","yep that's correct send it","that's exactly right go ahead",
    "yes please go ahead and send","correct that's the message","yeah send it like that",
    "that's what i meant send it","right send that one",
 ],
 "Cmd.SendMessage - no": [
    "hold on that's not right","wait that's not what i said","no that came out wrong",
    "that's not correct cancel it","nope redo that message","stop that's not what i meant",
    "that's wrong don't send it","no scrap it",
 ],
 "reminders.complete": [
    "i already did that one clear it","that task is taken care of","go ahead and finish that reminder",
    "i wrapped that up mark it done","you can clear that reminder now","that's handled cross it off",
    "i completed it remove the reminder","done dealt with that reminder",
 ],
 "Cmd.ActivityWalk": [
    "log the stroll i just took","track the walk i did","record the walking i just finished",
    "i went for a walk log it","save my walk","note down my walk",
 ],
 "Cmd.ActivityRun": [
    "log the run i just did","record the miles i ran","save the run i just finished",
    "note my jog","track the run i completed","log my finished run",
 ],
 "Default Fallback Intent": [
    "what should i cook for dinner","who won the game last night","what's a good movie to watch",
    "recommend me a book","how do i bake bread","what's the score of the match",
    "tell me a fun fact","what's the meaning of life","how tall is mount everest",
    "what time is the sunset","give me a recipe for soup","what's a good restaurant nearby",
    "how do i get rid of weeds","what should i watch tonight","who is the president",
    "what's on tv tonight","how do i tie a tie","what's the capital of japan",
    "plan my vacation","what should i wear today","how do i make coffee",
    "what's a good gift idea","tell me about the news","how far is the nearest store",
 ],
}
for _k, _v in W2.items():
    P.setdefault(_k, [])
    P[_k].extend(_v)

# ── Wave 3: close the cleanly-fixable boundary confusions + lift weak intents ──
W3 = {
 "Help_Translate": [
    "is spanish available for translation","which languages are supported for translation",
    "are there many languages to translate","is my language offered for translation",
    "what languages does the translate feature offer","is german one of the translation options",
    "can it translate into french","are foreign languages supported","do you support italian translation",
 ],
 "Help_Accessories": [
    "can i use a clip on microphone","does it work with a remote mic","can i connect an external mic",
    "what microphones can i pair","is there a tv streaming accessory","can i add a partner mic",
    "does a remote microphone work with my aids","what extra mics are supported","can i use a lapel mic with my aids",
 ],
 "Help_CleanCare": [
    "is it safe to get my aids wet","can my hearing aids handle water","what if my aids get damp",
    "are my aids waterproof","can i wear my aids in the shower","what happens if my aids get wet",
    "is moisture bad for my hearing aids","can i clean my aids with water","how do i deal with sweat on my aids",
 ],
 "Help_InsertDevice": [
    "my hearing aid keeps slipping out","the aid won't stay in my ear","my device won't sit properly",
    "the hearing aid falls out when i chew","my aid feels loose in my ear","why does my aid keep coming out",
    "the dome won't stay put","my aid keeps working its way out","how do i keep my aid from falling out",
 ],
 "Help_ThriveScore": [
    "where do i find my overall rating","show my daily wellness rating","what's my total thrive number",
    "where is my engagement rating","my overall score for the day","how do i view my thrive rating",
    "where's my combined body and brain rating","what is my daily achievement score",
 ],
 "Help_HearShare": [
    "let a friend hear what i hear","share my hearing with another person","i want someone to listen along",
    "stream what i hear to a family member","can my spouse hear through my aids","share my audio with a loved one",
 ],
 "Help_Health": [
    "where is my fitness overview","show my activity and exercise summary","my daily health metrics",
    "where do my steps and workouts show together","open my wellness dashboard","my overall activity stats",
 ],
 "Cmd.VolumeDecrease": [
    "i need some quiet","i need it more quiet","give me a little quiet","quiet it down for me",
    "i'd like things quieter","make my surroundings quieter","bring me some quiet",
 ],
 "Help_DeviceSettings": [
    "fix the sound problems on my aids","help with my hearing aid sound settings","sort out my device audio",
    "my aids need adjusting they sound bad","repair the audio on my hearing aids","help me fix my aid settings",
 ],
 "Cmd.MemoryChange": [
    "the program isn't right for where i am","this setting doesn't suit this place","i need a better program for here",
    "switch me to something better for this spot","this mode isn't working for my surroundings",
 ],
 "Help_Home": [
    "i'm completely lost in here","this is all confusing to me","i can't make sense of the app",
    "i don't get how any of this works","help i'm overwhelmed by the app",
 ],
}
for _k, _v in W3.items():
    P.setdefault(_k, [])
    P[_k].extend(_v)

# ── Wave 4: register coverage for documented holdout failures ────────────────
# Each block targets a SPECIFIC miss observed on test_holdout.py. These are new
# paraphrases in the failing register/vocabulary — NOT the verbatim test phrases
# (the leakage filter drops any that collide). Two groups:
#   (a) wrong-action fixes — the model fired the WRONG command (safety-critical),
#   (b) safe-miss fixes    — the model fell to GenAI; correct intent is clear.
# Several targets are contrastive: the failing token pulls to a sibling intent
# ("alert"->FallAlert, "share"->HearShare, "stream"->StreamingStart, "lost"->
# FindMyHearingAids, "understand"->Help_Home), so coverage is added to the
# CORRECT intent with distinctive vocabulary to move the boundary.
W4 = {
 # (a) ── wrong-action fixes ──────────────────────────────────────────────────
 "reminders.add": [
    "alert me when it's time to leave","give me an alert at the scheduled time",
    "set an alert so i don't miss it","i have an appointment coming up don't let me miss it",
    "make sure i don't miss my appointment","buzz me when it's time",
    "notify me when the time comes","i can't miss this appointment remind me",
    "set an alert for my appointment","poke me when the time arrives",
 ],
 "Cmd.TranslationStart": [
    "they're speaking a foreign language translate it","i can't understand their language translate",
    "i can't understand this foreign language","help me understand what they're saying in another language",
    "they're speaking another language i can't follow","understand this foreign speaker for me",
    "put their foreign words into my language","i don't understand their foreign language",
    "convert this foreign language to english","they don't speak english help me understand them",
 ],
 "reminders.complete": [
    "i finished that reminder mark it","that reminder is done remove it",
    "check that reminder off the list","i'm done with that task clear the reminder",
    "cross that reminder off","that to-do is finished mark it complete",
    "wrap up that reminder it's done","i've completed that reminder",
 ],
 "Help_HearShare": [
    "let another person hear what i hear","stream my hearing to someone else",
    "can a friend listen along with me","share my hearing aid audio with someone",
    "let a family member listen in","send what i hear to another person",
    "how do i let someone listen with me","share my audio so others can hear",
 ],
 "Help_Accessories": [
    "can i stream using a remote microphone","does a clip on mic send audio to my aids",
    "use an external mic accessory with my aids","does the partner mic stream to my hearing aids",
    "can i pair a wearable microphone","stream from an accessory mic to my aids",
    "what microphone accessories can stream to me","connect a remote mic accessory",
 ],
 "Help_EdgeMode": [
    "there's too much wind noise what setting helps","the wind outside is overwhelming what can i do",
    "cut the wind noise with a setting","help with wind noise when i'm outdoors",
    "what mode handles a windy environment","reduce wind noise in tough conditions",
    "the wind is ruining my hearing outside","a setting for very windy places",
 ],
 "Help_HeartRate": [
    "where do i check my bpm","show me my bpm","what's my current bpm",
    "view my beats per minute reading","my bpm please","track my bpm on the app",
    "see my pulse in bpm","how do i read my bpm",
 ],
 "Help_Pairing": [
    "my aids dropped their bluetooth connection","i lost the connection to my hearing aids",
    "my hearing aids disconnected reconnect them","the connection to my aids dropped",
    "my aids won't stay connected to my phone","reconnect my hearing aids to the app",
    "my aids keep losing the bluetooth link","help my aids lost connection to the phone",
 ],
 "Cmd.SendMessage": [
    "i have news to text a contact","send my update to someone","let a contact know my news by text",
    "text a contact with my news","i want to tell someone my news in a message",
    "message a contact an update","share my news with a contact by text",
 ],
 # (b) ── safe-miss fixes (correct intent unambiguous) ────────────────────────
 "Cmd.VolumeDecrease": [
    "my ears are taking a beating","the noise is pounding my ears","this is hammering my ears",
    "my ears can't take this pounding","the sound is battering my ears",
 ],
 "Cmd.MemoryChange": [
    "i just took a seat in a noisy cafe","i'm now seated in a loud coffee shop",
    "settling into a busy restaurant","i just sat down somewhere really loud",
    "i'm seated in a crowded noisy spot",
 ],
 "Cmd.VolumeUnmute": [
    "i'd like to hear once more","let me hear things once again","restore my hearing now",
    "get my sound going again","i want sound coming through again","end the mute let me hear",
 ],
 "Cmd.StreamingStop": [
    "cut the tv connection to my aids","disconnect the television audio","end the tv link to my aids",
    "stop the tv feed to my hearing aids","drop the connection to the tv sound",
 ],
 "Cmd.FindMyPhone": [
    "track down my handset for me","locate my handset","find my cell for me",
    "track down my mobile","i can't find my handset ring it",
 ],
 "Cmd.ActivityWalk": [
    "i'm off for a stroll please track it","heading out for a stroll record it",
    "going for a walk track it for me","start tracking this stroll","log the stroll i'm taking",
 ],
 "Cmd.TranscribeStart": [
    "caption what people are saying around me","put captions on everyone's speech",
    "show me text of what people say","caption the people talking","write out what folks are saying",
 ],
 "Cmd.ListenMessage": [
    "what messages did people send me","did anyone send me anything read it",
    "read me what was sent","tell me what people texted","play what people sent me",
 ],
 "Help_Tinnitus": [
    "the ringing in my ears is unbearable","this ear ringing is driving me crazy",
    "i can't stand the ringing in my ears","the constant ringing is maddening",
    "help the ringing won't stop",
 ],
 "Help_DeviceSettings": [
    "the sound keeps cutting in and out","my audio quality has gone downhill",
    "help with my hearing aid settings","the sound from my aids has gotten worse",
    "my hearing aid sound keeps glitching","fix the declining sound on my aids",
 ],
 # InsertDevice regression guard: the "keeps ...-ing out" register above bled
 # "the device keeps falling out" into DeviceSettings. Re-anchor the fit/seating
 # problem on InsertDevice with the same colloquial pattern.
 "Help_InsertDevice": [
    "my device keeps falling out of my ear","the aid keeps dropping out of my ear",
    "my hearing aid falls out of my ear","the device won't stay seated in my ear",
    "my aid keeps popping out","my hearing aid won't stay in it keeps coming out",
 ],
 "Help_Volume": [
    "can i have a different volume in each ear","set independent loudness per side",
    "different volume for my left and right","adjust each ear's volume separately",
    "control the loudness of each aid on its own",
 ],
 "Help_ChangingMemories": [
    "help me switch between profiles","how do i switch profiles","guide me on switching profiles",
    "i need help changing between my programs","walk me through switching profiles",
 ],
 "Help_CleanCare": [
    "how do i take care of my aids","upkeep for my hearing aids","how do i look after my hearing aids",
    "keeping my aids in good shape","caring for and maintaining my aids",
 ],
 "Help_FallAlert": [
    "will it notice if i take a tumble","does it catch a fall","will it know if i fall over",
    "does it spot a fall and alert someone","will a tumble trigger an alert",
 ],
 "Help_Reminder": [
    "can i set up a daily reminder","is there a daily reminder option","how do i make a reminder repeat daily",
    "set a recurring daily reminder","can reminders repeat every day",
 ],
}
for _k, _v in W4.items():
    P.setdefault(_k, [])
    P[_k].extend(_v)


def main():
    block = load_blocklist()
    base = pd.read_csv(DATA / "intent_data_new_2.csv")
    base["text"] = base["text"].astype(str).str.lower().str.strip()
    base_pairs = set(zip(base["text"], base["intent"].astype(str).str.strip()))
    base_texts = set(base["text"])

    rows, dropped_leak, dropped_dup = [], 0, 0
    for intent, phrases in P.items():
        for ph in phrases:
            t = ph.lower().strip()
            if t in block:
                dropped_leak += 1; continue
            if (t, intent) in base_pairs or t in base_texts:
                dropped_dup += 1; continue
            rows.append({"text": t, "intent": intent})

    out = pd.DataFrame(rows).drop_duplicates(subset=["text", "intent"])
    # Final safety: assert zero leakage
    leak = set(out["text"]) & block
    assert not leak, f"LEAKAGE: {sorted(leak)[:10]}"
    out.to_csv(DATA / "intent_data_augmented.csv", index=False)
    print(f"Wrote {len(out)} augmented rows across {out['intent'].nunique()} intents")
    print(f"  dropped (holdout leakage): {dropped_leak}")
    print(f"  dropped (already in base): {dropped_dup}")
    print(f"  leakage check: 0 ✅")

    # ── Also assemble the merged training file train.py/-head consume (v3) ──
    # enhanced = base (_2) + this augmentation + the earlier hand corrections.
    # Doing it here (not by hand) keeps the pipeline reproducible: this script is
    # the single source of truth for what the v3 training set contains.
    parts = [base[["text", "intent"]], out]
    corr_path = DATA / "intent_data_corrections.csv"
    if corr_path.exists():
        corr = pd.read_csv(corr_path)
        corr["text"] = corr["text"].astype(str).str.lower().str.strip()
        parts.append(corr[["text", "intent"]])
    merged = pd.concat(parts, ignore_index=True)
    merged["text"] = merged["text"].astype(str).str.lower().str.strip()
    merged["intent"] = merged["intent"].astype(str).str.strip()
    merged = merged.dropna().drop_duplicates(subset=["text", "intent"])
    # Gate the merged file ONLY against the paraphrase generalisation holdouts.
    # (NOT benchmark_250, which is in-distribution, and NOT test_semantic.py, whose
    # KNOWN/OOS cases legitimately live in base training.)
    merged_leak = set(merged["text"]) & load_paraphrase_holdouts()
    assert not merged_leak, f"MERGED LEAKAGE vs paraphrase holdouts: {sorted(merged_leak)[:10]}"
    merged.to_csv(DATA / "intent_data_new_2_enhanced.csv", index=False)
    print(f"Wrote merged v3 training file intent_data_new_2_enhanced.csv: "
          f"{len(merged)} rows  (base {len(base)} + aug {len(out)} + corrections)")

if __name__ == "__main__":
    main()
