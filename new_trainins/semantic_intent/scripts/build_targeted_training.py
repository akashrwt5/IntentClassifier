"""Phase 17/23 — targeted dataset iteration.

Every block here exists because a specific suite failed, not because more data
is generally nice. Generated items are filtered against the leakage keys of
validation, test AND every challenge suite, so nothing we add can leak into
what we measure on.

Failure modes addressed:
  F1  negation      — the model never saw the P2/P3 policy at training time
  F2  direction     — up/down, mute/unmute, start/stop under paraphrase
  F3  cmd_vs_help   — imperative vs interrogative frame on the same content
  F4  near_ood      — another device's volume must not become an aid command
  F5  stt           — offline ASR artifacts on real training sentences
  F6  tail classes  — 13 intents have fewer than 60 examples
"""
from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from build_challenge_sets import stt_corrupt  # noqa: E402
from corrective_pairs import generate as gen_corrective  # noqa: E402
from corrective_pairs import HELD_OUT_FAMILIES, TRAIN_FRAMES  # noqa: E402
from long_forms import gen_contextual  # noqa: E402
import foreign_objects  # noqa: E402
from foreign_objects import gen_foreign_objects  # noqa: E402
from accessories import generate as gen_accessories  # noqa: E402
from role_balance import generate as gen_role_balance  # noqa: E402
import role_balance  # noqa: E402
from symptom_pairs import generate as gen_symptom  # noqa: E402
from common import leakage_key, normalize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FALLBACK = "Default Fallback Intent"
SEED = 99

# --- F1 negation ------------------------------------------------------------
NEG_OPENERS = ["do not", "don't", "please do not", "i do not want you to",
               "i did not ask you to", "no need to", "there is no need to",
               "stop trying to", "i never asked you to"]
NEG_ACTIONS = {
    "turn it up": "Cmd.VolumeIncrease", "turn it down": "Cmd.VolumeDecrease",
    "make it louder": "Cmd.VolumeIncrease", "make it quieter": "Cmd.VolumeDecrease",
    "raise the volume": "Cmd.VolumeIncrease", "lower the volume": "Cmd.VolumeDecrease",
    "mute the aids": "Cmd.VolumeMute", "unmute the aids": "Cmd.VolumeUnmute",
    "switch my program": "Cmd.MemoryChange", "change my memory": "Cmd.MemoryChange",
    "start streaming": "Cmd.StreamingStart", "stop streaming": "Cmd.StreamingStop",
    "set a reminder": "reminders.add", "send a message": "Cmd.SendMessage",
    "play my messages": "Cmd.ListenMessage", "start translating": "Cmd.TranslationStart",
    "start transcribing": "Cmd.TranscribeStart", "find my phone": "Cmd.FindMyPhone",
}
CORRECTIVE = [  # (wrong_action, right_action, right_intent)
    ("quieter", "louder", "Cmd.VolumeIncrease"),
    ("louder", "quieter", "Cmd.VolumeDecrease"),
    ("down", "up", "Cmd.VolumeIncrease"),
    ("up", "down", "Cmd.VolumeDecrease"),
    ("muted", "unmuted", "Cmd.VolumeUnmute"),
    ("unmuted", "muted", "Cmd.VolumeMute"),
    ("stopped", "started", "Cmd.StreamingStart"),
    ("started", "stopped", "Cmd.StreamingStop"),
]
CORRECTIVE_FRAMES = [
    "not {w}, {r}", "i said {r}, not {w}", "no, {r} not {w}",
    "{r} please, not {w}", "i wanted it {r}, not {w}",
]

# --- F2 direction -----------------------------------------------------------
DIRECTION_FRAMES = [
    ("could you {v} the sound", "{v} the sound"),
]
UP_VERBS = ["raise", "lift", "increase", "boost", "step up", "pump up",
            "bring up", "push up", "turn up", "crank up", "amplify"]
DOWN_VERBS = ["lower", "reduce", "decrease", "soften", "step down", "bring down",
              "push down", "turn down", "dial down", "damp down", "cut back"]
UP_FRAMES = ["{v} the volume", "can you {v} the volume", "{v} my hearing aids",
             "{v} the sound a bit", "please {v} the level", "{v} it a little",
             "would you {v} the volume for me", "{v} the audio please"]
OBJ_SWAP = {"the volume": "the sound", "my hearing aids": "my aids"}

# --- F3 cmd vs help ---------------------------------------------------------
CMD_HELP = [
    # (imperative, cmd_intent, interrogative, help_intent)
    ("{v} the volume", "VOL", "how do i {v} the volume", "Help_Volume"),
    ("{v} the volume", "VOL", "where do i {v} the volume", "Help_Volume"),
    ("{v} the volume", "VOL", "can you show me how to {v} the volume", "Help_Volume"),
    ("{v} the volume", "VOL", "what is the way to {v} the volume", "Help_Volume"),
]
MEMORY_TARGETS = ["restaurant", "outdoor", "car", "church", "party", "office",
                  "tv", "music", "quiet", "windy", "meeting"]
MEM_CMD = ["switch to the {m} program", "put me on the {m} setting",
           "change to {m} mode", "go to my {m} memory", "use the {m} program"]
MEM_HELP = ["how do i switch to the {m} program", "how do i get to the {m} setting",
            "where do i find the {m} program", "explain how to use the {m} memory"]

# --- F4 near OOD ------------------------------------------------------------
OTHER_DEVICES = ["television", "tv set", "phone", "mobile", "radio", "laptop",
                 "speaker", "car stereo", "tablet", "alarm clock", "doorbell"]
OOD_FRAMES = ["turn the {d} volume up", "turn the {d} volume down",
              "mute the {d}", "make the {d} louder", "make the {d} quieter",
              "increase the {d} brightness", "pause the {d}"]


def gen_negation(rng) -> list[dict]:
    rows = []
    # P2: bare negation -> no action
    for opener, action in itertools.product(NEG_OPENERS, NEG_ACTIONS):
        rows.append(dict(text=f"{opener} {action}", intent=FALLBACK,
                         source="F1_negation_bare"))
    # P3: corrective -> affirmed action wins
    for (w, r, intent), frame in itertools.product(CORRECTIVE, CORRECTIVE_FRAMES):
        rows.append(dict(text=frame.format(w=w, r=r), intent=intent,
                         source="F1_negation_corrective"))
    # P3 variant with an explicit second clause
    for opener, (w, r, intent) in itertools.product(NEG_OPENERS[:5], CORRECTIVE):
        rows.append(dict(text=f"{opener} make it {w}, make it {r}",
                         intent=intent, source="F1_negation_corrective"))
    rng.shuffle(rows)
    return rows


def gen_direction(rng) -> list[dict]:
    rows = []
    for verbs, intent in ((UP_VERBS, "Cmd.VolumeIncrease"),
                          (DOWN_VERBS, "Cmd.VolumeDecrease")):
        for v, frame in itertools.product(verbs, UP_FRAMES):
            t = frame.format(v=v)
            rows.append(dict(text=t, intent=intent, source="F2_direction"))
            for a, b in OBJ_SWAP.items():
                if a in t:
                    rows.append(dict(text=t.replace(a, b), intent=intent,
                                     source="F2_direction"))
    # symptom -> action, both directions, matched frames
    for a, b in [("i can hardly hear anything", "everything is deafening"),
                 ("this is far too faint", "this is far too harsh"),
                 ("the speech is too weak", "the speech is too strong"),
                 ("sounds are coming through very thin", "sounds are coming through very heavy")]:
        rows.append(dict(text=a, intent="Cmd.VolumeIncrease", source="F2_direction"))
        rows.append(dict(text=b, intent="Cmd.VolumeDecrease", source="F2_direction"))
    # mute / unmute and start / stop
    for t, i in [("switch the sound off for now", "Cmd.VolumeMute"),
                 ("switch the sound on again", "Cmd.VolumeUnmute"),
                 ("no sound at all please", "Cmd.VolumeMute"),
                 ("give me sound again please", "Cmd.VolumeUnmute"),
                 ("put the audio through to my aids", "Cmd.StreamingStart"),
                 ("take the audio off my aids", "Cmd.StreamingStop"),
                 ("open the audio link", "Cmd.StreamingStart"),
                 ("close the audio link", "Cmd.StreamingStop")]:
        rows.append(dict(text=t, intent=i, source="F2_direction"))
    rng.shuffle(rows)
    return rows


def gen_cmd_help(rng) -> list[dict]:
    rows = []
    for v, intent in ([(v, "Cmd.VolumeIncrease") for v in UP_VERBS]
                      + [(v, "Cmd.VolumeDecrease") for v in DOWN_VERBS]):
        for cmd_f, _, help_f, help_i in CMD_HELP:
            rows.append(dict(text=cmd_f.format(v=v), intent=intent,
                             source="F3_cmd_vs_help"))
            rows.append(dict(text=help_f.format(v=v), intent=help_i,
                             source="F3_cmd_vs_help"))
    for m in MEMORY_TARGETS:
        for f in MEM_CMD:
            rows.append(dict(text=f.format(m=m), intent="Cmd.MemoryChange",
                             source="F3_cmd_vs_help"))
        for f in MEM_HELP:
            rows.append(dict(text=f.format(m=m), intent="Help_ChangingMemories",
                             source="F3_cmd_vs_help"))
    rng.shuffle(rows)
    return rows


def gen_near_ood(rng) -> list[dict]:
    rows = [dict(text=f.format(d=d), intent=FALLBACK, source="F4_near_ood")
            for d, f in itertools.product(OTHER_DEVICES, OOD_FRAMES)]
    rng.shuffle(rows)
    return rows


# --- F6 generic "activity goal" ------------------------------------------------
# Product decision: follow the dataset. In this corpus the generic phrase
# "activity goal" is labelled Cmd.ActivityAerobics, while "activity summary /
# overview / stats" is Help_Health. That split is not semantically obvious, so
# the model has to be taught it explicitly rather than inferred from 8 rows.
# The boundary is the NOUN, not the verb — both lists below use the same
# openers on purpose so nothing but the noun carries the label.
ACTIVITY_OPENERS = [
    "show me my", "what is my", "tell me my", "display my", "check my",
    "have i reached my", "did i hit my", "am i close to my", "how far off is my",
    "please show me my", "can you show me my", "what about my", "read out my",
]
AEROBICS_NOUNS = ["activity goal", "activity target", "activity goal today",
                  "activity goal for today", "daily activity goal"]
HEALTH_NOUNS = ["activity summary", "activity overview", "activity stats",
                "activity history", "overall activity"]

MUTE_ON = ["mute", "silence", "shut off the sound on", "turn the sound off on",
           "kill the sound on", "no sound from", "quiet completely"]
MUTE_OFF = ["unmute", "unsilence", "turn the sound back on for",
            "bring the sound back to", "give sound back to", "sound on for",
            "wake the sound on"]
MUTE_OBJ = ["my aids", "my hearing aids", "both aids", "the left aid",
            "the right aid", "everything", "them"]


def gen_activity_goal(rng) -> list[dict]:
    rows = []
    for opener, noun in itertools.product(ACTIVITY_OPENERS, AEROBICS_NOUNS):
        rows.append(dict(text=f"{opener} {noun}", intent="Cmd.ActivityAerobics",
                         source="F6_activity_goal"))
    for opener, noun in itertools.product(ACTIVITY_OPENERS, HEALTH_NOUNS):
        rows.append(dict(text=f"{opener} {noun}", intent="Help_Health",
                         source="F6_activity_goal"))
    rng.shuffle(rows)
    return rows


def gen_mute_direction(rng) -> list[dict]:
    """mute_direction scored 0.63 on the minimal-pair suite. A hearing aid that
    mutes when asked to unmute leaves the user unable to hear why, and unable to
    ask again, so this axis gets its own batch."""
    rows = []
    for verbs, intent in ((MUTE_ON, "Cmd.VolumeMute"), (MUTE_OFF, "Cmd.VolumeUnmute")):
        for v, obj in itertools.product(verbs, MUTE_OBJ):
            rows.append(dict(text=f"{v} {obj}", intent=intent,
                             source="F7_mute_direction"))
            rows.append(dict(text=f"please {v} {obj}", intent=intent,
                             source="F7_mute_direction"))
    rng.shuffle(rows)
    return rows


# --- F8 result-request vs feature-question -----------------------------------
# The real boundary is NOT the question word. Both of these start with "how":
#
#     how much battery is left      -> Cmd.BatteryLevel   (asking for MY data)
#     how do i check the battery    -> Help_Battery       (asking how it works)
#
# The question that decides it: is the user asking the app to DO something it
# has a function for, or asking how that function works? A rule keyed on
# "how/what" gets 10.3% of the corpus wrong, and the wrong 10% is not random —
# it is every data-request intent: BatteryLevel, all 8 Activity intents,
# FindMyPhone.
#
# The map below is read off the dataset, not invented, and it is deliberately
# ASYMMETRIC because the app is asymmetric:
#   * battery has both a result intent and a how-to intent
#   * heart rate has NO result intent — "what's my heart rate" is Help_HeartRate
#   * finding an aid is an action but lives under a Help_ label
# Anything generated here is checked back against the dataset before it is kept.
RESULT_VS_HOWTO = [
    # (topic, result templates, result intent, how-to templates, how-to intent)
    ("battery",
     ["how much battery is left on {obj}", "what is the battery level on {obj}",
      "how much charge is left on {obj}", "is there enough battery left on {obj}",
      "what percent battery does {obj} have"],
     "Cmd.BatteryLevel",
     ["how do i check the battery on {obj}", "how do i charge {obj}",
      "what battery size does {obj} take", "where do i put the battery in {obj}",
      "explain the battery on {obj}"],
     "Help_Battery"),
    ("steps",
     ["what is my step count {when}", "how many steps have i taken {when}",
      "have i hit my step goal {when}", "read out my steps {when}"],
     "Cmd.ActivityStep",
     ["how do i change my step goal", "where do i see my step count",
      "explain how step tracking works", "how do i set my daily step target"],
     "Help_Health"),
    ("calories",
     ["how many calories have i burned {when}", "what is my calorie total {when}",
      "how much energy did i burn {when}", "read out my calories {when}"],
     "Cmd.ActivityCalories",
     ["how do i change my calorie goal", "where do i see calories burned",
      "explain how calories are counted"],
     "Help_Health"),
    ("workout",
     ["how long did i work out {when}", "how long have i been exercising {when}",
      "what is my workout total {when}"],
     "Cmd.ActivityExercise",
     ["how do i log a workout", "where do i see my workout history",
      "explain the exercise tracking feature"],
     "Help_Health"),
    ("volume",
     ["{updown} the volume on {obj}", "make {obj} {louder}",
      "put {obj} {louder}"],
     None,  # filled per-direction below
     ["how do i change the volume on {obj}", "where do i adjust the volume for {obj}",
      "explain how volume works on {obj}", "how do i reset the volume on {obj}"],
     "Help_Volume"),
]
OBJ = ["my aids", "my hearing aids", "the left aid", "the right aid", "both aids"]
WHEN = ["today", "this week", "so far", "this month", "yesterday"]

# Topics where the app has NO separate result intent — the same intent absorbs
# both frames. Generating a Cmd here would invent a function that does not exist.
SINGLE_INTENT_TOPICS = [
    (["what is my heart rate {when}", "read out my heart rate",
      "how does the heart rate feature work", "how do i check my heart rate"],
     "Help_HeartRate"),
    (["what is my recovery rate {when}", "how does heart rate recovery work",
      "how do i check my recovery after exercise"],
     "Help_HeartRateRecovery"),
    (["what is my score {when}", "how do i improve my score",
      "explain what the score measures"],
     "Help_ThriveScore"),
    (["i cannot find my left aid", "i lost one of my hearing aids",
      "where did my right aid go", "how do i search for a lost aid",
      "help me locate my missing aid"],
     "Help_FindMyHearingAids"),
]


def gen_result_vs_howto(rng) -> list[dict]:
    rows = []
    for topic, res_t, res_i, how_t, how_i in RESULT_VS_HOWTO:
        if topic == "volume":
            for t in res_t:
                for obj in OBJ:
                    for updown, louder, intent in (
                            ("turn up", "louder", "Cmd.VolumeIncrease"),
                            ("turn down", "quieter", "Cmd.VolumeDecrease")):
                        rows.append(dict(
                            text=t.format(obj=obj, updown=updown, louder=louder),
                            intent=intent, source="F8_result_vs_howto"))
        else:
            for t in res_t:
                for obj in OBJ:
                    for when in WHEN:
                        rows.append(dict(text=t.format(obj=obj, when=when).strip(),
                                         intent=res_i, source="F8_result_vs_howto"))
        for t in how_t:
            for obj in OBJ:
                rows.append(dict(text=t.format(obj=obj).strip(), intent=how_i,
                                 source="F8_result_vs_howto"))
    for templates, intent in SINGLE_INTENT_TOPICS:
        for t in templates:
            for when in WHEN:
                rows.append(dict(text=t.format(when=when).strip(), intent=intent,
                                 source="F8_result_vs_howto"))
    # dedupe templates that ignored their slots
    seen, uniq = set(), []
    for r in rows:
        if r["text"] not in seen:
            seen.add(r["text"]); uniq.append(r)
    rng.shuffle(uniq)
    return uniq


# Words that legitimately flip an intent. If two near-identical sentences differ
# by one of these, that is a MINIMAL PAIR (which is the point of the exercise),
# not a labelling conflict. Without this list the validator deletes exactly the
# training data the model most needs: "boost the volume" and "cut the volume"
# score 80% similar and carry opposite labels, and both are correct.
POLARITY = {
    "up", "down", "upward", "downward", "higher", "lower", "louder", "quieter",
    # The comparatives were here but the base adjectives were not, so
    # "the volume is far too loud" was dropped for contradicting
    # "the volume is too low" — a genuine minimal pair whose ONLY differing
    # token is the direction word. These carry direction as much as "louder".
    "loud", "low", "quiet", "high", "soft", "faint", "faintly", "loudly",
    "sharper", "sharp", "higher", "muffled", "harsh", "booming",
    "softer", "harder", "raise", "lift", "increase", "boost", "amplify",
    "reduce", "decrease", "soften", "cut", "drop", "dial", "damp",
    "mute", "unmute", "silence", "unsilence", "off", "on", "back",
    "start", "stop", "begin", "end", "halt", "pause", "resume", "open", "close",
    "not", "no", "never", "dont", "add", "complete", "done", "finish",
    "send", "play", "listen", "record", "walk", "walking", "run", "running",
    "jog", "cycle", "cycling", "bike", "step", "steps", "stand", "standing",
    "aerobics", "exercise", "workout", "calorie", "calories",
    "goal", "summary", "overview", "stats", "history",
    "phone", "mobile", "aid", "aids", "earpiece",
    # the "whose device is it" axis — policy P5
    "tv", "television", "radio", "speaker", "laptop", "tablet", "stereo",
    "doorbell", "clock", "microwave", "thermostat", "car",
    # the "result vs how-to" axis — policy P1
    "how", "what", "where", "why", "explain", "guide", "way", "show",
    # the "named content vs routed audio" axis — F9
    "youtube", "spotify", "pandora", "netflix", "sinatra", "beatles",
    "playlist", "podcast", "audiobook", "jazz", "music", "media", "audio",
    # the "pair the device vs route the audio" axis — F10
    "sync", "pair", "connect", "link", "bluetooth", "streamer",
    # the "accessory vs phone/app" axis — F15. Help_Accessories and
    # Help_Pairing take the same frames and are separated ONLY by the object,
    # so the object word is what carries the label.
    "mic", "microphone", "accessory", "accessories", "app", "auracast",
    "connector", "companion", "handset", "table", "partner",
    # streaming direction — Cmd.StreamingStop is the smaller class and loses
    # ties, exactly as Cmd.VolumeDecrease did before F2
    "done", "finished", "unhook", "disconnect", "release", "join", "leave",
    "tune", "hook", "pipe", "route", "beam",
}

# F14 extends the P5 "whose device is it" axis. Without this the validator
# deletes the entire batch: "the air fryer is far too loud" has a nearest real
# neighbour of "it is far too loud" -> Cmd.VolumeDecrease, a different label,
# and the only differing token is the appliance name. That token IS the reason
# the label differs, which is precisely what POLARITY is for — the same
# exemption the volume minimal pairs needed. Sourced from foreign_objects so
# the two files cannot drift apart.
POLARITY |= {w for obj in
             (foreign_objects.SWITCHABLE
              + [o for o, _ in foreign_objects.SETTABLE]
              + foreign_objects.TRAIN_HEALTH)
             for w in obj.split()}


# --- F9 named external content is NOT a streaming command ---------------------
# 9 of the 21 distinct false executions were Default Fallback rows that the
# model fired a command on, and the biggest cluster was named content:
# "stream, youtube music" scored 0.997 on Cmd.StreamingStart.
#
# The dataset's boundary is what is being asked for, not the verb:
#     stream the music / play media through my hearing aids  -> Cmd.StreamingStart
#         (route whatever is playing INTO the aids)
#     play hank williams junior / stream youtube music        -> Fallback
#         (fetch a NAMED service, artist or track — the aid cannot do this)
# Both use "stream" and "play". Only the object differs.
NAMED_CONTENT = [
    "youtube music", "spotify", "pandora", "apple music", "netflix",
    "amazon music", "an audiobook", "a podcast", "the radio station",
    "frank sinatra", "hank williams junior", "some jazz", "the beatles",
    "my playlist", "the top forty", "a comedy special", "the football match",
]
CONTENT_VERBS = ["play", "stream", "put on", "start", "queue up", "shuffle"]
ROUTING_OBJECTS = [
    "the tv audio", "the television sound", "the remote mic", "the presenter",
    "whatever is playing", "the media", "the audio source", "the tv stream",
]
ROUTING_VERBS = ["stream", "send", "route", "put through", "pipe"]

# --- F10 pairing the DEVICE vs routing the AUDIO ------------------------------
# "sync hearing aid to tv" is Help_Pairing but the model fired
# Cmd.StreamingStart on all three variants. The dataset also has "sync the audio
# to my aids" as Cmd.StreamingStart, so the verb cannot decide it — the OBJECT
# does. Syncing the device is pairing; syncing the audio is streaming.
PAIR_VERBS = ["sync", "connect", "pair", "link", "hook up"]
PAIR_DEVICE_OBJ = ["my hearing aid to the tv", "my aids to the television",
                   "my hearing aids to the phone", "my aid to bluetooth",
                   "my hearing aids to my laptop", "my aids to the streamer"]
PAIR_AUDIO_OBJ = ["the audio to my aids", "the sound to my hearing aids",
                  "the tv audio to my ears", "the media audio to my aids"]


def gen_named_content(rng) -> list[dict]:
    rows = []
    for v, c in itertools.product(CONTENT_VERBS, NAMED_CONTENT):
        rows.append(dict(text=f"{v} {c}", intent=FALLBACK,
                         source="F9_named_content"))
    for v, o in itertools.product(ROUTING_VERBS, ROUTING_OBJECTS):
        rows.append(dict(text=f"{v} {o} to my aids", intent="Cmd.StreamingStart",
                         source="F9_named_content"))
    rng.shuffle(rows)
    return rows


def gen_pair_vs_route(rng) -> list[dict]:
    rows = []
    for v, o in itertools.product(PAIR_VERBS, PAIR_DEVICE_OBJ):
        rows.append(dict(text=f"{v} {o}", intent="Help_Pairing",
                         source="F10_pair_vs_route"))
    for v, o in itertools.product(PAIR_VERBS, PAIR_AUDIO_OBJ):
        rows.append(dict(text=f"{v} {o}", intent="Cmd.StreamingStart",
                         source="F10_pair_vs_route"))
    rng.shuffle(rows)
    return rows


def validate_against_dataset(rows: list[dict], df: pd.DataFrame,
                             threshold: int = 85) -> tuple[list[dict], list[dict]]:
    """Drop a generated row when its closest real dataset sentence carries a
    different label AND the two differ by nothing that could justify the flip.

    Generated data that contradicts the corpus teaches the model the opposite of
    what the app actually does — that is how a plausible-looking augmentation
    batch silently makes a model worse. But a naive similarity check cannot tell
    that apart from a minimal pair, because a minimal pair is BY CONSTRUCTION a
    near-identical sentence with a different label. The polarity test is what
    separates them: if the differing tokens contain a direction/negation/target
    word, the different label is expected.
    """
    from rapidfuzz import fuzz, process

    corpus = df["norm"].tolist()
    labels = df["intent"].tolist()
    keep, conflicts = [], []
    for r in rows:
        mine = normalize(r["text"])
        best = process.extractOne(mine, corpus, scorer=fuzz.token_sort_ratio)
        if not best or best[1] < threshold or labels[best[2]] == r["intent"]:
            keep.append(r)
            continue
        diff = set(mine.split()) ^ set(best[0].split())
        if diff & POLARITY:
            keep.append(r)          # legitimate minimal pair
            continue
        conflicts.append(dict(**r, nearest=df["text"].iloc[best[2]],
                              dataset_label=labels[best[2]],
                              sim=round(best[1], 1),
                              differing_tokens=" ".join(sorted(diff))))
    return keep, conflicts


def gen_stt_augmentation(train: pd.DataFrame, rng, per_row: int = 1) -> list[dict]:
    rows = []
    srng = random.Random(SEED + 5)
    for _, r in train.iterrows():
        for _ in range(per_row):
            c, ops = stt_corrupt(normalize(r["text"]), srng)
            if c != normalize(r["text"]):
                rows.append(dict(text=c, intent=r["intent"],
                                 source=f"F5_stt:{ops}"))
    rng.shuffle(rows)
    return rows


def main() -> None:
    rng = random.Random(SEED)
    train = pd.read_csv(DATA / "train.csv")

    # forbidden keys: anything we evaluate on
    forbidden_keys, forbidden_norm = set(), set()
    for f in ["validation.csv", "test.csv", "hard_negative_test.csv",
              "ood_test.csv", "minimal_pair_test.csv", "negation_test.csv",
              "contextual_test.csv", "accessories_test.csv", "stt_test.csv"]:
        d = pd.read_csv(DATA / f)
        forbidden_keys |= set(d["text"].map(leakage_key))
        forbidden_norm |= set(d["text"].map(normalize))

    blocks = {
        "F1_negation": gen_negation(rng),
        "F2_direction": gen_direction(rng),
        "F3_cmd_vs_help": gen_cmd_help(rng),
        "F4_near_ood": gen_near_ood(rng),
        "F6_activity_goal": gen_activity_goal(rng),
        "F7_mute_direction": gen_mute_direction(rng),
        "F8_result_vs_howto": gen_result_vs_howto(rng),
        "F9_named_content": gen_named_content(rng),
        "F10_pair_vs_route": gen_pair_vs_route(rng),
        "F11_corrective_all_families": gen_corrective(
            yaml.safe_load((ROOT / "configs" / "intents.yaml").read_text())
            ["families"], TRAIN_FRAMES, rng, "F11_corrective_all_families",
            exclude_families=HELD_OUT_FAMILIES),
        "F12_symptom_shortcut": gen_symptom(rng, "train"),
        "F13_long_form": [dict(text=r["text"], intent=r["intent"],
                              source="F13_long_form")
                          for r in gen_contextual(rng, per_payload=6,
                                                  split="train")],
        "F14_foreign_object": gen_foreign_objects(rng),
        "F15_accessories": gen_accessories(rng, "train"),
        "F16_role_balance": gen_role_balance(rng),
        "F5_stt": gen_stt_augmentation(train, rng),
    }

    raw = pd.read_csv(DATA / "raw" / "en.csv")
    raw["norm"] = raw["text"].map(normalize)

    kept, dropped = [], 0
    seen = set(train["text"].map(normalize))
    all_conflicts = []
    for name, rows in blocks.items():
        n_before = len(rows)
        # F1 (negation) and F5 (STT) are exempt: a negation example is
        # SUPPOSED to differ from its affirmative neighbour, and an STT row is a
        # deliberate corruption of a row whose label is already correct.
        if name not in ("F1_negation", "F5_stt", "F11_corrective_all_families"):
            rows, conf = validate_against_dataset(rows, raw)
            all_conflicts += conf
        for r in rows:
            n, k = normalize(r["text"]), leakage_key(r["text"])
            if n in forbidden_norm or k in forbidden_keys or n in seen:
                dropped += 1
                continue
            seen.add(n)
            kept.append(r)
        print(f"{name:16s} generated={n_before:5d} kept={sum(1 for r in kept if r['source'].startswith(name.split('_')[0])):5d}")

    if all_conflicts:
        cdf = pd.DataFrame(all_conflicts)
        cdf.to_csv(DATA / "augmentation_conflicts.csv", index=False)
        print(f"\n{len(cdf)} generated rows contradicted the dataset and were "
              f"DROPPED -> data/augmentation_conflicts.csv")
        print(cdf[["text", "intent", "nearest", "dataset_label", "sim"]]
              .head(12).to_string(index=False))

    aug = pd.DataFrame(kept)
    aug.to_csv(DATA / "targeted_augmentation.csv", index=False)

    combined = pd.concat([
        train.assign(source="original"),
        aug[["text", "intent", "source"]],
    ], ignore_index=True)
    combined.to_csv(DATA / "train_augmented.csv", index=False)

    # final safety check
    ck = set(combined["text"].map(leakage_key)) & forbidden_keys
    cn = set(combined["text"].map(normalize)) & forbidden_norm
    print(f"\naugmented train: {len(train)} -> {len(combined)} "
          f"(+{len(aug)}), dropped {dropped} colliding")
    print(f"VERIFY leakage against all eval suites: keys={len(ck)} norm={len(cn)} (expect 0/0)")

    # --- shortcut check -----------------------------------------------------
    # A direction word whose bare presence predicts the label is a shortcut the
    # training set is actively teaching, and a model that uses it is fitting,
    # not failing. "faint" was 100% Cmd.VolumeIncrease and "quieter" 63.6%
    # Cmd.VolumeDecrease when the shipped model answered
    # "it's a bit quieter here can you make it louder" with VolumeDecrease.
    #
    # Only DUAL-ROLE words are checked. "faint" at 100% is not a shortcut, it
    # is a fact — nothing describes a too-quiet room and then asks for less
    # volume. Words with two honest roles must not lean on either.
    import re as _re
    low = combined["text"].astype(str).str.lower()
    print("\nshortcut check on dual-role direction words (must stay under 0.65):")
    worst = []
    for w, _, _ in role_balance.DUAL_ROLE:
        hit = combined[low.str.contains(rf"\b{w}\b", na=False, regex=True)]
        if len(hit) < 10:
            continue
        top, n = hit["intent"].value_counts().index[0], hit["intent"].value_counts().iloc[0]
        share = n / len(hit)
        flag = "  <-- SHORTCUT" if share > 0.65 else ""
        print(f"  {w:9} {len(hit):5} rows   {share:.3f}  {top}{flag}")
        if share > 0.65:
            worst.append((w, round(share, 3), top))
    if worst:
        print("\nWARNING: these words still predict their label on their own.")
        print("A model that answers from them is fitting the data it was given.")
        print("Balance the roles in role_balance.py rather than adding more words.")
    assert not ck and not cn
    print(combined["source"].str.split(":").str[0].value_counts().to_string())


if __name__ == "__main__":
    main()
