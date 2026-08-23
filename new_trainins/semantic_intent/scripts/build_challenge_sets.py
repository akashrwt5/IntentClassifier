"""Phases 4-9 — Minimal pairs, hard negatives, negation, contextual, STT, OOD.

Everything here is authored to be HELD OUT: it is never added to train unless
Phase 17 (targeted iteration) explicitly promotes a slice of it, and the promoted
slice is tracked so the challenge tests stay honest.

Outputs into data/challenge/.
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import leakage_key, normalize  # noqa: E402
from corrective_pairs import TEST_FRAMES  # noqa: E402
from corrective_pairs import generate as gen_corrective  # noqa: E402
from long_forms import gen_contextual, gen_negation  # noqa: E402
from symptom_pairs import generate as gen_symptom  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "challenge"
FALLBACK = "Default Fallback Intent"

# ---------------------------------------------------------------------------
# 1. MINIMAL PAIRS — only the intent direction changes.
# ---------------------------------------------------------------------------
MINIMAL_PAIRS: list[tuple[str, str, str, str, str]] = [
    # (text_a, intent_a, text_b, intent_b, axis)
    ("nudge the level higher", "Cmd.VolumeIncrease", "nudge the level lower", "Cmd.VolumeDecrease", "volume_direction"),
    ("it is too quiet, raise it", "Cmd.VolumeIncrease", "it is too loud, lower it", "Cmd.VolumeDecrease", "volume_direction"),
    ("bump the sound up a notch", "Cmd.VolumeIncrease", "bump the sound down a notch", "Cmd.VolumeDecrease", "volume_direction"),
    ("i need a little more volume", "Cmd.VolumeIncrease", "i need a little less volume", "Cmd.VolumeDecrease", "volume_direction"),
    ("tone things upward", "Cmd.VolumeIncrease", "tone things downward", "Cmd.VolumeDecrease", "volume_direction"),
    ("the sound is too low, raise it", "Cmd.VolumeIncrease", "the sound is too high, drop it", "Cmd.VolumeDecrease", "volume_direction"),
    ("give me more sound", "Cmd.VolumeIncrease", "give me less sound", "Cmd.VolumeDecrease", "volume_direction"),
    ("lift the loudness", "Cmd.VolumeIncrease", "drop the loudness", "Cmd.VolumeDecrease", "volume_direction"),
    ("kill all output", "Cmd.VolumeMute", "bring sound back on", "Cmd.VolumeUnmute", "mute_direction"),
    ("silence my aids", "Cmd.VolumeMute", "unsilence my aids", "Cmd.VolumeUnmute", "mute_direction"),
    ("mute both hearing aids", "Cmd.VolumeMute", "unmute both hearing aids", "Cmd.VolumeUnmute", "mute_direction"),
    ("cut all sound from my aids", "Cmd.VolumeMute", "restore sound to my aids", "Cmd.VolumeUnmute", "mute_direction"),
    ("begin streaming to my aids", "Cmd.StreamingStart", "end streaming to my aids", "Cmd.StreamingStop", "stream_direction"),
    ("start the tv audio", "Cmd.StreamingStart", "stop the tv audio", "Cmd.StreamingStop", "stream_direction"),
    ("hook me into the television audio", "Cmd.StreamingStart", "unhook me from the television audio", "Cmd.StreamingStop", "stream_direction"),
    ("set a reminder for my pills", "reminders.add", "my pill reminder is done", "reminders.complete", "reminder_direction"),
    ("remind me to call the clinic", "reminders.add", "i already called the clinic, clear that reminder", "reminders.complete", "reminder_direction"),
    ("record a message for my daughter", "Cmd.SendMessage", "play the message from my daughter", "Cmd.ListenMessage", "message_direction"),
    ("dictate a quick note for my son", "Cmd.SendMessage", "replay what my son sent", "Cmd.ListenMessage", "message_direction"),
    # Cmd vs Help — the highest-volume confusion axis across 57 intents
    ("boost my hearing aid loudness", "Cmd.VolumeIncrease", "how would i boost my hearing aid loudness", "Help_Volume", "cmd_vs_help"),
    ("soften what i am hearing", "Cmd.VolumeDecrease", "how would i soften what i am hearing", "Help_Volume", "cmd_vs_help"),
    ("shut off all output", "Cmd.VolumeMute", "how would i shut off all output", "Help_Volume", "cmd_vs_help"),
    ("flip me over to the cafe setting", "Cmd.MemoryChange", "how would i flip over to the cafe setting", "Help_ChangingMemories", "cmd_vs_help"),
    ("jump to setting number three", "Cmd.MemoryChange", "how would i jump to setting number three", "Help_ChangingMemories", "cmd_vs_help"),
    ("kick off live captions", "Cmd.TranscribeStart", "how would i kick off live captions", "Help_Transcribe", "cmd_vs_help"),
    ("translate this for me", "Cmd.TranslationStart", "how does translating work", "Help_Translate", "cmd_vs_help"),
    ("how much battery is left", "Cmd.BatteryLevel", "what kind of battery goes in my aid", "Help_Battery", "cmd_vs_help"),
    ("locate my mobile", "Cmd.FindMyPhone", "locate my left earpiece", "Help_FindMyHearingAids", "find_target"),
    ("where did my mobile go", "Cmd.FindMyPhone", "which earpiece is missing", "Help_FindMyHearingAids", "find_target"),
    ("kick off television audio", "Cmd.StreamingStart", "how would i configure a television box", "Help_Accessories", "cmd_vs_help"),
    ("remind me at six", "reminders.add", "what abilities does the reminder tool have", "Help_Reminder", "cmd_vs_help"),
    # Activity type swaps — same frame, different activity
    ("how far have i walked today", "Cmd.ActivityWalk", "how far have i cycled today", "Cmd.ActivityCycle", "activity_type"),
    ("log this stroll of mine", "Cmd.ActivityWalk", "log the jog i just did", "Cmd.ActivityRun", "activity_type"),
    ("what is my step count today", "Cmd.ActivityStep", "how long did i stand today", "Cmd.ActivityStand", "activity_type"),
    ("log my gym session", "Cmd.ActivityExercise", "log this cardio dance session", "Cmd.ActivityAerobics", "activity_type"),
    ("how long was my gym session", "Cmd.ActivityExercise", "how much energy did my gym session burn", "Cmd.ActivityCalories", "activity_type"),
    ("did i reach my walking target", "Cmd.ActivityWalk", "did i reach my upright target", "Cmd.ActivityStand", "activity_type"),
    # Health how-to swaps
    ("explain my heart rate reading", "Help_HeartRate", "explain my heart rate recovery reading", "Help_HeartRateRecovery", "health_metric"),
    ("where do i see my activity summary", "Help_Health", "where do i see my engagement score", "Help_ThriveScore", "health_metric"),
    # Feature how-to swaps
    ("what abilities does edge mode have", "Help_EdgeMode", "what abilities does mask mode have", "Help_MaskMode", "feature_swap"),
    ("how do i turn on the tinnitus masker", "Help_Tinnitus", "how do i turn on speech enhancement", "Help_IntelliVoice", "feature_swap"),
    ("how do i clean my aids", "Help_CleanCare", "how do i put in my aids", "Help_InsertDevice", "feature_swap"),
    ("how do i pair my aids", "Help_Pairing", "how do i run a self check on my aids", "Help_SelfCheck", "feature_swap"),
    ("how do i share my data with my son", "Help_HearShare", "how do i book a remote appointment", "Help_HearingCareAnywhereConnect", "feature_swap"),
]

# ---------------------------------------------------------------------------
# 2. HARD NEGATIVES — heavy lexical overlap with a sibling, opposite label.
#    The distractor word from the sibling class is present on purpose.
# ---------------------------------------------------------------------------
HARD_NEGATIVES: list[tuple[str, str, str]] = [
    ("it is too loud in here, can you make it quieter", "Cmd.VolumeDecrease", "contains 'loud'"),
    ("it is too quiet in here, can you make it louder", "Cmd.VolumeIncrease", "contains 'quiet'"),
    ("everything sounds too loud, bring it down", "Cmd.VolumeDecrease", "contains 'loud'"),
    ("everything sounds too soft, bring it up", "Cmd.VolumeIncrease", "contains 'soft'"),
    ("i turned it up too far, take it back down", "Cmd.VolumeDecrease", "contains 'up'"),
    ("i turned it down too far, take it back up", "Cmd.VolumeIncrease", "contains 'down'"),
    ("stop making it louder and lower it instead", "Cmd.VolumeDecrease", "contains 'louder'"),
    ("stop making it quieter and raise it instead", "Cmd.VolumeIncrease", "contains 'quieter'"),
    ("that is far too much volume, cut it back", "Cmd.VolumeDecrease", "contains 'volume'"),
    ("the volume dropped too far, put it back up", "Cmd.VolumeIncrease", "contains 'dropped'"),
    ("i do not want it muted, i want it louder", "Cmd.VolumeIncrease", "contains 'muted'"),
    ("i do not want it louder, i want it muted", "Cmd.VolumeMute", "contains 'louder'"),
    ("it is muted, turn the sound back on", "Cmd.VolumeUnmute", "contains 'muted'"),
    ("the sound is on, mute it", "Cmd.VolumeMute", "contains 'on'"),
    ("i am not asking to change programs, just turn it up", "Cmd.VolumeIncrease", "contains 'programs'"),
    ("i am not asking to change the volume, switch my program", "Cmd.MemoryChange", "contains 'volume'"),
    ("do not start streaming, stop it", "Cmd.StreamingStop", "contains 'start'"),
    ("do not stop streaming, start it", "Cmd.StreamingStart", "contains 'stop'"),
    ("i do not need a new reminder, mark the old one done", "reminders.complete", "contains 'reminder'"),
    ("i do not want to mark it done, add another reminder", "reminders.add", "contains 'done'"),
    ("do not play the message, record a new one", "Cmd.SendMessage", "contains 'play'"),
    ("do not record a message, play the one i got", "Cmd.ListenMessage", "contains 'record'"),
    ("i am not looking for my phone, i lost a hearing aid", "Help_FindMyHearingAids", "contains 'phone'"),
    ("i am not looking for my hearing aid, i lost my phone", "Cmd.FindMyPhone", "contains 'hearing aid'"),
    ("do not tell me how, just change the volume", "Cmd.VolumeDecrease", "how-to distractor"),
    ("do not change anything, just tell me how volume works", "Help_Volume", "command distractor"),
    ("i did not ask about the battery type, just tell me the charge", "Cmd.BatteryLevel", "contains 'battery type'"),
    ("i did not ask for the charge level, what battery size do i need", "Help_Battery", "contains 'charge'"),
    ("not my step count, my stand time", "Cmd.ActivityStand", "contains 'step'"),
    ("not my stand time, my step count", "Cmd.ActivityStep", "contains 'stand'"),
    ("not my heart rate, my heart rate recovery", "Help_HeartRateRecovery", "contains 'heart rate'"),
    ("not the recovery number, just my heart rate", "Help_HeartRate", "contains 'recovery'"),
    ("not edge mode, i meant mask mode", "Help_MaskMode", "contains 'edge mode'"),
    ("not mask mode, i meant edge mode", "Help_EdgeMode", "contains 'mask mode'"),
    ("i do not want to transcribe, i want to translate", "Cmd.TranslationStart", "contains 'transcribe'"),
    ("i do not want to translate, i want to transcribe", "Cmd.TranscribeStart", "contains 'translate'"),
]

# ---------------------------------------------------------------------------
# 3. NEGATION — governed by policies P2 and P3 in configs/intents.yaml.
# ---------------------------------------------------------------------------
NEGATION: list[tuple[str, str, str]] = [
    # P2: bare negation, no alternative -> no action -> fallback
    ("do not make it louder", FALLBACK, "P2_bare_negation"),
    ("do not turn it up", FALLBACK, "P2_bare_negation"),
    ("do not turn it down", FALLBACK, "P2_bare_negation"),
    ("please do not change the volume", FALLBACK, "P2_bare_negation"),
    ("do not mute my aids", FALLBACK, "P2_bare_negation"),
    ("do not switch my program", FALLBACK, "P2_bare_negation"),
    ("do not start streaming", FALLBACK, "P2_bare_negation"),
    ("do not set a reminder", FALLBACK, "P2_bare_negation"),
    ("i did not ask you to lower it", FALLBACK, "P2_bare_negation"),
    ("i never said to change anything", FALLBACK, "P2_bare_negation"),
    ("stop, do not do that", FALLBACK, "P2_bare_negation"),
    ("no, leave it alone", FALLBACK, "P2_bare_negation"),
    ("do not translate anything", FALLBACK, "P2_bare_negation"),
    ("do not send that message", FALLBACK, "P2_bare_negation"),
    # P3: corrective negation, alternative stated -> the affirmed action
    ("not quieter, louder", "Cmd.VolumeIncrease", "P3_corrective"),
    ("not louder, quieter", "Cmd.VolumeDecrease", "P3_corrective"),
    ("i said down, not up", "Cmd.VolumeDecrease", "P3_corrective"),
    ("i said up, not down", "Cmd.VolumeIncrease", "P3_corrective"),
    ("no, mute it instead", "Cmd.VolumeMute", "P3_corrective"),
    ("no, unmute it instead", "Cmd.VolumeUnmute", "P3_corrective"),
    ("not the restaurant program, the outdoor one", "Cmd.MemoryChange", "P3_corrective"),
    ("not stop, start the streaming", "Cmd.StreamingStart", "P3_corrective"),
    ("not start, stop the streaming", "Cmd.StreamingStop", "P3_corrective"),
    ("not a new reminder, complete the one i have", "reminders.complete", "P3_corrective"),
    ("do not raise it, drop it a bit", "Cmd.VolumeDecrease", "P3_corrective"),
    ("do not drop it, raise it a bit", "Cmd.VolumeIncrease", "P3_corrective"),
]

# ---------------------------------------------------------------------------
# 4. CONTEXTUAL — long, natural, polite, disfluent.
# ---------------------------------------------------------------------------
CONTEXTUAL: list[tuple[str, str]] = [
    ("i am sitting in a noisy restaurant and i am having real trouble following the person across the table, could you please increase the volume a bit", "Cmd.VolumeIncrease"),
    ("um so the television is on and it is honestly blasting right now, would you mind bringing my hearing aids down a little", "Cmd.VolumeDecrease"),
    ("i am about to go into a meeting and i do not want any sound at all coming through, please shut them off", "Cmd.VolumeMute"),
    ("okay the meeting is over now so you can go ahead and put the sound back on for me", "Cmd.VolumeUnmute"),
    ("we just walked into the restaurant so could you put me on the restaurant setting please", "Cmd.MemoryChange"),
    ("i am heading out for my afternoon walk now so please start tracking it", "Cmd.ActivityWalk"),
    ("before i leave the house i want to know whether these will last me until this evening", "Cmd.BatteryLevel"),
    ("i have set my phone down somewhere in the house again and i cannot find it anywhere, can you help", "Cmd.FindMyPhone"),
    ("my wife is going to be speaking spanish with her family tonight and i would like to follow along, can you start that up", "Cmd.TranslationStart"),
    ("could you please remind me tomorrow morning at eight that i need to take my blood pressure tablets", "reminders.add"),
    ("i wanted to ask, when i am watching the television in the evening how exactly do i get the sound to come straight into my aids", "Help_Accessories"),
    ("my audiologist mentioned something about her being able to adjust these without me coming in, how does that actually work", "Help_RemoteProgramming"),
    ("i have been getting this ringing in my ears in the evenings and i believe there is a feature for that, can you explain it", "Help_Tinnitus"),
    ("my son wants to keep an eye on how i am getting on with these, is there a way to let him see that", "Help_HearShare"),
    ("every morning i struggle a bit to get the right one seated properly in my ear, is there a trick to it", "Help_InsertDevice"),
    ("i think one of them might not be working properly today, is there some kind of test i can run", "Help_SelfCheck"),
    ("i updated the app last night and everything looks different now, what has actually changed", "Help_WhatsNew"),
    ("when i am out in the wind at the golf course it gets very rough, i heard there is a mode that helps with that", "Help_EdgeMode"),
    ("if i were to have a fall while i am home on my own, what would these actually do about it", "Help_FallAlert"),
    ("i keep hearing about this score in the app and i do not really understand what it is measuring or how to make it go up", "Help_ThriveScore"),
    ("so i went for quite a long bike ride this morning and i am curious how far i actually got", "Cmd.ActivityCycle"),
    ("could you tell me roughly how many calories i have burned since i got up this morning", "Cmd.ActivityCalories"),
    ("i would like to leave a quick voice note for my daughter letting her know i got home safely", "Cmd.SendMessage"),
    ("there was a message that came through earlier while i was out, could you play it for me now", "Cmd.ListenMessage"),
    ("that prescription pickup i asked you about earlier, i have taken care of it so you can clear it", "reminders.complete"),
]

# ---------------------------------------------------------------------------
# 5. OOD — near and far. None of these are supported commands.
# ---------------------------------------------------------------------------
OOD_NEAR = [
    "turn the television volume down",
    "make the phone louder",
    "increase the screen brightness",
    "turn up my car radio",
    "lower the thermostat",
    "mute the microwave beeping",
    "switch my phone to silent mode",
    "pair my headphones to the laptop",
    "find my car keys",
    "where did i leave my glasses",
    "how is my blood pressure today",
    "what is my blood sugar reading",
    "check my sleep score",
    "how many hours did i sleep",
    "call my audiologist",
    "book me a dentist appointment",
    "text my son that i am running late",
    "read my emails to me",
    "start recording a video",
    "translate this document into french",
    "transcribe the meeting recording on my laptop",
    "turn on the living room lights",
    "set my oven to 180 degrees",
    "how far is it to the pharmacy",
    "order more hearing aid batteries online",
    "how much do new hearing aids cost",
    "is my warranty still valid",
    "what is the wifi password",
    "update my phone software",
    "increase the font size on my phone",
]
OOD_FAR = [
    "what is the weather going to be tomorrow",
    "amuse me with something funny",
    "play some jazz music",
    "what time is it in tokyo",
    "who won the football last night",
    "what is the capital of portugal",
    "how do i make a lasagne",
    "book me a flight to dublin",
    "what is twelve times fourteen",
    "read me the news headlines",
    "how high is the tallest mountain on earth",
    "define photosynthesis",
    "what is the exchange rate for euros",
    "sing me a song",
    "how do i change a flat tyre",
]

# ---------------------------------------------------------------------------
# 6. STT corruption — modelled on real offline-ASR failure modes, not random
#    character noise.
# ---------------------------------------------------------------------------
HOMOPHONES = {
    "louder": ["lauder", "lowder"], "quieter": ["quiter", "quieta"],
    "volume": ["volumn", "volum", "value"], "mute": ["moot", "mut", "newt"],
    "unmute": ["un mute", "on mute"], "aid": ["aide", "8"], "aids": ["aides", "8s"],
    "hearing": ["herring", "hearin"], "memory": ["memery", "mammary"],
    "memories": ["memorys", "mammaries"], "program": ["programme", "pro gram"],
    "tinnitus": ["tinitis", "tenitus", "tinnitis"], "streaming": ["streamin", "screaming"],
    "stream": ["scream"], "transcribe": ["transcribed", "trans cribe"],
    "translate": ["trans late", "translated"], "battery": ["batter e", "batteries"],
    "pairing": ["paring", "pearing"], "pair": ["pear", "pare"],
    "reminder": ["remind her", "reminda"], "remind": ["remind", "re mind"],
    "turn": ["turned", "tern"], "down": ["dow", "doun"], "up": ["ep"],
    "edge": ["hedge", "edg"], "mask": ["masque", "mast"],
    "thrive": ["thrived", "drive"], "steps": ["step", "stepped"],
    "calories": ["calorie", "colories"], "increase": ["in crease", "increased"],
    "decrease": ["de crease", "decreased"], "phone": ["fone", "foam"],
    "message": ["messege", "massage"], "settings": ["setting", "sittings"],
    "cros": ["cross"], "sound": ["sounds", "sund"],
}
FILLER = ["um", "uh", "er"]


def stt_corrupt(text: str, rng: random.Random) -> tuple[str, str]:
    """Return (corrupted_text, applied_ops). Applies 1-2 realistic ASR ops."""
    toks = text.split()
    ops: list[str] = []
    choices = ["homophone", "drop_function_word", "truncate_tail", "filler",
               "merge_boundary", "drop_plural"]
    rng.shuffle(choices)
    applied = 0
    for op in choices:
        if applied >= 2:
            break
        if op == "homophone":
            idx = [i for i, t in enumerate(toks) if t in HOMOPHONES]
            if idx:
                i = rng.choice(idx)
                toks[i] = rng.choice(HOMOPHONES[toks[i]])
                ops.append("homophone"); applied += 1
        elif op == "drop_function_word":
            idx = [i for i, t in enumerate(toks)
                   if t in {"the", "a", "my", "please", "can", "you", "to", "is", "it"}]
            if idx and len(toks) > 3:
                i = rng.choice(idx)
                toks.pop(i)
                ops.append("drop_function_word"); applied += 1
        elif op == "truncate_tail":
            if len(toks) >= 6:
                toks = toks[: len(toks) - 1]
                ops.append("truncate_tail"); applied += 1
        elif op == "filler":
            toks.insert(0, rng.choice(FILLER))
            ops.append("filler"); applied += 1
        elif op == "merge_boundary":
            if len(toks) >= 3:
                i = rng.randrange(len(toks) - 1)
                toks[i : i + 2] = [toks[i] + toks[i + 1]]
                ops.append("merge_boundary"); applied += 1
        elif op == "drop_plural":
            idx = [i for i, t in enumerate(toks) if len(t) > 4 and t.endswith("s")]
            if idx:
                i = rng.choice(idx)
                toks[i] = toks[i][:-1]
                ops.append("drop_plural"); applied += 1
    return " ".join(toks), "+".join(ops) if ops else "none"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(13)

    # minimal pairs -> long format with pair ids
    mp = []
    for i, (ta, ia, tb, ib, axis) in enumerate(MINIMAL_PAIRS):
        mp.append(dict(text=ta, intent=ia, pair_id=f"mp{i:03d}", side="a", axis=axis))
        mp.append(dict(text=tb, intent=ib, pair_id=f"mp{i:03d}", side="b", axis=axis))
    pd.DataFrame(mp).to_csv(OUT / "minimal_pairs.csv", index=False)

    # The 36 hand-written hard negatives stay, and the corrective-negation
    # frames extend them across every confusable pair in the taxonomy. 36 rows
    # could not distinguish 0.58 from 0.50 or 0.70; this can.
    import yaml
    fams = yaml.safe_load((ROOT / "configs" / "intents.yaml").read_text())["families"]
    hn = [dict(text=t, intent=i, reason=r) for t, i, r in HARD_NEGATIVES]
    for r in gen_corrective(fams, TEST_FRAMES, rng, "corrective"):
        hn.append(dict(text=r["text"], intent=r["intent"],
                       reason=("corrective_HELD_OUT_family" if r["held_out"]
                               else "corrective_taught_family")))
    for r in gen_symptom(rng, "test", cap=120):
        hn.append(dict(text=r["text"], intent=r["intent"],
                       reason="symptom_unseen_vocab"))
    pd.DataFrame(hn).to_csv(OUT / "hard_negatives.csv", index=False)

    # 26 rows carried +-9.4 points of noise across seeds — unreadable. The
    # hand-written 26 stay; the generated rows use openers and surface forms
    # that appear nowhere in the F1 training block.
    neg = [dict(text=t, intent=i, policy=p) for t, i, p in NEGATION]
    neg += gen_negation(rng)
    pd.DataFrame(neg).to_csv(OUT / "negation.csv", index=False)

    # Accessories are the product's first priority and had no suite at all, so
    # the 22 accessory errors in reports/errors_final.csv were only visible by
    # grepping the error CSV by hand. Objects AND frames here are disjoint from
    # the F15 training block, so this measures the rule rather than recall.
    from accessories import generate as gen_accessories
    acc = gen_accessories(rng, "test")
    pd.DataFrame(acc).to_csv(OUT / "accessories.csv", index=False)

    ctx = [dict(text=t, intent=i, length="long") for t, i in CONTEXTUAL]
    ctx += gen_contextual(rng)
    pd.DataFrame(ctx).to_csv(OUT / "contextual.csv", index=False)

    # OOD moved out to ood_generate.py. The 45 rows that used to be written
    # here (OOD_NEAR + OOD_FAR, kept below for reference) are now one labelled
    # family inside a 287-row suite, and they go through the same two validity
    # checks as everything else — four of them do not pass. At 45 rows this
    # suite had no measured noise floor, so the teacher/student OOD gap was
    # four rows and could not be called a result.
    import ood_generate
    ood = ood_generate.validate(
        ood_generate.build_candidates(random.Random(13)),
        pd.read_csv(ROOT / "data" / "raw" / "en.csv"))[0]
    pd.DataFrame(ood)[["text", "intent", "ood_type", "family", "source"]] \
        .to_csv(OUT / "ood.csv", index=False)

    # Collision check against training corpus — a challenge item that already
    # exists in train proves nothing.
    train = pd.read_csv(ROOT / "data" / "raw" / "en.csv")
    train_keys = set(train["text"].map(leakage_key))
    train_norm = set(train["text"].map(normalize))
    all_new = pd.concat([
        pd.DataFrame(mp)[["text", "intent"]],
        pd.read_csv(OUT / "hard_negatives.csv")[["text", "intent"]],
        pd.DataFrame(neg)[["text", "intent"]],
        pd.DataFrame(ctx)[["text", "intent"]],
        pd.DataFrame(ood)[["text", "intent"]],
        pd.DataFrame(acc)[["text", "intent"]],
    ])
    all_new["norm"] = all_new["text"].map(normalize)
    all_new["key"] = all_new["text"].map(leakage_key)
    n_norm = int(all_new["norm"].isin(train_norm).sum())
    n_key = int(all_new["key"].isin(train_keys).sum())
    all_new.to_csv(OUT / "_all_challenge.csv", index=False)

    print(f"minimal_pairs   : {len(mp)} rows ({len(MINIMAL_PAIRS)} pairs)")
    print(f"hard_negatives  : {len(hn)} ({len(HARD_NEGATIVES)} hand-written + {len(hn)-len(HARD_NEGATIVES)} corrective)")
    print(f"negation        : {len(neg)} ({len(NEGATION)} hand-written)")
    print(f"contextual      : {len(ctx)} ({len(CONTEXTUAL)} hand-written)")
    print(f"accessories     : {len(acc)} "
          f"(objects and frames disjoint from the F15 training block)")
    n_near = sum(r["ood_type"] == "near" for r in ood)
    print(f"ood             : {len(ood)} (near={n_near} far={len(ood)-n_near}) "
          f"— see ood_rejected.csv for what was refused and why")
    print(f"collision with train: exact-normalized={n_norm}  leakage-key={n_key}")
    if n_key:
        print(all_new[all_new["key"].isin(train_keys)][["text", "intent"]].to_string())


if __name__ == "__main__":
    main()
