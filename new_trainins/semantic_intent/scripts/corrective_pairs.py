"""F11 — corrective negation across every confusable pair, not just volume.

All 15 hard-negative failures were the same sentence shape and the same
mistake:

    "i do not want to transcribe, i want to translate"  -> predicted Transcribe
    "not the recovery number, just my heart rate"       -> predicted Recovery
    "not edge mode, i meant mask mode"                  -> predicted EdgeMode

The model takes the FIRST-MENTIONED option and ignores that it was the one
being rejected. Policy P3 says the affirmed alternative wins, and the training
data did encode P3 — but only along the volume and streaming axes
(louder/quieter, up/down, muted/unmuted, started/stopped). Every failure is on
a pair P3 did not cover. So this is not the model failing to learn the rule;
it is the rule having been taught in one corner of the label space and not
generalising out of it.

This module states each intent as a short phrase and generates the corrective
frame over every confusable sibling pair in the taxonomy. Train and test use
DISJOINT frame sets, so the suite still measures generalisation rather than
memorised templates.
"""

from __future__ import annotations

import itertools
import random

# One short surface phrase per intent, in the corpus's own words. Only intents
# that share a family with something else need one — a pair needs two sides.
PHRASE = {
    "Cmd.VolumeIncrease": "louder",
    "Cmd.VolumeDecrease": "quieter",
    "Cmd.VolumeMute": "muted",
    "Cmd.VolumeUnmute": "unmuted",
    "Help_Volume": "how the volume works",
    "Cmd.MemoryChange": "a different program",
    "Help_ChangingMemories": "how to change programs",
    "Help_MemoryOptions": "how to make a new program",
    "Help_Customize": "how to personalise the sound",
    "Cmd.StreamingStart": "streaming started",
    "Cmd.StreamingStop": "streaming stopped",
    "Help_Accessories": "help with my tv streamer",
    "Cmd.TranscribeStart": "to transcribe",
    "Help_Transcribe": "how transcribe works",
    "Cmd.TranslationStart": "to translate",
    "Help_Translate": "how translate works",
    "Cmd.FindMyPhone": "my phone found",
    "Help_FindMyHearingAids": "my hearing aid found",
    "Cmd.BatteryLevel": "the charge level",
    "Help_Battery": "what battery size i need",
    "reminders.add": "a new reminder",
    "reminders.complete": "that reminder marked done",
    "Help_Reminder": "how reminders work",
    "Cmd.SendMessage": "to record a message",
    "Cmd.ListenMessage": "to play my messages",
    "Cmd.ActivityWalk": "my walking",
    "Cmd.ActivityRun": "my running",
    "Cmd.ActivityCycle": "my cycling",
    "Cmd.ActivityStep": "my step count",
    "Cmd.ActivityStand": "my standing time",
    "Cmd.ActivityExercise": "my workout",
    "Cmd.ActivityAerobics": "my aerobics",
    "Cmd.ActivityCalories": "my calories",
    "Help_Health": "my activity summary",
    "Help_HeartRate": "my heart rate",
    "Help_HeartRateRecovery": "my heart rate recovery",
    "Help_ThriveScore": "my score",
    "Help_CleanCare": "how to clean them",
    "Help_InsertDevice": "how to put them in",
    "Help_SelfCheck": "a self check",
    "Help_DeviceSettings": "the device settings",
    "Help_RemoteProgramming": "a remote adjustment",
    "Help_HearingCareAnywhereConnect": "a remote appointment",
    "Help_HearShare": "to share my data",
    "Help_EdgeMode": "edge mode",
    "Help_MaskMode": "mask mode",
    "Help_IntelliVoice": "speech enhancement",
    "Help_Tinnitus": "the tinnitus masker",
    "Help_WiCROS": "the cros system",
    "Help_DemoMode": "demo mode",
    "Help_Home": "the basics",
    "Help_AppSettings": "the app settings",
    "Help_WhatsNew": "what is new",
    "Help_VoiceAssistant": "the voice assistant",
    "Help_Pairing": "to pair them",
}

# {w} = the rejected option, {r} = the affirmed one. The affirmed option is
# second in every frame, because that is the shape the model gets wrong.
TRAIN_FRAMES = [
    "not {w}, {r}",
    "i do not want {w}, i want {r}",
    "do not give me {w}, give me {r}",
    "{r} please, not {w}",
    "i said {r}, not {w}",
    "forget {w}, i need {r}",
    "i asked for {r} not {w}",
]
# Held out: never used in training, so the suite tests the rule and not the frame.
TEST_FRAMES = [
    "not {w} — i meant {r}",
    "i am not after {w}, i am after {r}",
    "skip {w}, do {r} instead",
    "{w} is not what i wanted, {r} is",
]


# Families deliberately kept OUT of the corrective TRAINING data. If every pair
# is taught the corrective frame, the suite can only tell us that the model
# memorised the frames — it cannot tell us whether the rule "the affirmed option
# wins" transferred to a pair it never saw stated that way. These three are the
# control group, and they are reported separately.
HELD_OUT_FAMILIES = ("features", "health", "messages")


def _pairs(families: dict, exclude: tuple = ()) -> list[tuple[str, str]]:
    out = []
    for fam, members in families.items():
        if fam in exclude:
            continue
        usable = [m for m in members if m in PHRASE]
        for a, b in itertools.permutations(usable, 2):
            out.append((a, b))  # both directions: which is rejected matters
    return out


def generate(
    families: dict, frames: list[str], rng: random.Random, source: str, exclude_families: tuple = ()
) -> list[dict]:
    fam_of = {m: f for f, ms in families.items() for m in ms}
    rows = []
    for (rejected, affirmed), frame in itertools.product(
        _pairs(families, exclude_families), frames
    ):
        rows.append(
            dict(
                text=frame.format(w=PHRASE[rejected], r=PHRASE[affirmed]),
                intent=affirmed,
                source=source,
                rejected_intent=rejected,
                family=fam_of.get(affirmed, "other"),
                held_out=fam_of.get(affirmed) in HELD_OUT_FAMILIES,
            )
        )
    rng.shuffle(rows)
    return rows
