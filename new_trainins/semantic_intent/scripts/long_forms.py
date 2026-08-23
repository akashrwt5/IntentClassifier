"""Enlarge the two suites that were too small to read.

`negation` had 26 rows and `contextual` had 25. Across 5 seeds their 2-sigma
was +-0.094 and +-0.098 — ten points of noise, which is larger than any effect
we were trying to measure. Every reading taken from those columns during
development was uninterpretable, and several were reported as results anyway.

Enlarging is not cosmetic: `hard_negatives` went from 36 rows to 916 and its
2-sigma fell from roughly +-0.09 to +-0.021, which is what turned it from
decoration into a number worth acting on.

Both generators use vocabulary and frames DISJOINT from the training
augmentation, so the suites still measure generalisation.
"""

from __future__ import annotations

import itertools
import random

FALLBACK = "Default Fallback Intent"

# ---------------------------------------------------------------------------
# NEGATION — policy P2 (bare negation, no alternative -> no action)
# Training (F1) uses: do not / don't / please do not / i do not want you to /
# i did not ask you to / no need to / there is no need to / stop trying to /
# i never asked you to. None of those appear here.
# ---------------------------------------------------------------------------
TEST_NEG_OPENERS = [
    "i would rather you did not",
    "there is no call to",
    "hold off on",
    "you do not need to",
    "i am not asking you to",
    "let us not",
    "kindly refrain from",
    "it would be wrong to",
]
# Surface forms also disjoint from F1's NEG_ACTIONS wording.
TEST_NEG_ACTIONS = [
    "raise the sound",
    "drop the sound",
    "silence my aids",
    "bring the sound back",
    "swap my program",
    "open the audio link",
    "close the audio link",
    "note down a reminder",
    "dictate a message",
    "replay my messages",
    "begin translating",
    "begin captions",
    "hunt for my phone",
    "clear that reminder",
]

# P3 (corrective) — the affirmed option wins. Kept small here because the
# corrective shape is measured in depth by the hard-negative suite.
TEST_CORRECTIVE = [
    ("drop the sound", "raise the sound", "Cmd.VolumeIncrease"),
    ("raise the sound", "drop the sound", "Cmd.VolumeDecrease"),
    ("silence my aids", "bring the sound back", "Cmd.VolumeUnmute"),
    ("bring the sound back", "silence my aids", "Cmd.VolumeMute"),
    ("close the audio link", "open the audio link", "Cmd.StreamingStart"),
    ("open the audio link", "close the audio link", "Cmd.StreamingStop"),
]
TEST_CORRECTIVE_FRAMES = [
    "{r} — do not {w}",
    "rather than {w}, {r}",
    "{r}, that is, not {w}",
]

# ---------------------------------------------------------------------------
# CONTEXTUAL — a real request buried in conversational scaffolding.
# The command itself is short; everything around it is what makes it hard.
# ---------------------------------------------------------------------------
# Training rows with >=18 words: 23 out of 6702 — and 20 of those 23 are
# Default Fallback Intent. The model did not fail to understand long sentences;
# it learned, correctly from this data, that long text means "unsupported".
# That is the strongest single shortcut in the corpus and it hits exactly the
# input real users produce: people explain their situation before asking.
#
# Scenarios and payloads are split TRAIN / TEST. Training on the same scaffolds
# the suite uses would teach the templates and prove nothing — the same trap the
# corrective batch fell into.
SCENARIOS_TEST = [
    "i am sitting in a busy cafe and it is hard to follow the conversation, so",
    "we have just got back from a walk and before i settle down",
    "my daughter is visiting this afternoon and before she arrives",
    "i have been meaning to ask you all morning, and now that i have a moment,",
    "the television has been on for hours and my ears are tired, so",
    "i am about to head into a meeting in a few minutes so",
    "it is quite windy out here on the golf course today, so",
    "my wife keeps telling me i have it set wrong, and honestly",
    "i just woke up and i am still a bit muddled, but",
    "the grandchildren are running around making a racket and",
    "i am on the bus and it is rather noisy in here, so",
    "there is building work going on next door all week, so",
]
SCENARIOS_TRAIN = [
    "i have just come in from the garden and before i sit down",
    "there is a lot of traffic noise where i am standing so",
    "my neighbour has the radio on very loud again and",
    "i am at my grandson's football match and it is chaos here, so",
    "the kettle and the extractor fan are both going and",
    "i have got the family round for lunch and it is very lively, so",
    "i am in the waiting room at the surgery and it is quiet in here, so",
    "we are driving and the road noise is quite bad, so",
    "i have been reading for an hour and it is very still in the house, so",
    "the washing machine is on its spin cycle and",
    "i am at the garden centre and there is music playing everywhere, so",
    "before i take the dog out this evening",
]
FILLERS = ["um,", "well,", "so,", "er,", "right,", ""]
POLITE_TEST = [
    "could you please",
    "would you mind if you",
    "i would like you to",
    "if it is not too much trouble,",
    "please",
    "can you",
]
POLITE_TRAIN = [
    "be a dear and",
    "i need you to",
    "go ahead and",
    "when you get a moment,",
    "would you",
    "do me a favour and",
]

# (short command, intent). Deliberately different wording from the training
# augmentation — these are the payloads the scaffolding wraps.
PAYLOADS_TEST = [
    ("lift the sound a notch", "Cmd.VolumeIncrease"),
    ("ease the sound back a notch", "Cmd.VolumeDecrease"),
    ("cut all sound for now", "Cmd.VolumeMute"),
    ("let the sound through again", "Cmd.VolumeUnmute"),
    ("move me onto the cafe setting", "Cmd.MemoryChange"),
    ("open the link to the television", "Cmd.StreamingStart"),
    ("close the link to the television", "Cmd.StreamingStop"),
    ("tell me what charge is left", "Cmd.BatteryLevel"),
    ("track down my handset", "Cmd.FindMyPhone"),
    ("start putting their words on screen", "Cmd.TranscribeStart"),
    ("start turning their spanish into english", "Cmd.TranslationStart"),
    ("note down that i must take my tablets at six", "reminders.add"),
    ("mark that tablet reminder as done", "reminders.complete"),
    ("dictate a short note for my daughter", "Cmd.SendMessage"),
    ("replay whatever came through earlier", "Cmd.ListenMessage"),
    ("tell me how far i walked", "Cmd.ActivityWalk"),
    ("tell me how many steps i have done", "Cmd.ActivityStep"),
    ("tell me what i burned this morning", "Cmd.ActivityCalories"),
    ("explain how the wind setting works", "Help_EdgeMode"),
    ("explain how the ringing feature works", "Help_Tinnitus"),
    ("explain how i get these paired up", "Help_Pairing"),
    ("explain how i keep these clean", "Help_CleanCare"),
    ("explain how i seat these properly", "Help_InsertDevice"),
    ("explain how my audiologist adjusts these from afar", "Help_RemoteProgramming"),
    ("explain how i let my son see how i am doing", "Help_HearShare"),
    ("explain what that score in the app is measuring", "Help_ThriveScore"),
    ("explain what happens if i take a fall", "Help_FallAlert"),
    ("explain how the volume control works", "Help_Volume"),
    ("explain how i switch between settings", "Help_ChangingMemories"),
    ("explain what changed in this new version", "Help_WhatsNew"),
]


def gen_negation(rng: random.Random) -> list[dict]:
    rows = []
    for opener, action in itertools.product(TEST_NEG_OPENERS, TEST_NEG_ACTIONS):
        rows.append(dict(text=f"{opener} {action}", intent=FALLBACK, policy="P2_bare_negation"))
    for (w, r, intent), frame in itertools.product(TEST_CORRECTIVE, TEST_CORRECTIVE_FRAMES):
        rows.append(dict(text=frame.format(w=w, r=r), intent=intent, policy="P3_corrective"))
    rng.shuffle(rows)
    return rows


# Payload wordings for TRAINING — same intents, different words from the suite.
PAYLOADS_TRAIN = [
    ("push the volume up a bit", "Cmd.VolumeIncrease"),
    ("pull the volume down a bit", "Cmd.VolumeDecrease"),
    ("switch the sound off entirely", "Cmd.VolumeMute"),
    ("switch the sound back on", "Cmd.VolumeUnmute"),
    ("put me on the outdoor program", "Cmd.MemoryChange"),
    ("start the tv audio coming through", "Cmd.StreamingStart"),
    ("stop the tv audio coming through", "Cmd.StreamingStop"),
    ("check how much power is left", "Cmd.BatteryLevel"),
    ("find where my phone is", "Cmd.FindMyPhone"),
    ("begin the live captions", "Cmd.TranscribeStart"),
    ("begin translating for me", "Cmd.TranslationStart"),
    ("set a reminder for my medication", "reminders.add"),
    ("tick off the reminder i had", "reminders.complete"),
    ("record a message for my son", "Cmd.SendMessage"),
    ("play back the messages i have", "Cmd.ListenMessage"),
    ("show me my walking distance", "Cmd.ActivityWalk"),
    ("show me my step total", "Cmd.ActivityStep"),
    ("show me the calories i used", "Cmd.ActivityCalories"),
    ("tell me about the edge feature", "Help_EdgeMode"),
    ("tell me about the tinnitus feature", "Help_Tinnitus"),
    ("tell me how pairing works", "Help_Pairing"),
    ("tell me how cleaning works", "Help_CleanCare"),
    ("tell me how to fit them in", "Help_InsertDevice"),
    ("tell me about remote adjustments", "Help_RemoteProgramming"),
    ("tell me about sharing my data", "Help_HearShare"),
    ("tell me what the score means", "Help_ThriveScore"),
    ("tell me about fall detection", "Help_FallAlert"),
    ("tell me how to adjust the volume", "Help_Volume"),
    ("tell me how to change programs", "Help_ChangingMemories"),
    ("tell me what is new in the app", "Help_WhatsNew"),
]


def gen_contextual(rng: random.Random, per_payload: int = 4, split: str = "test") -> list[dict]:
    """Long conversational requests. TRAIN and TEST share no scenario, no
    politeness phrase and no payload wording — only the shape."""
    if split == "train":
        scenarios, polite, payloads = (SCENARIOS_TRAIN, POLITE_TRAIN, PAYLOADS_TRAIN)
        source = "F13_long_form"
    else:
        scenarios, polite, payloads = (SCENARIOS_TEST, POLITE_TEST, PAYLOADS_TEST)
        source = "contextual"
    rows = []
    for payload, intent in payloads:
        combos = list(itertools.product(scenarios, polite, FILLERS))
        rng.shuffle(combos)
        for scenario, pol, filler in combos[:per_payload]:
            text = f"{filler} {scenario} {pol} {payload}".strip()
            rows.append(
                dict(text=" ".join(text.split()), intent=intent, length="long", source=source)
            )
    rng.shuffle(rows)
    return rows
