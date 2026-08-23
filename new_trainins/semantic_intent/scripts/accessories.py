"""F15 — accessories: the product's first priority, and its weakest area.

WHY THIS BATCH EXISTS
---------------------
The TV streamer, the remote mic and Auracast connect to the aids. They are the
first thing this product is for, and they are where the shipped model is worst.
From reports/errors_final.csv:

    8x  Help_Accessories   -> Help_Pairing
   14x  Default Fallback   -> Cmd.StreamingStart
    1x  Cmd.StreamingStop  -> Cmd.StreamingStart   "done streaming"        0.796
    1x  Cmd.StreamingStart -> Default Fallback     "can you play tv"
    1x  Cmd.StreamingStop  -> Default Fallback     "unhook me from the television audio"

Every one of these was refused by the gate, so nothing wrong reached hardware —
but a refused accessory request is still a user asking their hearing aids to
connect to the television and being told to repeat themselves.

WHAT THE DATASET SAYS, AND WHAT DECIDES
---------------------------------------
The Accessories/Pairing split is not about the wording. It is about the OBJECT:

    Help_Pairing (263)     the aid and a PHONE or the APP
                           "how do i check if the hearing aids are connected
                            to the app?"
    Help_Accessories (152) a physical ACCESSORY — remote mic, table mic,
                           tv streamer
                           "how do i check what accessories i have connected
                            to my aid?"
                           "how do i delete my remote mic?"

Those two sentences are near-identical in structure and differ only in the
object, which is exactly the shape the model gets wrong — and exactly policy P5,
the axis already used for "whose device is it". The class imbalance pushes the
wrong way too: Pairing has 263 rows against Accessories' 152, so when the object
is ambiguous the model guesses the bigger class.

Note that Help_Accessories also absorbs how-to questions about streaming FROM an
accessory ("how do i start streaming from my tv streamer?"), while the bare
command ("stream tv") is Cmd.StreamingStart. That is policy P1 — the request
type decides — and it is followed here rather than reinvented.

Direction, separately: Cmd.StreamingStop is the smaller class (71 vs 209) and
loses the same way the volume pair used to. "done streaming" went to
StreamingStart at 0.796.

AURACAST — READ THIS BEFORE TRUSTING THE LABELS
-----------------------------------------------
`auracast` appears in en.csv exactly ZERO times. So do `mini mic` and
`remote control`. Nothing in the dataset can tell us where an Auracast request
belongs; every other label in this file was checked against the corpus and this
one cannot be.

The mapping used here follows the corpus's own treatment of the TV streamer,
which is the closest thing it knows:

    connect to the auracast broadcast   -> Cmd.StreamingStart
    stop the auracast stream            -> Cmd.StreamingStop
    how do i use auracast               -> Help_Accessories

This is consistent with the instruction that stream/tv/accessory phrases route
to StreamingStart/Stop, and with P1 for the how-to form. It is still an
EXTENSION beyond the data. If the product disagrees, change AURACAST_* below —
it is isolated here for exactly that reason, and no other batch depends on it.
"""
from __future__ import annotations

import itertools
import random

ACCESSORIES = "Help_Accessories"
PAIRING = "Help_Pairing"
START = "Cmd.StreamingStart"
STOP = "Cmd.StreamingStop"

# ---------------------------------------------------------------------------
# The object axis. Same frame, swapped object, opposite label — the pair is the
# lesson, so both halves must be generated together.
# ---------------------------------------------------------------------------
ACCESSORY_OBJECTS_TRAIN = [
    "remote mic", "table mic", "tv streamer", "remote microphone",
    "partner mic", "tv connector",
]
# Held out. Two kinds on purpose: some share the head noun with training
# ("companion mic" vs "remote mic") and test whether an unseen MODIFIER still
# routes correctly; others share no token at all ("tv box", "neck loop") and
# test whether the rule survives a genuinely new accessory name. Reporting one
# number over both would hide which kind the model can actually do.
ACCESSORY_OBJECTS_TEST = [
    "companion mic", "clip on mic", "lapel mic",          # modifier unseen
    "tv box", "streaming accessory", "audio dongle",       # fully unseen
    "neck loop", "sound bridge",
]
PHONE_OBJECTS_TRAIN = ["phone", "app", "mobile"]
PHONE_OBJECTS_TEST = ["handset", "phone app", "tablet"]

OBJECT_FRAMES_TRAIN = [
    "how do i check what {o} is connected",
    "how do i remove my {o}",
    "how do i rename my {o}",
    "what do i do if my {o} keeps dropping out",
]
OBJECT_FRAMES_TEST = [
    "where do i see my {o}",
    "how do i get rid of my {o}",
    "my {o} will not stay connected",
    "can you help me set up my {o}",
    "is my {o} still linked",
    "what should i do about my {o}",
]

# ---------------------------------------------------------------------------
# The direction axis. Cmd.StreamingStop is the smaller class (71 vs 209) and
# loses ties, the same way Cmd.VolumeDecrease used to before F2.
# Wordings taken from how the corpus already phrases these: "disconnect from
# tv", "cut the streamed audio to my aids", "i don't want the stream anymore".
# ---------------------------------------------------------------------------
STOP_TRAIN = [
    "done streaming", "finished with the stream", "i am done with the tv audio",
    "unhook me from the television audio", "that is enough tv sound",
    "take the tv audio away", "no more streaming for now",
    "release the connection to the tv", "i have finished watching",
]
STOP_TEST = [
    "wrap up the streaming", "i am through with the tv sound",
    "let go of the tv audio", "that will do for the stream",
    "i have had enough of the telly sound", "shut the tv feed down",
    "pull the plug on the streaming", "quit sending me the tv",
]
START_TRAIN = [
    "can you play tv", "put the tv through to my aids",
    "send the television sound over", "let me hear the tv",
    "bring the tv audio in", "hook me up to the television audio",
    "i want the tv in my ears", "route the tv sound to me",
]
START_TEST = [
    "pipe the tv through to me", "give me the television audio",
    "let the tv sound in", "i would like the telly in my ears",
    "feed me the television audio", "switch the tv feed on",
    "start sending me the tv", "put the telly through",
]
# REMOVED from START_TEST: "connect me to the tv sound". It shares a leakage key
# with the training row "connect the tv sound to me" (Cmd.StreamingStart), so it
# would have measured memorisation. The assert in build_targeted_training.py
# caught it — which is the assert doing its job, not a nuisance.

# ---------------------------------------------------------------------------
# Auracast. Zero corpus rows — see the module docstring. Isolated so it can be
# deleted or relabelled without touching anything else.
# ---------------------------------------------------------------------------
AURACAST_START_TRAIN = [
    "connect to the auracast broadcast", "join the auracast channel",
    "tune into auracast", "start the auracast stream",
]
AURACAST_STOP_TRAIN = [
    "leave the auracast broadcast", "stop the auracast stream",
    "disconnect from auracast",
]
AURACAST_HELP_TRAIN = [
    "how do i use auracast", "what is auracast for",
    "how do i find an auracast broadcast",
]
AURACAST_TEST = [
    ("hook onto the auracast broadcast", START),
    ("drop out of the auracast channel", STOP),
    ("explain how auracast works", ACCESSORIES),
]


def generate(rng: random.Random, split: str = "train") -> list[dict]:
    """Both halves of every pair, in one batch.

    Generating only the accessory half would teach "accessory word ->
    Help_Accessories" and damage Help_Pairing, which is the larger and
    currently better-performing class. The pair is the lesson.
    """
    train = split == "train"
    acc_obj = ACCESSORY_OBJECTS_TRAIN if train else ACCESSORY_OBJECTS_TEST
    ph_obj = PHONE_OBJECTS_TRAIN if train else PHONE_OBJECTS_TEST
    frames = OBJECT_FRAMES_TRAIN if train else OBJECT_FRAMES_TEST
    stops = STOP_TRAIN if train else STOP_TEST
    starts = START_TRAIN if train else START_TEST
    src = "F15_accessories" if train else "accessories"

    rows: list[dict] = []

    def add(text: str, intent: str, axis: str):
        rows.append(dict(text=" ".join(text.split()), intent=intent,
                         source=src, axis=axis))

    # object axis — the accessory half and the phone half of the same frame
    for obj, frame in itertools.product(acc_obj, frames):
        add(frame.format(o=obj), ACCESSORIES, "object")
    for obj, frame in itertools.product(ph_obj, frames):
        add(frame.format(o=obj), PAIRING, "object")

    # direction axis
    for t in stops:
        add(t, STOP, "direction")
    for t in starts:
        add(t, START, "direction")

    # auracast — unattested in the corpus, see docstring
    if train:
        for t in AURACAST_START_TRAIN:
            add(t, START, "auracast")
        for t in AURACAST_STOP_TRAIN:
            add(t, STOP, "auracast")
        for t in AURACAST_HELP_TRAIN:
            add(t, ACCESSORIES, "auracast")
    else:
        for t, i in AURACAST_TEST:
            add(t, i, "auracast")

    rng.shuffle(rows)
    return rows
