"""F14 — teach the model that a named object in the room is not the aids.

THE FAILURE
-----------
On the enlarged 286-row OOD suite the leaks are not spread evenly. Six families
reject 100%; three do not:

    home_control        0.786
    health_other        0.844
    other_device_audio  0.889

and the accepted rows look like this:

    0.990  Cmd.VolumeDecrease   lower the thermostat
    0.990  Cmd.VolumeDecrease   the extractor fan is too loud
    0.986  Cmd.VolumeDecrease   the dishwasher is too loud
    0.986  Cmd.VolumeDecrease   close the blinds
    0.996  Cmd.MemoryChange     set the oven to one eighty

That is the dangerous half of the suite. A leaked weather question is harmless —
it never reaches hardware. A leaked "the dishwasher is too loud" turns the
user's hearing aids down while they are trying to hear over the dishwasher.

THE CAUSE — AND IT IS AN EARLIER FIX OF OURS
--------------------------------------------
The symptom shortcut was fixed by teaching policy P4b: a complaint about
loudness implies the corresponding action, even with no explicit request. That
worked — 0.95 on held-out symptom vocabulary. But look at what it was taught on.
`train_augmented` has 41 rows containing "too loud", every one of them
Cmd.VolumeDecrease, and every one with a pronoun subject:

    it's too loud turn it down · it sounds too loud · that sounds too loud
    this sounds too loud

No training row anywhere pairs a loudness complaint with a NAMED object. So the
model learned "complaint about loudness -> turn my aids down" and had no
opportunity to learn "...unless they named something else in the room". 0.99
confidence is not the model being reckless; it is the model applying, exactly,
the rule we gave it.

THE SHAPE OF THE FIX
--------------------
Negative rows alone would be wrong. Adding only "the dishwasher is too loud ->
Fallback" teaches "too loud -> Fallback" and breaks P4b, which took real work to
get right. The batch has to be MINIMAL PAIRS: the same frame, differing only in
whether a foreign object is named.

    the food mixer is far too loud   -> Default Fallback Intent
    the sound is far too loud        -> Cmd.VolumeDecrease

Two controls, both asserted at generation time rather than assumed:

  * objects are disjoint from the ones the OOD suite uses. If the suite's own
    nouns were trained on, the suite would measure recall of a word list
    instead of the rule. This is the trap the corrective batch fell into, where
    taught families scored 0.74 and held-out families 0.48.
  * frame WORDINGS are disjoint too ("turn down the {x}" here, "turn the {x}
    down" there), so the suite is not measuring template recall either.

A note on what survives into train_augmented. Of 125 generated rows about 114
are kept, and most of the losses are on the COMMAND side — "turn down the
sound" collides with a held-out eval suite and is dropped, correctly, since
training on it would leak. That looks alarming (100 Fallback vs 14 command) and
is not, because the pairing lives at the pattern level and the corpus already
supplies the other half from its own rows:

    turn down the volume on the hearing aid -> Cmd.VolumeDecrease   (38 in train)
    turn down the air fryer                 -> Default Fallback     (F14)

The generated command rows were largely redundant with real data. Check this
again if the corpus ever changes: if the aid-side counts above fall away, this
batch starts teaching "turn down X -> Fallback" unopposed.

And one guard against re-creating an old failure: the corpus originally had 23
rows of 18+ words, 20 of them Fallback, which taught the model that long text
means unsupported and cost 0.4 on the contextual suite. Adding a pile of
Fallback rows is exactly how that gets re-introduced, so the length
distributions of the two halves are checked against each other before the batch
is returned.
"""
from __future__ import annotations

import itertools
import re
import random

FALLBACK = "Default Fallback Intent"

# ---------------------------------------------------------------------------
# Objects. NONE of these may appear in the OOD suite — asserted below against
# ood_generate's own lists, so the two files cannot drift apart silently.
# All are checked against the corpus too: a noun the dataset already knows is
# not a foreign object.
# ---------------------------------------------------------------------------
# Objects are grouped by what can sensibly be said ABOUT them. A flat list
# crossed with a flat frame list produced "the bedside lamp is far too quiet",
# "set the printer to twenty degrees" and "the curtains is far too quiet" —
# sentences no user will ever say. A model rejecting those has not been taught
# anything, and the same mistake had just been caught and fixed in
# ood_generate.py, so there is no excuse for it recurring here.
NOISY = [                      # can be too loud, hum, buzz, be quietened
    "fridge", "freezer", "vacuum cleaner", "coffee machine", "air fryer",
    "boiler", "food mixer", "hairdryer", "printer", "car alarm",
    "burglar alarm", "cooker hood", "shower pump", "strimmer",
    "pressure washer", "sewing machine", "juicer",
]
SWITCHABLE = NOISY + [         # can be turned on or off
    "bedside lamp", "ceiling light", "radiator", "immersion heater",
    "electric blanket", "toaster",
]
# Explicit pairs, not a product — an air fryer takes a temperature, a boiler
# takes a schedule, and crossing the two lists invents neither.
SETTABLE = [
    ("air fryer", "to one eighty"), ("boiler", "to twenty degrees"),
    ("immersion heater", "to run at six"), ("radiator", "to the low setting"),
    ("electric blanket", "to the highest setting"),
    ("coffee machine", "to start at seven"), ("printer", "to double sided"),
    ("sewing machine", "to the slow stitch"), ("freezer", "to fast freeze"),
    ("toaster", "to a darker setting"),
]

# Wordings deliberately different from the suite's AUDIO_FRAMES / HOME_THINGS.
# suite: "turn the {d} down"          here: "turn down the {x}"
# suite: "can you silence the {d}"    here: "quieten the {x}"
# suite: "switch the {d} off"         here: "shut the {x} off"
# suite: "make the {d} stop beeping"  here: "stop the {x} buzzing"
# suite: "why is the {d} making that noise"  here: "what is that noise from the {x}"
NOISE_FRAMES = [
    "the {x} is far too loud",
    "turn down the {x}",
    "quieten the {x}",
    "stop the {x} buzzing",
    "what is that noise from the {x}",
]
# "the burglar alarm keeps humming" is not a sentence anyone says — alarms
# sound, motors hum. Restricted to the things that actually hum.
HUMMING = ["fridge", "freezer", "boiler", "printer", "cooker hood",
           "shower pump", "coffee machine"]
SWITCH_FRAMES = [
    "shut the {x} off",
    "put the {x} on",
    "leave the {x} running",
]

# The other half of every pair: the SAME frames, aimed at the aids. Without
# these the batch teaches "too loud -> Fallback" and undoes P4b.
# Subjects carry their own article so no frame double-inserts one.
# Every subject here was checked against en.csv before being used, because the
# first version was not and four of the six were wrong:
#
#   the sound     90 rows, top labels VolumeDecrease 24 / VolumeIncrease 18  OK
#   the volume   193 rows, VolumeDecrease 52 / VolumeIncrease 45             OK
#   my aids      301 rows, volume intents present under imperative frames    OK
#   my left aid   50 rows, incl. VolumeUnmute 6                              OK
#   my right aid  61 rows, incl. VolumeIncrease 6                            OK
#
#   the speech   106 rows -> MemoryChange 51, Help_IntelliVoice 30.  REMOVED.
#                In this corpus "speech" is about clarity, not loudness.
#   background noise 4 rows -> Help_Customize 3. The corpus literally contains
#                "how do i turn down background noise?" labelled Help_Customize.
#                REMOVED.
#   echo          1 row -> Cmd.MemoryChange ("i'm in a big echoey room now").
#                REMOVED.
#   hiss, whistling  0 rows. I invented both the vocabulary and the label.
#                REMOVED.
#
# The validator did not catch any of these, and it is worth understanding why:
# "the background noise is far too loud" differs from the real
# "how do i turn down background noise?" by the tokens loud/how/turn/down, and
# all four are in POLARITY — three of them because I widened POLARITY earlier in
# this same session. The polarity exemption exists so minimal pairs survive, and
# the wider it gets the more it also waves through labels that are simply wrong.
# It is not a substitute for checking the dataset.
AID_SUBJECTS = ["the sound", "my aids", "the volume",
                "my left aid", "my right aid"]
AID_FRAMES = [
    ("{x} {v} far too loud", "Cmd.VolumeDecrease"),
    ("{x} {v} far too quiet", "Cmd.VolumeIncrease"),
    ("turn down {x}", "Cmd.VolumeDecrease"),
    ("turn up {x}", "Cmd.VolumeIncrease"),
    ("quieten {x}", "Cmd.VolumeDecrease"),
    # Added after the first training run. 11 of the 25 command rows collided
    # with held-out suites and were correctly dropped, leaving 14 against 100
    # Fallback rows — a 7:1 ratio where 4:1 was intended, because the cap was
    # applied before the dedup rather than after it. More command wordings give
    # the dedup something to take without gutting the contrast.
    ("bring {x} down", "Cmd.VolumeDecrease"),
    ("bring {x} up", "Cmd.VolumeIncrease"),
    ("{x} needs to come down", "Cmd.VolumeDecrease"),
    ("{x} needs to come up", "Cmd.VolumeIncrease"),
    ("ease {x} down a bit", "Cmd.VolumeDecrease"),
    ("nudge {x} up a bit", "Cmd.VolumeIncrease"),
]
# "my aids is far too loud" came out of the first version. Agreement is decided
# by the subject, not left to the frame.
PLURAL_SUBJECTS = {"my aids"}


def _verb(subject: str) -> str:
    return "are" if subject in PLURAL_SUBJECTS else "is"
# REMOVED: an AID_NOISE_ONLY block covering "the background noise", "the hiss",
# "the whistling" and "the echo in here", all mapped to Cmd.VolumeDecrease.
# The dataset disagrees with every one of them (see the note on AID_SUBJECTS).
# Nothing replaces it: "somewhere noisy" is already Cmd.MemoryChange in 9 corpus
# rows, so that ground is covered by real data rather than data I made up.

# Health readings were the second-worst family. Same principle: a named reading
# the product does not measure is not an activity request. Disjoint from the
# suite's HEALTH_OTHER list.
TRAIN_HEALTH = [
    "my iron levels", "my thyroid result", "my bone density",
    "my lung function", "my grip strength", "my kidney test",
    "my b12 result", "my blood count",
]
HEALTH_FRAMES_TRAIN = [
    "what does {t} say", "pull up {t}", "have you got {t}",
]


def _assert_disjoint_from_suite() -> None:
    """The two files must not share nouns. Checked, not trusted."""
    import ood_generate as og
    suite_nouns = " ".join(
        og.OTHER_DEVICES + [t for t, _ in og.HOME_THINGS]
        + og.HEALTH_OTHER + og.FOREIGN_NOUNS).lower()
    mine = set(SWITCHABLE) | {o for o, _ in SETTABLE} | set(TRAIN_HEALTH)

    # Accessories are NOT foreign objects. The TV streamer, remote mic and
    # Auracast connect to the aids and are the product's first priority — the
    # corpus routes them to Help_Accessories and Cmd.StreamingStart/Stop. An
    # appliance list that quietly grew to include "microphone" or "streamer"
    # would teach the gate to refuse exactly the hardware this product is for.
    accessory = ["streamer", "mic", "microphone", "auracast", "accessor",
                 "tv connector", "remote control", "hearing", "aid", "aids",
                 "tv", "television", "companion"]
    bad = [o for o in sorted(mine)
           if any(a in o.lower().split() or a in o.lower() for a in accessory)]
    if bad:
        raise SystemExit(
            f"F14 treats hearing-aid accessories as foreign objects: {bad}\n"
            "Accessories are in-domain. Remove them from the appliance lists.")

    clash = [o for o in sorted(mine) if o.lower() in suite_nouns]
    if clash:
        raise SystemExit(
            f"F14 trains on objects the OOD suite also uses: {clash}\n"
            "That would turn the suite into a test of word-list recall. Pick "
            "different nouns for one side or the other.")


def _assert_wellformed(rows: list[dict]) -> None:
    """Same guard as ood_generate.assert_wellformed, for the same reason.

    Both files build sentences by filling slots, and both produced ungrammatical
    output the first time. A training row nobody would say teaches the model a
    pattern nobody will ever send it.
    """
    bad = [r["text"] for r in rows
           if re.search(r"\b(my aids|curtains|levels|results) is\b", r["text"])
           or re.search(r"\bthe (everything|the)\b", r["text"])
           or re.search(r"\bturn up the (background noise|hiss|whistling|echo)\b",
                        r["text"])]
    if bad:
        raise SystemExit("F14 produced ungrammatical or nonsensical rows:\n  "
                         + "\n  ".join(repr(b) for b in bad[:10]))


def _assert_length_matched(rows: list[dict], tol: float = 1.0) -> None:
    """Fallback rows must not be systematically longer than command rows.

    The corpus once had 20 of its 23 longest rows labelled Fallback, and the
    model correctly learned that long text means unsupported — which cost 0.4 on
    the contextual suite and took a whole extra batch (F13) to undo. Adding
    Fallback rows in bulk is exactly how that comes back.
    """
    fb = [len(r["text"].split()) for r in rows if r["intent"] == FALLBACK]
    cmd = [len(r["text"].split()) for r in rows if r["intent"] != FALLBACK]
    if not fb or not cmd:
        raise SystemExit("F14 must contain both halves of every pair")
    mfb, mcmd = sum(fb) / len(fb), sum(cmd) / len(cmd)
    if abs(mfb - mcmd) > tol:
        raise SystemExit(
            f"F14 length-imbalanced: Fallback rows average {mfb:.1f} words, "
            f"command rows {mcmd:.1f}. That re-teaches 'long means unsupported'. "
            f"Shorten one side or lengthen the other.")


def gen_foreign_objects(rng: random.Random) -> list[dict]:
    _assert_disjoint_from_suite()
    rows: list[dict] = []

    def add(text: str, intent: str):
        rows.append(dict(text=" ".join(text.split()), intent=intent,
                         source="F14_foreign_object"))

    # 1. named object + a frame that fits that object -> not us
    for obj, frame in itertools.product(NOISY, NOISE_FRAMES):
        add(frame.format(x=obj), FALLBACK)
    for obj, frame in itertools.product(SWITCHABLE, SWITCH_FRAMES):
        add(frame.format(x=obj), FALLBACK)
    for obj in HUMMING:
        add(f"the {obj} keeps humming", FALLBACK)

    # 2. the same frames aimed at the aids -> the real command. This is what
    #    stops the batch from simply teaching "too loud -> Fallback".
    for subj, (frame, intent) in itertools.product(AID_SUBJECTS, AID_FRAMES):
        add(frame.format(x=subj, v=_verb(subj)), intent)

    # 3. settings on a named appliance -> not a memory change
    for obj, val in SETTABLE:
        add(f"set the {obj} {val}", FALLBACK)

    # 4. named readings we do not measure -> not an activity request
    for topic, frame in itertools.product(TRAIN_HEALTH, HEALTH_FRAMES_TRAIN):
        add(frame.format(t=topic), FALLBACK)

    # The Fallback half is capped at an ABSOLUTE number, not a ratio, because
    # the absolute count is what the training run showed actually matters.
    #
    # The first version added 100 Fallback rows and coverage fell from 0.708 to
    # 0.593 — four in every ten genuine commands refused. The mechanism is in
    # the operating-point ablation: at conf 0.870 the model met the 0.97
    # precision target with coverage 0.835, but its fallback leak was 0.0812,
    # above the 0.07 cap the gate must respect. To get the leak back under the
    # cap the threshold went to 0.93, and that is where the coverage went.
    #
    # More Fallback training rows -> more real commands drift into Fallback ->
    # higher leak -> stricter threshold -> lost coverage. A ratio does not bound
    # that; a count does.
    MAX_FALLBACK = 60
    fb = [r for r in rows if r["intent"] == FALLBACK]
    cmd = [r for r in rows if r["intent"] != FALLBACK]
    rng.shuffle(fb)
    fb = fb[:min(len(cmd) * 3, MAX_FALLBACK)]

    out = fb + cmd
    _assert_wellformed(out)
    _assert_length_matched(out)
    rng.shuffle(out)
    return out
