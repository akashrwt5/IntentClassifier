"""F12 — symptom words must not decide the direction.

    "it is too loud in here, can you make it quieter"  -> predicted INCREASE

No negation, no correction, no ambiguity — a plain sentence with the requested
action spelled out. The model took the word "loud" and stopped reading. This is
the lexical shortcut Section 5 of the plan warns about, and it is a different
failure from the corrective-negation one: there the structure defeats the model,
here a single emotive word does.

The corpus explains why. Only 54 rows use symptom phrasing at all, split 31
Cmd.VolumeIncrease to 14 Cmd.VolumeDecrease. The Increase side is over-represented
two to one, so "there is a complaint about sound" has been learned as evidence
for turning it UP.

The fix is not more data in general — it is data where the symptom word and the
requested direction are decorrelated, in both directions, in equal number.

Vocabulary is split into TRAIN and TEST halves. A model that has memorised
"blaring -> down" has learned nothing; a model that handles the unseen
"thunderous -> down" has learned to read the verb instead of the adjective.
"""

from __future__ import annotations

import itertools
import random

# --- symptoms that mean IT IS TOO LOUD -> the request will be a decrease ------
LOUD_TRAIN = ["loud", "blaring", "deafening", "harsh", "booming"]
LOUD_TEST = ["thunderous", "piercing", "overwhelming", "shrill"]
DOWN_TRAIN = ["make it quieter", "turn it down", "bring it down", "lower it", "soften it"]
DOWN_TEST = ["ease it off", "take it down a bit", "dial it back"]

# --- symptoms that mean IT IS TOO QUIET -> the request will be an increase ----
QUIET_TRAIN = ["quiet", "faint", "soft", "low", "muffled"]
QUIET_TEST = ["thin", "distant", "weak", "washed out"]
UP_TRAIN = ["make it louder", "turn it up", "bring it up", "raise it", "boost it"]
UP_TEST = ["lift it a bit", "push it up", "give me more"]

CONTEXT = ["in here", "in this room", "right now", "on this setting", "at the moment", "today", ""]

FRAMES = [
    "it is too {sym} {ctx}, {act}",
    "{act}, it is too {sym} {ctx}",
    "my aids are too {sym} {ctx}, {act}",
    "everything is too {sym} {ctx}, please {act}",
    "the sound is too {sym} {ctx} so {act}",
]


def _build(syms, acts, intent, rng, source):
    rows = []
    for sym, act, ctx, frame in itertools.product(syms, acts, CONTEXT, FRAMES):
        text = frame.format(sym=sym, act=act, ctx=ctx)
        rows.append(
            dict(
                text=" ".join(text.split()).replace(" ,", ","),
                intent=intent,
                source=source,
                symptom=sym,
            )
        )
    rng.shuffle(rows)
    return rows


# A symptom with NO request in the sentence. Here the adjective IS the signal —
# there is nothing else to read — and the implied action follows from it
# (policy P4). Balanced for the same reason as above: the corpus's 31-to-14 tilt
# taught "complaint about sound" as evidence for turning it up, and that tilt
# has to be removed from BOTH sentence shapes or it just moves.
SYMPTOM_ONLY_FRAMES = [
    "it is too {sym} {ctx}",
    "everything is too {sym} {ctx}",
    "my aids are too {sym} {ctx}",
    "the sound is far too {sym} {ctx}",
    "this is much too {sym} {ctx}",
]


def _build_symptom_only(syms, intent, rng, source):
    rows = []
    for sym, ctx, frame in itertools.product(syms, CONTEXT, SYMPTOM_ONLY_FRAMES):
        text = frame.format(sym=sym, ctx=ctx)
        rows.append(dict(text=" ".join(text.split()), intent=intent, source=source, symptom=sym))
    rng.shuffle(rows)
    return rows


def generate(rng: random.Random, split: str = "train", cap: int = 260) -> list[dict]:
    """Equal numbers each way — the whole point is to break the correlation,
    and an unbalanced batch would just move the bias rather than remove it."""
    if split == "train":
        loud, down, quiet, up = LOUD_TRAIN, DOWN_TRAIN, QUIET_TRAIN, UP_TRAIN
        source = "F12_symptom_shortcut"
    else:
        loud, down, quiet, up = LOUD_TEST, DOWN_TEST, QUIET_TEST, UP_TEST
        source = "symptom_shortcut_unseen_vocab"

    half = cap // 2
    dec = _build(loud, down, "Cmd.VolumeDecrease", rng, source)[: half * 3 // 4]
    inc = _build(quiet, up, "Cmd.VolumeIncrease", rng, source)[: half * 3 // 4]
    # a quarter of the batch is the symptom-only shape
    dec += _build_symptom_only(loud, "Cmd.VolumeDecrease", rng, source + "_only")[: half // 4]
    inc += _build_symptom_only(quiet, "Cmd.VolumeIncrease", rng, source + "_only")[: half // 4]
    rows = dec + inc
    rng.shuffle(rows)
    return rows
