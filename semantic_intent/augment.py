"""
Contrastive augmentation for state-vs-action utterances.

The bug this exists to kill
---------------------------
Users describe a *problem* and request an *action* in one breath:

    "it's too quiet here, can you make it louder"   -> volume.increase
    "it's too loud here, can you make it quieter"   -> volume.decrease

Both sentences contain a loud-word and a quiet-word. Only the **action** word
decides the intent; the **state** word is the complaint. In the shipped corpus
just 1.4% of volume.increase and 2.8% of volume.decrease phrases contain both,
so no model — bag-of-words or embedding — has enough evidence to learn that
rule. Swapping TF-IDF for a semantic encoder does not fix it; measured, the
canonical pair still flips.

What this generates
-------------------
State x action crossings for the four volume intents, in the shapes people
actually speak them. Generation is seeded and deduplicated against the source
corpus, and is meant to be applied to the **training split only** — augmenting
before splitting would leak templates into test and manufacture a score.

    from semantic_intent.augment import contrastive_volume_phrases
    extra = contrastive_volume_phrases(per_intent=400, seed=0)
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

# ----------------------------------------------------------------- lexicons
LOUD_STATE = [
    "too loud",
    "too noisy",
    "way too loud",
    "really loud",
    "far too loud",
    "painfully loud",
    "deafening",
    "blasting",
    "too harsh",
    "overwhelming",
    "much too loud",
    "unbearably loud",
    "too strong",
    "too intense",
]
QUIET_STATE = [
    "too quiet",
    "too soft",
    "too faint",
    "way too quiet",
    "really quiet",
    "barely audible",
    "too low",
    "much too quiet",
    "too weak",
    "hard to hear",
    "almost inaudible",
    "very muffled",
    "far too soft",
    "not loud enough",
]
SILENT_STATE = [
    "completely silent",
    "totally muted",
    "on mute",
    "silenced",
    "not making any sound",
    "dead quiet with no audio",
    "muted right now",
]

UP_ACTION = [
    "make it louder",
    "turn it up",
    "raise the volume",
    "increase the volume",
    "boost it",
    "turn them up",
    "make them louder",
    "crank it up",
    "give me more volume",
    "amplify it",
    "bump it up",
    "push the volume up",
    # 'stronger' deliberately overlaps LOUD_STATE's "too strong": the model has
    # to learn that the verb decides, not the adjective.
    "make it stronger",
    "strengthen it",
    "make it more powerful",
]
DOWN_ACTION = [
    "make it quieter",
    "turn it down",
    "lower the volume",
    "decrease the volume",
    "soften it",
    "turn them down",
    "make them softer",
    "tone it down",
    "reduce the volume",
    "dial it back",
    "bring it down",
    "take it down a notch",
    "make it gentler",
    "make it less harsh",
    "make it milder",
    "ease it off",
]

# Venue words double as hearing-aid *program* names (restaurant, meeting...),
# so "this restaurant is deafening, tone it down" gets pulled toward
# device.memory.change. These templates teach that a venue mentioned alongside
# a volume action is context, not a program request.
VENUES = [
    "restaurant",
    "cafe",
    "bar",
    "party",
    "meeting",
    "church",
    "cinema",
    "train station",
    "shopping centre",
    "concert",
]
MUTE_ACTION = [
    "mute them",
    "turn the sound off",
    "silence them",
    "switch the mics off",
    "mute my hearing aids",
    "cut the audio",
    "turn off the sound completely",
]
UNMUTE_ACTION = [
    "turn the sound back on",
    "unmute them",
    "take them off mute",
    "re-enable the audio",
    "switch the mics back on",
    "bring the sound back",
    "unmute my hearing aids",
]

# --------------------------------------------------------------- templates
TEMPLATES = [
    "it's {state} {action}",
    "it's {state} can you {action}",
    "it's {state} please {action}",
    "it's {state} here {action}",
    "{state} in here {action}",
    "the sound is {state} {action}",
    "everything is {state} {action}",
    "this is {state} so {action}",
    "i think it's {state} {action}",
    "it feels {state} could you {action}",
    "{state} right now {action}",
    "my hearing aids are {state} {action}",
    "it's {state} would you {action}",
    "since it's {state} {action}",
    # venue-context templates (see VENUES)
    "this {venue} is {state} {action}",
    "i'm in a {venue} and it's {state} {action}",
    "the {venue} is {state} can you {action}",
]

# Terse, state-free commands. Without these, short utterances like "dial it
# back please" have no near neighbour in the corpus at all.
ACTION_ONLY_TEMPLATES = [
    "{action}",
    "can you {action}",
    "please {action}",
    "{action} please",
    "could you {action}",
    "i need you to {action}",
    "just {action}",
]

# Complaint-only utterances with no action verb at all. People say "I can't
# make out a word she's saying" and expect the volume up; the corpus has no
# such phrasing, so these land far from every prototype and get rejected as
# out-of-scope despite being the single most common real request.
TROUBLE_TEMPLATES = [
    "i can't make out {thing}",
    "i can barely hear {thing}",
    "i'm missing {thing}",
    "i can't follow {thing}",
    "{thing} is impossible to hear",
    "i keep missing {thing}",
    "i'm not catching {thing}",
    "it's hard to hear {thing}",
]
TROUBLE_THINGS = [
    "what she is saying",
    "a single word",
    "the conversation",
    "what people are saying",
    "anything at all",
    "what he just said",
    "the speaker",
    "half of what is said",
    "the person in front of me",
    "a word of this",
]

OVERWHELM_TEMPLATES = [
    "{thing} is drowning everything out",
    "{thing} is overwhelming me",
    "{thing} is hurting my ears",
    "i can't cope with {thing}",
    "{thing} is giving me a headache",
    "{thing} is too much for me",
]
OVERWHELM_THINGS = [
    "the background noise",
    "all this noise",
    "the racket in here",
    "the crowd",
    "the music",
    "everything around me",
    "the clatter",
]

# state pool -> action pool -> intent
PAIRINGS: List[Tuple[Sequence[str], Sequence[str], str]] = [
    (QUIET_STATE, UP_ACTION, "device.volume.increase"),
    (LOUD_STATE, DOWN_ACTION, "device.volume.decrease"),
    (LOUD_STATE, MUTE_ACTION, "device.volume.mute"),
    (SILENT_STATE, UNMUTE_ACTION, "device.volume.unmute"),
]


def contrastive_volume_phrases(
    per_intent: int = 400, seed: int = 0, exclude: Iterable[str] = ()
) -> pd.DataFrame:
    """Generate `per_intent` unique state+action phrases per volume intent.

    `exclude` should be the existing corpus texts (normalised) so augmentation
    never duplicates a real phrase.
    """
    from .data import normalize

    rng = random.Random(seed)
    seen = {normalize(t) for t in exclude}
    rows: List[Dict[str, str]] = []

    for states, actions, intent in PAIRINGS:
        combos = [(s, a, t) for s in states for a in actions for t in TEMPLATES]
        # ~15% of each intent's budget goes to terse, state-free commands.
        combos += [("", a, t) for a in actions for t in ACTION_ONLY_TEMPLATES] * 2
        rng.shuffle(combos)
        made = 0
        for state, action, template in combos:
            if made >= per_intent:
                break
            text = template.format(state=state, action=action, venue=rng.choice(VENUES))
            key = normalize(text)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"text": text, "intent": intent})
            made += 1
        if made < per_intent:
            print(f"  warning: only {made}/{per_intent} unique phrases for {intent}")

    # Complaint-only phrasings, ~25% of a normal intent budget each.
    for templates, things, intent in (
        (TROUBLE_TEMPLATES, TROUBLE_THINGS, "device.volume.increase"),
        (OVERWHELM_TEMPLATES, OVERWHELM_THINGS, "device.volume.decrease"),
    ):
        combos = [(t, th) for t in templates for th in things]
        rng.shuffle(combos)
        made = 0
        for template, thing in combos:
            if made >= per_intent // 4:
                break
            text = template.format(thing=thing)
            key = normalize(text)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"text": text, "intent": intent})
            made += 1

    return pd.DataFrame(rows)


def augment_training_split(df: pd.DataFrame, per_intent: int = 400, seed: int = 0) -> pd.DataFrame:
    """Append contrastive phrases to the *train* rows of an already-split frame.

    Dev and test are left untouched, so reported scores stay honest: the model
    is rewarded only if the new phrases help it on phrasings it never saw.
    """
    if "split" not in df.columns:
        raise ValueError("call grouped_split() before augmenting")

    from .data import core_of

    extra = contrastive_volume_phrases(per_intent, seed, exclude=df.text)
    extra["core"] = extra.text.map(core_of)
    extra["split"] = "train"

    # Do not collide with a dev/test core group — that would reintroduce leakage.
    held = set(df.loc[df.split.isin(["dev", "test"]), "core"])
    before = len(extra)
    extra = extra[~extra.core.isin(held)]
    dropped = before - len(extra)
    if dropped:
        print(f"  dropped {dropped} generated phrases colliding with dev/test groups")

    print(
        f"  + {len(extra)} contrastive phrases into train "
        f"({extra.intent.value_counts().to_dict()})"
    )
    return pd.concat([df, extra], ignore_index=True)


def contrastive_coverage(df: pd.DataFrame) -> float:
    """Fraction of rows containing both a state word and an opposite action."""
    import re

    lower = df.text.str.lower()
    state = re.compile("|".join(re.escape(s) for s in LOUD_STATE + QUIET_STATE))
    action = re.compile("|".join(re.escape(a) for a in UP_ACTION + DOWN_ACTION))
    both = lower.str.contains(state, regex=True) & lower.str.contains(action, regex=True)
    return float(both.mean())
