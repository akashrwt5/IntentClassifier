"""Enlarge the OOD suite from 45 rows to something that can be read.

`ood_test` was the last suite never enlarged. At 45 rows the teacher/student
gap of 0.956 vs 0.867 is four rows, which is not a result — no noise floor was
ever measured on it, so nobody could say whether four rows meant anything.

WHY THIS IS NOT JUST "WRITE MORE SENTENCES"
-------------------------------------------
An OOD suite is the one suite where a mislabelled row is worse than a missing
row. Every other suite says "this input means X"; this one says "this input
means NOTHING WE SUPPORT". Get that wrong and you are training and grading the
gate to refuse a request the product actually handles.

Checking the corpus before writing showed the original 45 rows had exactly that
problem. Three examples:

  "turn the television volume down"   Help_Accessories contains, verbatim,
                                      "how do i turn up the tv sound?" — TV
                                      volume is in-domain. 'television' appears
                                      54 times, 48 of them Cmd.MemoryChange.
  "check my sleep score"              borderline; sleep appears only 7 times
                                      and never as its own intent, so this one
                                      survives.
  "translate this document into
   french"                            Cmd.TranslationStart and Help_Translate
                                      are both supported intents.

So this file does not trust my judgement about what is out of domain. It
generates candidates, then puts every candidate — the new ones AND the original
45 — through two independent checks, and writes the failures to a file instead
of dropping them quietly.

  check 1, lexical:  does the sentence contain a domain-object noun? These are
                     derived from the 57 intent names and from reading the
                     corpus, not guessed. 'tv', 'heart rate', 'accessories',
                     'memory', 'tinnitus' make a sentence about this product no
                     matter what is asked about them.
  check 2, fuzzy:    is any real corpus sentence >= 85 token_sort similar? This
                     catches paraphrases the token list cannot.

A row failing check 1 is not necessarily OOD-invalid — "how much do new hearing
aids cost" is genuinely unsupported and genuinely about the device. Those go to
ADJUDICATED with a written reason, so the exception is visible in the diff
rather than buried in a threshold.

    python scripts/ood_generate.py                # writes data/challenge/ood.csv
    python scripts/ood_generate.py --report       # also prints what was dropped
"""
from __future__ import annotations

import argparse
import itertools
import random
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import normalize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/en.csv"
OUT = ROOT / "data/challenge"
FALLBACK = "Default Fallback Intent"

# ---------------------------------------------------------------------------
# Two lists, because two different things went wrong in the first run.
#
# OBJECT nouns name the product or one of its features. A sentence containing
# one is ABOUT this product whatever it asks, so the hit is never waived.
# Sources: the 57 intent names plus corpus counts — 'television' 54 rows of
# which 48 are Cmd.MemoryChange, ' tv' 124 rows of which 53 are
# Cmd.StreamingStart, 'heart rate' 84 rows across two supported Help intents,
# 'music' 66 rows of which 50 are Cmd.MemoryChange.
#
# ACTION words name something the product does. On their own they are NOT
# evidence of domain: "mute the microwave beeping" is one of the best near-OOD
# rows in the suite and the first version of this file threw it away for
# containing 'mute'. So an action hit is waived when the sentence is anchored
# to a foreign object — a noun the corpus barely knows. The anchor's corpus
# count is measured, not asserted; see foreign_anchor().
# ---------------------------------------------------------------------------
OBJECT_TOKENS = [
    "hearing aid", "hearing aids", "my aid", "my aids", "the aids", "earpiece",
    "receiver", "wax guard", "charger case",
    # Hearing-aid ACCESSORIES are in-domain and are the product's first
    # priority: the TV streamer, the remote mic and Auracast all connect to the
    # aids, and the corpus already routes them (streamer 21 rows ->
    # Help_Accessories 19; remote mic 43 -> Help_Accessories 26 /
    # Cmd.StreamingStart 14). "streamer" would NOT have matched the "stream "
    # token below, so an OOD list containing one could have slipped through and
    # taught the gate to refuse the product's core accessory.
    "streamer", "mic", "microphone", "auracast", "tv connector",
    "remote control", "companion", "hearing device",
    "tv", "television", "accessor", "streaming", "stream", "program",
    "programme", "memory", "memories", "tinnitus", "thrive", "edge mode",
    "mask mode", "fall alert", "fall detection", "self check", "selfcheck",
    "heart rate", "bpm", "resting pulse", "intellivoice", "hearshare",
    "wicros", "demo mode", "remote programming", "audiologist", "hearing care",
    "music", "voice assistant", "thrive assistant", "the app", "app settings",
]
ACTION_TOKENS = [
    "volume", "mute", "unmute", "captions", "transcribe", "transcription",
    "translate", "translation", "pairing", "pair my", "bluetooth",
    "battery", "batteries", "reminder", "remind me",
    "step count", "steps", "calories", "activity", "workout",
]
# Tokens matched as PREFIXES, because the suffix varies: accessory/accessories,
# pairing/paired. Everything else is matched as a whole word.
#
# This distinction is not cosmetic. Matching as bare substrings, "mic" matched
# "microwave" and silently removed six good near-OOD rows about a microwave —
# the exact rows the suite most needs, since a microwave is the thing sitting
# next to the user making noise.
PREFIX_TOKENS = {"accessor", "pair my", "stream"}


def _compile(tokens: list[str]) -> re.Pattern:
    parts = []
    for t in tokens:
        esc = re.escape(t)
        parts.append(rf"\b{esc}" if t in PREFIX_TOKENS else rf"\b{esc}\b")
    return re.compile("|".join(parts))


OBJECT_RE = _compile(OBJECT_TOKENS)
ACTION_RE = _compile(ACTION_TOKENS)

# Nouns that anchor a sentence to something other than the hearing aids.
# Every one of these is checked against the corpus at run time and dropped
# from the set if the corpus turns out to know it well.
FOREIGN_NOUNS = [
    "microwave", "doorbell", "alarm clock", "oven", "smoke alarm",
    "washing machine", "games console", "kitchen radio", "baby monitor",
    "landline", "smart speaker", "computer speakers", "laptop", "thermostat",
    "blinds", "heating", "kettle", "front door", "garage door", "sprinklers",
    "porch light", "hallway lights", "bedroom lights", "wifi", "wallpaper",
    "screenshot", "torch", "aeroplane mode", "downloads folder",
]


def foreign_anchor(text: str, allowed: set[str]) -> str | None:
    low = " " + text.lower() + " "
    for n in allowed:
        if n in low:
            return n
    return None


def domain_hit(text: str, foreign_ok: set[str]) -> tuple[str, str] | None:
    """Returns (kind, token) or None."""
    low = " " + text.lower() + " "
    m = OBJECT_RE.search(low)
    if m:
        return ("object", m.group(0).strip())
    m = ACTION_RE.search(low)
    if m:
        if foreign_anchor(text, foreign_ok):
            return None
        return ("action", m.group(0).strip())
    return None


# ---------------------------------------------------------------------------
# NEAR-OOD — lexically and pragmatically adjacent, genuinely unsupported.
# Each family is a separate column in the report, because "OOD rejection 0.87"
# averaged over families hides which kind of input actually gets through.
# ---------------------------------------------------------------------------
# Devices deliberately chosen for having ZERO or near-zero corpus presence:
# laptop 0, thermostat 0, brightness 0, wifi 0, glasses 0, keys 0, radio 1.
OTHER_DEVICES = [
    "microwave", "doorbell", "alarm clock", "oven timer", "smoke alarm",
    "washing machine", "computer speakers", "games console", "kitchen radio",
    "baby monitor", "landline", "smart speaker", "tumble dryer",
    "dishwasher", "extractor fan", "carbon monoxide alarm",
]
AUDIO_FRAMES = [
    "turn the {d} down", "turn the {d} up", "can you silence the {d}",
    "the {d} is too loud", "why is the {d} making that noise",
    "switch the {d} off", "make the {d} stop beeping",
]

HOME_THINGS = [
    ("the hallway lights", "turn on {t}"), ("the bedroom lights", "turn off {t}"),
    ("the blinds", "close {t}"), ("the blinds", "open {t}"),
    ("the heating", "put {t} on for an hour"), ("the heating", "turn {t} down"),
    ("the thermostat", "set {t} to twenty degrees"),
    ("the front door", "is {t} locked"), ("the kettle", "has {t} boiled"),
    ("the oven", "set {t} to one eighty"), ("the sprinklers", "run {t} tonight"),
    ("the garage door", "close {t} for me"),
    ("the fan", "put {t} on low"), ("the porch light", "leave {t} on tonight"),
]

# Health readings this product does not measure. Heart rate, steps, calories,
# walking, running, cycling, standing and exercise are ALL supported intents
# and are excluded on purpose.
# Every entry must be a singular noun phrase that reads correctly in ALL of
# HEALTH_FRAMES. The first version included "how well i slept" and the plural
# "my oxygen levels", which produced "what is how well i slept" and "what is my
# oxygen levels" — four malformed rows out of 45. Nobody says those, so a model
# that rejects them has been credited for nothing, and generated grammar errors
# are noise I introduced rather than variation users produce. assert_wellformed()
# below stops this returning.
HEALTH_OTHER = [
    "my blood pressure", "my blood sugar", "my cholesterol",
    "my oxygen level", "my temperature", "my weight this week",
    "my sleep score", "my hydration for today",
    "my peak flow reading", "my last cholesterol test",
    "my eyesight test result", "my vitamin d level",
    "my breathing rate overnight", "my resting temperature",
]
HEALTH_FRAMES = [
    "what is {t}", "can you tell me {t}", "show me {t}",
    "has {t} changed since last month", "read me {t}",
    "log {t} for the doctor",
]

# Cheap structural check. It cannot judge meaning — it catches the two shapes
# that actually went wrong: a slot that is a clause rather than a noun phrase,
# and plural agreement after "what is".
_MALFORMED = [
    (r"^(what is|has) (my )?\w+ (levels|readings|results|scores)\b",
     "plural noun after a singular verb"),
    (r"\b(what is|has|read me) (how|why|when|where|whether)\b",
     "clause used where a noun phrase belongs"),
]


def assert_wellformed(rows: list[dict]) -> None:
    bad = [(r["text"], why) for r in rows
           for pat, why in _MALFORMED if re.search(pat, r["text"])]
    if bad:
        lines = "\n".join(f"  {t!r}: {w}" for t, w in bad)
        raise SystemExit(
            f"{len(bad)} generated rows are not well-formed English:\n{lines}\n"
            "Fix the slot list or the frame list — do not ship these. A model "
            "rejecting a sentence no human would say has not been tested.")

# Placing a call is not a supported intent. Wording avoids message/note/voice
# note entirely, because Cmd.SendMessage and Cmd.ListenMessage ARE supported.
CALL_TARGETS = ["my daughter", "the surgery", "my son at work", "the pharmacy",
                "the district nurse", "my brother", "the taxi company"]
CALL_FRAMES = [
    "give {t} a ring", "put me through to {t}", "dial {t} for me",
    "call {t} back", "get {t} on the line",
]

PHONE_OTHER = [
    "what is the wifi password here", "how do i free up storage on my phone",
    "take a screenshot of this", "increase the font size on my screen",
    "turn on the torch", "put my phone on aeroplane mode",
    "how many photos are on this phone", "empty my downloads folder",
    "change my phone wallpaper", "install the latest system update",
    "why is my phone so slow today", "show me my recent downloads",
]

COMMERCE = [
    "where is the nearest chemist", "what time does the pharmacy shut",
    "order me a taxi for four o'clock", "add milk to my shopping list",
    "how much is a first class stamp now", "when does the post office open",
    "is the library open on a sunday", "book a table for two at seven",
    "renew my prescription online", "what is on at the cinema tonight",
]

# ---------------------------------------------------------------------------
# FAR-OOD — no plausible reading as a device command.
# ---------------------------------------------------------------------------
KNOWLEDGE = [
    "what is the capital of portugal", "how tall is ben nevis",
    "define photosynthesis", "who wrote wuthering heights",
    "what is twelve times fourteen", "how many miles in a kilometre",
    "when did the second world war end", "what is the exchange rate for euros",
    "how far away is the moon", "what does gdp actually mean",
    "who painted the haywain", "what year did the titanic sink",
    "how long do tortoises live", "what is the boiling point of water",
    "how many bones are in the human body", "who was the first man in space",
    "what is the longest river in africa", "how do volcanoes actually form",
    "what language do they speak in brazil", "how many pints in a gallon",
    "when was the printing press invented", "what causes the northern lights",
    "how does a fridge keep things cold", "what is the population of scotland",
    "why is the sky blue",
]
WEATHER_TIME = [
    "what is the weather going to do tomorrow", "will it rain this afternoon",
    "what time is it in sydney", "when do the clocks go back",
    "how cold is it outside right now", "is it going to be icy in the morning",
    "what time does the sun set today", "how many days until christmas",
    "what is the forecast for the weekend", "is it warm enough to sit outside",
    "what day of the week is the fourteenth", "how windy is it meant to get",
    "do i need an umbrella today", "when is the next bank holiday",
    "what time does it get dark now",
]
NEWS_SPORT = [
    "read me the headlines", "who won the cricket yesterday",
    "what is the football score", "any news about the election",
    "how did the horses do at cheltenham", "what happened on the news today",
    "is there any traffic on the motorway", "when is the next test match",
    "what is happening with the trains this week",
    "did anything happen overnight", "who is top of the league",
    "give me the local news",
]
CHITCHAT = [
    "tell me a joke", "sing me a song", "how are you feeling today",
    "what is your favourite colour", "do you ever get bored",
    "say something interesting", "amuse me for a minute",
    "what should i have for lunch", "keep me company for a bit",
    "who made you", "are you a real person", "cheer me up",
    "do you like being switched on", "what did you do all night",
    "have a guess at my age", "tell me something i do not know",
    "do you get lonely in there", "say good morning to me properly",
    "what would you do if you were me", "i am a bit fed up today",
]
TASK_ASSIST = [
    "how do i make a proper lasagne", "book me a flight to dublin",
    "what is the quickest way to the station", "how do i get rid of greenfly",
    "give me a recipe for scones", "how do i change a flat tyre",
    "what should i plant in october", "how do i get a stain out of a carpet",
    "write a birthday card verse for my wife",
    "how do i knit a simple scarf", "help me plan a week of meals",
    "what is a good present for a ten year old",
    "how do i descale a coffee machine", "work out my mortgage payment",
    "convert two hundred pounds into dollars",
    "how long should i roast a chicken", "what is a good film for tonight",
    "how do i prune an apple tree",
]

# ---------------------------------------------------------------------------
# The original 45. Kept as a labelled subset so every number reported before
# today stays reproducible — but they go through the same checks as everything
# else, and three of them do not survive.
# ---------------------------------------------------------------------------
ORIGINAL_45 = [
    ("turn the television volume down", "near"), ("make the phone louder", "near"),
    ("increase the screen brightness", "near"), ("turn up my car radio", "near"),
    ("lower the thermostat", "near"), ("mute the microwave beeping", "near"),
    ("switch my phone to silent mode", "near"),
    ("pair my headphones to the laptop", "near"), ("find my car keys", "near"),
    ("where did i leave my glasses", "near"),
    ("how is my blood pressure today", "near"),
    ("what is my blood sugar reading", "near"), ("check my sleep score", "near"),
    ("how many hours did i sleep", "near"), ("call my audiologist", "near"),
    ("book me a dentist appointment", "near"),
    ("text my son that i am running late", "near"),
    ("read my emails to me", "near"), ("start recording a video", "near"),
    ("translate this document into french", "near"),
    ("transcribe the meeting recording on my laptop", "near"),
    ("turn on the living room lights", "near"),
    ("set my oven to 180 degrees", "near"),
    ("how far is it to the pharmacy", "near"),
    ("order more hearing aid batteries online", "near"),
    ("how much do new hearing aids cost", "near"),
    ("is my warranty still valid", "near"), ("what is the wifi password", "near"),
    ("update my phone software", "near"),
    ("increase the font size on my phone", "near"),
    ("what is the weather going to be tomorrow", "far"),
    ("amuse me with something funny", "far"), ("play some jazz music", "far"),
    ("what time is it in tokyo", "far"), ("who won the football last night", "far"),
    ("what is the capital of portugal", "far"), ("how do i make a lasagne", "far"),
    ("book me a flight to dublin", "far"),
    ("what is twelve times fourteen", "far"), ("read me the news headlines", "far"),
    ("how high is the tallest mountain on earth", "far"),
    ("define photosynthesis", "far"),
    ("what is the exchange rate for euros", "far"), ("sing me a song", "far"),
    ("how do i change a flat tyre", "far"),
]

# ---------------------------------------------------------------------------
# Rows that trip the lexical check but are still valid OOD, each with the
# reason it is an exception. This list is deliberately short and deliberately
# in source control — an exception you have to type out is one you have to
# defend.
# ---------------------------------------------------------------------------
ADJUDICATED = {
    "how much do new hearing aids cost":
        "pricing. no Help_ intent covers commercial questions; the device noun "
        "is the subject, not the request.",
    "order more hearing aid batteries online":
        "Cmd.BatteryLevel reports charge and Help_Battery explains charging. "
        "Neither places an order.",
    "when is my hearing test due":
        "appointments are not in the taxonomy; Help_HearingCareAnywhereConnect "
        "is about connecting to a clinician, not scheduling.",
    # REMOVED: "can i wear my hearing aids in the shower". I wrote the
    # justification "Help_CleanCare covers cleaning, not water tolerance" and
    # the fuzzy check immediately answered it — Help_CleanCare contains
    # "can i wear my aids in the shower" at 89 similarity. The exception was
    # wrong and an automated check caught it, which is the entire argument for
    # having the check sit downstream of the adjudication list rather than
    # upstream of it.
    "will my hearing aids set off the airport scanner":
        "no intent covers travel. Included because users really ask it.",
}


def build_candidates(rng: random.Random) -> list[dict]:
    rows: list[dict] = []

    def add(text: str, fam: str, kind: str, source: str = "generated"):
        rows.append(dict(text=" ".join(text.split()), intent=FALLBACK,
                         ood_type=kind, family=fam, source=source))

    for dev, frame in itertools.product(OTHER_DEVICES, AUDIO_FRAMES):
        add(frame.format(d=dev), "other_device_audio", "near")
    for thing, frame in HOME_THINGS:
        add(frame.format(t=thing), "home_control", "near")
    for topic, frame in itertools.product(HEALTH_OTHER, HEALTH_FRAMES):
        add(frame.format(t=topic), "health_other", "near")
    for target, frame in itertools.product(CALL_TARGETS, CALL_FRAMES):
        add(frame.format(t=target), "phone_calls", "near")
    for t in PHONE_OTHER:
        add(t, "phone_other", "near")
    for t in COMMERCE:
        add(t, "errands", "near")
    for t in ADJUDICATED:
        add(t, "device_but_unsupported", "near")

    for t in KNOWLEDGE:
        add(t, "general_knowledge", "far")
    for t in WEATHER_TIME:
        add(t, "weather_time", "far")
    for t in NEWS_SPORT:
        add(t, "news_sport", "far")
    for t in CHITCHAT:
        add(t, "chitchat", "far")
    for t in TASK_ASSIST:
        add(t, "task_assistant", "far")

    for text, kind in ORIGINAL_45:
        add(text, "original_45", kind, source="original_45")

    # Sampling down the big product families keeps one family from dominating
    # the headline number. other_device_audio alone would otherwise be 84 rows.
    capped: list[dict] = []
    by_family: dict[str, list[dict]] = {}
    for r in rows:
        by_family.setdefault(r["family"], []).append(r)
    for fam, items in by_family.items():
        if fam in ("other_device_audio", "health_other", "phone_calls"):
            rng.shuffle(items)
            items = items[:45]
        capped.extend(items)
    assert_wellformed(capped)
    return capped


def measure_foreign_nouns(df: pd.DataFrame, max_rows: int = 3) -> set[str]:
    """Keep only anchors the corpus genuinely does not know.

    Asserting 'the corpus has never heard of a thermostat' is the kind of claim
    that is true right up until the dataset changes underneath you. So count.
    """
    low = df["text"].astype(str).str.lower()
    keep, dropped = set(), []
    for n in FOREIGN_NOUNS:
        c = int(low.str.contains(re.escape(n), na=False).sum())
        (keep.add(n) if c <= max_rows else dropped.append((n, c)))
    if dropped:
        print("foreign anchors rejected — corpus knows them:")
        for n, c in dropped:
            print(f"  {n!r} appears {c} times")
    return keep


def validate(rows: list[dict], df: pd.DataFrame,
             fuzzy_threshold: int = 85) -> tuple[list[dict], list[dict]]:
    """Three checks. Returns (kept, rejected-with-reason).

    Order matters: original_45 rows are validated FIRST so that when a row also
    appears in one of the new generated lists, the survivor keeps its
    original_45 attribution. The first version processed generated rows first,
    which reported seven original rows as 'dropped: duplicate' when they had in
    fact survived — an artefact that made the headline count wrong.
    """
    from rapidfuzz import fuzz, process

    corpus = df["text"].astype(str).map(normalize).tolist()
    labels = df["intent"].tolist()
    corpus_exact = set(corpus)
    foreign_ok = measure_foreign_nouns(df)

    rows = sorted(rows, key=lambda r: r["source"] != "original_45")

    kept, rejected = [], []
    seen: set[str] = set()
    for r in rows:
        norm = normalize(r["text"])
        if norm in seen:
            rejected.append({**r, "reject_reason": "duplicate", "detail": ""})
            continue
        seen.add(norm)

        # Verbatim presence in the corpus is disqualifying even when the label
        # agrees. "tell me a joke" is in en.csv as Default Fallback Intent, so
        # the model has seen it and rejecting it measures memory, not the
        # ability to reject an unfamiliar request. Agreement is the reason it
        # is uninteresting, not a reason to keep it.
        if norm in corpus_exact:
            rejected.append({**r, "reject_reason": "verbatim_in_corpus",
                             "detail": f"labelled {labels[corpus.index(norm)]!r}"})
            continue

        hit = domain_hit(r["text"], foreign_ok)
        if hit and r["text"] not in ADJUDICATED:
            kind, tok = hit
            rejected.append({**r, "reject_reason": f"{kind}_token",
                             "detail": tok})
            continue

        match = process.extractOne(norm, corpus, scorer=fuzz.token_sort_ratio,
                                   score_cutoff=fuzzy_threshold)
        if match:
            # A near-identical corpus row that is itself Default Fallback
            # Intent is not a conflict — it is the dataset agreeing that this
            # input is unsupported. Rejecting those threw away
            # "how do i make a lasagne", which the corpus labels Fallback.
            anchor = foreign_anchor(r["text"], foreign_ok)
            if labels[match[2]] == FALLBACK:
                r = {**r, "corpus_confirms": corpus[match[2]]}
            elif anchor and anchor not in corpus[match[2]]:
                # token_sort_ratio is unreliable on five-word sentences: it
                # scored "turn the heating down" at 87 against
                # Cmd.VolumeDecrease "turn down the gain", which share only
                # 'turn' and 'down'. When the sentence is anchored to an object
                # the corpus does not know and the matched row does not contain
                # that object, the similarity is function words and nothing else.
                r = {**r, "similarity_waived": f"{match[1]:.0f} vs "
                                               f"{corpus[match[2]]}"}
            else:
                rejected.append({**r, "reject_reason": "corpus_similarity",
                                 "detail": f"{match[1]:.0f} vs "
                                           f"{labels[match[2]]!r}: "
                                           f"{corpus[match[2]]}"})
                continue
        kept.append(r)
    return kept, rejected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--fuzzy-threshold", type=int, default=85)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(RAW)
    rng = random.Random(args.seed)

    candidates = build_candidates(rng)
    kept, rejected = validate(candidates, df, args.fuzzy_threshold)

    OUT.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(kept)[["text", "intent", "ood_type", "family", "source"]]
    out = out.sort_values(["ood_type", "family", "text"]).reset_index(drop=True)
    out.to_csv(OUT / "ood.csv", index=False)
    pd.DataFrame(rejected).to_csv(OUT / "ood_rejected.csv", index=False)

    print(f"candidates {len(candidates)} -> kept {len(kept)}, "
          f"rejected {len(rejected)}")
    print(f"  near {sum(r['ood_type']=='near' for r in kept)}   "
          f"far {sum(r['ood_type']=='far' for r in kept)}")
    print("\nby family:")
    for fam, n in out["family"].value_counts().sort_index().items():
        print(f"  {fam:26} {n}")

    orig_kept = sum(r["source"] == "original_45" for r in kept)
    orig_dropped = [r for r in rejected if r["source"] == "original_45"]
    print(f"\noriginal 45: {orig_kept} survive, {len(orig_dropped)} dropped")
    for r in orig_dropped:
        print(f"  - {r['text']!r}\n      {r['reject_reason']}: {r['detail']}")

    if args.report:
        print("\nall rejections:")
        for r in rejected:
            print(f"  [{r['reject_reason']}] {r['text']!r}  {r['detail']}")

    print(f"\nwrote {OUT/'ood.csv'} and {OUT/'ood_rejected.csv'}")


if __name__ == "__main__":
    main()
