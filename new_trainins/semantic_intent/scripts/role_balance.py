"""F16 — make direction words stop predicting the label on their own.

THE MEASUREMENT THIS ANSWERS
----------------------------
In the current training set, the bare presence of a word decides the answer:

    'soft'      32 rows -> 100.0% Cmd.VolumeIncrease
    'faint'     35 rows -> 100.0% Cmd.VolumeIncrease
    'harsh'     31 rows -> 100.0% Cmd.VolumeDecrease
    'quieter'  143 rows ->  63.6% Cmd.VolumeDecrease

A model that answers from those words is not broken. It is fitting, and the fit
is nearly free. That is why this fails:

    "it's a bit quieter here can you make it louder"
      -> Cmd.VolumeDecrease 0.7634   (correct answer was the runner-up)

WHAT NOT TO DO
--------------
The obvious fix — add "quieter" to the symptom vocabulary — repairs one word and
leaves the next unseen comparative broken. It also does not touch the reason:
the word is *usable as a shortcut at all*.

THE APPROACH
------------
Some direction words are DUAL-ROLE. "quieter" is both a request ("make it
quieter" -> Decrease) and a description ("it is a bit quieter in here" -> the
room is quiet, so the request will be Increase). The training data only ever
showed the request role. This batch generates both, so the word's own
distribution moves toward 50/50 and stops being evidence.

Two things make this different from the earlier, failed attempts:

  * **Every sentence is natural.** No combination is forced. "it is a bit
    quieter in here, could you push the volume up" is something a person says.
    Earlier generators produced "the bedside lamp is far too quiet" and
    "what is how well i slept" by crossing lists blindly; both halves here are
    real uses of the word.
  * **Both orders are generated.** Request-first and description-first. That
    teaches "the clause carrying the request decides" rather than "the last
    direction word wins", which would be a new shortcut replacing the old one.

Single-role words are deliberately left alone. "faint" at 100% Increase is not a
shortcut, it is a fact — nothing describes a too-quiet room and asks to turn it
down. Only words with two honest roles are balanced.

DISJOINTNESS FROM THE PROBE
---------------------------
`structure_probe.py` measures whether this worked. It uses the same head words
on purpose — that is what is being generalised — but different surrounding
phrasing and different frames, and `assert_disjoint_from_probe()` fails the
build if any exact phrase is shared. Without that, the probe would measure
recall of these very sentences and report success either way.
"""
from __future__ import annotations

import itertools
import random

UP, DOWN = "Cmd.VolumeIncrease", "Cmd.VolumeDecrease"

# ---------------------------------------------------------------------------
# Dual-role words: each is a legitimate REQUEST in one direction and a
# legitimate DESCRIPTION implying the other.
#   (word, direction it requests, direction it implies when describing a room)
# ---------------------------------------------------------------------------
DUAL_ROLE = [
    ("quieter", DOWN, UP),
    ("louder", UP, DOWN),
    ("lower", DOWN, UP),
    ("higher", UP, DOWN),
    ("softer", DOWN, UP),
    ("sharper", UP, DOWN),
]

# Phrasings around the head word. DISJOINT from the probe's
# "a bit {w}" / "on the {w} side" / "quite {w}" / "rather {w}".
DESC_PHRASINGS = [
    "somewhat {w}", "a shade {w}", "a touch {w}", "noticeably {w}",
    "a good deal {w}",
]

# Request wordings. DISJOINT from the probe's "can you make it {x}" /
# "could you turn it {x}" / "please raise the volume" / "please lower the volume".
REQUEST = {
    UP: ["push the volume up", "give me more volume", "step the sound up",
         "crank it up a bit", "boost the level"],
    DOWN: ["pull the volume down", "give me less volume", "step the sound down",
           "damp it down a bit", "cut the level back"],
}

# Frames. DISJOINT from the probe's "it is {sym} in here {req}" /
# "{req}, it is {sym} in here". Both orders are present on purpose.
# Frames take a context clause and a request clause. Both orders are generated
# for BOTH roles, so neither order nor clause position becomes the new shortcut.
FRAMES_CTX_FIRST = [
    "{ctx}, {req}",
    "since {ctx}, {req}",
    "{ctx} at the moment so {req}",
]
FRAMES_REQ_FIRST = [
    "{req} — {ctx}",
    "{req} because {ctx}",
    "{req}, {ctx}",
]

# Context clauses for role 2, where the dual-role word is the REQUEST. These
# describe the situation without using any dual-role word, so they cannot
# themselves carry the label.
NEUTRAL_CTX = {
    UP: ["i can barely follow the conversation",
         "i am struggling to catch what people say",
         "the speech is hard to make out",
         "i keep missing words in here"],
    DOWN: ["the room is noisy today",
           "there is a lot going on in here",
           "my ears are getting tired",
           "it is very busy in this room"],
}

# Ways the dual-role word appears AS the request.
REQUEST_WITH_WORD = ["make it {w}", "i need it {w}", "set it {w}"]

# Rows per word per role. 55 is what it takes to pull "lower" — the most
# lopsided of the six, because the corpus and F2 both lean on it — under the
# 0.65 shortcut threshold that build_targeted_training.py asserts. Raising this
# adds bulk to classes that already hold ~1400 rows; the aim is to move a ratio.
PER_WORD_PER_ROLE = 55


def assert_disjoint_from_probe() -> None:
    """The probe must not be measuring recall of this batch."""
    import structure_probe as sp

    probe_phrases = set()
    for descs, reqs, _ in sp.ARMS.values():
        probe_phrases |= {d.lower() for d in descs}
        probe_phrases |= {r.lower() for r in reqs}

    mine = {p.format(w=w).lower()
            for p, (w, _, _) in itertools.product(DESC_PHRASINGS, DUAL_ROLE)}
    mine |= {r.lower() for rs in REQUEST.values() for r in rs}

    clash = sorted(probe_phrases & mine)
    if clash:
        raise SystemExit(
            f"F16 trains on phrases the structure probe also uses: {clash}\n"
            "The probe would then measure recall of these sentences and report "
            "success whether or not anything generalised. Reword one side.")

    probe_frames = {f.replace("{sym}", "{ctx}") for f in sp.CONFLICT_FRAMES}
    if probe_frames & set(FRAMES_CTX_FIRST + FRAMES_REQ_FIRST):
        raise SystemExit("F16 shares a frame with the structure probe")


def generate(rng: random.Random) -> list[dict]:
    assert_disjoint_from_probe()
    rows: list[dict] = []

    def add(text: str, intent: str, role: str):
        rows.append(dict(text=" ".join(text.split()), intent=intent,
                         source="F16_role_balance", role=role))

    frames = FRAMES_CTX_FIRST + FRAMES_REQ_FIRST

    for word, requests_dir, describes_dir in DUAL_ROLE:
        # ROLE 1 — the word DESCRIBES the room; the request points the other
        # way and is what decides. This is the role the training data never had.
        for phrasing, req, frame in itertools.product(
                DESC_PHRASINGS, REQUEST[describes_dir], frames):
            ctx = f"the room is {phrasing.format(w=word)}"
            add(frame.format(ctx=ctx, req=req), describes_dir, "description")

        # ROLE 2 — the word IS the request, with a context clause that contains
        # no dual-role word. Generated in the SAME frames and the same orders,
        # so the two roles cannot be told apart by sentence shape. If role 1
        # were always long and role 2 always short, frame length would simply
        # replace the word as the shortcut.
        for req_tpl, ctx, frame in itertools.product(
                REQUEST_WITH_WORD, NEUTRAL_CTX[requests_dir], frames):
            add(frame.format(ctx=ctx, req=req_tpl.format(w=word)),
                requests_dir, "request")

    # Cap PER WORD, not globally. A global cap divides evenly across words and
    # ignores the fact that they start from different places: "lower" appears
    # far more often than "softer" in the corpus and in F2, so an even split
    # left it at 0.724 while the others landed near 0.55. The word that needs
    # the most correction is the one a global cap starves.
    rng.shuffle(rows)
    out: list[dict] = []
    for word, _, _ in DUAL_ROLE:
        for role in ("description", "request"):
            sub = [r for r in rows
                   if r["role"] == role and f" {word}" in f" {r['text']}"]
            out.extend(sub[:PER_WORD_PER_ROLE])
    rng.shuffle(out)
    return out
