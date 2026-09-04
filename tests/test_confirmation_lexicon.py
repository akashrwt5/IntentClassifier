"""The authored Cmd.SendMessage confirmation phrases, against the real resolver.

WHY THIS FILE EXISTS. Confirmation is resolved deterministically from the
content's `affirmative` / `negative` lists, not by the classifier — there is no
`Cmd.SendMessage - yes` class in the model, and there should not be one. That
makes these two lists the entire contract for "did the user agree?", and a
contract with fifty authored utterances on each side deserves to be checked
against all hundred rather than against the half-dozen someone remembers.

It caught three inversions when it was written — utterances where the user
DECLINED and the resolver said yes:

    "I don’t want to send."   ->  True     (curly apostrophe; `don't` never matched)
    "Forget send."            ->  True     (`send` is affirmative, nothing negative matched)
    "Skip send."              ->  True     (same)

An inversion here is not a missed match. It sends a message the user just
refused, so the assertions below separate the two: a `None` is a re-prompt and
merely unhelpful, a wrong boolean is the failure that matters.

Calls `NLUEngine._yes_no` UNBOUND against a stub carrying the real content
lists. That keeps the test honest — it exercises the shipped function, not a
copy of it — while needing no trained model, so it runs in a fresh clone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in ("packages/runtime", "packages/buildtime"):
    if str(_ROOT / _p) not in sys.path:
        sys.path.insert(0, str(_ROOT / _p))

SCHEMA = json.loads(
    (_ROOT / "language_packs" / "en" / "nlu_schema.json").read_text(encoding="utf-8"))

# Authored for the legacy `Cmd.SendMessage - no` intent. Kept verbatim,
# typographic apostrophes included — those are what dictation produces and what
# broke three of them.
DECLINE = [
    "No.", "No thank you.", "No don't send.", "No please don't.",
    "No I changed my mind.", "No stop.", "No cancel.", "Cancel send.", "Cancel.",
    "Cancel please.", "Stop send.", "Don't send.", "Don't send please.",
    "Please don't send.", "I don’t want to send.", "I changed my mind.",
    "No not now.", "Not now.", "Wait no.", "No hold on.", "Hold on cancel.",
    "No don't do it.", "No don't push.", "Don't push send.", "Forget it.",
    "Forget send.", "Leave it.", "Leave message.", "Skip send.", "Skip.",
    "No stop now.", "Stop it.", "Please cancel.", "No that’s wrong.",
    "No that’s not right.", "No not okay.", "No that’s not fine.",
    "No no send.", "No not this time.", "No stop please.", "Cancel message.",
    "Cancel my send.", "Don't send my message.", "No don't send message.",
    "No cancel message.", "No cancel audio.", "Cancel audio send.",
    "Cancel recording.", "Don't send recording.", "Please don't send my recording.",
]

# Authored for the legacy `Cmd.SendMessage - yes` intent.
ACCEPT = [
    "Yes.", "Yes please.", "Yes send it.", "Yes send.", "Yes do it.",
    "Yes please send.", "Yes I want to send.", "Yes I do.", "Yes okay send.",
    "Yes go ahead.", "Sure.", "Sure send it.", "Sure go ahead.",
    "Sure please send.", "Okay.", "Okay send.", "Okay yes send.",
    "Okay go ahead.", "Please send.", "Send it.", "Send now.", "Do it.",
    "Do it now.", "Go ahead.", "Go ahead send.", "That’s right.",
    "That’s okay.", "That’s fine.", "Alright.", "Alright send.",
    "Alright yes.", "Alright please send.", "Yes send now.", "Send away.",
    "Yes that’s fine.", "Yes that’s okay.", "I confirm.", "I agree.",
    "I approve.", "Yes approved.", "Approved send.", "Yes go.", "Yes push it.",
    "Yes push send.", "Yes please do.", "Yes I said send.",
    "Yes I want this sent.", "Yes that’s what I want.", "Yes I mean it.",
    "Yes please go.",
]


@pytest.fixture(scope="module")
def resolve():
    """`NLUEngine._yes_no` bound to the shipped content lists, without a model."""
    from nlu_engine.engine import NLUEngine

    class _Stub:
        affirmative = set(SCHEMA["affirmative"])
        negative = set(SCHEMA["negative"])
        _UNCERTAIN = NLUEngine._UNCERTAIN
        _NO_IDIOMS = NLUEngine._NO_IDIOMS
        _APOSTROPHES = NLUEngine._APOSTROPHES

    stub = _Stub()
    return lambda text: NLUEngine._yes_no(stub, text)


def _report(resolve, phrases, want):
    inverted, unresolved = [], []
    for p in phrases:
        got = resolve(p)
        if got is want:
            continue
        (unresolved if got is None else inverted).append((p, got))
    return inverted, unresolved


def test_no_declining_phrase_is_read_as_consent(resolve):
    """The one that matters. A wrong boolean here SENDS a refused message."""
    inverted, _ = _report(resolve, DECLINE, False)
    assert not inverted, (
        "declining utterances resolved as CONSENT:\n  "
        + "\n  ".join(f"{p!r} -> {g}" for p, g in inverted))


def test_no_accepting_phrase_is_read_as_refusal(resolve):
    inverted, _ = _report(resolve, ACCEPT, True)
    assert not inverted, (
        "accepting utterances resolved as REFUSAL:\n  "
        + "\n  ".join(f"{p!r} -> {g}" for p, g in inverted))


def test_every_authored_decline_resolves(resolve):
    """Unresolved is a re-prompt, not a wrong action — asserted separately so a
    regression says which kind it is."""
    _, unresolved = _report(resolve, DECLINE, False)
    assert not unresolved, (
        f"{len(unresolved)}/{len(DECLINE)} decline utterances re-prompt instead "
        "of resolving:\n  " + "\n  ".join(repr(p) for p, _ in unresolved))


def test_every_authored_accept_resolves(resolve):
    _, unresolved = _report(resolve, ACCEPT, True)
    assert not unresolved, (
        f"{len(unresolved)}/{len(ACCEPT)} accept utterances re-prompt instead "
        "of resolving:\n  " + "\n  ".join(repr(p) for p, _ in unresolved))


def test_typographic_apostrophes_read_the_same_as_straight_ones(resolve):
    """`don't` is authored with a straight apostrophe; dictation emits U+2019.
    Before the fold, "Don’t send" matched no negative, matched `send`, and
    returned True."""
    for straight, curly in (("Don't send.", "Don’t send."),
                            ("I don't want to send.", "I don’t want to send."),
                            ("No please don't.", "No please don’t.")):
        assert resolve(straight) == resolve(curly) is False, (
            f"{curly!r} does not resolve like {straight!r}")


def test_confirmation_vocabulary_is_language_level_not_per_intent(resolve):
    """No intent carries its own yes/no lists.

    They were tried and reverted: `workflows.schema.json` fixes an intent's key
    set with `additionalProperties: false`, so a per-intent list cannot reach a
    device and would have been Python-only. The safety a per-intent list is
    reached for comes from the CALL SITE instead — `_yes_no` runs only with a
    confirmation context already active.
    """
    offenders = [name for name, cfg in SCHEMA["intents"].items()
                 if "affirmative_lexicon" in cfg or "negative_lexicon" in cfg]
    assert not offenders, (
        f"per-intent confirmation lexicons found on {offenders} — these cannot "
        "reach a device pack; put the phrases in platform.yaml's affirmative/"
        "negative instead")
