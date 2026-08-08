"""
Negation suppression must be language-aware (charter A4).

`_is_negated` guards `contains` keyword rules: "I don't want to translate this"
must not fire Cmd.TranslationStart. It previously consulted a hardcoded
English tuple with no language input, so the guard was English-only for every
language.

SCOPE NOTE — this defect is LATENT, not live. The shipped schema declares 28
`regex` and 4 `exact` keyword triggers and **zero** `contains` rules, so
`_is_negated` is never reached in the current configuration. It becomes live the
moment a `contains` rule is authored. These tests therefore drive the guard
directly and through a synthetic `contains` rule, rather than through the
shipped schema.
"""

import importlib
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_LOC = _ROOT / "language_packs"


def _load_classifier_module():
    """Import the classifier submodule directly.

    `importlib.spec_from_file_location` cannot be used here: classifier.py does
    a relative `from .manifest import ...`, which needs a real package context.
    Importing the submodule by name gives it one without evaluating the package
    __init__'s engine import chain.
    """
    pkg_root = str(_ROOT / "packages" / "runtime")
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    return importlib.import_module("nlu_engine.classifier")


_M = _load_classifier_module()


def _cues(lang):
    path = _LOC / lang / "nlu_schema.json"
    if not path.exists():
        pytest.skip(f"no localization for {lang}")
    return tuple(json.loads(path.read_text(encoding="utf-8"))["negation_cues"])


# --------------------------------------------------------------------------- #
# The regression: a non-English negation must suppress a non-English term
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("lang,text,term", [
    ("fr", "je ne veux pas traduire ça", "traduire"),
    ("fr", "sans traduire", "traduire"),
    ("fr", "annule la traduction", "traduction"),
    ("de", "ich will nicht übersetzen", "übersetzen"),
    ("de", "ohne übersetzen", "übersetzen"),
    ("de", "kein übersetzen", "übersetzen"),
    ("da", "jeg vil ikke oversætte", "oversætte"),
    ("da", "uden oversætte", "oversætte"),
    ("da", "aldrig oversætte", "oversætte"),
])
def test_negation_suppresses_in_each_language(lang, text, term):
    assert _M._is_negated(text, term, _cues(lang)), (
        f"[{lang}] {text!r}: negation before {term!r} was not detected"
    )


@pytest.mark.parametrize("lang,text,term", [
    ("fr", "traduire ça", "traduire"),
    ("de", "übersetzen bitte", "übersetzen"),
    ("da", "oversætte det", "oversætte"),
])
def test_plain_command_is_not_suppressed(lang, text, term):
    """The guard must not fire on an unnegated command."""
    assert not _M._is_negated(text, term, _cues(lang))


def test_english_default_still_works():
    """English behaviour via the fallback table is unchanged."""
    assert _M._is_negated("i don't want to translate", "translate")
    assert _M._is_negated("never translate", "translate")
    assert not _M._is_negated("translate this", "translate")


def test_english_cues_do_not_apply_to_other_languages():
    """The actual regression: French text with English cues must NOT suppress.

    This is what the old hardcoded table did for every non-English language.
    """
    assert not _M._is_negated("je ne veux pas traduire", "traduire",
                              _M._DEFAULT_NEGATIONS)
    # ...and the French cues do.
    assert _M._is_negated("je ne veux pas traduire", "traduire", _cues("fr"))


# --------------------------------------------------------------------------- #
# Word-boundary matching — why short cues are safe
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("lang,text,term", [
    # de "ne" must not match inside "ohne"/"eine"; "kein" not inside "keinesfalls"
    ("de", "eine übersetzen", "übersetzen"),
    # da "ingen" must not match inside a longer word
    ("da", "ingenting oversætte", "oversætte"),
])
def test_short_cues_do_not_match_inside_words(lang, text, term):
    assert not _M._is_negated(text, term, _cues(lang)), (
        f"[{lang}] a cue matched inside an unrelated word in {text!r} — "
        f"boundary matching regressed, and this would block a real command"
    )


def test_english_substring_false_positive_is_gone():
    """'not' inside 'nothing'/'another' must no longer suppress.

    The old implementation used a bare substring test, so "another translate"
    contained "not" and was silently suppressed.
    """
    assert not _M._is_negated("another translate", "translate")
    assert not _M._is_negated("nothing here, translate", "translate")


# --------------------------------------------------------------------------- #
# Wiring — the cues reach the classifier's `contains` path
# --------------------------------------------------------------------------- #

def test_classifier_defaults_to_english_table():
    cls = _M.IntentClassifier.__new__(_M.IntentClassifier)
    cls.negation_cues = None or _M._DEFAULT_NEGATIONS
    assert cls.negation_cues == _M._DEFAULT_NEGATIONS


def test_contains_rule_is_suppressed_via_injected_cues():
    """End-to-end through _keyword_match with a synthetic `contains` rule.

    The shipped schema has none, so this proves the wiring that a future
    `contains` rule would depend on.
    """
    cls = _M.IntentClassifier.__new__(_M.IntentClassifier)
    cls._kw_rules = [{"type": "contains", "terms": ["traduire"],
                      "intent": "Cmd.TranslationStart"}]
    cls.negation_cues = _cues("fr")
    cls.last_keyword_tier = None

    intent = cls._keyword_match("traduire ça")
    assert intent == "Cmd.TranslationStart"
    assert cls.last_keyword_tier == "contains"

    intent = cls._keyword_match("je ne veux pas traduire ça")
    assert intent is None, "French negation did not suppress the contains hit"


def test_every_shipped_lexicon_declares_cues():
    for lang in ("fr", "de", "da"):
        path = _LOC / lang / "nlu_schema.json"
        if not path.exists():
            continue
        lex = json.loads(path.read_text(encoding="utf-8"))
        assert lex.get("negation_cues"), f"{lang} lexicon has no negation_cues"
        assert lex.get("_negation_cues_note"), f"{lang} cues lack the provenance note"


def test_cues_are_distinct_from_the_confirmation_lexicon():
    """`negation_cues` and `negative` are different linguistic functions.

    `negative` is the yes/no ANSWER lexicon ('non merci', 'nej tak'); these are
    grammatical negators that precede a command. Reusing one for the other was
    the tempting shortcut and would have missed French 'ne ... pas' entirely.
    """
    for lang in ("fr", "de", "da"):
        path = _LOC / lang / "nlu_schema.json"
        if not path.exists():
            continue
        lex = json.loads(path.read_text(encoding="utf-8"))
        assert set(lex["negation_cues"]) != set(lex["negative"]), lang
