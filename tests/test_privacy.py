"""
Privacy invariants. This is a hearing-aid app: utterances are medical-context
speech, so what the engine retains and emits is a safety property, not a
preference.

Locks in the two guarantees stated in engine.py:
  - the raw utterance is never embedded in an NLUResult;
  - raw-utterance logging is OPT-IN and off by default.
"""

import importlib
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "packages"))
logging.disable(logging.CRITICAL)

from nlu import NLUEngine  # noqa: E402


def test_utterance_logging_is_off_by_default(monkeypatch):
    monkeypatch.delenv("NLU_LOG_UTTERANCES", raising=False)
    assert NLUEngine()._log_utterances is False


def test_utterance_logging_requires_an_explicit_opt_in(monkeypatch):
    # Only the documented values enable it; anything else must stay off, so a
    # stray value can never silently start logging medical-context speech.
    for value in ("1", "true", "TRUE", "yes"):
        monkeypatch.setenv("NLU_LOG_UTTERANCES", value)
        assert NLUEngine()._log_utterances is True, value
    for value in ("", "0", "false", "no", "off", "maybe"):
        monkeypatch.setenv("NLU_LOG_UTTERANCES", value)
        assert NLUEngine()._log_utterances is False, value


def test_fallback_result_never_echoes_the_raw_utterance():
    eng = NLUEngine()
    secret = "qzxw blarg flooble zzqq wibble"
    r = eng.handle("privacy-1", secret)
    assert r.type == "FALLBACK"
    blob = str(r.to_dict()).lower()
    for token in ("qzxw", "flooble", "zzqq", "wibble"):
        assert token not in blob, f"raw utterance token {token!r} leaked into the result"


def test_result_carries_no_url_field():
    # The GenAI URL is configuration, not a per-result field — shipping it on
    # the result would put the utterance in a query string.
    r = NLUEngine().handle("privacy-2", "asdfghjkl qwerty zxcvb")
    assert not hasattr(r, "url")
    assert "url" not in r.to_dict()
