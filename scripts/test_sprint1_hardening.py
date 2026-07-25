#!/usr/bin/env python3
"""
Sprint-1 production-hardening regression tests.

Each test pins a specific fix from the architecture review so the failure can
never silently return:

  * Item 1 — no raw utterance is ever carried in an NLUResult
  * Item 3 — time-based context + session expiry
  * Item 4 — keyword negation guard + calibrated (non-1.0) confidence
  * Item 5 — slot-fill attempt limit (user is never trapped)

Run:
    python scripts/test_sprint1_hardening.py
    pytest scripts/test_sprint1_hardening.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nlu import NLUEngine                       # noqa: E402
from nlu.context import SessionStore            # noqa: E402
from nlu.classifier import IntentClassifier, KEYWORD_CONFIDENCE  # noqa: E402

_counter = {"n": 0}


def _sid():
    _counter["n"] += 1
    return f"sprint1-{_counter['n']}"


# ---- a controllable clock so expiry is deterministic, not wall-clock based ---
class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


# ============================ Item 1: no PII leak ============================

def test_fallback_result_has_no_url_field():
    """NLUResult must not expose a `url` attribute carrying the utterance."""
    engine = NLUEngine()
    r = engine.handle(_sid(), "asdfghjkl qwerty zxcvb")
    assert r.type == "FALLBACK"
    assert not hasattr(r, "url"), "NLUResult still has a url field"


def test_fallback_dict_never_contains_raw_text():
    """
    On the GenAI fallback path, the serialized result (what a caller would log)
    must not echo the raw utterance. Note: legitimate slot values — e.g. a
    reminder's name — DO contain user content by design; that is the app's data,
    not telemetry, and is out of scope for this check. Here we use a truly
    out-of-scope utterance that falls through to FALLBACK.
    """
    engine = NLUEngine()
    secret = "qzxw blarg flooble zzqq wibble"
    r = engine.handle(_sid(), secret)
    assert r.type == "FALLBACK", f"expected FALLBACK, got {r.type}"
    blob = str(r.to_dict()).lower()
    assert "zzqq" not in blob and "flooble" not in blob, \
        "raw utterance leaked into the fallback result"


# ====================== Item 4: keyword honesty + negation ===================

def test_keyword_confidence_is_calibrated_not_one():
    clf = IntentClassifier()
    intent, conf = clf._keyword_match("mute")          # exact rule
    assert intent == "Cmd.VolumeMute"
    assert conf == KEYWORD_CONFIDENCE["exact"] < 1.0


def test_keyword_negation_suppresses_contains_hit():
    """'don't translate' must not fire the translate `contains` rule."""
    clf = IntentClassifier()
    intent, _ = clf._keyword_match("i don't want to translate anything right now")
    assert intent != "Cmd.TranslationStart"


def test_keyword_positive_contains_still_fires():
    # The translate rule is a guarded regex (it carries a `not_regex` of
    # help-question markers), so a positive hit is priced at the
    # `regex_guarded` tier — not `contains`, which is what it used to be.
    clf = IntentClassifier()
    intent, conf = clf._keyword_match("please translate this menu")
    assert intent == "Cmd.TranslationStart"
    assert conf == KEYWORD_CONFIDENCE["regex_guarded"]


# ======================= Item 3: time-based expiry ==========================

def test_context_expires_after_ttl():
    clock = FakeClock()
    store = SessionStore(clock=clock)
    sess = store.get("s1")
    sess.set_context("confirm_ctx", lifespan=5, now=clock(), ttl_seconds=90.0)
    assert sess.has_context("confirm_ctx")
    clock.advance(120)                 # past the 90s TTL
    sess.expire_contexts(clock())
    assert not sess.has_context("confirm_ctx"), "stale context survived its TTL"


def test_context_survives_within_ttl():
    clock = FakeClock()
    store = SessionStore(clock=clock)
    sess = store.get("s2")
    sess.set_context("confirm_ctx", lifespan=5, now=clock(), ttl_seconds=90.0)
    clock.advance(30)
    sess.expire_contexts(clock())
    assert sess.has_context("confirm_ctx")


def test_idle_session_is_reset():
    """An idle session past the session TTL comes back fresh (no stale pending state)."""
    clock = FakeClock()
    store = SessionStore(clock=clock, session_ttl_seconds=600.0)
    sess = store.get("s3")
    sess.pending_intent = "reminders.add"
    sess.pending_slots = {"name": "old topic"}
    clock.advance(601)                 # idle longer than the session TTL
    fresh = store.get("s3")
    assert fresh.pending_intent is None
    assert fresh.pending_slots == {}


def test_stale_confirmation_does_not_fire_on_later_yes():
    """
    End-to-end: a confirmation left open should not fire its action when an
    unrelated 'yes' arrives after the context TTL has elapsed.
    """
    clock = FakeClock()
    engine = NLUEngine()
    engine.sessions = SessionStore(clock=clock)   # inject controllable clock
    sid = _sid()

    # Drive a turn that opens a confirmation context, if the schema has one.
    # Use the public API: find any intent with a followup.
    followup_intent = next((name for name, cfg in engine.intents.items()
                            if cfg.get("followup")), None)
    if followup_intent is None:
        return  # no confirmation flow in this schema — nothing to assert

    sess = engine.sessions.get(sid)
    fu = engine.intents[followup_intent]["followup"]
    sess.set_context(fu["context"], lifespan=5, now=clock(), ttl_seconds=90.0)

    clock.advance(120)                 # let the confirmation go stale
    r = engine.handle(sid, "yes")
    # The stale confirmation must NOT have produced its yes-branch fulfilment.
    assert not (r.type == "FULFILL" and r.intent == followup_intent), \
        "stale confirmation fired on a later unrelated 'yes'"


# ===================== Item 5: slot-fill attempt limit ======================

def test_slot_fill_abandons_after_max_attempts():
    """
    A user who cannot supply a parseable slot value is routed out of the flow
    instead of being prompted forever.
    """
    engine = NLUEngine()
    sid = _sid()
    engine.reset(sid)

    # Start a slot-filling intent, then feed un-parseable answers.
    engine.handle(sid, "set a reminder")
    last = None
    for _ in range(engine.MAX_SLOT_ATTEMPTS + 2):
        last = engine.handle(sid, "qzxw nonsense blarg")
    # After exhausting attempts the engine must abandon (not keep PROMPTing).
    sess = engine.sessions.get(sid)
    assert last.type == "FALLBACK" or sess.pending_intent is None, \
        "engine kept the user trapped in slot filling past the attempt budget"


# ------------------------------- runner -------------------------------------
def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ✅ {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {fn.__name__}: {e}")
            failed += 1
        except Exception as e:  # surface unexpected errors, don't hide them
            print(f"  💥 {fn.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
