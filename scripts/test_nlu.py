#!/usr/bin/env python3
"""
End-to-end tests for the on-device NLU engine.

Run:
    python scripts/test_nlu.py        # plain runner, no pytest needed
    pytest scripts/test_nlu.py        # also works under pytest
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nlu import NLUEngine  # noqa: E402

engine = NLUEngine()
_counter = {"n": 0}


def _sid():
    _counter["n"] += 1
    return f"test-{_counter['n']}"


def run(turns):
    """Run a fresh conversation, return the final NLUResult."""
    sid = _sid()
    engine.reset(sid)
    result = None
    for t in turns:
        result = engine.handle(sid, t)
    return result


def run_all(turns):
    """Run a conversation, return the list of every NLUResult."""
    sid = _sid()
    engine.reset(sid)
    return [engine.handle(sid, t) for t in turns]


# ----------------------------- slot filling ---------------------------------
def test_reminder_step_by_step():
    rs = run_all(["set a reminder", "take medication", "at 5 pm"])
    assert rs[0].type == "PROMPT" and "remind" in rs[0].message.lower()
    assert rs[1].type == "PROMPT" and "when" in rs[1].message.lower()
    assert rs[2].type == "FULFILL"
    assert rs[2].parameters["name"] == "Take Medication"
    assert rs[2].parameters["date-time"].endswith("17:00")
    assert rs[2].action == "reminders.add"


def test_reminder_one_shot():
    r = run(["remind me to drink water in 10 minutes"])
    assert r.type == "FULFILL"
    assert r.parameters["name"] == "Drink Water"
    assert "date-time" in r.parameters


def test_reminder_freeform_topic():
    r = run(["set a reminder", "call my dentist", "tomorrow at 9 am"])
    assert r.type == "FULFILL"
    assert r.parameters["name"] == "call my dentist"


def test_reminder_recurrence():
    r = run(["remind me to clean hearing aids every morning"])
    assert r.parameters.get("recurrence") == "Daily"


# ------------------------------- memory -------------------------------------
def test_memory_step_by_step():
    rs = run_all(["change memory", "restaurant"])
    assert rs[0].type == "PROMPT"
    assert rs[1].type == "FULFILL"
    assert rs[1].parameters["MemoryName"] == "Restaurant"


def test_memory_one_shot():
    r = run(["switch to car memory"])
    assert r.type == "FULFILL" and r.parameters["MemoryName"] == "Car"


def test_memory_fuzzy_asr_error():
    r = run(["change memory", "restraunt"])
    assert r.type == "FULFILL" and r.parameters["MemoryName"] == "Restaurant"


# --------------------------- confirmation flow ------------------------------
def test_send_message_yes():
    rs = run_all(["send a message", "yes"])
    assert rs[0].type == "CONFIRM"
    assert rs[1].type == "FULFILL" and rs[1].action == "message.send"


def test_send_message_no():
    rs = run_all(["send a message", "no"])
    assert rs[1].action == "message.cancel"


def test_confirmation_reprompt():
    rs = run_all(["send a message", "hmm not sure", "yes"])
    assert rs[1].type == "CONFIRM"          # didn't understand → re-prompt
    assert rs[2].action == "message.send"


# ----------------------- simple fire-and-forget -----------------------------
def test_simple_intents():
    cases = {
        "increase the volume": ("VOLUME_INCREASE", "volume.increase"),
        "turn it down": ("VOLUME_DECREASE", "volume.decrease"),
        "mute": ("VOLUME_MUTE", "volume.mute"),
        "unmute": ("VOLUME_UNMUTE", "volume.unmute"),
        "open translate": ("TRANSLATE", "translate.open"),
        "run selfcheck": ("SELFCHECK", "selfcheck.run"),
        "check my battery": ("BATTERY", "battery.level"),
        "find my phone": ("FIND_MY_PHONE", "phone.find"),
    }
    for text, (intent, action) in cases.items():
        r = run([text])
        assert r.type == "FULFILL", f"{text} → {r.type}"
        assert r.intent == intent, f"{text} → {r.intent} (want {intent})"
        assert r.action == action


# ------------------------------ fallback ------------------------------------
def test_out_of_scope_fallback():
    r = run(["how is the weather today"])
    assert r.type == "FALLBACK" and r.intent == "GENAI"
    assert r.url and "genai" in r.url


def test_gibberish_fallback():
    r = run(["asdfghjkl qwerty"])
    assert r.type == "FALLBACK"


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
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
