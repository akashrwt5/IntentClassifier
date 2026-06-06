#!/usr/bin/env python3
"""
End-to-end tests for the on-device NLU engine.

Run:
    python scripts/test_nlu.py
    pytest scripts/test_nlu.py
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
    sid = _sid()
    engine.reset(sid)
    result = None
    for t in turns:
        result = engine.handle(sid, t)
    return result


def run_all(turns):
    sid = _sid()
    engine.reset(sid)
    return [engine.handle(sid, t) for t in turns]


# ----------------------------- slot filling ---------------------------------
def test_reminder_step_by_step():
    rs = run_all(["set a reminder", "take medication", "at 5 pm"])
    assert rs[0].type == "PROMPT"
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


def test_reminder_oneshot_freeform_topic():
    r = run(["do not let me forget to water flowers at 7 a.m. tomorrow"])
    assert r.type == "FULFILL", f"got {r.type}"
    assert r.parameters["name"] == "water flowers", r.parameters
    assert r.parameters["date-time"].endswith("07:00")


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
    assert rs[1].action == "memory.change"


def test_memory_one_shot():
    r = run(["switch to car memory"])
    assert r.type == "FULFILL" and r.parameters["MemoryName"] == "Car"


def test_memory_fuzzy_asr_error():
    r = run(["change memory", "restraunt"])
    assert r.type == "FULFILL" and r.parameters["MemoryName"] == "Restaurant"


# --------------------------- send message -----------------------------------
def test_send_message_yes():
    r = run(["yes send the message"])
    assert r.type == "FULFILL" and r.action == "message.send"


def test_send_message_no():
    r = run(["no don't send"])
    assert r.type == "FULFILL" and r.action == "message.cancel"


# ----------------------- simple fire-and-forget -----------------------------
def test_simple_intents():
    cases = {
        "increase the volume":    ("Cmd.VolumeIncrease",  "volume.increase"),
        "turn it down":           ("Cmd.VolumeDecrease",  "volume.decrease"),
        "mute":                   ("Cmd.VolumeMute",      "volume.mute"),
        "unmute":                 ("Cmd.VolumeUnmute",    "volume.unmute"),
        "open translate":         ("Cmd.TranslationStart","translate.open"),
        "check my battery":       ("Cmd.BatteryLevel",    "battery.level"),
        "find my phone":          ("Cmd.FindMyPhone",     "phone.find"),
    }
    for text, (intent, action) in cases.items():
        r = run([text])
        assert r.type == "FULFILL", f"{text} → {r.type}"
        assert r.intent == intent, f"{text} → {r.intent} (want {intent})"
        assert r.action == action


def test_activity_intents():
    cases = {
        "start a run":   ("Cmd.ActivityRun",  "activity.run"),
        "go for a walk": ("Cmd.ActivityWalk", "activity.walk"),
    }
    for text, (intent, action) in cases.items():
        r = run([text])
        assert r.type == "FULFILL", f"{text} → {r.type}"
        assert r.intent == intent, f"{text} → {r.intent}"


def test_help_intents():
    cases = {
        "how do i pair my hearing aids": "Help_Pairing",
        "help with tinnitus":            "Help_Tinnitus",
        "how does fall alert work":      "Help_FallAlert",
    }
    for text, intent in cases.items():
        r = run([text])
        assert r.type == "FULFILL", f"{text} → {r.type}"
        assert r.intent == intent, f"{text} → {r.intent} (want {intent})"


# ------------------------------ fallback ------------------------------------
def test_out_of_scope_fallback():
    r = run(["how is the weather today"])
    assert r.type == "FALLBACK" and r.intent == "GENAI"


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
