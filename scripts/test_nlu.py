#!/usr/bin/env python3
"""
End-to-end tests for the on-device NLU engine.

Run:
    python scripts/test_nlu.py
    pytest scripts/test_nlu.py
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nlu import NLUEngine  # noqa: E402


def _local_hm(iso: str):
    """Convert a stored UTC ISO timestamp back to local wall-clock (h, m).

    Reminders are stored in UTC; a spoken local time round-trips back to the
    same local hour regardless of the test runner's timezone.
    """
    return (lambda d: (d.hour, d.minute))(datetime.fromisoformat(iso).astimezone())

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
    assert _local_hm(rs[2].parameters["date-time"]) == (17, 0)
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
    assert _local_hm(r.parameters["date-time"]) == (7, 0)


def test_reminder_freeform_topic():
    r = run(["set a reminder", "call my dentist", "tomorrow at 9 am"])
    assert r.type == "FULFILL"
    assert r.parameters["name"] == "call my dentist"


def test_reminder_topic_strips_leading_connector():
    # "...for dinner" must not keep the dangling leading "for".
    r = run(["remind me at 9 pm for dinner"])
    assert r.type == "FULFILL"
    assert r.parameters["name"] == "dinner", r.parameters


def test_reminder_topic_strips_orphaned_at_number():
    # "at 5" must be removed whole — no orphaned "5" left in the topic.
    r = run(["remind me to call mom at 5"])
    assert r.type == "FULFILL"
    assert r.parameters["name"] == "call mom", r.parameters


def test_reminder_topic_strips_in_the_morning_phrase():
    r = run(["remind me to water the plants in the morning"])
    assert r.type == "FULFILL"
    assert r.parameters["name"] == "water the plants", r.parameters


def test_reminder_topic_keeps_midsentence_for():
    # The leading-connector strip must NOT eat a legitimate mid-sentence "for".
    r = run(["remind me to buy a gift for mom at 3pm"])
    assert r.type == "FULFILL"
    assert r.parameters["name"] == "buy a gift for mom", r.parameters


def test_reminder_recurrence():
    r = run(["remind me to clean hearing aids every morning"])
    assert r.parameters.get("recurrence") == "Daily"


def test_reminder_day_only_prompts_for_time():
    # A day with no time must ask for the time instead of defaulting to 9am.
    rs = run_all(["i need to go to airport tomorrow", "at 6 am"])
    assert rs[0].type == "PROMPT" and "when" in rs[0].message.lower()
    assert rs[1].type == "FULFILL"
    assert rs[1].parameters["name"] == "go to airport"
    # The parked day ("tomorrow") must be preserved by the bare-time answer,
    # and 6am must NOT roll forward a day.
    from datetime import datetime, timedelta
    answered = datetime.fromisoformat(rs[1].parameters["date-time"]).astimezone()
    tomorrow = (datetime.now().astimezone() + timedelta(days=1)).date()
    assert answered.date() == tomorrow, (answered, tomorrow)
    assert (answered.hour, answered.minute) == (6, 0)


def test_reminder_day_only_step_by_step_keeps_day():
    rs = run_all(["set a reminder", "buy milk", "tomorrow", "9 am"])
    assert rs[2].type == "PROMPT"      # "tomorrow" alone still needs a time
    assert rs[3].type == "FULFILL"
    assert _local_hm(rs[3].parameters["date-time"]) == (9, 0)
    from datetime import datetime, timedelta
    when = datetime.fromisoformat(rs[3].parameters["date-time"]).astimezone()
    assert when.date() == (datetime.now().astimezone() + timedelta(days=1)).date()


def test_reminder_explicit_time_still_one_shot():
    # Day + explicit time must fulfill directly, no extra prompt.
    r = run(["remind me to call mom tomorrow at 3pm"])
    assert r.type == "FULFILL"
    assert _local_hm(r.parameters["date-time"]) == (15, 0)


def test_reminder_time_answer_repeating_day_does_not_advance():
    # Regression: after "...tomorrow" prompts for time, answering "tomorrow at 4"
    # must NOT advance to the day after — the answer's own day wins, not the
    # parked day. (Previously anchored "tomorrow" onto the parked tomorrow.)
    from datetime import datetime, timedelta
    rs = run_all(["i need to go to airport tomorrow", "tomorrow at 4"])
    assert rs[0].type == "PROMPT"
    assert rs[1].type == "FULFILL"
    when = datetime.fromisoformat(rs[1].parameters["date-time"]).astimezone()
    assert when.date() == (datetime.now().astimezone() + timedelta(days=1)).date(), when
    assert when.hour == 16  # "4" with no am/pm resolves to the afternoon


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
    rs = run_all(["send a message", "yes send it"])
    assert rs[0].type == "FULFILL" and rs[0].action == "message.compose"
    assert rs[1].type == "FULFILL" and rs[1].action == "message.send"


def test_send_message_no():
    rs = run_all(["send a message", "no don't send"])
    assert rs[0].type == "FULFILL" and rs[0].action == "message.compose"
    assert rs[1].type == "FULFILL" and rs[1].action == "message.cancel"


# ----------------------- intent interruption --------------------------------
def test_interrupt_reminder_with_volume():
    """Saying 'mute' mid-reminder flow should abandon reminder and mute."""
    rs = run_all(["set a reminder", "mute"])
    assert rs[0].type == "PROMPT"                          # reminder started
    assert rs[1].type == "FULFILL"                         # mute fired
    assert rs[1].intent == "Cmd.VolumeMute"
    assert rs[1].interrupted_intent == "reminders.add"    # abandoned flow reported


def test_interrupt_memory_with_volume():
    """Saying 'turn it down' mid-memory flow should abandon memory and decrease volume."""
    rs = run_all(["change memory", "turn it down"])
    assert rs[0].type == "PROMPT"
    assert rs[1].type == "FULFILL"
    assert rs[1].intent == "Cmd.VolumeDecrease"
    assert rs[1].interrupted_intent == "Cmd.MemoryChange"


def test_no_interrupt_on_slot_answer():
    """A slot answer that happens to match another intent weakly should NOT interrupt."""
    rs = run_all(["set a reminder", "take medication", "at 5 pm"])
    # "take medication" could weakly resemble reminders.add but confidence
    # should be below INTERRUPT_THRESHOLD — slot filling must continue
    assert rs[1].type == "PROMPT"   # still collecting date-time, not interrupted
    assert rs[2].type == "FULFILL"


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


# ----------------------- context parameter memory ---------------------------
def test_memory_change_back():
    """Change memory → do something else → change back resolves previous memory."""
    rs = run_all([
        "change memory to restaurant",   # last=Restaurant, prev=None
        "remind me to take medication tomorrow at 9 am",  # unrelated turn
        "change back",                   # should resolve to None (no prev yet)...
    ])
    # After turn 1: last_memory=Restaurant, prev_memory=None
    assert rs[0].type == "FULFILL" and rs[0].parameters["MemoryName"] == "Restaurant"
    assert rs[1].type == "FULFILL"
    # "change back" with no prev_memory falls through to slot prompt
    assert rs[2].type in ("PROMPT", "FULFILL")


def test_memory_change_back_full_flow():
    """Three memory changes — third resolves to first via prev_memory."""
    rs = run_all([
        "change memory to restaurant",   # last=Restaurant, prev=None
        "change memory to personal",     # last=Personal,    prev=Restaurant
        "change back",                   # should go back to Restaurant
    ])
    assert rs[0].parameters["MemoryName"] == "Restaurant"
    assert rs[1].parameters["MemoryName"] == "Personal"
    assert rs[2].type == "FULFILL"
    assert rs[2].parameters["MemoryName"] == "Restaurant"


def test_reminder_again():
    """'Remind me again' reuses the last reminder's name and time."""
    rs = run_all([
        "remind me to take medication tomorrow at 9 am",
        "remind me again",
    ])
    assert rs[0].type == "FULFILL"
    assert rs[1].type == "FULFILL"
    assert rs[1].parameters.get("name") == rs[0].parameters.get("name")


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
