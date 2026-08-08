"""Unit tests for the wrong-action budget SCOPE (no model/data needed).

The medical wrong-action budget governs SAFETY events — firing a wrong
state-changing command. Read-only queries (which only read a value back to the
user) are an accuracy concern, not a safety one, and must NOT be charged to the
budget. These tests pin that boundary against the exact intents that showed up
in the English failure sample, so the scope can never silently drift.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "buildtime"))

from nlu_training.wrong_action_harness import (  # noqa: E402
    is_actionable,
    is_read_only,
    is_state_changing,
)


class TestReadOnlyIsNotCharged:
    """Read-only queries are actionable-looking but change no state."""

    READ_ONLY = [
        "Cmd.ActivityCalories",  # "how many calories did i burn jogging"
        "Cmd.ActivityRun",
        "Cmd.ActivityWalk",
        "Cmd.BatteryLevel",  # "help with battery questions" mis-fires here
    ]

    STATE_CHANGING = [
        "Cmd.VolumeDecrease",  # "lower how loud it is"
        "Cmd.VolumeIncrease",
        "Cmd.VolumeMute",  # "hush it up"
        "Cmd.StreamingStart",  # "stream from an accessory mic ..."
        "Cmd.SendMessage",
        "Cmd.TranslationStart",
        "reminders.complete",
    ]

    def test_read_only_intents_are_read_only(self):
        for label in self.READ_ONLY:
            assert is_read_only(label), f"{label} should be read-only"

    def test_read_only_intents_are_not_state_changing(self):
        # This is the fix: a wrong read-only query is NOT a budget-charged action.
        for label in self.READ_ONLY:
            assert not is_state_changing(label), f"{label} must not be budget-charged"

    def test_state_changing_intents_are_charged(self):
        for label in self.STATE_CHANGING:
            assert is_state_changing(label), f"{label} is a safety event"
            assert not is_read_only(label)

    def test_help_and_sys_never_charged(self):
        for label in ("Help_Battery", "Default Fallback Intent"):
            assert not is_state_changing(label)
            assert not is_actionable(label)


def test_query_suffix_convention_holds():
    # Any *.query intent is read-only by the domain.object.action taxonomy.
    assert is_read_only("Cmd.ActivityStand")
    assert not is_state_changing("Cmd.ActivityStand")
