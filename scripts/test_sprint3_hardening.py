#!/usr/bin/env python3
"""
Sprint-3 hardening regression tests.

Pins the Sprint-3 fixes:

  * A1 — end-to-end holdout gate exists and is enforceable (real, not a no-op)
  * C1 — a bare `contains` keyword hit cannot interrupt an active slot flow,
         but a strong signal still can
  * C2 — fuzzy enum matching is length-gated (short-word collisions rejected)
         and disabled for one-shot scans; genuine typos still match
  * C3 — the legacy 1-NN semantic index path is gone; the head still classifies

Run:
    python scripts/test_sprint3_hardening.py
    pytest scripts/test_sprint3_hardening.py
"""

import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
from nlu import NLUEngine                       # noqa: E402
from nlu.entities import EntityExtractor        # noqa: E402
from nlu.semantic import SemanticFallback       # noqa: E402

engine = NLUEngine()
entities = EntityExtractor()
_counter = {"n": 0}


def _sid():
    _counter["n"] += 1
    return f"sprint3-{_counter['n']}"


# ============================ C1: interrupt demotion =========================

def test_bare_contains_does_not_interrupt_slot_flow():
    sid = _sid()
    engine.reset(sid)
    engine.handle(sid, "set a reminder")
    r = engine.handle(sid, "ask about the translate feature")
    # Incidental "translate" must NOT abandon the reminder flow.
    assert r.interrupted_intent is None, r.interrupted_intent
    assert r.intent == "reminders.add", r.intent


def test_strong_signal_still_interrupts():
    sid = _sid()
    engine.reset(sid)
    engine.handle(sid, "set a reminder")
    r = engine.handle(sid, "mute")
    # An exact keyword command is a genuine topic switch and should interrupt.
    assert r.interrupted_intent == "reminders.add", r.interrupted_intent
    assert r.intent == "Cmd.VolumeMute", r.intent


# ============================ C2: fuzzy hardening ===========================

def test_short_word_collisions_not_fuzzy_matched():
    # Common words close to short memory names must NOT resolve to a memory.
    for w in ("care", "cab", "pup", "gum", "cute"):
        val, _, _ = entities.extract_enum("memory", w, fuzzy=True)
        assert val is None, f"{w!r} wrongly fuzzy-matched to {val}"


def test_genuine_typo_still_fuzzy_matches():
    val, _, _ = entities.extract_enum("memory", "restraunt", fuzzy=True)
    assert val == "Restaurant", val


def test_fuzzy_disabled_for_one_shot_scan():
    val, _, _ = entities.extract_enum("memory", "restraunt", fuzzy=False)
    assert val is None, val


# ============================ C3: 1-NN path removed =========================

def test_semantic_has_no_legacy_index():
    s = SemanticFallback()
    assert not hasattr(s, "_embeddings")
    intent, conf = s.classify("i cannot hear you")
    assert isinstance(intent, str) and 0.0 <= conf <= 1.0


def test_legacy_index_file_absent():
    assert not (BASE.parent / "models" / "semantic_index.npz").exists(), \
        "legacy semantic_index.npz should have been removed"


# ============================ A1: holdout gate ==============================

def test_holdout_gate_passes_at_default_floor():
    # The authoritative end-to-end gate must pass on the current system.
    r = subprocess.run(
        [sys.executable, str(BASE / "test_holdout.py"), "--strict"],
        capture_output=True, text=True)
    assert r.returncode == 0, f"holdout --strict failed:\n{r.stdout[-500:]}"


def test_holdout_gate_fails_when_floor_impossible():
    # Proves the gate is real, not a no-op.
    import os
    env = dict(os.environ, MIN_HOLDOUT_TOTAL="99")
    r = subprocess.run(
        [sys.executable, str(BASE / "test_holdout.py"), "--strict"],
        capture_output=True, text=True, env=env)
    assert r.returncode == 1, "gate should fail at an impossible floor"


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
        except Exception as e:
            print(f"  💥 {fn.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
