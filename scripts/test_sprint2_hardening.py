#!/usr/bin/env python3
"""
Sprint-2 production-hardening regression tests.

Pins the Sprint-2 fixes so they cannot silently regress:

  * Item 7  — learned out-of-scope class rejects general-knowledge queries
  * Item 11 — semantic threshold sourced from the schema (single source)
  * Item 12 — semantic stage warm-up does not crash construction
  * Item 13 — "yes, no worries" idioms parse as agreement, not refusal

Run:
    python scripts/test_sprint2_hardening.py
    pytest scripts/test_sprint2_hardening.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nlu import NLUEngine  # noqa: E402

engine = NLUEngine()
_counter = {"n": 0}


def _sid():
    _counter["n"] += 1
    return f"sprint2-{_counter['n']}"


# ===================== Item 11: threshold from the schema ====================

def test_semantic_threshold_comes_from_schema():
    expected = engine.schema.get("semantic_threshold")
    assert expected is not None, "schema is missing semantic_threshold"
    assert engine.semantic_threshold == expected
    # And it is the value the semantic stage was actually built with.
    if engine.semantic is not None:
        assert engine.semantic.threshold == expected


# ===================== Item 12: warm-up didn't break load ====================

def test_semantic_stage_loaded():
    # If artifacts are present the stage must be live (warm-up must not have
    # thrown and disabled it).
    if (Path(__file__).parent.parent / "models" / "semantic_head.npz").exists():
        assert engine.semantic is not None


# ===================== Item 13: idiom polarity ==============================

def test_yes_no_worries_is_agreement():
    assert engine._yes_no("yes, no worries") is True
    assert engine._yes_no("no problem, go ahead") is True
    assert engine._yes_no("sure, no doubt") is True


def test_genuine_negatives_still_negative():
    assert engine._yes_no("no") is False
    assert engine._yes_no("no thanks") is False
    assert engine._yes_no("cancel that") is False


def test_plain_affirmatives_still_affirmative():
    assert engine._yes_no("yes") is True
    assert engine._yes_no("go ahead") is True


# ===================== Item 7: out-of-scope rejection =======================

def test_out_of_scope_does_not_fire_a_command():
    """
    General-knowledge / chit-chat must NOT execute an in-scope action. They may
    route to GenAI fallback; what they must never do is FULFILL a real command.
    """
    oos = [
        "sing me a song",
        "what's the weather like today",
        "who painted the mona lisa",
        "what is twelve times seven",
    ]
    for utt in oos:
        r = engine.handle(_sid(), utt)
        assert not (r.type == "FULFILL" and r.intent not in (None, "GENAI")), \
            f"OOS utterance fired a command: {utt!r} -> {r.intent}"


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
