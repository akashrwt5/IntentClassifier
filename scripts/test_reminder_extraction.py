#!/usr/bin/env python3
"""
Test script to verify reminder intent and slot/title extraction.

Usage:
    # Run built-in sample test cases:
    python scripts/test_reminder_extraction.py

    # Test a specific phrase:
    python scripts/test_reminder_extraction.py "set reminder to take medicine tomorrow 5 pm"

    # Interactive prompt mode:
    python scripts/test_reminder_extraction.py --interactive
"""

import sys
from pathlib import Path

# Add packages/runtime to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "runtime"))

from nlu_engine import NLUEngine
from nlu_langpack import load_pack


def get_engine():
    pack_path = PROJECT_ROOT / "dist" / "nlu_pack"
    if not pack_path.exists():
        raise FileNotFoundError(f"Language pack not found at {pack_path}")
    pack = load_pack(str(pack_path))
    return NLUEngine(pack=pack)


def test_phrase(engine, phrase: str, session_id: str = "test_user"):
    res = engine.handle(session_id, phrase)

    print("\n" + "=" * 60)
    print(f'📥 Input Phrase  : "{phrase}"')
    print("-" * 60)
    print(f"🎯 Intent        : {res.intent}")
    print(f"⚡ Action        : {res.action}")
    print(f"📊 Status / Type : {res.type}")
    print(f"📈 Confidence    : {res.confidence:.2f}")

    title = res.parameters.get("name")
    dt = res.parameters.get("date-time")
    recurrence = res.parameters.get("recurrence")

    print(f"🏷️  Reminder Title: {title or '(None)'}")
    print(f"⏰ Date / Time   : {dt or '(None)'}")
    if recurrence:
        print(f"🔄 Recurrence    : {recurrence}")

    if res.message:
        print(f"💬 Engine Prompt : {res.message}")

    print("=" * 60)
    return res


def main():
    try:
        engine = get_engine()
    except Exception as e:
        print(f"❌ Error loading engine: {e}")
        sys.exit(1)

    args = sys.argv[1:]

    if "--interactive" in args or "-i" in args:
        print("=== Interactive Reminder Tester ===")
        print("  Type 'exit' to quit, 'reset' to clear session context.\n")
        session_id = "interactive_session"
        while True:
            try:
                user_input = input("\nEnter reminder phrase: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not user_input or user_input.lower() in ("exit", "quit"):
                break
            if user_input.lower() == "reset":
                engine.reset(session_id)
                print("🔄 Session reset.")
                continue
            test_phrase(engine, user_input, session_id=session_id)
        return

    if args:
        # User passed a custom phrase as arguments
        phrase = " ".join(args)
        test_phrase(engine, phrase)
        return

    # Default: Run test suite with various reminder phrases
    sample_phrases = [
        "set reminder to take medicine tomorrow 5 pm",
        "remind me to call Mom at 8pm",
        "set a reminder for doctor appointment next Monday at 10 am",
        "remind me to drink water",  # Missing time (slot prompt)
        "create a reminder to submit project report by Friday 6 pm",
    ]

    print("Running sample reminder test cases...")
    for idx, phrase in enumerate(sample_phrases):
        test_phrase(engine, phrase, session_id=f"test_session_{idx}")


if __name__ == "__main__":
    main()
