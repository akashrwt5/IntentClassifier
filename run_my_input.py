import sys
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "runtime"))

from nlu_engine import NLUEngine
from nlu_langpack import load_pack


def process_phrase(engine, phrase: str, session_id: str = "editor_user"):
    result = engine.handle(session_id, phrase)

    print("\n" + "=" * 60)
    print(f'📥 Input Phrase  : "{phrase}"')
    print("-" * 60)
    print(f"🎯 Intent        : {result.intent}")
    print(f"🏷️  Reminder Title: {result.parameters.get('name') or '(None)'}")
    print(f"⏰ Date / Time   : {result.parameters.get('date-time') or '(None)'}")
    print(f"⚡ Action        : {result.action}")
    print(f"📊 Status / Type : {result.type}")
    if result.message:
        print(f"💬 Message       : {result.message}")
    print("=" * 60)
    return result


def main():
    print("⏳ Loading NLU Engine with BIO Slot Tagger...")
    pack = load_pack(str(PROJECT_ROOT / "dist" / "nlu_pack"))
    engine = NLUEngine(pack=pack)
    print("✅ Loaded successfully!\n")

    print("=" * 60)
    print("🎯 NLU Reminder Tester (Interactive Mode)")
    print("   Console me apna sentence likhein aur Enter dabayein.")
    print("   Band karne ke liye 'exit' likhein.")
    print("=" * 60)

    turn = 1
    while True:
        try:
            phrase = input(f"\n👉 [{turn}] Enter phrase: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not phrase:
            continue
        if phrase.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        process_phrase(engine, phrase, session_id=f"turn_{turn}")
        turn += 1


if __name__ == "__main__":
    main()
