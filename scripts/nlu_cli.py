#!/usr/bin/env python3
"""
Interactive multi-turn NLU demo — the on-device Dialogflow replacement.

Usage:
    python scripts/nlu_cli.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nlu import NLUEngine  # noqa: E402

SESSION = "cli-user"


def render(r):
    if r.interrupted_intent:
        print(f"  ⚠️  Interrupted: {r.interrupted_intent} flow cancelled")
    via = "🧠 semantic" if r.semantic_rescue else "⚡ tf-idf"
    if r.type == "FULFILL":
        params = f"  {r.parameters}" if r.parameters else ""
        print(f"  ✅ {r.intent}  →  action={r.action}{params}")
        print(f"     [{via}, confidence {r.confidence:.2f}]")
        if r.message:
            print(f"  💬 {r.message}")
    elif r.type == "PROMPT":
        print(f"  ❓ {r.message}")
        print(f"     [{via}]")
        if r.parameters:
            print(f"     (collected so far: {r.parameters})")
    elif r.type == "CONFIRM":
        print(f"  ❓ {r.message}  [yes/no]")
    elif r.type == "FALLBACK":
        print(f"  🤖 GenAI fallback  (confidence {r.confidence:.2f})")
        print(f"  🔗 {r.url}")


def main():
    print("=== On-device NLU (Dialogflow replacement) ===")
    print("    Type 'exit' to quit, 'reset' to clear the conversation.\n")
    engine = NLUEngine()
    while True:
        try:
            text = input("you ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.lower() == "exit":
            break
        if text.lower() == "reset":
            engine.reset(SESSION)
            print("  (conversation reset)\n")
            continue
        render(engine.handle(SESSION, text))
        print()


if __name__ == "__main__":
    main()
