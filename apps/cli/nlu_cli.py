#!/usr/bin/env python3
"""
Interactive multi-turn NLU demo — the on-device Dialogflow replacement.

Usage:
    python apps/cli/nlu_cli.py
"""

import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages" / "runtime"))
from nlu_engine import NLUEngine  # noqa: E402

SESSION = "cli-user"


def render(r, engine, text):
    if r.interrupted_intent:
        print(f"  ⚠️  Interrupted: {r.interrupted_intent} flow cancelled")
    if r.semantic_rescue:
        via = f"🧠 semantic {r.confidence:.2f}  |  tf-idf said: {r.tfidf_intent} ({r.tfidf_confidence:.2f})"
    else:
        via = f"⚡ tf-idf {r.confidence:.2f}"
    if r.type == "FULFILL":
        params = f"  {r.parameters}" if r.parameters else ""
        print(f"  ✅ {r.intent}  →  action={r.action}{params}")
        print(f"     [{via}]")
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
        if r.message:
            print(f"  💬 {r.message}")
        # The app layer (here, the CLI) builds the GenAI URL from the text it
        # already holds — the raw utterance is never returned in the result.
        if engine.genai_url:
            url = engine.genai_url + urllib.parse.quote(text)
            print(f"  🔗 {url}")
        else:
            print("  🔗 (no GenAI endpoint configured — set NLU_GENAI_URL)")


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
        render(engine.handle(SESSION, text), engine, text)
        print()


if __name__ == "__main__":
    main()
