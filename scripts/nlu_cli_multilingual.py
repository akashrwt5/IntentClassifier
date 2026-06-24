#!/usr/bin/env python3
"""
Interactive multi-turn NLU demo with language/model selection.

This is identical to nlu_cli.py except you can select which TF-IDF model to use.

Usage:
    python scripts/nlu_cli_multilingual.py                  # production model (default)
    python scripts/nlu_cli_multilingual.py --model en       # English multilingual model
    python scripts/nlu_cli_multilingual.py --model fr       # French multilingual model
    python scripts/nlu_cli_multilingual.py --model de       # German multilingual model
    python scripts/nlu_cli_multilingual.py --model da       # Danish multilingual model
    python scripts/nlu_cli_multilingual.py --model multilingual  # combined multilingual model

All models use the SAME NLU business logic (slot-filling, reminders, multi-turn context).
The --model flag only selects which TF-IDF classifier handles intent prediction.
"""

import argparse
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nlu import NLUEngine  # noqa: E402

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
        url = engine.genai_url + urllib.parse.quote(text)
        print(f"  🔗 {url}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive multi-turn NLU with language selection"
    )
    parser.add_argument(
        "--model", "-m",
        choices=["production", "en", "fr", "de", "da", "multilingual"],
        default="production",
        help="Which TF-IDF model to use for intent classification. "
             "production=scripts/models/intent_model.onnx (default), "
             "or any multilingual model (en/fr/de/da/multilingual)."
    )
    args = parser.parse_args()

    try:
        engine = NLUEngine(model_name=args.model)
    except FileNotFoundError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)

    model_label = "production" if args.model == "production" else args.model
    print("=== On-device NLU (Dialogflow replacement) ===")
    print(f"    Model: {model_label}")
    print("    Type 'exit' to quit, 'reset' to clear the conversation.\n")

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
