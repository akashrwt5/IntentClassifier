import sys
from pathlib import Path
import argparse

# Add the necessary paths to sys.path
# This assumes the script is run from the IntentClassifier root
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'packages' / 'runtime'))

from nlu_engine.engine import NLUEngine

def interactive_test():
    parser = argparse.ArgumentParser(description="Interactive NLU Engine Test")
    parser.add_argument("--lang", "-l", default="en",
                        help="Language to test (default: en)")
    args = parser.parse_args()

    print(f"Initializing NLU Engine for language: {args.lang}...")
    engine = NLUEngine(language=args.lang)
    print("NLU Engine ready. Type your utterances below. Type 'exit' to quit.")

    session_id = "interactive_session"
    engine.reset(session_id) # Reset session for a clean start

    while True:
        try:
            utterance = input("You: ")
            if utterance.lower() == 'exit':
                break

            result = engine.handle(session_id, utterance)
            print(f"NLU Result: {result.to_dict()}")
            print("-" * 30)

        except Exception as e:
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    interactive_test()
