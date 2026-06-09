#!/usr/bin/env python3
"""
Phase 3 threshold tuning — semantic fallback test suite.

Tests three categories:
  1. Novel phrasings that TF-IDF currently fails (should be rescued by semantic)
  2. Out-of-scope queries (should still fall to GenAI — NOT rescued)
  3. Known good phrasings (should still work via TF-IDF, semantic_rescue=False)

Run from the repo root:
    python scripts/test_semantic.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.nlu.engine import NLUEngine

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


# ── Test cases ────────────────────────────────────────────────────────────────

NOVEL_PHRASINGS = [
    # (utterance, expected_intent)
    # From the design doc — all previously fell to GenAI via TF-IDF
    ("kill the sound",          "Cmd.VolumeMute"),
    ("dim the audio",           "Cmd.VolumeDecrease"),
    ("log a jog",               "Cmd.ActivityRun"),
    # VolumeMute or VolumeDecrease are both reasonable here; head picks decrease
    ("I need some quiet",       "Cmd.VolumeDecrease"),
    ("switch listening profile","Cmd.MemoryChange"),
    ("cant hear well",          "Cmd.VolumeIncrease"),
    ("aid keeps cutting out",   "Help_DeviceSettings"),
    # Additional novel phrasings worth testing
    ("everything sounds muffled",   "Help_DeviceSettings"),
    ("crank it up",                 "Cmd.VolumeIncrease"),
    ("my ears are ringing",         "Help_Tinnitus"),

]

OUT_OF_SCOPE = [
    # These should NEVER be rescued — genuinely out of domain
    ("what is the weather today",       "GENAI"),
    ("who won the world cup",           "GENAI"),
    ("tell me a joke",                  "GENAI"),
    ("what is the capital of France",   "GENAI"),
    ("how do I make pasta",             "GENAI"),
    ("convert 10 dollars to euros",     "GENAI"),
]

KNOWN_PHRASINGS = [
    # Should be handled by TF-IDF (Stage 2), semantic_rescue must be False
    ("mute",                    "Cmd.VolumeMute"),
    ("silence please",          "Cmd.VolumeMute"),
    ("turn it down",            "Cmd.VolumeDecrease"),
    ("increase the volume",     "Cmd.VolumeIncrease"),
    ("start running",           "Cmd.ActivityRun"),
    ("change to restaurant",    "Cmd.MemoryChange"),
    ("remind me tomorrow at 9am to call chris", "reminders.add"),
]


# ── Runner ────────────────────────────────────────────────────────────────────

def run_suite(engine: NLUEngine, cases, label: str, expect_rescue: bool = None):
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {label}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")

    passed = 0
    failed = 0

    for text, expected_intent in cases:
        result = engine.handle("test-session", text)
        engine.reset("test-session")

        actual_intent = result.intent if result.type != "FALLBACK" else "GENAI"
        conf = result.confidence
        rescued = result.semantic_rescue

        ok = actual_intent == expected_intent
        rescue_ok = True
        if expect_rescue is True  and not rescued: rescue_ok = False
        if expect_rescue is False and rescued:     rescue_ok = False

        overall = ok and rescue_ok

        if overall:
            passed += 1
            status = f"{GREEN}✅{RESET}"
        else:
            failed += 1
            status = f"{RED}❌{RESET}"

        rescue_tag = f"{YELLOW}[semantic]{RESET}" if rescued else "[tfidf]  "
        intent_display = actual_intent[:28].ljust(28)
        mismatch = f"  {RED}← expected {expected_intent}{RESET}" if not ok else ""
        rescue_mismatch = f"  {RED}← rescue={rescued}, expected {expect_rescue}{RESET}" if not rescue_ok else ""

        print(f"  {status} {rescue_tag}  conf={conf:.2f}  {intent_display}  \"{text}\"{mismatch}{rescue_mismatch}")

    print(f"\n  {passed}/{passed+failed} passed")
    return passed, failed


def main():
    print(f"\n{BOLD}Semantic Fallback — Phase 3 Threshold Tuning{RESET}")

    engine = NLUEngine()

    if engine.semantic is None:
        print(f"\n{RED}ERROR: SemanticFallback not loaded.{RESET}")
        print("Model files not found. Make sure you have run:")
        print("  python scripts/download_minilm.py")
        print("  python scripts/build_semantic_index.py")
        sys.exit(1)

    print(f"\nThreshold: {engine.semantic.threshold}")
    if engine.semantic._head is not None:
        print(f"Mode: classification head ({len(engine.semantic._head[2])} classes)")
    else:
        print(f"Mode: 1-NN index ({len(engine.semantic._intents)} phrases)")

    total_passed = 0
    total_failed = 0

    p, f = run_suite(engine, NOVEL_PHRASINGS,
                     "1. Novel phrasings — must be rescued by semantic",
                     expect_rescue=True)
    total_passed += p; total_failed += f

    p, f = run_suite(engine, OUT_OF_SCOPE,
                     "2. Out-of-scope — must NOT be rescued (still GenAI)",
                     expect_rescue=False)
    total_passed += p; total_failed += f

    p, f = run_suite(engine, KNOWN_PHRASINGS,
                     "3. Known phrasings — handled by TF-IDF (no semantic needed)",
                     expect_rescue=False)
    total_passed += p; total_failed += f

    print(f"\n{BOLD}{'='*60}{RESET}")
    overall = f"{GREEN}ALL PASSED{RESET}" if total_failed == 0 else f"{RED}{total_failed} FAILED{RESET}"
    print(f"{BOLD}  Total: {total_passed}/{total_passed+total_failed}  —  {overall}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    if total_failed > 0:
        print("Threshold tuning hints:")
        print("  • Novel phrasings failing → lower threshold (try 0.78–0.80)")
        print("  • Out-of-scope being rescued → raise threshold (try 0.84–0.86)")
        sys.exit(1)


if __name__ == "__main__":
    main()
