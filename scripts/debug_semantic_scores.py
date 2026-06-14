#!/usr/bin/env python3
"""
Diagnostic: show raw semantic similarity scores for each novel phrasing.
Tells us whether the model is scoring too low (embedding mismatch)
or just barely below threshold (threshold too strict).

Run:
    python scripts/debug_semantic_scores.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.nlu.semantic import SemanticFallback

NOVEL_PHRASINGS = [
    ("kill the sound",           "Cmd.VolumeMute"),
    ("dim the audio",            "Cmd.VolumeDecrease"),
    ("log a jog",                "Cmd.ActivityRun"),
    ("I need some quiet",        "Cmd.VolumeMute"),
    ("switch listening profile", "Cmd.MemoryChange"),
    ("cant hear well",           "Cmd.VolumeIncrease"),
    ("aid keeps cutting out",    "Help_DeviceSettings"),
    ("everything sounds muffled","Help_DeviceSettings"),
    ("crank it up",              "Cmd.VolumeIncrease"),
    ("my ears are ringing",      "Help_Tinnitus"),
    # Reference: known phrases — these should score high
    ("turn it down",             "Cmd.VolumeDecrease"),
    ("mute",                     "Cmd.VolumeMute"),
    ("I can't hear",             "Cmd.VolumeIncrease"),
]

sf = SemanticFallback()
print(f"\n{'Utterance':<35} {'Score':>6}  {'Top intent returned':<35} {'Expected'}")
print("─" * 110)
for text, expected in NOVEL_PHRASINGS:
    intent, score = sf.classify(text)
    match = "✅" if intent == expected else "❌"
    print(f"{text:<35} {score:>6.3f}  {intent:<35} {match} {expected}")
