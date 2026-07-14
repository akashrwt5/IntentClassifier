#!/usr/bin/env python3
"""
Heuristic mislabel scanner for datasets/01_source_base_training_data.csv.

It does NOT use the trained model (which would just echo its own biases).
Instead it defines strong, mutually-exclusive keyword signatures for the
most-confusable intent families (the activity trackers, the volume/stream
commands, etc.) and flags any row whose text matches a DIFFERENT intent's
signature far more strongly than its own label.

Output is a REVIEW LIST, not an auto-delete — a human confirms before
anything is removed.

Run:
    python scripts/scan_mislabels.py
    python scripts/scan_mislabels.py --added-only   # only the 252 new rows
"""

import csv
import re
import sys
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parents[3]
DATA_PATH = BASE_DIR / "datasets" / "01_source_base_training_data.csv"

# Strong, fairly exclusive signatures. A row labelled intent A but whose text
# clearly matches intent B's signature (and not A's) is a mislabel candidate.
# Keep these tight to avoid false positives.
SIGNATURES = {
    "Cmd.ActivityWalk":      [r"\bwalk", r"\bwalking\b"],
    "Cmd.ActivityRun":       [r"\brun\b", r"\brunning\b", r"\bjog"],
    "Cmd.ActivityCycle":     [r"\bcycl", r"\bbik(e|ing)\b", r"\bbicycl", r"\bride\b", r"\briding\b"],
    "Cmd.ActivityStep":      [r"\bstep", r"\bsteps\b"],
    "Cmd.ActivityStand":     [r"\bstand", r"\bstanding\b", r"\bstood\b"],
    "Cmd.ActivityAerobics":  [r"\baerobic"],
    "Cmd.ActivityCalories":  [r"\bcalorie"],
    "Cmd.ActivityExercise":  [r"\bexercis", r"\bwork\s*out", r"\bworking out\b"],
    "Cmd.VolumeIncrease":    [r"\blouder\b", r"\bturn it up\b", r"\bvolume up\b", r"\bincrease.*volume\b", r"\bcrank"],
    "Cmd.VolumeDecrease":    [r"\bquieter\b", r"\bturn it down\b", r"\bvolume down\b", r"\bdecrease.*volume\b", r"\blower.*volume\b"],
    "Cmd.VolumeMute":        [r"\bmute\b", r"\bsilence\b"],
    "Cmd.VolumeUnmute":      [r"\bunmute\b"],
    "Cmd.StreamingStart":    [r"\bstart stream", r"\bbegin stream", r"\bstart streaming\b"],
    "Cmd.StreamingStop":     [r"\bstop stream", r"\bend stream", r"\bturn off.*stream", r"\bdisconnect.*stream"],
    "Cmd.BatteryLevel":      [r"battery (level|percentage|status)\b", r"how much battery\b", r"\bbattery left\b"],
    "Cmd.FindMyPhone":       [r"find my phone\b", r"where.*my phone\b", r"locate.*phone\b"],
    "Help_FindMyHearingAids":[r"find my hearing aid", r"where.*hearing aid", r"locate.*hearing aid", r"lost.*hearing aid"],
    "Cmd.TranscribeStart":   [r"\btranscri", r"\bcaption", r"\brecord this\b", r"speech (in)?to text\b"],
    "Cmd.TranslationStart":  [r"\btranslat"],
    "Help_HeartRateRecovery":[r"heart rate recovery\b", r"recovery (rate|number|score)\b"],
    "Help_HeartRate":        [r"heart rate\b", r"\bheartbeat\b"],
    "Help_Tinnitus":         [r"\btinnitus\b", r"\bringing\b", r"\bbuzzing\b"],
    "Help_EdgeMode":         [r"\bedge mode\b"],
    "Help_MaskMode":         [r"\bmask mode\b", r"\bface mask\b"],
    "Help_DemoMode":         [r"\bdemo mode\b", r"\bdemonstrat", r"without (a )?hearing aid", r"without aids\b"],
    "Help_CleanCare":        [r"\bwax\b", r"\bclean", r"\bwash"],
    "Help_WiCROS":           [r"\bcros\b", r"\bwicros\b", r"\btransmitter\b", r"\breceiver\b", r"\bbalance control\b"],
}

# Some intents legitimately share words. Suppress known-benign overlaps:
# (label_intent, matched_signature_intent) pairs that are NOT mislabels.
ALLOWED_OVERLAP = {
    # Recovery rows naturally contain "heart rate"
    ("Help_HeartRateRecovery", "Help_HeartRate"),
    # Calorie questions exist inside every activity tracker
    ("Cmd.ActivityWalk", "Cmd.ActivityCalories"),
    ("Cmd.ActivityRun", "Cmd.ActivityCalories"),
    ("Cmd.ActivityCycle", "Cmd.ActivityCalories"),
    ("Cmd.ActivityStep", "Cmd.ActivityCalories"),
    ("Cmd.ActivityStand", "Cmd.ActivityCalories"),
    ("Cmd.ActivityAerobics", "Cmd.ActivityCalories"),
    ("Cmd.ActivityExercise", "Cmd.ActivityCalories"),
    # Exercise is an umbrella that mentions specific activities
    ("Cmd.ActivityExercise", "Cmd.ActivityWalk"),
    ("Cmd.ActivityExercise", "Cmd.ActivityRun"),
    ("Cmd.ActivityExercise", "Cmd.ActivityCycle"),
    # Health screen rows reference walking/biking/steps as examples
    ("Help_Health", "Cmd.ActivityWalk"),
    ("Help_Health", "Cmd.ActivityCycle"),
    ("Help_Health", "Cmd.ActivityStep"),
    ("Help_Health", "Cmd.ActivityRun"),
    ("Help_Health", "Cmd.ActivityAerobics"),
    # Streaming stop vs start both contain "stream"
    ("Cmd.StreamingStop", "Cmd.StreamingStart"),
    ("Cmd.StreamingStart", "Cmd.StreamingStop"),
}


def compile_sigs():
    return {intent: [re.compile(p) for p in pats] for intent, pats in SIGNATURES.items()}


def matches(text, compiled):
    return [intent for intent, pats in compiled.items()
            if any(p.search(text) for p in pats)]


def main():
    added_only = "--added-only" in sys.argv

    with open(DATA_PATH, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    # Heuristic: the last 252 rows are the additions (appended in order).
    added_set = set()
    for r in rows[-252:]:
        added_set.add((r["text"].strip().lower(), r["intent"].strip()))

    compiled = compile_sigs()
    flagged = []

    for idx, r in enumerate(rows):
        text   = r["text"].strip().lower()
        label  = r["intent"].strip()
        if added_only and (text, label) not in added_set:
            continue
        if label == "Default Fallback Intent":
            continue

        hit_intents = matches(text, compiled)
        if not hit_intents:
            continue

        own_sig = compiled.get(label)
        own_match = own_sig and any(p.search(text) for p in own_sig)

        # Other intents whose signature this text matches
        others = [h for h in hit_intents
                  if h != label
                  and (label, h) not in ALLOWED_OVERLAP]
        if not others:
            continue

        # Flag only if text matches another intent but NOT its own label's
        # signature (clear cross-mapping) — or matches another with no own sig.
        if not own_match:
            is_added = (text, label) in added_set
            flagged.append((idx, text, label, others, is_added))

    print("=" * 70)
    print("MISLABEL SCAN" + ("  (added rows only)" if added_only else "  (full file)"))
    print("=" * 70)
    print(f"\nRows scanned: {len(rows)}   Flagged candidates: {len(flagged)}\n")

    if not flagged:
        print("No cross-mapping mislabels detected by the keyword signatures.")
        return

    added_flags = [f for f in flagged if f[4]]
    print(f"Of which in MY 252 additions: {len(added_flags)}\n")
    print(f"{'#':>5}  {'labelled':28} {'signature points to':28} text")
    print("-" * 70)
    for idx, text, label, others, is_added in flagged:
        tag = " *ADDED*" if is_added else ""
        print(f"{idx:>5}  {label[:28]:28} {', '.join(others)[:28]:28} \"{text}\"{tag}")


if __name__ == "__main__":
    main()
