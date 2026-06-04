#!/usr/bin/env python3
"""
Auto-label unknown/low-confidence inputs using keyword rules.

Reads:  data/unknown_data.csv
Writes: data/intent_data_new.csv (appends matching rows)

Run periodically to grow training data from real user inputs.
Always review the output manually before retraining.
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
UNKNOWN_PATH = BASE_DIR / "data" / "unknown_data.csv"
TRAINING_PATH = BASE_DIR / "data" / "intent_data_new.csv"

# ---------- Keyword rules ----------
# Add rules here as you discover patterns in unknown_data.csv
KEYWORD_RULES = {
    "volume":       "VOLUME",
    "louder":       "VOLUME",
    "quieter":      "VOLUME",
    "mute":         "VOLUME",
    "remind":       "REMINDER",
    "reminder":     "REMINDER",
    "weather":      "WEATHER_FORECAST",
    "notification": "NOTIFICATIONS",
    "translate":    "TRANSLATE",
    "transcribe":   "TRANSCRIBE",
    "telehear":     "TELEHEARAI",
    "selfcheck":    "SELFCHECK",
    "self check":   "SELFCHECK",
    "memory":       "MEMORY",
}


def auto_label(text: str) -> str | None:
    t = text.lower().strip()
    for keyword, intent in KEYWORD_RULES.items():
        if keyword in t:
            return intent
    return None


def main():
    if not UNKNOWN_PATH.exists():
        print("❌ No unknown_data.csv found")
        return

    unknown = pd.read_csv(UNKNOWN_PATH)
    rows = []

    for _, r in unknown.iterrows():
        intent = auto_label(str(r["text"]))
        if intent:
            rows.append([r["text"], intent])

    if rows:
        df = pd.DataFrame(rows, columns=["text", "intent"])
        df.to_csv(TRAINING_PATH, mode="a", header=False, index=False)
        print(f"✅ Added {len(rows)} samples to training data")
        print("⚠️  Review data/intent_data_new.csv before retraining!")
    else:
        print("❌ No rows could be auto-labeled")


if __name__ == "__main__":
    main()
