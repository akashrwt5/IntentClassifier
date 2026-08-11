#!/usr/bin/env python3
"""
V6 ENGLISH PRODUCTION VOCABULARY / COVERAGE BUILDER

Purpose
-------
Build a REVIEW-ONLY candidate vocabulary set for the 57-intent English
classifier.

IMPORTANT:
- Does NOT modify train.csv.
- Does NOT modify V4/V5 checkpoints.
- Does NOT read the 1686-row locked test for training.
- Candidate phrases are NOT automatically used for training.
- Generated candidates must be reviewed/approved before V6 training.
- No ONNX.
- No quantization.
- English only.

Inputs
------
Primary training data:
    /Users/shuklam/IntentClassifier/semantic_project/57semanitc/train.csv

Existing V5 hard-negative candidates, if present:
    .../v5_e5_english_hard_negative_refinement/hard_negative_predictions.csv

Outputs
-------
    v6_english_vocab_review/
        existing_intent_coverage.csv
        candidate_vocab_review.csv
        candidate_by_intent.csv
        ambiguity_review.csv
        production_vocab_summary.json

The candidate CSV has:
    intent
    text
    source
    status

status is ALWAYS "REVIEW_REQUIRED".

Do not train V6 from this file until the rows are reviewed.
"""

from pathlib import Path
import json
import re
import pandas as pd


PROJECT = Path(
    "/Users/shuklam/IntentClassifier/semantic_project/57semanitc"
)

TRAIN_CSV = PROJECT / "train.csv"

V5_DIR = (
    PROJECT / "v5_e5_english_hard_negative_refinement"
)

V5_HARD_NEGATIVE_CSV = (
    V5_DIR / "hard_negative_predictions.csv"
)

OUT_DIR = PROJECT / "v6_english_vocab_review"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# HIGH-CONFIDENCE PRODUCTION VOCABULARY
#
# These are REVIEW candidates only.
# They are deliberately conservative and concentrated on short/ambiguous
# commands that caused real-world failures.
#
# Do not silently assume an ambiguous phrase's intent. Every row is marked
# REVIEW_REQUIRED and must be accepted/rejected by the product owner.
# ---------------------------------------------------------------------

CANDIDATES = {

    "Cmd.VolumeIncrease": [
        "make it louder",
        "make the volume louder",
        "turn the volume up",
        "turn it up",
        "increase the volume",
        "increase volume",
        "volume up",
        "raise the volume",
        "raise volume",
        "make it a little louder",
        "make it much louder",
        "it's too quiet",
        "can you make it louder",
        "please make it louder",
        "louder please",
    ],

    "Cmd.VolumeDecrease": [
        "make it quieter",
        "make the volume quieter",
        "turn the volume down",
        "turn it down",
        "decrease the volume",
        "decrease volume",
        "volume down",
        "lower the volume",
        "lower volume",
        "make it a little quieter",
        "make it much quieter",
        "it's too loud",
        "can you make it quieter",
        "please make it quieter",
        "quieter please",
    ],

    "Cmd.VolumeMute": [
        "mute",
        "mute it",
        "mute the volume",
        "silence it",
        "silence the volume",
        "stop the sound",
        "stop sound",
        "turn the sound off",
        "turn off the sound",
        "turn the audio off",
        "make it silent",
        "make the sound silent",
        "no sound",
        "quiet please",
    ],

    "Cmd.VolumeUnmute": [
        "unmute",
        "unmute it",
        "unmute the volume",
        "turn the sound on",
        "turn the audio on",
        "restore the sound",
        "restore the volume",
        "bring the sound back",
        "turn sound back on",
        "turn volume back on",
    ],

    "Cmd.StreamingStart": [
        "start streaming",
        "start the stream",
        "begin streaming",
        "begin the stream",
        "turn streaming on",
        "stream now",
        "start streaming now",
        "can you start streaming",
    ],

    "Cmd.StreamingStop": [
        "stop streaming",
        "stop the stream",
        "end streaming",
        "end the stream",
        "turn streaming off",
        "streaming off",
        "stop the streaming",
        "can you stop streaming",
    ],

    "Cmd.ListenMessage": [
        "listen to my message",
        "read my message",
        "read the message",
        "play my message",
        "listen to the message",
        "hear my message",
    ],

    "Cmd.SendMessage": [
        "send a message",
        "send the message",
        "send this message",
        "send my message",
        "send it",
        "send this",
        "please send the message",
        "can you send this message",
    ],

    "Cmd.FindMyPhone": [
        "find my phone",
        "locate my phone",
        "where is my phone",
        "find the phone",
        "locate the phone",
        "help me find my phone",
    ],

    "Cmd.BatteryLevel": [
        "check the battery",
        "check my battery",
        "what is the battery level",
        "what's the battery level",
        "how much battery is left",
        "how much battery do I have",
        "show battery level",
        "tell me the battery level",
    ],

    "Cmd.MemoryChange": [
        "change the memory",
        "change memory",
        "switch memory",
        "switch the memory",
        "change my memory",
        "change to another memory",
        "switch to another memory",
    ],

    "Cmd.TranscribeStart": [
        "start transcription",
        "start transcribing",
        "begin transcription",
        "transcribe this",
        "start the transcription",
        "turn transcription on",
    ],

    "Cmd.TranslationStart": [
        "start translation",
        "start translating",
        "begin translation",
        "translate this",
        "start the translation",
        "turn translation on",
    ],

    "reminders.add": [
        "remind me",
        "set a reminder",
        "create a reminder",
        "add a reminder",
        "make a reminder",
        "remember this for me",
        "remind me about this",
        "set me a reminder",
        "please remind me",
    ],

    "reminders.complete": [
        "complete the reminder",
        "mark the reminder complete",
        "finish the reminder",
        "complete my reminder",
        "mark my reminder as done",
        "done with the reminder",
    ],

    "Default Fallback Intent": [
        # These are intentionally generic/OOD candidates.
        # They should remain fallback unless the product explicitly adds
        # a new supported intent.
        "play some music",
        "show me a funny video",
        "send an email",
        "book me a flight",
        "take me to the airport",
        "go to the airport tomorrow at 9 pm",
        "turn off the television",
        "turn off the lights",
        "turn off my phone",
        "open the camera",
        "what is the weather",
        "call my friend",
    ],

    # Help intents: only very conservative lexical candidates are added.
    "Help_Volume": [
        "how do I change the volume",
        "how can I change the volume",
        "how do I adjust the volume",
        "help with volume",
        "help me with volume",
    ],

    "Help_Pairing": [
        "how do I pair my hearing aids",
        "how can I pair my hearing aids",
        "help me pair my hearing aids",
        "how do I connect my hearing aids",
        "help with pairing",
    ],

    "Help_Battery": [
        "how do I check the battery",
        "how can I check the battery",
        "help with battery",
        "where can I see the battery level",
    ],

    "Help_FindMyHearingAids": [
        "how do I find my hearing aids",
        "help me find my hearing aids",
        "where are my hearing aids",
        "how can I locate my hearing aids",
    ],

    "Help_Reminder": [
        "how do I set a reminder",
        "how can I create a reminder",
        "help with reminders",
        "how do reminders work",
    ],

    "Help_Translate": [
        "how do I translate",
        "how can I translate",
        "help with translation",
        "how does translation work",
    ],

    "Help_Transcribe": [
        "how do I transcribe",
        "how can I transcribe",
        "help with transcription",
        "how does transcription work",
    ],

    "Help_VoiceAssistant": [
        "how do I use the voice assistant",
        "how can I use the voice assistant",
        "help with voice assistant",
    ],

    "Help_SelfCheck": [
        "how do I run a self check",
        "how can I run a self check",
        "help with self check",
        "how does self check work",
    ],

    "Help_DeviceSettings": [
        "how do I change device settings",
        "how can I change device settings",
        "help with device settings",
        "where are the device settings",
    ],

    "Help_AppSettings": [
        "how do I change app settings",
        "how can I change app settings",
        "help with app settings",
        "where are the app settings",
    ],
}


def normalize(text):
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def load_training():
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"train.csv not found:\n{TRAIN_CSV}"
        )

    df = pd.read_csv(TRAIN_CSV)

    if "text" not in df.columns or "intent" not in df.columns:
        raise RuntimeError(
            f"train.csv must contain text and intent columns. "
            f"Found: {list(df.columns)}"
        )

    df = df[["text", "intent"]].copy()
    df["text_norm"] = df["text"].map(normalize)

    return df


def build_coverage(df):
    rows = []

    for intent, part in df.groupby("intent"):
        texts = set(part["text_norm"])

        rows.append({
            "intent": intent,
            "train_rows": len(part),
            "unique_texts": len(texts),
            "unique_candidates_added": 0,
        })

    coverage = pd.DataFrame(rows)

    return coverage.sort_values("train_rows")


def main():

    print("=" * 78)
    print("V6 ENGLISH PRODUCTION VOCABULARY / COVERAGE BUILDER")
    print("=" * 78)

    df = load_training()

    train_texts = set(
        df["text_norm"].tolist()
    )

    train_intents = set(
        df["intent"].tolist()
    )

    print()
    print(f"Training rows : {len(df)}")
    print(f"Training intents: {len(train_intents)}")

    if len(train_intents) != 57:
        raise RuntimeError(
            f"Expected 57 intents, found {len(train_intents)}."
        )

    # ---------------------------------------------------------------
    # BUILD REVIEW CANDIDATES
    # ---------------------------------------------------------------

    rows = []
    seen = set()

    for intent, phrases in CANDIDATES.items():

        if intent not in train_intents:
            print(
                f"WARNING: candidate intent not in train.csv: "
                f"{intent}"
            )
            continue

        for phrase in phrases:

            text = normalize(phrase)

            if not text:
                continue

            key = (intent, text)

            if key in seen:
                continue

            seen.add(key)

            source = (
                "curated_production_candidate"
            )

            status = "REVIEW_REQUIRED"

            if text in train_texts:
                status = "ALREADY_IN_TRAIN"

            rows.append({
                "intent": intent,
                "text": phrase,
                "source": source,
                "status": status,
            })

    candidates = pd.DataFrame(rows)

    # ---------------------------------------------------------------
    # EXISTING HARD NEGATIVE PREDICTIONS
    # ---------------------------------------------------------------

    ambiguity_rows = []

    if V5_HARD_NEGATIVE_CSV.exists():

        hn = pd.read_csv(
            V5_HARD_NEGATIVE_CSV
        )

        # Keep this flexible because V5 output column names may vary.
        text_col = next(
            (
                c for c in [
                    "text",
                    "utterance",
                    "query",
                ]
                if c in hn.columns
            ),
            None,
        )

        pred_col = next(
            (
                c for c in [
                    "prediction",
                    "predicted_intent",
                    "pred",
                ]
                if c in hn.columns
            ),
            None,
        )

        true_col = next(
            (
                c for c in [
                    "intent",
                    "true_intent",
                    "label",
                    "expected_intent",
                ]
                if c in hn.columns
            ),
            None,
        )

        if text_col:
            keep = [
                c for c in [
                    text_col,
                    true_col,
                    pred_col,
                ]
                if c
            ]

            ambiguity_rows = hn[keep].copy()

            rename = {
                text_col: "text",
            }

            if true_col:
                rename[true_col] = "expected_intent"

            if pred_col:
                rename[pred_col] = "predicted_intent"

            ambiguity_rows = (
                ambiguity_rows.rename(
                    columns=rename
                )
            )

            ambiguity_rows[
                "text_norm"
            ] = ambiguity_rows[
                "text"
            ].map(normalize)

            ambiguity_rows[
                "needs_review"
            ] = True

    # ---------------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------------

    coverage = build_coverage(df)

    candidate_counts = (
        candidates[
            candidates["status"]
            != "ALREADY_IN_TRAIN"
        ]
        .groupby("intent")
        .size()
        .rename(
            "unique_candidates_added"
        )
    )

    coverage = coverage.merge(
        candidate_counts,
        on="intent",
        how="left",
        suffixes=("", "_new"),
    )

    if "unique_candidates_added_new" in coverage.columns:
        coverage[
            "unique_candidates_added"
        ] = coverage[
            "unique_candidates_added_new"
        ].fillna(0).astype(int)

        coverage = coverage.drop(
            columns=[
                "unique_candidates_added_new"
            ]
        )

    coverage_path = (
        OUT_DIR / "existing_intent_coverage.csv"
    )

    candidates_path = (
        OUT_DIR / "candidate_vocab_review.csv"
    )

    by_intent_path = (
        OUT_DIR / "candidate_by_intent.csv"
    )

    ambiguity_path = (
        OUT_DIR / "ambiguity_review.csv"
    )

    summary_path = (
        OUT_DIR / "production_vocab_summary.json"
    )

    coverage.to_csv(
        coverage_path,
        index=False,
    )

    candidates.to_csv(
        candidates_path,
        index=False,
    )

    candidates.sort_values(
        ["intent", "text"]
    ).to_csv(
        by_intent_path,
        index=False,
    )

    if isinstance(
        ambiguity_rows,
        pd.DataFrame
    ) and len(ambiguity_rows):
        ambiguity_rows.to_csv(
            ambiguity_path,
            index=False,
        )
    else:
        pd.DataFrame(
            columns=[
                "text",
                "expected_intent",
                "predicted_intent",
                "needs_review",
            ]
        ).to_csv(
            ambiguity_path,
            index=False,
        )

    review_count = int(
        (
            candidates["status"]
            == "REVIEW_REQUIRED"
        ).sum()
    )

    already_count = int(
        (
            candidates["status"]
            == "ALREADY_IN_TRAIN"
        ).sum()
    )

    summary = {
        "purpose": (
            "English production vocabulary review "
            "before V6 training"
        ),
        "training_csv": str(TRAIN_CSV),
        "training_rows": int(len(df)),
        "training_intents": int(len(train_intents)),
        "candidate_rows": int(len(candidates)),
        "review_required": review_count,
        "already_in_train": already_count,
        "hard_negative_source": (
            str(V5_HARD_NEGATIVE_CSV)
            if V5_HARD_NEGATIVE_CSV.exists()
            else None
        ),
        "candidate_policy": (
            "Candidates are review-only and must not be "
            "automatically added to training."
        ),
        "synthetic_training": False,
        "locked_test_used_for_training": False,
        "quantization": False,
        "onnx": False,
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------------
    # PRINT
    # ---------------------------------------------------------------

    print()
    print("=" * 78)
    print("V6 PRODUCTION VOCABULARY REVIEW SET CREATED")
    print("=" * 78)

    print(
        f"Candidate rows       : {len(candidates)}"
    )
    print(
        f"Review required      : {review_count}"
    )
    print(
        f"Already in train.csv : {already_count}"
    )

    print()
    print(
        "Important: candidates are REVIEW_REQUIRED."
    )
    print(
        "Do NOT train V6 from candidate_vocab_review.csv "
        "until expected intents are approved."
    )

    print()
    print("Saved:")
    print(coverage_path)
    print(candidates_path)
    print(by_intent_path)
    print(ambiguity_path)
    print(summary_path)

    print()
    print(
        "STATUS: V6 ENGLISH VOCABULARY REVIEW "
        "CANDIDATES READY"
    )


if __name__ == "__main__":
    main()
