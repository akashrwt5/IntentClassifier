#!/usr/bin/env python3
"""
V7-02 TARGETED VOCABULARY REVIEW

Creates a conservative, human-reviewable candidate set from V6 smoke-test
errors and existing V6 training data.

NO model training.
NO locked-test access.
NO synthetic generation.
NO automatic addition of candidates to training.

Important:
- Candidate rows come ONLY from existing train.csv or existing smoke-test
  error examples.
- A smoke-test error is not automatically considered safe training data.
- The output marks each candidate as REVIEW_REQUIRED.
"""

from pathlib import Path
import json
import pandas as pd

PROJECT = Path(
    "/Users/shuklam/IntentClassifier/semantic_project/57semanitc"
)

SMOKE_DIR = PROJECT / "v6_production_smoke_test"
ERROR_FILE = SMOKE_DIR / "smoke_errors.csv"

TRAIN_CSV = PROJECT / "train.csv"

OUT_DIR = PROJECT / "v7_02_targeted_vocabulary_review"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FALLBACK = "Default Fallback Intent"


def clean(x):
    return " ".join(str(x).strip().split())


def load_train():
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"train.csv not found:\n{TRAIN_CSV}"
        )

    df = pd.read_csv(TRAIN_CSV)

    text_col = None
    intent_col = None

    for c in df.columns:
        if str(c).lower() in {
            "text", "utterance", "query", "sentence"
        }:
            text_col = c
            break

    for c in df.columns:
        if str(c).lower() in {
            "intent", "label", "true_intent", "expected_intent"
        }:
            intent_col = c
            break

    if text_col is None or intent_col is None:
        raise RuntimeError(
            f"Could not identify text/intent columns in {list(df.columns)}"
        )

    out = pd.DataFrame({
        "text": df[text_col].map(clean),
        "intent": df[intent_col].map(clean),
    })

    return out[
        (out["text"] != "")
        & (out["intent"] != "")
    ].drop_duplicates("text")


def load_errors():
    if not ERROR_FILE.exists():
        raise FileNotFoundError(
            f"Smoke errors not found:\n{ERROR_FILE}"
        )

    df = pd.read_csv(ERROR_FILE)

    required = {
        "text",
        "intent",
        "prediction",
        "confidence",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing smoke-error columns: {sorted(missing)}"
        )

    for col in [
        "text",
        "intent",
        "prediction",
    ]:
        df[col] = df[col].map(clean)

    return df


def add_cluster(row):
    true_intent = row["intent"]
    prediction = row["prediction"]

    if true_intent == "Cmd.BatteryLevel":
        if prediction == FALLBACK:
            return "battery_vs_oos"
        if prediction == "Help_DeviceSettings":
            return "battery_vs_device_settings"

    if true_intent == "Help_Home":
        return "home_vs_oos"

    if true_intent == "Help_ChangingMemories":
        return "changing_memories_vs_memory_options"

    if true_intent == "reminders.complete":
        return "reminder_completion_vs_oos"

    if true_intent == "Help_WhatsNew":
        return "whatsnew_vs_home"

    if true_intent == "Help_InsertDevice":
        return "insert_device_vs_oos"

    if true_intent == "Help_EdgeMode":
        return "edge_mode_vs_volume"

    if true_intent == "Help_HearingCareAnywhereConnect":
        return "care_anywhere_vs_remote_programming"

    if true_intent == "Cmd.VolumeMute":
        return "mute_vs_volume_decrease"

    if true_intent == "Help_Battery":
        return "battery_help_vs_oos"

    if true_intent == "Help_ThriveScore":
        return "thrive_score_vs_oos"

    if true_intent == "Cmd.SendMessage":
        return "send_message_vs_oos"

    if true_intent == "Help_HeartRateRecovery":
        return "heart_rate_recovery_vs_thrive_score"

    return "other_targeted_confusion"


def main():

    print("=" * 78)
    print("V7-02 TARGETED VOCABULARY REVIEW")
    print("=" * 78)

    train = load_train()
    errors = load_errors()

    print(
        f"Training rows available for coverage check: {len(train)}"
    )
    print(
        f"Smoke errors: {len(errors)}"
    )

    train_text = set(
        train["text"].str.lower()
    )

    # Candidate examples are existing smoke-test rows.
    candidates = errors.copy()

    candidates["cluster"] = candidates.apply(
        add_cluster,
        axis=1,
    )

    candidates["already_in_train"] = (
        candidates["text"]
        .str.lower()
        .isin(train_text)
    )

    candidates["review_status"] = "REVIEW_REQUIRED"

    # High-confidence mistakes deserve extra review because they indicate
    # a stronger semantic boundary problem.
    candidates["priority"] = "NORMAL"

    candidates.loc[
        candidates["confidence"] >= 0.80,
        "priority"
    ] = "HIGH"

    candidates.loc[
        candidates["confidence"] < 0.35,
        "priority"
    ] = "LOW"

    # Never recommend automatic training from an already-seen exact row.
    candidates.loc[
        candidates["already_in_train"],
        "review_status"
    ] = "ALREADY_IN_TRAIN"

    # Sort by cluster and then confidence.
    candidates = candidates.sort_values(
        [
            "review_status",
            "cluster",
            "confidence",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    )

    # ---------------------------------------------------------------
    # Candidate list
    # ---------------------------------------------------------------

    candidates[
        [
            "text",
            "intent",
            "prediction",
            "confidence",
            "cluster",
            "priority",
            "already_in_train",
            "review_status",
        ]
    ].to_csv(
        OUT_DIR / "targeted_vocab_candidates.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Cluster summary
    # ---------------------------------------------------------------

    cluster_summary = (
        candidates.groupby(
            [
                "cluster",
                "intent",
                "prediction",
            ]
        )
        .agg(
            candidate_count=("text", "count"),
            mean_confidence=("confidence", "mean"),
            high_priority_count=(
                "priority",
                lambda s: int(
                    (s == "HIGH").sum()
                ),
            ),
        )
        .reset_index()
        .sort_values(
            "candidate_count",
            ascending=False,
        )
    )

    cluster_summary.to_csv(
        OUT_DIR / "targeted_cluster_summary.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Only rows that are potentially usable after human approval
    # ---------------------------------------------------------------

    review_only = candidates[
        candidates["review_status"]
        == "REVIEW_REQUIRED"
    ].copy()

    review_only[
        [
            "text",
            "intent",
            "prediction",
            "confidence",
            "cluster",
            "priority",
        ]
    ].to_csv(
        OUT_DIR / "review_required.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Explicit recommendations
    # ---------------------------------------------------------------

    recommendations = [
        {
            "cluster": "battery_vs_oos",
            "true_intent": "Cmd.BatteryLevel",
            "action": "REVIEW",
            "reason": (
                "Add/retain vocabulary around remaining battery, "
                "battery running low, charging soon, and runtime."
            ),
        },
        {
            "cluster": "battery_vs_device_settings",
            "true_intent": "Cmd.BatteryLevel",
            "action": "REVIEW",
            "reason": (
                "Strengthen explicit battery-state wording so it "
                "separates from generic device settings."
            ),
        },
        {
            "cluster": "home_vs_oos",
            "true_intent": "Help_Home",
            "action": "REVIEW",
            "reason": (
                "Home/help navigation examples are currently broad "
                "and can look like OOD/general assistance."
            ),
        },
        {
            "cluster": "changing_memories_vs_memory_options",
            "true_intent": "Help_ChangingMemories",
            "action": "REVIEW",
            "reason": (
                "Separate changing/selecting programs or memories "
                "from asking what memory options are."
            ),
        },
        {
            "cluster": "reminder_completion_vs_oos",
            "true_intent": "reminders.complete",
            "action": "REVIEW",
            "reason": (
                "Natural completion phrases such as 'cross it off' "
                "need stronger representation."
            ),
        },
        {
            "cluster": "whatsnew_vs_home",
            "true_intent": "Help_WhatsNew",
            "action": "REVIEW",
            "reason": (
                "Quick-start/guide language overlaps with Home/help."
            ),
        },
        {
            "cluster": "insert_device_vs_oos",
            "true_intent": "Help_InsertDevice",
            "action": "REVIEW",
            "reason": (
                "Wearing/inserting/dome/earmold vocabulary needs "
                "clearer coverage."
            ),
        },
        {
            "cluster": "edge_mode_vs_volume",
            "true_intent": "Help_EdgeMode",
            "action": "REVIEW",
            "reason": (
                "Noise-reduction/background-noise wording overlaps "
                "with volume commands."
            ),
        },
        {
            "cluster": "care_anywhere_vs_remote_programming",
            "true_intent": "Help_HearingCareAnywhereConnect",
            "action": "REVIEW",
            "reason": (
                "Professional/audiologist connection terminology "
                "overlaps with remote programming."
            ),
        },
        {
            "cluster": "mute_vs_volume_decrease",
            "true_intent": "Cmd.VolumeMute",
            "action": "REVIEW",
            "reason": (
                "Zero-sound/cut-all-sound wording must be separated "
                "from merely lowering volume."
            ),
        },
        {
            "cluster": "send_message_vs_oos",
            "true_intent": "Cmd.SendMessage",
            "action": "REVIEW",
            "reason": (
                "Natural messaging verbs such as text, send, fire off "
                "a text need stronger coverage."
            ),
        },
    ]

    pd.DataFrame(
        recommendations
    ).to_csv(
        OUT_DIR / "review_recommendations.csv",
        index=False,
    )

    summary = {
        "smoke_error_rows": int(len(errors)),
        "candidate_rows": int(len(candidates)),
        "review_required": int(
            (candidates["review_status"] == "REVIEW_REQUIRED").sum()
        ),
        "already_in_train": int(
            (candidates["review_status"] == "ALREADY_IN_TRAIN").sum()
        ),
        "high_priority": int(
            (candidates["priority"] == "HIGH").sum()
        ),
        "clusters": int(
            candidates["cluster"].nunique()
        ),
        "automatic_training": False,
        "locked_test_read": False,
        "synthetic_text_generated": False,
        "model_modified": False,
    }

    (
        OUT_DIR / "v7_02_review_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("V7-02 REVIEW SET")
    print("=" * 78)

    print(
        f"Candidate rows       : {len(candidates)}"
    )
    print(
        f"Review required      : "
        f"{(candidates['review_status'] == 'REVIEW_REQUIRED').sum()}"
    )
    print(
        f"Already in train.csv : "
        f"{(candidates['review_status'] == 'ALREADY_IN_TRAIN').sum()}"
    )
    print(
        f"High priority        : "
        f"{(candidates['priority'] == 'HIGH').sum()}"
    )

    print()
    print(
        "--- CLUSTER SUMMARY ---"
    )
    print(
        cluster_summary.head(20).to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    print(
        OUT_DIR / "targeted_vocab_candidates.csv"
    )
    print(
        OUT_DIR / "targeted_cluster_summary.csv"
    )
    print(
        OUT_DIR / "review_required.csv"
    )
    print(
        OUT_DIR / "review_recommendations.csv"
    )
    print(
        OUT_DIR / "v7_02_review_summary.json"
    )

    print()
    print("IMPORTANT:")
    print("No training performed.")
    print("V6 model was NOT modified.")
    print("Locked 1686-row test was NOT read.")
    print("No synthetic text was generated.")
    print(
        "Do NOT train from review_required.csv until examples are approved."
    )

    print()
    print(
        "STATUS: V7-02 TARGETED VOCABULARY REVIEW READY"
    )


if __name__ == "__main__":
    main()
