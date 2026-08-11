#!/usr/bin/env python3
"""
BUILD TARGETED V2 DATASET FOR THE 57-INTENT V3 MODEL

This script DOES NOT invent or relabel text.

It uses:
  1) original train.csv
  2) confusion_examples.csv from v3_57intent_error_analysis/
  3) top_confusion_pairs.csv from v3_57intent_error_analysis/

It creates:
  - train_v2_targeted.csv
  - hard_error_rows.csv
  - v2_dataset_summary.json

Strategy:
  - keep every original unique row once
  - oversample actual model-error rows (hard examples)
  - oversample minority classes up to MIN_CLASS_TARGET
  - cap total hard-example copies
  - never change an intent label
  - never generate synthetic text

This is deliberately conservative: first improve the model on examples it
already gets wrong, rather than inventing potentially incorrect labels.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path("/Users/shuklam/IntentClassifier/semantic_project/57semanitc")

TRAIN_CSV = ROOT / "train.csv"
ERROR_EXAMPLES = ROOT / "v3_57intent_error_analysis" / "confusion_examples.csv"
PAIRS_CSV = ROOT / "v3_57intent_error_analysis" / "top_confusion_pairs.csv"

OUT_DIR = ROOT / "v3_57intent_v2_dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "train_v2_targeted.csv"
HARD_CSV = OUT_DIR / "hard_error_rows.csv"
SUMMARY_JSON = OUT_DIR / "v2_dataset_summary.json"

MIN_CLASS_TARGET = 180
HARD_REPEAT = 3
MAX_HARD_ROWS_PER_INTENT = 180
RANDOM_SEED = 42


def detect_columns(columns):
    lower = {str(c).strip().lower(): c for c in columns}
    text_col = next(
        (lower[x] for x in ["text", "utterance", "query", "sentence", "input"] if x in lower),
        None,
    )
    label_col = next(
        (lower[x] for x in ["intent", "label", "category", "class"] if x in lower),
        None,
    )
    if text_col is None or label_col is None:
        raise RuntimeError(f"Could not detect text/intent columns: {list(columns)}")
    return text_col, label_col


def clean_base(path):
    df = pd.read_csv(path)
    text_col, label_col = detect_columns(df.columns)
    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "intent"]
    df["text"] = df["text"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["intent"] != "")]
    df = df.drop_duplicates(["text", "intent"]).reset_index(drop=True)
    return df


def main():
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Missing: {TRAIN_CSV}")

    if not ERROR_EXAMPLES.exists():
        raise FileNotFoundError(
            f"Missing:\n{ERROR_EXAMPLES}\n\n"
            "Copy your local v3_57intent_error_analysis folder into "
            "57semanitc first."
        )

    base = clean_base(TRAIN_CSV)

    errors = pd.read_csv(ERROR_EXAMPLES)

    required = {"text", "true_intent", "predicted_intent"}
    missing = required - set(errors.columns)
    if missing:
        raise RuntimeError(
            f"confusion_examples.csv missing columns: {sorted(missing)}"
        )

    errors = errors.rename(columns={"true_intent": "intent"})
    errors["text"] = errors["text"].astype(str).str.strip()
    errors["intent"] = errors["intent"].astype(str).str.strip()

    # Only retain genuine mistakes.
    errors = errors[
        errors["intent"] != errors["predicted_intent"]
    ].copy()

    errors = errors[
        errors["text"].isin(set(base["text"]))
    ].copy()

    # Deduplicate by text + intent. We don't want the same hard example
    # copied repeatedly merely because it appeared in multiple reports.
    errors = errors.drop_duplicates(
        ["text", "intent"]
    ).reset_index(drop=True)

    hard_by_intent = {
        k: g.copy()
        for k, g in errors.groupby("intent")
    }

    parts = [base.copy()]

    # 1. Actual model mistakes get repeated more strongly.
    hard_added = 0
    for intent, group in hard_by_intent.items():
        if intent not in set(base["intent"]):
            continue

        group = group.head(MAX_HARD_ROWS_PER_INTENT)

        hard = group[["text", "intent"]].copy()

        # HARD_REPEAT additional copies.
        for _ in range(HARD_REPEAT):
            parts.append(hard)
            hard_added += len(hard)

    # 2. Minority classes are brought to a moderate floor.
    # We deliberately use 180, not 300, to avoid exploding the dataset.
    counts = base["intent"].value_counts()

    for intent, count in counts.items():
        if count >= MIN_CLASS_TARGET:
            continue

        needed = MIN_CLASS_TARGET - int(count)

        # Prefer hard-error examples for that intent.
        pool = hard_by_intent.get(intent)

        if pool is not None and len(pool) > 0:
            pool = pool[["text", "intent"]].drop_duplicates()
        else:
            pool = base[
                base["intent"] == intent
            ][["text", "intent"]].drop_duplicates()

        if len(pool) == 0:
            continue

        rng = np.random.default_rng(
            RANDOM_SEED + abs(hash(intent)) % 100000
        )

        take_idx = rng.choice(
            len(pool),
            size=needed,
            replace=True,
        )

        parts.append(
            pool.iloc[take_idx].copy()
        )

    result = pd.concat(
        parts,
        ignore_index=True,
    )

    result = result.sample(
        frac=1.0,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    result.to_csv(
        OUT_CSV,
        index=False,
    )

    errors.to_csv(
        HARD_CSV,
        index=False,
    )

    final_counts = result["intent"].value_counts()

    summary = {
        "source_rows": int(len(base)),
        "source_intents": int(base["intent"].nunique()),
        "hard_error_rows_used": int(len(errors)),
        "hard_error_extra_copies": int(hard_added),
        "final_rows": int(len(result)),
        "min_class_target": MIN_CLASS_TARGET,
        "hard_repeat": HARD_REPEAT,
        "final_class_distribution": {
            str(k): int(v)
            for k, v in final_counts.items()
        },
        "important_rule": (
            "No synthetic text and no labels were changed. "
            "Only original rows were retained/oversampled."
        ),
    }

    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=" * 78)
    print("TARGETED V2 DATASET CREATED")
    print("=" * 78)
    print(f"Original rows       : {len(base)}")
    print(f"57-intent classes   : {base['intent'].nunique()}")
    print(f"Hard error rows     : {len(errors)}")
    print(f"Final training rows : {len(result)}")
    print()
    print("Saved:")
    print(OUT_CSV)
    print(HARD_CSV)
    print(SUMMARY_JSON)
    print()
    print("NO SYNTHETIC TEXT GENERATED.")
    print("NO LABELS CHANGED.")


if __name__ == "__main__":
    main()
