#!/usr/bin/env python3
"""
CREATE A PROPER LOCKED 57-INTENT HOLDOUT

IMPORTANT:
The existing 57-intent V3/V2 checkpoints were trained using the original
8430-row train.csv. Therefore, an evaluation split made NOW from that same
CSV is NOT an unseen benchmark for those already-trained checkpoints.

This script creates a permanent, stratified 20% holdout and a reference
80% training set. To obtain a scientifically valid V3-vs-V2 comparison,
BOTH models must subsequently be retrained using ONLY reference_train.csv.

This script performs NO training and does not touch the locked 11-intent
595 CSV.

Outputs:
  v3_57intent_locked_eval/
    reference_train.csv
    locked_test_57intent.csv
    split_manifest.json
    test_sha256.txt

Run:
  python3 create_locked_57intent_split.py
"""

from pathlib import Path
import hashlib
import json
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "train.csv"

OUT = ROOT / "v3_57intent_locked_eval"
OUT.mkdir(exist_ok=True)

REFERENCE = OUT / "reference_train.csv"
LOCKED_TEST = OUT / "locked_test_57intent.csv"
MANIFEST = OUT / "split_manifest.json"
SHA_FILE = OUT / "test_sha256.txt"

SEED = 20260809
TEST_SIZE = 0.20
EXPECTED_INTENTS = 57


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_columns(columns):
    lower = {str(c).strip().lower(): c for c in columns}
    text = next(
        (lower[x] for x in ["text", "utterance", "query", "sentence", "input"]
         if x in lower), None
    )
    label = next(
        (lower[x] for x in ["intent", "label", "category", "class"]
         if x in lower), None
    )
    if text is None or label is None:
        raise RuntimeError(f"Cannot identify columns: {list(columns)}")
    return text, label


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing source CSV:\n{SOURCE}")

    df0 = pd.read_csv(SOURCE)
    text_col, label_col = detect_columns(df0.columns)

    df = df0[[text_col, label_col]].copy()
    df.columns = ["text", "intent"]

    df["text"] = df["text"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()

    before = len(df)

    df = df[
        (df["text"] != "") &
        (df["intent"] != "")
    ].drop_duplicates(
        ["text", "intent"]
    ).reset_index(drop=True)

    if df["intent"].nunique() != EXPECTED_INTENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_INTENTS} intents, "
            f"got {df['intent'].nunique()}"
        )

    # Every class must have enough rows for a stratified holdout.
    counts = df["intent"].value_counts()
    too_small = counts[counts < 2]
    if len(too_small):
        raise RuntimeError(
            f"Cannot stratify classes with <2 rows:\n{too_small}"
        )

    reference, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["intent"],
    )

    reference = reference.sample(
        frac=1.0,
        random_state=SEED,
    ).reset_index(drop=True)

    test = test.sample(
        frac=1.0,
        random_state=SEED,
    ).reset_index(drop=True)

    # Safety: no exact text overlap.
    overlap = set(reference["text"]) & set(test["text"])
    if overlap:
        raise RuntimeError(
            f"Text leakage detected: {len(overlap)} overlapping texts."
        )

    reference.to_csv(REFERENCE, index=False)
    test.to_csv(LOCKED_TEST, index=False)

    test_hash = sha256_file(LOCKED_TEST)
    source_hash = sha256_file(SOURCE)

    manifest = {
        "source_csv": str(SOURCE.resolve()),
        "source_sha256": source_hash,
        "source_rows_before_cleaning": before,
        "source_rows_after_cleaning": len(df),
        "intents": int(df["intent"].nunique()),
        "seed": SEED,
        "test_size": TEST_SIZE,
        "reference_rows": len(reference),
        "locked_test_rows": len(test),
        "locked_test_sha256": test_hash,
        "reference_sha256": sha256_file(REFERENCE),
        "text_overlap_between_reference_and_test": 0,
        "class_distribution_reference": {
            str(k): int(v)
            for k, v in reference["intent"].value_counts().sort_index().items()
        },
        "class_distribution_test": {
            str(k): int(v)
            for k, v in test["intent"].value_counts().sort_index().items()
        },
        "IMPORTANT": (
            "Existing checkpoints trained on the full original train.csv "
            "must NOT be evaluated on this holdout as an unseen test. "
            "Retrain both baseline V3 and targeted V2 using reference_train.csv only."
        ),
    }

    MANIFEST.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    SHA_FILE.write_text(
        test_hash + "  locked_test_57intent.csv\n",
        encoding="utf-8",
    )

    print("=" * 78)
    print("LOCKED 57-INTENT HOLDOUT CREATED")
    print("=" * 78)
    print(f"Source rows          : {len(df)}")
    print(f"Intents              : {df['intent'].nunique()}")
    print(f"Reference train rows : {len(reference)}")
    print(f"Locked test rows     : {len(test)}")
    print()
    print("Locked test SHA256:")
    print(test_hash)
    print()
    print("Saved:")
    print(REFERENCE)
    print(LOCKED_TEST)
    print(MANIFEST)
    print(SHA_FILE)
    print()
    print("NEXT:")
    print("1. Retrain BASE V3 using reference_train.csv ONLY.")
    print("2. Build targeted V2 from reference_train.csv + training-only errors.")
    print("3. Retrain V2 using reference_train.csv ONLY.")
    print("4. Benchmark both on locked_test_57intent.csv.")
    print()
    print("DO NOT benchmark the existing full-data checkpoints as 'unseen'.")


if __name__ == "__main__":
    main()
