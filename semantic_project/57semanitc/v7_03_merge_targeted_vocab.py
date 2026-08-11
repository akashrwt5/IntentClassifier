cat > ~/Downloads/v7_03_merge_targeted_vocab.py <<'PY'
#!/usr/bin/env python3

from pathlib import Path
import json
import pandas as pd

PROJECT = Path("/Users/shuklam/IntentClassifier/semantic_project/57semanitc")

TRAIN_CSV = PROJECT / "train.csv"
REVIEW_CSV = PROJECT / "v7_02_targeted_vocabulary_review" / "review_required.csv"
OUT_DIR = PROJECT / "v7_03_targeted_training_set"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def clean(x):
    return " ".join(str(x).strip().split())


def find_col(df, candidates):
    for col in df.columns:
        if str(col).lower() in candidates:
            return col
    return None


def main():

    print("=" * 78)
    print("V7-03 TARGETED VOCABULARY MERGE")
    print("=" * 78)

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"train.csv not found:\n{TRAIN_CSV}"
        )

    if not REVIEW_CSV.exists():
        raise FileNotFoundError(
            f"review_required.csv not found:\n{REVIEW_CSV}"
        )

    train = pd.read_csv(TRAIN_CSV)
    review = pd.read_csv(REVIEW_CSV)

    train_text = find_col(
        train,
        {"text", "utterance", "query", "sentence"}
    )

    train_intent = find_col(
        train,
        {"intent", "label", "true_intent"}
    )

    review_text = find_col(
        review,
        {"text", "utterance", "query", "sentence"}
    )

    review_intent = find_col(
        review,
        {"intent", "label", "true_intent"}
    )

    if train_text is None or train_intent is None:
        raise RuntimeError(
            f"Could not identify train columns: {list(train.columns)}"
        )

    if review_text is None or review_intent is None:
        raise RuntimeError(
            f"Could not identify review columns: {list(review.columns)}"
        )

    base = pd.DataFrame({
        "text": train[train_text].map(clean),
        "intent": train[train_intent].map(clean),
    })

    candidates = pd.DataFrame({
        "text": review[review_text].map(clean),
        "intent": review[review_intent].map(clean),
    })

    base = base[
        (base["text"] != "") &
        (base["intent"] != "")
    ].copy()

    candidates = candidates[
        (candidates["text"] != "") &
        (candidates["intent"] != "")
    ].copy()

    existing_texts = set(
        base["text"].str.lower()
    )

    candidates["already_in_train"] = (
        candidates["text"]
        .str.lower()
        .isin(existing_texts)
    )

    candidates = candidates[
        ~candidates["already_in_train"]
    ].copy()

    candidates = candidates.drop_duplicates(
        subset=["text"]
    )

    unknown_intents = sorted(
        set(candidates["intent"])
        - set(base["intent"])
    )

    if unknown_intents:
        raise RuntimeError(
            "Unknown intents found:\n"
            + "\n".join(unknown_intents)
        )

    before_counts = (
        base["intent"]
        .value_counts()
        .rename("before_count")
        .to_frame()
    )

    additions = (
        candidates["intent"]
        .value_counts()
        .rename("added_count")
        .to_frame()
    )

    final = pd.concat(
        [
            base,
            candidates[["text", "intent"]]
        ],
        ignore_index=True
    )

    final = final.drop_duplicates(
        subset=["text"]
    ).reset_index(drop=True)

    after_counts = (
        final["intent"]
        .value_counts()
        .rename("after_count")
        .to_frame()
    )

    distribution = (
        before_counts
        .join(additions, how="outer")
        .join(after_counts, how="outer")
        .fillna(0)
    )

    for col in [
        "before_count",
        "added_count",
        "after_count"
    ]:
        distribution[col] = (
            distribution[col]
            .astype(int)
        )

    distribution["percent_change"] = (
        (
            distribution["after_count"]
            - distribution["before_count"]
        )
        / distribution["before_count"]
        * 100.0
    )

    distribution = (
        distribution
        .reset_index()
        .rename(columns={"index": "intent"})
        .sort_values(
            "added_count",
            ascending=False
        )
    )

    final_csv = (
        OUT_DIR /
        "train_v7_targeted.csv"
    )

    final.to_csv(
        final_csv,
        index=False
    )

    distribution.to_csv(
        OUT_DIR /
        "intent_distribution_v7.csv",
        index=False
    )

    candidates.to_csv(
        OUT_DIR /
        "added_targeted_examples.csv",
        index=False
    )

    summary = {
        "original_training_rows": int(len(base)),
        "review_candidates_input": int(len(review)),
        "duplicate_candidates_removed": int(
            len(review) - len(candidates)
        ),
        "added_targeted_rows": int(len(candidates)),
        "final_training_rows": int(len(final)),
        "original_intents": int(
            base["intent"].nunique()
        ),
        "final_intents": int(
            final["intent"].nunique()
        ),
        "locked_test_read": False,
        "synthetic_text_generated": False,
        "model_modified": False,
        "training_performed": False
    }

    (
        OUT_DIR /
        "v7_merge_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2
        ),
        encoding="utf-8"
    )

    print()
    print(
        f"Original training rows : {len(base)}"
    )

    print(
        f"Review candidates input: {len(review)}"
    )

    print(
        f"Duplicate candidates   : "
        f"{len(review) - len(candidates)}"
    )

    print(
        f"Added targeted rows    : {len(candidates)}"
    )

    print(
        f"Final training rows    : {len(final)}"
    )

    print()
    print(
        "--- PER-INTENT ADDITIONS ---"
    )

    print(
        distribution[
            [
                "intent",
                "before_count",
                "added_count",
                "after_count",
                "percent_change"
            ]
        ].to_string(index=False)
    )

    print()
    print("Saved:")
    print(final_csv)
    print(
        OUT_DIR /
        "intent_distribution_v7.csv"
    )
    print(
        OUT_DIR /
        "added_targeted_examples.csv"
    )
    print(
        OUT_DIR /
        "v7_merge_summary.json"
    )

    print()
    print("IMPORTANT:")
    print("No model training performed.")
    print("V6 model was NOT modified.")
    print("Locked 1686-row test was NOT read.")
    print("No synthetic text was generated.")

    print()
    print(
        "STATUS: V7-03 TRAINING SET READY"
    )


if __name__ == "__main__":
    main()
PY
