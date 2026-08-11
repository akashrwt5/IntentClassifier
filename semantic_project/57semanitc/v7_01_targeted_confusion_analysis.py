#!/usr/bin/env python3

from pathlib import Path
import json
import pandas as pd

PROJECT = Path(
    "/Users/shuklam/IntentClassifier/semantic_project/57semanitc"
)

SMOKE_DIR = PROJECT / "v6_production_smoke_test"
ERROR_FILE = SMOKE_DIR / "smoke_errors.csv"

OUT_DIR = PROJECT / "v7_01_targeted_confusion_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def clean(x):
    return " ".join(str(x).strip().split())


def main():
    print("=" * 78)
    print("V7-01 TARGETED CONFUSION ANALYSIS")
    print("=" * 78)

    if not ERROR_FILE.exists():
        raise FileNotFoundError(
            f"Smoke error file not found:\n{ERROR_FILE}"
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
            f"Missing columns: {sorted(missing)}"
        )

    df["text"] = df["text"].map(clean)
    df["intent"] = df["intent"].map(clean)
    df["prediction"] = df["prediction"].map(clean)

    pairs = (
        df.groupby(["intent", "prediction"])
        .size()
        .reset_index(name="error_count")
        .sort_values("error_count", ascending=False)
    )

    pairs.to_csv(
        OUT_DIR / "targeted_confusion_pairs.csv",
        index=False,
    )

    targeted = df[
        ["text", "intent", "prediction", "confidence"]
    ].copy()

    targeted["word_count"] = (
        targeted["text"].str.split().str.len()
    )

    targeted = targeted.sort_values(
        ["intent", "prediction", "confidence"]
    )

    targeted.to_csv(
        OUT_DIR / "targeted_error_examples.csv",
        index=False,
    )

    by_true = (
        df.groupby("intent")
        .size()
        .reset_index(name="error_count")
        .sort_values("error_count", ascending=False)
    )

    by_true.to_csv(
        OUT_DIR / "intent_error_counts.csv",
        index=False,
    )

    clusters = []

    for _, row in pairs.iterrows():
        true_intent = row["intent"]
        predicted = row["prediction"]
        count = int(row["error_count"])

        examples = df[
            (df["intent"] == true_intent)
            & (df["prediction"] == predicted)
        ][["text", "confidence"]].head(10)

        clusters.append({
            "true_intent": true_intent,
            "predicted_intent": predicted,
            "error_count": count,
            "examples": examples.to_dict(orient="records"),
        })

    (OUT_DIR / "confusion_clusters.json").write_text(
        json.dumps(clusters, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Total smoke errors : {len(df)}")

    print()
    print("--- TOP CONFUSION PAIRS ---")
    print(pairs.head(20).to_string(index=False))

    print()
    print("--- INTENTS WITH MOST ERRORS ---")
    print(by_true.head(20).to_string(index=False))

    print()
    print("--- TARGETED CLUSTERS ---")

    for cluster in clusters[:15]:
        print()
        print(
            f"{cluster['true_intent']}  →  "
            f"{cluster['predicted_intent']}  "
            f"({cluster['error_count']} errors)"
        )

        for example in cluster["examples"][:5]:
            print(
                f"   {float(example['confidence']):.4f}"
                f" | {example['text']}"
            )

    print()
    print("Saved:")
    print(OUT_DIR / "targeted_confusion_pairs.csv")
    print(OUT_DIR / "targeted_error_examples.csv")
    print(OUT_DIR / "intent_error_counts.csv")
    print(OUT_DIR / "confusion_clusters.json")

    print()
    print("IMPORTANT:")
    print("V6 model was NOT modified.")
    print("No training was performed.")
    print("Locked 1686-row test was NOT read.")

    print()
    print("STATUS: V7-01 ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
