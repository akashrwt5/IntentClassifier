import os
import json
import pandas as pd
from pathlib import Path


def main():
    print("Scanning project for CSV files...")
    all_csv_files = Path(".").rglob("*.csv")

    # Load migration map to map device.xyz back to Cmd.Xyz
    try:
        with open("datasets/label_migration_map.json", "r") as f:
            migration_data = json.load(f)
            # Reverse map: value -> key (e.g. 'device.volume.increase' -> 'Cmd.VolumeIncrease')
            # But only keep ones that aren't null
            intent_map = {v: k for k, v in migration_data["map"].items() if v is not None}
        try:
            with open("models/semantic_student/en/labels.json", "r") as f:
                valid_labels = set(json.load(f))
        except:
            print("Warning: Could not load valid labels. Will not filter.")
            valid_labels = set()

    except Exception as e:
        print(f"Warning: Could not load label map: {e}")
        intent_map = {}
        valid_labels = set()

    combined_df = pd.DataFrame()
    processed_files = []
    skipped_files = []

    for file_path in all_csv_files:
        if file_path.name == "complete_csv.csv":
            continue

        if "eval" in file_path.parts:
            skipped_files.append(f"{file_path} (Skipped eval data)")
            continue

        lower_path = str(file_path).lower()
        if any(
            x in lower_path
            for x in [
                "danish",
                "french",
                "german",
                "spanish",
                "italian",
                "dutch",
                "fr_label",
                "da_label",
                "de_label",
                "multilingual",
                "language_packs",
                "/fr/",
                "/da/",
                "/de/",
                "/es/",
                "/it/",
                "/nl/",
                "/nl.csv",
                "/fr.csv",
                "/da.csv",
                "/de.csv",
                "/es.csv",
                "/it.csv",
                "/ja.csv",
                "/ko.csv",
                "/pt.csv",
                "/zh.csv",
            ]
        ):
            skipped_files.append(f"{file_path} (Skipped non-English)")
            continue

        try:
            df = pd.read_csv(file_path)

            if "text" in df.columns and "intent" in df.columns:
                # Keep only text and intent
                df = df[["text", "intent"]].dropna()

                # Apply intent mapping (e.g., device.volume.increase -> Cmd.VolumeIncrease)
                if intent_map:
                    df["intent"] = df["intent"].apply(lambda x: intent_map.get(x, x))

                # Also apply the user's specific request: anywhere "device." appears, change to "Cmd."
                df = df.replace("device\.", "Cmd.", regex=True)
                df = df.replace("Device\.", "Cmd.", regex=True)

                # Fix known typos in intents
                df["intent"] = df["intent"].replace(
                    {
                        "Help.MemoryOptions": "Help_MemoryOptions",
                        "Help.Reminder": "Help_Reminder",
                        "Help.RemoteProgramming": "Help_RemoteProgramming",
                        "Help.Transcribe": "Help_Transcribe",
                        "OOD": "Default Fallback Intent",
                    }
                )

                combined_df = pd.concat([combined_df, df], ignore_index=True)
                processed_files.append(str(file_path))
            else:
                skipped_files.append(str(file_path))
        except Exception as e:
            skipped_files.append(f"{file_path} (Error: {str(e)})")

    if combined_df.empty:
        print("No valid CSV files with 'text' and 'intent' columns found.")
        return

    print(f"\nFound {len(processed_files)} valid intent datasets.")
    print(f"Total rows before filtering/deduplication: {len(combined_df)}")

    # Strictly filter out invalid labels if valid_labels is loaded
    if valid_labels:
        invalid_count = len(combined_df[~combined_df["intent"].isin(valid_labels)])
        if invalid_count > 0:
            print(f"Dropping {invalid_count} rows with invalid/obsolete intent labels...")
            combined_df = combined_df[combined_df["intent"].isin(valid_labels)]

    # Drop exact duplicates
    combined_df = combined_df.drop_duplicates(subset=["text", "intent"])

    # Check for conflicts (same text, different intents)
    conflict_counts = combined_df["text"].value_counts()
    conflicting_texts = conflict_counts[conflict_counts > 1].index
    if not conflicting_texts.empty:
        print(
            f"Warning: Found {len(conflicting_texts)} texts with conflicting intents. Removing conflicts for safety..."
        )
        combined_df = combined_df.drop_duplicates(subset=["text"], keep=False)

    print(f"Total rows after deduplication and conflict resolution: {len(combined_df)}")

    # Remove data leak: filter out any texts present in eval datasets
    import re, unicodedata

    _TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")

    def _token_key(text):
        t = unicodedata.normalize("NFKD", str(text)).replace("’", "'")
        return " ".join(_TOKEN_RE.findall(t.lower()))

    eval_texts = set()
    for eval_file in Path("new_semantic/data/eval").glob("*.csv"):
        try:
            eval_df = pd.read_csv(eval_file)
            if "text" in eval_df.columns:
                eval_texts.update(eval_df["text"].dropna().apply(_token_key).tolist())
        except Exception:
            pass
    if eval_texts:
        before_leak = len(combined_df)
        combined_df = combined_df[~combined_df["text"].apply(_token_key).isin(eval_texts)]
        print(
            f"Removed {before_leak - len(combined_df)} rows that were in eval datasets (data leak)."
        )

    output_file = "complete_csv.csv"
    combined_df.to_csv(output_file, index=False)

    print(f"\n✅ Successfully saved combined data to: {output_file}")


if __name__ == "__main__":
    main()
