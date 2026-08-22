#!/usr/bin/env python3
"""
Google Colab Stage 2 Script (Contrastive Fine-Tuning)
=====================================================
Shatters overlapping vectors (like "louder" and "quieter") by forcing
the neural network to pull same-intent phrases together and aggressively
repel conflicting-intent phrases.

INSTRUCTIONS:
1. Run Stage 1 first to get `distilled_minilm_l3.zip`.
2. Unzip it and upload the folder to your Colab workspace.
3. Upload `train.csv`.
4. Run this script!
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import shutil

# --- Reproducibility ---
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# --- Configuration ---
INPUT_MODEL_DIR = "distilled_bge_small_l3"
OUTPUT_MODEL_DIR = "stage2_contrastive_bge_small_l3"
# --- Training data identity -------------------------------------------------
# Colab has no repository checkout: the file is uploaded by hand, so a path
# cannot pin it. Three files in this repository could be uploaded under the
# name "train.csv", and two of them leak 40% and 99% of holdout_honest.csv
# into training:
#
#     language_packs/en/train.csv       8,430 rows      0 holdout rows
#     new_semantic/data/en/train.csv   23,989 rows    585 holdout rows
#     complete_csv.csv                 31,699 rows  1,461 holdout rows
#
# A previous Stage-2 run left no record of which one it used, which is why the
# baseline it produced still carries an asterisk. The identity of the file is
# therefore pinned instead of its path, and the run refuses to start on a
# mismatch.
#
# Update these constants DELIBERATELY when the training data legitimately
# changes, and say why in the commit that changes them.
TRAIN_CSV_PATH = "train.csv"
EXPECTED_SOURCE = "language_packs/en/train.csv"
EXPECTED_SHA256 = "803a97d57a5397b1db522b18c982fe0a70092719265aabbe375830b55da6074b"
EXPECTED_ROWS = 8430


def verify_training_data(path=TRAIN_CSV_PATH):
    """Refuse to train on an unidentified file, and return the provenance block.

    Deliberately duplicated in both Colab scripts rather than imported: these
    files are pasted into a notebook cell and cannot rely on the rest of the
    directory being present.
    """
    import hashlib

    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Upload {EXPECTED_SOURCE} to Colab.")

    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    rows = sum(1 for _ in open(path, encoding="utf-8-sig")) - 1

    print("Training data")
    print("-" * 25)
    print(f"expected : {EXPECTED_SOURCE}  {EXPECTED_ROWS} rows")
    print(f"sha256   : {digest}")
    print(f"rows     : {rows}")

    if digest != EXPECTED_SHA256 or rows != EXPECTED_ROWS:
        raise SystemExit(
            "\nTRAINING DATA DOES NOT MATCH THE PINNED IDENTITY.\n"
            f"  expected sha256 {EXPECTED_SHA256} with {EXPECTED_ROWS} rows\n"
            f"  got      sha256 {digest} with {rows} rows\n\n"
            "Either the wrong file was uploaded, or the training data changed.\n"
            "If the change is intended, update EXPECTED_SHA256 and EXPECTED_ROWS\n"
            "in this script and record why. Do not bypass this check: an\n"
            "unidentified training file makes every number the run produces\n"
            "unverifiable after the fact."
        )

    print("identity : OK\n")
    return {
        "training_file": EXPECTED_SOURCE,
        "rows": rows,
        "sha256": digest,
    }


# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 3

print("GPU Available:", torch.cuda.is_available())

from torch.utils.data import Dataset, DataLoader, Sampler
from collections import defaultdict


class DynamicIntentDataset(Dataset):
    """
    Dynamically generates (Anchor, Positive) pairs on the fly every epoch.
    This provides much richer positive supervision than fixing pairs once.
    """

    def __init__(self, df):
        self.intent_to_texts = {}
        self.intent_to_id = {}

        valid_intents = [intent for intent, group in df.groupby("intent") if len(group) >= 2]
        self.singletons_skipped = len(df.groupby("intent")) - len(valid_intents)

        for idx, intent in enumerate(valid_intents):
            self.intent_to_texts[intent] = df[df["intent"] == intent]["text"].astype(str).tolist()
            self.intent_to_id[intent] = idx

        self.samples = []
        for intent, texts in self.intent_to_texts.items():
            for text in texts:
                self.samples.append((intent, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        intent, anchor = self.samples[idx]
        texts = self.intent_to_texts[intent]

        # Sample positive dynamically on the fly
        positive = random.choice(texts)
        attempts = 0
        while positive == anchor and attempts < 10:
            positive = random.choice(texts)
            attempts += 1

        intent_id = self.intent_to_id[intent]
        return InputExample(texts=[anchor, positive], label=intent_id)


class NoDuplicatesBatchSampler(Sampler):
    """
    Yields batches of indices where no two indices share the same intent label.
    This significantly reduces false-negatives within the same intent during MNRL training.

    This Sampler uses a "replenishing" strategy. When a small intent runs out of data,
    its pool is mathematically replenished. This ensures that large intents are fully trained
    and not prematurely truncated, while small intents receive beneficial oversampling.
    """

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.num_samples = len(dataset)

        # Cap the batch size to the number of unique intents to guarantee no duplicates
        self.unique_intents = list(dataset.intent_to_texts.keys())
        self.effective_batch_size = min(batch_size, len(self.unique_intents))

        self.label_to_indices_master = defaultdict(list)
        for idx, sample in enumerate(dataset.samples):
            intent = sample[0]
            self.label_to_indices_master[intent].append(idx)

        self.expected_batches = self.num_samples // self.effective_batch_size

    def __iter__(self):
        # Create a working copy of the pools
        working_pools = {k: list(v) for k, v in self.label_to_indices_master.items()}
        for k in working_pools:
            random.shuffle(working_pools[k])

        available_labels = list(working_pools.keys())

        for _ in range(self.expected_batches):
            batch = []
            random.shuffle(available_labels)
            labels_for_batch = available_labels[: self.effective_batch_size]

            for label in labels_for_batch:
                # If a small intent runs out, replenish its pool!
                if len(working_pools[label]) == 0:
                    working_pools[label] = list(self.label_to_indices_master[label])
                    random.shuffle(working_pools[label])

                batch.append(working_pools[label].pop())

            yield batch

    def __len__(self):
        return self.expected_batches


def run_sanity_eval(model_path):
    print("\nEvaluating Intent Accuracy (Sanity Check Only)...")
    print("NOTE: This is a lightweight evaluation to check for catastrophic failure.")
    print("Production evaluation should happen via the local downstream pipeline.")
    student = SentenceTransformer(model_path)

    df = pd.read_csv(TRAIN_CSV_PATH)
    texts = df["text"].astype(str).str.lower().str.strip().tolist()
    labels = df["intent"].astype(str).str.strip().tolist()

    with torch.inference_mode():
        embeddings = student.encode(texts, batch_size=256, show_progress_bar=True)

    X_tr, X_te, y_tr, y_te = train_test_split(
        embeddings, labels, test_size=0.15, stratify=labels, random_state=42
    )

    clf = LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)

    print(f"── Stage 2 Evaluation ──")
    print(f"  Overall accuracy: {acc:.3f}")
    print(f"  Macro-F1 score:   {f1:.3f}")



def write_provenance(provenance, output_dir):
    """Record what this run was trained on, beside what it produced.

    A previous Stage-2 run left no record of which of three candidate
    train.csv files it consumed, and two of them leak holdout rows into
    training. The baseline it produced still carries an asterisk because of
    it. This file is how that question stops being unanswerable.
    """
    import json

    record = dict(provenance)
    record["stage"] = Path(__file__).stem
    record["output_dir"] = str(output_dir)
    path = Path(output_dir) / "provenance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"Provenance written to {path}")


def main():
    provenance = verify_training_data()
    if not os.path.exists(TRAIN_CSV_PATH):
        print(f"ERROR: {TRAIN_CSV_PATH} not found. Upload it to Colab.")
        sys.exit(1)

    if not os.path.exists(INPUT_MODEL_DIR):
        print(f"ERROR: {INPUT_MODEL_DIR} not found. Upload your Stage 1 model folder to Colab.")
        sys.exit(1)

    print(f"\nLoading Stage 1 Model from {INPUT_MODEL_DIR}...")
    model = SentenceTransformer(INPUT_MODEL_DIR)

    df = pd.read_csv(TRAIN_CSV_PATH)

    print(f"\nInitializing Dynamic Dataset & Sampler from {len(df)} samples...")
    train_dataset = DynamicIntentDataset(df)

    print("\nDataset Statistics:")
    print("-" * 25)
    print(f"Total Intents        : {len(df['intent'].unique())}")
    print(f"Intents Used         : {len(train_dataset.intent_to_texts)}")
    print(f"Singletons Skipped   : {train_dataset.singletons_skipped} (No Stage 2 supervision)")

    total_utterances = len(df)
    used_utterances = len(train_dataset)
    coverage = (used_utterances / total_utterances) * 100 if total_utterances > 0 else 0
    print(f"Total Utterances Used: {used_utterances} / {total_utterances}")
    print(f"Coverage             : {coverage:.1f}%")

    intent_sizes = [len(v) for v in train_dataset.intent_to_texts.values()]
    if intent_sizes:
        print(f"Largest Intent       : {max(intent_sizes)} utterances")
        print(f"Smallest Intent      : {min(intent_sizes)} utterances")
        print(f"Avg Utterances       : {sum(intent_sizes) / len(intent_sizes):.1f}")
    print("-" * 25)

    # The custom batch sampler significantly reduces false-negatives in MNRL.
    batch_sampler = NoDuplicatesBatchSampler(train_dataset, batch_size=BATCH_SIZE)

    if batch_sampler.effective_batch_size < 8:
        print("\n⚠️  WARNING: Effective batch size is below 8 due to low unique intents.")
        print("   Contrastive learning (MNRL) benefits heavily from large batches.")
        print("   Consider adding more intents before running Stage 2.\n")
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler)

    # HACK: PyTorch DataLoaders with a batch_sampler set batch_size=None, and
    # newer versions of PyTorch actively forbid modifying .batch_size after initialization.
    # SentenceTransformers has a bug where it crashes trying to divide by None.
    # We bypass PyTorch's __setattr__ guardrail by modifying __dict__ directly!
    train_dataloader.__dict__["batch_size"] = batch_sampler.effective_batch_size

    # State-of-the-Art Metric Learning Loss (automatically mines hard negatives in the batch)
    # Note: MNRL ignores `label_id`. The label is ONLY used by our custom batch sampler.
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    dynamic_warmup = int(0.1 * len(train_dataloader) * EPOCHS)

    print(f"\nStarting Stage 2 Contrastive Training...")
    print("-" * 25)
    print(f"Target Batch Size : {BATCH_SIZE}")
    print(f"Actual Batch Size : {batch_sampler.effective_batch_size}")
    print(f"Epochs            : {EPOCHS}")
    print(f"Total Steps       : {len(train_dataloader) * EPOCHS}")
    print(f"Warmup Steps      : {dynamic_warmup}")
    print("-" * 25)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=dynamic_warmup,
        output_path=OUTPUT_MODEL_DIR,
        show_progress_bar=True,
    )

    print("\n✅ Stage 2 Training Complete!")
    write_provenance(provenance, OUTPUT_MODEL_DIR)
    run_sanity_eval(OUTPUT_MODEL_DIR)

    print("Zipping the final output for you to download...")
    shutil.make_archive(OUTPUT_MODEL_DIR, "zip", OUTPUT_MODEL_DIR)
    print(f"🎉 Done! Download {OUTPUT_MODEL_DIR}.zip and run the ONNX export script on it!")


if __name__ == "__main__":
    main()
