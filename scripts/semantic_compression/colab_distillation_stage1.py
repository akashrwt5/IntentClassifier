#!/usr/bin/env python3
"""
Google Colab Distillation Script (Stage 1 Pipeline)
===================================================
NOTE: This script is responsible ONLY for Stage 1 semantic recovery (knowledge distillation).
Production classifier training, calibration, vocabulary pruning, ONNX export, and quantization
are intentionally handled by downstream local tooling on the MacBook.

INSTRUCTIONS FOR GOOGLE COLAB:
1. Open Google Colab (colab.research.google.com)
2. Create a new notebook and select Runtime -> Change runtime type -> T4 GPU.
3. In the first cell, run:
   !pip install sentence-transformers datasets pandas scikit-learn transformers torch
4. Upload your `train.csv` and `semantic_oos_2.csv` to the Colab files pane.
5. Paste this entire script into a new cell and run it!
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
import shutil

# --- Reproducibility ---
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# --- Configuration ---
TEACHER_MODEL_NAME = "BAAI/bge-small-en-v1.5"
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


OOS_CSV_PATH = "language_packs/en/extras/oos_2.csv"
OUTPUT_DIR = "distilled_bge_small_l3"

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 1
MAX_SEQ_LENGTH = 64
TOTAL_DISTILLATION_SAMPLES = 500_000

print("gpu available:", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"


def build_mixed_corpus():
    """Builds the 60/20/15/5 distillation corpus."""
    print("Building Mixed Distillation Corpus...")
    corpus = []

    # 1. 60% Domain Data (train.csv)
    # Verify dataset schema
    if os.path.exists(TRAIN_CSV_PATH):
        df = pd.read_csv(TRAIN_CSV_PATH)
        if "text" not in df.columns or "intent" not in df.columns:
            raise ValueError(f"ERROR: {TRAIN_CSV_PATH} must contain 'text' and 'intent' columns.")
        domain_texts = df["text"].astype(str).tolist()
        # Oversample to hit 60%
        target_domain = int(TOTAL_DISTILLATION_SAMPLES * 0.60)
        corpus.extend(random.choices(domain_texts, k=target_domain))
        domain_count = target_domain
        print(f" -> Added {target_domain} in-domain samples")
    else:
        domain_count = 0
        print(f"WARNING: {TRAIN_CSV_PATH} not found. Skipping 60% domain data.")

    # 2. 20% Oversampled Domain Data (Placeholder for True Synthetic Data)
    target_synth = int(TOTAL_DISTILLATION_SAMPLES * 0.20)
    synth_count = 0
    if os.path.exists(TRAIN_CSV_PATH):
        corpus.extend(random.choices(domain_texts, k=target_synth))
        synth_count = target_synth
        print(f" -> Added {target_synth} oversampled domain samples")

    # 3. 15% SNLI/MultiNLI
    target_snli = int(TOTAL_DISTILLATION_SAMPLES * 0.15)
    snli_count = 0
    try:
        snli = load_dataset("stanfordnlp/snli", split="train")
        snli_texts = snli["premise"][:target_snli]
        corpus.extend(snli_texts)
        snli_count = len(snli_texts)
        print(f" -> Added {len(snli_texts)} SNLI samples")
    except Exception as e:
        print(f"WARNING: Could not load SNLI: {e}")

    # 4. 5% Wikipedia
    target_wiki = int(TOTAL_DISTILLATION_SAMPLES * 0.05)
    wiki_count = 0
    try:
        wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        wiki_texts = []
        for row in wiki:
            # Just take the first ~150 chars of the article to avoid fragile splitting
            sentence = row["text"][:150].strip()
            wiki_texts.append(sentence)
            if len(wiki_texts) >= target_wiki:
                break
        corpus.extend(wiki_texts)
        wiki_count = len(wiki_texts)
        print(f" -> Added {len(wiki_texts)} Wikipedia samples")
    except Exception as e:
        print(f"WARNING: Could not load Wikipedia: {e}")

    random.shuffle(corpus)

    # Print Final Corpus Composition
    print("\nFinal Corpus Composition")
    print("-" * 25)
    print(f"Domain      : {domain_count}")
    print(f"Oversampled : {synth_count}")
    print(f"SNLI        : {snli_count}")
    print(f"Wikipedia   : {wiki_count}")
    print("-" * 25)
    print(f"Total       : {len(corpus)}")
    if len(corpus) < TOTAL_DISTILLATION_SAMPLES:
        print(
            "WARNING: Final corpus size is less than intended (external datasets may have failed to load)."
        )

    return corpus


def create_layer_dropped_student():
    """Initializes the Student by dropping the last 9 layers of the Teacher."""
    print("\nInitializing Layer-Dropped Student (L12 -> L3)...")
    teacher = SentenceTransformer(TEACHER_MODEL_NAME)

    # 1. Initialize an independent copy
    student_base = SentenceTransformer(TEACHER_MODEL_NAME)
    auto_model = student_base._first_module().auto_model
    tokenizer = student_base._first_module().tokenizer

    # 2. Surgically chop it down to 3 layers
    layers_to_keep = 3
    auto_model.encoder.layer = nn.ModuleList(
        [auto_model.encoder.layer[i] for i in range(layers_to_keep)]
    )
    auto_model.config.num_hidden_layers = layers_to_keep

    # 3. Save the chopped model to disk (bulletproof way to initialize a fresh Transformer)
    tmp_dir = "tmp_chopped_base"
    auto_model.save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)

    # 4. Load the cleanly chopped model back as a fresh SentenceTransformers module
    word_embedding_model = models.Transformer(tmp_dir, max_seq_length=MAX_SEQ_LENGTH)

    modules_to_keep = [word_embedding_model]
    for idx, module in enumerate(teacher):
        if idx > 0:
            modules_to_keep.append(module)

    student = SentenceTransformer(modules=modules_to_keep)

    # Sanity Checks (with fallback for v3.0 syntax)
    get_dim = getattr(student, "get_embedding_dimension", student.get_sentence_embedding_dimension)
    get_dim_teacher = getattr(
        teacher, "get_embedding_dimension", teacher.get_sentence_embedding_dimension
    )

    actual_dim = get_dim()
    expected_dim = get_dim_teacher()
    assert (
        actual_dim == expected_dim
    ), f"Dimension mismatch! Expected {expected_dim}, got {actual_dim}"

    actual_layers = len(student._first_module().auto_model.encoder.layer)
    assert (
        actual_layers == layers_to_keep
    ), f"Layer mismatch! Expected {layers_to_keep}, got {actual_layers}"

    print(f"Student created successfully with {layers_to_keep} layers and {actual_dim} dimensions.")
    return teacher, student


def run_distillation(teacher, student, corpus):
    """Runs the MSE Distillation Loop by pre-computing teacher embeddings."""
    from sentence_transformers import InputExample

    print("\nStarting Knowledge Distillation (GPU recommended)...")

    print("1. Pre-computing Teacher Embeddings (This takes a few minutes)...")
    teacher_embeddings = teacher.encode(corpus, batch_size=256, show_progress_bar=True)

    print("2. Preparing Dataset for Student...")
    train_examples = []
    for text, emb in zip(corpus, teacher_embeddings):
        # We pass the teacher's embedding as the ground-truth label for MSELoss
        train_examples.append(InputExample(texts=[text], label=emb))

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

    # MSELoss will automatically compute MSE between Student's output and the provided label
    train_loss = losses.MSELoss(model=student)

    print("3. Training Student...")
    dynamic_warmup = int(0.1 * len(train_dataloader) * EPOCHS)

    student.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=dynamic_warmup,
        output_path=OUTPUT_DIR,
        optimizer_params={"lr": 1e-4},
        show_progress_bar=True,
    )
    print("✅ Distillation Complete!")


def run_memorisation_check(student_path):
    """Re-fit a LogReg head on a random split of train.csv -- a memorisation check.

    Renamed from run_head_retraining_and_eval. The old name invited its output to
    be read as downstream accuracy, and it was: figures from this function reached
    the plan of record as quality claims. A random split of a corpus that is 44.7%
    near-duplicate within itself puts paraphrases of the same sentence on both
    sides, so a high score here is the expected result for any encoder that has not
    catastrophically failed. Its only legitimate use is detecting that failure.
    """
    print("\n=== MEMORISATION CHECK (not an accuracy result) ===")
    print("A random 85/15 split of train.csv. 44.7% of this corpus is near-duplicated")
    print("within itself, so paraphrases of the same utterance land on both sides of")
    print("the split. The number below says the encoder can retrieve what it was shown;")
    print("it says nothing about generalisation and MUST NOT be quoted as accuracy.")
    print("The generalisation figure comes from dev_hard.csv, scored locally.")
    student = SentenceTransformer(student_path)

    if not os.path.exists(TRAIN_CSV_PATH):
        print("No train.csv found, skipping evaluation.")
        return

    df = pd.read_csv(TRAIN_CSV_PATH)
    texts = df["text"].astype(str).str.lower().str.strip().tolist()
    labels = df["intent"].astype(str).str.strip().tolist()

    print("Embedding train.csv through new Distilled Student...")
    embeddings = student.encode(texts, batch_size=256, show_progress_bar=True)

    X_tr, X_te, y_tr, y_te = train_test_split(
        embeddings, labels, test_size=0.15, stratify=labels, random_state=42
    )

    clf = LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced")
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)

    print("── Memorisation check (NOT accuracy) ──")
    print(f"  retrieval on seen-paraphrase split: {acc:.3f}")
    print(f"  macro-F1 on the same split:         {f1:.3f}")
    print("  interpretation: >0.90 means the encoder trained; nothing more.")


def prepare_for_vocab_pruning(student_path):
    """Post-distillation vocabulary pruning placeholder as a deployment optimization."""
    print("\nPost-Distillation Vocabulary Pruning...")
    # Extract unique words from train.csv to retain
    retained_tokens = set()
    if os.path.exists(TRAIN_CSV_PATH):
        df = pd.read_csv(TRAIN_CSV_PATH)
        for text in df["text"]:
            words = str(text).lower().split()
            retained_tokens.update(words)

    if not retained_tokens:
        print("No tokens to retain, skipping pruning.")
        return

    print(f"Retaining {len(retained_tokens)} unique tokens from domain data.")
    # In a full run, we would rewrite the tokenizer.json and slice the embedding matrix here.
    # For this script, we just demonstrate the pipeline structure.
    print("Pruning candidates identified. Actual matrix slicing happens downstream.")



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
    corpus = build_mixed_corpus()
    teacher, student = create_layer_dropped_student()

    run_distillation(teacher, student, corpus)

    write_provenance(provenance, OUTPUT_DIR)

    run_memorisation_check(OUTPUT_DIR)

    prepare_for_vocab_pruning(OUTPUT_DIR)

    shutil.make_archive(OUTPUT_DIR, "zip", OUTPUT_DIR)
    print(f"\n🎉 Pipeline Finished! You can now download {OUTPUT_DIR}.zip")


if __name__ == "__main__":
    main()
