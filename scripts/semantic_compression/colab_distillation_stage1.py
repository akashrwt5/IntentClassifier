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
TRAIN_CSV_PATH = "train.csv"
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


def run_head_retraining_and_eval(student_path):
    """Re-fits LogReg head using the exact production strategy and evaluates."""
    print("\nEvaluating Intent Accuracy (Head Retraining)...")
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

    print(f"── Student Downstream Evaluation ──")
    print(f"  Overall accuracy: {acc:.3f}")
    print(f"  Macro-F1 score:   {f1:.3f}")


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


def main():
    corpus = build_mixed_corpus()
    teacher, student = create_layer_dropped_student()

    run_distillation(teacher, student, corpus)

    run_head_retraining_and_eval(OUTPUT_DIR)

    prepare_for_vocab_pruning(OUTPUT_DIR)

    shutil.make_archive(OUTPUT_DIR, "zip", OUTPUT_DIR)
    print(f"\n🎉 Pipeline Finished! You can now download {OUTPUT_DIR}.zip")


if __name__ == "__main__":
    main()
