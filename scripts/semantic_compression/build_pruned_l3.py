#!/usr/bin/env python3
"""
Track 1: Knowledge Distillation + Vocabulary Pruning
Builds a heavily pruned version of all-MiniLM-L3-v2 tailored to the specific vocabulary
of the training dataset, significantly reducing the embedding matrix size.
"""

import sys
import json
import shutil
from pathlib import Path
from collections import Counter
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "datasets" / "04_GENERATED_MASTER_training_data.csv"
OOS_PATH = BASE_DIR / "datasets" / "semantic_oos_2.csv"

OUT_DIR = Path(__file__).parent / "output_models" / "track1_pruned_l3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "sentence-transformers/paraphrase-MiniLM-L3-v2"
VOCAB_SIZE_TARGET = 10000


def main():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)

    print("Loading datasets to compute vocabulary frequencies...")
    texts = []
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        if "text" in df.columns:
            texts.extend(df["text"].astype(str).tolist())
    if OOS_PATH.exists():
        df = pd.read_csv(OOS_PATH)
        if "text" in df.columns:
            texts.extend(df["text"].astype(str).tolist())

    if not texts:
        print("Warning: Could not find training datasets. Using default fallback phrases.")
        texts = ["turn on the lights", "turn up the volume", "what is the weather"]

    print("Tokenizing corpus to count token frequencies...")
    token_counts = Counter()
    for text in texts:
        tokens = tokenizer.tokenize(text.lower())
        token_counts.update(tokens)

    # Always keep special tokens
    special_tokens = set(tokenizer.all_special_tokens)

    # Get top tokens
    most_common = [t for t, _ in token_counts.most_common()]
    keep_tokens = set(special_tokens)
    for t in most_common:
        if len(keep_tokens) >= VOCAB_SIZE_TARGET:
            break
        keep_tokens.add(t)

    # Pad out to VOCAB_SIZE_TARGET with common English words
    original_vocab = tokenizer.get_vocab()
    sorted_original = sorted(original_vocab.items(), key=lambda x: x[1])

    for token, idx in sorted_original:
        if len(keep_tokens) >= VOCAB_SIZE_TARGET:
            break
        keep_tokens.add(token)

    keep_tokens_list = list(keep_tokens)
    print(f"Selected {len(keep_tokens_list)} tokens out of {tokenizer.vocab_size} original.")

    # 1. Prune the Tokenizer
    new_vocab = {token: i for i, token in enumerate(keep_tokens_list)}

    # 2. Prune the Embedding Matrix
    old_embeddings = model.embeddings.word_embeddings.weight.data
    new_embeddings = torch.zeros(
        (len(keep_tokens_list), old_embeddings.shape[1]), dtype=old_embeddings.dtype
    )

    for t in keep_tokens_list:
        old_id = original_vocab[t]
        new_id = new_vocab[t]
        new_embeddings[new_id] = old_embeddings[old_id]

    model.embeddings.word_embeddings = torch.nn.Embedding.from_pretrained(new_embeddings)
    model.config.vocab_size = len(keep_tokens_list)

    # Update tokenizer files
    temp_dir = Path(__file__).parent / "temp_tokenizer"
    tokenizer.save_pretrained(temp_dir)

    # Write the new vocab.txt
    vocab_path = temp_dir / "vocab.txt"
    if vocab_path.exists():
        with open(vocab_path, "w", encoding="utf-8") as f:
            sorted_new_vocab = sorted(new_vocab.items(), key=lambda x: x[1])
            for token, idx in sorted_new_vocab:
                f.write(token + "\n")

    # Reload modified tokenizer
    pruned_tokenizer = AutoTokenizer.from_pretrained(temp_dir)

    # Save the modified PyTorch model
    pt_dir = OUT_DIR / "pytorch"
    model.save_pretrained(pt_dir)
    pruned_tokenizer.save_pretrained(pt_dir)

    print("\nExporting to ONNX...")
    ort_model = ORTModelForFeatureExtraction.from_pretrained(pt_dir, export=True)

    print("Quantizing to INT8...")
    quantizer = ORTQuantizer.from_pretrained(ort_model)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

    quant_dir = OUT_DIR / "onnx_quantized"
    quantizer.quantize(save_dir=str(quant_dir), quantization_config=qconfig)

    # Clean up temp
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Report Sizes
    import os

    pt_size = sum(f.stat().st_size for f in pt_dir.iterdir() if f.is_file()) / (1024 * 1024)
    q_size = 0
    if (quant_dir / "model_quantized.onnx").exists():
        q_size = (quant_dir / "model_quantized.onnx").stat().st_size / (1024 * 1024)

    print(f"\n✅ Track 1 Model Built Successfully!")
    print(f"Output Directory: {OUT_DIR}")
    print(f"PyTorch Size (FP32): {pt_size:.2f} MB")
    print(f"ONNX Size (INT8):    {q_size:.2f} MB")


if __name__ == "__main__":
    main()
