#!/usr/bin/env python3
"""
Build the semantic nearest-neighbour index from training data.

Embeds every training phrase in data/intent_data_new.csv using MiniLM-L6-v2
and stores the result as models/semantic_index.npz (float16, ~5.7 MB).

Run this once after training, or whenever the training data changes:
    python scripts/build_semantic_index.py

Requirements:
    pip install sentence-transformers
    (model is downloaded from HuggingFace on first run ~90 MB)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR   = Path(__file__).parent.parent
DATA_PATH  = BASE_DIR / "data" / "intent_data_new.csv"
MODEL_DIR  = BASE_DIR / "models"
INDEX_PATH = MODEL_DIR / "semantic_index.npz"
ONNX_PATH  = MODEL_DIR / "minilm-l6-v2.onnx"
VOCAB_PATH = MODEL_DIR / "minilm-vocab.txt"


def load_data():
    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    data.columns = [c.strip().lower() for c in data.columns]
    data["text"]   = data["text"].astype(str).str.lower().str.strip()
    data["intent"] = data["intent"].astype(str).str.strip()
    data = data.dropna().drop_duplicates(subset=["text", "intent"])
    # Exclude Default Fallback Intent — semantically noisy, should stay as GenAI
    data = data[data["intent"] != "Default Fallback Intent"]
    print(f"Phrases to embed: {len(data)}  (across {data['intent'].nunique()} intents)")
    return data


def embed_with_sentence_transformers(texts: list[str]) -> np.ndarray:
    """
    Use sentence-transformers to embed in batches.
    Falls back to ONNX model if sentence-transformers is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        print("Loading all-MiniLM-L6-v2 via sentence-transformers...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"Embedding {len(texts)} phrases in batches of 256...")
        embeddings = model.encode(
            texts,
            batch_size=256,
            show_progress_bar=True,
            normalize_embeddings=True,    # L2 normalise — match runtime behaviour
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)
    except ImportError:
        print("sentence-transformers not installed — trying ONNX path...")
        return embed_with_onnx(texts)


def embed_with_onnx(texts: list[str]) -> np.ndarray:
    """Embed using the ONNX model directly (no sentence-transformers needed)."""
    if not ONNX_PATH.exists():
        print(f"\nERROR: ONNX model not found at {ONNX_PATH}")
        print("Please either:")
        print("  1. pip install sentence-transformers  (recommended)")
        print("  2. Download the ONNX model manually — see scripts/download_minilm.py")
        sys.exit(1)

    sys.path.insert(0, str(BASE_DIR / "scripts"))
    from nlu.semantic import SemanticFallback
    sf = SemanticFallback.__new__(SemanticFallback)
    import onnxruntime as ort
    sf.session   = ort.InferenceSession(str(ONNX_PATH))
    sf.threshold = 0.82
    sf._vocab_cache = None

    embeddings = []
    for i, text in enumerate(texts):
        if i % 500 == 0:
            print(f"  {i}/{len(texts)}...")
        embeddings.append(sf._embed(text))
    return np.array(embeddings, dtype=np.float32)


def export_onnx_and_vocab():
    """Export ONNX model + vocab from sentence-transformers for runtime use."""
    try:
        from sentence_transformers import SentenceTransformer
        from optimum.onnxruntime import ORTModelForFeatureExtraction  # type: ignore
        from transformers import AutoTokenizer                         # type: ignore
    except ImportError:
        print("Skipping ONNX export (optimum not installed). ONNX model not saved.")
        return

    if ONNX_PATH.exists() and VOCAB_PATH.exists():
        print(f"ONNX model already exists at {ONNX_PATH} — skipping export.")
        return

    print("Exporting MiniLM to ONNX (INT8 quantised)...")
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    ort_model = ORTModelForFeatureExtraction.from_pretrained(
        model_id, export=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Quantise to INT8
    from optimum.onnxruntime import ORTQuantizer
    quantizer = ORTQuantizer.from_pretrained(ort_model)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    MODEL_DIR.mkdir(exist_ok=True)
    quantizer.quantize(save_dir=str(MODEL_DIR), quantization_config=qconfig)

    # Save vocab for runtime tokeniser
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        for token, _ in sorted_vocab:
            f.write(token + "\n")

    # Rename quantised model to our expected filename
    import shutil
    candidates = list(MODEL_DIR.glob("model_quantized.onnx"))
    if candidates:
        shutil.move(str(candidates[0]), str(ONNX_PATH))
        print(f"ONNX model saved to {ONNX_PATH} ({ONNX_PATH.stat().st_size/1024/1024:.1f} MB)")


def build_index():
    MODEL_DIR.mkdir(exist_ok=True)
    data = load_data()

    texts   = data["text"].tolist()
    intents = data["intent"].tolist()

    embeddings = embed_with_sentence_transformers(texts)

    # Save as float16 to halve storage (~5.7 MB vs 11.3 MB)
    np.savez_compressed(
        INDEX_PATH,
        embeddings=embeddings.astype(np.float16),
        intents=np.array(intents, dtype=object),
    )

    size_mb = INDEX_PATH.stat().st_size / 1024 / 1024
    print(f"\n✅ Semantic index saved to {INDEX_PATH}")
    print(f"   Phrases: {len(texts)}  Dimensions: {embeddings.shape[1]}  Size: {size_mb:.1f} MB")


if __name__ == "__main__":
    export_onnx_and_vocab()
    build_index()
