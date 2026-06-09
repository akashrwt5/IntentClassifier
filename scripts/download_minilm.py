#!/usr/bin/env python3
"""
Download all-MiniLM-L6-v2 ONNX model + vocab to models/.

Run this once:
    python scripts/download_minilm.py

What it downloads (~90 MB total from HuggingFace):
    models/minilm-l6-v2.onnx    — INT8 quantised ONNX model (~6 MB after quant)
    models/minilm-vocab.txt     — WordPiece vocabulary (230 KB)
"""

from pathlib import Path
import sys

BASE_DIR   = Path(__file__).parent.parent
MODEL_DIR  = BASE_DIR / "models"
ONNX_PATH  = MODEL_DIR / "minilm-l6-v2.onnx"
VOCAB_PATH = MODEL_DIR / "minilm-vocab.txt"

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction  # type: ignore
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        from optimum.onnxruntime import ORTQuantizer
        from transformers import AutoTokenizer                         # type: ignore
    except ImportError:
        print("Missing dependencies. Run:")
        print("  pip install optimum[onnxruntime] transformers")
        sys.exit(1)

    if ONNX_PATH.exists() and VOCAB_PATH.exists():
        print(f"Model already exists at {ONNX_PATH} — nothing to do.")
        return

    print(f"Downloading and exporting {MODEL_ID} to ONNX (INT8)...")
    print("This downloads ~90 MB from HuggingFace and may take a few minutes.\n")

    # Export to ONNX
    ort_model = ORTModelForFeatureExtraction.from_pretrained(MODEL_ID, export=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Quantise INT8 (reduces ~22 MB → ~6 MB with minimal quality loss)
    quantizer = ORTQuantizer.from_pretrained(ort_model)
    qconfig   = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=str(MODEL_DIR), quantization_config=qconfig)

    # Rename to our expected filename
    import shutil
    for candidate in MODEL_DIR.glob("model_quantized.onnx"):
        shutil.move(str(candidate), str(ONNX_PATH))
        break
    else:
        # Fallback: save unquantised
        ort_model.save_pretrained(str(MODEL_DIR))
        for candidate in MODEL_DIR.glob("model.onnx"):
            shutil.move(str(candidate), str(ONNX_PATH))
            break

    # Save vocabulary
    vocab = tokenizer.get_vocab()
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        for token, _ in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(token + "\n")

    size_mb = ONNX_PATH.stat().st_size / 1024 / 1024
    print(f"\n✅ ONNX model saved: {ONNX_PATH}  ({size_mb:.1f} MB)")
    print(f"✅ Vocab saved:      {VOCAB_PATH}  ({VOCAB_PATH.stat().st_size//1024} KB)")
    print(f"\nNext step: python scripts/build_semantic_index.py")


if __name__ == "__main__":
    main()
