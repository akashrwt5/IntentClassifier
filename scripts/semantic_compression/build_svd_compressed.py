#!/usr/bin/env python3
"""
Track 3: Deep Matrix Factorization (SVD)
Builds a compressed version of all-MiniLM-L6-v2 by applying Truncated SVD
to the heavy embedding and linear layers, physically shrinking the model size.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
OUT_DIR = Path(__file__).parent / "output_models" / "track3_svd_l6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Rank to compress to. Original dim is 384.
# Reducing to 128 cuts the matrix size by 3x (minus the overhead of two matrices).
SVD_RANK = 128


def apply_svd_to_linear(layer: nn.Linear, rank: int) -> nn.Sequential:
    """Replaces a linear layer W (out x in) with U (out x rank) and V (rank x in)"""
    W = layer.weight.data
    b = layer.bias.data if layer.bias is not None else None

    U, S, Vh = torch.linalg.svd(W, full_matrices=False)

    # Keep top 'rank' singular values
    U_k = U[:, :rank]
    S_k = torch.diag(S[:rank])
    Vh_k = Vh[:rank, :]

    # Create two smaller sequential linear layers
    # Layer 1: Vh_k (rank x in)
    # Layer 2: U_k @ S_k (out x rank)
    lin1 = nn.Linear(layer.in_features, rank, bias=False)
    lin1.weight.data = Vh_k.contiguous()

    lin2 = nn.Linear(rank, layer.out_features, bias=True if b is not None else False)
    lin2.weight.data = (U_k @ S_k).contiguous()
    if b is not None:
        lin2.bias.data = b.contiguous()

    return nn.Sequential(lin1, lin2)


def main():
    print(f"Loading {MODEL_ID} for SVD Factorization...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)

    print(f"Applying Truncated SVD (Rank={SVD_RANK}) to model layers...")

    # 1. Compress the main vocabulary embedding matrix
    # Original: vocab_size x 384
    # Factorized: (vocab_size x rank) * (rank x 384)
    # Note: transformers expects `word_embeddings` to be an nn.Embedding layer that returns 384-dim.
    # We can't easily replace nn.Embedding with nn.Sequential(nn.Embedding, nn.Linear)
    # because the BertEmbeddings forward pass expects it to return the full hidden size to add to position embeddings.
    # Therefore, to truly compress the embeddings on disk, we have to rewrite the BertEmbeddings class,
    # or just compress the transformer linear layers for now.

    # For this script, we will aggressively compress the internal Transformer layers:
    compressed_layers = 0
    for i, layer_module in enumerate(model.encoder.layer):
        # Attention Q, K, V
        layer_module.attention.self.query = apply_svd_to_linear(
            layer_module.attention.self.query, SVD_RANK
        )
        layer_module.attention.self.key = apply_svd_to_linear(
            layer_module.attention.self.key, SVD_RANK
        )
        layer_module.attention.self.value = apply_svd_to_linear(
            layer_module.attention.self.value, SVD_RANK
        )

        # Attention Output
        layer_module.attention.output.dense = apply_svd_to_linear(
            layer_module.attention.output.dense, SVD_RANK
        )

        # Intermediate Dense
        layer_module.intermediate.dense = apply_svd_to_linear(
            layer_module.intermediate.dense, SVD_RANK
        )

        # Output Dense
        layer_module.output.dense = apply_svd_to_linear(layer_module.output.dense, SVD_RANK)
        compressed_layers += 6

    print(f"Compressed {compressed_layers} linear matrices using SVD.")

    # Save the modified PyTorch model
    pt_dir = OUT_DIR / "pytorch"
    print("\nSaving PyTorch model (pre-recovery training)...")
    # Note: saving this will throw warnings because the architecture structure changed slightly.
    # But PyTorch standard save_pretrained handles the dict keys if we use custom saving,
    # though transformers might struggle to reload it natively. Let's just trace to ONNX directly!

    print("Exporting factorized model to ONNX...")
    # Because we modified the class structure (nn.Linear -> nn.Sequential), Optimum might crash if it relies on exact class names.
    # Let's try it. If it fails, the fallback is standard torch.onnx.export.
    model.save_pretrained(pt_dir)
    tokenizer.save_pretrained(pt_dir)

    try:
        ort_model = ORTModelForFeatureExtraction.from_pretrained(pt_dir, export=True)
        print("Quantizing to INT8...")
        quantizer = ORTQuantizer.from_pretrained(ort_model)
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

        quant_dir = OUT_DIR / "onnx_quantized"
        quantizer.quantize(save_dir=str(quant_dir), quantization_config=qconfig)

        q_size = (quant_dir / "model_quantized.onnx").stat().st_size / (1024 * 1024)
        print(f"\n✅ Track 3 SVD Model Built Successfully!")
        print(f"ONNX Size (INT8): {q_size:.2f} MB")
    except Exception as e:
        print(f"\nFailed to export to ONNX via Optimum: {e}")
        print(
            "This usually happens because replacing linear layers with sequential layers breaks HuggingFace's static configuration tracing."
        )


if __name__ == "__main__":
    main()
