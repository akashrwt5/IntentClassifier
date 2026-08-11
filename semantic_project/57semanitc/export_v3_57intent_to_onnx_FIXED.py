#!/usr/bin/env python3
"""
FINAL: Export the 57-intent V3 PyTorch checkpoint to ONNX.

Checkpoint:
    student_v3_57intent_fp32.pt

The exporter INSPECTS the checkpoint first and reconstructs the model
from the state-dict shapes instead of guessing the architecture.

Expected V3 contract:
    input_ids : int64 [1, 24]
    logits    : float32 [1, 57]

Run:
    python3 export_v3_57intent_to_onnx.py
"""

from pathlib import Path
import json
import sys

import torch
import torch.nn as nn


# ============================================================
# PATHS
# ============================================================

# Change only this path if your .pt is somewhere else.
CHECKPOINT = Path(
    "student_v3_57intent_fp32.pt"
)

# Production project output directory.
OUTPUT_DIR = Path(
    "/Users/shuklam/IntentClassifier/semantic_project/"
    "v3_57intent_onnx"
)

OUTPUT_ONNX = (
    OUTPUT_DIR /
    "v3_semantic_student_57intent_fp32.onnx"
)

MANIFEST = (
    OUTPUT_DIR /
    "export_manifest.json"
)


# ============================================================
# FIXED V3 CONFIG
# ============================================================

MAX_LEN = 24
NUM_CLASSES = 57
PAD_ID = 0


# ============================================================
# MODEL
# ============================================================

class V3Student57(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_layers,
        num_heads,
        ffn_dim,
        num_classes,
        max_len,
        dropout=0.10,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=PAD_ID,
        )

        self.position_embedding = nn.Embedding(
            max_len,
            embed_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, input_ids):

        batch_size, seq_len = input_ids.shape

        positions = torch.arange(
            seq_len,
            device=input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0)

        x = (
            self.embedding(input_ids)
            + self.position_embedding(positions)
        )

        padding_mask = input_ids.eq(PAD_ID)

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        # Padding-aware mean pooling.
        mask = (~padding_mask).unsqueeze(-1).to(x.dtype)

        summed = (x * mask).sum(dim=1)

        count = mask.sum(
            dim=1
        ).clamp(min=1.0)

        pooled = summed / count

        pooled = self.norm(pooled)

        return self.classifier(pooled)


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def unwrap_checkpoint(obj):
    """
    Accept common PyTorch checkpoint formats.
    """
    if isinstance(obj, nn.Module):
        return obj.state_dict(), {}

    if not isinstance(obj, dict):
        raise RuntimeError(
            "Unsupported checkpoint object type: "
            f"{type(obj)}"
        )

    metadata = {}

    # Common state-dict containers.
    for key in (
        "model_state_dict",
        "state_dict",
        "model",
    ):
        if key in obj and isinstance(obj[key], dict):
            metadata = {
                k: v
                for k, v in obj.items()
                if k != key
            }
            return obj[key], metadata

    # Raw state_dict.
    tensor_values = [
        v for v in obj.values()
        if torch.is_tensor(v)
    ]

    if tensor_values:
        return obj, metadata

    raise RuntimeError(
        "Could not find a PyTorch state_dict in checkpoint."
    )


def clean_state_dict(state_dict):
    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value

    return cleaned


def get_required(state, key):
    if key not in state:
        raise RuntimeError(
            f"Required checkpoint tensor missing: {key}"
        )
    return state[key]


def inspect_architecture(state):
    """
    Infer the dimensions from the checkpoint itself.

    This avoids silently assuming the wrong architecture.
    """

    emb = get_required(
        state,
        "embedding.weight",
    )

    pos = get_required(
        state,
        "position_embedding.weight",
    )

    classifier0 = get_required(
        state,
        "classifier.0.weight",
    )

    classifier_last = get_required(
        state,
        "classifier.3.weight",
    )

    vocab_size, embed_dim = emb.shape
    max_len, pos_dim = pos.shape

    if embed_dim != pos_dim:
        raise RuntimeError(
            "Embedding and position dimensions differ: "
            f"{embed_dim} vs {pos_dim}"
        )

    cls_hidden, cls_in = classifier0.shape
    num_classes, cls_last_in = classifier_last.shape

    if cls_in != embed_dim:
        raise RuntimeError(
            "Classifier input does not match embedding dimension: "
            f"{cls_in} vs {embed_dim}"
        )

    if cls_last_in != cls_hidden:
        raise RuntimeError(
            "Classifier dimensions are inconsistent: "
            f"{cls_last_in} vs {cls_hidden}"
        )

    if num_classes != NUM_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_CLASSES} classes, "
            f"checkpoint has {num_classes}."
        )

    # Detect transformer depth.
    layer_ids = set()

    prefix = "encoder.layers."

    for key in state:
        if key.startswith(prefix):
            remainder = key[len(prefix):]
            first = remainder.split(".", 1)[0]

            if first.isdigit():
                layer_ids.add(int(first))

    if not layer_ids:
        raise RuntimeError(
            "Could not detect Transformer encoder layers."
        )

    num_layers = max(layer_ids) + 1

    # Detect FFN dimension from the first encoder layer.
    ffn_key = (
        "encoder.layers.0.linear1.weight"
    )

    ffn = get_required(state, ffn_key)

    ffn_dim = ffn.shape[0]

    # Infer number of attention heads from in_proj.
    # d_model = embed_dim.
    # The exact head count cannot always be inferred from
    # the state dict alone, so we use the V3 architecture's
    # known 4-head configuration after validating divisibility.
    num_heads = 4

    if embed_dim % num_heads != 0:
        raise RuntimeError(
            f"V3 4-head configuration is incompatible "
            f"with embedding dimension {embed_dim}."
        )

    return {
        "vocab_size": int(vocab_size),
        "embed_dim": int(embed_dim),
        "max_len": int(max_len),
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "ffn_dim": int(ffn_dim),
        "classifier_hidden": int(cls_hidden),
        "num_classes": int(num_classes),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("V3 57-INTENT FP32 -> ONNX EXPORT")
    print("=" * 78)

    print("\nCheckpoint:")
    print(CHECKPOINT)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            "Checkpoint not found:\n"
            f"{CHECKPOINT}"
        )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
    )

    state_dict, checkpoint_metadata = (
        unwrap_checkpoint(checkpoint)
    )

    state_dict = clean_state_dict(
        state_dict
    )

    print("\nCheckpoint tensors:", len(state_dict))

    # --------------------------------------------------------
    # Inspect architecture
    # --------------------------------------------------------

    arch = inspect_architecture(
        state_dict
    )

    print("\n--- CHECKPOINT ARCHITECTURE ---")

    for key, value in arch.items():
        print(f"{key:20s}: {value}")

    if arch["vocab_size"] != 895:
        print(
            "\nWARNING: vocabulary size is not 895."
        )

    if arch["embed_dim"] != 64:
        raise RuntimeError(
            "This is not the expected V3 64-dim model."
        )

    if arch["num_layers"] != 2:
        raise RuntimeError(
            "This is not the expected V3 2-layer model."
        )

    if arch["ffn_dim"] != 128:
        raise RuntimeError(
            "This is not the expected V3 FFN=128 model."
        )

    if arch["max_len"] != MAX_LEN:
        raise RuntimeError(
            f"Expected max length {MAX_LEN}, "
            f"got {arch['max_len']}."
        )

    # --------------------------------------------------------
    # Build exact model
    # --------------------------------------------------------

    model = V3Student57(
        vocab_size=arch["vocab_size"],
        embed_dim=arch["embed_dim"],
        num_layers=arch["num_layers"],
        num_heads=arch["num_heads"],
        ffn_dim=arch["ffn_dim"],
        num_classes=arch["num_classes"],
        max_len=arch["max_len"],
        dropout=0.10,
    )

    # --------------------------------------------------------
    # Strict load
    # --------------------------------------------------------

    print("\nLoading checkpoint STRICTLY...")

    try:
        model.load_state_dict(
            state_dict,
            strict=True,
        )
    except RuntimeError as exc:
        print("\nSTRICT CHECKPOINT LOAD FAILED.")
        print(exc)
        print(
            "\nONNX export ABORTED. "
            "Do not export a mismatched architecture."
        )
        raise

    model.eval()

    print("Strict checkpoint load: PASS")

    # --------------------------------------------------------
    # Verify output shape
    # --------------------------------------------------------

    dummy = torch.zeros(
        (1, MAX_LEN),
        dtype=torch.long,
    )

    with torch.no_grad():
        torch_logits = model(dummy)

    print("\n--- PYTORCH CHECK ---")
    print("Input shape :", tuple(dummy.shape))
    print("Input dtype :", dummy.dtype)
    print(
        "Output shape:",
        tuple(torch_logits.shape),
    )
    print(
        "Output dtype:",
        torch_logits.dtype,
    )

    if tuple(torch_logits.shape) != (
        1,
        NUM_CLASSES,
    ):
        raise RuntimeError(
            "Unexpected PyTorch output shape: "
            f"{tuple(torch_logits.shape)}"
        )

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n--- ONNX EXPORT ---")
    print("Output:")
    print(OUTPUT_ONNX)

    torch.onnx.export(
        model,
        dummy,
        str(OUTPUT_ONNX),
        input_names=["input_ids"],
        output_names=["logits"],
        opset_version=18,
        do_constant_folding=True,
        dynamic_axes=None,
    )

    print("\nONNX export: PASS")

    # --------------------------------------------------------
    # Verify ONNX with ONNX Runtime
    # --------------------------------------------------------

    try:
        import onnx
        import onnxruntime as ort
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Install required packages first:\n"
            "pip install onnx onnxruntime"
        ) from exc

    print("\n--- ONNX VALIDATION ---")

    onnx_model = onnx.load(
        str(OUTPUT_ONNX)
    )

    onnx.checker.check_model(
        onnx_model
    )

    print("onnx.checker: PASS")

    session = ort.InferenceSession(
        str(OUTPUT_ONNX),
        providers=["CPUExecutionProvider"],
    )

    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]

    print(
        "Input name :",
        input_meta.name,
    )
    print(
        "Input shape:",
        input_meta.shape,
    )
    print(
        "Input type :",
        input_meta.type,
    )

    print(
        "Output name:",
        output_meta.name,
    )
    print(
        "Output shape:",
        output_meta.shape,
    )
    print(
        "Output type :",
        output_meta.type,
    )

    if input_meta.shape != [1, 24]:
        raise RuntimeError(
            f"Unexpected ONNX input shape: "
            f"{input_meta.shape}"
        )

    if output_meta.shape != [1, 57]:
        raise RuntimeError(
            f"Unexpected ONNX output shape: "
            f"{output_meta.shape}"
        )

    # --------------------------------------------------------
    # PyTorch vs ONNX numerical parity
    # --------------------------------------------------------

    sample = np.asarray(
        [
            [
                40, 2, 6, 11, 3, 4,
                238, 41, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0
            ]
        ],
        dtype=np.int64,
    )

    with torch.no_grad():
        pt = (
            model(
                torch.from_numpy(sample)
            )
            .cpu()
            .numpy()
        )

    ort_out = session.run(
        [output_meta.name],
        {
            input_meta.name: sample
        },
    )[0]

    max_abs_diff = float(
        np.max(
            np.abs(pt - ort_out)
        )
    )

    mean_abs_diff = float(
        np.mean(
            np.abs(pt - ort_out)
        )
    )

    pt_pred = int(
        np.argmax(pt[0])
    )

    ort_pred = int(
        np.argmax(ort_out[0])
    )

    print("\n--- PYTORCH vs ONNX PARITY ---")
    print(
        f"Max absolute difference : "
        f"{max_abs_diff:.10f}"
    )
    print(
        f"Mean absolute difference: "
        f"{mean_abs_diff:.10f}"
    )
    print(
        f"PyTorch prediction      : {pt_pred}"
    )
    print(
        f"ONNX prediction         : {ort_pred}"
    )

    if pt_pred != ort_pred:
        raise RuntimeError(
            "PyTorch and ONNX predictions differ."
        )

    if max_abs_diff > 1e-4:
        raise RuntimeError(
            "PyTorch/ONNX numerical parity failed."
        )

    print("Prediction parity: PASS")
    print("Numerical parity : PASS")

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = {
        "model": "V3 57-intent semantic student",
        "checkpoint": str(CHECKPOINT),
        "onnx": str(OUTPUT_ONNX),
        "format": "ONNX FP32",
        "opset": 18,
        "input": {
            "name": "input_ids",
            "dtype": "int64",
            "shape": [1, 24],
        },
        "output": {
            "name": "logits",
            "dtype": "float32",
            "shape": [1, 57],
        },
        "architecture": arch,
        "intents": 57,
        "int8": False,
        "dynamic_batch": False,
        "locked_595_used": False,
        "pytorch_onnx_max_abs_diff": max_abs_diff,
        "pytorch_onnx_mean_abs_diff": mean_abs_diff,
        "prediction_parity": True,
    }

    MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nManifest:")
    print(MANIFEST)

    print("\n" + "=" * 78)
    print("EXPORT + VALIDATION COMPLETE")
    print("=" * 78)

    print("\nFinal contract:")
    print("Input  : input_ids int64 [1,24]")
    print("Output : logits float32 [1,57]")
    print("FP32   : YES")
    print("INT8   : NO")
    print("595-row benchmark: NOT RUN")
    print("PyTorch/ONNX parity: PASS")
    print("\nNext: run the 57-intent benchmark before mobile integration.")


if __name__ == "__main__":
    main()
