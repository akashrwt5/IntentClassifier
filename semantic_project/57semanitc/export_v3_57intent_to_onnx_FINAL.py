#!/usr/bin/env python3

"""
FINAL V3 57-INTENT PYTORCH -> ONNX EXPORTER

Checkpoint:
    student_v3_57intent_fp32.pt

Run from the same directory:
    python3 export_v3_57intent_to_onnx.py

Expected:
    input_ids : int64 [1, 24]
    logits    : float32 [1, 57]

This version uses the checkpoint's actual key:
    position.weight
NOT:
    position_embedding.weight
"""

from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# PATHS
# ============================================================

CHECKPOINT = Path("student_v3_57intent_fp32.pt")

OUTPUT_DIR = Path("v3_57intent_onnx")

OUTPUT_ONNX = (
    OUTPUT_DIR /
    "v3_semantic_student_57intent_fp32.onnx"
)

MANIFEST = (
    OUTPUT_DIR /
    "export_manifest.json"
)


# ============================================================
# FIXED CONTRACT
# ============================================================

MAX_LEN = 24
NUM_CLASSES = 57
PAD_ID = 0

VOCAB_SIZE = 895
EMBED_DIM = 64
NUM_LAYERS = 2
NUM_HEADS = 4
FFN_DIM = 128
CLASSIFIER_HIDDEN = 64
DROPOUT = 0.10


# ============================================================
# EXACT V3 57-INTENT MODEL
# ============================================================

class V3Student57(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            EMBED_DIM,
            padding_idx=PAD_ID,
        )

        # IMPORTANT:
        # Actual checkpoint key is position.weight
        self.position = nn.Embedding(
            MAX_LEN,
            EMBED_DIM,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=FFN_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_LAYERS,
        )

        self.norm = nn.LayerNorm(
            EMBED_DIM
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                EMBED_DIM,
                CLASSIFIER_HIDDEN,
            ),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(
                CLASSIFIER_HIDDEN,
                NUM_CLASSES,
            ),
        )

    def forward(self, input_ids):

        batch_size, seq_len = input_ids.shape

        if seq_len > MAX_LEN:
            raise RuntimeError(
                f"Sequence length {seq_len} exceeds "
                f"MAX_LEN={MAX_LEN}"
            )

        positions = torch.arange(
            seq_len,
            device=input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0)

        x = (
            self.embedding(input_ids)
            + self.position(positions)
        )

        padding_mask = input_ids.eq(
            PAD_ID
        )

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        # Padding-aware mean pooling.
        mask = (
            (~padding_mask)
            .unsqueeze(-1)
            .to(x.dtype)
        )

        summed = (
            x * mask
        ).sum(dim=1)

        count = (
            mask.sum(dim=1)
            .clamp(min=1.0)
        )

        pooled = summed / count

        pooled = self.norm(
            pooled
        )

        return self.classifier(
            pooled
        )


# ============================================================
# CHECKPOINT
# ============================================================

def load_state_dict():

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"\nCheckpoint not found:\n"
            f"{CHECKPOINT.resolve()}\n\n"
            f"Make sure this file exists in the current "
            f"directory:\n"
            f"student_v3_57intent_fp32.pt"
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:
            state = checkpoint[
                "state_dict"
            ]

        else:
            state = checkpoint

    else:
        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    cleaned = {}

    for key, value in state.items():

        if key.startswith("module."):
            key = key[
                len("module.") :
            ]

        cleaned[key] = value

    return cleaned


# ============================================================
# CHECKPOINT INSPECTION
# ============================================================

def inspect_checkpoint(state):

    print("\n--- CHECKPOINT SHAPES ---")

    required = [
        "embedding.weight",
        "position.weight",
        "encoder.layers.0.self_attn.in_proj_weight",
        "encoder.layers.0.linear1.weight",
        "norm.weight",
        "classifier.0.weight",
        "classifier.3.weight",
    ]

    for key in required:

        if key not in state:
            raise RuntimeError(
                f"Required checkpoint tensor missing: "
                f"{key}"
            )

        print(
            f"{key:55s} "
            f"{tuple(state[key].shape)}"
        )

    # Explicit architecture validation.

    if tuple(
        state["embedding.weight"].shape
    ) != (VOCAB_SIZE, EMBED_DIM):

        raise RuntimeError(
            "Unexpected embedding shape."
        )

    if tuple(
        state["position.weight"].shape
    ) != (MAX_LEN, EMBED_DIM):

        raise RuntimeError(
            "Unexpected position shape."
        )

    if tuple(
        state["encoder.layers.0.linear1.weight"].shape
    ) != (FFN_DIM, EMBED_DIM):

        raise RuntimeError(
            "Unexpected FFN shape."
        )

    if tuple(
        state["classifier.0.weight"].shape
    ) != (
        CLASSIFIER_HIDDEN,
        EMBED_DIM,
    ):

        raise RuntimeError(
            "Unexpected classifier hidden shape."
        )

    if tuple(
        state["classifier.3.weight"].shape
    ) != (
        NUM_CLASSES,
        CLASSIFIER_HIDDEN,
    ):

        raise RuntimeError(
            "Checkpoint is not a 57-intent model."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("V3 57-INTENT FP32 -> ONNX")
    print("=" * 78)

    print("\nCheckpoint:")
    print(CHECKPOINT.resolve())

    state = load_state_dict()

    print(
        f"\nCheckpoint tensors: {len(state)}"
    )

    inspect_checkpoint(
        state
    )

    # --------------------------------------------------------
    # Build exact model
    # --------------------------------------------------------

    model = V3Student57()

    print(
        "\nLoading checkpoint STRICTLY..."
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.eval()

    print(
        "Strict checkpoint load: PASS"
    )

    # --------------------------------------------------------
    # PyTorch sanity test
    # --------------------------------------------------------

    dummy = torch.zeros(
        (1, MAX_LEN),
        dtype=torch.long,
    )

    with torch.no_grad():

        pt_output = model(
            dummy
        )

    print(
        "\n--- PYTORCH SANITY ---"
    )

    print(
        "Input shape :",
        tuple(dummy.shape),
    )

    print(
        "Input dtype :",
        dummy.dtype,
    )

    print(
        "Output shape:",
        tuple(pt_output.shape),
    )

    print(
        "Output dtype:",
        pt_output.dtype,
    )

    if tuple(
        pt_output.shape
    ) != (
        1,
        NUM_CLASSES,
    ):

        raise RuntimeError(
            "PyTorch output is not [1,57]."
        )

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n--- ONNX EXPORT ---"
    )

    print(
        "Output:",
        OUTPUT_ONNX.resolve(),
    )

    torch.onnx.export(
        model,
        dummy,
        str(OUTPUT_ONNX),
        input_names=[
            "input_ids"
        ],
        output_names=[
            "logits"
        ],
        opset_version=18,
        do_constant_folding=True,
        dynamic_axes=None,
    )

    print(
        "ONNX export: PASS"
    )

    # --------------------------------------------------------
    # ONNX validation
    # --------------------------------------------------------

    try:
        import onnx
        import onnxruntime as ort

    except ImportError as exc:

        raise RuntimeError(
            "\nInstall required packages:\n"
            "pip install onnx onnxruntime"
        ) from exc

    print(
        "\n--- ONNX VALIDATION ---"
    )

    onnx_model = onnx.load(
        str(OUTPUT_ONNX)
    )

    onnx.checker.check_model(
        onnx_model
    )

    print(
        "onnx.checker: PASS"
    )

    session = ort.InferenceSession(
        str(OUTPUT_ONNX),
        providers=[
            "CPUExecutionProvider"
        ],
    )

    input_meta = (
        session.get_inputs()[0]
    )

    output_meta = (
        session.get_outputs()[0]
    )

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

    if input_meta.name != "input_ids":
        raise RuntimeError(
            "Unexpected ONNX input name."
        )

    if output_meta.name != "logits":
        raise RuntimeError(
            "Unexpected ONNX output name."
        )

    if input_meta.shape != [
        1,
        24,
    ]:
        raise RuntimeError(
            f"Unexpected ONNX input shape: "
            f"{input_meta.shape}"
        )

    if output_meta.shape != [
        1,
        57,
    ]:
        raise RuntimeError(
            f"Unexpected ONNX output shape: "
            f"{output_meta.shape}"
        )

    # --------------------------------------------------------
    # PyTorch vs ONNX numerical parity
    # --------------------------------------------------------

    sample = np.array(
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

        pt_logits = (
            model(
                torch.from_numpy(sample)
            )
            .cpu()
            .numpy()
        )

    ort_logits = session.run(
        [output_meta.name],
        {
            input_meta.name: sample
        },
    )[0]

    max_abs_diff = float(
        np.max(
            np.abs(
                pt_logits -
                ort_logits
            )
        )
    )

    mean_abs_diff = float(
        np.mean(
            np.abs(
                pt_logits -
                ort_logits
            )
        )
    )

    pt_pred = int(
        np.argmax(
            pt_logits[0]
        )
    )

    ort_pred = int(
        np.argmax(
            ort_logits[0]
        )
    )

    print(
        "\n--- PYTORCH vs ONNX PARITY ---"
    )

    print(
        f"Max absolute difference : "
        f"{max_abs_diff:.10f}"
    )

    print(
        f"Mean absolute difference: "
        f"{mean_abs_diff:.10f}"
    )

    print(
        f"PyTorch prediction      : "
        f"{pt_pred}"
    )

    print(
        f"ONNX prediction         : "
        f"{ort_pred}"
    )

    if pt_pred != ort_pred:
        raise RuntimeError(
            "Prediction parity FAILED."
        )

    if max_abs_diff > 1e-4:
        raise RuntimeError(
            "Numerical parity FAILED."
        )

    print(
        "Prediction parity: PASS"
    )

    print(
        "Numerical parity : PASS"
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = {
        "model": "V3 57-intent semantic student",
        "checkpoint": str(
            CHECKPOINT.resolve()
        ),
        "onnx": str(
            OUTPUT_ONNX.resolve()
        ),
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
        "architecture": {
            "vocab_size": VOCAB_SIZE,
            "embedding": EMBED_DIM,
            "position_length": MAX_LEN,
            "transformer_layers": NUM_LAYERS,
            "attention_heads": NUM_HEADS,
            "ffn": FFN_DIM,
            "classifier": "64 -> 64 -> 57",
        },
        "int8": False,
        "dynamic_batch": False,
        "prediction_parity": True,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
    }

    MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nManifest:"
    )

    print(
        MANIFEST.resolve()
    )

    print(
        "\n" + "=" * 78
    )

    print(
        "EXPORT + VALIDATION COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        "\nFinal contract:"
    )

    print(
        "Input  : input_ids int64 [1,24]"
    )

    print(
        "Output : logits float32 [1,57]"
    )

    print(
        "FP32   : YES"
    )

    print(
        "INT8   : NO"
    )

    print(
        "PyTorch/ONNX parity: PASS"
    )

    print(
        "\nNext: benchmark this ONNX model "
        "against the 57-intent test CSV."
    )


if __name__ == "__main__":
    main()
