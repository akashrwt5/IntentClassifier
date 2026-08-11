#!/usr/bin/env python3
"""
FINAL V2.1 -> ONNX FP32 EXPORT

Source:
  v3_57intent_v2_1_controlled/student_v3_57intent_v2_1_best_fp32.pt

Contract:
  input_ids : int64 [1, 24]
  logits    : float32 [1, 57]

This script:
- loads V2.1 strictly
- exports FP32 ONNX
- validates ONNX
- checks PyTorch vs ONNX numerical parity
- checks prediction parity
- writes a manifest
- does NOT train
- does NOT read the locked test
"""

from pathlib import Path
import json

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent

CHECKPOINT = (
    ROOT
    / "v3_57intent_v2_1_controlled"
    / "student_v3_57intent_v2_1_best_fp32.pt"
)

OUT_DIR = ROOT / "v3_57intent_v2_1_onnx"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ONNX_PATH = OUT_DIR / "v2_1_57intent_fp32.onnx"
MANIFEST_PATH = OUT_DIR / "export_manifest.json"

VOCAB_SIZE = 895
EMBED_DIM = 64
HEADS = 4
LAYERS = 2
FF_DIM = 128
MAX_LEN = 24
NUM_CLASSES = 57
DROPOUT = 0.10


class V3Student57(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            EMBED_DIM,
            padding_idx=0,
        )

        self.position = nn.Embedding(
            MAX_LEN,
            EMBED_DIM,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=HEADS,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=LAYERS,
        )

        self.norm = nn.LayerNorm(EMBED_DIM)

        self.classifier = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(EMBED_DIM, NUM_CLASSES),
        )

    def forward(self, x):
        padding_mask = x.eq(0)

        pos = torch.arange(
            x.size(1),
            device=x.device,
        ).unsqueeze(0)

        h = self.embedding(x) + self.position(pos)

        h = self.encoder(
            h,
            src_key_padding_mask=padding_mask,
        )

        valid = (~padding_mask).unsqueeze(-1).float()

        pooled = (
            (h * valid).sum(dim=1)
            / valid.sum(dim=1).clamp(min=1.0)
        )

        return self.classifier(self.norm(pooled))


def main():
    print("=" * 78)
    print("V2.1 57-INTENT -> ONNX FP32 EXPORT")
    print("=" * 78)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"V2.1 checkpoint not found:\n{CHECKPOINT}"
        )

    torch.manual_seed(42)

    model = V3Student57()

    state = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state, strict=True)
    model.eval()

    # Fixed deterministic test input.
    dummy = torch.tensor(
        [[
            2, 15, 31, 47, 63, 79,
            95, 111, 127, 143, 159, 175,
            191, 207, 223, 239, 255, 271,
            287, 303, 319, 335, 351, 0
        ]],
        dtype=torch.long,
    )

    with torch.no_grad():
        torch_logits = model(dummy).cpu().numpy()

    print("\n--- PYTORCH REFERENCE ---")
    print(f"Input shape  : {list(dummy.shape)}")
    print(f"Output shape : {list(torch_logits.shape)}")
    print(f"Prediction   : {int(torch_logits.argmax(axis=1)[0])}")

    print("\n--- ONNX EXPORT ---")

    torch.onnx.export(
        model,
        (dummy,),
        str(ONNX_PATH),
        input_names=["input_ids"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=True,
    )

    print("ONNX export: PASS")
    print(f"ONNX path: {ONNX_PATH}")

    print("\n--- ONNX VALIDATION ---")

    model_onnx = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(model_onnx)

    print("onnx.checker: PASS")

    graph_input = model_onnx.graph.input[0]
    graph_output = model_onnx.graph.output[0]

    input_name = graph_input.name
    output_name = graph_output.name

    input_shape = [
        d.dim_value if d.HasField("dim_value") else "?"
        for d in graph_input.type.tensor_type.shape.dim
    ]

    output_shape = [
        d.dim_value if d.HasField("dim_value") else "?"
        for d in graph_output.type.tensor_type.shape.dim
    ]

    print(f"Input name : {input_name}")
    print(f"Input shape: {input_shape}")
    print(f"Output name: {output_name}")
    print(f"Output shape: {output_shape}")

    print("\n--- PYTORCH vs ONNX PARITY ---")

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    ort_input_name = session.get_inputs()[0].name
    ort_output_name = session.get_outputs()[0].name

    onnx_logits = session.run(
        [ort_output_name],
        {
            ort_input_name: dummy.numpy(),
        },
    )[0]

    max_abs = float(
        np.max(np.abs(torch_logits - onnx_logits))
    )

    mean_abs = float(
        np.mean(np.abs(torch_logits - onnx_logits))
    )

    torch_pred = int(torch_logits.argmax(axis=1)[0])
    onnx_pred = int(onnx_logits.argmax(axis=1)[0])

    print(f"Max absolute difference : {max_abs:.10f}")
    print(f"Mean absolute difference: {mean_abs:.10f}")
    print(f"PyTorch prediction      : {torch_pred}")
    print(f"ONNX prediction         : {onnx_pred}")

    prediction_parity = torch_pred == onnx_pred
    numerical_parity = max_abs < 1e-3

    print(
        "Prediction parity:",
        "PASS" if prediction_parity else "FAIL",
    )

    print(
        "Numerical parity:",
        "PASS" if numerical_parity else "FAIL",
    )

    if not prediction_parity or not numerical_parity:
        raise RuntimeError(
            "ONNX parity validation FAILED."
        )

    size_mb = ONNX_PATH.stat().st_size / (1024 * 1024)

    manifest = {
        "model": "V2.1 Controlled 57-intent",
        "checkpoint": str(CHECKPOINT.resolve()),
        "onnx": str(ONNX_PATH.resolve()),
        "input_name": input_name,
        "input_type": "int64",
        "input_shape": [1, 24],
        "output_name": output_name,
        "output_type": "float32",
        "output_shape": [1, 57],
        "precision": "FP32",
        "int8": False,
        "onnx_checker": "PASS",
        "prediction_parity": "PASS",
        "numerical_parity": "PASS",
        "max_absolute_difference": max_abs,
        "mean_absolute_difference": mean_abs,
        "model_size_mb": size_mb,
        "locked_test_read": False,
        "training_performed": False,
    }

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\n--- FINAL CONTRACT ---")
    print("Input  : input_ids int64 [1,24]")
    print("Output : logits float32 [1,57]")
    print("FP32   : YES")
    print("INT8   : NO")
    print("PyTorch/ONNX parity: PASS")
    print(f"Model size: {size_mb:.3f} MB")

    print("\nManifest:")
    print(MANIFEST_PATH)

    print("\nSTATUS:")
    print("V2.1 ONNX FP32 EXPORT COMPLETE")


if __name__ == "__main__":
    main()
