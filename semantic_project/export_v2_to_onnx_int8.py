#!/usr/bin/env python3
"""
Export V2 Balanced FP32 -> ONNX -> INT8.

Source:
    tiny_semantic_student_v2_balanced/student_v2_balanced_fp32.pt

This script:
1. Recreates the SAME V2 architecture.
2. Loads V2 FP32 weights.
3. Exports V2 to ONNX.
4. Quantizes ONNX weights to INT8.
5. Does NOT modify the current INT8 baseline.
6. Writes all V2 artifacts into a separate directory.

Run:
    python3 export_v2_to_onnx_int8.py

Requirements:
    pip install torch onnx onnxruntime onnxruntime-tools
"""

from pathlib import Path
import json
import shutil
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
V2_DIR = ROOT / "tiny_semantic_student_v2_balanced"
OUT_DIR = ROOT / "tiny_semantic_student_v2_int8"

FP32_PATH = V2_DIR / "student_v2_balanced_fp32.pt"
VOCAB_PATH = V2_DIR / "vocab.json"
CONFIG_PATH = V2_DIR / "config.json"
LABELS_PATH = V2_DIR / "intent_labels.txt"

ONNX_PATH = OUT_DIR / "v2_semantic_student_fp32.onnx"
INT8_PATH = OUT_DIR / "v2_semantic_student_int8.onnx"

# ------------------------------------------------------------
# Check files
# ------------------------------------------------------------

for p in [
    FP32_PATH,
    VOCAB_PATH,
    CONFIG_PATH,
    LABELS_PATH,
]:
    if not p.exists():
        raise FileNotFoundError(f"Missing V2 artifact: {p}")

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (
        quantize_dynamic,
        QuantType,
    )
except ImportError as e:
    raise ImportError(
        "\nMissing ONNX packages.\n"
        "Install with:\n\n"
        "pip install onnx onnxruntime\n"
    ) from e

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = [x.strip() for x in f if x.strip()]

def cfg(*keys, default=None):
    for key in keys:
        if key in config:
            return config[key]
    return default

ED = int(cfg("embed_dim", "embedding_dim", default=64))
NH = int(cfg("num_heads", "nhead", default=4))
NL = int(cfg("num_layers", "layers", default=2))
FF = int(cfg("ff_dim", "feedforward_dim", default=128))
ML = int(cfg("max_len", "max_length", "sequence_length", default=24))

PAD = int(vocab.get("<PAD>", vocab.get("[PAD]", 0)))

print("=" * 72)
print("V2 FP32 -> ONNX -> INT8 EXPORT")
print("=" * 72)
print("V2 FP32:", FP32_PATH)
print("Output:", OUT_DIR)
print("Vocab:", len(vocab))
print("Layers:", NL)
print("Heads:", NH)
print("Embedding:", ED)
print("Max length:", ML)
print("Intents:", len(labels))

# ------------------------------------------------------------
# SAME V2 architecture
# ------------------------------------------------------------

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            len(vocab),
            ED,
            padding_idx=PAD,
        )

        self.position = nn.Embedding(
            ML,
            ED,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=ED,
            nhead=NH,
            dim_feedforward=FF,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            NL,
        )

        self.norm = nn.LayerNorm(ED)

        self.classifier = nn.Sequential(
            nn.Linear(ED, ED),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(ED, len(labels)),
        )

    def forward(self, x):
        mask = x.eq(PAD)

        pos = torch.arange(
            x.size(1),
            device=x.device,
        ).unsqueeze(0)

        h = self.embedding(x) + self.position(pos)

        h = self.encoder(
            h,
            src_key_padding_mask=mask,
        )

        valid = (~mask).unsqueeze(-1).float()

        h = (
            h * valid
        ).sum(1) / valid.sum(1).clamp(min=1.0)

        return self.classifier(
            self.norm(h)
        )

model = Model()

state = torch.load(
    FP32_PATH,
    map_location="cpu",
)

if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]

model.load_state_dict(state)
model.eval()

# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# Preserve tokenizer/config/labels.
for p in [
    VOCAB_PATH,
    CONFIG_PATH,
    LABELS_PATH,
]:
    shutil.copy2(
        p,
        OUT_DIR / p.name,
    )

# ------------------------------------------------------------
# Export ONNX
# ------------------------------------------------------------

dummy = torch.zeros(
    (1, ML),
    dtype=torch.long,
)

print("\nExporting FP32 ONNX...")

with torch.no_grad():
    torch.onnx.export(
        model,
        dummy,
        str(ONNX_PATH),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {
                0: "batch"
            },
            "logits": {
                0: "batch"
            },
        },
        opset_version=17,
        do_constant_folding=True,
    )

onnx_model = onnx.load(str(ONNX_PATH))
onnx.checker.check_model(onnx_model)

print("ONNX export successful.")
print("File:", ONNX_PATH)
print(
    "Size:",
    f"{ONNX_PATH.stat().st_size / 1024 / 1024:.3f} MB",
)

# ------------------------------------------------------------
# ONNX Runtime FP32 sanity check
# ------------------------------------------------------------

session_fp32 = ort.InferenceSession(
    str(ONNX_PATH),
    providers=["CPUExecutionProvider"],
)

input_name = session_fp32.get_inputs()[0].name
output_name = session_fp32.get_outputs()[0].name

sample = np.zeros(
    (2, ML),
    dtype=np.int64,
)

ort_output = session_fp32.run(
    [output_name],
    {
        input_name: sample,
    },
)[0]

torch_output = (
    model(
        torch.from_numpy(sample)
    )
    .detach()
    .numpy()
)

max_diff = float(
    np.max(
        np.abs(
            ort_output - torch_output
        )
    )
)

print("\nONNX FP32 sanity check:")
print("Input :", input_name)
print("Output:", output_name)
print("Shape :", ort_output.shape)
print("Max Torch/ONNX difference:", max_diff)

if not np.allclose(
    ort_output,
    torch_output,
    rtol=1e-3,
    atol=1e-4,
):
    print(
        "WARNING: Torch/ONNX outputs differ more than expected."
    )
else:
    print("Torch/ONNX numerical check: PASS")

# ------------------------------------------------------------
# Dynamic INT8 quantization
# ------------------------------------------------------------

print("\nQuantizing ONNX to INT8...")

quantize_dynamic(
    model_input=str(ONNX_PATH),
    model_output=str(INT8_PATH),
    weight_type=QuantType.QInt8,
    per_channel=True,
    reduce_range=False,
)

print("INT8 quantization successful.")
print("File:", INT8_PATH)
print(
    "Size:",
    f"{INT8_PATH.stat().st_size / 1024 / 1024:.3f} MB",
)

# ------------------------------------------------------------
# INT8 runtime sanity check
# ------------------------------------------------------------

session_int8 = ort.InferenceSession(
    str(INT8_PATH),
    providers=["CPUExecutionProvider"],
)

int8_input_name = session_int8.get_inputs()[0].name
int8_output_name = session_int8.get_outputs()[0].name

int8_output = session_int8.run(
    [int8_output_name],
    {
        int8_input_name: sample,
    },
)[0]

print("\nINT8 runtime sanity check:")
print("Input :", int8_input_name)
print("Output:", int8_output_name)
print("Shape :", int8_output.shape)

if int8_output.shape == torch_output.shape:
    print("INT8 runtime shape check: PASS")
else:
    raise RuntimeError(
        "INT8 output shape does not match FP32 output."
    )

# ------------------------------------------------------------
# Save metadata
# ------------------------------------------------------------

metadata = {
    "source_fp32": str(FP32_PATH),
    "onnx_path": str(ONNX_PATH),
    "int8_path": str(INT8_PATH),
    "vocab_size": len(vocab),
    "embedding_dim": ED,
    "transformer_layers": NL,
    "attention_heads": NH,
    "max_length": ML,
    "num_intents": len(labels),
    "onnx_opset": 17,
    "quantization": "dynamic_weight_qint8_per_channel",
    "current_int8_baseline_modified": False,
    "onnx_fp32_torch_max_abs_diff": max_diff,
}

with open(
    OUT_DIR / "export_metadata.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metadata,
        f,
        indent=2,
    )

print("\n" + "=" * 72)
print("EXPORT COMPLETE")
print("=" * 72)

print(
    "V2 FP32 ONNX:",
    f"{ONNX_PATH.stat().st_size / 1024 / 1024:.3f} MB",
)

print(
    "V2 INT8 ONNX:",
    f"{INT8_PATH.stat().st_size / 1024 / 1024:.3f} MB",
)

print("\nIMPORTANT:")
print(
    "Current INT8 baseline was NOT modified."
)
print(
    "Do NOT replace it until V2 INT8 passes the same benchmark."
)

print("\nNext:")
print(
    "Run the exact benchmark against V2 INT8."
)
