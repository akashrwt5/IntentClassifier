#!/usr/bin/env python3
"""
V2 FP32 -> ONNX -> INT8 export, fixed for PyTorch 2.x Transformer export.

Why this version exists:
PyTorch's legacy ONNX exporter cannot export the fused
aten::_transformer_encoder_layer_fwd operator produced by
nn.TransformerEncoderLayer.

This script keeps the SAME V2 weights and architecture semantics, but
implements the Transformer encoder explicitly using:
    MultiheadAttention
    Linear
    GELU
    LayerNorm
so the ONNX graph contains exportable primitive operators.

The current 0.236 MB INT8 baseline is never modified.
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

for p in [FP32_PATH, VOCAB_PATH, CONFIG_PATH, LABELS_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"Missing V2 artifact: {p}")

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError as e:
    raise ImportError(
        "Install dependencies with:\n"
        "python3 -m pip install onnx onnxruntime"
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
print("V2 FP32 -> ONNX -> INT8 EXPORT (EXPLICIT TRANSFORMER)")
print("=" * 72)
print("V2 FP32:", FP32_PATH)
print("Output:", OUT_DIR)
print("Vocab:", len(vocab))
print("Layers:", NL)
print("Heads:", NH)
print("Embedding:", ED)
print("FFN:", FF)
print("Max length:", ML)
print("Intents:", len(labels))


# ------------------------------------------------------------------
# Explicit encoder layer
#
# This matches the trained TransformerEncoderLayer:
#   norm_first=True
#   activation='gelu'
#   batch_first=True
#
# State-dict names intentionally match the original layer names:
#   self_attn, linear1, linear2, norm1, norm2
# ------------------------------------------------------------------

class ExplicitEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=ED,
            num_heads=NH,
            dropout=0.10,
            batch_first=True,
        )

        self.linear1 = nn.Linear(ED, FF)
        self.linear2 = nn.Linear(FF, ED)

        self.norm1 = nn.LayerNorm(ED)
        self.norm2 = nn.LayerNorm(ED)

        self.dropout = nn.Dropout(0.10)
        self.dropout1 = nn.Dropout(0.10)
        self.dropout2 = nn.Dropout(0.10)

    def forward(self, src, key_padding_mask):
        # Original layer was norm_first=True:
        # x = x + self_attn(norm1(x))
        # x = x + linear2(dropout(gelu(linear1(norm2(x)))))
        q = self.norm1(src)

        attn_out, _ = self.self_attn(
            q,
            q,
            q,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        src = src + self.dropout1(attn_out)

        ff = self.norm2(src)
        ff = self.linear2(
            self.dropout(
                torch.nn.functional.gelu(
                    self.linear1(ff)
                )
            )
        )

        src = src + self.dropout2(ff)
        return src


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            len(vocab), ED, padding_idx=PAD
        )

        self.position = nn.Embedding(
            ML, ED
        )

        # Use ModuleList, but preserve the original state-dict
        # hierarchy: encoder.layers.0..., encoder.layers.1...
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(
            [ExplicitEncoderLayer() for _ in range(NL)]
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

        for layer in self.encoder.layers:
            h = layer(h, mask)

        valid = (~mask).unsqueeze(-1).float()

        h = (
            h * valid
        ).sum(1) / valid.sum(1).clamp(min=1.0)

        return self.classifier(self.norm(h))


model = Model()

state = torch.load(
    FP32_PATH,
    map_location="cpu",
)

if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]

missing, unexpected = model.load_state_dict(
    state,
    strict=False,
)

if missing or unexpected:
    raise RuntimeError(
        "V2 weights did not match the explicit export architecture.\n"
        f"Missing: {missing}\n"
        f"Unexpected: {unexpected}"
    )

model.eval()

print("\nLoaded V2 FP32 weights into explicit export architecture.")


OUT_DIR.mkdir(parents=True, exist_ok=True)

for p in [VOCAB_PATH, CONFIG_PATH, LABELS_PATH]:
    shutil.copy2(p, OUT_DIR / p.name)


# ------------------------------------------------------------------
# Torch sanity check
# ------------------------------------------------------------------

dummy = torch.zeros((1, ML), dtype=torch.long)

with torch.no_grad():
    torch_out = model(dummy).cpu().numpy()

print("\nTorch sanity check:")
print("Output shape:", torch_out.shape)


# ------------------------------------------------------------------
# ONNX export
# ------------------------------------------------------------------

print("\nExporting FP32 ONNX...")

with torch.no_grad():
    torch.onnx.export(
        model,
        dummy,
        str(ONNX_PATH),
        input_names=["input_ids"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

onnx_model = onnx.load(str(ONNX_PATH))
onnx.checker.check_model(onnx_model)

print("ONNX export successful.")
print(
    "FP32 ONNX size:",
    f"{ONNX_PATH.stat().st_size / 1024 / 1024:.3f} MB",
)


# ------------------------------------------------------------------
# ONNX Runtime FP32 numerical validation
# ------------------------------------------------------------------

session = ort.InferenceSession(
    str(ONNX_PATH),
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

ort_out = session.run(
    [output_name],
    {input_name: dummy.numpy()},
)[0]

max_diff = float(
    np.max(np.abs(ort_out - torch_out))
)

print("\nTorch vs ONNX:")
print("Input:", input_name)
print("Output:", output_name)
print("Shape:", ort_out.shape)
print("Max absolute difference:", max_diff)

if np.allclose(
    ort_out,
    torch_out,
    rtol=1e-3,
    atol=1e-4,
):
    print("Torch/ONNX check: PASS")
else:
    print("WARNING: Torch/ONNX numerical difference is larger than expected.")


# ------------------------------------------------------------------
# INT8 quantization
# ------------------------------------------------------------------

print("\nQuantizing ONNX to INT8...")

quantize_dynamic(
    model_input=str(ONNX_PATH),
    model_output=str(INT8_PATH),
    weight_type=QuantType.QInt8,
    per_channel=True,
    reduce_range=False,
)

print("INT8 quantization successful.")
print(
    "INT8 ONNX size:",
    f"{INT8_PATH.stat().st_size / 1024 / 1024:.3f} MB",
)


# ------------------------------------------------------------------
# INT8 runtime validation
# ------------------------------------------------------------------

int8_session = ort.InferenceSession(
    str(INT8_PATH),
    providers=["CPUExecutionProvider"],
)

int8_input = int8_session.get_inputs()[0].name
int8_output = int8_session.get_outputs()[0].name

int8_out = int8_session.run(
    [int8_output],
    {int8_input: dummy.numpy()},
)[0]

print("\nINT8 runtime check:")
print("Input:", int8_input)
print("Output:", int8_output)
print("Shape:", int8_out.shape)

if int8_out.shape != torch_out.shape:
    raise RuntimeError(
        f"INT8 output shape mismatch: "
        f"{int8_out.shape} vs {torch_out.shape}"
    )

print("INT8 runtime shape check: PASS")


# ------------------------------------------------------------------
# Metadata
# ------------------------------------------------------------------

metadata = {
    "source_fp32": str(FP32_PATH),
    "onnx_fp32": str(ONNX_PATH),
    "onnx_int8": str(INT8_PATH),
    "vocab_size": len(vocab),
    "embedding_dim": ED,
    "transformer_layers": NL,
    "attention_heads": NH,
    "feedforward_dim": FF,
    "max_length": ML,
    "num_intents": len(labels),
    "opset": 17,
    "exporter": "legacy_torchscript_explicit_transformer",
    "quantization": "dynamic_weight_qint8_per_channel",
    "torch_onnx_max_abs_diff": max_diff,
    "current_int8_baseline_modified": False,
}

with open(
    OUT_DIR / "export_metadata.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 72)
print("EXPORT COMPLETE")
print("=" * 72)
print("FP32 ONNX:", ONNX_PATH)
print("INT8 ONNX:", INT8_PATH)
print("\nCurrent INT8 baseline was NOT modified.")
print("Next: benchmark V2 INT8 against the locked Current INT8 baseline.")
