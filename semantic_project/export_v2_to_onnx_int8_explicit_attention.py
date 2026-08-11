#!/usr/bin/env python3
"""
V2 FP32 -> ONNX -> INT8 export, fully explicit attention.

Why:
PyTorch 2.x legacy ONNX export does not support the fused
aten::_transformer_encoder_layer_fwd or aten::_native_multi_head_attention
operators on this environment.

This exporter therefore implements multi-head self-attention using only
exportable tensor operations:
    Linear projections -> reshape/transpose -> MatMul -> Softmax -> MatMul
    -> output projection

The trained V2 checkpoint is loaded directly. Current INT8 baseline is
never modified.
"""

from pathlib import Path
import json
import shutil
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

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

if ED % NH != 0:
    raise ValueError(f"Embedding dim {ED} must be divisible by heads {NH}.")

HEAD_DIM = ED // NH

print("=" * 76)
print("V2 FP32 -> ONNX -> INT8")
print("FULLY EXPLICIT ATTENTION EXPORT")
print("=" * 76)
print("V2 FP32:", FP32_PATH)
print("Output:", OUT_DIR)
print("Vocab:", len(vocab))
print("Layers:", NL)
print("Heads:", NH)
print("Head dim:", HEAD_DIM)
print("Embedding:", ED)
print("FFN:", FF)
print("Max length:", ML)
print("Intents:", len(labels))


# ------------------------------------------------------------------
# Explicit Multi-Head Attention
#
# Parameter names deliberately match nn.MultiheadAttention:
#   in_proj_weight
#   in_proj_bias
#   out_proj.weight
#   out_proj.bias
#
# This lets the original V2 checkpoint load without conversion.
# ------------------------------------------------------------------

class ExplicitSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.in_proj_weight = nn.Parameter(
            torch.empty(3 * ED, ED)
        )
        self.in_proj_bias = nn.Parameter(
            torch.empty(3 * ED)
        )

        self.out_proj = nn.Linear(
            ED,
            ED,
        )

    def forward(self, x, key_padding_mask):
        batch = x.shape[0]
        seq = x.shape[1]

        # Equivalent to nn.MultiheadAttention's Q/K/V projection.
        qkv = F.linear(
            x,
            self.in_proj_weight,
            self.in_proj_bias,
        )

        q, k, v = torch.chunk(
            qkv,
            3,
            dim=-1,
        )

        # [B,S,E] -> [B,H,S,D]
        q = q.reshape(
            batch, seq, NH, HEAD_DIM
        ).transpose(1, 2)

        k = k.reshape(
            batch, seq, NH, HEAD_DIM
        ).transpose(1, 2)

        v = v.reshape(
            batch, seq, NH, HEAD_DIM
        ).transpose(1, 2)

        # Scaled dot-product attention.
        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        ) / float(HEAD_DIM) ** 0.5

        # key_padding_mask: [B,S]
        # Broadcast to [B,1,1,S].
        mask = key_padding_mask.unsqueeze(1).unsqueeze(2)

        # Avoid -inf where possible for ONNX/mobile compatibility.
        # The very negative value behaves as masked attention.
        scores = scores.masked_fill(
            mask,
            -1.0e4,
        )

        weights = torch.softmax(
            scores,
            dim=-1,
        )

        context = torch.matmul(
            weights,
            v,
        )

        # [B,H,S,D] -> [B,S,E]
        context = context.transpose(
            1, 2
        ).contiguous().reshape(
            batch, seq, ED
        )

        return self.out_proj(context)


# ------------------------------------------------------------------
# Explicit Transformer Encoder Layer
# Same norm_first=True behavior as training.
# ------------------------------------------------------------------

class ExplicitEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.self_attn = ExplicitSelfAttention()

        self.linear1 = nn.Linear(
            ED, FF
        )

        self.linear2 = nn.Linear(
            FF, ED
        )

        self.norm1 = nn.LayerNorm(ED)
        self.norm2 = nn.LayerNorm(ED)

        self.dropout = nn.Dropout(0.10)
        self.dropout1 = nn.Dropout(0.10)
        self.dropout2 = nn.Dropout(0.10)

    def forward(self, src, key_padding_mask):

        # norm_first=True:
        # src = src + self_attn(norm1(src))
        q = self.norm1(src)

        attn_out = self.self_attn(
            q,
            key_padding_mask,
        )

        src = src + self.dropout1(
            attn_out
        )

        # Feed-forward block.
        ff = self.norm2(src)

        ff = self.linear1(ff)
        ff = F.gelu(ff)
        ff = self.dropout(ff)
        ff = self.linear2(ff)

        src = src + self.dropout2(
            ff
        )

        return src


# ------------------------------------------------------------------
# V2 model.
# ------------------------------------------------------------------

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

        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(
            [
                ExplicitEncoderLayer()
                for _ in range(NL)
            ]
        )

        self.norm = nn.LayerNorm(ED)

        self.classifier = nn.Sequential(
            nn.Linear(ED, ED),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(
                ED,
                len(labels),
            ),
        )

    def forward(self, x):

        mask = x.eq(PAD)

        pos = torch.arange(
            x.size(1),
            device=x.device,
        ).unsqueeze(0)

        h = (
            self.embedding(x)
            + self.position(pos)
        )

        for layer in self.encoder.layers:
            h = layer(
                h,
                mask,
            )

        valid = (
            (~mask)
            .unsqueeze(-1)
            .float()
        )

        h = (
            h * valid
        ).sum(1) / valid.sum(1).clamp(
            min=1.0
        )

        return self.classifier(
            self.norm(h)
        )


# ------------------------------------------------------------------
# Load V2 weights.
# ------------------------------------------------------------------

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
        "V2 weights did not match exporter architecture.\n"
        f"Missing: {missing}\n"
        f"Unexpected: {unexpected}"
    )

model.eval()

print("\nV2 FP32 weights loaded successfully.")


# ------------------------------------------------------------------
# Prepare output.
# ------------------------------------------------------------------

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

for p in [
    VOCAB_PATH,
    CONFIG_PATH,
    LABELS_PATH,
]:
    shutil.copy2(
        p,
        OUT_DIR / p.name,
    )


# ------------------------------------------------------------------
# Torch sanity.
# ------------------------------------------------------------------

dummy = torch.zeros(
    (1, ML),
    dtype=torch.long,
)

with torch.no_grad():
    torch_out = (
        model(dummy)
        .cpu()
        .numpy()
    )

print("\nTorch sanity:")
print("Output shape:", torch_out.shape)


# ------------------------------------------------------------------
# Export.
# Static batch=1 is intentional for the mobile deployment graph.
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

onnx_model = onnx.load(
    str(ONNX_PATH)
)

onnx.checker.check_model(
    onnx_model
)

print("ONNX export successful.")
print(
    "FP32 ONNX size:",
    f"{ONNX_PATH.stat().st_size / 1024 / 1024:.3f} MB",
)


# ------------------------------------------------------------------
# ONNX Runtime FP32 validation.
# ------------------------------------------------------------------

session = ort.InferenceSession(
    str(ONNX_PATH),
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

ort_out = session.run(
    [output_name],
    {
        input_name: dummy.numpy()
    },
)[0]

max_diff = float(
    np.max(
        np.abs(
            ort_out - torch_out
        )
    )
)

print("\nTorch vs ONNX:")
print("Input :", input_name)
print("Output:", output_name)
print("Shape :", ort_out.shape)
print(
    "Max absolute difference:",
    max_diff,
)

if np.allclose(
    ort_out,
    torch_out,
    rtol=1e-3,
    atol=1e-4,
):
    print(
        "Torch/ONNX numerical check: PASS"
    )
else:
    print(
        "WARNING: Torch/ONNX difference "
        "is larger than expected."
    )


# ------------------------------------------------------------------
# INT8.
# ------------------------------------------------------------------

print("\nQuantizing ONNX to INT8...")

quantize_dynamic(
    model_input=str(ONNX_PATH),
    model_output=str(INT8_PATH),
    weight_type=QuantType.QInt8,
    per_channel=True,
    reduce_range=False,
)

print(
    "INT8 quantization successful."
)

print(
    "INT8 ONNX size:",
    f"{INT8_PATH.stat().st_size / 1024 / 1024:.3f} MB",
)


# ------------------------------------------------------------------
# INT8 runtime check.
# ------------------------------------------------------------------

int8_session = ort.InferenceSession(
    str(INT8_PATH),
    providers=["CPUExecutionProvider"],
)

int8_input = (
    int8_session
    .get_inputs()[0]
    .name
)

int8_output = (
    int8_session
    .get_outputs()[0]
    .name
)

int8_out = int8_session.run(
    [int8_output],
    {
        int8_input: dummy.numpy()
    },
)[0]

print("\nINT8 runtime check:")
print("Input :", int8_input)
print("Output:", int8_output)
print("Shape :", int8_out.shape)

if int8_out.shape != torch_out.shape:
    raise RuntimeError(
        "INT8 output shape mismatch: "
        f"{int8_out.shape} vs "
        f"{torch_out.shape}"
    )

print(
    "INT8 runtime shape check: PASS"
)


# ------------------------------------------------------------------
# Metadata.
# ------------------------------------------------------------------

metadata = {
    "source_fp32": str(FP32_PATH),
    "onnx_fp32": str(ONNX_PATH),
    "onnx_int8": str(INT8_PATH),
    "vocab_size": len(vocab),
    "embedding_dim": ED,
    "transformer_layers": NL,
    "attention_heads": NH,
    "head_dim": HEAD_DIM,
    "feedforward_dim": FF,
    "max_length": ML,
    "num_intents": len(labels),
    "opset": 17,
    "export_type": "fully_explicit_attention",
    "quantization": (
        "dynamic_weight_qint8_per_channel"
    ),
    "torch_onnx_max_abs_diff": max_diff,
    "current_int8_baseline_modified": False,
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

print("\n" + "=" * 76)
print("EXPORT COMPLETE")
print("=" * 76)
print(
    "FP32 ONNX:",
    ONNX_PATH,
)
print(
    "INT8 ONNX:",
    INT8_PATH,
)
print(
    "\nCurrent INT8 baseline was NOT modified."
)
print(
    "Next: benchmark V2 INT8 against "
    "the locked Current INT8 baseline."
)
