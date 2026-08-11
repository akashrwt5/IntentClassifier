#!/usr/bin/env python3
"""
export_v3_fp32_to_onnx.py

Production-candidate export for V3 FP32.

Source:
  tiny_semantic_student_v3_error_driven/student_v3_fp32.pt

Output:
  tiny_semantic_student_v3_fp32/
    v3_semantic_student_fp32.onnx
    labels.json
    export_manifest.json

IMPORTANT:
  - FP32 only. No INT8 quantization.
  - V2 INT8 is untouched.
  - The 595-row unseen test is NOT used.
  - The exported ONNX graph is checked against PyTorch on several fixed
    sentences before the script reports success.
  - The model uses fixed batch=1 and max_length=24, matching the mobile
    inference use case.
"""

from pathlib import Path
import json
import sys
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "tiny_semantic_student_v3_error_driven"
PT = BASE / "student_v3_fp32.pt"
VOCAB_JSON = ROOT / "tiny_semantic_student_v2_balanced" / "vocab.json"
OUT = ROOT / "tiny_semantic_student_v3_fp32"
OUT.mkdir(parents=True, exist_ok=True)
ONNX = OUT / "v3_semantic_student_fp32.onnx"

ED = 64
NH = 4
FF = 128
NL = 2
ML = 24
DROPOUT = 0.10

LABELS = [
    "device.memory.change",
    "device.volume.decrease",
    "device.volume.increase",
    "device.volume.mute",
    "device.volume.unmute",
    "find.phone.locate",
    "help.reminder.show",
    "reminders.task.complete",
    "reminders.task.create",
    "streaming.session.start",
    "streaming.session.stop",
]

if not PT.exists():
    raise FileNotFoundError(PT)
if not VOCAB_JSON.exists():
    raise FileNotFoundError(VOCAB_JSON)

raw = json.loads(VOCAB_JSON.read_text(encoding="utf-8"))
if isinstance(raw.get("model"), dict) and isinstance(raw["model"].get("vocab"), dict):
    vocab = raw["model"]["vocab"]
elif isinstance(raw.get("vocab"), dict):
    vocab = raw["vocab"]
else:
    vocab = raw
vocab = {str(k): int(v) for k, v in vocab.items()}

PAD = int(vocab.get("<pad>", vocab.get("[PAD]", 0)))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(
            len(vocab), ED, padding_idx=PAD
        )
        self.position = nn.Embedding(ML, ED)

        layer = nn.TransformerEncoderLayer(
            ED,
            NH,
            FF,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, NL)
        self.norm = nn.LayerNorm(ED)
        self.classifier = nn.Sequential(
            nn.Linear(ED, ED),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(ED, len(LABELS)),
        )

    def forward(self, x):
        mask = x.eq(PAD)
        pos = torch.arange(
            x.size(1),
            device=x.device
        ).unsqueeze(0)

        h = self.encoder(
            self.embedding(x) + self.position(pos),
            src_key_padding_mask=mask,
        )

        valid = (~mask).unsqueeze(-1).float()
        h = (h * valid).sum(1) / valid.sum(1).clamp(min=1)

        return self.classifier(self.norm(h))

# Load V3.
model = Model()
state = torch.load(PT, map_location="cpu")
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
model.load_state_dict(state, strict=True)
model.eval()

# Disable training-only behavior and freeze parameters.
for p in model.parameters():
    p.requires_grad_(False)

# Fixed batch=1 is intentional: the mobile app sends one utterance at a time.
dummy = torch.zeros((1, ML), dtype=torch.long)

with torch.no_grad():
    torch_ref = model(dummy).cpu().numpy()

print("=" * 78)
print("V3 FP32 -> ONNX EXPORT")
print("=" * 78)
print("PyTorch checkpoint :", PT)
print("Vocab size         :", len(vocab))
print("Transformer layers :", NL)
print("Attention heads    :", NH)
print("Embedding          :", ED)
print("FFN                :", FF)
print("Max length         :", ML)
print("Export batch       : 1")
print()

# Modern PyTorch exporter requires onnxscript.
try:
    import onnx  # noqa: F401
    import onnxscript  # noqa: F401
except ImportError as e:
    print("Missing ONNX exporter dependency:", e)
    print()
    print("Run inside the SAME .venv:")
    print("  python3 -m pip install -U onnx onnxscript onnxruntime")
    print()
    print("Then rerun this script.")
    sys.exit(2)

# Use the modern exporter. Fixed shapes avoid the batch/reshape issue seen
# during earlier experiments with a dynamic batch dimension.
print("Exporting FP32 ONNX...")
try:
    torch.onnx.export(
        model,
        (dummy,),
        str(ONNX),
        input_names=["input_ids"],
        output_names=["logits"],
        opset_version=18,
        dynamo=True,
        external_data=False,
    )
except Exception as e:
    print()
    print("ONNX export FAILED.")
    print("Error:", repr(e))
    print()
    print("V2 INT8 was not modified.")
    print("No production artifact was promoted.")
    raise

if not ONNX.exists():
    raise RuntimeError("Exporter returned without creating the ONNX file.")

print("Exported:", ONNX)
print("Size: %.3f MB" % (ONNX.stat().st_size / 1024 / 1024))

# Validate ONNX structure.
import onnx
onnx_model = onnx.load(str(ONNX))
onnx.checker.check_model(onnx_model)
print("ONNX checker: PASS")

# Runtime check.
session = ort.InferenceSession(
    str(ONNX),
    providers=["CPUExecutionProvider"],
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("ONNX input :", input_name)
print("ONNX output:", output_name)
print("Input shape:", session.get_inputs()[0].shape)
print("Output shape:", session.get_outputs()[0].shape)

# Compare PyTorch and ONNX logits on fixed cases.
tests = {
    "zero": np.zeros((1, ML), dtype=np.int64),
    "increase_like": np.array(
        [[2, 15, 31, 48] + [PAD] * (ML - 4)],
        dtype=np.int64
    ),
    "decrease_like": np.array(
        [[2, 19, 44, 57] + [PAD] * (ML - 4)],
        dtype=np.int64
    ),
    "mute_like": np.array(
        [[2, 11, 29, 63] + [PAD] * (ML - 4)],
        dtype=np.int64
    ),
}

max_abs = 0.0
for name, ids in tests.items():
    with torch.no_grad():
        pt_logits = model(
            torch.from_numpy(ids)
        ).cpu().numpy()

    ort_logits = session.run(
        [output_name],
        {input_name: ids}
    )[0]

    diff = float(np.max(np.abs(pt_logits - ort_logits)))
    max_abs = max(max_abs, diff)

    pt_pred = int(np.argmax(pt_logits[0]))
    ort_pred = int(np.argmax(ort_logits[0]))

    print(
        f"Runtime check | {name:<18} "
        f"max_abs_diff={diff:.7f} "
        f"prediction={'PASS' if pt_pred == ort_pred else 'FAIL'}"
    )

    if pt_pred != ort_pred:
        raise RuntimeError(
            f"PyTorch/ONNX prediction mismatch on test '{name}'."
        )

print("Maximum absolute logit difference:", max_abs)
print("PyTorch -> ONNX runtime equivalence: PASS")

# Save labels.
(OUT / "labels.json").write_text(
    json.dumps(LABELS, indent=2),
    encoding="utf-8"
)

manifest = {
    "model": "tiny_semantic_student_v3_error_driven",
    "format": "ONNX FP32",
    "checkpoint": str(PT),
    "onnx": str(ONNX),
    "size_mb": ONNX.stat().st_size / 1024 / 1024,
    "vocab_size": len(vocab),
    "transformer_layers": NL,
    "attention_heads": NH,
    "embedding_dim": ED,
    "feed_forward": FF,
    "max_length": ML,
    "batch_size": 1,
    "opset": 18,
    "quantized": False,
    "v2_int8_modified": False,
    "unseen_test_used": False,
    "runtime_equivalence": True,
    "max_abs_logit_difference": max_abs,
    "next_step": (
        "Run the complete V3 FP32 ONNX benchmark against the locked V2 INT8 "
        "baseline before any mobile integration."
    ),
}

(OUT / "export_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8"
)

print()
print("=" * 78)
print("EXPORT SUCCESS")
print("=" * 78)
print("FP32 ONNX:", ONNX)
print("Size      : %.3f MB" % (ONNX.stat().st_size / 1024 / 1024))
print("Quantized : NO")
print("Runtime   : PASS")
print()
print("V2 INT8 was NOT modified.")
print("V3 INT8 was NOT created.")
print("595-row unseen test was NOT used.")
print()
print("NEXT:")
print("Benchmark this V3 FP32 ONNX artifact using the SAME")
print("unseen/contextual/targeted/OOD suite before mobile deployment.")
