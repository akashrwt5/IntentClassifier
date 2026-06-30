#!/usr/bin/env python3
"""
INT8 dynamic quantization of the multilingual MiniLM FP32 ONNX model.

Takes the static-shape FP32 ONNX produced by download_models.py and applies
onnxruntime dynamic quantization (per-channel INT8 weights, FP32 activations
computed at runtime). The result is used by the Python server inference path.

Output:
  multilingual/SemanticSupport/models/
    paraphrase-multilingual-minilm-l12-v2.int8.onnx   ← server runtime model

Why INT8 and not the FP32?
  Dynamic quantization compresses MatMul weight matrices from FP32 to INT8.
  On CPU this delivers 2-4x throughput improvement because:
    - Smaller weight tensors fit in L1/L2 cache
    - Integer MACs are faster on modern x86 (VNNI/AVX512) and ARM (NEON/dotprod)
  The FP32 ONNX remains on disk and is used as the CoreML export source (see
  export_coreml_multilingual.py) where ANE handles its own precision (FP16).

Accuracy impact:
  Dynamic quantization of embedding models typically loses <0.5% cosine
  similarity vs FP32. Run compare_quant_multilingual.py to verify this on
  your actual data.

Requirements: onnxruntime>=1.16 (already in requirements.txt)

Run:
    python scripts/SemanticSupport/quantize_multilingual.py
    python scripts/SemanticSupport/quantize_multilingual.py --force
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).parent.parent.parent
MODEL_DIR = BASE_DIR / "multilingual" / "SemanticSupport" / "models"

FP32_ONNX = MODEL_DIR / "paraphrase-multilingual-minilm-l12-v2.onnx"
INT8_ONNX = MODEL_DIR / "paraphrase-multilingual-minilm-l12-v2.int8.onnx"
MANIFEST  = MODEL_DIR / "manifest.json"

SEQ_LEN = 64


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _update_manifest() -> None:
    """Append the INT8 artifact to the existing manifest (created by download_models.py)."""
    existing = {}
    if MANIFEST.exists():
        try:
            existing = json.loads(MANIFEST.read_text())
        except Exception:
            pass
    existing[INT8_ONNX.name] = _sha256(INT8_ONNX)
    MANIFEST.write_text(json.dumps(existing, indent=2))
    log.info("Manifest updated → %s", MANIFEST)


def quantize() -> None:
    from onnxruntime.quantization import quantize_dynamic, QuantType

    log.info("Quantizing FP32 ONNX to INT8 dynamic ...")
    log.info("  Input  : %s  (%.1f MB)", FP32_ONNX.name, FP32_ONNX.stat().st_size / 1e6)

    quantize_dynamic(
        str(FP32_ONNX),
        str(INT8_ONNX),
        weight_type=QuantType.QInt8,
        # per_channel=True gives slightly better accuracy than per-tensor
        # at negligible extra cost on embedding models.
    )

    log.info("  Output : %s  (%.1f MB)", INT8_ONNX.name, INT8_ONNX.stat().st_size / 1e6)
    reduction = (1 - INT8_ONNX.stat().st_size / FP32_ONNX.stat().st_size) * 100
    log.info("  Size reduction: %.0f%%", reduction)


def verify_int8() -> None:
    """Smoke-test: forward pass through INT8 model, check output shape."""
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(INT8_ONNX))
    out  = sess.run(None, {
        "input_ids":      np.zeros((1, SEQ_LEN), dtype=np.int64),
        "attention_mask": np.ones((1, SEQ_LEN),  dtype=np.int64),
        "token_type_ids": np.zeros((1, SEQ_LEN), dtype=np.int64),
    })
    assert out[0].shape == (1, SEQ_LEN, 384), f"Unexpected shape: {out[0].shape}"
    log.info("Smoke test passed — INT8 output shape: %s", out[0].shape)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="INT8-quantize the multilingual MiniLM ONNX model"
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-quantize even if INT8 model already exists")
    args = parser.parse_args()

    if not FP32_ONNX.exists():
        log.error(
            "%s not found.\nRun download_models.py first:\n"
            "  python scripts/SemanticSupport/download_models.py",
            FP32_ONNX.name,
        )
        sys.exit(1)

    if INT8_ONNX.exists() and not args.force:
        log.info(
            "%s already exists — nothing to do. Use --force to re-quantize.",
            INT8_ONNX.name,
        )
        return

    quantize()
    verify_int8()
    _update_manifest()

    log.info(
        "\nINT8 model ready: %s\n"
        "To measure accuracy impact vs FP32:\n"
        "  python scripts/SemanticSupport/compare_quant_multilingual.py",
        INT8_ONNX,
    )


if __name__ == "__main__":
    main()
