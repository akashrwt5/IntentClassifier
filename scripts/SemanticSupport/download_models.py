#!/usr/bin/env python3
"""
Download, export, and quantize paraphrase-multilingual-MiniLM-L12-v2 for
runtime use via onnxruntime with strict static input shapes (1, 64).

The static shapes are a hard requirement for downstream Core ML / ANE
compilation: the ANE does not support dynamic sequence dimensions. Every
tensor that enters the ONNX graph has its shape frozen at export time.

Artifacts produced (all in multilingual/SemanticSupport/models/):
  paraphrase-multilingual-minilm-l12-v2.onnx       — FP32 static-shape graph
  paraphrase-multilingual-minilm-l12-v2.int8.onnx  — INT8 quantized (runtime)
  multilingual-vocab.txt                            — WordPiece vocab
  tokenizer_config.json                             — tokenizer metadata
  manifest.json                                     — SHA-256 checksums

This script is idempotent: if all artifacts already exist and their checksums
match the manifest, it exits without re-downloading or re-exporting.

Runtime requirements (only for this script, NOT for inference):
  pip install torch transformers optimum[exporters] onnx onnxruntime

Inference only requires: onnxruntime numpy
"""

import argparse
import hashlib
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent.parent.parent
OUT_DIR    = BASE_DIR / "multilingual" / "SemanticSupport" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FP32_ONNX  = OUT_DIR / "paraphrase-multilingual-minilm-l12-v2.onnx"
INT8_ONNX  = OUT_DIR / "paraphrase-multilingual-minilm-l12-v2.int8.onnx"
VOCAB_TXT  = OUT_DIR / "multilingual-vocab.txt"
TOK_CFG    = OUT_DIR / "tokenizer_config.json"
MANIFEST   = OUT_DIR / "manifest.json"

MODEL_ID   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEQ_LEN    = 64   # static sequence length — must match tokeniser.py SEQ_LEN


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(paths: list) -> None:
    data = {p.name: _sha256(p) for p in paths if p.exists()}
    MANIFEST.write_text(json.dumps(data, indent=2))
    log.info("Manifest written: %s", MANIFEST)


def _manifest_ok() -> bool:
    """Return True if all artifacts exist and their SHA-256 hashes match the manifest."""
    if not MANIFEST.exists():
        return False
    try:
        recorded = json.loads(MANIFEST.read_text())
    except Exception:
        return False
    for path in (FP32_ONNX, INT8_ONNX, VOCAB_TXT, TOK_CFG):
        if not path.exists():
            return False
        if recorded.get(path.name) != _sha256(path):
            return False
    return True


def _check_import_deps() -> None:
    missing = []
    for pkg in ("torch", "transformers", "optimum", "onnx", "onnxruntime"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        log.error(
            "Missing packages required for model export: %s\n"
            "Install with:\n"
            "  pip install torch transformers 'optimum[exporters]' onnx onnxruntime",
            " ".join(missing),
        )
        sys.exit(1)


# ── Export ───────────────────────────────────────────────────────────────────

def export_fp32_onnx(hf_cache_dir: Path) -> None:
    """Export the model to a static-shape FP32 ONNX graph.

    We use torch.onnx.export directly (rather than optimum-cli) so we can
    enforce a completely static graph: dynamic_axes={} means no axis in any
    input or output is marked as variable. This is mandatory for Core ML / ANE.

    The dummy inputs have shape (1, SEQ_LEN) — exactly the shapes that the
    tokeniser will always produce at runtime.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    log.info("Loading HuggingFace model: %s", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=hf_cache_dir)
    model     = AutoModel.from_pretrained(MODEL_ID, cache_dir=hf_cache_dir)
    model.eval()

    # Fixed-shape dummy inputs — shapes are locked into the exported graph.
    dummy = {
        "input_ids":      torch.zeros((1, SEQ_LEN), dtype=torch.long),
        "attention_mask": torch.ones((1, SEQ_LEN),  dtype=torch.long),
        "token_type_ids": torch.zeros((1, SEQ_LEN), dtype=torch.long),
    }

    log.info("Exporting FP32 ONNX with static shape (1, %d)...", SEQ_LEN)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
            str(FP32_ONNX),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["last_hidden_state", "pooler_output"],
            # dynamic_axes intentionally omitted — static shapes for ANE
            opset_version=14,
            do_constant_folding=True,
        )
    log.info("FP32 ONNX saved: %s (%.1f MB)", FP32_ONNX, FP32_ONNX.stat().st_size / 1e6)

    # ── Save tokenizer artifacts ──────────────────────────────────────────
    # Save vocab.txt so the runtime tokeniser does not need transformers.
    vocab = tokenizer.get_vocab()
    # Sort by token id to produce the standard line-index format (line N = token with id N).
    sorted_vocab = sorted(vocab.items(), key=lambda kv: kv[1])
    VOCAB_TXT.write_text("\n".join(tok for tok, _ in sorted_vocab), encoding="utf-8")
    log.info("Vocab saved: %s (%d tokens)", VOCAB_TXT, len(sorted_vocab))

    tok_cfg = {
        "model_id":      MODEL_ID,
        "do_lower_case": getattr(tokenizer, "do_lower_case", False),
        "seq_len":       SEQ_LEN,
        "vocab_size":    len(sorted_vocab),
    }
    TOK_CFG.write_text(json.dumps(tok_cfg, indent=2))
    log.info("Tokenizer config saved: %s", TOK_CFG)


def quantize_int8() -> None:
    """Produce INT8 dynamic quantization of the FP32 graph for CPU/server use.

    onnxruntime dynamic quantization targets MatMul and Embedding operators,
    which account for >95% of the model's arithmetic. The resulting INT8
    model is ~25% of the FP32 size and 2-4x faster on CPU with no accuracy
    loss on embedding tasks.

    Note: ANE inference uses the FP16 Core ML path (compiled from FP32 ONNX
    separately). The INT8 ONNX is used for the Python server stage only.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType
    log.info("Quantizing to INT8...")
    quantize_dynamic(
        str(FP32_ONNX),
        str(INT8_ONNX),
        weight_type=QuantType.QInt8,
    )
    log.info("INT8 ONNX saved: %s (%.1f MB)", INT8_ONNX, INT8_ONNX.stat().st_size / 1e6)


def verify_onnx_outputs() -> None:
    """Smoke-test: run a forward pass through the INT8 model and check output shape."""
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(INT8_ONNX))
    dummy_ids   = np.zeros((1, SEQ_LEN), dtype=np.int64)
    dummy_mask  = np.ones((1, SEQ_LEN),  dtype=np.int64)
    dummy_types = np.zeros((1, SEQ_LEN), dtype=np.int64)

    outputs = sess.run(None, {
        "input_ids":      dummy_ids,
        "attention_mask": dummy_mask,
        "token_type_ids": dummy_types,
    })
    lhs = outputs[0]
    assert lhs.shape == (1, SEQ_LEN, 384), f"Unexpected output shape: {lhs.shape}"
    log.info("INT8 ONNX smoke test passed — output shape: %s", lhs.shape)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download and export multilingual MiniLM")
    parser.add_argument("--force", action="store_true",
                        help="Re-export even if artifacts already exist")
    parser.add_argument("--hf-cache", type=Path,
                        default=Path.home() / ".cache" / "huggingface",
                        help="HuggingFace cache directory (default: ~/.cache/huggingface)")
    args = parser.parse_args()

    if not args.force and _manifest_ok():
        log.info("All artifacts present and checksums match — nothing to do. "
                 "Use --force to re-export.")
        return

    _check_import_deps()

    if not FP32_ONNX.exists() or args.force:
        export_fp32_onnx(args.hf_cache)
    else:
        log.info("FP32 ONNX already exists, skipping export.")

    if not INT8_ONNX.exists() or args.force:
        quantize_int8()
    else:
        log.info("INT8 ONNX already exists, skipping quantization.")

    verify_onnx_outputs()
    _write_manifest([FP32_ONNX, INT8_ONNX, VOCAB_TXT, TOK_CFG])

    log.info(
        "\nArtifacts ready in: %s\n"
        "  FP32 ONNX : %s\n"
        "  INT8 ONNX : %s  ← used at runtime\n"
        "  Vocab     : %s\n"
        "  Manifest  : %s",
        OUT_DIR, FP32_ONNX.name, INT8_ONNX.name, VOCAB_TXT.name, MANIFEST.name,
    )


if __name__ == "__main__":
    main()
