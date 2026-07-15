#!/usr/bin/env python3
"""
Download and export paraphrase-multilingual-MiniLM-L12-v2 to a static-shape
FP32 ONNX graph suitable for direct use and for downstream CoreML/ANE compilation.

The static shapes (batch=1, seq=64) are a hard requirement for the Apple Neural
Engine: ANE does not support dynamic sequence dimensions. Shapes are frozen at
export time by omitting dynamic_axes entirely from torch.onnx.export.

Artifacts produced (multilingual/SemanticSupport/models/):
  paraphrase-multilingual-minilm-l12-v2.onnx  — FP32 static-shape graph
  multilingual-vocab.txt                       — WordPiece vocabulary (line N = token id N)
  tokenizer_config.json                        — do_lower_case, seq_len, vocab_size
  manifest.json                                — SHA-256 checksums of all artifacts

This script is idempotent: if all artifacts exist and checksums match, it exits
immediately without re-downloading or re-exporting. Use --force to re-run.

For INT8 quantization of the FP32 ONNX for server-side Python inference, run:
  python scripts/SemanticSupport/quantize_multilingual.py

For CoreML / ANE export, run:
  python scripts/SemanticSupport/export_coreml_multilingual.py

Requirements (only for this script — not for inference):
  pip install torch transformers onnx onnxruntime
  (optimum is NOT required — we use torch.onnx.export directly for shape control)

Inference only requires: onnxruntime numpy
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR  = Path(__file__).resolve().parents[4]
OUT_DIR   = BASE_DIR / "multilingual" / "SemanticSupport" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FP32_ONNX = OUT_DIR / "paraphrase-multilingual-minilm-l12-v2.onnx"
VOCAB_TXT = OUT_DIR / "multilingual-vocab.txt"
TOK_CFG   = OUT_DIR / "tokenizer_config.json"
MANIFEST  = OUT_DIR / "manifest.json"

MODEL_ID  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEQ_LEN   = 64   # frozen into ONNX graph — must match tokeniser.py SEQ_LEN


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(paths: list) -> None:
    data = {p.name: _sha256(p) for p in paths if p.exists()}
    MANIFEST.write_text(json.dumps(data, indent=2))
    log.info("Manifest written → %s", MANIFEST)


def _manifest_ok() -> bool:
    if not MANIFEST.exists():
        return False
    try:
        recorded = json.loads(MANIFEST.read_text())
    except Exception:
        return False
    for path in (FP32_ONNX, VOCAB_TXT, TOK_CFG):
        if not path.exists():
            return False
        if recorded.get(path.name) != _sha256(path):
            return False
    return True


def _check_deps() -> None:
    missing = []
    for pkg in ("torch", "transformers", "onnx", "onnxruntime"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        log.error(
            "Missing packages: %s\n"
            "Install with: pip install torch transformers onnx onnxruntime",
            " ".join(missing),
        )
        sys.exit(1)


# ── Export ────────────────────────────────────────────────────────────────────

def export_fp32_onnx(hf_cache_dir: Path) -> None:
    """Export to a fully static-shape FP32 ONNX graph.

    Key ANE / CoreML design decisions:
    - dynamic_axes is intentionally omitted → all input/output shapes are baked
      in as constants (1, 64). The ANE requires this; dynamic shapes fall back
      to GPU or CPU and lose the low-power acceleration.
    - opset 14: stable, well-supported by both onnxruntime and coremltools.
    - do_constant_folding=True: folds BN/LayerNorm scale/bias into weights at
      export time, reducing the number of runtime ops.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    log.info("Loading %s from HuggingFace (cache: %s)", MODEL_ID, hf_cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=hf_cache_dir)
    model     = AutoModel.from_pretrained(MODEL_ID, cache_dir=hf_cache_dir)
    model.eval()

    dummy = {
        "input_ids":      torch.zeros((1, SEQ_LEN), dtype=torch.long),
        "attention_mask": torch.ones((1, SEQ_LEN),  dtype=torch.long),
        "token_type_ids": torch.zeros((1, SEQ_LEN), dtype=torch.long),
    }

    log.info("Exporting FP32 ONNX — static shape (1, %d), no dynamic axes ...", SEQ_LEN)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
            str(FP32_ONNX),
            input_names  = ["input_ids", "attention_mask", "token_type_ids"],
            output_names = ["last_hidden_state", "pooler_output"],
            opset_version        = 14,
            do_constant_folding  = True,
            # No dynamic_axes — shapes (1, 64) are frozen for ANE/CoreML.
        )
    log.info("FP32 ONNX saved → %s  (%.1f MB)", FP32_ONNX.name,
             FP32_ONNX.stat().st_size / 1e6)

    # ── Save vocab.txt (line N = token with id N) ─────────────────────────
    vocab         = tokenizer.get_vocab()
    sorted_tokens = sorted(vocab.items(), key=lambda kv: kv[1])
    VOCAB_TXT.write_text("\n".join(tok for tok, _ in sorted_tokens), encoding="utf-8")
    log.info("Vocab saved → %s  (%d tokens)", VOCAB_TXT.name, len(sorted_tokens))

    tok_cfg = {
        "model_id":      MODEL_ID,
        "do_lower_case": getattr(tokenizer, "do_lower_case", False),
        "seq_len":       SEQ_LEN,
        "vocab_size":    len(sorted_tokens),
    }
    TOK_CFG.write_text(json.dumps(tok_cfg, indent=2))
    log.info("Tokenizer config saved → %s", TOK_CFG.name)


def verify_onnx(path: Path) -> None:
    """Forward-pass smoke test: verify output shape is (1, SEQ_LEN, 384)."""
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path))
    out  = sess.run(None, {
        "input_ids":      __import__("numpy").zeros((1, SEQ_LEN), dtype=__import__("numpy").int64),
        "attention_mask": __import__("numpy").ones((1, SEQ_LEN),  dtype=__import__("numpy").int64),
        "token_type_ids": __import__("numpy").zeros((1, SEQ_LEN), dtype=__import__("numpy").int64),
    })
    import numpy as np
    assert out[0].shape == (1, SEQ_LEN, 384), f"Unexpected output shape: {out[0].shape}"
    log.info("Smoke test passed — %s output shape: %s", path.name, out[0].shape)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and export multilingual MiniLM to static-shape FP32 ONNX"
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-export even if artifacts already exist")
    parser.add_argument("--hf-cache", type=Path,
                        default=Path.home() / ".cache" / "huggingface",
                        help="HuggingFace cache directory")
    args = parser.parse_args()

    if not args.force and _manifest_ok():
        log.info(
            "All artifacts present and checksums match — nothing to do.\n"
            "         Use --force to re-export."
        )
        return

    _check_deps()
    export_fp32_onnx(args.hf_cache)
    verify_onnx(FP32_ONNX)
    _write_manifest([FP32_ONNX, VOCAB_TXT, TOK_CFG])

    log.info(
        "\nArtifacts ready:\n"
        "  %s  (FP32 ONNX — use directly or as CoreML source)\n"
        "  %s  (vocab)\n"
        "  %s  (manifest)\n\n"
        "Next steps:\n"
        "  Quantize for server inference : python scripts/SemanticSupport/quantize_multilingual.py\n"
        "  Export to CoreML / ANE        : python scripts/SemanticSupport/export_coreml_multilingual.py\n"
        "  Train the semantic head       : python scripts/SemanticSupport/train_multilingual_semantic_head.py",
        FP32_ONNX.name, VOCAB_TXT.name, MANIFEST.name,
    )


if __name__ == "__main__":
    main()
