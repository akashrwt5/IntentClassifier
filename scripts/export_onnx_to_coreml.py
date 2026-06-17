#!/usr/bin/env python3
"""
Convert minilm-l6-v2.onnx DIRECTLY to CoreML, to close the ONNX->CoreML gap.

Background
----------
The semantic head (train_semantic_head.py) was trained on embeddings produced
by the INT8 ONNX model (embedder="onnx"). The shipping FP16 CoreML embedder,
however, was built from the raw FP32 PyTorch model via torch.jit.trace in
export_coreml.py. That is a DIFFERENT embedder than the head learned on, which
is why compare_coreml_quant.py shows ~4% lower in-scope accuracy and only a
0.96 cosine between ONNX and FP16 CoreML embeddings.

This script builds the CoreML embedder from the SAME minilm-l6-v2.onnx file the
head was trained against, so the embeddings land in the head's expected space.
If it works, the ONNX->CoreML accuracy gap should largely disappear.

This script is SELF-CONTAINED and does NOT modify export_coreml.py or
compare_coreml_quant.py. It writes a new artifact:
    models/MiniLMEmbedder_fromONNX.mlpackage

After running, point the comparison at it (rename to MiniLMEmbedder.mlpackage,
or copy over it) and re-run compare_coreml_quant.py to confirm the gap closed.

Conversion strategy
-------------------
coremltools 9 removed the ONNX frontend (ct.convert only accepts
tensorflow / pytorch / milinternal). So we go ONNX -> PyTorch via onnx2torch,
then trace the resulting nn.Module and convert with the pytorch frontend.

CAVEAT: minilm-l6-v2.onnx is dynamically INT8-quantized (download_minilm.py
uses avx512_vnni, is_static=False). onnx2torch may not support the quantized
ops (MatMulInteger / DynamicQuantizeLinear). If conversion fails on those, the
script prints a clear diagnosis and the recommended fallbacks.

Usage
-----
    pip install onnx onnx2torch torch coremltools onnxruntime numpy
    python scripts/export_onnx_to_coreml.py
"""

import sys
from pathlib import Path

import numpy as np

BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
ONNX_PATH  = MODELS_DIR / "minilm-l6-v2.onnx"
DST_PATH   = MODELS_DIR / "MiniLMEmbedder_fromONNX.mlpackage"
VOCAB_PATH = MODELS_DIR / "minilm-vocab.txt"

MAX_LEN = 64

# A few holdout-style probes for the inline cosine sanity check.
PROBE_TEXTS = [
    "give me a power reading",
    "soften the audio for me",
    "add some more loudness please",
    "what is the weather today",
    "remind me to call chris tomorrow",
]


def _fail(msg):
    print(f"\nFAILED: {msg}")
    sys.exit(1)


# ── Tokeniser (shared WordPiece, same as the runtime) ─────────────────────────

def _make_tokeniser():
    sys.path.insert(0, str(BASE_DIR))
    from scripts.nlu.semantic import SemanticFallback

    if not VOCAB_PATH.exists():
        _fail(f"{VOCAB_PATH} not found — run scripts/download_minilm.py")
    with open(VOCAB_PATH, encoding="utf-8") as f:
        vocab = {line.strip(): i for i, line in enumerate(f)}

    def tokenise(text):
        toks = ["[CLS]"] + SemanticFallback._wordpiece(text.lower(), vocab) + ["[SEP]"]
        toks = toks[:MAX_LEN]
        ids = [vocab.get(t, vocab["[UNK]"]) for t in toks]
        return ids, len(ids)

    return tokenise


def _meanpool_normalise(token_emb, n):
    """token_emb: [seq, 384] -> L2-normalised [384] (mask is all-ones here)."""
    vec = token_emb[:n].mean(axis=0)
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > 0 else vec.astype(np.float32)


# ── Step 1: load ONNX -> PyTorch ──────────────────────────────────────────────

def load_onnx_as_torch():
    try:
        import onnx
        from onnx2torch import convert as onnx2torch_convert
    except ImportError as e:
        _fail(
            "onnx2torch (and onnx) are required.\n"
            "    pip install onnx onnx2torch\n"
            f"    (import error: {e})"
        )

    if not ONNX_PATH.exists():
        _fail(f"{ONNX_PATH} not found — run scripts/download_minilm.py")

    print(f"  Loading ONNX graph: {ONNX_PATH.name}")
    onnx_model = onnx.load(str(ONNX_PATH))

    # Report the graph's input names/order — onnx2torch forward() is positional.
    input_names = [i.name for i in onnx_model.graph.input]
    print(f"  ONNX inputs (in order): {input_names}")

    # Detect dynamic-INT8 ops up front so the failure message is actionable.
    op_types = {node.op_type for node in onnx_model.graph.node}
    quant_ops = op_types & {
        "MatMulInteger", "DynamicQuantizeLinear", "ConvInteger",
        "QLinearMatMul", "QLinearConv", "QuantizeLinear", "DequantizeLinear",
    }
    if quant_ops:
        print(f"  NOTE: ONNX contains quantization ops: {sorted(quant_ops)}")
        print("        onnx2torch support for these is limited; if conversion")
        print("        fails below, see the fallback guidance at the end.")

    print("  Converting ONNX -> PyTorch (onnx2torch) ...")
    try:
        torch_model = onnx2torch_convert(onnx_model)
    except Exception as e:
        _diagnose_onnx2torch_failure(e, quant_ops)
    torch_model.eval()
    return torch_model, input_names


def _diagnose_onnx2torch_failure(exc, quant_ops):
    print(f"\n  onnx2torch conversion failed: {exc}")
    print("\n  Why this likely happened:")
    if quant_ops:
        print("    The ONNX model is dynamically INT8-quantized. onnx2torch cannot")
        print("    translate ops like MatMulInteger / DynamicQuantizeLinear into")
        print("    PyTorch, so the graph can't be rebuilt for CoreML conversion.")
    else:
        print("    An op in the ONNX graph has no onnx2torch translation.")
    print("\n  Recommended fallbacks (in order of effort):")
    print("    1. Re-export a NON-quantized FP32 ONNX and convert that instead:")
    print("         python scripts/export_onnx_to_coreml.py --reexport-fp32")
    print("       (This produces minilm-l6-v2-fp32.onnx and converts from it.")
    print("        It will NOT be bit-identical to the INT8 ONNX the head trained")
    print("        on, but FP32 is much closer to it than the raw HF trace was.)")
    print("    2. Retrain the head on the FP16-CoreML embeddings instead, so the")
    print("       head matches the embedder iOS actually runs (train_semantic_head.py).")
    print("    3. Run the INT8 ONNX directly on-device via onnxruntime-mobile (SPM).")
    print("       This guarantees a 0% gap but adds ~8 MB of runtime.")
    sys.exit(1)


# ── Step 2: trace + convert to CoreML ─────────────────────────────────────────

def convert_to_coreml(torch_model, input_names):
    import torch
    import coremltools as ct

    # Wrap so the traced graph returns a single named tensor.
    class Wrapper(torch.nn.Module):
        def __init__(self, m, names):
            super().__init__()
            self.m = m
            self.names = names

        def forward(self, input_ids, attention_mask, token_type_ids):
            # onnx2torch GraphModule takes positional args in ONNX input order.
            ordered = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
            args = [ordered[n] for n in self.names if n in ordered]
            out = self.m(*args)
            # onnx2torch may return a tensor or a tuple/list — take the first.
            if isinstance(out, (tuple, list)):
                out = out[0]
            return out

    wrapper = Wrapper(torch_model, input_names).eval()

    ids  = torch.ones((1, 8), dtype=torch.long)
    mask = torch.ones((1, 8), dtype=torch.long)
    tids = torch.zeros((1, 8), dtype=torch.long)

    print("  Tracing PyTorch graph ...")
    try:
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, (ids, mask, tids), strict=False)
    except Exception as e:
        _fail(f"torch.jit.trace failed: {e}")

    print("  Converting traced model to CoreML (FP16) ...")
    seq = ct.RangeDim(lower_bound=1, upper_bound=MAX_LEN, default=8)
    try:
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="input_ids",      shape=(1, seq), dtype=np.int32),
                ct.TensorType(name="attention_mask", shape=(1, seq), dtype=np.int32),
                ct.TensorType(name="token_type_ids", shape=(1, seq), dtype=np.int32),
            ],
            outputs=[ct.TensorType(name="last_hidden_state")],
            minimum_deployment_target=ct.target.iOS16,
            compute_precision=ct.precision.FLOAT16,
        )
    except Exception as e:
        _fail(f"ct.convert failed: {e}")

    mlmodel.short_description = "MiniLM-L6-v2 encoder (converted from minilm-l6-v2.onnx)"
    mlmodel.output_description["last_hidden_state"] = "Token embeddings [1, seq, 384] — mean-pool in Swift"
    mlmodel.save(str(DST_PATH))
    size_mb = sum(f.stat().st_size for f in DST_PATH.rglob("*") if f.is_file()) / 1e6
    print(f"  Saved {DST_PATH}  ({size_mb:.1f} MB)")
    return mlmodel


# ── Step 3: inline sanity check — cosine vs the ONNX reference ─────────────────

def sanity_check(tokenise):
    import coremltools as ct
    try:
        import onnxruntime as ort
    except ImportError:
        print("\n  (skipping cosine check — onnxruntime not installed)")
        return

    print(f"\n  {'─'*60}")
    print("  Sanity check: ONNX vs new CoreML embeddings (cosine)")
    print(f"  {'─'*60}")

    onnx_sess = ort.InferenceSession(str(ONNX_PATH))
    cml = ct.models.MLModel(str(DST_PATH), compute_units=ct.ComputeUnit.CPU_ONLY)

    cosines = []
    for text in PROBE_TEXTS:
        ids, n = tokenise(text)
        # ONNX
        o = onnx_sess.run(None, {
            "input_ids":      np.array([ids], dtype=np.int64),
            "attention_mask": np.ones((1, n), dtype=np.int64),
            "token_type_ids": np.zeros((1, n), dtype=np.int64),
        })
        vo = _meanpool_normalise(o[0][0], n)
        # CoreML
        c = cml.predict({
            "input_ids":      np.array([ids], dtype=np.int32),
            "attention_mask": np.ones((1, n), dtype=np.int32),
            "token_type_ids": np.zeros((1, n), dtype=np.int32),
        })
        vc = _meanpool_normalise(np.array(c[next(iter(c))])[0], n)
        cos = float(np.dot(vo, vc))
        cosines.append(cos)
        print(f"    cos={cos:.4f}   \"{text}\"")

    mean = float(np.mean(cosines))
    print(f"\n  mean cosine ONNX vs new CoreML: {mean:.4f}")
    if mean >= 0.999:
        print("  ✅ Near-perfect match — the ONNX->CoreML gap should be gone.")
        print("     Rename MiniLMEmbedder_fromONNX.mlpackage -> MiniLMEmbedder.mlpackage")
        print("     and re-run compare_coreml_quant.py to confirm.")
    elif mean >= 0.99:
        print("  ✅ Close match — gap should largely close. Verify with the comparison.")
    else:
        print("  ⚠️  Still drifting. The INT8 ONNX could not be matched in FP16.")
        print("     Consider retraining the head on CoreML embeddings, or running")
        print("     the ONNX directly on-device via onnxruntime-mobile.")


# ── Optional: re-export an FP32 ONNX (fallback path) ──────────────────────────

def reexport_fp32():
    """Export a NON-quantized FP32 ONNX, which onnx2torch can convert cleanly."""
    fp32_path = MODELS_DIR / "minilm-l6-v2-fp32.onnx"
    print(f"  Re-exporting FP32 ONNX to {fp32_path.name} ...")
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        import shutil, tempfile
    except ImportError as e:
        _fail(f"optimum required for --reexport-fp32: {e}\n    pip install optimum[onnxruntime]")

    tmp = Path(tempfile.mkdtemp())
    m = ORTModelForFeatureExtraction.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2", export=True
    )
    m.save_pretrained(str(tmp))
    src = next(tmp.glob("model.onnx"), None)
    if src is None:
        _fail("FP32 ONNX export produced no model.onnx")
    shutil.move(str(src), str(fp32_path))
    print(f"  Saved {fp32_path}")
    print("  NOTE: edit ONNX_PATH at the top of this script to point at the FP32")
    print("        file, or copy it over minilm-l6-v2.onnx, then re-run without")
    print("        --reexport-fp32.")


def main():
    reexport = "--reexport-fp32" in sys.argv

    print("=" * 64)
    print("  ONNX -> CoreML direct conversion (close the 4% gap)")
    print("=" * 64)

    if reexport:
        reexport_fp32()
        return

    tokenise = _make_tokeniser()
    torch_model, input_names = load_onnx_as_torch()
    convert_to_coreml(torch_model, input_names)
    sanity_check(tokenise)

    print(f"\n{'='*64}")
    print("  Done. New artifact: models/MiniLMEmbedder_fromONNX.mlpackage")
    print("  Compare it against the existing models:")
    print("    cp models/MiniLMEmbedder.mlpackage models/MiniLMEmbedder.fp16.bak  # backup")
    print("    cp -r models/MiniLMEmbedder_fromONNX.mlpackage models/MiniLMEmbedder.mlpackage")
    print("    python scripts/compare_coreml_quant.py")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
