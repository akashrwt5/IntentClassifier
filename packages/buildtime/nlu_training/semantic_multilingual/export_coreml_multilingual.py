#!/usr/bin/env python3
"""
Export the multilingual MiniLM FP32 ONNX model to CoreML (.mlpackage) for
deployment on Apple Neural Engine (ANE).

The FP32 ONNX produced by download_models.py has strictly static shapes (1, 64)
and standard operators — both are prerequisites for full ANE hardware residency.
Dynamic shapes or unsupported ops would cause the CoreML compiler to fall back
to CPU/GPU, losing the low-power ANE acceleration.

Export path:
  FP32 ONNX  →  onnx2torch (rebuild as PyTorch graph)
             →  torch.jit.trace (TorchScript)
             →  coremltools ct.convert (FP16 CoreML)

Why FP32 ONNX → CoreML (not INT8 ONNX → CoreML)?
  The INT8 ONNX contains MatMulInteger and DynamicQuantizeLinear ops that
  onnx2torch cannot translate to PyTorch. The FP32 ONNX has only standard
  transformer ops and converts cleanly. The ANE natively runs FP16 arithmetic,
  so ct.convert with compute_precision=FLOAT16 achieves full ANE acceleration
  without quantized-op complications.

Output:
  multilingual/SemanticSupport/models/
    MiniLMMultilingual.mlpackage   — FP16 CoreML model for iOS/macOS ANE

Cosine sanity check:
  After conversion the script runs a set of multilingual probe sentences
  through both ONNX and CoreML and reports the mean cosine similarity.
  A mean ≥ 0.999 confirms the CoreML embeddings land in the same space
  as the Python ONNX embeddings the head was trained on.

Requirements:
  pip install onnx onnx2torch torch coremltools onnxruntime

Run:
    python scripts/SemanticSupport/export_coreml_multilingual.py
    python scripts/SemanticSupport/export_coreml_multilingual.py --force
    python scripts/SemanticSupport/export_coreml_multilingual.py --skip-verify
"""

import argparse
import sys
from pathlib import Path

import numpy as np

BASE_DIR  = Path(__file__).resolve().parents[4]
MODEL_DIR = BASE_DIR / "multilingual" / "SemanticSupport" / "models"
sys.path.insert(0, str(BASE_DIR))

FP32_ONNX  = MODEL_DIR / "paraphrase-multilingual-minilm-l12-v2.onnx"
VOCAB_PATH = MODEL_DIR / "multilingual-vocab.txt"
OUT_PATH   = MODEL_DIR / "MiniLMMultilingual.mlpackage"

SEQ_LEN = 64

# Probe sentences used for post-conversion cosine sanity check.
# Covers all three target languages plus English to catch cross-lingual drift.
PROBE_TEXTS = [
    "Quand dois-je prendre mes médicaments ?",            # fr
    "Wann nehme ich meine Medikamente?",                   # de
    "Hvornår skal jeg tage min medicin?",                  # da
    "what time should I take my medication",               # en
    "pouvez-vous baisser le volume ?",                     # fr
    "Können Sie die Lautstärke verringern?",               # de
    "set a reminder for tomorrow morning",                  # en
]

parser = argparse.ArgumentParser(
    description="Export multilingual MiniLM FP32 ONNX to CoreML FP16"
)
parser.add_argument("--force",       action="store_true",
                    help="Re-export even if .mlpackage already exists")
parser.add_argument("--skip-verify", action="store_true",
                    help="Skip cosine sanity check (useful on non-Mac hosts)")
parser.add_argument("--seq-len",     type=int, default=SEQ_LEN,
                    help=f"Static sequence length (must match SEQ_LEN in tokeniser.py, default={SEQ_LEN})")
args = parser.parse_args()

SEQ_LEN = args.seq_len


def _check_deps() -> None:
    missing = []
    for pkg in ("onnx", "onnx2torch", "torch", "coremltools"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"ERROR: Missing packages: {', '.join(missing)}")
        print("Install with: pip install onnx onnx2torch torch coremltools onnxruntime")
        sys.exit(1)


def _embed_onnx(sess, tokeniser, texts: list) -> np.ndarray:
    """Return L2-normalised (N, 384) float32 embeddings via the ONNX session."""
    batch_ids, batch_mask, batch_types = [], [], []
    for text in texts:
        ids, mask, types = tokeniser.encode(text)
        batch_ids.append(ids[0])
        batch_mask.append(mask[0])
        batch_types.append(types[0])

    input_ids      = np.stack(batch_ids).astype(np.int64)
    attention_mask = np.stack(batch_mask).astype(np.int64)
    token_type_ids = np.stack(batch_types).astype(np.int64)

    out       = sess.run(None, {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    })
    token_emb = out[0]
    expanded  = attention_mask[:, :, np.newaxis].astype(np.float32)
    summed    = (token_emb * expanded).sum(axis=1)
    counts    = np.where(expanded.sum(axis=1) == 0, 1.0, expanded.sum(axis=1))
    vecs      = summed / counts
    norms     = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms     = np.where(norms == 0, 1.0, norms)
    return (vecs / norms).astype(np.float32)


def load_onnx_as_torch():
    """Convert FP32 ONNX → PyTorch via onnx2torch.

    The FP32 ONNX has only standard transformer ops (LayerNorm, GELU, MatMul,
    Attention) — all translatable by onnx2torch without quantization issues.
    """
    import onnx
    from onnx2torch import convert as onnx2torch_convert

    if not FP32_ONNX.exists():
        print(f"ERROR: {FP32_ONNX} not found.\n"
              "Run: python scripts/SemanticSupport/download_models.py")
        sys.exit(1)

    print(f"  Loading FP32 ONNX: {FP32_ONNX.name}  ({FP32_ONNX.stat().st_size/1e6:.1f} MB)")
    onnx_model  = onnx.load(str(FP32_ONNX))
    input_names = [i.name for i in onnx_model.graph.input]
    print(f"  ONNX inputs: {input_names}")

    # Verify no quantized ops (INT8 ops would block onnx2torch conversion).
    quant_ops = {n.op_type for n in onnx_model.graph.node} & {
        "MatMulInteger", "DynamicQuantizeLinear", "QLinearMatMul",
        "QuantizeLinear", "DequantizeLinear",
    }
    if quant_ops:
        print(f"  WARNING: quantized ops detected: {quant_ops}")
        print("  You may have pointed at the INT8 model by mistake. "
              "Use the FP32 ONNX.")

    print("  Converting ONNX → PyTorch (onnx2torch) ...")
    torch_model = onnx2torch_convert(onnx_model)
    torch_model.eval()
    return torch_model, input_names


def export_to_coreml(torch_model, input_names) -> None:
    import torch
    import coremltools as ct

    class _Wrapper(torch.nn.Module):
        """Wrap the onnx2torch GraphModule so the traced graph returns last_hidden_state."""
        def __init__(self, m, names):
            super().__init__()
            self.m     = m
            self.names = names

        def forward(self, input_ids, attention_mask, token_type_ids):
            # onnx2torch takes positional args in ONNX input order.
            arg_map = {
                "input_ids":      input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
            args = [arg_map[n] for n in self.names if n in arg_map]
            out  = self.m(*args)
            # onnx2torch may return a tuple or single tensor.
            return out[0] if isinstance(out, (tuple, list)) else out

    wrapper = _Wrapper(torch_model, input_names).eval()

    # Trace with fixed (1, SEQ_LEN) tensors — shapes are static, matching the
    # ONNX graph and the runtime tokeniser output.
    dummy_ids   = torch.zeros((1, SEQ_LEN), dtype=torch.long)
    dummy_mask  = torch.ones((1, SEQ_LEN),  dtype=torch.long)
    dummy_types = torch.zeros((1, SEQ_LEN), dtype=torch.long)

    print(f"  Tracing PyTorch graph (shape=(1, {SEQ_LEN})) ...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_ids, dummy_mask, dummy_types),
                                 strict=False)

    print("  Converting traced graph → CoreML FP16 (target: iOS16 / macOS13) ...")
    # Fixed (1, SEQ_LEN) input shapes — ANE requires static dimensions.
    # compute_precision=FLOAT16: ANE runs FP16 natively; this avoids the
    # GPU fallback that FP32 CoreML triggers on most Apple Silicon chips.
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids",      shape=(1, SEQ_LEN), dtype=np.int32),
            ct.TensorType(name="attention_mask",  shape=(1, SEQ_LEN), dtype=np.int32),
            ct.TensorType(name="token_type_ids",  shape=(1, SEQ_LEN), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="last_hidden_state")],
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16,
    )

    mlmodel.short_description = (
        "Multilingual MiniLM-L12-v2 encoder — "
        "mean-pool last_hidden_state in Swift/ObjC; "
        f"static input shape (1, {SEQ_LEN})"
    )
    mlmodel.output_description["last_hidden_state"] = (
        f"Token embeddings shape (1, {SEQ_LEN}, 384). "
        "Apply attention-mask-weighted mean pool + L2 norm to get a 384-dim sentence vector."
    )
    mlmodel.save(str(OUT_PATH))
    size_mb = sum(f.stat().st_size for f in OUT_PATH.rglob("*") if f.is_file()) / 1e6
    print(f"  Saved: {OUT_PATH.name}  ({size_mb:.1f} MB)")


def cosine_sanity_check() -> None:
    """Compare ONNX vs CoreML embeddings on multilingual probes.

    Runs on CPU_ONLY compute unit so this works on non-Mac build machines.
    On Apple Silicon, switch to ct.ComputeUnit.ALL to verify ANE execution.
    """
    import onnxruntime as ort
    import coremltools as ct
    from multilingual.SemanticSupport.tokeniser import MultilingualTokeniser

    print(f"\n  {'─'*60}")
    print("  Cosine sanity check: ONNX vs CoreML embeddings")
    print(f"  {'─'*60}")

    onnx_sess = ort.InferenceSession(str(FP32_ONNX))
    tokeniser = MultilingualTokeniser(VOCAB_PATH)

    try:
        cml = ct.models.MLModel(str(OUT_PATH), compute_units=ct.ComputeUnit.CPU_ONLY)
    except Exception as e:
        print(f"  Could not load CoreML model (likely non-Mac host): {e}")
        print("  Skipping cosine check — run on macOS to verify.")
        return

    cosines = []
    for text in PROBE_TEXTS:
        # ONNX embedding
        ids, mask, types = tokeniser.encode(text)
        out_onnx = onnx_sess.run(None, {
            "input_ids":      ids,
            "attention_mask": mask,
            "token_type_ids": types,
        })
        token_emb = out_onnx[0][0]                          # (64, 384)
        mask_f    = mask[0].astype(np.float32)              # (64,)
        summed    = (token_emb * mask_f[:, np.newaxis]).sum(axis=0)
        vec_onnx  = summed / mask_f.sum()
        n         = np.linalg.norm(vec_onnx)
        vec_onnx  = (vec_onnx / n if n > 0 else vec_onnx).astype(np.float32)

        # CoreML embedding (INT32 inputs required by ct.convert dtype=np.int32)
        cml_out = cml.predict({
            "input_ids":      ids.astype(np.int32),
            "attention_mask": mask.astype(np.int32),
            "token_type_ids": types.astype(np.int32),
        })
        lhs      = np.array(cml_out["last_hidden_state"])[0]    # (64, 384)
        summed_c = (lhs * mask_f[:, np.newaxis]).sum(axis=0)
        vec_cml  = summed_c / mask_f.sum()
        n        = np.linalg.norm(vec_cml)
        vec_cml  = (vec_cml / n if n > 0 else vec_cml).astype(np.float32)

        cos = float(np.dot(vec_onnx, vec_cml))
        cosines.append(cos)
        print(f"    cos={cos:.5f}   {text[:60]!r}")

    mean_cos = float(np.mean(cosines))
    print(f"\n  Mean cosine (ONNX FP32 vs CoreML FP16): {mean_cos:.5f}")
    if mean_cos >= 0.999:
        print("  ✅ Near-perfect match — CoreML embeddings land in the same "
              "space as the training embeddings. Head is directly compatible.")
    elif mean_cos >= 0.995:
        print("  ✅ Very close match — minor FP16 rounding. Acceptable for production.")
    else:
        print("  ⚠  Cosine below 0.995 — check for graph differences between "
              "ONNX and CoreML paths. Consider retraining head on CoreML embeddings.")


def main() -> None:
    print("\n" + "=" * 64)
    print("  Multilingual MiniLM → CoreML export")
    print("=" * 64)

    if OUT_PATH.exists() and not args.force:
        size_mb = sum(f.stat().st_size for f in OUT_PATH.rglob("*") if f.is_file()) / 1e6
        print(f"\n{OUT_PATH.name} already exists ({size_mb:.1f} MB). "
              "Use --force to re-export.")
        if not args.skip_verify:
            print("Running cosine check on existing model ...")
            cosine_sanity_check()
        return

    _check_deps()

    print("\nStep 1: Load FP32 ONNX as PyTorch graph")
    torch_model, input_names = load_onnx_as_torch()

    print("\nStep 2: Trace and convert to CoreML FP16")
    export_to_coreml(torch_model, input_names)

    if not args.skip_verify:
        print("\nStep 3: Cosine sanity check")
        cosine_sanity_check()

    print(f"\n{'='*64}")
    print(f"  Done.")
    print(f"  CoreML model : {OUT_PATH}")
    print(f"\n  Usage in Swift:")
    print(f"    let model = try MiniLMMultilingual(configuration: .init())")
    print(f"    // Input: input_ids, attention_mask, token_type_ids — all Int32, shape (1, {SEQ_LEN})")
    print(f"    // Output: last_hidden_state — Float32, shape (1, {SEQ_LEN}, 384)")
    print(f"    // Post-process: attention-mask-weighted mean pool + L2 normalise → 384-dim vector")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
