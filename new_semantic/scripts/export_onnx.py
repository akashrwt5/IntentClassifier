#!/usr/bin/env python3
"""
Export a trained student to ONNX (+ INT8) and PROVE parity.

Repo constraints (.claude/memory/mobile.md) — non-negotiable:
  * static batch size 1, static sequence length (no dynamic axes: the ANE does
    not support a dynamic sequence dimension)
  * output is LOGITS; the confidence gate / temperature is applied at runtime
  * accuracy delta ~ 0 and 0 gate disagreements vs the PyTorch model

The script fails loudly rather than writing an artifact it could not verify.

Usage:
    python scripts/export_onnx.py --tag unkaug_s1 --threshold 0.40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts.common import encode, load_rows, load_vocab  # noqa: E402


def _prepare_for_quant(onnx_path: Path) -> Path:
    """Return a graph the dynamic quantizer can run shape inference on.

    The quantizer calls onnx.shape_inference on the model before doing anything.
    torch.onnx.export leaves stale intermediate `value_info` entries behind, and
    inference then aborts:

        InferenceError: Inferred shape and existing shape differ in
        dimension 0: (64) vs (57)

    64 is EMBED_DIM, 57 is n_classes — a wrong annotation on the classifier
    Linear. The graph computes correctly (FP32 parity passes); only the metadata
    is wrong, so dropping it and re-inferring is safe.
    """
    import onnx

    # 1) targeted fix: drop stale annotations, let inference rebuild them
    try:
        m = onnx.load(str(onnx_path))
        del m.graph.value_info[:]
        try:
            m = onnx.shape_inference.infer_shapes(m, strict_mode=False)
        except Exception as e:  # noqa: BLE001
            print(f"  re-infer skipped ({type(e).__name__}); leaving value_info empty")
            m = onnx.load(str(onnx_path))
            del m.graph.value_info[:]
        clean = onnx_path.with_name(onnx_path.stem + "_clean.onnx")
        onnx.save(m, str(clean))
        print(f"  cleared stale value_info -> {clean.name}")
        return clean
    except Exception as e:  # noqa: BLE001
        print(f"  value_info cleanup failed ({type(e).__name__}: {e})")

    # 2) fallback: ORT's own pre-processor
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = onnx_path.with_name(onnx_path.stem + "_pre.onnx")
    quant_pre_process(str(onnx_path), str(pre), skip_symbolic_shape=True)
    print(f"  quant_pre_process -> {pre.name}")
    return pre


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument(
        "--threshold", type=float, required=True, help="the gate this model is being shipped with"
    )
    ap.add_argument("--skip-int8", action="store_true")
    ap.add_argument(
        "--int8-per-tensor",
        action="store_true",
        help="use per-TENSOR int8 scales (default is per-channel). Per-tensor "
        "gives one scale for a whole weight matrix, which is poor for embedding "
        "tables where row magnitudes vary a lot — that is the usual cause of "
        "argmax flips after quantization.",
    )
    ap.add_argument(
        "--int8-keep-embedding-fp32",
        action="store_true",
        help="exclude MatMul/Gather on the embedding table from quantization. "
        "Costs size, usually removes the remaining flips.",
    )
    args = ap.parse_args()

    import torch

    from scripts.train_en import build_student

    vocab, tok_mode = load_vocab(config.MODELS / f"vocab_{args.tag}.json")
    meta_path = config.REPORTS / f"train_{args.tag}_summary.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    max_len = meta.get("max_len", config.MAX_LEN)
    tok_mode = meta.get("tokenizer", tok_mode)
    label_list = json.loads((config.MODELS / f"labels_{args.tag}.json").read_text(encoding="utf-8"))

    model = build_student(len(vocab), len(label_list), dim=meta.get("embed_dim", config.EMBED_DIM))
    model.load_state_dict(torch.load(config.MODELS / f"student_{args.tag}.pt"))
    model.eval()
    print(f"model  : {args.tag}  vocab {len(vocab)}  max_len {max_len}  tok {tok_mode}")

    onnx_path = config.MODELS / f"student_{args.tag}.onnx"
    int8_path = config.MODELS / f"student_{args.tag}_int8.onnx"

    # ---------------------------------------------------------- export
    ids = torch.zeros(1, max_len, dtype=torch.int64)
    mask = torch.ones(1, max_len, dtype=torch.bool)
    torch.onnx.export(
        model,
        (ids, mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        opset_version=14,
        do_constant_folding=True,
        # NO dynamic_axes on purpose — shapes are frozen at (1, max_len)
    )
    # torch.onnx.export writes initializers to a SIDECAR (`model.onnx.data`)
    # once they exceed a size threshold. The .onnx is then only the graph, and
    # shipping it alone gives a model with no weights.
    #
    # This bit us: 0.166 MB was recorded as the shipping size when the real
    # artifact was 0.166 + 0.787 = 0.953 MB, and INT8 was then declared "2x
    # LARGER than FP32" by comparing a graph-only file against a self-contained
    # one. INT8 is in fact ~2.7x smaller.
    #
    # Fold everything back into one file so the artifact is what it claims to be.
    sidecar = onnx_path.with_name(onnx_path.name + ".data")
    if sidecar.exists():
        import onnx as _onnx

        m = _onnx.load(str(onnx_path))  # pulls the external data in
        _onnx.save(m, str(onnx_path), save_as_external_data=False)
        sidecar.unlink()
        print(f"  folded {sidecar.name} back into the model (was external)")

    fp32_mb = onnx_path.stat().st_size / 1e6
    print(f"exported {onnx_path}  ({fp32_mb:.3f} MB, self-contained)")

    stray = list(onnx_path.parent.glob(onnx_path.name + ".*"))
    if stray:
        raise SystemExit(
            f"ABORT: sidecar files still present {[[p.name for p in stray]]} — "
            f"the artifact is not self-contained and would ship broken."
        )

    # ---------------------------------------------------------- parity
    import onnxruntime as ort

    eval_texts = []
    for p in (config.LOCKED_TEST, config.STRESS_TEST, config.OOD_TEST):
        if p.exists():
            eval_texts += [t for t, _ in load_rows(p)]
    print(f"parity set: {len(eval_texts)} utterances (locked + stress + OOD)")

    X = np.array([encode(t, vocab, max_len, tok_mode)[0] for t in eval_texts], dtype=np.int64)
    M = X != config.PAD_ID

    with torch.no_grad():
        ref = model(torch.tensor(X), torch.tensor(M)).numpy()

    def run(path):
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        out = []
        for i in range(len(X)):  # batch is fixed at 1 by design
            out.append(
                sess.run(
                    None,
                    {
                        "input_ids": X[i : i + 1],
                        "attention_mask": M[i : i + 1],
                    },
                )[
                    0
                ][0]
            )
        return np.array(out)

    def softmax(z):
        z = z - z.max(-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(-1, keepdims=True)

    def check(name, got):
        d = float(np.abs(got - ref).max())
        flips = got.argmax(1) != ref.argmax(1)
        mism = int(flips.sum())
        pr, pg = softmax(ref), softmax(got)
        gate = int(((pr.max(1) >= args.threshold) != (pg.max(1) >= args.threshold)).sum())
        print(f"\n{name}")
        print(f"  max |delta logit|      {d:.3e}")
        print(f"  argmax mismatches      {mism} / {len(ref)}")
        print(f"  gate disagreements     {gate} / {len(ref)}   (threshold {args.threshold})")

        # Are the flips near-ties (harmless) or confident errors (not harmless)?
        detail = []
        if mism:
            top2 = np.sort(pr, axis=1)[:, -2:]
            margin = top2[:, 1] - top2[:, 0]
            idx = np.where(flips)[0]
            print("  flipped rows (reference margin between top-1 and top-2):")
            for i in idx[:10]:
                a, b = label_list[ref[i].argmax()], label_list[got[i].argmax()]
                print(
                    f"     margin {margin[i]:.4f}  conf {pr[i].max():.3f}  "
                    f"{a} -> {b}   {eval_texts[i][:44]!r}"
                )
                detail.append(
                    {
                        "text": eval_texts[i],
                        "ref": a,
                        "quant": b,
                        "ref_margin": round(float(margin[i]), 4),
                        "ref_confidence": round(float(pr[i].max()), 4),
                    }
                )
            near_tie = int((margin[idx] < 0.05).sum())
            print(f"  of {mism} flips, {near_tie} are near-ties (margin < 0.05)")
        return {
            "max_abs_logit_delta": d,
            "argmax_mismatches": mism,
            "gate_disagreements": gate,
            "flips": detail,
        }

    res = {"fp32": check("FP32 ONNX vs PyTorch", run(onnx_path))}

    if not args.skip_int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        # torch.onnx.export leaves stale value_info on the graph (the classifier
        # Linear(64 -> 57) is annotated with the wrong dim), and the quantizer's
        # shape inference then aborts with
        #   "Inferred shape and existing shape differ in dimension 0: (64) vs (57)"
        # Clearing the intermediate value_info and re-inferring fixes it; the
        # graph itself is fine, only the annotations are wrong.
        src = _prepare_for_quant(onnx_path)

        qkw = {"weight_type": QuantType.QInt8, "per_channel": not args.int8_per_tensor}
        if args.int8_keep_embedding_fp32:
            import onnx as _onnx

            g = _onnx.load(str(src)).graph
            skip = [n.name for n in g.node if n.op_type in ("Gather", "GatherElements")]
            qkw["nodes_to_exclude"] = skip
            print(f"  keeping {len(skip)} embedding node(s) in fp32")
        print(f"  per_channel={qkw['per_channel']}")

        quantize_dynamic(str(src), str(int8_path), **qkw)
        int8_mb = int8_path.stat().st_size / 1e6
        print(f"\nquantized {int8_path}  ({int8_mb:.3f} MB)")

        print(
            f"  size: FP32 {fp32_mb:.3f} MB -> INT8 {int8_mb:.3f} MB "
            f"({fp32_mb / max(int8_mb, 1e-9):.2f}x)"
        )
        if int8_mb >= fp32_mb:
            print("\n  INT8 is not smaller — skipping it.")
            int8_path.unlink(missing_ok=True)
        else:
            res["int8"] = check("INT8 ONNX vs PyTorch", run(int8_path))

    manifest = {
        "tag": args.tag,
        "threshold": args.threshold,
        "tokenizer": tok_mode,
        "max_len": max_len,
        "vocab_size": len(vocab),
        "intents": len(label_list),
        "static_shape": [1, max_len],
        "dynamic_axes": False,
        "output": "logits",
        "fp32_mb": round(fp32_mb, 4),
        "int8_mb": round(int8_path.stat().st_size / 1e6, 4) if int8_path.exists() else None,
        "parity": res,
        "synthetic_text": meta.get("synthetic_text", False),
        "synthetic_rows": meta.get("synthetic_rows", 0),
    }
    out = config.REPORTS / f"export_{args.tag}.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    # ---------------------------------------------------------- ship bar
    fail = []
    if res["fp32"]["argmax_mismatches"] > 0:
        fail.append("FP32 ONNX argmax mismatch")
    if res["fp32"]["gate_disagreements"] > 0:
        fail.append("FP32 ONNX gate disagreement")
    if "int8" in res and res["int8"]["argmax_mismatches"] > 0:
        fail.append(f"INT8 argmax mismatch ({res['int8']['argmax_mismatches']})")
    if fail:
        raise SystemExit("\nPARITY FAILED: " + "; ".join(fail))
    print("\nparity OK — artifacts are safe to ship")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
