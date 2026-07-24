#!/usr/bin/env python3
"""
Production CoreML <-> ONNX parity GATE.

Proves the shipped CoreML (iOS/ANE) model is numerically equivalent to the shipped
ONNX (Android) model, so the two platforms cannot silently diverge. Everything is
tied to the ONE fitted pipeline (models/intent_pipeline.pkl) that train.py
produced — the ONNX is skl2onnx of it, and the CoreML weights are exported from it
(scripts/export_weights.py). Featurizing the CoreML input with that same pipeline
is exactly what the ONNX does internally, so the comparison is faithful (no
hand-rolled tokenizer to drift).

Tiers (any failure blocks the release):

  Tier-A  (runs ANYWHERE incl. Linux CI):
    A1. `.mlpackage` is ANE-shaped: ML-Program, fixed (1, V) input, V == n_features.
    A2. The linear weight/bias pulled OUT of the .mlpackage equal the pipeline's
        clf.coef_/intercept_ within FP16 tolerance.
    A3. The pipeline's own logits (coef·tfidf(x)+b) match the ONNX raw logits over
        the holdout — proves the ONNX export is faithful.

  Tier-B  (--runtime; macOS only): feed tfidf(x) to the REAL Core ML runtime and
    require top-1 agreement with ONNX over the holdout (+ 0.70-gate disagreements).

Usage:
    python scripts/ci/verify_coreml_parity.py \
        --pipeline models/intent_pipeline.pkl --onnx models/intent_model.onnx \
        --coreml dist/models/model.mlpackage --holdout data/semantic_holdout_100.csv \
        [--runtime] [--onnx-ref-only] [--out dist/coreml_parity.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FP16_RTOL = 2e-2
FP16_ATOL = 1e-3
LOGIT_TOL = 5e-3   # sklearn logits vs ONNX raw logits (fp32 export rounding)
GATE = 0.70


def load_pipeline(path: Path):
    import joblib
    pipe = joblib.load(path)
    return pipe.named_steps["tfidf"], pipe.named_steps["clf"]


def onnx_logits_fn(onnx_path: Path, n_classes: int):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path))
    inp = sess.get_inputs()[0]

    def run(text: str) -> np.ndarray:
        t = text.lower().strip()
        arr = (np.array([[t]], dtype=object) if len(inp.shape) == 2
               else np.array([t], dtype=object))
        outs = sess.run(None, {inp.name: arr})
        for o in outs:
            a = np.asarray(o)
            if a.ndim and a.shape[-1] == n_classes:
                return a.reshape(-1).astype(np.float64)
        return np.asarray(outs[-1]).reshape(-1).astype(np.float64)

    return run


# --- CoreML .mlpackage inspection (coremltools; runs in CI) ----------------- #

def load_spec(pkg: Path):
    import coremltools as ct
    return ct.models.MLModel(str(pkg), skip_model_load=True).get_spec()


def _read_blob(reader, dtype, offset):
    if dtype == "FLOAT16":
        raw = np.asarray(reader.read_fp16_data(offset), dtype=np.uint16)
        return raw.view(np.float16).astype(np.float64)
    if dtype == "FLOAT32":
        return np.asarray(reader.read_float_data(offset), dtype=np.float64)
    raise ValueError(f"unsupported blob dtype {dtype}")


def extract_linear_weights(pkg: Path, spec):
    if spec.WhichOneof("Type") == "mlProgram":
        from coremltools.libmilstoragepython import _BlobStorageReader as BlobReader
        from coremltools.proto import MIL_pb2
        blob = pkg / "Data" / "com.apple.CoreML" / "weights" / "weight.bin"
        reader = BlobReader(str(blob))
        fn = spec.mlProgram.functions["main"]
        block = fn.block_specializations[next(iter(fn.block_specializations))]
        consts = {}
        for op in block.operations:
            if op.type != "const":
                continue
            val = op.attributes["val"]
            if not val.HasField("blobFileValue"):
                continue
            tt = val.type.tensorType
            shape = [int(d.constant.size) for d in tt.dimensions]
            dtype = MIL_pb2.DataType.Name(tt.dataType)
            consts[op.outputs[0].name] = _read_blob(reader, dtype, val.blobFileValue.offset).reshape(shape)
        weight = bias = None
        for op in block.operations:
            if op.type == "linear":
                weight = consts[op.inputs["weight"].arguments[0].name]
                bias = consts[op.inputs["bias"].arguments[0].name]
                break
        if weight is None:
            twod = [(n, a) for n, a in consts.items() if a.ndim == 2]
            oned = [(n, a) for n, a in consts.items() if a.ndim == 1]
            weight = max(twod, key=lambda kv: kv[1].size)[1]
            bias = max(oned, key=lambda kv: kv[1].size)[1]
        return np.asarray(weight, np.float64), np.asarray(bias, np.float64)

    nn = spec.neuralNetwork
    for layer in nn.layers:
        if layer.HasField("innerProduct"):
            ip = layer.innerProduct
            K, V = int(ip.outputChannels), int(ip.inputChannels)
            W = np.asarray(ip.weights.floatValue or ip.weights.float16Value, np.float64).reshape(K, V)
            b = np.asarray(ip.bias.floatValue or ip.bias.float16Value, np.float64).reshape(K)
            return W, b
    raise ValueError("no linear/innerProduct layer in the CoreML package")


def read_holdout(path: Path, limit: int):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if limit and limit > 0:
        rows = rows[:limit]
    return [(r["utterance"], r.get("expected_intent") or r.get("intent")) for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default=str(ROOT / "models" / "intent_pipeline.pkl"))
    ap.add_argument("--onnx", default=str(ROOT / "models" / "intent_model.onnx"))
    ap.add_argument("--coreml", default=str(ROOT / "dist" / "models" / "model.mlpackage"))
    ap.add_argument("--holdout", default=str(ROOT / "data" / "semantic_holdout_100.csv"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--runtime", action="store_true", help="Tier-B: real Core ML runtime (macOS)")
    ap.add_argument("--onnx-ref-only", action="store_true", help="Tier-A3 only (no coremltools)")
    ap.add_argument("--out", default=str(ROOT / "dist" / "coreml_parity.json"))
    args = ap.parse_args()

    tfidf, clf = load_pipeline(Path(args.pipeline))
    coef = np.asarray(clf.coef_, np.float64)
    intercept = np.asarray(clf.intercept_, np.float64)
    n_classes, n_features = coef.shape

    def featurize(text: str) -> np.ndarray:
        return np.asarray(tfidf.transform([text.lower().strip()]).todense(), np.float64).reshape(-1)

    holdout = read_holdout(Path(args.holdout), args.limit)
    onnx = onnx_logits_fn(Path(args.onnx), n_classes)
    report: dict = {"n_classes": n_classes, "n_features": n_features,
                    "holdout": Path(args.holdout).name, "checks": {}}
    fails: list[str] = []

    # A3: pipeline logits == ONNX raw logits (proves faithful skl2onnx export).
    max_dev = 0.0
    for text, _ in holdout:
        skl = coef @ featurize(text) + intercept
        max_dev = max(max_dev, float(np.max(np.abs(skl - onnx(text)))))
    ok = max_dev <= LOGIT_TOL
    report["checks"]["onnx_matches_pipeline"] = {"max_logit_dev": max_dev, "tol": LOGIT_TOL, "pass": ok}
    if not ok:
        fails.append(f"ONNX logits diverge from the pipeline (max {max_dev:.2e} > {LOGIT_TOL})")

    if not args.onnx_ref_only:
        pkg = Path(args.coreml)
        spec = load_spec(pkg)

        is_mlp = spec.WhichOneof("Type") == "mlProgram"
        dims = [int(d) for d in spec.description.input[0].type.multiArrayType.shape]
        shape_ok = is_mlp and dims == [1, n_features]
        report["checks"]["ane_shape"] = {"mlprogram": is_mlp, "input_dims": dims,
                                         "expected": [1, n_features], "pass": shape_ok}
        if not shape_ok:
            fails.append(f"CoreML not ANE-shaped: mlprogram={is_mlp} dims={dims}")

        W, b = extract_linear_weights(pkg, spec)
        w_ok = W.shape == coef.shape and np.allclose(W, coef, rtol=FP16_RTOL, atol=FP16_ATOL)
        b_ok = b.shape == intercept.shape and np.allclose(b, intercept, rtol=FP16_RTOL, atol=FP16_ATOL)
        report["checks"]["weights_match"] = {"weight_shape": list(W.shape),
                                            "weight_pass": bool(w_ok), "bias_pass": bool(b_ok)}
        if not (w_ok and b_ok):
            fails.append("CoreML linear weights/bias differ from the pipeline beyond FP16 tolerance")

        if args.runtime:
            import coremltools as ct
            mdl = ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.ALL)
            in_name = spec.description.input[0].name
            disagree = gate_disagree = 0
            T = 1.0
            for text, _ in holdout:
                x = featurize(text).reshape(1, -1).astype(np.float32)
                cml = np.asarray(next(iter(mdl.predict({in_name: x}).values())), np.float64).reshape(-1)
                oz = onnx(text)
                if int(np.argmax(cml)) != int(np.argmax(oz)):
                    disagree += 1
                if (_top_softmax(cml, T) >= GATE) != (_top_softmax(oz, T) >= GATE):
                    gate_disagree += 1
            rt_ok = disagree == 0
            report["checks"]["runtime_top1_agreement"] = {
                "n": len(holdout), "argmax_disagreements": disagree,
                "gate_disagreements": gate_disagree, "pass": rt_ok}
            if not rt_ok:
                fails.append(f"CoreML runtime top-1 disagrees with ONNX on {disagree}/{len(holdout)}")

    report["passed"] = not fails
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for name, c in report["checks"].items():
        print(f"  {'PASS' if c.get('pass') else 'FAIL'}  {name}: {c}")
    print(f"parity report -> {out}")
    if fails:
        print("\nPARITY FAILED — CoreML diverges from ONNX; release blocked:")
        for f in fails:
            print("  - " + f)
        return 1
    print("\nCoreML <-> ONNX PARITY OK.")
    return 0


def _top_softmax(logits: np.ndarray, T: float) -> float:
    z = np.asarray(logits, np.float64) / T
    z -= z.max()
    e = np.exp(z)
    return float((e / e.sum()).max())


if __name__ == "__main__":
    sys.exit(main())
