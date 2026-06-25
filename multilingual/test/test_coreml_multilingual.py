#!/usr/bin/env python3
"""
CoreML <-> ONNX numeric-equivalence tests for the multilingual intent
classifiers.

Two tiers:

  Tier-A  (default; runs ANYWHERE incl. Linux/CI — no Core ML runtime needed)
    1. Load each .mlpackage SPEC: assert fixed (1, V) input, 'logits' output,
       class labels embedded, and 'temperature' metadata present and == the
       weights-JSON value.
    2. Extract the linear weight matrix + bias FROM the CoreML package and assert
       they equal the JSON coef/intercept within FP16 tolerance (proves the
       export carried the right numbers).
    3. Run a NumPy reference scorer (identical math to scripts/test_ios_conformance
       ._ios_predict) over each model's holdout and assert it matches the ONNX raw
       logits within tolerance. Transitively this proves CoreML == ONNX.

  Tier-B  (--runtime; macOS-only, auto-skips elsewhere)
    Loads each .mlpackage with the real Core ML runtime, runs predict() over the
    holdout, and compares CoreML logits vs ONNX logits vs the NumPy reference;
    reports accuracy delta and 0.70-gate disagreements on the 30-utterance
    conformance set.

Extra:
  --emit-fixtures   Write multilingual/test/coreml_golden_fixtures.json (S6) —
                    the cross-language source of truth for the iOS XCTest.

Usage:
    python multilingual/test/test_coreml_multilingual.py
    python multilingual/test/test_coreml_multilingual.py --full
    python multilingual/test/test_coreml_multilingual.py --runtime     # macOS
    python multilingual/test/test_coreml_multilingual.py --emit-fixtures
"""

import argparse
import csv
import json
import math
import os
import platform
import re
import sys
from pathlib import Path

import numpy as np

# Ensure the ONNX string-normalizer locale exists for this process.
os.environ.setdefault("LC_ALL", "en_US.UTF-8")
os.environ.setdefault("LANG", "en_US.UTF-8")

BASE_DIR = Path(__file__).resolve().parent.parent          # multilingual/
MODELS_DIR = BASE_DIR / "models"
TEST_DIR = BASE_DIR / "test"
MODEL_NAMES = ["en", "fr", "de", "da", "multilingual", "multilingual_small"]

# FP16 weight tolerance is RELATIVE to coefficient magnitude, not absolute:
# round-to-nearest float16 has a half-ULP error of 2^-11 of the value, so the
# absolute error of the largest coefficient grows with its magnitude (e.g. a
# coef near 19 rounds to ~6e-3). A fixed absolute tol would wrongly flag the
# large-vocab models. tol = 2^-11 * max(|coef|) + floor.
FP16_REL = 2 ** -11          # float16 half-ULP (relative)
FP16_FLOOR = 1e-4            # subnormal / accumulation floor
LOGIT_REF_TOL = 1e-3         # NumPy reference vs ONNX raw logits (both ~fp32)

# The 30 conformance utterances (mirrors scripts/test_ios_conformance.py).
CONF_UTTERANCES = [
    "mute", "unmute", "turn it up", "turn it down", "increase volume",
    "decrease the volume", "change to restaurant", "switch to music",
    "send a message", "find my phone", "battery level", "start a run",
    "start walking", "how many steps", "show my calories", "set a reminder",
    "open translate", "start transcribe", "start streaming", "stop streaming",
    "check my heart rate", "open health", "help with volume", "help with pairing",
    "help with battery", "what is mask mode", "tell me about edge mode",
    "help with tinnitus", "show thrive score", "how do i insert my device",
]
CONF_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# NumPy reference scorer — identical math to _ios_predict (device path)
# ---------------------------------------------------------------------------

def swift_tokens(text: str) -> list:
    words = [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w]
    tokens = list(words)
    for i in range(len(words) - 1):
        tokens.append(words[i] + " " + words[i + 1])
    return tokens


class RefScorer:
    """Sublinear TF-IDF over the pruned vocab -> L2 normalize -> coef.x + b."""

    def __init__(self, weights: dict):
        self.vocab = weights["vocab"]
        self.idf = np.asarray(weights["idf"], dtype=np.float64)
        self.coef = np.asarray(weights["coef"], dtype=np.float64)
        self.intercept = np.asarray(weights["intercept"], dtype=np.float64)
        self.labels = [str(x) for x in weights["labels"]]
        self.T = float(weights.get("temperature", 1.0))
        # bigrams only when the model's ngram_range includes 2.
        ngr = weights.get("ngram_range", [1, 2])
        self.use_bigrams = len(ngr) >= 2 and int(ngr[1]) >= 2

    def _tokens(self, text: str):
        if self.use_bigrams:
            return swift_tokens(text)
        return [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w]

    def vector(self, text: str) -> np.ndarray:
        counts = {}
        for tok in self._tokens(text):
            idx = self.vocab.get(tok)
            if idx is not None:
                counts[idx] = counts.get(idx, 0) + 1
        vec = np.zeros(len(self.idf), dtype=np.float64)
        for idx, cnt in counts.items():
            vec[idx] = math.log(1 + cnt) * self.idf[idx]
        n = np.linalg.norm(vec)
        if n > 0:
            vec /= n
        return vec

    def logits(self, text: str) -> np.ndarray:
        return self.coef @ self.vector(text) + self.intercept

    def predict(self, text: str):
        z = self.logits(text)
        top = int(np.argmax(z))
        scaled = z / self.T
        scaled -= scaled.max()
        e = np.exp(scaled)
        probs = e / e.sum()
        return self.labels[top], float(probs[top]), z


def softmax_T(logits: np.ndarray, T: float) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / T
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


# ---------------------------------------------------------------------------
# ONNX raw-logits scorer
# ---------------------------------------------------------------------------

def onnx_logits_fn(model: str, n_classes: int):
    import onnxruntime as ort
    path = MODELS_DIR / model / f"{model}_intent_model.onnx"
    sess = ort.InferenceSession(str(path))
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


# ---------------------------------------------------------------------------
# Package helpers (build-if-missing, spec inspection, weight extraction)
# ---------------------------------------------------------------------------

def package_path(model: str) -> Path:
    return MODELS_DIR / model / f"IntentClassifier_{model}.mlpackage"


def ensure_package(model: str):
    pkg = package_path(model)
    if pkg.exists():
        return pkg
    print(f"  [{model}] package missing — building it now ...")
    sys.path.insert(0, str(BASE_DIR))
    import importlib
    exp = importlib.import_module("export_coreml_multilingual")
    import coremltools as ct
    use_torch = True
    try:
        import torch  # noqa: F401
    except ImportError:
        use_torch = False
    exp.export_model(ct, model, fp16=True, fp32=False, use_torch=use_torch)
    return pkg


def load_spec(pkg: Path):
    import coremltools as ct
    return ct.models.MLModel(str(pkg), skip_model_load=True).get_spec()


def spec_input_shape(spec):
    """Return (name, [dims]) for the single fixed input of an mlprogram."""
    inp = spec.description.input[0]
    tt = inp.type.multiArrayType
    dims = [int(d) for d in tt.shape]
    return inp.name, dims


def spec_output_names(spec):
    return [o.name for o in spec.description.output]


def spec_metadata(spec):
    return dict(spec.description.metadata.userDefined)


def extract_linear_weights(pkg: Path, spec):
    """Pull (weight (K,V), bias (K,)) back out of the saved package.

    mlprogram: weights live as fp16/fp32 consts in weights/weight.bin, referenced
    by blobFileValue offsets. neuralnetwork: weights are inline in the layer.
    """
    if spec.WhichOneof("Type") == "mlProgram":
        return _extract_mlprogram_weights(pkg, spec)
    return _extract_nn_weights(spec)


def _read_blob(reader, dtype, offset):
    if dtype == "FLOAT16":
        raw = np.asarray(reader.read_fp16_data(offset), dtype=np.uint16)
        return raw.view(np.float16).astype(np.float64)
    if dtype == "FLOAT32":
        return np.asarray(reader.read_float_data(offset), dtype=np.float64)
    raise ValueError(f"unsupported blob dtype {dtype}")


def _extract_mlprogram_weights(pkg: Path, spec):
    from coremltools.libmilstoragepython import _BlobStorageReader as BlobReader
    blob = pkg / "Data" / "com.apple.CoreML" / "weights" / "weight.bin"
    reader = BlobReader(str(blob))

    fn = spec.mlProgram.functions["main"]
    block = fn.block_specializations[next(iter(fn.block_specializations))]

    from coremltools.proto import MIL_pb2
    consts = {}   # const output_name -> np_array (decoded from the blob)
    for op in block.operations:
        if op.type != "const":
            continue
        val = op.attributes["val"]
        if not val.HasField("blobFileValue"):
            continue
        tt = val.type.tensorType
        shape = [int(d.constant.size) for d in tt.dimensions]
        dtype = MIL_pb2.DataType.Name(tt.dataType)   # 'FLOAT16' / 'FLOAT32'
        arr = _read_blob(reader, dtype, val.blobFileValue.offset)
        consts[op.outputs[0].name] = arr.reshape(shape)

    # The 'linear' op consumes (x, weight, bias); resolve weight/bias by the
    # const names bound to those inputs.
    weight = bias = None
    for op in block.operations:
        if op.type == "linear":
            wname = op.inputs["weight"].arguments[0].name
            bname = op.inputs["bias"].arguments[0].name
            weight = consts[wname]
            bias = consts[bname]
            break
    if weight is None:
        # fall back: largest 2D const is the weight, matching 1D const is bias
        twod = [(n, a) for n, a in consts.items() if a.ndim == 2]
        oned = [(n, a) for n, a in consts.items() if a.ndim == 1]
        weight = max(twod, key=lambda kv: kv[1].size)[1]
        bias = max(oned, key=lambda kv: kv[1].size)[1]
    return weight, bias


def _extract_nn_weights(spec):
    nn = spec.neuralNetwork
    for layer in nn.layers:
        if layer.HasField("innerProduct"):
            ip = layer.innerProduct
            K = int(ip.outputChannels)
            V = int(ip.inputChannels)
            W = np.asarray(ip.weights.floatValue or ip.weights.float16Value,
                           dtype=np.float64).reshape(K, V)
            b = np.asarray(ip.bias.floatValue or ip.bias.float16Value,
                           dtype=np.float64).reshape(K)
            return W, b
    raise ValueError("no innerProduct layer found in neuralnetwork spec")


# ---------------------------------------------------------------------------
# Holdout
# ---------------------------------------------------------------------------

def read_holdout(model: str, limit):
    path = TEST_DIR / f"{model}_holdout.csv"
    rows = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            txt = (r.get("text") or "").strip()
            if txt:
                rows.append((txt, (r.get("intent") or "").strip()))
    if limit and limit > 0:
        rows = rows[:limit]
    return rows


# ---------------------------------------------------------------------------
# Tier-A
# ---------------------------------------------------------------------------

def tier_a_model(model: str, limit: int) -> dict:
    weights = json.loads(_weights_path(model).read_text(encoding="utf-8"))
    coef = np.asarray(weights["coef"], dtype=np.float64)
    intercept = np.asarray(weights["intercept"], dtype=np.float64)
    K, V = coef.shape
    T = float(weights.get("temperature", 1.0))

    pkg = ensure_package(model)
    spec = load_spec(pkg)

    # 1. spec shape / outputs / labels / temperature
    in_name, dims = spec_input_shape(spec)
    assert dims == [1, V], f"[{model}] input shape {dims} != [1, {V}]"
    assert in_name == "tfidf_vector", f"[{model}] input name {in_name!r}"
    outs = spec_output_names(spec)
    assert "logits" in outs, f"[{model}] 'logits' not in outputs {outs}"
    meta = spec_metadata(spec)
    assert "temperature" in meta, f"[{model}] no temperature metadata"
    meta_T = float(meta["temperature"])
    assert abs(meta_T - T) < 1e-9, f"[{model}] meta T {meta_T} != JSON T {T}"
    labels_meta = json.loads(meta["class_labels"])
    assert labels_meta == [str(x) for x in weights["labels"]], \
        f"[{model}] embedded labels mismatch"
    assert len(labels_meta) == K

    # 2. weights extracted from package == JSON within FP16 tol
    W, b = extract_linear_weights(pkg, spec)
    assert W.shape == (K, V), f"[{model}] extracted W shape {W.shape} != {(K, V)}"
    w_delta = float(np.abs(W - coef).max())
    b_delta = float(np.abs(b - intercept).max())
    w_tol = FP16_REL * float(np.abs(coef).max()) + FP16_FLOOR
    b_tol = FP16_REL * float(np.abs(intercept).max()) + FP16_FLOOR
    assert w_delta <= w_tol, f"[{model}] weight Δ {w_delta:.2e} > FP16 tol {w_tol:.2e}"
    assert b_delta <= b_tol, f"[{model}] bias Δ {b_delta:.2e} > FP16 tol {b_tol:.2e}"

    # 3. NumPy reference logits == ONNX raw logits
    scorer = RefScorer(weights)
    onnx = onnx_logits_fn(model, K)
    rows = read_holdout(model, limit)
    max_ref_delta = 0.0
    for txt, _ in rows:
        d = float(np.abs(scorer.logits(txt) - onnx(txt)).max())
        max_ref_delta = max(max_ref_delta, d)
    assert max_ref_delta <= LOGIT_REF_TOL, \
        f"[{model}] ref-vs-ONNX Δ {max_ref_delta} > {LOGIT_REF_TOL}"

    fp16_kb = sum(f.stat().st_size for f in pkg.rglob("*") if f.is_file()) / 1024.0
    fp32_pkg = MODELS_DIR / model / f"IntentClassifier_{model}_fp32.mlpackage"
    fp32_kb = (sum(f.stat().st_size for f in fp32_pkg.rglob("*") if f.is_file()) / 1024.0
               if fp32_pkg.exists() else None)

    return {
        "model": model, "V": V, "K": K, "T": T,
        "fp16_kb": fp16_kb, "fp32_kb": fp32_kb,
        "weight_delta": w_delta, "bias_delta": b_delta, "weight_tol": w_tol,
        "ref_onnx_delta": max_ref_delta, "n_rows": len(rows),
        "verdict": "PASS",
    }


def _weights_path(model: str) -> Path:
    return MODELS_DIR / model / f"{model}_intent_classifier_weights.json"


def run_tier_a(models, limit) -> list:
    print("=" * 70)
    print("  Tier-A  (platform-independent numeric equivalence — no CoreML runtime)")
    print("=" * 70)
    results = []
    failures = 0
    for m in models:
        try:
            r = tier_a_model(m, limit)
            print(f"  ✅ {m:20} V={r['V']:5d}  FP16={r['fp16_kb']:7.1f}KB  "
                  f"wΔ={r['weight_delta']:.2e}  refΔ={r['ref_onnx_delta']:.2e}  "
                  f"(n={r['n_rows']})")
            results.append(r)
        except AssertionError as e:
            failures += 1
            print(f"  ❌ {m:20} {e}")
            results.append({"model": m, "verdict": "FAIL", "error": str(e)})
        except Exception as e:
            failures += 1
            print(f"  ❌ {m:20} ERROR {type(e).__name__}: {e}")
            results.append({"model": m, "verdict": "FAIL", "error": repr(e)})
    print(f"\n  Tier-A: {len(models) - failures}/{len(models)} models PASS")
    return results, failures


# ---------------------------------------------------------------------------
# Tier-B (macOS runtime)
# ---------------------------------------------------------------------------

def run_tier_b(models, limit):
    if platform.system() != "Darwin":
        print("\n" + "=" * 70)
        print("  Tier-B  (real Core ML runtime) — SKIPPED")
        print("  Reason: requires macOS (this runner is "
              f"{platform.system()}). The Core ML runtime/libcoremlpython is "
              "macOS-only. Tier-A already proved numeric equivalence here.")
        print("=" * 70)
        return None, 0

    import coremltools as ct
    print("=" * 70)
    print("  Tier-B  (real Core ML runtime, macOS)")
    print("=" * 70)
    results, failures = [], 0
    for m in models:
        try:
            weights = json.loads(_weights_path(m).read_text(encoding="utf-8"))
            K = len(weights["coef"]); T = float(weights.get("temperature", 1.0))
            scorer = RefScorer(weights)
            onnx = onnx_logits_fn(m, K)
            mlmodel = ct.models.MLModel(str(package_path(m)))
            rows = read_holdout(m, limit)
            labels = [str(x) for x in weights["labels"]]

            max_logit_d = max_conf_d = 0.0
            onnx_correct = cm_correct = 0
            for txt, gold in rows:
                x = scorer.vector(txt).astype(np.float32).reshape(1, -1)
                cm = np.asarray(mlmodel.predict({"tfidf_vector": x})["logits"]).reshape(-1)
                oz = onnx(txt)
                max_logit_d = max(max_logit_d, float(np.abs(cm - oz).max()))
                cm_p = softmax_T(cm, T); oz_p = softmax_T(oz, T)
                max_conf_d = max(max_conf_d, abs(cm_p.max() - oz_p.max()))
                if gold:
                    onnx_correct += int(labels[int(oz.argmax())] == gold)
                    cm_correct += int(labels[int(cm.argmax())] == gold)

            # 0.70 gate parity on conformance set
            disagree = 0
            for utt in CONF_UTTERANCES:
                x = scorer.vector(utt).astype(np.float32).reshape(1, -1)
                cm = np.asarray(mlmodel.predict({"tfidf_vector": x})["logits"]).reshape(-1)
                oz = onnx(utt)
                cm_fire = softmax_T(cm, T).max() >= CONF_THRESHOLD
                oz_fire = softmax_T(oz, T).max() >= CONF_THRESHOLD
                disagree += int(cm_fire != oz_fire)

            n = len(rows)
            r = {
                "model": m,
                "onnx_acc": onnx_correct / n if n else 0.0,
                "coreml_acc": cm_correct / n if n else 0.0,
                "max_logit_delta": max_logit_d,
                "max_conf_delta": max_conf_d,
                "gate_disagree": disagree,
                "agreement_pct": 100.0 * (len(CONF_UTTERANCES) - disagree) / len(CONF_UTTERANCES),
            }
            results.append(r)
            ok = disagree == 0
            failures += int(not ok)
            print(f"  {'✅' if ok else '❌'} {m:20} accΔ={r['coreml_acc']-r['onnx_acc']:+.4f}  "
                  f"logitΔ={max_logit_d:.2e}  confΔ={max_conf_d:.2e}  gate-disagree={disagree}/30")
        except Exception as e:
            failures += 1
            print(f"  ❌ {m:20} ERROR {type(e).__name__}: {e}")
            results.append({"model": m, "verdict": "FAIL", "error": repr(e)})
    return results, failures


# ---------------------------------------------------------------------------
# Golden fixtures (S6)
# ---------------------------------------------------------------------------

def emit_fixtures(models, sample_per_model=20):
    fixtures = []
    for m in models:
        weights = json.loads(_weights_path(m).read_text(encoding="utf-8"))
        scorer = RefScorer(weights)
        T = float(weights.get("temperature", 1.0))
        items = list(CONF_UTTERANCES)
        for txt, _ in read_holdout(m, sample_per_model):
            items.append(txt)
        for utt in items:
            intent, conf, z = scorer.predict(utt)
            fixtures.append({
                "utterance": utt,
                "model": m,
                "temperature": T,
                "expected_intent": intent,
                "expected_top1_confidence": round(conf, 6),
                "expected_logits": [round(float(v), 6) for v in z],
            })
    out = TEST_DIR / "coreml_golden_fixtures.json"
    out.write_text(json.dumps(fixtures, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {out}  ({len(fixtures)} fixtures across {len(models)} models)")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="all",
                        choices=["all"] + MODEL_NAMES)
    parser.add_argument("--runtime", action="store_true",
                        help="Also run Tier-B (real Core ML runtime; macOS only).")
    parser.add_argument("--emit-fixtures", action="store_true",
                        help="Write coreml_golden_fixtures.json and exit.")
    parser.add_argument("--limit", type=int, default=400,
                        help="Max holdout rows per model (default 400; 0 = all).")
    parser.add_argument("--full", action="store_true",
                        help="Use the entire holdout (overrides --limit).")
    args = parser.parse_args()

    models = MODEL_NAMES if args.model == "all" else [args.model]
    limit = 0 if args.full else args.limit

    if args.emit_fixtures:
        emit_fixtures(models)
        return

    _, a_fail = run_tier_a(models, limit)
    _, b_fail = run_tier_b(models, limit)

    total_fail = a_fail + (b_fail or 0)
    if total_fail:
        print(f"\n❌ {total_fail} failure(s).")
        sys.exit(1)
    print("\n✅ All checks passed.")


if __name__ == "__main__":
    main()
