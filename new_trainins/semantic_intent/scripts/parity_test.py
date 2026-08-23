"""Phase 21 — Python vs ONNX parity, and the INT8 quality delta.

Two separate questions, kept separate:
  1. NUMERICAL parity — does the ONNX graph reproduce the Python probabilities
     within tolerance on the same inputs?
  2. DECISION parity — does the safety gate reach the same accept/reject verdict?
Decision parity is the one that can break a product; a tiny numeric drift that
flips a borderline case is a real defect even if max|delta| looks small.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from calibration import margin_of  # noqa: E402
from pipeline import DATA, IntentModel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def onnx_probs(sess, model, texts, max_len=64) -> np.ndarray:
    names = {i.name for i in sess.get_inputs()}
    if "input_ids" in names:
        tok = model.encoder.tok
        prefix = getattr(model.encoder, "prefix", "")
        enc = tok([prefix + t for t in texts], padding="max_length",
                  truncation=True, max_length=max_len, return_tensors="np")
        feeds = {"input_ids": enc["input_ids"].astype(np.int64),
                 "attention_mask": enc["attention_mask"].astype(np.int64)}
    else:
        inp = sess.get_inputs()[0]
        if "string" in inp.type:
            feeds = {inp.name: np.array(texts, dtype=object).reshape(-1, 1)}
        else:
            # graph starts at the embedding (reference encoder path — see
            # onnx/EXPORT_NOTE.txt); feed it what the encoder produces
            feeds = {inp.name: model.encoder.encode(texts).astype(np.float32)}
    names = [o.name for o in sess.get_outputs()]
    out = sess.run(None, feeds)
    # by name, not by shape: the graph now also returns whitened_embedding, and
    # matching on "second dimension equals the class count" would silently pick
    # the wrong tensor the day an encoder has as many dims as there are intents
    idx = names.index("probs") if "probs" in names else 0
    return np.asarray(out[idx], dtype=np.float64)


def compare(model, sess, texts, tol=1e-4, decision_tol=0.99) -> dict:
    # The transformer export fuses temperature scaling into the graph; the
    # reference sklearn export cannot, so compare against uncalibrated Python
    # probabilities there. Comparing a calibrated tensor to an uncalibrated one
    # would report a huge delta that is a bookkeeping artefact, not a defect.
    graph_is_calibrated = any(i.name == "input_ids" for i in sess.get_inputs())
    p_py = model.probs(texts, calibrated=graph_is_calibrated)
    p_ox = onnx_probs(sess, model, texts)
    d = np.abs(p_py - p_ox)
    top_py, top_ox = p_py.argmax(1), p_ox.argmax(1)

    gate = model.gate
    def verdicts(p):
        conf, marg = p.max(1), margin_of(p)
        idx = p.argmax(1)
        return np.array([
            (idx[i] != gate.reject_index) and conf[i] >= gate.conf_threshold
            and marg[i] >= gate.margin_threshold for i in range(len(p))])
    v_py, v_ox = verdicts(p_py), verdicts(p_ox)

    return dict(n=len(texts),
                graph_includes_temperature=bool(graph_is_calibrated),
                max_abs_delta=float(d.max()),
                mean_abs_delta=float(d.mean()),
                p99_abs_delta=float(np.quantile(d, 0.99)),
                top1_agreement=float((top_py == top_ox).mean()),
                gate_agreement=float((v_py == v_ox).mean()),
                within_tolerance=bool(d.max() <= tol),
                tolerance=tol,
                decision_parity_ok=bool((v_py == v_ox).mean() >= decision_tol),
                decision_tolerance=decision_tol,
                disagreements=[
                    dict(text=texts[i], py=model.labels[top_py[i]],
                         onnx=model.labels[top_ox[i]],
                         py_conf=float(p_py[i].max()), onnx_conf=float(p_ox[i].max()))
                    for i in np.where(top_py != top_ox)[0][:15]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/final")
    ap.add_argument("--onnx-dir", default="models/final/onnx")
    ap.add_argument("--tolerance", type=float, default=1e-3)
    ap.add_argument("--int8-tolerance", type=float, default=5e-2,
                    help="informational only; int8 passes on decision parity")
    ap.add_argument("--decision-tolerance", type=float, default=0.99,
                    help="minimum gate agreement for the int8 build to pass")
    args = ap.parse_args()

    model = IntentModel.load(ROOT / args.model)
    d = ROOT / args.onnx_dir

    # Parity inputs deliberately include the nasty suites, not just clean test.
    texts = []
    for f in ("test.csv", "hard_negative_test.csv", "negation_test.csv",
              "ood_test.csv", "stt_test.csv", "contextual_test.csv",
              "accessories_test.csv"):
        texts += pd.read_csv(DATA / f)["text"].tolist()[:400]

    # fp32 and int8 are judged on DIFFERENT criteria, on purpose.
    #
    #   fp32 must be numerically faithful: it is supposed to be the same model,
    #   so any real delta is an export bug.
    #
    #   int8 is a different model by construction — throwing away 24 bits per
    #   weight and expecting identical probabilities is incoherent. Judging it
    #   on max|delta| answers a question nobody asked. What must hold is that
    #   it reaches the same ACCEPT/REJECT verdicts and scores the same on the
    #   suites, so that is what decides its verdict. max|delta| is still
    #   reported, as information rather than as a gate.
    report = {}
    for tag, fname, tol in (("fp32", "intent_fp32.onnx", args.tolerance),
                            ("int8", "intent_int8.onnx", args.int8_tolerance)):
        p = d / fname
        if not p.exists():
            continue
        sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        r = compare(model, sess, texts, tol, args.decision_tolerance)
        r["size_mb"] = round(p.stat().st_size / 1e6, 3)
        r["criterion"] = ("numerical (max abs delta)" if tag == "fp32"
                          else "decision parity (gate agreement)")
        report[tag] = r
        passed = r["within_tolerance"] if tag == "fp32" else r["decision_parity_ok"]
        status = "PASS" if passed else "FAIL"
        print(f"[{tag}] {status}  ({r['criterion']})  max|d|={r['max_abs_delta']:.3e} "
              f"top1_agree={r['top1_agreement']:.5f} "
              f"gate_agree={r['gate_agreement']:.5f} size={r['size_mb']}MB")
        if r["disagreements"]:
            print(f"      {len(r['disagreements'])} shown top-1 disagreements, "
                  f"first: {r['disagreements'][0]}")

    (ROOT / "reports" / "parity.json").write_text(json.dumps(report, indent=2))
    print("-> reports/parity.json")


if __name__ == "__main__":
    main()
