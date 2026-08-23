"""Phase 26 — run the FULL evaluation matrix through an ONNX session.

Parity alone is not enough. Parity answers "does the graph reproduce Python on
these inputs"; it does not answer "is the quantized model still good enough to
ship". The plan is explicit: the quantized model must pass the full evaluation
suite again. This script is what makes that claim checkable.

    python scripts/evaluate_onnx.py --model models/final \
           --onnx models/final/onnx/intent_int8.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent))
from calibration import SafetyGate  # noqa: E402
from evaluate_model import headline, run_all  # noqa: E402
from pipeline import IntentModel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class OnnxIntentModel:
    """Presents an ONNX session through the same surface as IntentModel, so the
    evaluation code is byte-for-byte the same for Python and ONNX. If the suites
    were re-implemented for ONNX, a difference in the harness could masquerade
    as a difference in the model."""

    def __init__(self, ref: IntentModel, onnx_path: Path, max_len: int = 64, batch_size: int = 64):
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.labels = ref.labels
        self.temperature = ref.temperature
        self.tok = ref.encoder.tok
        self.prefix = getattr(ref.encoder, "prefix", "")
        self.max_len = max_len
        self.batch_size = batch_size
        self.gate = SafetyGate(
            ref.gate.conf_threshold,
            ref.gate.margin_threshold,
            ref.labels,
            ref.gate.reject_label,
            temperature=1.0,  # graph already applied T
            ood_threshold=ref.gate.ood_threshold,
            ood_percentile=ref.gate.ood_percentile,
            risk_of=ref.gate.risk_of,
            conf_by_risk=ref.gate.conf_by_risk,
            reject_corrective=ref.gate.reject_corrective,
        )
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.centroids = (
            ref.ood.mu_white_
            if getattr(ref, "ood", None) is not None and ref.ood.method == "mahalanobis"
            else None
        )

    def _run(self, texts):
        texts = [self.prefix + t for t in list(texts)]
        probs, emb = [], []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tok(
                texts[i : i + self.batch_size],
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="np",
            )
            o = self.sess.run(
                None,
                {
                    "input_ids": enc["input_ids"].astype(np.int64),
                    "attention_mask": enc["attention_mask"].astype(np.int64),
                },
            )
            probs.append(o[self.out_names.index("probs")])
            if "whitened_embedding" in self.out_names:
                emb.append(o[self.out_names.index("whitened_embedding")])
        return (
            np.vstack(probs).astype(np.float64),
            np.vstack(emb).astype(np.float64) if emb else None,
        )

    def probs(self, texts, calibrated: bool = True) -> np.ndarray:
        return self._run(texts)[0]

    def decide(self, texts) -> list:
        """The whole runtime decision computed the way the phone will compute
        it: everything from the graph's own two outputs, nothing recomputed in
        Python. If this disagrees with the Python model, the app will too."""
        p, Z = self._run(texts)
        ood = None
        if Z is not None and self.centroids is not None:
            d2 = (
                (Z**2).sum(1)[:, None]
                - 2 * Z @ self.centroids.T
                + (self.centroids**2).sum(1)[None, :]
            )
            ood = np.sqrt(np.clip(d2.min(1), 0, None))
        return self.gate.decide(np.log(np.clip(p, 1e-12, None)), ood, texts=list(texts))

    def logits(self, texts) -> np.ndarray:
        # The gate re-applies softmax(logits / T). With T pinned to 1.0 above,
        # feeding log(probs) round-trips exactly back to probs.
        return np.log(np.clip(self.probs(texts), 1e-12, None))

    def y_index(self, labels) -> np.ndarray:
        idx = {l: i for i, l in enumerate(self.labels)}
        return np.array([idx[l] for l in labels])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/final")
    ap.add_argument("--onnx", default="models/final/onnx/intent_int8.onnx")
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ref = IntentModel.load(ROOT / args.model)
    onnx_path = ROOT / args.onnx
    om = OnnxIntentModel(ref, onnx_path, max_len=args.max_len)

    res_onnx = run_all(om)
    res_py = run_all(ref)

    tag = onnx_path.stem.replace("intent_", "")
    out = ROOT / (args.out or f"reports/onnx_suite_{tag}.json")
    out.write_text(json.dumps(res_onnx, indent=2, default=float))

    print(f"python : {headline(res_py)}")
    print(f"{tag:7s}: {headline(res_onnx)}")

    print("\nDelta (onnx - python), negative means the export lost quality:")
    rows = [
        (
            "test accuracy",
            res_py["standard_test"]["accuracy"],
            res_onnx["standard_test"]["accuracy"],
        ),
        (
            "test macro F1",
            res_py["standard_test"]["macro_f1"],
            res_onnx["standard_test"]["macro_f1"],
        ),
        ("ECE (lower better)", res_py["standard_test"]["ece"], res_onnx["standard_test"]["ece"]),
        ("contextual", res_py["contextual"]["accuracy"], res_onnx["contextual"]["accuracy"]),
        (
            "minimal pairs",
            res_py["minimal_pairs"]["pair_accuracy"],
            res_onnx["minimal_pairs"]["pair_accuracy"],
        ),
        (
            "hard negatives",
            res_py["hard_negatives"]["accuracy"],
            res_onnx["hard_negatives"]["accuracy"],
        ),
        ("negation", res_py["negation"]["accuracy"], res_onnx["negation"]["accuracy"]),
        ("stt", res_py["stt"]["accuracy"], res_onnx["stt"]["accuracy"]),
        ("ood rejection", res_py["ood"]["rejection_rate"], res_onnx["ood"]["rejection_rate"]),
        ("gated coverage", res_py["gated_test"]["coverage"], res_onnx["gated_test"]["coverage"]),
        (
            "accepted precision",
            res_py["gated_test"]["accepted_precision"],
            res_onnx["gated_test"]["accepted_precision"],
        ),
    ]
    print(f"{'metric':22s} {'python':>8s} {tag:>8s} {'delta':>8s}")
    for name, a, b in rows:
        if a is None or b is None:
            print(f"{name:22s} {'n/a':>8s} {'n/a':>8s} {'-':>8s}")
            continue
        print(f"{name:22s} {a:8.4f} {b:8.4f} {b-a:+8.4f}")

    pa = res_py["gated_test"]["accepted_precision"]
    oa = res_onnx["gated_test"]["accepted_precision"]
    crit = (pa - oa) if (pa is not None and oa is not None) else 0.0
    if crit > 0.01:
        print(
            f"\nFAIL: accepted-precision dropped {crit:.4f} — this is the "
            "number the safety gate promises. Do not ship this build."
        )
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
