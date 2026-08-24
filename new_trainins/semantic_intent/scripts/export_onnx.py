"""Phase 25 — ONNX export, and Phase 26 — INT8 quantization.

The exported graph is the WHOLE decision stack except tokenization:

    input_ids, attention_mask
        -> transformer encoder
        -> mean pooling over the attention mask
        -> L2 normalize                (the embedding the classifier saw)
        -> Gemm(W, b)                  (the trained linear classifier)
        -> Div(temperature)            (the fitted calibration)
        -> Softmax
        -> probs [batch, n_intents]

Calibration lives inside the graph on purpose: if temperature scaling is left
to hand-written app code it will eventually drift out of sync with the weights,
and then the confidence the safety gate reads is not the confidence that was
validated. Thresholds stay outside, in runtime_config.json, because product may
want to retune them without re-exporting.

Tokenization is deliberately NOT in the graph: the Android app owns it, and
vocab.txt plus tokenizer_config.json are written alongside so the Kotlin
WordPiece implementation is driven by the exact same vocabulary.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import IntentModel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
def linear_head_from(clf, n_dim: int):
    """Extract (W, b) for a linear decision function, or None if not linear."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    if isinstance(clf, LogisticRegression):
        W, b = clf.coef_, clf.intercept_
        return W.astype(np.float32), b.astype(np.float32), []
    if isinstance(clf, MLPClassifier):
        return None  # handled by mlp_layers_from
    if isinstance(clf, CalibratedClassifierCV):
        return None
    return None


def mlp_layers_from(clf):
    from sklearn.neural_network import MLPClassifier

    if not isinstance(clf, MLPClassifier):
        return None
    return [
        (w.T.astype(np.float32), b.astype(np.float32)) for w, b in zip(clf.coefs_, clf.intercepts_)
    ]


# ---------------------------------------------------------------------------
class FusedIntentNet:
    """Builds a torch module: encoder -> pool -> norm -> classifier -> T -> softmax."""

    def __init__(self, hf_encoder, clf, temperature: float, whitening=None):
        import torch
        import torch.nn as nn

        self.torch = torch
        base = hf_encoder.model

        layers = mlp_layers_from(clf)
        lin = linear_head_from(clf, base.config.hidden_size)
        if layers is None and lin is None:
            raise ValueError(
                f"{type(clf).__name__} cannot be fused into the graph. Use "
                "logreg or mlp for the exported model, or export the encoder "
                "alone and run the classifier natively."
            )

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = base
                self.T = float(temperature)
                # The OOD score is a distance in a whitened embedding space.
                # Folding the whitening matrix into the graph means the phone
                # never does matrix algebra: it gets a whitened vector out and
                # only has to compare it to 57 stored centroids.
                self.whiten = (
                    None
                    if whitening is None
                    else nn.Parameter(
                        torch.from_numpy(whitening.astype("float32")), requires_grad=False
                    )
                )
                if layers is not None:
                    mods = []
                    for i, (w, b) in enumerate(layers):
                        l = nn.Linear(w.shape[1], w.shape[0])
                        l.weight.data = torch.from_numpy(w)
                        l.bias.data = torch.from_numpy(b)
                        mods.append(l)
                        if i < len(layers) - 1:
                            mods.append(nn.ReLU())
                    self.head = nn.Sequential(*mods)
                else:
                    W, b, _ = lin
                    l = nn.Linear(W.shape[1], W.shape[0])
                    l.weight.data = torch.from_numpy(W)
                    l.bias.data = torch.from_numpy(b)
                    self.head = l

            def forward(self, input_ids, attention_mask):
                h = self.encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state
                m = attention_mask.unsqueeze(-1).float()
                emb = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                emb = emb / emb.norm(dim=1, keepdim=True).clamp(min=1e-9)
                logits = self.head(emb)
                probs = torch.softmax(logits / self.T, dim=-1)
                if self.whiten is None:
                    return probs
                return probs, emb @ self.whiten

        self.net = Net().eval()


def export_transformer(
    model: IntentModel, out_dir: Path, max_len: int = 64, opset: int = 17
) -> Path:
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    whitening = (
        model.ood.L_
        if getattr(model, "ood", None) is not None and model.ood.method == "mahalanobis"
        else None
    )
    fused = FusedIntentNet(model.encoder, model.clf, model.temperature, whitening)
    tok = model.encoder.tok
    dummy = tok(
        ["turn the volume up", "how do i change my program"],
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    path = out_dir / "intent_fp32.onnx"
    torch.onnx.export(
        fused.net,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(path),
        input_names=["input_ids", "attention_mask"],
        output_names=(["probs"] if whitening is None else ["probs", "whitened_embedding"]),
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "probs": {0: "batch"},
            "whitened_embedding": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    _consolidate(path)
    tok.save_pretrained(out_dir / "tokenizer")
    return path


def _consolidate(path: Path) -> None:
    """Fold external weight files back into one .onnx and re-run shape
    inference. Two reasons: a single file is far easier to ship in an Android
    asset bundle, and the dynamic quantizer's shape-inference pass trips over
    the stale value_info the exporter leaves behind."""
    import onnx

    m = onnx.load(str(path), load_external_data=True)
    del m.graph.value_info[:]
    try:
        m = onnx.shape_inference.infer_shapes(m, strict_mode=False)
    except Exception as e:  # noqa: BLE001
        print(f"  (shape inference skipped: {e})")
    onnx.save(m, str(path), save_as_external_data=False)
    ext = path.with_name(path.name + ".data")
    if ext.exists():
        ext.unlink()


def export_sklearn(model: IntentModel, out_dir: Path) -> Path:
    """Fallback path for the non-transformer reference encoder.

    skl2onnx cannot convert a character-n-gram TfidfVectorizer (`char_wb`), so
    the text->features stage of the reference encoder has no ONNX equivalent.
    We export the dense tail (SVD projection -> classifier) instead, which is
    enough to validate the parity tooling. The reference encoder was never a
    deployment candidate; the shipped model is the transformer, whose whole
    graph including pooling does export.
    """
    from skl2onnx import to_onnx
    from sklearn.pipeline import make_pipeline

    out_dir.mkdir(parents=True, exist_ok=True)
    enc = model.encoder
    path = out_dir / "intent_fp32.onnx"
    sample = np.array(["turn the volume up", "how do i change my program"])
    try:
        pipe = make_pipeline(enc.vec, enc.svd, model.clf)
        onx = to_onnx(pipe, sample, options={id(model.clf): {"zipmap": False}})
    except NotImplementedError as e:
        print(
            f"  full-pipeline export unavailable ({e.__class__.__name__}); "
            "exporting the dense tail only"
        )
        X = enc.vec.transform(sample.tolist())
        dense = enc.svd.transform(X).astype(np.float32)
        onx = to_onnx(make_pipeline(model.clf), dense, options={id(model.clf): {"zipmap": False}})
        (out_dir / "EXPORT_NOTE.txt").write_text(
            "This ONNX graph starts at the 384-d embedding, not at raw text: "
            "the reference encoder uses char_wb TF-IDF, which skl2onnx cannot "
            "convert. Reference encoder only — not a deployment candidate.\n"
        )
    path.write_bytes(onx.SerializeToString())
    return path


def quantize(path: Path, per_channel: bool = True, keep_embeddings_fp32: bool = True) -> Path:
    """Dynamic INT8.

    Two settings matter far more than the defaults suggest:

    `per_channel=True` — one scale per output channel instead of one for the
    whole weight tensor. Transformer projection matrices have wildly different
    per-channel ranges, so a single tensor-wide scale crushes the small
    channels. This is usually the difference between a usable and an unusable
    INT8 encoder.

    `keep_embeddings_fp32` — the token embedding is a Gather, not a MatMul, and
    quantizing it injects noise at layer zero that every later layer amplifies.
    It also buys little on a small vocabulary. Excluded by default.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out = path.with_name(path.name.replace("fp32", "int8"))
    op_types = ["MatMul", "Gemm"] if keep_embeddings_fp32 else None
    quantize_dynamic(
        str(path),
        str(out),
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        reduce_range=False,
        op_types_to_quantize=op_types,
    )
    return out


def write_runtime_config(model: IntentModel, out_dir: Path, max_len: int):
    cfg = dict(
        labels=model.labels,
        temperature=model.temperature,
        note="temperature is already applied inside the ONNX graph; it is "
        "recorded here for traceability only",
        gate=model.gate.to_dict(),
        ood=(
            model.ood.export()
            if getattr(model, "ood", None) is not None and model.ood.method == "mahalanobis"
            else None
        ),
        ood_threshold=(
            None if model.gate.ood_threshold == float("inf") else model.gate.ood_threshold
        ),
        tokenizer=dict(
            type="wordpiece",
            max_len=max_len,
            lowercase=True,
            prefix=getattr(model.encoder, "prefix", ""),
        ),
        outputs=dict(
            probs=dict(shape=["batch", len(model.labels)], meaning="calibrated probabilities"),
            whitened_embedding=dict(
                shape=["batch", "dim"],
                meaning="OOD score = min over classes of the euclidean distance "
                "to ood.whitened_centroids; reject if it exceeds "
                "ood_threshold",
            ),
        ),
    )
    (out_dir / "runtime_config.json").write_text(json.dumps(cfg, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/final_student_256")
    ap.add_argument("--out", default="models/final_student_256/onnx")
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--no-quantize", action="store_true")
    ap.add_argument(
        "--no-per-channel",
        action="store_true",
        help="tensor-wide INT8 scales (usually much worse on transformers)",
    )
    ap.add_argument(
        "--quantize-embeddings",
        action="store_true",
        help="also quantize the token embedding Gather",
    )
    args = ap.parse_args()

    model = IntentModel.load(ROOT / args.model)
    out_dir = ROOT / args.out
    is_transformer = hasattr(model.encoder, "tok")

    if is_transformer:
        p = export_transformer(model, out_dir, args.max_len)
    else:
        print(
            "NOTE: reference (non-transformer) encoder — exporting the "
            "sklearn pipeline. This path exists so parity tooling is "
            "exercised; the shipped model is expected to be the transformer."
        )
        p = export_sklearn(model, out_dir)
    size = p.stat().st_size / 1e6
    print(f"fp32 ONNX: {p.name}  {size:.3f} MB")

    write_runtime_config(model, out_dir, args.max_len)

    if not args.no_quantize:
        q = quantize(
            p,
            per_channel=not args.no_per_channel,
            keep_embeddings_fp32=not args.quantize_embeddings,
        )
        print(
            f"int8 ONNX: {q.name}  {q.stat().st_size/1e6:.3f} MB "
            f"({size / (q.stat().st_size/1e6):.2f}x smaller)"
        )


if __name__ == "__main__":
    main()
