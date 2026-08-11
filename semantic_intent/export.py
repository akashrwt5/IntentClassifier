"""
Fuse encoder + pooling + head + OOD scoring into ONE ONNX graph.

Runtime then owns exactly one model file and one tokenizer vocab. Pooling
happens inside the graph, so there is no chance of the app and the training
code disagreeing about how a sentence becomes a vector — which is the usual
source of silent on-device accuracy loss.

Graph appended to the encoder's `last_hidden_state`:

    mask -> Cast(f32) -> Unsqueeze(-1)                     [B,S,1]
    hidden * mask -> ReduceSum(axis=1)                     [B,H]
    mask -> ReduceSum(axis=1) -> Clip(min=eps)             [B,1]
    Div -> LpNormalization(p=2, axis=-1)                   [B,H]  = embedding
    embedding @ Wt + b -> Div(T) -> Softmax                [B,C]  = probabilities
    embedding @ Pt -> ReduceMax(axis=1)                    [B]    = ood_score

Outputs: `probabilities`, `ood_score`, `embedding`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .head import SemanticHead

EPS = 1e-9
_REQUIRED_OPSET = 18  # ReduceMax with axes-as-input


def _const(name: str, array: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.ascontiguousarray(array), name=name)


def fuse(
    encoder_path: str | Path, head: SemanticHead, output_path: str | Path, batch_one: bool = True
) -> Path:
    """Write a single ONNX file containing encoder + pooling + head."""
    encoder_path, output_path = Path(encoder_path), Path(output_path)
    model = onnx.load(str(encoder_path))

    opset = {o.domain: o.version for o in model.opset_import}.get("", 0)
    if opset < _REQUIRED_OPSET:
        model = onnx.version_converter.convert_version(model, _REQUIRED_OPSET)

    graph = model.graph
    hidden_name = graph.output[0].name
    mask_name = next(i.name for i in graph.input if i.name == "attention_mask")

    wt = head.weights.T.astype(np.float32)  # (H, C)
    pt = head.prototypes.T.astype(np.float32)  # (H, P)

    graph.initializer.extend(
        [
            _const("si_axis1", np.array([1], np.int64)),
            _const("si_axis_last", np.array([-1], np.int64)),
            _const("si_eps", np.array(EPS, np.float32)),
            _const("si_W", wt),
            _const("si_b", head.bias.astype(np.float32)),
            _const("si_T", np.array(head.temperature, np.float32)),
            _const("si_P", pt),
        ]
    )

    nodes = [
        helper.make_node(
            "Cast", [mask_name], ["si_mask_f"], to=TensorProto.FLOAT, name="si_cast_mask"
        ),
        helper.make_node(
            "Unsqueeze", ["si_mask_f", "si_axis_last"], ["si_mask3"], name="si_unsqueeze_mask"
        ),
        helper.make_node("Mul", [hidden_name, "si_mask3"], ["si_masked"], name="si_apply_mask"),
        helper.make_node(
            "ReduceSum", ["si_masked", "si_axis1"], ["si_sum"], keepdims=0, name="si_sum_tokens"
        ),
        helper.make_node(
            "ReduceSum", ["si_mask3", "si_axis1"], ["si_count"], keepdims=0, name="si_count_tokens"
        ),
        helper.make_node("Max", ["si_count", "si_eps"], ["si_count_safe"], name="si_clip_count"),
        helper.make_node("Div", ["si_sum", "si_count_safe"], ["si_mean"], name="si_mean_pool"),
        helper.make_node(
            "LpNormalization", ["si_mean"], ["embedding"], p=2, axis=-1, name="si_l2_normalise"
        ),
        # classification head
        helper.make_node("MatMul", ["embedding", "si_W"], ["si_wx"], name="si_head_matmul"),
        helper.make_node("Add", ["si_wx", "si_b"], ["si_logits"], name="si_head_bias"),
        helper.make_node("Div", ["si_logits", "si_T"], ["si_scaled"], name="si_temperature"),
        helper.make_node("Softmax", ["si_scaled"], ["probabilities"], axis=-1, name="si_softmax"),
        # out-of-scope score
        helper.make_node("MatMul", ["embedding", "si_P"], ["si_sims"], name="si_proto_sims"),
        helper.make_node(
            "ReduceMax", ["si_sims", "si_axis1"], ["ood_score"], keepdims=0, name="si_ood_max"
        ),
    ]
    graph.node.extend(nodes)

    n_classes, dim = head.weights.shape
    batch = 1 if batch_one else "batch"
    del graph.output[:]
    graph.output.extend(
        [
            helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [batch, n_classes]),
            helper.make_tensor_value_info("ood_score", TensorProto.FLOAT, [batch]),
            helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [batch, dim]),
        ]
    )

    if batch_one:
        for tensor in list(graph.input):
            d = tensor.type.tensor_type.shape.dim[0]
            d.ClearField("dim_param")
            d.dim_value = 1

    # Labels travel with the model so the runtime cannot drift out of sync.
    meta = model.metadata_props.add()
    meta.key, meta.value = "intent_labels", ",".join(map(str, head.labels))
    for key, value in (
        ("ood_threshold", head.ood_threshold),
        ("conf_threshold", head.conf_threshold),
        ("temperature", head.temperature),
    ):
        m = model.metadata_props.add()
        m.key, m.value = key, f"{value:.6f}"

    onnx.checker.check_model(model, full_check=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    return output_path


def verify_parity(
    onnx_path: str | Path, encoder, head: SemanticHead, texts, atol: float = 2e-4
) -> dict:
    """Compare the fused graph against the Python pipeline on the same inputs."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = {i.name for i in sess.get_inputs()}

    emb_ref = encoder.encode(list(texts))
    prob_ref = head.probabilities(emb_ref)
    ood_ref = head.ood_score(emb_ref)

    probs, oods, embs = [], [], []
    for text in texts:
        ids, mask = encoder.tokenize([text])
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        p, o, e = sess.run(["probabilities", "ood_score", "embedding"], feeds)
        probs.append(p[0])
        oods.append(o[0])
        embs.append(e[0])

    d_emb = float(np.abs(np.array(embs) - emb_ref).max())
    d_prob = float(np.abs(np.array(probs) - prob_ref).max())
    d_ood = float(np.abs(np.array(oods) - ood_ref).max())
    agree = float((np.array(probs).argmax(1) == prob_ref.argmax(1)).mean())
    return {
        "max_abs_diff_embedding": d_emb,
        "max_abs_diff_probs": d_prob,
        "max_abs_diff_ood": d_ood,
        "argmax_agreement": agree,
        "passed": bool(d_prob < atol and agree == 1.0),
    }
