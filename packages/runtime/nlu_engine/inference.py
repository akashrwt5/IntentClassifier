"""InferenceBackend — the inference-inversion seam (runtime-contract-v1 §2).

The conversation logic never owns an ML runtime: it calls a backend the host
injects. In Python the host backend is ONNX Runtime; iOS injects CoreML and
Android injects ORT Mobile behind the same two methods. This module closes
the first row of the contract's conformance-gap table.

Adapter note (documented deviation from the contract's signatures): the
contract types the seam as ``tfidf_logits(features)`` / ``embed(token_ids)``.
The Python intent model compiles featurization INTO the ONNX graph (string
tensor in), so this adapter's tfidf entry point takes normalized text — the
featurizer simply lives on the host side of the seam here, which the
contract permits. The embedder entry point takes token tensors exactly as
specified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class InferenceBackend(Protocol):
    """What the conversation logic is allowed to ask of an ML runtime."""

    def tfidf_logits(self, text: str) -> np.ndarray:
        """RAW decision-function logits over the fixed label order (never
        softmaxed — the core applies softmax(logits / T) itself)."""
        ...

    def embed_tokens(self, input_ids: np.ndarray, attention_mask: np.ndarray,
                     token_type_ids: np.ndarray) -> np.ndarray:
        """Encoder forward pass: token embeddings [seq_len, dim]. Batch 1,
        static shapes (on-device constraint)."""
        ...

    def unigram_vocabulary(self) -> frozenset[str]:
        """Single-word terms the featurizer can represent, or an empty set.

        The out-of-vocabulary guard needs to know what the model CANNOT see.
        A backend that cannot answer returns an empty set and the guard
        disables itself rather than guessing.
        """
        ...


class OrtIntentBackend:
    """ONNX Runtime adapter for the TF-IDF intent model (string-tensor graph)."""

    def __init__(self, model_path: Path, n_labels: int):
        import onnxruntime as ort

        self._model_path = Path(model_path)
        self._session = ort.InferenceSession(str(model_path))
        self._inp = self._session.get_inputs()[0]
        self._n_labels = n_labels
        self._unigrams: frozenset[str] | None = None

    def unigram_vocabulary(self) -> frozenset[str]:
        """Single-word terms read out of the graph's own TfIdfVectorizer node.

        Deliberately read from the SHIPPED GRAPH rather than from a vocabulary
        file beside it. The guard's whole job is to answer "can the featurizer
        represent this word?", and the only authority on that is the featurizer.
        A separate vocab artifact would be one more pair that can silently
        disagree — the failure mode this codebase has already paid for twice
        (the device/server temperature, blocker B8; the device/server training
        subset).

        `pool_strings` holds every n-gram order concatenated; `ngram_counts`
        gives each order's offset, so the unigrams are the first slice. Parsed
        lazily and cached — this is not on the per-turn path.

        Returns an empty set if the graph has no such node (a different
        featurizer, or an injected test backend), which switches the guard off.
        """
        if self._unigrams is not None:
            return self._unigrams
        self._unigrams = frozenset()
        try:
            import onnx

            graph = onnx.load(str(self._model_path)).graph
            node = next(n for n in graph.node if n.op_type == "TfIdfVectorizer")
            attrs = {a.name: a for a in node.attribute}
            counts = list(attrs["ngram_counts"].ints)
            pool = attrs["pool_strings"].strings
            end = counts[1] if len(counts) > 1 else len(pool)
            self._unigrams = frozenset(s.decode("utf-8") for s in pool[counts[0]:end])
        except Exception:
            pass  # guard disables itself; never fail a turn over telemetry-grade data
        return self._unigrams

    def tfidf_logits(self, text: str) -> np.ndarray:
        t = text.lower().strip()
        x = (np.array([[t]], dtype=object) if len(self._inp.shape) == 2
             else np.array([t], dtype=object))
        outputs = self._session.run(None, {self._inp.name: x})
        scores = None
        for o in outputs:
            if hasattr(o, "shape") and o.shape and o.shape[-1] == self._n_labels:
                scores = o[0]
                break
        if scores is None:
            scores = outputs[-1][0]
        if isinstance(scores, dict):
            # zipmap-style output: order by the session's label order is the
            # caller's job — return values in dict-key order handled upstream.
            return scores
        return np.asarray(scores, dtype=float)

    def embed_tokens(self, *args, **kwargs):  # pragma: no cover - wrong backend
        raise NotImplementedError("intent backend does not embed")


class OrtEmbedderBackend:
    """ONNX Runtime adapter for the MiniLM encoder (token tensors in)."""

    def __init__(self, model_path: Path):
        import onnxruntime as ort

        self._session = ort.InferenceSession(str(model_path))

    def tfidf_logits(self, text: str):  # pragma: no cover - wrong backend
        raise NotImplementedError("embedder backend does not classify")

    def embed_tokens(self, input_ids, attention_mask, token_type_ids) -> np.ndarray:
        outputs = self._session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        })
        return outputs[0][0]
