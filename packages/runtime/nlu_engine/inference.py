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


class OrtIntentBackend:
    """ONNX Runtime adapter for the TF-IDF intent model (string-tensor graph)."""

    def __init__(self, model_path: Path, n_labels: int):
        import onnxruntime as ort

        self._session = ort.InferenceSession(str(model_path))
        self._inp = self._session.get_inputs()[0]
        self._n_labels = n_labels

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
