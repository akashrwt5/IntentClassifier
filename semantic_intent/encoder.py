"""
Frozen MiniLM sentence encoder over ONNX Runtime.

The encoder is never fine-tuned. All task knowledge lives in the small linear
head trained on top (see `head.py`), which keeps the artifact story simple:
one general-purpose encoder + a ~17 KB task head per deployment.

Pooling is mean-over-tokens with the attention mask, then L2 normalisation —
this must match `export.py`'s ONNX subgraph exactly, or the fused model and
the Python path will disagree.

Batching caveat (important)
---------------------------
The shipped encoder is INT8 with *dynamic* quantisation: DynamicQuantizeLinear
derives its scale from the whole input tensor at run time. Padding a short
sentence to a longer neighbour's length therefore changes its own embedding —
measured cosine 0.982 between the same sentence encoded alone vs. in a padded
batch of four. On device the model runs one sentence at a time, so training
embeddings must be produced the same way or the head is fitted in a slightly
different space than it is served in.

Hence `batch_size` defaults to 1. It costs nothing here: throughput is
~1,270 utterances/s either way, because these sequences are far too short to
saturate the batch dimension.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np

DEFAULT_MAX_LEN = 64  # commands are short; 64 covers >99.9% of the corpus


class MiniLMEncoder:
    def __init__(
        self,
        onnx_path: str | Path,
        vocab_path: str | Path,
        max_len: int = DEFAULT_MAX_LEN,
        threads: int = 2,
    ):
        import onnxruntime as ort
        from tokenizers import BertWordPieceTokenizer

        onnx_path, vocab_path = Path(onnx_path), Path(vocab_path)
        for p in (onnx_path, vocab_path):
            if not p.exists():
                raise FileNotFoundError(p)

        self.max_len = max_len
        self.tokenizer = BertWordPieceTokenizer(str(vocab_path), lowercase=True)
        self.tokenizer.enable_truncation(max_len)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(onnx_path), opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.dim = int(self.session.get_outputs()[0].shape[-1])

    # ------------------------------------------------------------------
    def tokenize(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        enc = [self.tokenizer.encode(t) for t in texts]
        width = max(len(e.ids) for e in enc)
        ids = np.zeros((len(texts), width), np.int64)
        mask = np.zeros((len(texts), width), np.int64)
        for j, e in enumerate(enc):
            ids[j, : len(e.ids)] = e.ids
            mask[j, : len(e.ids)] = 1
        return ids, mask

    def encode(self, texts: Sequence[str], batch_size: int = 1) -> np.ndarray:
        """Return L2-normalised mean-pooled embeddings, shape (n, dim).

        Keep batch_size=1 unless you have verified that a larger batch does not
        shift embeddings for your encoder — see the module docstring.
        """
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        out: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            ids, mask = self.tokenize(texts[i : i + batch_size])
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feeds["token_type_ids"] = np.zeros_like(ids)
            hidden = self.session.run(None, feeds)[0]
            out.append(mean_pool(hidden, mask))
        return np.vstack(out)


def mean_pool(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Masked mean pooling followed by L2 normalisation."""
    m = mask[..., None].astype(np.float32)
    summed = (hidden * m).sum(axis=1)
    counts = np.clip(m.sum(axis=1), 1e-9, None)
    vec = summed / counts
    vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9
    return vec.astype(np.float32)
