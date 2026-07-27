"""
MultilingualSemanticFallback — Stage 3 of the NLU pipeline for non-English languages.

Mirrors the API of scripts/nlu/semantic.py's SemanticFallback exactly:
  classify(text) -> (intent, confidence)

Differences from the English path:
  * Encoder: paraphrase-multilingual-MiniLM-L12-v2 (INT8 ONNX)
  * Tokeniser: MultilingualTokeniser (case-preserving, 105k vocab, SEQ_LEN=64)
  * Vocab: multilingual-vocab.txt (saved from the HuggingFace tokenizer)
  * Head:  semantic_head_multilingual.npz

ANE / Core ML constraints enforced here:
  * The ONNX session always receives (1, 64) tensors — static shapes, no
    dynamic axes. This is a pre-condition of the exported graph and is
    guaranteed by MultilingualTokeniser.encode().
  * No sentence-transformers dependency at runtime; only onnxruntime is used.
  * No network calls; all artifacts are local files.

Typical latency: ~10ms embed (INT8 ONNX) + <1ms head.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("nlu.multilingual.semantic")

BASE_DIR     = Path(__file__).parent.parent.parent
MODEL_DIR    = Path(__file__).parent / "models"

ONNX_PATH    = MODEL_DIR / "paraphrase-multilingual-minilm-l12-v2.int8.onnx"
VOCAB_PATH   = MODEL_DIR / "multilingual-vocab.txt"
HEAD_PATH    = MODEL_DIR / "semantic_head_multilingual.npz"

FALLBACK_INTENT   = "Default Fallback Intent"
DEFAULT_THRESHOLD = 0.55


class MultilingualSemanticFallback:
    """Embedding + logistic-regression head for fr/de/da intent classification.

    Public API matches scripts/nlu/semantic.py SemanticFallback so the engine
    can call it without branching on language after construction.
    """

    def __init__(
        self,
        head_path:  Path = HEAD_PATH,
        model_path: Path = ONNX_PATH,
        vocab_path: Path = VOCAB_PATH,
        threshold:  float = DEFAULT_THRESHOLD,
    ):
        self.threshold = threshold
        self._session  = None
        self._tokeniser = None
        self._head     = None   # (weights, bias, labels)

        # ── Load head weights ────────────────────────────────────────────
        if not head_path.exists():
            raise FileNotFoundError(
                f"Multilingual semantic head not found: {head_path}\n"
                "Run `python scripts/SemanticSupport/train_multilingual_semantic_head.py` first."
            )
        data = np.load(head_path, allow_pickle=True)
        self._head = (
            data["weights"].astype(np.float32),   # (n_classes, 384)
            data["bias"].astype(np.float32),       # (n_classes,)
            data["labels"],                        # object array of intent strings
        )

        # ── Load ONNX encoder ────────────────────────────────────────────
        if not model_path.exists():
            raise FileNotFoundError(
                f"Multilingual MiniLM ONNX model not found: {model_path}\n"
                "Run `python scripts/SemanticSupport/download_models.py` first."
            )
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"Multilingual vocab not found: {vocab_path}\n"
                "Run `python scripts/SemanticSupport/download_models.py` first."
            )

        import onnxruntime as ort
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(str(model_path), sess_opts)

        from .tokeniser import MultilingualTokeniser
        self._tokeniser = MultilingualTokeniser(vocab_path)

        # Warm-up: prime the ONNX JIT and memory allocator so the first
        # real user turn does not pay the initialisation cost.
        try:
            self._embed("warm up")
        except Exception:
            pass  # best-effort; never block construction

    # ── Public API (identical signature to SemanticFallback) ─────────────────

    def classify(self, text: str) -> Tuple[str, float]:
        """Return (intent, confidence) using the multilingual head.

        Confidence below self.threshold means the caller should fall back to
        GenAI. The fallback class is the explicit OOS class trained into the
        head, so rejection is learned rather than guessed.
        """
        vec = self._embed(text)
        weights, bias, labels = self._head
        logits = weights @ vec + bias
        logits -= logits.max()            # numerically stable softmax
        probs = np.exp(logits)
        probs /= probs.sum()
        top = int(np.argmax(probs))
        return str(labels[top]), float(probs[top])

    def is_available(self) -> bool:
        return True

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        """Return an L2-normalised 384-dim float32 embedding vector."""
        input_ids, attention_mask, token_type_ids = self._tokeniser.encode(text)

        # The ONNX graph expects static (1, 64) inputs — the tokeniser guarantees this.
        outputs = self._session.run(
            None,
            {
                "input_ids":      input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        # outputs[0] is last_hidden_state: (1, 64, 384)
        token_embeddings = outputs[0][0]                             # (64, 384)
        mask             = attention_mask[0].astype(np.float32)      # (64,)
        expanded         = mask[:, np.newaxis]                       # (64, 1)
        summed           = (token_embeddings * expanded).sum(axis=0) # (384,)
        vec              = summed / mask.sum()                       # mean pool

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)
