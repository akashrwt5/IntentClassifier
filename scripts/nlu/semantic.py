"""
SemanticFallback — Stage 3 of the NLU pipeline.

Activates only when TF-IDF (Stage 2) confidence falls below threshold.
Embeds the utterance with MiniLM-L6-v2 and classifies it with a trained
logistic-regression head (SetFit-style, frozen encoder).

The head is trained by scripts/train_semantic_head.py on embeddings of
ALL training phrases — including Default Fallback Intent phrases as an
explicit out-of-scope class. Rejection is therefore learned, not
threshold-guessed: when the head predicts the fallback class, the engine
routes to GenAI.

Artifacts:
  models/semantic_head.npz   — head weights (~85 KB)
  models/minilm-l6-v2.onnx   — INT8 MiniLM encoder (~23 MB)
  models/minilm-vocab.txt    — WordPiece vocab for the runtime tokeniser

Legacy: if the head is missing but models/semantic_index.npz exists,
falls back to 1-NN cosine search over the index (deprecated — absolute
cosine thresholds proved uncalibratable; see docs/semantic-understanding-plan.md).

Typical latency: ~8ms embed + <1ms head.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple

BASE_DIR   = Path(__file__).parent.parent.parent
MODEL_DIR  = BASE_DIR / "models"
HEAD_PATH  = MODEL_DIR / "semantic_head.npz"
INDEX_PATH = MODEL_DIR / "semantic_index.npz"
ONNX_PATH  = MODEL_DIR / "minilm-l6-v2.onnx"

FALLBACK_INTENT = "Default Fallback Intent"

# Minimum softmax probability for the head's prediction to rescue a turn.
# Measured on held-out data (train_semantic_head.py rejection curve):
# 0.50 keeps 95.5% of in-scope rescues, rejects 73% of out-of-scope.
DEFAULT_THRESHOLD = 0.50

# Near-exact match: an utterance this close to a training phrase is
# almost certainly that intent even when the head is unsure.
NEAR_MATCH_THRESHOLD = 0.80


class SemanticFallback:
    def __init__(
        self,
        head_path: Path = HEAD_PATH,
        index_path: Path = INDEX_PATH,
        model_path: Path = ONNX_PATH,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.threshold    = threshold
        self._st_model    = None
        self._ort_session = None
        self._head        = None   # (weights, bias, labels)
        self._embeddings  = None   # legacy 1-NN index
        self._intents     = None

        embedder = None
        if head_path.exists():
            data   = np.load(head_path, allow_pickle=True)
            self._head = (
                data["weights"].astype(np.float32),
                data["bias"].astype(np.float32),
                data["labels"],
            )
            embedder = str(data["embedder"][0]) if "embedder" in data else "onnx"
        if index_path.exists():
            data = np.load(index_path, allow_pickle=True)
            index_embedder = (str(data["embedder"][0])
                              if "embedder" in data else "st")
            # Only load the index if it was built with the same embedder as
            # the head — mixed embedding spaces produced the original bug.
            if embedder is None or index_embedder == embedder:
                self._embeddings = data["embeddings"].astype(np.float32)
                self._intents    = data["intents"]
                norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                self._embeddings = self._embeddings / norms
                embedder = embedder or index_embedder
        if self._head is None and self._embeddings is None:
            raise FileNotFoundError(
                f"Neither {head_path} nor {index_path} found. "
                "Run `python scripts/train_semantic_head.py` first."
            )

        # The runtime embedder MUST match the one that built the artifact,
        # otherwise query vectors land in a slightly different space.
        if embedder == "st":
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                embedder = "onnx"  # best effort
        if embedder == "onnx":
            if not model_path.exists():
                raise FileNotFoundError(
                    f"MiniLM ONNX model not found: {model_path}. "
                    "Run `python scripts/download_minilm.py` first."
                )
            import onnxruntime as ort  # type: ignore
            self._ort_session = ort.InferenceSession(str(model_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, text: str) -> Tuple[str, float]:
        """
        Return (intent, confidence). Hybrid decision:

          1. Classification head — calibrated softmax probability,
             generalises across each intent's whole phrase cluster.
          2. Near-exact match — if the head is unsure but the utterance
             sits within NEAR_MATCH_THRESHOLD cosine of a single training
             phrase, trust that phrase's intent (1-NN catches near-
             duplicates the head's averaged boundary can dilute).

        Confidence below self.threshold means the caller should fall back.
        """
        vec = self._embed(text)

        if self._head is not None:
            weights, bias, labels = self._head
            logits = weights @ vec + bias
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            top = int(np.argmax(probs))
            intent, conf = str(labels[top]), float(probs[top])
            if conf >= self.threshold:
                return intent, conf
            # Head unsure — check for a near-exact training phrase match
            if self._embeddings is not None:
                nn_intent, nn_score = self._nearest(vec)
                if nn_score >= NEAR_MATCH_THRESHOLD:
                    return nn_intent, nn_score
            return intent, conf

        # Legacy: index only, raw cosine
        return self._nearest(vec)

    def _nearest(self, vec: np.ndarray) -> Tuple[str, float]:
        scores  = self._embeddings @ vec
        top_idx = int(np.argmax(scores))
        return str(self._intents[top_idx]), float(scores[top_idx])

    def is_available(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Embedding — must match the path used to build the artifact
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        """Return an L2-normalised 384-dim float32 embedding vector."""
        if self._st_model is not None:
            vec = self._st_model.encode(
                text, normalize_embeddings=True, convert_to_numpy=True
            )
            return vec.astype(np.float32)
        return self._embed_onnx(text)

    def _embed_onnx(self, text: str) -> np.ndarray:
        input_ids, attention_mask, token_type_ids = self._tokenise(text)

        outputs = self._ort_session.run(
            None,
            {
                "input_ids":      input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        token_embeddings = outputs[0][0]
        mask             = attention_mask[0]
        expanded         = mask[:, np.newaxis].astype(np.float32)
        summed           = (token_embeddings * expanded).sum(axis=0)
        vec              = summed / expanded.sum()

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    # ------------------------------------------------------------------
    # WordPiece tokeniser — used by the ONNX path
    # ------------------------------------------------------------------

    _VOCAB_PATH = MODEL_DIR / "minilm-vocab.txt"

    def _tokenise(self, text: str, max_len: int = 64):
        vocab  = self._load_vocab()
        tokens = ["[CLS]"] + self._wordpiece(text.lower(), vocab) + ["[SEP]"]
        tokens = tokens[:max_len]
        ids    = [vocab.get(t, vocab["[UNK]"]) for t in tokens]
        n      = len(ids)
        return (
            np.array([ids], dtype=np.int64),
            np.ones((1, n), dtype=np.int64),
            np.zeros((1, n), dtype=np.int64),
        )

    @staticmethod
    def _wordpiece(text: str, vocab: dict, unk: str = "[UNK]") -> list:
        import re
        tokens = []
        for word in re.findall(r"\w+|[^\w\s]", text):
            chars = list(word)
            if len(chars) > 100:
                tokens.append(unk); continue
            is_bad     = False
            start      = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in vocab:
                        cur = substr; break
                    end -= 1
                if cur is None:
                    is_bad = True; break
                sub_tokens.append(cur)
                start = end
            tokens.extend([unk] if is_bad else sub_tokens)
        return tokens

    _vocab_cache: Optional[dict] = None

    def _load_vocab(self) -> dict:
        if SemanticFallback._vocab_cache is None:
            with open(self._VOCAB_PATH, encoding="utf-8") as f:
                SemanticFallback._vocab_cache = {
                    line.strip(): i for i, line in enumerate(f)
                }
        return SemanticFallback._vocab_cache
