"""
SemanticFallback — Stage 3 of the NLU pipeline.

Activates only when TF-IDF (Stage 2) confidence falls below threshold.
Embeds the utterance with MiniLM-L6-v2 and finds the nearest training
phrase in a pre-built index, returning the intent of the closest match.

The index is built once by scripts/build_semantic_index.py and stored as
models/semantic_index.npz.  At runtime the index is loaded into RAM
(~5.7 MB float16) and searched with a vectorised cosine similarity.

Runtime embedding strategy (in priority order):
  1. sentence-transformers — highest quality, used when installed
  2. ONNX + custom WordPiece — mobile/edge fallback, no heavy deps

Typical latency: ~8ms inference + ~1ms search over 6,469 phrases.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple

BASE_DIR   = Path(__file__).parent.parent.parent
MODEL_DIR  = BASE_DIR / "models"
INDEX_PATH = MODEL_DIR / "semantic_index.npz"
ONNX_PATH  = MODEL_DIR / "minilm-l6-v2.onnx"

# Similarity must exceed this to rescue a fallback.
# Below this we consider the query genuinely out-of-scope → GenAI.
DEFAULT_THRESHOLD = 0.78


class SemanticFallback:
    def __init__(
        self,
        index_path: Path = INDEX_PATH,
        model_path: Path = ONNX_PATH,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        if not index_path.exists():
            raise FileNotFoundError(
                f"Semantic index not found: {index_path}. "
                "Run `python scripts/build_semantic_index.py` first."
            )

        self.threshold  = threshold
        self._st_model  = None   # sentence-transformers model (preferred)
        self._ort_session = None  # ONNX session (fallback)

        # Try sentence-transformers first — same library used to build the index
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            # Fall back to ONNX runtime
            if not model_path.exists():
                raise FileNotFoundError(
                    f"sentence-transformers not installed and ONNX model not found: {model_path}. "
                    "Either `pip install sentence-transformers` or run `python scripts/download_minilm.py`."
                )
            import onnxruntime as ort  # type: ignore
            self._ort_session = ort.InferenceSession(str(model_path))

        data = np.load(index_path, allow_pickle=True)
        # embeddings: float16 array (n_phrases, 384)
        # intents: object array of intent name strings (n_phrases,)
        self._embeddings = data["embeddings"].astype(np.float32)
        self._intents    = data["intents"]

        # Pre-normalise index vectors so runtime cosine = simple dot product
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._embeddings = self._embeddings / norms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, text: str) -> Tuple[str, float]:
        """
        Return (intent, similarity).
        If similarity < threshold the caller should treat it as a fallback.
        """
        vec    = self._embed(text)
        scores  = self._embeddings @ vec
        top_idx = int(np.argmax(scores))
        return str(self._intents[top_idx]), float(scores[top_idx])

    def is_available(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Embedding — sentence-transformers preferred, ONNX fallback
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        """Return an L2-normalised 384-dim float32 embedding vector."""
        if self._st_model is not None:
            return self._embed_st(text)
        return self._embed_onnx(text)

    def _embed_st(self, text: str) -> np.ndarray:
        """Embed using sentence-transformers (matches index build exactly)."""
        vec = self._st_model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.astype(np.float32)

    def _embed_onnx(self, text: str) -> np.ndarray:
        """Embed using ONNX runtime + custom WordPiece tokeniser (mobile path)."""
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
    # WordPiece tokeniser — used only by the ONNX path
    # ------------------------------------------------------------------

    _VOCAB_PATH = MODEL_DIR / "minilm-vocab.txt"

    def _tokenise(self, text: str, max_len: int = 128):
        vocab  = self._load_vocab()
        tokens = ["[CLS]"] + self._wordpiece(text.lower(), vocab) + ["[SEP]"]
        tokens = tokens[:max_len]
        ids    = [vocab.get(t, vocab["[UNK]"]) for t in tokens]
        pad    = max_len - len(ids)
        mask   = [1] * len(ids) + [0] * pad
        ids    = ids + [0] * pad
        return (
            np.array([ids],  dtype=np.int64),
            np.array([mask], dtype=np.int64),
            np.zeros((1, max_len), dtype=np.int64),
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
