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

Rejection is learned via an explicit out-of-scope class in the head, so
there is no cosine-threshold tiebreak. (The legacy 1-NN cosine index was
removed in Sprint 3 — absolute cosine thresholds proved uncalibratable and
the index was an untracked, unverified influence on predictions.)

Typical latency: ~8ms embed + <1ms head.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple

BASE_DIR   = Path(__file__).resolve().parents[3]
MODEL_DIR  = BASE_DIR / "models"
HEAD_PATH  = MODEL_DIR / "semantic_head.npz"
ONNX_PATH  = MODEL_DIR / "minilm-l6-v2.onnx"

FALLBACK_INTENT = "Default Fallback Intent"

# Minimum softmax probability for the head's prediction to rescue a turn.
# Measured on held-out data (train_semantic_head.py rejection curve):
# 0.55 keeps the bulk of in-scope rescues while rejecting more out-of-scope
# queries (e.g. "how is the weather today" scores ~0.51 to MemoryChange and
# must be rejected). Tuned against semantic_holdout_100.csv.
DEFAULT_THRESHOLD = 0.55


class SemanticFallback:
    def __init__(
        self,
        head_path: Path = HEAD_PATH,
        model_path: Path = ONNX_PATH,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.threshold    = threshold
        self._st_model    = None
        self._ort_session = None
        self._head        = None   # (weights, bias, labels)

        embedder = None
        if head_path.exists():
            data   = np.load(head_path, allow_pickle=True)
            self._head = (
                data["weights"].astype(np.float32),
                data["bias"].astype(np.float32),
                data["labels"],
            )
            embedder = str(data["embedder"][0]) if "embedder" in data else "onnx"
        if self._head is None:
            raise FileNotFoundError(
                f"{head_path} not found. "
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

        # Warm-up: ONNX Runtime does graph/JIT setup and allocator priming on
        # the first inference. Doing one throwaway embed here moves that cost
        # off the first real user turn (otherwise the first semantic rescue is
        # 3-5x slower than steady state).
        if self._ort_session is not None:
            try:
                self._embed("warm up")
            except Exception:
                pass  # warm-up is best-effort; never block construction

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, text: str) -> Tuple[str, float]:
        """
        Return (intent, confidence) from the classification head: a calibrated
        softmax probability that generalises across each intent's phrase
        cluster, with an explicit out-of-scope class for learned rejection.

        Confidence below self.threshold means the caller should fall back.
        """
        vec = self._embed(text)
        weights, bias, labels = self._head
        logits = weights @ vec + bias
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        top = int(np.argmax(probs))
        return str(labels[top]), float(probs[top])

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
