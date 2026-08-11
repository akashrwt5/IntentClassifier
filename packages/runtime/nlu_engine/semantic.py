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

import json
import re
import unicodedata

import numpy as np
from pathlib import Path
from typing import Optional, Tuple

BASE_DIR   = Path(__file__).resolve().parents[3]
MODEL_DIR  = BASE_DIR / "models"
HEAD_PATH  = MODEL_DIR / "semantic_head.npz"
ONNX_PATH  = MODEL_DIR / "minilm-l6-v2.onnx"

# Where a distilled single-file student lives, if one is installed. See
# StudentSemantic below. Absent -> the MiniLM path above is used unchanged.
STUDENT_DIR = MODEL_DIR / "semantic_student"

FALLBACK_INTENT = "Default Fallback Intent"

# Minimum softmax probability for the head's prediction to rescue a turn.
# Measured on held-out data (train_semantic_head.py rejection curve):
# 0.55 keeps the bulk of in-scope rescues while rejecting more out-of-scope
# queries (e.g. "how is the weather today" scores ~0.51 to MemoryChange and
# must be rejected). Tuned against semantic_holdout_100.csv.
DEFAULT_THRESHOLD = 0.55


class StudentSemantic:
    """Stage 3 backed by a distilled single-file student (new_semantic/).

    Shape difference from SemanticFallback, which is why this is a separate
    class rather than a flag:

                        MiniLM path                 student path
        tokenizer       WordPiece, 30k vocab        word-level, own vocab
        artifacts       encoder ONNX + head .npz    ONE ONNX
        flow            text -> 384-d embedding     text -> ids -> logits
                             -> LogReg head

    The student has no embedding stage to share, so `_embed`/`_tokenise`/
    mean-pooling from SemanticFallback do not apply to it at all.

    Installed layout (models/semantic_student/<lang>/):
        student.onnx   ids (1, max_len) + mask -> logits (1, n_intents)
        vocab.json     {"mode": "word", "vocab": {...}}  (or a bare dict)
        labels.json    ["Cmd.ActivityAerobics", ...]  column order of the logits
        meta.json      {"max_len": 24, "threshold": 0.40, ...}   optional

    THE TOKENIZER BELOW MUST STAY BYTE-IDENTICAL to the one used at training
    time (new_semantic/scripts/common.py). It is the model's input contract:
    change the regex and every id shifts, silently.
    """

    _TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
    PAD_ID = 0
    UNK_ID = 1

    def __init__(self, model_dir: Path, threshold: float = 0.40):
        import onnxruntime as ort

        model_dir = Path(model_dir)
        onnx_path = model_dir / "student.onnx"
        for required in (onnx_path, model_dir / "vocab.json",
                         model_dir / "labels.json"):
            if not required.exists():
                raise FileNotFoundError(f"student artifact missing: {required}")

        # A sidecar means torch wrote the weights outside the graph; loading the
        # .onnx alone would give a model with no weights and silently wrong
        # answers. Refuse rather than serve that.
        sidecar = onnx_path.with_name(onnx_path.name + ".data")
        if sidecar.exists():
            raise FileNotFoundError(
                f"{onnx_path.name} has an external-data sidecar ({sidecar.name}); "
                f"re-export it self-contained before installing."
            )

        raw = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "vocab" in raw and "mode" in raw:
            self.vocab = raw["vocab"]
            raw_mode = raw.get("mode", "word")
        else:
            self.vocab = raw
            raw_mode = "word"

        self.labels = json.loads(
            (model_dir / "labels.json").read_text(encoding="utf-8"))

        meta_path = model_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        self.tokenizer_mode = meta.get("tokenizer", raw_mode)
        self.max_len = int(meta.get("max_len", 32 if self.tokenizer_mode == "subword" else 24))
        self.threshold = float(meta.get("threshold", threshold))

        # Temperature scaling, fitted offline by new_semantic/scripts/calibrate.py
        # on the dev half and written here by install_student.py.
        #
        # This is NOT a tuning knob — it is what makes the returned number mean
        # what it says. Uncalibrated, the student reported ECE 0.2029: "0.9
        # confident" was right about 70% of the time, and the engine gates on
        # exactly that number. At T=0.68, ECE is 0.0187 and accuracy at the 0.40
        # gate goes 0.8374 -> 0.8835, purely from better-shaped confidence.
        #
        # T is rank-preserving, so argmax NEVER changes; only the confidence
        # does. That is why the omission survived unnoticed until now — every
        # accuracy check still passed while the gate was reading a miscalibrated
        # scale, and meta.json had recorded temperature 0.68 for a while.
        #
        # Default 1.0 = identity, so a meta.json without the key behaves exactly
        # as before.
        self.temperature = float(meta.get("temperature", 1.0))
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature} from {meta_path}")

        self._sess = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"])
        self._in_ids = self._sess.get_inputs()[0].name
        self._in_mask = self._sess.get_inputs()[1].name

        self.classify("warm up")  # move ORT's graph setup off the first real turn

    # ------------------------------------------------------------------
    def _wordpiece(self, word: str) -> list[str]:
        out, start = [], 0
        while start < len(word):
            end = len(word)
            cur = None
            while start < end:
                piece = word[start:end] if start == 0 else "##" + word[start:end]
                if piece in self.vocab:
                    cur = piece
                    break
                end -= 1
            if cur is None:
                out.append("[UNK]")
                start += 1
            else:
                out.append(cur)
                start = end
        return out

    def _encode(self, text: str):
        t = unicodedata.normalize("NFKD", str(text)).replace("’", "'")
        toks = self._TOKEN_RE.findall(t.lower())
        if self.tokenizer_mode == "subword":
            pieces = [p for w in toks for p in self._wordpiece(w)]
            ids = [self.vocab.get(p, self.UNK_ID) for p in pieces][: self.max_len]
        else:
            ids = [self.vocab.get(w, self.UNK_ID) for w in toks][: self.max_len]
        ids += [self.PAD_ID] * (self.max_len - len(ids))
        arr = np.array([ids], dtype=np.int64)
        return arr, arr != self.PAD_ID

    def classify(self, text: str) -> Tuple[str, float]:
        ids, mask = self._encode(text)
        logits = self._sess.run(None, {self._in_ids: ids, self._in_mask: mask})[0][0]
        z = logits / self.temperature
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        top = int(np.argmax(p))
        return str(self.labels[top]), float(p[top])

    def is_available(self) -> bool:
        return True


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
            from .inference import OrtEmbedderBackend
            self._backend = OrtEmbedderBackend(model_path)
            self._ort_session = self._backend  # legacy truthiness checks

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

        token_embeddings = self._backend.embed_tokens(
            input_ids, attention_mask, token_type_ids)
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
