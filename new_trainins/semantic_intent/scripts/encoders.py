"""Encoder abstraction — Phase 11.

Every encoder exposes the same contract:
    fit(texts)                 # no-op for pretrained transformers
    encode(texts) -> (n, d) float32, L2-normalized
    meta() -> dict             # dim, params, disk size, latency

This keeps the classifier / calibration / gate code identical across
candidates, so the benchmark compares encoders and nothing else.

Transformer weights are loaded from a LOCAL directory. Nothing is downloaded
at inference time — the whole point is an offline model.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-9, None)


# ---------------------------------------------------------------------------
class TfidfSvdEncoder:
    """Offline, dependency-free reference encoder.

    Word 1-2 grams + character 3-5 grams (character grams are what give it any
    robustness to the STT misspellings) reduced to `dim` dense components.
    It is NOT expected to win; it exists so the whole downstream pipeline —
    calibration, gating, ONNX parity — is exercised and measurable without a
    network, and so the transformer has an honest floor to beat.
    """

    name = "tfidf-svd"
    needs_fit = True

    def __init__(self, dim: int = 384):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion

        self.dim = dim
        self.vec = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True
                    ),
                ),
            ]
        )
        self.svd = TruncatedSVD(n_components=dim, random_state=0)
        self._fitted = False

    def fit(self, texts):
        X = self.vec.fit_transform(texts)
        self.dim = min(self.dim, X.shape[1] - 1)
        self.svd.n_components = self.dim
        self.svd.fit(X)
        self._fitted = True
        return self

    def encode(self, texts) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit() first")
        return _l2(self.svd.transform(self.vec.transform(texts)).astype(np.float32))

    def meta(self) -> dict:
        n_feat = sum(len(v.vocabulary_) for _, v in self.vec.transformer_list)
        params = n_feat * self.dim
        return dict(
            name=self.name,
            dim=self.dim,
            vocab_features=n_feat,
            params=params,
            fp32_mb=round(params * 4 / 1e6, 2),
        )


# ---------------------------------------------------------------------------
class HFEncoder:
    """Local sentence-transformer style encoder with mean pooling."""

    needs_fit = False

    # prefix conventions that matter for retrieval-trained encoders
    PREFIX = {"e5": "query: ", "bge": "", "minilm": "", "gte": ""}

    def __init__(
        self, path: str | Path, name: str | None = None, max_len: int = 64, batch_size: int = 128
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.path = Path(path)
        self.name = name or self.path.name
        self.max_len = max_len
        self.batch_size = batch_size
        self.tok = AutoTokenizer.from_pretrained(str(self.path), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(self.path), local_files_only=True)
        self.model.eval()
        key = next((k for k in self.PREFIX if k in self.name.lower()), None)
        self.prefix = self.PREFIX.get(key, "")

    def fit(self, texts):
        return self

    def encode(self, texts) -> np.ndarray:
        import torch

        texts = [self.prefix + t for t in texts]
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                enc = self.tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors="pt",
                )
                h = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                out.append(emb.cpu().numpy())
        return _l2(np.vstack(out).astype(np.float32))

    def meta(self) -> dict:
        params = sum(p.numel() for p in self.model.parameters())
        size = sum(
            f.stat().st_size
            for f in self.path.rglob("*")
            if f.suffix in {".bin", ".safetensors", ".onnx"}
        )
        return dict(
            name=self.name,
            dim=self.model.config.hidden_size,
            params=params,
            fp32_mb=round(params * 4 / 1e6, 2),
            on_disk_mb=round(size / 1e6, 2),
            layers=self.model.config.num_hidden_layers,
        )


# ---------------------------------------------------------------------------
def discover_local_encoders() -> dict[str, Path]:
    """Any directory under models/encoders/ containing a config.json."""
    base = ROOT / "models" / "encoders"
    found = {}
    if base.exists():
        for d in sorted(base.iterdir()):
            if d.is_dir() and (d / "config.json").exists():
                found[d.name] = d
    return found


def get_encoder(spec: str):
    if spec == "tfidf-svd":
        return TfidfSvdEncoder()
    local = discover_local_encoders()
    if spec in local:
        return HFEncoder(local[spec], name=spec)
    p = Path(spec)
    if p.exists() and (p / "config.json").exists():
        return HFEncoder(p)
    raise ValueError(f"unknown encoder '{spec}'. local: {list(local)}")


def measure_latency(enc, texts, n_warm: int = 3, n_rep: int = 20) -> dict:
    """Single-sentence latency — that is what the phone actually does."""
    sample = list(texts[:n_rep])
    for t in sample[:n_warm]:
        enc.encode([t])
    times = []
    for t in sample:
        s = time.perf_counter()
        enc.encode([t])
        times.append((time.perf_counter() - s) * 1000)
    times.sort()
    return dict(
        p50_ms=round(times[len(times) // 2], 2),
        p90_ms=round(times[int(len(times) * 0.9)], 2),
        mean_ms=round(sum(times) / len(times), 2),
    )
