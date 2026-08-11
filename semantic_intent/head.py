"""
The task head: a linear probe over frozen embeddings, plus two things that a
bare `LogisticRegression` does not give you and that an on-device NLU needs.

1. **Temperature scaling** — one scalar fitted on out-of-fold logits so the
   softmax score means what it says. Without it the confidence gate is tuned
   against a number that drifts every retrain.

2. **Prototype-based out-of-scope detection** — k-means centroids of the
   training embeddings per intent. Max cosine similarity to any prototype is a
   far better in-domain/out-of-domain signal than max softmax probability
   (measured AUROC 0.998 vs 0.968), because a linear head is forced to assign
   *some* class to every input, however alien.

Why linear and not an MLP: on the 11-intent corpus an MLP(512,256) buys
+0.1pt in-distribution but loses 2 points on held-out paraphrases and becomes
markedly overconfident on out-of-scope input — while costing 78x the
parameters. The linear head is both smaller and better behaved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class SemanticHead:
    weights: np.ndarray  # (n_classes, dim)
    bias: np.ndarray  # (n_classes,)
    labels: np.ndarray  # (n_classes,) str
    temperature: float  # softmax divisor
    prototypes: np.ndarray  # (n_protos, dim), L2-normalised
    ood_threshold: float  # min max-cosine to be considered in-scope
    conf_threshold: float  # min calibrated probability to accept

    # ------------------------------------------------------------------
    def logits(self, emb: np.ndarray) -> np.ndarray:
        return emb @ self.weights.T + self.bias

    def probabilities(self, emb: np.ndarray) -> np.ndarray:
        z = self.logits(emb) / self.temperature
        z -= z.max(axis=-1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=-1, keepdims=True)

    def ood_score(self, emb: np.ndarray) -> np.ndarray:
        """Max cosine similarity to any training prototype. Higher = in-domain."""
        return (emb @ self.prototypes.T).max(axis=-1)

    def predict(self, emb: np.ndarray):
        probs = self.probabilities(emb)
        idx = probs.argmax(axis=-1)
        return self.labels[idx], probs[np.arange(len(idx)), idx], self.ood_score(emb)

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            weights=self.weights.astype(np.float32),
            bias=self.bias.astype(np.float32),
            labels=self.labels,
            temperature=np.array([self.temperature], np.float32),
            prototypes=self.prototypes.astype(np.float16),  # half precision is ample
            ood_threshold=np.array([self.ood_threshold], np.float32),
            conf_threshold=np.array([self.conf_threshold], np.float32),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SemanticHead":
        d = np.load(Path(path), allow_pickle=True)
        return cls(
            weights=d["weights"].astype(np.float32),
            bias=d["bias"].astype(np.float32),
            labels=d["labels"],
            temperature=float(d["temperature"][0]),
            prototypes=d["prototypes"].astype(np.float32),
            ood_threshold=float(d["ood_threshold"][0]),
            conf_threshold=float(d["conf_threshold"][0]),
        )


# ----------------------------------------------------------------------
# Fitting helpers
# ----------------------------------------------------------------------
def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Fit a single scalar T minimising NLL of softmax(logits / T)."""
    from scipy.optimize import minimize_scalar

    def nll(log_t: float) -> float:
        z = logits / np.exp(log_t)
        z = z - z.max(axis=1, keepdims=True)
        return float(-(z[np.arange(len(z)), y] - np.log(np.exp(z).sum(1))).mean())

    res = minimize_scalar(nll, bounds=(-2.0, 3.0), method="bounded")
    return float(np.exp(res.x))


def build_prototypes(
    emb: np.ndarray, y: np.ndarray, labels: Sequence[str], per_class: int = 64, seed: int = 0
) -> np.ndarray:
    """k-means centroids per intent, L2-normalised.

    64/class was chosen empirically: it matches the OOD separation of keeping
    every training vector (AUROC 0.998) at 1/20th the size.
    """
    from sklearn.cluster import KMeans

    chunks = []
    for i, _ in enumerate(labels):
        z = emb[y == i]
        if len(z) == 0:
            continue
        k = min(per_class, len(z))
        centers = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(z).cluster_centers_
        centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-9
        chunks.append(centers.astype(np.float32))
    return np.vstack(chunks)


def choose_threshold(
    in_domain: np.ndarray, out_of_scope: np.ndarray, min_retention: float = 0.95
) -> float:
    """Pick a gate threshold maximising (in-domain kept − out-of-scope kept).

    Youden's J, restricted to thresholds that still retain `min_retention` of
    in-domain traffic. Rejecting a real request is worse than occasionally
    letting an out-of-scope one through, so retention is a hard constraint and
    separation is the objective.

    `in_domain` should be scored on *realistic* utterances, not on the
    template corpus. Template phrases score near-identically to each other, so
    a percentile taken on them sets an operating point far tighter than real
    speech ever reaches — which shows up as correct predictions being routed
    to fallback.
    """
    candidates = np.unique(np.concatenate([in_domain, out_of_scope]))
    best, best_j = float(candidates.min()), -np.inf
    for t in candidates:
        retention = float((in_domain >= t).mean())
        if retention < min_retention:
            continue
        j = retention - float((out_of_scope >= t).mean())
        if j > best_j:
            best_j, best = j, float(t)
    return best


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, bins: int = 12) -> float:
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)
