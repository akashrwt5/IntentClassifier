"""A real OOD score for the safety gate.

The gate currently has two signals, confidence and top1-top2 margin, and BOTH
are read off the same softmax. A softmax is a normalized comparison between the
57 known classes — it answers "which of these is it", never "is it any of
these". So an ASR fragment like "and push it down for dramatics" lands near the
volume-down region, gets 0.974 confidence and a 0.95 margin, and sails through
the gate. No amount of extra training data fixes that, because the fragment
really does look like a volume command.

This module adds a third, independent signal computed in EMBEDDING space,
before the classifier: how far is this input from anything the model was
trained on?

    Mahalanobis: class-conditional Gaussians with a shared covariance.
        d(x) = min_c (x - mu_c)^T P (x - mu_c)

Shared covariance (rather than per-class) keeps it estimable — several intents
have only ~50 training rows, far too few for a 384x384 per-class covariance —
and it is what makes the score deployable: with P = L L^T, whitening x once by
L^T turns every class distance into a plain squared euclidean distance against
a stored centroid. The whitening can be folded into the ONNX graph, leaving the
phone with 57 cheap distance computations and no matrix algebra.

Energy is provided too, as a logits-only fallback for comparison:
    E(x) = -logsumexp(logits)
It needs no extra state but it is still a function of the classifier head, so
it is far less independent of the softmax than the embedding-space score.
"""

from __future__ import annotations

import numpy as np


class OODScorer:
    """Higher score = more out-of-distribution."""

    def __init__(self, method: str = "mahalanobis", shrinkage: float = 0.10):
        if method not in ("mahalanobis", "energy", "none"):
            raise ValueError(method)
        self.method = method
        self.shrinkage = shrinkage
        self.labels_: list = []
        self.mu_: np.ndarray | None = None  # (n_classes, dim)
        self.L_: np.ndarray | None = None  # whitening, (dim, dim)
        self.mu_white_: np.ndarray | None = None  # (n_classes, dim)
        self.train_quantiles_: dict = {}

    # -- fitting -----------------------------------------------------------
    def fit(self, X: np.ndarray, y: list, logits: np.ndarray | None = None):
        if self.method == "none":
            return self
        if self.method == "energy":
            if logits is not None:
                s = self._energy(logits)
                self.train_quantiles_ = self._quantiles(s)
            return self

        X = np.asarray(X, dtype=np.float64)
        self.labels_ = sorted(set(y))
        idx = {l: i for i, l in enumerate(self.labels_)}
        yi = np.array([idx[v] for v in y])

        d = X.shape[1]
        mu = np.zeros((len(self.labels_), d))
        centred = np.empty_like(X)
        for c in range(len(self.labels_)):
            m = yi == c
            mu[c] = X[m].mean(0)
            centred[m] = X[m] - mu[c]
        cov = centred.T @ centred / max(1, len(X) - len(self.labels_))
        # Ledoit-Wolf style shrinkage towards a scaled identity. Without it the
        # covariance of a 384-d embedding estimated from ~14k rows is badly
        # conditioned and the inverse amplifies noise directions.
        cov = (1 - self.shrinkage) * cov + self.shrinkage * np.trace(cov) / d * np.eye(d)
        P = np.linalg.pinv(cov)
        # symmetric PSD square root so that d(x) = ||L^T x - L^T mu||^2
        w, V = np.linalg.eigh((P + P.T) / 2)
        w = np.clip(w, 0, None)
        self.L_ = (V * np.sqrt(w)) @ V.T
        self.mu_ = mu
        self.mu_white_ = mu @ self.L_
        self.train_quantiles_ = self._quantiles(self.score(X))
        return self

    @staticmethod
    def _quantiles(s: np.ndarray) -> dict:
        return {str(q): float(np.quantile(s, q / 100)) for q in (50, 90, 95, 97.5, 99, 99.5, 100)}

    # -- scoring -----------------------------------------------------------
    @staticmethod
    def _energy(logits: np.ndarray) -> np.ndarray:
        m = logits.max(1, keepdims=True)
        return -(m.ravel() + np.log(np.exp(logits - m).sum(1)))

    def score(self, X: np.ndarray | None = None, logits: np.ndarray | None = None) -> np.ndarray:
        if self.method == "none":
            return np.zeros(len(X if X is not None else logits))
        if self.method == "energy":
            return self._energy(np.asarray(logits, dtype=np.float64))
        Z = np.asarray(X, dtype=np.float64) @ self.L_
        # ||z - mu||^2 for every class, take the closest
        d2 = (Z**2).sum(1)[:, None] - 2 * Z @ self.mu_white_.T + (self.mu_white_**2).sum(1)[None, :]
        return np.sqrt(np.clip(d2.min(1), 0, None))

    def threshold_at(self, percentile: float) -> float:
        """Score below which `percentile`% of TRAINING data falls."""
        key = str(percentile)
        if key in self.train_quantiles_:
            return self.train_quantiles_[key]
        return float("inf")

    # -- deployment --------------------------------------------------------
    def export(self) -> dict:
        """Everything the phone needs. The whitening matrix L is folded into
        the ONNX graph, so the runtime only stores the whitened centroids."""
        if self.method != "mahalanobis":
            return dict(method=self.method)
        return dict(
            method="mahalanobis",
            labels=self.labels_,
            whitened_centroids=self.mu_white_.astype(np.float32).tolist(),
            dim=int(self.mu_white_.shape[1]),
            train_quantiles=self.train_quantiles_,
        )
