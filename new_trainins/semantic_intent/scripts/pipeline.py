"""Shared train/predict plumbing so benchmark, calibration and final training
all use exactly the same code path."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FALLBACK = "Default Fallback Intent"


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv")


def make_classifier(spec: str, seed: int = 0):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import LinearSVC

    if spec == "logreg":
        return LogisticRegression(max_iter=3000, C=8.0, random_state=seed)
    if spec == "logreg_balanced":
        return LogisticRegression(max_iter=3000, C=8.0, class_weight="balanced", random_state=seed)
    if spec == "linsvm":
        # LinearSVC has no probabilities; wrap it rather than pretending its
        # decision_function is a probability (plan Section 14.B).
        return CalibratedClassifierCV(LinearSVC(C=1.0, random_state=seed), method="sigmoid", cv=3)
    if spec == "linsvm_balanced":
        return CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced", random_state=seed), method="sigmoid", cv=3
        )
    if spec == "mlp":
        # early_stopping is left off deliberately: sklearn's internal scorer
        # chokes on string labels in this version, and the loss-plateau stop
        # below is enough for a 256-unit head on 384-d embeddings.
        return MLPClassifier(
            hidden_layer_sizes=(256,),
            alpha=1e-4,
            max_iter=400,
            early_stopping=False,
            n_iter_no_change=15,
            tol=1e-4,
            random_state=seed,
        )
    raise ValueError(spec)


def decision_logits(clf, X: np.ndarray) -> np.ndarray:
    """Return pre-softmax scores. For models that only expose probabilities
    (calibrated SVM), take log-probabilities so temperature scaling still has
    a meaningful surface to act on."""
    if hasattr(clf, "decision_function"):
        d = clf.decision_function(X)
        if d.ndim == 1:
            d = np.column_stack([-d, d])
        return d.astype(np.float64)
    return np.log(np.clip(clf.predict_proba(X), 1e-12, None))


class IntentModel:
    def __init__(self, encoder, clf_spec: str, seed: int = 0):
        self.encoder = encoder
        self.clf_spec = clf_spec
        self.clf = make_classifier(clf_spec, seed)
        self.labels: list[str] = []
        self.temperature = 1.0
        self.gate = None
        self.ood = None
        self.fit_seconds = 0.0

    def fit(self, texts, labels):
        t0 = time.perf_counter()
        if getattr(self.encoder, "needs_fit", False):
            self.encoder.fit(list(texts))
        X = self.encoder.encode(list(texts))
        self.clf.fit(X, list(labels))
        self.labels = list(self.clf.classes_)
        self.fit_seconds = round(time.perf_counter() - t0, 2)
        return self

    def embed(self, texts) -> np.ndarray:
        return self.encoder.encode(list(texts))

    def logits(self, texts) -> np.ndarray:
        return decision_logits(self.clf, self.encoder.encode(list(texts)))

    def ood_scores(self, texts=None, X=None, logits=None) -> np.ndarray:
        """None when no scorer is fitted, so the gate falls back to the
        two-signal behaviour instead of silently scoring everything zero."""
        if self.ood is None or self.ood.method == "none":
            return None
        if X is None:
            X = self.embed(texts)
        if logits is None and self.ood.method == "energy":
            logits = decision_logits(self.clf, X)
        return self.ood.score(X, logits)

    def decide(self, texts) -> list[dict]:
        """One encode, one classify, one OOD score, one gate verdict."""
        X = self.embed(texts)
        lg = decision_logits(self.clf, X)
        return self.gate.decide(lg, self.ood_scores(X=X, logits=lg), texts=list(texts))

    def probs(self, texts, calibrated: bool = True) -> np.ndarray:
        from calibration import softmax

        T = self.temperature if calibrated else 1.0
        return softmax(self.logits(texts) / T)

    def y_index(self, labels) -> np.ndarray:
        idx = {l: i for i, l in enumerate(self.labels)}
        return np.array([idx[l] for l in labels])

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(self, f)
        meta = dict(
            clf=self.clf_spec,
            labels=self.labels,
            temperature=self.temperature,
            encoder=self.encoder.meta(),
            gate=self.gate.to_dict() if self.gate else None,
            ood=(
                dict(method=self.ood.method, train_quantiles=self.ood.train_quantiles_)
                if self.ood
                else None
            ),
        )
        (path / "meta.json").write_text(json.dumps(meta, indent=2))

    @staticmethod
    def load(path: Path) -> "IntentModel":
        with open(Path(path) / "model.pkl", "rb") as f:
            return pickle.load(f)
