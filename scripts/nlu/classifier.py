"""
Intent classifier wrapper around the ONNX model.

Thin, reusable layer over the same TF-IDF + LogisticRegression ONNX model that
scripts/predict.py uses, plus the keyword pre-filter for app-name intents.
Returns a clean (intent, confidence) tuple for the NLU engine to consume.
"""

import numpy as np
import joblib
import onnxruntime as ort
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "intent_model.onnx"
LABELS_PATH = BASE_DIR / "models" / "intent_labels.pkl"


def _keyword_match(text: str):
    """Deterministic rules for intents that always contain the feature name."""
    t = text.lower().strip()
    if "translate" in t:
        return "TRANSLATE"
    if "transcribe" in t or "transcription" in t:
        return "TRANSCRIBE"
    if "telehear" in t:
        return "TELEHEARAI"
    if "selfcheck" in t or "self check" in t or "self-check" in t:
        return "SELFCHECK"
    if t in ("mute", "silence"):
        return "VOLUME_MUTE"
    if t == "unmute":
        return "VOLUME_UNMUTE"
    return None


class IntentClassifier:
    def __init__(self, model_path: Path = MODEL_PATH, labels_path: Path = LABELS_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. Run `python scripts/train.py` first."
            )
        self.session = ort.InferenceSession(str(model_path))
        self.inp = self.session.get_inputs()[0]
        self.input_name = self.inp.name
        self.labels = joblib.load(str(labels_path))

    def _format(self, text: str):
        t = text.lower().strip()
        return (np.array([[t]], dtype=object)
                if len(self.inp.shape) == 2
                else np.array([t], dtype=object))

    def classify(self, text: str):
        """Return (intent, confidence). Keyword rules win with confidence 1.0."""
        kw = _keyword_match(text)
        if kw:
            return kw, 1.0

        outputs = self.session.run(None, {self.input_name: self._format(text)})
        # locate the probability tensor
        scores = None
        for o in outputs:
            if hasattr(o, "shape") and o.shape and o.shape[-1] == len(self.labels):
                scores = o[0]
                break
        if scores is None:
            scores = outputs[-1][0]

        if isinstance(scores, dict):
            scores = np.array([scores[l] for l in self.labels], dtype=float)
        else:
            scores = np.asarray(scores, dtype=float)

        top = int(np.argmax(scores))
        return self.labels[top], float(scores[top])
