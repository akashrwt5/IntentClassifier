"""
Semantic intent classifier: frozen MiniLM encoder + calibrated linear head.

Replaces the TF-IDF bag-of-words classifier, whose failure mode was that a
word's *presence* decided the intent regardless of its role in the sentence —
so "it's too quiet, make it louder" was answered with volume.decrease, because
"quiet" carries decrease mass in the training corpus.

Public surface:
    SemanticIntentClassifier  — runtime predictor (one ONNX file + vocab)
    Prediction                — (intent, confidence, ood_score, accepted)
"""

from .predict import Prediction, SemanticIntentClassifier

__all__ = ["SemanticIntentClassifier", "Prediction"]
__version__ = "0.1.0"
