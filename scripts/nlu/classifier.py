"""
Intent classifier wrapper around the ONNX model.

Thin, reusable layer over the TF-IDF + LogisticRegression ONNX model,
plus a schema-driven keyword pre-filter (no intent names hardcoded here).
"""

import json
import re
import numpy as np
import joblib
import onnxruntime as ort
from pathlib import Path

from .manifest import verify_manifest

BASE_DIR = Path(__file__).parent.parent.parent
MODEL_PATH   = BASE_DIR / "models" / "intent_model.onnx"
LABELS_PATH  = BASE_DIR / "models" / "intent_labels.pkl"
SCHEMA_PATH  = BASE_DIR / "data"   / "nlu_schema.json"
WEIGHTS_PATH = BASE_DIR / "models" / "intent_classifier_weights.json"


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax (subtract per-row max)."""
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)


def _load_temperature(weights_path: Path) -> float:
    """Read the calibration temperature `T` exported alongside the model.

    The ONNX graph emits raw decision-function logits; confidence is calibrated
    post-hoc by single-parameter temperature scaling: softmax(logits / T). A
    missing file or missing "temperature" key means T = 1.0 (plain softmax), so
    older artifacts stay backward compatible.
    """
    try:
        meta = json.loads(Path(weights_path).read_text(encoding="utf-8"))
        return float(meta.get("temperature", 1.0))
    except (FileNotFoundError, ValueError, TypeError):
        return 1.0


# Honest, match-type-calibrated confidences for keyword hits. A keyword match
# is evidence, not certainty: a full-string exact match is near-certain, a bare
# substring is the weakest signal. Returning a calibrated value (instead of a
# blanket 1.0) keeps the keyword stage from always out-ranking a genuine
# mid-slot interrupt and stops a weak substring hit from masquerading as
# maximum confidence downstream.
KEYWORD_CONFIDENCE = {
    "exact": 0.97,
    "contains": 0.85,
    "regex_guarded": 0.90,   # regex with a not_regex exclusion
    "regex": 0.75,           # bare regex, the broadest pattern
}

# Negation cues that flip the meaning of a `contains` substring hit. "I don't
# want to translate this" should not fire Cmd.TranslationStart. We only guard
# `contains` rules; `exact`/`regex` authors express negation explicitly.
_NEGATIONS = ("not", "don't", "dont", "do not", "never", "without",
              "no need to", "stop", "cancel")


def _compile_keyword_rules(schema: dict) -> list:
    """
    Build a list of compiled rule dicts from schema["keyword_triggers"].

    Each rule is one of:
      {"intent": str, "contains": [str, ...]}       — substring match (any)
      {"intent": str, "exact": [str, ...]}           — full-string match (any)
      {"intent": str, "regex": str, "not_regex": str} — regex match + optional exclusion
    """
    rules = []
    for entry in schema.get("keyword_triggers", []):
        rule = {"intent": entry["intent"]}
        if "contains" in entry:
            rule["type"] = "contains"
            rule["terms"] = entry["contains"]
        elif "exact" in entry:
            rule["type"] = "exact"
            rule["terms"] = entry["exact"]
        elif "regex" in entry:
            rule["type"] = "regex"
            rule["pattern"] = re.compile(entry["regex"])
            rule["not_pattern"] = re.compile(entry["not_regex"]) if "not_regex" in entry else None
        rules.append(rule)
    return rules


def _is_negated(text: str, term: str) -> bool:
    """True if `term` appears negated in `text` (a negation cue precedes it)."""
    idx = text.find(term)
    if idx < 0:
        return False
    prefix = text[:idx]
    # Look only at the short window before the term so an unrelated earlier
    # "not" doesn't suppress a genuine later command.
    window = prefix[-30:]
    return any(neg in window for neg in _NEGATIONS)


class IntentClassifier:
    def __init__(self,
                 model_path:   Path = MODEL_PATH,
                 labels_path:  Path = LABELS_PATH,
                 schema_path:  Path = SCHEMA_PATH,
                 weights_path: Path = WEIGHTS_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. Run `python scripts/train.py` first."
            )
        verify_manifest(BASE_DIR)
        self._schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        self._kw_rules = _compile_keyword_rules(self._schema)
        self.session = ort.InferenceSession(str(model_path))
        self.inp = self.session.get_inputs()[0]
        self.input_name = self.inp.name
        self.labels = joblib.load(str(labels_path))
        # Calibration temperature for softmax(logits / T). Sourced from the
        # exported weights JSON; defaults to 1.0 (plain softmax) when absent.
        self.temperature = _load_temperature(weights_path)
        self.last_stage = None         # "keyword" | "tfidf" — set on each classify()
        self.last_keyword_tier = None  # "exact"|"contains"|"regex"|"regex_guarded"|None

    def _keyword_match(self, text: str):
        """
        Return (intent, confidence) if a declarative keyword rule fires, else
        (None, 0.0). Confidence is calibrated by match type (see
        KEYWORD_CONFIDENCE); `contains` hits are suppressed when negated.
        """
        t = text.lower().strip()
        self.last_keyword_tier = None
        for rule in self._kw_rules:
            kind = rule.get("type")
            if kind == "contains":
                hit = next((term for term in rule["terms"] if term in t), None)
                if hit and not _is_negated(t, hit):
                    self.last_keyword_tier = "contains"
                    return rule["intent"], KEYWORD_CONFIDENCE["contains"]
            elif kind == "exact":
                if t in rule["terms"]:
                    self.last_keyword_tier = "exact"
                    return rule["intent"], KEYWORD_CONFIDENCE["exact"]
            elif kind == "regex":
                if rule["pattern"].search(t):
                    if rule["not_pattern"] is None or not rule["not_pattern"].search(t):
                        tier = "regex" if rule["not_pattern"] is None else "regex_guarded"
                        self.last_keyword_tier = tier
                        return rule["intent"], KEYWORD_CONFIDENCE[tier]
        return None, 0.0

    def _format(self, text: str):
        t = text.lower().strip()
        return (np.array([[t]], dtype=object)
                if len(self.inp.shape) == 2
                else np.array([t], dtype=object))

    def classify(self, text: str):
        kw_intent, kw_conf = self._keyword_match(text)
        if kw_intent:
            self.last_stage = "keyword"
            return kw_intent, kw_conf

        self.last_stage = "tfidf"
        outputs = self.session.run(None, {self.input_name: self._format(text)})
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

        # `scores` are raw decision-function logits (the ONNX graph is exported
        # with raw_scores=True). Temperature scaling is rank-preserving — dividing
        # by a positive scalar leaves argmax unchanged — so intent selection
        # equals the raw-logit argmax; only the confidence is rescaled.
        scaled = scores / self.temperature
        top = int(np.argmax(scaled))
        conf = float(_stable_softmax(scaled)[top])
        return self.labels[top], conf
