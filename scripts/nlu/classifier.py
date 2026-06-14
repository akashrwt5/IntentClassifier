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
MODEL_PATH  = BASE_DIR / "models" / "intent_model.onnx"
LABELS_PATH = BASE_DIR / "models" / "intent_labels.pkl"
SCHEMA_PATH = BASE_DIR / "data"   / "nlu_schema.json"


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


class IntentClassifier:
    def __init__(self,
                 model_path:  Path = MODEL_PATH,
                 labels_path: Path = LABELS_PATH,
                 schema_path: Path = SCHEMA_PATH):
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

    def _keyword_match(self, text: str):
        """Return intent name if a declarative keyword rule fires, else None."""
        t = text.lower().strip()
        for rule in self._kw_rules:
            kind = rule.get("type")
            if kind == "contains":
                if any(term in t for term in rule["terms"]):
                    return rule["intent"]
            elif kind == "exact":
                if t in rule["terms"]:
                    return rule["intent"]
            elif kind == "regex":
                if rule["pattern"].search(t):
                    if rule["not_pattern"] is None or not rule["not_pattern"].search(t):
                        return rule["intent"]
        return None

    def _format(self, text: str):
        t = text.lower().strip()
        return (np.array([[t]], dtype=object)
                if len(self.inp.shape) == 2
                else np.array([t], dtype=object))

    def classify(self, text: str):
        kw = self._keyword_match(text)
        if kw:
            return kw, 1.0

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

        top = int(np.argmax(scores))
        return self.labels[top], float(scores[top])
