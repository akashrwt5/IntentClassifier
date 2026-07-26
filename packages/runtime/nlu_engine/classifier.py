"""
Intent classifier wrapper around the ONNX model.

Thin, reusable layer over the TF-IDF + LogisticRegression ONNX model,
plus a schema-driven keyword pre-filter (no intent names hardcoded here).
"""

import json
import re
from functools import lru_cache
import numpy as np
import joblib
from pathlib import Path

from .manifest import verify_manifest
from .text_norm import normalize_text

BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH   = BASE_DIR / "models" / "intent_model.onnx"
LABELS_PATH  = BASE_DIR / "models" / "intent_labels.pkl"
SCHEMA_PATH  = BASE_DIR / "content" / "nlu_schema.json"
WEIGHTS_PATH = BASE_DIR / "models" / "intent_classifier_weights.json"


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax (subtract per-row max)."""
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)


def _load_temperature(weights_path: Path,
                      calibration_path: Path | None = None) -> float:
    """The calibration temperature `T` for softmax(logits / T).

    PRECEDENCE — `calibration.json` beats `weights.json`, and the order is the
    whole point (Review-F5 blocker B8).

    `calibration.json` is written by `nlu_training.fit_calibration`: fit
    OUT-OF-FOLD on the same featurizer the shipped ONNX uses, with evaluation
    sets excluded, and carrying provenance. `weights.json` carries the iOS
    DEVICE temperature, fit against a pruned 1370-term vocabulary. Applying that
    device value to full-vocab server logits is what B8 is: it shipped as
    T=0.796 where the correct server value is 0.657, and no test could see it
    because temperature is rank-preserving — it changes only confidence, never
    which intent wins.

    A missing file means T = 1.0 (plain softmax), so older artifacts still load.
    """
    for path in (calibration_path, weights_path):
        if path is None:
            continue
        try:
            meta = json.loads(Path(path).read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, TypeError):
            continue
        if "temperature" in meta:
            return float(meta["temperature"])
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
# want to translate this" should not fire translation.session.start. We only
# guard `contains` rules; `exact`/`regex` authors express negation explicitly.
#
# ENGLISH FALLBACK ONLY. A pack/lexicon supplies its own cues via
# `IntentClassifier(negation_cues=...)`; this table is what a caller that
# provides none gets. The `_DEFAULT_` prefix is the language-neutrality guard's
# convention for an overridable data table (see
# scripts/ci/check_language_neutral.py).
#
# Previously named `_NEGATIONS` and consulted unconditionally, which made
# negation suppression English-only for every language. Note this path is
# currently DEAD in the shipped configuration — the schema declares 28 `regex`
# and 4 `exact` triggers and zero `contains` rules — so the defect is latent
# rather than live. It becomes live the moment a `contains` rule is authored.
_DEFAULT_NEGATIONS = ("not", "don't", "dont", "do not", "never", "without",
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


def _is_negated(text: str, term: str, cues=_DEFAULT_NEGATIONS) -> bool:
    """True if `term` appears negated in `text` (a negation cue precedes it).

    `cues` is language-specific and comes from the pack/lexicon; it defaults to
    the English table so a caller that supplies none behaves as before.

    Cues are matched on WORD BOUNDARIES, not as bare substrings. Short cues are
    the norm outside English — fr "ne", da "ikke", de "kein" — and a substring
    test would fire them inside unrelated words (de "ne" inside "ohne"/"eine",
    en "not" inside "nothing"/"another"), suppressing legitimate commands. The
    boundary is Unicode-aware so accented cues ("arrête", "n'") match correctly.
    """
    idx = text.find(term)
    if idx < 0:
        return False
    # Look only at the short window before the term so an unrelated earlier
    # negation doesn't suppress a genuine later command.
    window = text[:idx][-30:]
    return any(re.search(_negation_pattern(cue), window) for cue in cues)


@lru_cache(maxsize=256)
def _negation_pattern(cue: str) -> "re.Pattern":
    """Word-boundary matcher for one cue, compiled once.

    `\\b` is unreliable next to apostrophes and accented characters, so the
    boundary is expressed as explicit lookarounds over the word-character set
    this engine treats as letters.
    """
    return re.compile(rf"(?<![0-9A-Za-zÀ-ÿ]){re.escape(cue)}(?![0-9A-Za-zÀ-ÿ])")


class IntentClassifier:
    def __init__(self,
                 model_path:   Path = MODEL_PATH,
                 labels_path:  Path = LABELS_PATH,
                 schema_path:  Path = SCHEMA_PATH,
                 weights_path: Path = WEIGHTS_PATH,
                 backend=None,
                 negation_cues=None,
                 calibration_path: Path | None = None):
        if backend is None and not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. Run `python scripts/train.py` first."
            )
        verify_manifest(BASE_DIR)
        self._schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        self._kw_rules = _compile_keyword_rules(self._schema)
        self.labels = joblib.load(str(labels_path))
        # Inference-inversion seam (runtime-contract-v1 §2): the classifier
        # never owns an ML runtime — the host injects one (ORT by default).
        if backend is None:
            from .inference import OrtIntentBackend
            backend = OrtIntentBackend(model_path, len(self.labels))
        self.backend = backend
        # Calibration temperature for softmax(logits / T). Sourced from the
        # exported weights JSON; defaults to 1.0 (plain softmax) when absent.
        self.temperature = _load_temperature(weights_path, calibration_path)
        # Language-specific negation cues, supplied by the caller from the
        # pack/lexicon. None => the English fallback table.
        self.negation_cues = tuple(negation_cues) if negation_cues else _DEFAULT_NEGATIONS
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
                if hit and not _is_negated(t, hit, self.negation_cues):
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

    def classify(self, text: str):
        kw_intent, kw_conf = self._keyword_match(text)
        if kw_intent:
            self.last_stage = "keyword"
            return kw_intent, kw_conf

        self.last_stage = "tfidf"
        # Normalise the surface form (expand contractions, drop apostrophes) so
        # the ONNX tokenizer sees exactly what sklearn saw at fit time. Applied
        # to the TF-IDF path ONLY — the keyword stage above matches raw text.
        # MUST stay identical to the normalisation in nlu_training/train.py.
        scores = self.backend.tfidf_logits(normalize_text(text))
        if isinstance(scores, dict):  # zipmap-style graph output
            scores = np.array([scores[l] for l in self.labels], dtype=float)

        # `scores` are raw decision-function logits (the ONNX graph is exported
        # with raw_scores=True). Temperature scaling is rank-preserving — dividing
        # by a positive scalar leaves argmax unchanged — so intent selection
        # equals the raw-logit argmax; only the confidence is rescaled.
        scaled = scores / self.temperature
        top = int(np.argmax(scaled))
        conf = float(_stable_softmax(scaled)[top])
        return self.labels[top], conf
