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
# Dynamic paths provided by engine.py at runtime


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


# Match-type ordering for keyword hits, retained for TELEMETRY AND ORDERING ONLY.
#
# These values no longer reach `classify()`'s confidence output, and must not be
# put back there. A keyword rule is deterministic; it cannot produce a
# probability, and any constant written into the confidence field is a category
# error rather than a badly-chosen number.
#
# What went wrong when they were live: `regex` returned 0.75, which the engine
# compared against `uncertain_confirm.below_confidence` — a band fitted
# out-of-fold on temperature-calibrated softmax probabilities. Two incompatible
# scales, one comparison. `0.75 < 0.91` is arithmetically true and semantically
# meaningless, so every one of the schema's 28 `regex` rules became permanently
# un-fireable the day that band moved 0.80 -> 0.91 (blocker B8). "increase
# volume" asked the user for confirmation while the model scored it 0.9992.
#
# They were also inverted against measured precision: `regex` measures 95.4%
# accurate and was assigned the LOWEST value, `regex_guarded` measures 76.7% and
# was assigned the second-highest. See docs/confirm-gate-diagnosis.md.
KEYWORD_TIER_ORDER = ("exact", "regex_guarded", "contains", "regex")

# Negation cues that flip the meaning of a `contains` substring hit. "I don't
# want to translate this" should not fire Cmd.TranslationStart. We only
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
    # Confidence reported when a keyword rule fires but the model's top
    # prediction is a DIFFERENT intent. The rule still wins the label — it is a
    # deliberate product decision — but the disagreement is real evidence of
    # ambiguity and the number must say so.
    #
    # PROVISIONAL. This is the one constant left in the confidence path, and it
    # is currently a placeholder chosen to land inside the confirmation band. It
    # must be fitted out-of-fold on train.csv (never on the holdout — that is
    # blocker B9) by the joint (FIRE, FLOOR) sweep described in step 5 of
    # docs/confirm-gate-remediation-plan.md. Shipping a fitted 0.75 in place of
    # a guessed 0.75 would repeat the original defect with better manners.
    #
    # Basis for the placeholder: contested keyword predictions measure ~45%
    # correct on holdout_honest.csv (n=20) versus 99.1% when corroborated.
    CONTESTED_CONFIDENCE = 0.60

    def __init__(self,
                 model_path:   Path | None = None,
                 labels_path:  Path | None = None,
                 weights_path: Path | None = None,
                 schema_path:  Path | None = None,
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
        self.labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.suffix == ".json" else joblib.load(str(labels_path))
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
        # Label -> row index, built once. `calibrated_confidence` is on the turn
        # path and must not do a linear scan of the label list per call.
        self._label_index = {label: i for i, label in enumerate(self.labels)}
        # Per-turn observability, all set by classify(). Annotated because the
        # bare `= None` made mypy infer NoneType and reject every real value.
        self.last_stage: str | None = None            # "keyword" | "tfidf"
        self.last_keyword_tier: str | None = None     # "exact"|"contains"|"regex"|"regex_guarded"
        self.last_arbitration: str | None = None      # "corroborated" | "contested"
        self.last_distribution: np.ndarray | None = None  # calibrated probabilities

    def _keyword_match(self, text: str):
        """
        Return the intent of the first declarative keyword rule that fires, else
        None. The match tier is recorded on ``self.last_keyword_tier`` (the
        engine's interrupt logic reads it); `contains` hits are suppressed when
        negated.

        Deliberately returns NO confidence. A rule is deterministic and cannot
        express one — see the note on ``KEYWORD_TIER_ORDER``.
        """
        t = text.lower().strip()
        self.last_keyword_tier = None
        for rule in self._kw_rules:
            kind = rule.get("type")
            if kind == "contains":
                hit = next((term for term in rule["terms"] if term in t), None)
                if hit and not _is_negated(t, hit, self.negation_cues):
                    self.last_keyword_tier = "contains"
                    return rule["intent"]
            elif kind == "exact":
                if t in rule["terms"]:
                    self.last_keyword_tier = "exact"
                    return rule["intent"]
            elif kind == "regex":
                if rule["pattern"].search(t):
                    if rule["not_pattern"] is None or not rule["not_pattern"].search(t):
                        tier = "regex" if rule["not_pattern"] is None else "regex_guarded"
                        self.last_keyword_tier = tier
                        return rule["intent"]
        return None

    def _model_distribution(self, text: str) -> np.ndarray:
        """Calibrated probability over the full label space.

        Normalise the surface form (expand contractions, drop apostrophes) so
        the ONNX tokenizer sees exactly what sklearn saw at fit time. Applied to
        the MODEL path only — the keyword stage matches raw text. MUST stay
        identical to the normalisation in nlu_training/train.py.

        `scores` are raw decision-function logits (the ONNX graph is exported
        with raw_scores=True). Temperature scaling is rank-preserving — dividing
        by a positive scalar leaves argmax unchanged — so intent selection
        equals the raw-logit argmax; only the confidence is rescaled.
        """
        scores = self.backend.tfidf_logits(normalize_text(text))
        if isinstance(scores, dict):  # zipmap-style graph output
            scores = np.array([scores[l] for l in self.labels], dtype=float)
        return _stable_softmax(np.asarray(scores, dtype=float) / self.temperature)

    def classify(self, text: str):
        """Arbitrate between the keyword rules and the model.

        The model runs on EVERY turn, and is the sole author of the confidence
        this returns. The rule, when one fires, is the sole author of the label.
        Separating those two responsibilities is the point:

        * a rule is a deliberate, hand-authored product decision about what an
          utterance means, so it decides the label;
        * only the model produces a calibrated probability, and confidence is
          compared downstream against thresholds fitted on exactly that scale,
          so it decides the number.

        Previously the keyword stage short-circuited the model and returned a
        hardcoded constant, which put two incompatible scales in one field.
        "increase volume" returned 0.75 (the `regex` literal) and was held for
        confirmation, while the model scored it 0.9992.

        Corroborated (rule and model agree) measures 99.1% correct; contested
        (they disagree) measures ~45% on the honest holdout — a coin flip, and
        exactly the condition a confirmation exists to catch. See
        docs/confirm-gate-diagnosis.md.

        Cost: one extra inference on the ~9% of turns that hit a keyword rule;
        the rest already ran the model. Measured at 0.06 ms.
        """
        kw_intent = self._keyword_match(text)
        p = self._model_distribution(text)
        self.last_distribution = p
        top = int(np.argmax(p))
        model_intent, model_conf = self.labels[top], float(p[top])

        if not kw_intent:
            self.last_stage = "tfidf"
            self.last_arbitration = None
            return model_intent, model_conf

        self.last_stage = "keyword"
        if model_intent == kw_intent:
            self.last_arbitration = "corroborated"
            return kw_intent, model_conf
        self.last_arbitration = "contested"
        return kw_intent, self.CONTESTED_CONFIDENCE

    # Token pattern sklearn's TfidfVectorizer uses by default. The guard must
    # split text the same way the featurizer does, or it counts words the model
    # was never offered and reports an out-of-vocabulary share that describes
    # nothing.
    _TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")

    def oov_ratio(self, text: str) -> float:
        """Share of this utterance's tokens the featurizer cannot represent.

        WHY THIS EXISTS. TF-IDF's vocabulary is a fixed set of slots. A token
        outside it is not weighed and dismissed — there is nowhere to put it, so
        the sentence arrives without it:

            'turn off'          -> 3 non-zero features
            'turn off toshiba'  -> 3 non-zero features, cosine 1.000000

        The two are bit-identical, so no threshold, training row or
        hyperparameter can separate them: the model is never asked the
        question. And the word that makes an utterance out of scope is almost
        always a rare, specific one — a brand, an object, a topic — which is
        precisely the kind of word a finite vocabulary lacks.

        That unknown word is itself evidence, and it was being discarded. This
        recovers it: "the utterance contains content the model has never seen"
        is a reason to doubt any confident reading of the remainder.

        Returns 0.0 when the backend cannot report a vocabulary, which disables
        the guard rather than rejecting everything.
        """
        vocab = getattr(self.backend, "unigram_vocabulary", None)
        if vocab is None:
            return 0.0
        known = vocab()
        if not known:
            return 0.0
        tokens = self._TOKEN_RE.findall(normalize_text(text).lower())
        if not tokens:
            return 0.0
        return sum(1 for t in tokens if t not in known) / len(tokens)

    def calibrated_confidence(self, intent: str) -> float | None:
        """The model's calibrated probability for `intent` on the LAST turn.

        For callers that change the reported intent after `classify` has run —
        the engine's polarity and help-marker guards — so the confidence can be
        re-derived for the intent actually being reported instead of being
        inherited from the one that was blocked.

        Returns None when there is no distribution yet or the intent is outside
        the model's label space; the caller keeps whatever it had.
        """
        idx = self._label_index.get(intent)
        if self.last_distribution is None or idx is None:
            return None
        return float(self.last_distribution[idx])
