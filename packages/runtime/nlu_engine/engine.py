"""
NLUEngine — the orchestrator that replaces Dialogflow end-to-end.

Per user turn priority order:
  1. CONFIRMATION — active yes/no follow-up context
  2. SLOT FILLING — mid-collection intent (with interruption detection)
  3. CLASSIFY     — fresh turn

Intent interruption: if the user switches topic mid slot-filling with high
confidence (>= schema `interrupt_threshold`), the pending flow is abandoned and
the new intent is handled immediately. The NLUResult carries interrupted_intent
so the app can optionally notify the user. An utterance that is a valid value
for the slot currently being awaited is the ANSWER to the live question and
never interrupts, however confidently it classifies as something else.
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from .classifier import IntentClassifier
from .entities import EntityExtractor
from .context import SessionStore
from .model_paths import resolve_model_set
from . import label_compat

BASE_DIR = Path(__file__).resolve().parents[3]
# Paths are dynamic based on language pack

# Fallback default for the semantic-rescue threshold when the schema omits it.
# The schema's "semantic_threshold" is the single source of truth; this is only
# used if that key is absent.
DEFAULT_SEMANTIC_THRESHOLD = 0.55

# Placeholder GenAI endpoint base. NEVER used at runtime: the startup guard in
# NLUEngine.__init__ rejects it, so a real endpoint must be configured via the
# NLU_GENAI_URL env var or a "genai_url" schema key (Review-F5 Appendix A #5,
# risk RK1: user utterances must never be sent to an unregistered domain).
DEFAULT_GENAI_URL = "https://genai.yourcompany.com/chat?query="

# English carrier phrases — the FALLBACK table used when a pack/lexicon supplies
# none. A carrier is the polite wrapper around a command ("remind me to ...")
# that must be stripped before the remainder becomes a slot topic.
#
# `_DEFAULT_` is the neutrality guard's convention for an overridable DATA table:
# a language's own carriers come from its lexicon and are tried FIRST, with
# these as the tail. English ships no lexicon file, so it uses these alone.
_DEFAULT_CARRIERS = [
    # VIK-041. Politeness prefixes, and they MUST STAY AT THE FRONT.
    #
    # `_derive_topic` makes ONE pass in list order and every pattern is
    # `^`-anchored, so a pattern is only ever tested against the string as it
    # stands when its turn comes. With these last, "^remind me" is tried against
    # "can you remind me…", misses, and is never retried after "^can you" strips:
    #
    #   at the BACK  : 'Can you remind me to go for a walk' -> 'remind me to go for a walk'
    #   at the FRONT : 'Can you remind me to go for a walk' -> 'go for a walk'
    #
    # Which is also why `^please` has always been index 0. Any future politeness
    # prefix inherits the same requirement, in every language.
    #
    # Keep in step with `language_packs/*/platform.yaml`; a carrier that lives in
    # one and not the other is VIK-022 all over again.
    r"^\s*(?:can|could|would|will)\s+you\s+(?:please\s+)?",
    r"^\s*i\s+want\s+you\s+to\b\s*",
    r"^\s*please\s+",

    # Carriers by SHAPE, not by enumerated verb. The list used to name its verbs
    # (`remind|tell|alert|notify`), so every phrasing outside that set kept its
    # wrapper and the whole sentence became the reminder's name — "nudge me to
    # stretch" stored "nudge me to stretch". The set of verbs people use is open;
    # the sentence shape is not.
    #
    # These sit AFTER the prefix-strippers above and BEFORE the specific carriers
    # below, and both halves of that matter. `_derive_topic` makes ONE pass in list
    # order, so a prefix left in front ("please remind me to X") would stop these
    # matching, and a specific carrier that fired first would leave these to run on
    # the payload instead of the wrapper.
    #
    # `\w+\s+me\s+(?:to|about|that|when)` is deliberately verb-agnostic. It is safe
    # BECAUSE of where it runs, not because it is narrow: topic derivation only
    # happens for an OPEN slot, and `remind` is the only open entity in the pack —
    # so the classifier has already decided this is a reminder before any of this
    # executes. It never sees the other 56 intents.
    r"^\s*\w+\s+me\s+(?:to|about|that|when)\b\s*",
    r"^\s*i\s+(?:must|mustn'?t|must\s+not|can'?t|cannot|should|shouldn'?t)\s+forget\s*(?:(?:to|about)\b)?\s*",
    r"^\s*(?:make|add|create|leave)\s+(?:an?\s+)?(?:note|reminder|alarm)\s*(?:(?:to|about|that|for)\b)?\s*",
    r"^\s*(?:do\s*n[o']?t|don't|dont)\s+let\s+me\s+forget\b\s*(?:(?:to|about)\b)?\s*",
    r"^\s*(?:remind|tell|alert|notify)\s+me\b\s*(?:(?:to|that|about|of)\b)?\s*",
    # No `for` branch, and no negative lookahead. The previous form guarded it
    # with `for\s+(?!\d)`; lookahead is outside the portable-regex subset
    # (spec/bundle/portable-regex.md), so `compile_lexicon` silently dropped this
    # entire carrier from every bundle it built. The engine kept it, which meant
    # this table and the bundles disagreed — the engine stripped the carrier and
    # a pack-driven runtime did not, so the same utterance produced two different
    # reminder names.
    #
    # A leading "for" is removed by `_leading_connector` one step later, so the
    # branch was redundant anyway. Keep this in step with
    # `language_packs/*/platform.yaml`; the point of the rewrite is that the two
    # can no longer diverge.
    r"^\s*set(?:\s+up)?\s+(?:an?\s+)?(?:reminder|alarm)\b\s*(?:(?:to|about)\b)?\s*",
    r"^\s*make\s+sure\s+(?:i|to)\b\s*",
    r"^\s*i\s+(?:need|have|want)\s+to\b\s*",
]

# Connectors that dangle at the START of a derived topic once the carrier and
# the time expression have been stripped: "remind me at 9pm for dinner" ->
# "for dinner" -> "dinner". Overridable per language, same convention.
_DEFAULT_LEADING_CONNECTORS = ["for", "about", "of", "on", "to", "that",
                               "regarding", "with"]

logger = logging.getLogger("nlu.engine")


@dataclass
class NLUResult:
    type: str
    intent: Optional[str] = None
    action: Optional[str] = None
    parameters: dict = field(default_factory=dict)
    message: str = ""
    confidence: float = 0.0
    complete: bool = False
    interrupted_intent: Optional[str] = None
    semantic_rescue: bool = False         # True when MiniLM rescued a TF-IDF miss
    tfidf_intent: Optional[str] = None   # what TF-IDF predicted before semantic overruled
    tfidf_confidence: float = 0.0        # TF-IDF confidence before semantic overruled

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None and v != {} and v != ""}


class NLUEngine:
    # Fallback interruption bar for a schema that omits `interrupt_threshold`.
    # The live value is CONTENT-OWNED (content/platform.yaml, per-language via
    # overlay) and fitted by `nlu_training.fit_slot_thresholds`; read it from
    # `self.interrupt_threshold`, not from here.
    #
    # It used to be this constant, hardcoded at 0.75 and justified as "lowered
    # from 0.85 after isotonic calibration". The pipeline calibrates by
    # temperature scaling now, so that justification referred to a confidence
    # scale that no longer exists — and being a class constant, no language pack
    # could override it.
    DEFAULT_INTERRUPT_THRESHOLD = 0.75

    # A user who cannot complete a slot after this many tries is routed out of
    # the flow instead of being trapped in an unanswerable prompt loop.
    #
    # FALLBACK ONLY, same as DEFAULT_INTERRUPT_THRESHOLD above. The live value is
    # CONTENT-OWNED (`max_slot_attempts` in platform.yaml, per-language via
    # overlay) and reaches a device runtime as `policies.limits.max_slot_attempts`;
    # read it from `self.max_slot_attempts`, not from here. It was a bare class
    # constant, so the budget the pack declared and the budget both engines
    # applied were three independent 3s that only agreed by coincidence.
    DEFAULT_MAX_SLOT_ATTEMPTS = 3
    MAX_SLOT_ATTEMPTS = DEFAULT_MAX_SLOT_ATTEMPTS  # back-compat alias

    # How well an utterance must match a value of the slot being awaited before
    # it counts as the ANSWER and is allowed to suppress a topic switch (see
    # _answers_awaited_slot). Genuine answers match at 0.95-1.0 on the strict
    # path; anything looser is noise that would trap the user in the flow.
    SLOT_ANSWER_MATCH_FLOOR = 0.9

    # Relaxed floor used ONLY when two INDEPENDENT recognisers land on the same
    # real intent. Lower than either stage's own bar because mutual
    # corroboration offsets the softmax diffusion of a 57-class head, while
    # still rejecting a flat (genuinely-ambiguous) distribution.
    #
    # Applies to both agreements this engine can observe:
    #   * TF-IDF and the MiniLM semantic head (the original use), and
    #   * a keyword rule and the model (`last_arbitration == "corroborated"`).
    #
    # It is EVIDENCE STRENGTH, not a confidence. The reported confidence stays
    # the model's calibrated probability; only the bar it must clear moves. That
    # distinction is the whole point — inventing a higher number for a
    # corroborated turn would put a second scale back in the confidence field,
    # which is the defect this ladder was rebuilt to remove.
    #
    # Measured on holdout_honest.csv: corroborated keyword turns are 99.2%
    # correct overall (n=118) and 100% correct in the 0.50-0.70 band (n=4) —
    # turns where the rule and the model agree but a sibling class ("too quiet"
    # pulling volume.increase toward volume.decrease) splits the mass.
    #
    # FALLBACK ONLY. The live value is CONTENT-OWNED (`agreement_threshold` in
    # platform.yaml, per-language via overlay) and read into
    # `self.agreement_threshold`; use that, not this. As a bare class constant
    # it had the same defect this codebase already called out in
    # DEFAULT_INTERRUPT_THRESHOLD — no language pack could override it.
    DEFAULT_AGREEMENT_THRESHOLD = 0.50
    AGREEMENT_THRESHOLD = DEFAULT_AGREEMENT_THRESHOLD  # back-compat alias

    def __init__(self, schema_path: Path | None = None, model_name: str | None = None,
                 language: str = "en", semantic_enabled: bool | None = None,
                 pack=None, backend=None):
        # `pack` is the Language Pack seam (nlu_langpack.LanguagePack). When one
        # is supplied its manifest is authoritative for model artifacts and
        # nothing is inferred from the filesystem. When it is None the engine
        # falls back to the per-language build tree — the state today, since no
        # content->bundle compiler exists yet. Accepting it here is what makes
        # the eventual switch a caller change rather than an engine change.
        self.pack = pack
        self.backend = backend
        self.language = (getattr(pack, "language", None) or language)
        if not schema_path: 
            schema_path = BASE_DIR / "language_packs" / self.language / "nlu_schema.json"
            if not schema_path.exists():
                schema_path = BASE_DIR / "language_packs" / "en" / "nlu_schema.json"
        self._schema_path = schema_path
        self.schema = self._load_schema(schema_path, self.language)
        self.intents = self.schema["intents"]
        self.threshold = self.schema.get("confidence_threshold", 0.70)
        self.interrupt_threshold = self.schema.get(
            "interrupt_threshold", self.DEFAULT_INTERRUPT_THRESHOLD)
        self.agreement_threshold = self.schema.get(
            "agreement_threshold", self.DEFAULT_AGREEMENT_THRESHOLD)
        self.max_slot_attempts = self.schema.get(
            "max_slot_attempts", self.DEFAULT_MAX_SLOT_ATTEMPTS)
        # Share of an utterance's tokens that may be absent from the model's
        # vocabulary before the turn is refused outright. `None` disables the
        # guard. Content-owned so a language pack sets its own — the value
        # depends on the vocabulary that pack ships, not on the engine.
        self._oov_reject_ratio = self.schema.get("oov_reject_ratio")
        # Above this confidence an unknown word is read as an entity VALUE
        # rather than as evidence the utterance is out of scope, and the guard
        # stands down. Defaults to 1.01 (never bypass) so a pack that sets a
        # ratio without a ceiling gets the strict behaviour rather than a
        # silently disabled guard.
        self._oov_bypass_confidence = self.schema.get("oov_bypass_confidence", 1.01)
        self.affirmative = set(self.schema.get("affirmative", []))
        self.negative = set(self.schema.get("negative", []))
        # Explicit cancellation cues, honoured while a slot-filling flow is
        # active. Content-owned so a language pack overrides them; the default
        # covers English. A bare refusal (schema `negative`, e.g. "no") is also
        # treated as a cancel by _is_cancel — see there.
        self._cancel_cues = tuple(self.schema.get("cancel_cues", (
            "cancel", "stop", "never mind", "nevermind",
            "forget it", "forget about it", "quit", "abort")))
        # GenAI base URL is configuration, not a result field. The raw user
        # utterance is NEVER embedded into an NLUResult (it would otherwise be
        # captured by any caller that logs the result). The app layer, which
        # already holds the text, constructs the navigation URL itself.
        self.genai_url = self._resolve_genai_url(
            self.schema.get("genai_url") or os.environ.get("NLU_GENAI_URL"))
        # Opt-in raw-utterance logging. Off by default so medical-context
        # speech is never written to logs in production; enable in dev only.
        self._log_utterances = os.environ.get("NLU_LOG_UTTERANCES", "").lower() in ("1", "true", "yes")
        # Single source of truth for the semantic-rescue gate: the schema. The
        # same value constructs SemanticFallback AND gates its result in the
        # engine, so the two can never drift.
        self.semantic_threshold = self.schema.get("semantic_threshold", DEFAULT_SEMANTIC_THRESHOLD)
        # ND-11(b): polarity guards — high-precision lexical rules that stop a
        # confident prediction of the OPPOSITE action ("turn mute on" must
        # never fire unmute). Config-driven per language; see _apply_guards.
        self._polarity_guards = [
            (re.compile(g["pattern"], re.IGNORECASE), g["blocked_intent"], g["redirect_intent"])
            for g in self.schema.get("polarity_guards", [])
        ]
        # ND-14: help-marker guard — a high-precision safety rule that stops a
        # confident STATE-CHANGING action from firing when the user is clearly
        # ASKING how to use a feature ("how do I use transcription?" must show
        # help, never start recording). Redirects the action to its read-only
        # help.* sibling. Read-only queries (activity/battery) are deliberately
        # NOT paired — "how many steps" is a legitimate query, not a help ask.
        hmg = self.schema.get("help_marker_guard", {})
        self._help_pairs = dict(hmg.get("pairs", {}))
        _markers = hmg.get("markers", "")
        self._help_markers = re.compile(_markers, re.IGNORECASE) if _markers else None
        # ND-11(a): uncertainty-confirmation gate — flagged fire-and-forget
        # intents below this confidence get an ask-first turn instead of firing.
        # The decision ladder is BINARY: `confidence_threshold` and below it the
        # fallback intent. There is no confidence-triggered confirmation.
        #
        # There used to be one — `uncertain_confirm`, a band from 0.55 to 0.91
        # over a hand-curated 14-intent list. It was removed rather than
        # retuned, for three reasons:
        #
        #  * The band sat ABOVE the fire threshold, so it converted commands
        #    that would have fired into questions. On the honest holdout it
        #    produced 103 friction turns against 16 useful catches — 85% of
        #    every confirmation a user saw was asked about a CORRECT prediction.
        #  * The product never asked for it. Dialogflow, which this replaces,
        #    has no confidence-triggered confirmation at all: intent matching is
        #    threshold-or-fallback, and confirmation is authored dialogue.
        #    `legacy_label_map.json`'s `confirm_compound` carries exactly one
        #    entry, an explicitly designed send-message flow.
        #  * A confirmation is delivered through audio, which for a
        #    hearing-impaired user is the channel already failing — a "did you
        #    mean volume up?" arrives through the volume they cannot hear.
        #
        # Confirmation that a product genuinely wants is declared per intent as
        # a schema `followup` and is DETERMINISTIC (see `_handle_confirmation`);
        # it fires every time regardless of confidence, which is what a contract
        # with an app requires.
        self._confirm_cancel_msg = self.schema.get("uncertain_confirm", {}).get(
            "cancel_message", "Okay, I won't.")
        self.classifier = self._load_classifier(model_name)
        self.entities = self._load_entities(language)
        self._carrier = self._build_carrier_patterns(language)
        self._leading_connector = self._build_leading_connector(language)
        self.sessions = SessionStore()
        self._availability: dict = {}  # runtime-contract-v1 §5 snapshot
        # Semantic rescue: ONE plug-and-play flag. Resolution order:
        #   1. constructor param (tests/harness/apps),
        #   2. NLU_SEMANTIC_RESCUE env var (ops kill-switch, "0"/"1"),
        #   3. schema 'semantic_rescue_enabled' (content-owned default, per
        #      language via overlay), default True.
        # Disabled = artifacts are never even loaded (zero cost, zero risk of
        # stale-artifact influence).
        self.semantic_enabled = self._resolve_semantic_flag(semantic_enabled)
        if not self.semantic_enabled:
            self.semantic = None
        else:
            # INTERIM: The global semantic head is English-only.
            # Once fully pack-fed, the pack manifest will name its own semantic
            # artifact (e.g. models.semantic_head.<lang>.artifact) and this
            # hardcoded path inference will disappear — see nlu_langpack.
            self.semantic = self._load_semantic(self.semantic_threshold)
        self._assert_label_schema_parity()

    def _resolve_semantic_flag(self, param: bool | None) -> bool:
        if param is not None:
            return param
        env = os.environ.get("NLU_SEMANTIC_RESCUE")
        if env is not None:
            return env.strip().lower() in ("1", "true", "yes", "on")
        return bool(self.schema.get("semantic_rescue_enabled", True))

    @staticmethod
    def _resolve_genai_url(configured: str | None) -> str | None:
        """Startup guard for the GenAI endpoint (Review-F5 Appendix A #5, RK1).

        The placeholder ``DEFAULT_GENAI_URL`` points at an unregistered domain;
        shipping it would leak user utterances (as query params) to whoever
        registers it. It is therefore REJECTED, never silently used:

        - no URL configured  -> GenAI URL disabled (``None``); callers must
          handle the absence. Routing to the FALLBACK result is unaffected.
        - placeholder configured explicitly -> hard error: misconfiguration
          must fail loudly at startup, not at first fallback.
        - any other URL -> used as-is.
        """
        if configured is None:
            logger.warning(
                "No GenAI endpoint configured (NLU_GENAI_URL / schema 'genai_url'); "
                "GenAI fallback URL construction is disabled.")
            return None
        if configured == DEFAULT_GENAI_URL:
            raise RuntimeError(
                "Refusing to start with the placeholder GenAI URL "
                f"({DEFAULT_GENAI_URL!r}). Configure a real endpoint via the "
                "NLU_GENAI_URL env var or the schema 'genai_url' key. "
                "(Review-F5 Appendix A #5, risk RK1)")
        return configured

    @staticmethod
    def _load_schema(schema_path: Path, language: str) -> dict:
        """Load canonical schema then deep-merge the language overlay (if any).

        Overlay keys applied: intents[].fulfillment, intents[].slots[].prompt,
        affirmative, negative, followup prompts. Structural keys (entity, required,
        action) always come from the canonical schema. Missing overlay → English.
        """
        import copy
        schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        # Data-driven, not language-string-driven: a language with an overlay
        # file gets it merged; one without (English, which ships none) uses the
        # canonical schema. No `if language == "en"` needed — absence IS the
        # signal, so a new language is a file, not a branch.
        if not language:
            return schema
        overlay_path = BASE_DIR / "language_packs" / language / "extras" / f"nlu_schema.{language}.json"
        if not overlay_path.exists():
            logger.debug("nlu.schema.no_overlay lang=%s (using canonical schema)", language)
            return schema

        try:
            overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
        except Exception:
            logger.error("nlu.schema.overlay_decode_error lang=%s", language)
            return schema

        merged = copy.deepcopy(schema)

        # Merge per-intent fulfillment + slot prompts
        for intent_name, ov_intent in overlay.get("intents", {}).items():
            if intent_name not in merged["intents"]:
                continue
            if "fulfillment" in ov_intent:
                merged["intents"][intent_name]["fulfillment"] = ov_intent["fulfillment"]
            if "confirm_prompt" in ov_intent:
                merged["intents"][intent_name]["confirm_prompt"] = ov_intent["confirm_prompt"]
            ov_slots = {s["name"]: s for s in ov_intent.get("slots", []) if "name" in s}
            for slot in merged["intents"][intent_name].get("slots", []):
                if slot.get("name") in ov_slots and "prompt" in ov_slots[slot["name"]]:
                    slot["prompt"] = ov_slots[slot["name"]]["prompt"]

        # Merge followup prompts
        for key in ("followup",):
            if key in overlay and key in merged:
                for subkey, val in overlay[key].items():
                    if isinstance(val, dict):
                        merged[key].setdefault(subkey, {}).update(val)
                    else:
                        merged[key][subkey] = val

        # Per-language semantic-rescue override (one flag, overlay-scoped).
        if "semantic_rescue_enabled" in overlay:
            merged["semantic_rescue_enabled"] = overlay["semantic_rescue_enabled"]

        # ND-11: language-specific polarity guards + confirmation-gate texts
        # replace the English defaults wholesale (patterns are per-language).
        if "polarity_guards" in overlay:
            merged["polarity_guards"] = overlay["polarity_guards"]
        if "uncertain_confirm" in overlay:
            merged.setdefault("uncertain_confirm", {}).update(overlay["uncertain_confirm"])

        # ND-14: help-marker guard. The action->help PAIRS are language-neutral
        # (same intent taxonomy), but the MARKERS regex is language-specific, so
        # the overlay replaces only the markers and keeps the shared pairs.
        if "help_marker_guard" in overlay:
            hmg = dict(merged.get("help_marker_guard", {}))
            hmg.update(overlay["help_marker_guard"])
            merged["help_marker_guard"] = hmg

        # Merge yes/no sets
        if "affirmative" in overlay:
            merged["affirmative"] = overlay["affirmative"]
        if "negative" in overlay:
            merged["negative"] = overlay["negative"]

        return merged

    @staticmethod
    def _load_negation_cues(language: str):
        """Per-language negation cues for the classifier's `contains` guard.

        Data-driven, NOT language-string-driven: if a lexicon exists for this
        language its `negation_cues` are used, otherwise None lets the
        classifier fall back to its `_DEFAULT_NEGATIONS` table. There is
        deliberately no `if language == "en"` here — English simply ships no
        lexicon, so it takes the fallback by absence rather than by branch.
        """
        if not language:
            return None
        lex_path = BASE_DIR / "language_packs" / language / "nlu_lexicon.json"
        if not lex_path.exists():
            return None
        try:
            cues = json.loads(lex_path.read_text(encoding="utf-8")).get("negation_cues")
        except Exception:
            logger.warning("nlu.lexicon.negation_cues_unreadable lang=%s", language)
            return None
        return tuple(cues) if cues else None

    @staticmethod
    def _has_localization(language: str) -> bool:
        """True when this language ships a localization lexicon.

        The single data-driven signal this engine uses to tell "a language with
        its own tables" from "the built-in defaults". English ships none, so it
        takes the default path by absence rather than by name.
        """
        return bool(language) and (BASE_DIR / "language_packs" / language / "nlu_lexicon.json").exists()

    @staticmethod
    def _build_leading_connector(language: str):
        """Compile the leading-connector stripper from the lexicon, else defaults."""
        words = list(_DEFAULT_LEADING_CONNECTORS)
        lex_path = BASE_DIR / "language_packs" / language / "nlu_lexicon.json" if language else None
        if lex_path is not None and lex_path.exists():
            try:
                override = json.loads(lex_path.read_text(encoding="utf-8")).get(
                    "leading_connectors")
                if override:
                    words = list(override)
            except Exception:
                logger.warning("nlu.lexicon.connectors_unreadable lang=%s", language)
        alt = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
        # `(?:\s+|$)`, not `\s+`: a connector can be the ENTIRE remainder once the
        # time expression has been stripped out from behind it. "Set a reminder
        # for 5pm" reduces to "for", which `^(?:...)\s+` cannot match because
        # there is no trailing space — so the reminder was created with the name
        # "for". Anchoring on end-of-string as well removes it and leaves no
        # topic, which is correct: the user gave a time and no subject, and the
        # engine should ask for one.
        return re.compile(rf"^(?:{alt})(?:\s+|$)", re.I)

    @staticmethod
    def _build_carrier_patterns(language: str) -> list:
        """Carrier patterns: the language's own first, then the defaults.

        The lexicon's carrier_phrases are regex strings in the same format as
        `_DEFAULT_CARRIERS`, prepended so language-specific patterns win. A
        language with no lexicon file (English) simply gets the defaults —
        selection is by file presence, never by a language literal.
        """
        base = list(_DEFAULT_CARRIERS)
        if not language:
            return base
        lex_path = BASE_DIR / "language_packs" / language / "nlu_lexicon.json"
        if not lex_path.exists():
            return base
        try:
            lex = json.loads(lex_path.read_text(encoding="utf-8"))
            lang_carriers = lex.get("carrier_phrases", [])
            return lang_carriers + base
        except Exception:
            return base

    @staticmethod
    def _load_entities(language: str) -> EntityExtractor:
        """Build an EntityExtractor for the given language.

        A language with an entities file gets it, plus the lexicon-driven
        datetime parser. One without (English) gets the canonical entities and
        the table-driven parser fed by `_DEFAULT_DT_GRAMMAR`. Presence of the
        file is the switch; there is no language literal.
        """
        if not language:
            return EntityExtractor()
        entities_path = BASE_DIR / "language_packs" / language / "nlu_entities.json"
        if not entities_path.exists():
            logger.debug("nlu.entities.no_overlay lang=%s (using canonical entities)", language)
            return EntityExtractor()
        return EntityExtractor(entities_path=entities_path, language=language)

    def _load_classifier(self, model_name: str | None = None) -> IntentClassifier:
        """Load this language's intent model.

        Resolution is delegated to `model_paths.resolve_model_set`, the single
        place that knows where artifacts live: a Language Pack first (its
        manifest names them outright), then the per-language build tree
        `models/intent/<lang>/`, then the legacy flat tree for one transition.

        `model_name` overrides which LANGUAGE's model to load — it used to name
        a directory under the retired `multilingual/models/` tree, which only
        the combined-multilingual trainer wrote. There is no combined model any
        more: each Language Pack carries its own.
        """
        cues = self._load_negation_cues(self.language)
        # No literal default here: the language already defaulted at the
        # constructor's signature, which is configuration. Re-defaulting inside
        # the logic would be the engine deciding a language, which is coupling.
        language = model_name if model_name and model_name != "production" else self.language
        models = resolve_model_set(language, pack=self.pack)
        logger.debug("nlu.model.resolved lang=%s source=%s path=%s",
                     language, models.source, models.model)

        kwargs = {"model_path": models.model, "labels_path": models.labels,
                  "schema_path": self._schema_path, "negation_cues": cues}
        if getattr(self, "backend", None):
            kwargs["backend"] = self.backend
        # Calibration travels with the (model, featurizer) pair. When the
        # per-language artifact exists it wins; otherwise the classifier keeps
        # its legacy default. Fitting the value correctly is charter B2/B3 —
        # this is the plumbing, not the fix.
        if models.weights is not None:
            kwargs["weights_path"] = models.weights
        if models.calibration is not None:
            kwargs["calibration_path"] = models.calibration
        return IntentClassifier(**kwargs)

    @staticmethod
    def _load_semantic(threshold: float):
        """
        Construct the semantic stage. Returns None when its artifacts are
        absent (graceful degradation to TF-IDF only). An out-of-memory failure
        is surfaced loudly rather than silently swallowed so a low-memory
        device that loses the semantic stage is visible in telemetry.
        """
        try:
            from .semantic import SemanticFallback
            return SemanticFallback(threshold=threshold)
        except FileNotFoundError:
            logger.warning("nlu.semantic.unavailable",
                           extra={"nlu": {"reason": "artifacts_missing"}})
            return None
        except MemoryError:
            logger.error("nlu.semantic.oom",
                         extra={"nlu": {"reason": "out_of_memory_loading_minilm"}})
            return None
        except Exception as e:
            logger.error("nlu.semantic.load_failed",
                         extra={"nlu": {"reason": type(e).__name__}})
            return None

    def _assert_label_schema_parity(self):
        """Fail loudly at startup if trained labels and schema intents diverge."""
        if not getattr(self, "labels", None):
            return  # model not yet trained; skip during development
        labels = set(self.labels)
        schema_intents = set(self.intents.keys())
        only_in_model = labels - schema_intents
        only_in_schema = schema_intents - labels
        # Default Fallback Intent is allowed to be schema-only (it's a catch-all)
        only_in_schema.discard("Default Fallback Intent")
        if only_in_model or only_in_schema:
            raise RuntimeError(
                f"Label/schema mismatch detected.\n"
                f"  In model but not schema: {sorted(only_in_model)}\n"
                f"  In schema but not model: {sorted(only_in_schema)}\n"
                f"Re-run `python scripts/train.py` and update nlu_schema.json to match."
            )

    # -- runtime-contract-v1 seams (§1, §4, §5) -------------------------------

    def notify_execution(self, session_id: str, turn_id: str | None,
                         outcome: str, detail: dict | None = None) -> None:
        """Feedback edge from the Action Dispatcher (§4). v1 records the
        outcome for referent/telemetry use; dialogue reaction to async
        outcomes is schema-declared work that lands with the SDK."""
        session = self.sessions.get(session_id)
        session.last_execution = {"turn_id": turn_id, "outcome": outcome,
                                  "detail": detail or {}}
        logger.info("nlu.execution", extra={"nlu": {
            "session_id": session_id, "turn_id": turn_id, "outcome": outcome}})

    def push_availability(self, snapshot: dict) -> None:
        """Availability snapshot push (§5): atomic replace; recognized-but-
        unavailable intents route to an unavailable message, never an action
        (ADR-002 A6 — intents are NEVER removed from the label space)."""
        if snapshot.get("snapshot_id", 0) < self._availability.get("snapshot_id", -1):
            return  # stale push dropped
        self._availability = dict(snapshot)

    def _capability_of(self, intent: str) -> str | None:
        """Which capability owns this intent, read from the schema.

        This used to be a longest-prefix match of the intent id against the
        pushed capability ids — `device.volume.mute`.startswith(`device.volume`).
        That inferred structure from the LABEL, and when the taxonomy moved to
        `Cmd.*` the prefix relationship vanished: every capability pushed as
        `unavailable` quietly resumed firing actions, because nothing could be
        matched to it any more.

        The compiler records the owning capability per intent, so the lookup is
        now a fact rather than a guess and a rename cannot break it. The prefix
        walk is kept only as a fallback for a pack compiled before that field
        existed.
        """
        caps = self._availability.get("capabilities", {})
        if not caps:
            return None
        declared = (self.intents.get(intent) or {}).get("capability")
        if declared is not None:
            return declared if declared in caps else None
        best = None
        for cap_id in caps:
            if (intent == cap_id or intent.startswith(cap_id + ".")) and \
                    (best is None or len(cap_id) > len(best)):
                best = cap_id
        return best

    def _availability_block(self, intent: str) -> Optional["NLUResult"]:
        cap = self._capability_of(intent)
        if cap is None:
            return None
        entry = self._availability["capabilities"][cap]
        if entry.get("state") == "unavailable":
            msg = entry.get("unavailable_response") or self.schema.get(
                "unavailable_message",
                "That feature isn't available right now.")
            return NLUResult(type="FULFILL", intent=intent, action=None,
                             message=msg, confidence=1.0, complete=True)
        return None

    def handle(self, session_id: str, text: str, turn_id: str | None = None) -> NLUResult:
        t0 = time.perf_counter()
        self._current_turn_id = turn_id
        session = self.sessions.get(session_id)  # resets a long-idle session
        now = self.sessions.now()
        session.expire_contexts(now)             # drop TTL-expired dialogue state
        # Bound input length before any inference: very long ASR output or
        # adversarial input should not drive tokenizer/regex work unbounded.
        text = text.strip()[:500]

        confirm = self._active_confirmation(session)
        if confirm:
            result = self._handle_confirmation(session, confirm, text, now)
            entry_stage = "confirm"
        elif session.pending_intent:
            result = self._handle_slot_filling(session, text, now)
            entry_stage = "slot_fill"
        else:
            shortcut = self._try_back_reference(session, text)
            if shortcut:
                result, entry_stage = shortcut, "back_reference"
            else:
                result = self._handle_new_intent(session, text, now)
                entry_stage = None  # resolved from the classifier/result below

        self._log_decision(session_id, text, result, entry_stage,
                           (time.perf_counter() - t0) * 1000.0)
        # Telemetry above logs modern labels; app boundary gets legacy names.
        return label_compat.apply(result)

    def _log_decision(self, session_id, text, result, entry_stage, latency_ms):
        """Emit one structured telemetry record per turn (no raw text by default)."""
        if entry_stage == "slot_fill" and result.interrupted_intent:
            stage = "interrupt"
        elif entry_stage == "slot_fill" and result.type == "FALLBACK":
            stage = "slot_abandon"
        elif entry_stage is not None:
            stage = entry_stage
        elif result.semantic_rescue:
            stage = "semantic"
        elif result.type == "FALLBACK" and result.intent == "GENAI":
            stage = "genai"
        else:
            stage = getattr(self.classifier, "last_stage", "tfidf")

        record = {
            "session_id": session_id,
            "turn_id": getattr(self, "_current_turn_id", None),
            "stage": stage,
            "type": result.type,
            "intent": result.intent,
            "confidence": round(result.confidence, 4),
            "semantic_rescue": result.semantic_rescue,
            "interrupted_intent": result.interrupted_intent,
            # Rule-vs-model arbitration outcome for this turn (None when no
            # keyword rule fired). Contested turns are the ones a confirmation
            # exists to catch, so the split has to be visible in the field —
            # a rising contested rate means the rules and the model have drifted
            # apart and one of them needs attention.
            "arbitration": getattr(self.classifier, "last_arbitration", None),
            "text_len": len(text),
            "latency_ms": round(latency_ms, 2),
        }
        if self._log_utterances:
            record["text"] = text
        logger.info("nlu.decision", extra={"nlu": record})

    def reset(self, session_id: str):
        self.sessions.reset(session_id)

    def _active_confirmation(self, session):
        for intent_name, cfg in self.intents.items():
            fu = cfg.get("followup")
            if fu and session.has_context(fu["context"]):
                return (intent_name, fu)
        return None

    def _handle_confirmation(self, session, confirm, text, now=0.0):
        intent_name, fu = confirm
        polarity = self._yes_no(text)
        if polarity is None:
            # NOT yes and not no. Two very different things arrive here, and
            # re-prompting for both traps the user:
            #
            # This branch used to unconditionally re-ask AND re-set the context,
            # so the lifespan never counted down. "send a message" followed by
            # "increase volume" asked about sending a message forever, with no
            # way out. It was unreachable while no intent declared a `followup`;
            # giving Cmd.SendMessage one made it live.
            #
            # A confident, different command is the user moving on — it
            # interrupts, exactly as it does mid slot-filling, and the
            # abandoned intent is reported so the app can say so. Anything else
            # ("hmm", silence, a mumble) is a genuine non-answer and is
            # re-asked, but only within a bounded budget so a user who cannot
            # be understood is routed out instead of being held.
            new_intent, new_conf = self.classifier.classify(text)
            if (new_intent != intent_name
                    and new_intent != "Default Fallback Intent"
                    and new_conf >= self.interrupt_threshold
                    and self.intents.get(new_intent) is not None):
                session.clear_context(fu["context"])
                session.confirm_attempts = 0
                result = self._handle_new_intent(session, text, now)
                result.interrupted_intent = intent_name
                return result

            session.confirm_attempts = getattr(session, "confirm_attempts", 0) + 1
            if session.confirm_attempts >= self.max_slot_attempts:
                session.clear_context(fu["context"])
                session.confirm_attempts = 0
                return self._genai_fallback(0.0)
            session.set_context(fu["context"], fu.get("lifespan", 2), now=now)
            return NLUResult(type="CONFIRM", intent=intent_name,
                             message=fu["prompt"], confidence=1.0)
        session.clear_context(fu["context"])
        session.confirm_attempts = 0
        branch = fu["yes"] if polarity else fu["no"]
        result = NLUResult(type="FULFILL", intent=intent_name,
                           action=branch["action"], message=branch.get("fulfillment", ""),
                           confidence=1.0, complete=True)
        result._confirm_polarity = "yes" if polarity else "no"
        result._confirmed_intent = intent_name
        return result

    _UNCERTAIN = ("not sure", "maybe", "dunno", "don't know", "dont know",
                  "i don't know", "no idea", "unsure")

    # Idioms where "no" is part of an affirmative/neutral phrase, not a refusal.
    # "yes, no worries" is agreement; the bare "no" must not flip it to cancel.
    _NO_IDIOMS = ("no worries", "no problem", "no doubt", "no biggie",
                  "no probs", "no sweat", "not a problem")

    # Typographic apostrophes, folded to ASCII before the polarity scan.
    #
    # The negative list carries `don't` with a STRAIGHT apostrophe, and dictation
    # produces the curly one. `"Don't send"` matched `don't` and returned False;
    # `"Don’t send"` matched nothing negative, matched `send` (which is in the
    # affirmative list), and returned TRUE — the user declined and the message
    # went. Three of the fifty authored decline phrases inverted this way.
    #
    # Folded here rather than in the lists so every list is written one way and
    # every input is compared one way. U+02BC is the modifier letter apostrophe
    # some keyboards emit.
    _APOSTROPHES = ("\u2019", "\u02bc", "\u2018")

    def _yes_no(self, text: str):
        t = text.lower().strip()
        for ch in self._APOSTROPHES:
            t = t.replace(ch, "'")
        if any(p in t for p in self._UNCERTAIN):
            return None
        # Neutralise affirmative idioms containing "no" before polarity scan so
        # they don't register as negatives.
        scan = t
        for idiom in self._NO_IDIOMS:
            scan = scan.replace(idiom, " ")

        # ONE vocabulary, from the content. Confirmation phrasing is a LANGUAGE
        # fact, not an intent fact: "go ahead" means yes to whatever was asked.
        # A per-intent list was tried and reverted — it could not reach a device
        # (`workflows.schema.json` fixes the intent's key set with
        # `additionalProperties: false`), and it bought no safety, because this
        # function is only ever reached with a confirmation context already
        # active (`_handle_confirmation`, and `_is_cancel` mid-slot-flow). An
        # utterance arriving with no context goes to the classifier instead and
        # never consults these lists at all.
        neg = any(re.search(rf"\b{re.escape(p)}\b", scan) for p in self.negative)
        pos = any(re.search(rf"\b{re.escape(p)}\b", scan) for p in self.affirmative)
        if neg and not pos: return False
        if pos and not neg: return True
        if neg and pos:     return False
        return None

    def _is_cancel(self, text: str) -> bool:
        """True if `text` is a PURE cancellation of the active slot-filling flow.

        Two forms count: an explicit cue ("cancel", "never mind", "stop"), or a
        bare refusal ("no", "nope") with no other salient content. The purity
        guard matters — "no, tomorrow at 5" is a CORRECTION that carries a real
        value, not a cancel; it must fall through to normal extraction. So a
        refusal only cancels when the turn is essentially just the refusal
        (<= 2 tokens). Explicit cue words cancel regardless of length because
        "cancel the reminder" has no other interpretation while a flow is open.
        """
        t = text.lower().strip()
        if any(re.search(rf"\b{re.escape(c)}\b", t) for c in self._cancel_cues):
            return True
        return self._yes_no(t) is False and len(t.split()) <= 2

    def _answers_awaited_slot(self, session, cfg, text: str) -> bool:
        """True if `text` is a valid value for the slot we are waiting on.

        Only CLOSED enum entities are consulted. For an OPEN free-text entity
        (e.g. @remind) every utterance is a valid value, so there is nothing this
        test could assert; `sys.date-time` belongs to the parser for the same
        reason, and resolving it here would run the parser twice.

        Those two used to fall through to the confidence bar alone, which is the
        defect VIK-038 records: the classifier is asked about input it was never
        trained on, answers at 0.98+, and cancels the flow. They are now excluded
        from the probe entirely by `_slot_can_refuse_the_answer`, so this test is
        only ever reached for a slot whose answer can actually be checked.

        MATCHING IS STRICT — `fuzzy=False` and a high match floor. Extraction for
        STORAGE can afford to be lenient, because the user is already answering
        the question. This decision is the opposite: it SUPPRESSES a topic switch,
        so a loose match traps the user in the flow. With fuzzy matching on, the
        `memory` entity resolves "turn up the volume" to the memory "three" (via
        "the", 0.6 confidence), which silently blocked a genuine switch. Real
        answers score 0.95-1.0 and survive the strict path.
        """
        awaiting = session.awaiting_slot
        if not awaiting:
            return False
        slot = self._slot_def(cfg, awaiting)
        if not slot:
            return False
        entity = slot.get("entity", "")
        if self.entities.is_open(entity):
            return False
        value, _, conf = self.entities.extract(entity, text, fuzzy=False)
        return value is not None and conf >= self.SLOT_ANSWER_MATCH_FLOOR

    def _slot_can_refuse_the_answer(self, session, cfg) -> bool:
        """True if the awaited slot can supply evidence that a turn is NOT its answer.

        This is the gate on the topic-switch probe (VIK-038); the reasoning lives
        at the call site in `_handle_slot_filling`. In short: a closed enum can be
        missed, so a miss is a fact worth acting on. An open free-text slot accepts
        anything and a date-time slot belongs to the parser, so for those the
        classifier has no question it can answer and must not be asked.

        With no slot awaited the pre-existing behaviour stands: probe.
        """
        awaiting = session.awaiting_slot
        if not awaiting:
            return True
        slot = self._slot_def(cfg, awaiting)
        if not slot:
            return True
        entity = slot.get("entity", "")
        return not (self.entities.is_open(entity) or self.entities.is_date_time(entity))

    def _handle_slot_filling(self, session, text, now=0.0):
        intent_name = session.pending_intent
        cfg = self.intents[intent_name]

        # An answer to the question we just asked is NOT a topic switch, however
        # confidently it classifies as something else. Several memory names are
        # also commands — "mute", "quiet", "telephone" — so asking "What is the
        # name of the memory?" and hearing "mute" used to MUTE the device at 0.98
        # instead of switching to the Mute memory. No threshold can fix that:
        # "tinnitus" and "mask" classify at 1.000. Answering the live question
        # has to take precedence over re-classification.
        #
        # Also consulted by the cancellation guard below, so it is computed on
        # every turn, not only the ones that reach the probe.
        answers_prompt = self._answers_awaited_slot(session, cfg, text)

        # VIK-038. Topic-switch probe — but ONLY when the awaited slot can produce
        # evidence that this utterance is not an answer to it.
        #
        # The classifier is trained on COMMANDS. A slot answer is out-of-distribution
        # input for it, and a confidence score on OOD input is not a quantity that
        # can be thresholded. Measured on this language pack's own weights, answering
        # the reminder's "What do you want to be reminded?" with
        #
        #     "walk the dog"           -> Cmd.ActivityWalk      1.000
        #     "clean my hearing aids"  -> Help_CleanCare        1.000
        #     "start my workout"       -> Cmd.ActivityExercise  0.995
        #     "Need to go to walk"     -> Cmd.ActivityWalk      0.994
        #     "charge my hearing aids" -> Help_Battery          0.979
        #
        # cancelled the reminder the user was in the middle of setting and did the
        # other thing instead. Raising `interrupt_threshold` does not help: "start
        # my workout" (a legitimate reminder) outscores "start transcribing" (a real
        # command, 0.962). The two are structurally identical and differ only in
        # whether the object is a device capability or a thing in the world.
        #
        # The flow survived at all only by accident — "call mom", "buy milk" and
        # "drink water" produce no vocabulary features, so they collapse to the
        # vacuous floor (0.580) and land under the bar. That is luck, not a guard.
        #
        # What DOES carry evidence is the slot itself, so the gate is the awaited
        # entity's KIND — never the intent's name, which this engine does not
        # interpret:
        #
        #   closed enum      the value is in the list or it is not. A miss is a
        #                    fact, and the only honest reason to ask the classifier
        #                    where the user went instead. PROBE.
        #   open free-text   every utterance is a legal value. There is nothing to
        #                    be right about. DO NOT PROBE.
        #   date-time        the parser decides, not the classifier. DO NOT PROBE.
        #
        # For `en` that means the reminder flow (`remind` open + `sys.date-time`)
        # no longer interrupts, and the memory flow (`memory`, a closed enum) still
        # does — "increase volume" is not a memory, so it still switches correctly.
        #
        # A user is not trapped by this: `_is_cancel` below still abandons the flow
        # on an explicit cue, and MAX_SLOT_ATTEMPTS still applies to a slot that
        # cannot be filled.
        #
        # PARITY: mirrors VoiceAIKit `NLUEngine.handleSlotFilling`
        # (`slotCanRefuseTheAnswer`) condition for condition. The defect was fixed
        # there first and left open here as a question; the two must not diverge.
        if self._slot_can_refuse_the_answer(session, cfg):
            new_intent, new_conf = self.classifier.classify(text)
            # A bare `contains` keyword hit is the weakest signal — an incidental
            # mention ("ask about the translate feature") must NOT abandon an
            # in-progress flow. Only stronger signals (TF-IDF, exact/regex keyword)
            # may interrupt.
            weak_keyword = (self.classifier.last_stage == "keyword"
                            and self.classifier.last_keyword_tier == "contains")
            if (new_intent != intent_name
                    and new_intent != "Default Fallback Intent"
                    and new_conf >= self.interrupt_threshold
                    and not weak_keyword
                    and not answers_prompt
                    and self.intents.get(new_intent) is not None):
                abandoned = intent_name
                session.reset_slot_filling()
                result = self._handle_new_intent(session, text, now)
                result.interrupted_intent = abandoned
                return result

        # Meta-command: a pure cancellation/refusal ABANDONS the flow instead of
        # being mined for a slot value. Without this an open free-text slot
        # stores "no" as the reminder name, and a typed slot drags the user
        # through MAX_SLOT_ATTEMPTS dead reprompts. Guarded by `not answers_prompt`
        # so a refusal word that is itself a valid enum value stays an answer,
        # and by _is_cancel's purity check so "no, tomorrow at 5" (a correction
        # carrying a value) falls through to extraction below.
        if self._is_cancel(text) and not answers_prompt:
            session.reset_slot_filling()
            return NLUResult(type="FULFILL", intent="sys.slot.cancelled",
                             action=None, message=self._confirm_cancel_msg,
                             confidence=1.0, complete=True)

        awaiting = session.awaiting_slot
        if awaiting:
            slot = self._slot_def(cfg, awaiting)
            if slot["entity"] == "sys.date-time":
                iso, filled = self._resolve_datetime(session, text)
                if filled:
                    session.pending_slots[slot["name"]] = iso
            else:
                if self.entities.is_open(slot["entity"]):
                    # For open free-text entities (e.g. @remind) always store what the
                    # user actually said rather than a canonical English match from the
                    # synonym table. The synonym table is used for recognition only, not
                    # for storage — otherwise "prendre des médicaments" becomes
                    # "Take Medication" in a French session.
                    value = text.strip() or None
                else:
                    value, _, _conf = self.entities.extract(slot["entity"], text)
                if value is not None:
                    session.pending_slots[slot["name"]] = value
        # Opportunistically fill OTHER slots mentioned in the same answer, but
        # skip the slot we just handled — re-resolving it (e.g. a parked
        # date-time anchored to itself) would double-advance the day.
        self._extract_all_slots(session, cfg, text, session.pending_slots, skip=awaiting)

        # Slot-attempt accounting: if we were waiting on a specific slot and it
        # is still unfilled after this turn, count a failed attempt. Abandon the
        # flow gracefully once the budget is exhausted so the user is never
        # trapped re-answering an un-parseable prompt.
        if awaiting and awaiting not in session.pending_slots:
            session.slot_attempts += 1
            if session.slot_attempts >= self.max_slot_attempts:
                session.reset_slot_filling()
                return NLUResult(
                    type="FALLBACK", intent="GENAI", action="genai.fallback",
                    confidence=0.0,
                    message="Sorry, I'm having trouble with that. Let's try something else.")
        elif awaiting:
            session.slot_attempts = 0  # progress made on the awaited slot

        return self._advance_slots(session, intent_name, cfg)

    def _advance_slots(self, session, intent_name, cfg):
        """Prompt for the next missing required slot, or fulfil when none remain.

        This used to take an `entry_conf` so a flow completing on its very first
        turn could still be held by the uncertainty gate. That gate is gone (see
        __init__), so the parameter went with it — a slot-bearing intent is
        governed by the same single fire threshold as everything else, applied
        once in `_handle_new_intent` before the flow ever starts.
        """
        for slot in cfg["slots"]:
            if slot["required"] and slot["name"] not in session.pending_slots:
                session.pending_intent = intent_name
                session.awaiting_slot = slot["name"]
                return NLUResult(type="PROMPT", intent=intent_name,
                                 parameters=dict(session.pending_slots),
                                 message=slot["prompt"], confidence=1.0)
        params = dict(session.pending_slots)
        session.reset_slot_filling()
        session.record_fulfillment(intent_name, params)
        return NLUResult(type="FULFILL", intent=intent_name, action=cfg.get("action"),
                         parameters=params, message=cfg.get("fulfillment", ""),
                         confidence=1.0, complete=True)

    def _try_back_reference(self, session, text: str) -> Optional["NLUResult"]:
        """Pre-pass: resolve back/again phrases using declarative schema back_reference entries."""
        t = text.lower()
        for intent_name, cfg in self.intents.items():
            br = cfg.get("back_reference")
            if not br:
                continue
            if not re.search(br["pattern"], t, re.I):
                continue
            source = br.get("source")
            if source == "prev_memory":
                if session.prev_memory:
                    params = {br["slot"]: session.prev_memory}
                    session.record_fulfillment(intent_name, params)
                    return NLUResult(type="FULFILL", intent=intent_name,
                                     action=cfg.get("action"), parameters=params,
                                     message=cfg.get("fulfillment", ""),
                                     confidence=1.0, complete=True)
                # No prev value — fall through to slot-filling
                session.pending_intent = intent_name
                session.pending_slots = {}
                session.awaiting_slot = None
                return self._advance_slots(session, intent_name, cfg)
            if source == "last_fulfilled":
                last = session.get_last_params(intent_name)
                if last:
                    session.record_fulfillment(intent_name, last)
                    return NLUResult(type="FULFILL", intent=intent_name,
                                     action=cfg.get("action"), parameters=last,
                                     message=cfg.get("fulfillment", ""),
                                     confidence=1.0, complete=True)
        return None

    def _resolve_back_reference(self, session, intent: str, text: str) -> Optional[dict]:
        """Return pre-filled slots if the utterance matches the intent's back_reference pattern."""
        cfg = self.intents.get(intent, {})
        br = cfg.get("back_reference")
        if not br:
            return None
        if not re.search(br["pattern"], text.lower(), re.I):
            return None
        source = br.get("source")
        if source == "prev_memory" and session.prev_memory:
            return {br["slot"]: session.prev_memory}
        if source == "last_fulfilled":
            last = session.get_last_params(intent)
            if last:
                return last
        return None

    def _apply_polarity_guards(self, text: str, intent: str) -> str:
        """ND-11(b): redirect a prediction contradicted by explicit polarity
        words. Fires only when EXACTLY ONE guard matches the predicted intent
        AND the utterance does NOT also carry the opposite cue.

        The guard must never override a prediction the utterance itself supports.
        A phrase like "lower how LOUD it is" contains both a decrease cue
        ("lower") and an increase cue ("loud"); the model already resolves this
        correctly, so a guard that fired on "loud" alone would FLIP the correct
        answer to the wrong one. Whenever a mirror rule (redirect -> intent)
        also matches the text, the polarity signal is contradictory and the
        guard abstains, trusting the model."""
        if not self._polarity_guards:
            return intent
        low = text.lower()
        hits = []
        for rx, blocked, redirect in self._polarity_guards:
            if blocked != intent or not rx.search(low):
                continue
            # Contradiction check: does a mirror rule redirect BACK to `intent`,
            # i.e. is the opposite polarity cue also present? If so, abstain.
            opposite_present = any(
                b2 == redirect and r2 == intent and rx2.search(low)
                for rx2, b2, r2 in self._polarity_guards
            )
            if not opposite_present:
                hits.append(redirect)
        if len(hits) == 1 and hits[0] in self.intents:
            logger.info("nlu.polarity_guard", extra={"nlu": {
                "blocked": intent, "redirected": hits[0]}})
            return hits[0]
        return intent

    def _apply_help_guard(self, text: str, intent: str) -> str:
        """ND-14: if the model predicts a state-changing ACTION but the utterance
        carries explicit help/question markers ("how do I…", "guide", "comment…"),
        redirect to the action's read-only ``help.*`` sibling. Safety rule: asking
        HOW to use a feature must never TRIGGER it. Fires only for actions that
        have a paired help intent (queries are excluded on purpose) and only when
        the paired help intent actually exists in this bundle."""
        if not self._help_markers or intent not in self._help_pairs:
            return intent
        if not self._help_markers.search(text.lower()):
            return intent
        sibling = self._help_pairs[intent]
        if sibling not in self.intents:
            return intent
        logger.info("nlu.help_guard", extra={"nlu": {
            "blocked": intent, "redirected": sibling}})
        return sibling

    def _handle_new_intent(self, session, text, now=0.0):
        session.decrement_contexts()
        intent, conf = self.classifier.classify(text)
        guarded = self._apply_help_guard(text, self._apply_polarity_guards(text, intent))
        if guarded != intent:
            # A guard changed WHICH intent we report, so the confidence must be
            # re-read for the intent actually being reported. Inheriting the
            # blocked prediction's number describes something we are no longer
            # returning, and it is compared against the fire threshold moments
            # later.
            #
            # "how do i turn up the loudness on my hearing aids?" is the case:
            # the `turn up` regex proposes Cmd.VolumeIncrease, the model
            # says Help_Volume at 0.85, the help guard correctly redirects
            # to Help_Volume — and the turn then carried the *contested*
            # confidence of the blocked action, dropping it under the fire
            # threshold and deflecting a perfectly good help request to GenAI.
            # Latent before arbitration too: it inherited the `regex` literal
            # 0.75, which happened to clear the 0.70 bar, so nothing failed.
            reguarded_conf = self.classifier.calibrated_confidence(guarded)
            if reguarded_conf is not None:
                conf = reguarded_conf
            intent = guarded

        cfg = self.intents.get(intent)

        # ONE threshold, for every intent. Slot-bearing intents used to get a
        # lower bar (`slot_confidence_threshold`, 0.50) on the reasoning that a
        # prompt would resolve any ambiguity before anything happened.
        #
        # That reasoning only holds when the flow actually prompts. A
        # slot-bearing intent whose slots are ALL filled by the classifying
        # utterance completes immediately, and then the lower bar is just a
        # lower bar on a live action: "can you to us number one hits" classified
        # as Cmd.MemoryChange at 0.519, "one" filled the memory slot, and
        # the hearing-aid program changed — reported as confidence 1.0, which is
        # the slot-fill certainty rather than the intent's.
        #
        # That hole was previously plugged by handing `entry_conf` down to
        # `_advance_slots` so the completing turn still met the confirmation
        # gate. With the gate gone, the plug went with it, and the honest fix is
        # the one threshold rather than a second special case.
        #
        # The single exception is CORROBORATION: when a keyword rule and the
        # model independently name the same intent, two recognisers agree and
        # the bar drops to AGREEMENT_THRESHOLD — the same relaxation this engine
        # already applies to TF-IDF/MiniLM agreement below. "turn it up its too
        # quiet" is the case: rule and model both say volume.increase, but
        # "quiet" splits the mass with volume.decrease and leaves the top class
        # at 0.66. Rejecting a command both recognisers agree on, because a
        # sibling class took a third of the probability, is a worse error than
        # the one the threshold is there to prevent.
        corroborated = getattr(self.classifier, "last_arbitration", None) == "corroborated"
        fire_bar = self.agreement_threshold if corroborated else self.threshold

        # OUT-OF-VOCABULARY GUARD. A confident reading of the words the
        # featurizer CAN see says nothing about the words it cannot.
        #
        # "help me find a paper" -> the model receives `help me find`, because
        # "paper" appears nowhere in the training corpus and so has no slot in
        # the vocabulary at all. On that input `Help_FindMyHearingAids`
        # is the correct answer; the utterance that was actually spoken never
        # reached the model. Same shape as "turn off toshiba" reducing to
        # "turn off", whose vector is bit-identical to the bare command.
        #
        # So the unknown words are consulted directly rather than thrown away:
        # above `oov_reject_ratio` of the tokens being unrepresentable, the turn
        # goes to the fallback intent regardless of how confident the remainder
        # looks. This cannot be done by tuning a threshold — the confidence is
        # honest about the input the model was given.
        #
        # An unknown word is only evidence when the REST of the utterance is
        # also ambiguous. Entity values are unknown BY NATURE — a contact name,
        # a brand, a free-text reminder topic can never all be in a finite
        # vocabulary — so a bare ratio test refuses:
        #
        #   'send a message to john'   oov 0.25, conf 1.000   <- a real command
        #   'stream from netflix'      oov 0.33, conf 0.996   <- a real command
        #   'help me find a paper'     oov 0.25, conf 0.771   <- out of scope
        #
        # The ratio cannot tell a slot value from a foreign topic, but the
        # confidence can: when the remainder reads unambiguously the unknown
        # token is almost certainly the VALUE the command operates on; when the
        # remainder is merely plausible, it is the thing that puts the utterance
        # out of scope. So the guard stands down above
        # `oov_bypass_confidence`. Measured on the honest holdout, adding that
        # condition keeps the same out-of-scope reduction (10 -> 5) and returns
        # 7 correct commands the bare ratio was refusing.
        #
        # Applied AFTER the guards so it sees the intent actually being
        # returned, and before the fire test so it can only ever withhold an
        # action, never cause one.
        if self._oov_reject_ratio is not None and intent != "Default Fallback Intent" \
                and conf < self._oov_bypass_confidence:
            ratio = self.classifier.oov_ratio(text)
            if ratio >= self._oov_reject_ratio:
                logger.info("nlu.oov_guard", extra={"nlu": {
                    "blocked": intent, "oov_ratio": round(ratio, 3),
                    "confidence": round(conf, 3)}})
                return self._genai_fallback(conf)

        if intent == "Default Fallback Intent" or conf < fire_bar:
            # Stage 3: semantic rescue via MiniLM when TF-IDF is uncertain
            if self.semantic is not None:
                sem_intent, sem_conf = self.semantic.classify(text)
                if sem_intent != "Default Fallback Intent":
                    # Two ways to accept a rescue:
                    #   1. Standard: the head clears the absolute softmax floor.
                    #   2. Agreement gate: TF-IDF and the head INDEPENDENTLY land
                    #      on the SAME real intent. Over a 61-class head, a correct
                    #      prediction's probability is diffused across sibling
                    #      classes (e.g. VolumeDecrease/Mute/Increase) and can sit
                    #      just under the floor; corroboration by TF-IDF is stronger
                    #      evidence than either signal alone, so we relax the bar.
                    #      This cannot re-admit out-of-scope queries: a genuine OOS
                    #      utterance does not make two independent models agree on a
                    #      real command.
                    accept_threshold = self.semantic_threshold
                    if (sem_intent == intent
                            and intent != "Default Fallback Intent"):
                        accept_threshold = self.agreement_threshold
                    if sem_conf >= accept_threshold:
                        sem_intent = self._apply_polarity_guards(text, sem_intent)
                        sem_intent = self._apply_help_guard(text, sem_intent)
                        sem_cfg = self.intents.get(sem_intent)
                        if sem_cfg is not None:
                            result = self._fulfill_intent(session, sem_intent, sem_conf, sem_cfg, text, now)
                            result.semantic_rescue    = True
                            result.tfidf_intent       = intent
                            result.tfidf_confidence   = conf
                            return result
            # Below the fire threshold and not rescued: the fallback intent.
            # There is no confirmation tier under the threshold — see __init__.
            return self._genai_fallback(conf)
        if cfg is None:
            return self._genai_fallback(conf)
        return self._fulfill_intent(session, intent, conf, cfg, text, now)

    @staticmethod
    def _genai_fallback(conf: float) -> "NLUResult":
        # No URL, no text: the result carries only the routing decision. The
        # app holds the utterance and builds the GenAI request itself.
        return NLUResult(type="FALLBACK", intent="GENAI",
                         action="genai.fallback", confidence=conf)

    def _fulfill_intent(self, session, intent, conf, cfg, text, now=0.0):
        if cfg.get("followup"):
            fu = cfg["followup"]
            session.set_context(fu["context"], fu.get("lifespan", 2), now=now)
            return NLUResult(type="CONFIRM", intent=intent, action=cfg.get("action"),
                             message=fu["prompt"], confidence=conf)
        if cfg["slots"]:
            # Check for back-reference ("change back", "remind me again")
            # before doing normal entity extraction.
            slots = self._resolve_back_reference(session, intent, text) or {}
            if not slots:
                self._extract_all_slots(session, cfg, text, slots)
                self._fill_open_topics(cfg, text, slots)
            session.pending_intent = intent
            session.pending_slots = slots
            session.awaiting_slot = None
            return self._advance_slots(session, intent, cfg)
        blocked = self._availability_block(intent)
        if blocked is not None:
            return blocked
        result = NLUResult(type="FULFILL", intent=intent, action=cfg.get("action"),
                           message=cfg.get("fulfillment", ""), confidence=conf, complete=True)
        session.record_fulfillment(intent, {})
        return result

    def _extract_all_slots(self, session, cfg, text, slots: dict, skip: str = None):
        # One-shot / bulk full-sentence scan: disable fuzzy enum matching so a
        # common word (e.g. "care", "cup") doesn't get mis-read as a memory
        # name. Fuzzy is reserved for the awaited-slot answer in slot filling.
        for slot in cfg["slots"]:
            if slot["name"] in slots or slot["name"] == skip:
                continue
            if slot["entity"] == "sys.date-time":
                # Only fill when a time was actually given; a day-only mention
                # parks the day in session.partial_datetime and leaves the slot
                # open so the engine prompts for the time.
                iso, filled = self._resolve_datetime(session, text)
                if filled:
                    slots[slot["name"]] = iso
                continue
            # Open free-text entities (e.g. @remind) are handled by _fill_open_topics
            # via _derive_topic, which strips carrier phrases and datetime cleanly.
            # Attempting enum extraction here returns the English canonical name even
            # for foreign-language input (e.g. "prendre des médicaments" → "Take
            # Medication"), which is wrong for user-visible reminder names.
            if self.entities.is_open(slot["entity"]):
                continue
            value, _, _conf = self.entities.extract(slot["entity"], text, fuzzy=False)
            if value is not None:
                slots[slot["name"]] = value

    def _resolve_datetime(self, session, text: str):
        """Resolve a date-time slot value from `text`.

        Returns (iso, filled). `filled` is True only when an explicit time was
        given. When the user supplies a day but no time, the resolved day is
        stored in session.partial_datetime and (None, False) is returned so the
        engine prompts for the time; a later bare-time answer ("3pm") is anchored
        to that parked day so "tomorrow" is not lost.
        """
        # Probe with the real clock first. This tells us whether the answer
        # carries its OWN day ("tomorrow at 4") — in which case it wins and we
        # must NOT anchor, or the anchored day would advance ("tomorrow" relative
        # to the parked tomorrow → day after).
        iso, _span, _conf, time_explicit, explicit_day = self.entities.extract_datetime(text)
        if iso is None:
            return None, False

        # Bare time with no day of its own ("at 4") — anchor it to the parked day
        # so "tomorrow" is preserved, then re-resolve against that day's midnight.
        if not explicit_day and session.partial_datetime:
            try:
                anchor = datetime.fromisoformat(session.partial_datetime).astimezone()
                iso, _span, _conf, time_explicit, explicit_day = \
                    self.entities.extract_datetime(text, now=anchor)
            except ValueError:
                pass

        if time_explicit:
            session.partial_datetime = None
            return iso, True

        # Day given, no time — park the day at local midnight (not the 9am
        # default) so a later time answer like "6am" stays on this day instead of
        # tripping the "already past today → roll forward" guard.
        day_start = (datetime.fromisoformat(iso).astimezone()
                     .replace(hour=0, minute=0, second=0, microsecond=0))
        session.partial_datetime = day_start.isoformat()
        return None, False

    def _fill_open_topics(self, cfg, text, slots: dict):
        for slot in cfg["slots"]:
            if slot["name"] in slots or not slot.get("required"):
                continue
            if not self.entities.is_open(slot["entity"]):
                continue
            topic = self._derive_topic(text)
            if topic:
                slots[slot["name"]] = topic

    def _derive_topic(self, text: str):
        """Strip the request wrapper off an utterance so the open slot gets the payload.

        ORDER MATTERS, and the date/time goes FIRST. Every carrier is `^`-anchored,
        so a carrier is only reachable when it is at the front of the string — and a
        leading time expression pushes it out of reach:

            "tomorrow morning remind me to water the plants"
                carriers first -> "^remind me" misses -> "remind me to water the plants"
                date/time first -> "remind me to water the plants" -> "water the plants"

        This is safe in the other direction because `strip_datetime` only ever REMOVES
        text; it never prepends. So running it first can only move an `^`-anchored
        carrier closer to the front, never further from it — the change is strictly
        additive, and an utterance whose time is in the middle or at the end (the
        common shape: "remind me to drink water at 5") is unaffected. Verified over a
        41-case battery: only leading-time utterances change, all of them to the right
        answer.

        The one way that invariant could break is a language whose carrier CONTAINS a
        word `strip_datetime` removes. No English carrier does. A new language pack
        must be checked for it — that is what the parity fixtures are for.
        """
        t = self.entities.strip_datetime(text.strip())
        for pat in self._carrier:
            t = re.sub(pat, "", t, count=1, flags=re.I)
        t = self._leading_connector.sub("", t).strip(" .,")
        return t or None

    @staticmethod
    def _slot_def(cfg, name):
        return next(s for s in cfg["slots"] if s["name"] == name)
