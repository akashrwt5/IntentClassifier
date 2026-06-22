"""
NLUEngine — the orchestrator that replaces Dialogflow end-to-end.

Per user turn priority order:
  1. CONFIRMATION — active yes/no follow-up context
  2. SLOT FILLING — mid-collection intent (with interruption detection)
  3. CLASSIFY     — fresh turn

Intent interruption: if the user switches topic mid slot-filling with
high confidence (>= INTERRUPT_THRESHOLD), the pending flow is abandoned
and the new intent is handled immediately. The NLUResult carries
interrupted_intent so the app can optionally notify the user.
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

BASE_DIR = Path(__file__).parent.parent.parent
SCHEMA_PATH = BASE_DIR / "data" / "nlu_schema.json"
LABELS_JSON_PATH = BASE_DIR / "models" / "intent_labels.json"

# Fallback default for the semantic-rescue threshold when the schema omits it.
# The schema's "semantic_threshold" is the single source of truth; this is only
# used if that key is absent.
DEFAULT_SEMANTIC_THRESHOLD = 0.55

# Default GenAI endpoint base. Overridable via the NLU_GENAI_URL env var or a
# "genai_url" key in the schema so the placeholder is never shipped silently.
DEFAULT_GENAI_URL = "https://genai.yourcompany.com/chat?query="

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
    # Interruption requires stronger signal than the base 0.70 threshold to avoid
    # abandoning a slot flow on an ambiguous utterance like "take medication".
    # Lowered from 0.85 → 0.75 after isotonic calibration: calibrated probabilities
    # are more moderate so 0.85 was unreachable for genuine switch-intent signals.
    INTERRUPT_THRESHOLD = 0.75

    # A user who cannot complete a slot after this many tries is routed out of
    # the flow instead of being trapped in an unanswerable prompt loop.
    MAX_SLOT_ATTEMPTS = 3

    # Relaxed semantic-rescue floor used ONLY when the TF-IDF stage and the
    # MiniLM head independently agree on the same real intent (see the agreement
    # gate in _handle_new_intent). Lower than semantic_threshold because mutual
    # corroboration offsets the softmax diffusion of a 61-class head; still high
    # enough that a flat (genuinely-ambiguous) distribution is rejected.
    AGREEMENT_THRESHOLD = 0.50

    def __init__(self, schema_path: Path = SCHEMA_PATH):
        self.schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        self.intents = self.schema["intents"]
        self.threshold = self.schema.get("confidence_threshold", 0.70)
        self.affirmative = set(self.schema.get("affirmative", []))
        self.negative = set(self.schema.get("negative", []))
        # GenAI base URL is configuration, not a result field. The raw user
        # utterance is NEVER embedded into an NLUResult (it would otherwise be
        # captured by any caller that logs the result). The app layer, which
        # already holds the text, constructs the navigation URL itself.
        self.genai_url = (self.schema.get("genai_url")
                          or os.environ.get("NLU_GENAI_URL")
                          or DEFAULT_GENAI_URL)
        # Opt-in raw-utterance logging. Off by default so medical-context
        # speech is never written to logs in production; enable in dev only.
        self._log_utterances = os.environ.get("NLU_LOG_UTTERANCES", "").lower() in ("1", "true", "yes")
        # Single source of truth for the semantic-rescue gate: the schema. The
        # same value constructs SemanticFallback AND gates its result in the
        # engine, so the two can never drift.
        self.semantic_threshold = self.schema.get("semantic_threshold", DEFAULT_SEMANTIC_THRESHOLD)
        self.classifier = IntentClassifier()
        self.entities = EntityExtractor()
        self.sessions = SessionStore()
        self.semantic = self._load_semantic(self.semantic_threshold)
        self._assert_label_schema_parity()

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
        labels_path = LABELS_JSON_PATH
        if not labels_path.exists():
            return  # model not yet trained; skip during development
        labels = set(json.loads(labels_path.read_text(encoding="utf-8")))
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

    def handle(self, session_id: str, text: str) -> NLUResult:
        t0 = time.perf_counter()
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
        return result

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
            "stage": stage,
            "type": result.type,
            "intent": result.intent,
            "confidence": round(result.confidence, 4),
            "semantic_rescue": result.semantic_rescue,
            "interrupted_intent": result.interrupted_intent,
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
            session.set_context(fu["context"], fu.get("lifespan", 2), now=now)
            return NLUResult(type="CONFIRM", intent=intent_name,
                             message=fu["prompt"], confidence=1.0)
        session.clear_context(fu["context"])
        branch = fu["yes"] if polarity else fu["no"]
        return NLUResult(type="FULFILL", intent=intent_name,
                         action=branch["action"], message=branch.get("fulfillment", ""),
                         confidence=1.0, complete=True)

    _UNCERTAIN = ("not sure", "maybe", "dunno", "don't know", "dont know",
                  "i don't know", "no idea", "unsure")

    # Idioms where "no" is part of an affirmative/neutral phrase, not a refusal.
    # "yes, no worries" is agreement; the bare "no" must not flip it to cancel.
    _NO_IDIOMS = ("no worries", "no problem", "no doubt", "no biggie",
                  "no probs", "no sweat", "not a problem")

    def _yes_no(self, text: str):
        t = text.lower().strip()
        if any(p in t for p in self._UNCERTAIN):
            return None
        # Neutralise affirmative idioms containing "no" before polarity scan so
        # they don't register as negatives.
        scan = t
        for idiom in self._NO_IDIOMS:
            scan = scan.replace(idiom, " ")
        neg = any(re.search(rf"\b{re.escape(p)}\b", scan) for p in self.negative)
        pos = any(re.search(rf"\b{re.escape(p)}\b", scan) for p in self.affirmative)
        if neg and not pos: return False
        if pos and not neg: return True
        if neg and pos:     return False
        return None

    def _handle_slot_filling(self, session, text, now=0.0):
        intent_name = session.pending_intent
        cfg = self.intents[intent_name]

        # Check for intent interruption: re-classify every slot-filling turn.
        # A different intent at high confidence means the user switched topics.
        new_intent, new_conf = self.classifier.classify(text)
        # A bare `contains` keyword hit is the weakest signal — an incidental
        # mention ("ask about the translate feature") must NOT abandon an
        # in-progress flow. Only stronger signals (TF-IDF, exact/regex keyword)
        # may interrupt.
        weak_keyword = (self.classifier.last_stage == "keyword"
                        and self.classifier.last_keyword_tier == "contains")
        if (new_intent != intent_name
                and new_intent != "Default Fallback Intent"
                and new_conf >= self.INTERRUPT_THRESHOLD
                and not weak_keyword
                and self.intents.get(new_intent) is not None):
            abandoned = intent_name
            session.reset_slot_filling()
            result = self._handle_new_intent(session, text, now)
            result.interrupted_intent = abandoned
            return result

        awaiting = session.awaiting_slot
        if awaiting:
            slot = self._slot_def(cfg, awaiting)
            if slot["entity"] == "sys.date-time":
                iso, filled = self._resolve_datetime(session, text)
                if filled:
                    session.pending_slots[slot["name"]] = iso
            else:
                value, _, _conf = self.entities.extract(slot["entity"], text)
                # Open free-text entities (e.g. @remind) accept the raw answer as
                # a fallback — a None structured extraction is expected, not a failure.
                if value is None and self.entities.is_open(slot["entity"]):
                    value = text.strip()
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
            if session.slot_attempts >= self.MAX_SLOT_ATTEMPTS:
                session.reset_slot_filling()
                return NLUResult(
                    type="FALLBACK", intent="GENAI", action="genai.fallback",
                    confidence=0.0,
                    message="Sorry, I'm having trouble with that. Let's try something else.")
        elif awaiting:
            session.slot_attempts = 0  # progress made on the awaited slot

        return self._advance_slots(session, intent_name, cfg)

    def _advance_slots(self, session, intent_name, cfg):
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

    def _handle_new_intent(self, session, text, now=0.0):
        session.decrement_contexts()
        intent, conf = self.classifier.classify(text)

        cfg = self.intents.get(intent)

        # Slot-filling intents use a lower threshold: a prompt will resolve
        # any ambiguity, while a fire-and-forget intent executing at low
        # confidence causes a silent wrong action.
        has_slots = bool(cfg and cfg.get("slots"))
        effective_threshold = self.schema.get(
            "slot_confidence_threshold", 0.60
        ) if has_slots else self.threshold

        if intent == "Default Fallback Intent" or conf < effective_threshold:
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
                        accept_threshold = self.AGREEMENT_THRESHOLD
                    if sem_conf >= accept_threshold:
                        sem_cfg = self.intents.get(sem_intent)
                        if sem_cfg is not None:
                            result = self._fulfill_intent(session, sem_intent, sem_conf, sem_cfg, text, now)
                            result.semantic_rescue    = True
                            result.tfidf_intent       = intent
                            result.tfidf_confidence   = conf
                            return result
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

    _CARRIER = [
        r"^\s*please\s+",
        r"^\s*(?:do\s*n[o']?t|don't|dont)\s+let\s+me\s+forget\b\s*(?:to|about)?\s*",
        r"^\s*(?:remind|tell|alert|notify)\s+me\b\s*(?:to|that|about|of)?\s*",
        r"^\s*set(?:\s+up)?\s+(?:a\s+)?reminder\b\s*(?:to|for|about)?\s*",
        r"^\s*make\s+sure\s+(?:i|to)\b\s*",
        r"^\s*i\s+(?:need|have|want)\s+to\b\s*",
    ]

    def _fill_open_topics(self, cfg, text, slots: dict):
        for slot in cfg["slots"]:
            if slot["name"] in slots or not slot.get("required"):
                continue
            if not self.entities.is_open(slot["entity"]):
                continue
            topic = self._derive_topic(text)
            if topic:
                slots[slot["name"]] = topic

    # Connectors that dangle at the START of a derived topic once the carrier
    # phrase and the time expression have been stripped, e.g. "remind me at 9pm
    # for dinner" → "for dinner" → "dinner".
    _LEADING_CONNECTOR = re.compile(
        r"^(?:for|about|of|on|to|that|regarding|with)\s+", re.I)

    def _derive_topic(self, text: str):
        t = text.strip()
        for pat in self._CARRIER:
            t = re.sub(pat, "", t, count=1, flags=re.I)
        t = self.entities.strip_datetime(t)
        t = self._LEADING_CONNECTOR.sub("", t).strip(" .,")
        return t or None

    @staticmethod
    def _slot_def(cfg, name):
        return next(s for s in cfg["slots"] if s["name"] == name)
