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
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from .classifier import IntentClassifier
from .entities import EntityExtractor
from .context import SessionStore

BASE_DIR = Path(__file__).parent.parent.parent
SCHEMA_PATH = BASE_DIR / "data" / "nlu_schema.json"


@dataclass
class NLUResult:
    type: str
    intent: Optional[str] = None
    action: Optional[str] = None
    parameters: dict = field(default_factory=dict)
    message: str = ""
    confidence: float = 0.0
    complete: bool = False
    url: Optional[str] = None
    interrupted_intent: Optional[str] = None  # set when a slot-filling flow was abandoned

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v not in (None, {}, "")}


class NLUEngine:
    # Interruption requires stronger signal than normal classification to avoid
    # abandoning a slot flow on an ambiguous utterance like "take medication"
    # which could be a slot answer or a new reminder intent.
    INTERRUPT_THRESHOLD = 0.85

    def __init__(self, schema_path: Path = SCHEMA_PATH):
        self.schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        self.intents = self.schema["intents"]
        self.threshold = self.schema.get("confidence_threshold", 0.70)
        self.affirmative = set(self.schema.get("affirmative", []))
        self.negative = set(self.schema.get("negative", []))
        self.genai_url = "https://genai.yourcompany.com/chat?query="
        self.classifier = IntentClassifier()
        self.entities = EntityExtractor()
        self.sessions = SessionStore()

    def handle(self, session_id: str, text: str) -> NLUResult:
        session = self.sessions.get(session_id)
        text = text.strip()
        confirm = self._active_confirmation(session)
        if confirm:
            return self._handle_confirmation(session, confirm, text)
        if session.pending_intent:
            return self._handle_slot_filling(session, text)
        return self._handle_new_intent(session, text)

    def reset(self, session_id: str):
        self.sessions.reset(session_id)

    def _active_confirmation(self, session):
        for intent_name, cfg in self.intents.items():
            fu = cfg.get("followup")
            if fu and session.has_context(fu["context"]):
                return (intent_name, fu)
        return None

    def _handle_confirmation(self, session, confirm, text):
        intent_name, fu = confirm
        polarity = self._yes_no(text)
        if polarity is None:
            session.set_context(fu["context"], fu.get("lifespan", 2))
            return NLUResult(type="CONFIRM", intent=intent_name,
                             message=fu["prompt"], confidence=1.0)
        session.clear_context(fu["context"])
        branch = fu["yes"] if polarity else fu["no"]
        return NLUResult(type="FULFILL", intent=intent_name,
                         action=branch["action"], message=branch.get("fulfillment", ""),
                         confidence=1.0, complete=True)

    _UNCERTAIN = ("not sure", "maybe", "dunno", "don't know", "dont know",
                  "i don't know", "no idea", "unsure")

    def _yes_no(self, text: str):
        t = text.lower().strip()
        if any(p in t for p in self._UNCERTAIN):
            return None
        neg = any(re.search(rf"\b{re.escape(p)}\b", t) for p in self.negative)
        pos = any(re.search(rf"\b{re.escape(p)}\b", t) for p in self.affirmative)
        if neg and not pos: return False
        if pos and not neg: return True
        if neg and pos:     return False
        return None

    def _handle_slot_filling(self, session, text):
        intent_name = session.pending_intent
        cfg = self.intents[intent_name]

        # Check for intent interruption: re-classify every slot-filling turn.
        # A different intent at high confidence means the user switched topics.
        new_intent, new_conf = self.classifier.classify(text)
        if (new_intent != intent_name
                and new_intent != "Default Fallback Intent"
                and new_conf >= self.INTERRUPT_THRESHOLD
                and self.intents.get(new_intent) is not None):
            abandoned = intent_name
            session.reset_slot_filling()
            result = self._handle_new_intent(session, text)
            result.interrupted_intent = abandoned
            return result

        if session.awaiting_slot:
            slot = self._slot_def(cfg, session.awaiting_slot)
            value, _ = self.entities.extract(slot["entity"], text)
            if value is None and slot["entity"] in ("remind",):
                value = text.strip()
            if value is not None:
                session.pending_slots[slot["name"]] = value
        self._extract_all_slots(cfg, text, session.pending_slots)
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
        return NLUResult(type="FULFILL", intent=intent_name, action=cfg.get("action"),
                         parameters=params, message=cfg.get("fulfillment", ""),
                         confidence=1.0, complete=True)

    def _handle_new_intent(self, session, text):
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
            return NLUResult(type="FALLBACK", intent="GENAI", action="genai.fallback",
                             confidence=conf, url=self.genai_url + _quote(text))
        if cfg is None:
            return NLUResult(type="FALLBACK", intent="GENAI", action="genai.fallback",
                             confidence=conf, url=self.genai_url + _quote(text))
        if cfg.get("followup"):
            fu = cfg["followup"]
            session.set_context(fu["context"], fu.get("lifespan", 2))
            return NLUResult(type="CONFIRM", intent=intent, action=cfg.get("action"),
                             message=fu["prompt"], confidence=conf)
        if cfg["slots"]:
            slots = {}
            self._extract_all_slots(cfg, text, slots)
            self._fill_open_topics(cfg, text, slots)
            session.pending_intent = intent
            session.pending_slots = slots
            session.awaiting_slot = None
            return self._advance_slots(session, intent, cfg)
        return NLUResult(type="FULFILL", intent=intent, action=cfg.get("action"),
                         message=cfg.get("fulfillment", ""), confidence=conf, complete=True)

    def _extract_all_slots(self, cfg, text, slots: dict):
        for slot in cfg["slots"]:
            if slot["name"] in slots:
                continue
            value, _ = self.entities.extract(slot["entity"], text)
            if value is not None:
                slots[slot["name"]] = value

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

    def _derive_topic(self, text: str):
        t = text.strip()
        for pat in self._CARRIER:
            t = re.sub(pat, "", t, count=1, flags=re.I)
        t = self.entities.strip_datetime(t)
        return t or None

    @staticmethod
    def _slot_def(cfg, name):
        return next(s for s in cfg["slots"] if s["name"] == name)


def _quote(text: str) -> str:
    import urllib.parse
    return urllib.parse.quote(text)
