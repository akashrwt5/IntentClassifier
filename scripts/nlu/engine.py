"""
NLUEngine — the orchestrator that replaces Dialogflow end-to-end.

Per user turn it decides which of three modes applies, in priority order:

  1. CONFIRMATION   — a yes/no follow-up context is active (e.g. "send this
                      message?"). Map the utterance to yes/no and fire the
                      corresponding action.
  2. SLOT FILLING   — an intent is mid-collection and we're waiting on a slot.
                      Try to fill the awaited slot from this utterance; if more
                      required slots remain, prompt for the next one.
  3. CLASSIFY       — fresh turn: classify the intent, extract any entities
                      present up-front, then either fulfil immediately (no slots)
                      or begin slot filling / a confirmation follow-up.

The response shape is uniform regardless of mode so the app has a single
contract to render against.
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
    """Uniform engine response."""
    type: str                       # FULFILL | PROMPT | CONFIRM | FALLBACK
    intent: Optional[str] = None
    action: Optional[str] = None
    parameters: dict = field(default_factory=dict)
    message: str = ""               # what the assistant should say/do next
    confidence: float = 0.0
    complete: bool = False          # True when the intent is fully resolved
    url: Optional[str] = None       # GenAI fallback URL, if any

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v not in (None, {}, "")}


class NLUEngine:
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

    # ---------------------------------------------------------------- public
    def handle(self, session_id: str, text: str) -> NLUResult:
        session = self.sessions.get(session_id)
        text = text.strip()

        # Priority 1: an active yes/no confirmation context
        confirm = self._active_confirmation(session)
        if confirm:
            return self._handle_confirmation(session, confirm, text)

        # Priority 2: mid slot-filling
        if session.pending_intent:
            return self._handle_slot_filling(session, text)

        # Priority 3: fresh classification
        return self._handle_new_intent(session, text)

    def reset(self, session_id: str):
        self.sessions.reset(session_id)

    # ---------------------------------------------------- confirmation flow
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
            # didn't understand — re-prompt, keep context alive
            session.set_context(fu["context"], fu.get("lifespan", 2))
            return NLUResult(type="CONFIRM", intent=intent_name,
                             message=fu["prompt"], confidence=1.0)

        session.clear_context(fu["context"])
        branch = fu["yes"] if polarity else fu["no"]
        return NLUResult(
            type="FULFILL", intent=intent_name,
            action=branch["action"], message=branch.get("fulfillment", ""),
            confidence=1.0, complete=True,
        )

    # phrases that explicitly signal uncertainty → treat as "not understood"
    _UNCERTAIN = ("not sure", "maybe", "dunno", "don't know", "dont know",
                  "i don't know", "no idea", "unsure")

    def _yes_no(self, text: str):
        t = text.lower().strip()

        if any(p in t for p in self._UNCERTAIN):
            return None

        neg = any(re.search(rf"\b{re.escape(p)}\b", t) for p in self.negative)
        pos = any(re.search(rf"\b{re.escape(p)}\b", t) for p in self.affirmative)

        # explicit negatives win over incidental positives ("no, don't send")
        if neg and not pos:
            return False
        if pos and not neg:
            return True
        if neg and pos:
            return False  # mixed signal → safest is to not act
        return None

    # ------------------------------------------------------- slot filling
    def _handle_slot_filling(self, session, text):
        intent_name = session.pending_intent
        cfg = self.intents[intent_name]

        # Fill the slot we asked about (and opportunistically any others)
        if session.awaiting_slot:
            slot = self._slot_def(cfg, session.awaiting_slot)
            value, _ = self.entities.extract(slot["entity"], text)
            if value is None and slot["entity"] in ("remind",):
                # open-ended topic entity: accept the raw utterance as the value
                value = text.strip()
            if value is not None:
                session.pending_slots[slot["name"]] = value

        self._extract_all_slots(cfg, text, session.pending_slots)
        return self._advance_slots(session, intent_name, cfg)

    def _advance_slots(self, session, intent_name, cfg):
        """Prompt for the next missing required slot, or fulfil."""
        for slot in cfg["slots"]:
            if slot["required"] and slot["name"] not in session.pending_slots:
                session.pending_intent = intent_name
                session.awaiting_slot = slot["name"]
                return NLUResult(
                    type="PROMPT", intent=intent_name,
                    parameters=dict(session.pending_slots),
                    message=slot["prompt"], confidence=1.0,
                )

        # all required slots present → fulfil
        params = dict(session.pending_slots)
        session.reset_slot_filling()
        return NLUResult(
            type="FULFILL", intent=intent_name, action=cfg.get("action"),
            parameters=params, message=cfg.get("fulfillment", ""),
            confidence=1.0, complete=True,
        )

    # ---------------------------------------------------- new classification
    def _handle_new_intent(self, session, text):
        session.decrement_contexts()
        intent, conf = self.classifier.classify(text)

        # low confidence or explicit out-of-scope → GenAI fallback
        if intent == "OUT_OF_SCOPE" or conf < self.threshold:
            return NLUResult(
                type="FALLBACK", intent="GENAI", action="genai.fallback",
                confidence=conf,
                url=self.genai_url + _quote(text),
                message="",
            )

        cfg = self.intents.get(intent)
        if cfg is None:
            return NLUResult(type="FALLBACK", intent="GENAI",
                             action="genai.fallback", confidence=conf,
                             url=self.genai_url + _quote(text))

        # follow-up intent (e.g. send message → confirm yes/no)
        if cfg.get("followup"):
            fu = cfg["followup"]
            session.set_context(fu["context"], fu.get("lifespan", 2))
            return NLUResult(
                type="CONFIRM", intent=intent, action=cfg.get("action"),
                message=fu["prompt"], confidence=conf,
            )

        # slot-filling intent: grab whatever entities are already present
        if cfg["slots"]:
            slots = {}
            self._extract_all_slots(cfg, text, slots)
            session.pending_intent = intent
            session.pending_slots = slots
            session.awaiting_slot = None
            return self._advance_slots(session, intent, cfg)

        # simple fire-and-forget intent
        return NLUResult(
            type="FULFILL", intent=intent, action=cfg.get("action"),
            message=cfg.get("fulfillment", ""), confidence=conf, complete=True,
        )

    # ------------------------------------------------------------- helpers
    def _extract_all_slots(self, cfg, text, slots: dict):
        for slot in cfg["slots"]:
            if slot["name"] in slots:
                continue
            value, _ = self.entities.extract(slot["entity"], text)
            if value is not None:
                slots[slot["name"]] = value

    @staticmethod
    def _slot_def(cfg, name):
        return next(s for s in cfg["slots"] if s["name"] == name)


def _quote(text: str) -> str:
    import urllib.parse
    return urllib.parse.quote(text)
