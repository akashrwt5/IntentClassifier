"""
Session & context management — the conversational memory of Dialogflow.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Context:
    name: str
    lifespan: int
    parameters: dict = field(default_factory=dict)


@dataclass
class Session:
    session_id: str
    contexts: dict = field(default_factory=dict)
    pending_intent: Optional[str] = None
    pending_slots: dict = field(default_factory=dict)
    awaiting_slot: Optional[str] = None

    # --- context parameter memory ---
    # Stores the last fulfilled parameters per intent family so the engine
    # can resolve back-references ("change back", "remind me again") and
    # pre-fill slots in follow-up turns without re-asking the user.
    last_fulfilled: dict = field(default_factory=dict)  # {intent: {slot: value}}
    last_memory: Optional[str] = None       # last active memory preset
    prev_memory: Optional[str] = None       # memory before the last change

    def set_context(self, name: str, lifespan: int = 5, parameters: dict = None):
        self.contexts[name] = Context(name, lifespan, parameters or {})

    def has_context(self, name: str) -> bool:
        return name in self.contexts

    def clear_context(self, name: str):
        self.contexts.pop(name, None)

    def decrement_contexts(self):
        for name in list(self.contexts):
            self.contexts[name].lifespan -= 1
            if self.contexts[name].lifespan <= 0:
                del self.contexts[name]

    def reset_slot_filling(self):
        self.pending_intent = None
        self.pending_slots = {}
        self.awaiting_slot = None

    def record_fulfillment(self, intent: str, params: dict):
        """Called after every FULFILL to persist parameters for later reuse."""
        self.last_fulfilled[intent] = dict(params)
        if intent == "Cmd.MemoryChange" and params.get("MemoryName"):
            self.prev_memory = self.last_memory
            self.last_memory = params["MemoryName"]

    def get_last_params(self, intent: str) -> dict:
        return dict(self.last_fulfilled.get(intent, {}))


class SessionStore:
    def __init__(self):
        self._sessions: dict = {}

    def get(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id)
        return self._sessions[session_id]

    def reset(self, session_id: str):
        self._sessions.pop(session_id, None)
