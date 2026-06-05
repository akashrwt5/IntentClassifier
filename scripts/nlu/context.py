"""
Session & context management — the conversational memory of Dialogflow.

Dialogflow tracks per-session "contexts" with a lifespan (number of turns
before they expire) plus the partially-filled parameters of an in-progress
intent. This module reimplements that with a simple in-memory store keyed by
session_id. For a single on-device user a dict is plenty; a server-side
deployment can swap SessionStore for a Redis/SQLite-backed implementation
without touching the engine.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Context:
    """An active Dialogflow-style context with a turn-based lifespan."""
    name: str
    lifespan: int
    parameters: dict = field(default_factory=dict)


@dataclass
class Session:
    """Per-user conversational state."""
    session_id: str
    # Active contexts: name -> Context
    contexts: dict = field(default_factory=dict)
    # In-progress slot filling
    pending_intent: Optional[str] = None
    pending_slots: dict = field(default_factory=dict)   # slot_name -> value
    awaiting_slot: Optional[str] = None                 # slot currently prompted for

    def set_context(self, name: str, lifespan: int = 5, parameters: dict = None):
        self.contexts[name] = Context(name, lifespan, parameters or {})

    def has_context(self, name: str) -> bool:
        return name in self.contexts

    def clear_context(self, name: str):
        self.contexts.pop(name, None)

    def decrement_contexts(self):
        """Age all contexts by one turn; drop the expired ones."""
        for name in list(self.contexts):
            self.contexts[name].lifespan -= 1
            if self.contexts[name].lifespan <= 0:
                del self.contexts[name]

    def reset_slot_filling(self):
        self.pending_intent = None
        self.pending_slots = {}
        self.awaiting_slot = None


class SessionStore:
    """In-memory session registry. Swap out for persistent storage if needed."""

    def __init__(self):
        self._sessions: dict = {}

    def get(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id)
        return self._sessions[session_id]

    def reset(self, session_id: str):
        self._sessions.pop(session_id, None)
