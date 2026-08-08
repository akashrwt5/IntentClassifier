"""
Session & context management — the conversational memory of Dialogflow.

Sprint-1 hardening:
  * Time-based expiry. Contexts carry a created_at timestamp and a TTL;
    sessions track last_active. A clock is injected (defaults to time.time)
    so expiry is deterministically testable. This prevents a stale
    confirmation context from a previous conversation firing an action on
    an unrelated affirmative ("yes") hours later.
  * Slot-fill attempt tracking so the engine can abandon a flow the user
    cannot complete instead of prompting forever.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

# Confirmations and slot prompts are short-lived dialogue state. A user who
# walks away mid-confirmation should not have a stale "yes/no" fire on their
# next, unrelated utterance. 90s comfortably covers a real reply while
# expiring abandoned turns.
DEFAULT_CONTEXT_TTL_SECONDS = 90.0

# A session idle longer than this is treated as a new conversation: pending
# slot-filling and contexts are cleared on next access.
DEFAULT_SESSION_TTL_SECONDS = 600.0  # 10 minutes


@dataclass
class Context:
    name: str
    lifespan: int
    parameters: dict = field(default_factory=dict)
    created_at: float = 0.0
    ttl_seconds: Optional[float] = None

    def is_expired(self, now: float) -> bool:
        if self.ttl_seconds is None:
            return False
        return (now - self.created_at) > self.ttl_seconds


@dataclass
class Session:
    session_id: str
    contexts: dict = field(default_factory=dict)
    pending_intent: Optional[str] = None
    pending_slots: dict = field(default_factory=dict)
    awaiting_slot: Optional[str] = None
    slot_attempts: int = 0          # failed attempts to fill awaiting_slot
    # Non-answers to an authored yes/no `followup`. Bounded for the same reason
    # as slot_attempts: the confirmation re-sets its own context each time it
    # re-asks, so without a budget a user who is never understood is held in it
    # forever.
    confirm_attempts: int = 0
    # When a date-time slot answer gives a day but no time ("tomorrow"), the
    # resolved day is parked here so the time prompt's answer can be anchored to
    # it (keeping the day) instead of resolving against today.
    partial_datetime: Optional[str] = None
    last_active: float = 0.0        # wall-clock of the last handled turn
    # ND-11(a): an action held back by the uncertainty-confirmation gate,
    # awaiting the user's yes/no. {"intent", "action", "fulfillment"}.
    pending_confirm: Optional[dict] = None
    # runtime-contract-v1 §4: last execution outcome (notify_execution)
    last_execution: Optional[dict] = None

    # --- context parameter memory ---
    # Stores the last fulfilled parameters per intent family so the engine
    # can resolve back-references ("change back", "remind me again") and
    # pre-fill slots in follow-up turns without re-asking the user.
    last_fulfilled: dict = field(default_factory=dict)  # {intent: {slot: value}}
    last_memory: Optional[str] = None       # last active memory preset
    prev_memory: Optional[str] = None       # memory before the last change

    # -- session blob (runtime-contract-v1 §3) --------------------------------
    # The core owns serialization; the HOST persists/encrypts the blob.
    # Never contains raw utterance text. Unknown fields in a newer minor
    # version are preserved round-trip via the _extra bag.

    def to_blob(self) -> dict:
        frame = None
        if self.pending_intent:
            frame = {"intent": self.pending_intent,
                     "slots_filled": dict(self.pending_slots),
                     "awaiting_slot": self.awaiting_slot,
                     "attempts": self.slot_attempts,
                     "confirm_pending": False}
        elif self.pending_confirm:
            frame = {"intent": self.pending_confirm["intent"],
                     "slots_filled": {}, "awaiting_slot": None,
                     "attempts": 0, "confirm_pending": True,
                     "held_action": self.pending_confirm.get("action"),
                     "held_fulfillment": self.pending_confirm.get("fulfillment", "")}
        return {"v": 1, "session_id": self.session_id,
                "updated_at_mono_ms": int(self.last_active * 1000),
                "frames": [frame] if frame else [],
                "contexts": {n: {"lifespan": c.lifespan, "created_at": c.created_at}
                              for n, c in self.contexts.items()},
                "last_fulfilled": {k: dict(v) for k, v in self.last_fulfilled.items()},
                "last_memory": self.last_memory, "prev_memory": self.prev_memory,
                "partial_datetime": self.partial_datetime,
                "counters": {}}

    @classmethod
    def from_blob(cls, blob: dict) -> "Session":
        s = cls(session_id=blob["session_id"])
        s.last_active = blob.get("updated_at_mono_ms", 0) / 1000.0
        frames = blob.get("frames", [])
        if frames:
            f = frames[-1]
            if f.get("confirm_pending"):
                s.pending_confirm = {"intent": f["intent"],
                                     "action": f.get("held_action"),
                                     "fulfillment": f.get("held_fulfillment", "")}
            else:
                s.pending_intent = f["intent"]
                s.pending_slots = dict(f.get("slots_filled", {}))
                s.awaiting_slot = f.get("awaiting_slot")
                s.slot_attempts = f.get("attempts", 0)
        for name, c in blob.get("contexts", {}).items():
            s.set_context(name, c.get("lifespan", 2), now=c.get("created_at", 0.0))
        s.last_fulfilled = {k: dict(v) for k, v in blob.get("last_fulfilled", {}).items()}
        s.last_memory = blob.get("last_memory")
        s.prev_memory = blob.get("prev_memory")
        s.partial_datetime = blob.get("partial_datetime")
        return s

    def set_context(self, name: str, lifespan: int = 5, parameters: dict = None,
                    now: float = 0.0, ttl_seconds: Optional[float] = DEFAULT_CONTEXT_TTL_SECONDS):
        self.contexts[name] = Context(name, lifespan, parameters or {},
                                      created_at=now, ttl_seconds=ttl_seconds)

    def has_context(self, name: str) -> bool:
        return name in self.contexts

    def clear_context(self, name: str):
        self.contexts.pop(name, None)

    def expire_contexts(self, now: float):
        """Drop any context past its TTL. Called once per turn before routing."""
        for name in list(self.contexts):
            if self.contexts[name].is_expired(now):
                del self.contexts[name]

    def decrement_contexts(self):
        for name in list(self.contexts):
            self.contexts[name].lifespan -= 1
            if self.contexts[name].lifespan <= 0:
                del self.contexts[name]

    def reset_slot_filling(self):
        self.pending_intent = None
        self.pending_slots = {}
        self.awaiting_slot = None
        self.slot_attempts = 0
        self.partial_datetime = None

    def record_fulfillment(self, intent: str, params: dict):
        """Called after every FULFILL to persist parameters for later reuse."""
        self.last_fulfilled[intent] = dict(params)
        if intent == "Cmd.MemoryChange" and params.get("MemoryName"):
            self.prev_memory = self.last_memory
            self.last_memory = params["MemoryName"]

    def get_last_params(self, intent: str) -> dict:
        return dict(self.last_fulfilled.get(intent, {}))


class SessionStore:
    def __init__(self,
                 clock: Callable[[], float] = time.time,
                 session_ttl_seconds: float = DEFAULT_SESSION_TTL_SECONDS):
        self._sessions: dict = {}
        self._clock = clock
        self._session_ttl = session_ttl_seconds

    def now(self) -> float:
        return self._clock()

    def get(self, session_id: str) -> Session:
        """
        Return the live session, resetting it first if it has been idle longer
        than the session TTL. An idle session is treated as a fresh
        conversation so stale pending state never bleeds across a long gap.
        """
        now = self._clock()
        sess = self._sessions.get(session_id)
        if sess is None:
            sess = Session(session_id, last_active=now)
            self._sessions[session_id] = sess
            return sess
        if (now - sess.last_active) > self._session_ttl:
            sess = Session(session_id, last_active=now)
            self._sessions[session_id] = sess
            return sess
        sess.last_active = now
        return sess

    def reset(self, session_id: str):
        self._sessions.pop(session_id, None)
