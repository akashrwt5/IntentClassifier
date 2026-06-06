"""
On-device NLU engine — a full Dialogflow replacement.

Public API:
    from nlu import NLUEngine
    engine = NLUEngine()
    result = engine.handle(session_id, "set a reminder")
"""

from .engine import NLUEngine, NLUResult

__all__ = ["NLUEngine", "NLUResult"]
