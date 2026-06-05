"""
On-device NLU engine — a full Dialogflow replacement.

Pipeline:
    text → IntentClassifier (ONNX) → EntityExtractor → SlotFiller / ContextManager → Response

Public API:
    from nlu import NLUEngine
    engine = NLUEngine()
    result = engine.handle(session_id, "set a reminder")

The engine is intentionally written in dependency-light, pure Python so the
same logic can be ported to Kotlin (Android) / Swift (iOS) alongside the
ONNX Runtime Mobile model. All conversational config lives in declarative
JSON (data/nlu_schema.json, data/nlu_entities.json) rather than in code.
"""

from .engine import NLUEngine, NLUResult

__all__ = ["NLUEngine", "NLUResult"]
