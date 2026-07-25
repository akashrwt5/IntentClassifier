"""
The Language Pack interface contract — what the engine depends on.

These are the ONLY things the language-agnostic engine knows about. It never sees
"English" or "Japanese"; it sees a `TextNormalizer`, a `Tokenizer`, an
`IntentModel`, and so on. Every language provides implementations (or data that a
generic engine-side interpreter consumes) through a `LanguagePack`.

Design choice — structural typing via `typing.Protocol`:
  The engine programs against these Protocols, not concrete classes. A pack (or a
  future alternative implementation) satisfies a Protocol by shape, with no
  inheritance coupling. This keeps the engine ↔ pack boundary a pure contract.

Two provider styles are both valid under this contract, and the split is
deliberate (see IMPLEMENTATION-PLAN §2.2/§4):
  - LOGIC-bearing components the pack provides as objects: `Tokenizer`,
    `IntentModel`, `SemanticClassifier` (they wrap a model/vocab).
  - DATA the pack provides as tables, interpreted by a GENERIC engine-side
    interpreter: keyword rules, datetime grammar, lexicons, workflows. The
    corresponding Protocols below (`KeywordMatcher`, `EntityExtractor`,
    `Lexicon`, `WorkflowProvider`) are the interpreter's contract; the *tables*
    live in the pack, the *interpretation* lives in the engine.

Nothing here imports the engine or any model runtime — this module is a
dependency-free contract so both sides can depend on it without a cycle.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


# --------------------------------------------------------------------------- #
# Text pipeline
# --------------------------------------------------------------------------- #

@runtime_checkable
class TextNormalizer(Protocol):
    """Language-specific text normalization (casing, Unicode NFC, digit and
    diacritic policy). Driven by the pack's `normalizer` rules."""

    def normalize(self, text: str) -> str: ...


@runtime_checkable
class Tokenizer(Protocol):
    """Tokenization for the pack's models. `tokenize` yields surface tokens;
    `encode` yields model input ids (e.g. WordPiece) when a model needs them."""

    def tokenize(self, text: str) -> list[str]: ...

    def encode(self, text: str) -> list[int]: ...


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #

@runtime_checkable
class KeywordMatcher(Protocol):
    """Stage-1 deterministic pre-filter. Interprets the pack's keyword rule table
    (portable-regex subset, tiers, guards). Returns the top match or None."""

    def match(self, text: str) -> Optional["KeywordHit"]: ...


@runtime_checkable
class IntentModel(Protocol):
    """Stage-2 statistical intent classifier for one language (e.g. TF-IDF + LR
    exported to ONNX, temperature-calibrated). Returns ranked (intent, score)."""

    def classify(self, text: str) -> list[tuple[str, float]]: ...


@runtime_checkable
class SemanticClassifier(Protocol):
    """Optional Stage-3 semantic rescue (e.g. MiniLM encoder + LR head with a
    learned out-of-scope class). Present ONLY when the pack declares it and the
    runtime enables it. Returns (intent, confidence)."""

    def classify(self, text: str) -> tuple[str, float]: ...


# --------------------------------------------------------------------------- #
# Entities
# --------------------------------------------------------------------------- #

@runtime_checkable
class EntityExtractor(Protocol):
    """Entity/slot resolution. Enum entities come from the pack's synonym tables;
    system entities (datetime, number) are interpreted from the pack's grammar
    tables by the generic engine-side interpreter."""

    def extract(self, entity_type: str, text: str) -> Optional[str]: ...

    def extract_datetime(self, text: str, now: Optional[float] = None) -> Optional[str]: ...

    def is_open(self, entity_type: str) -> bool: ...


# --------------------------------------------------------------------------- #
# Language behavior facts (data the engine interprets)
# --------------------------------------------------------------------------- #

@runtime_checkable
class Lexicon(Protocol):
    """Per-language word tables the engine's generic logic consumes: yes/no,
    uncertainty markers, idioms, carrier phrases, universal verbs, negation cues.
    These replace the constants currently hardcoded in the engine."""

    def affirmatives(self) -> frozenset[str]: ...

    def negatives(self) -> frozenset[str]: ...

    def carriers(self) -> tuple[str, ...]: ...

    def idioms(self) -> tuple[str, ...]: ...


@runtime_checkable
class WorkflowProvider(Protocol):
    """The compiled intent workflows (slots, prompts, validation, confirmation,
    completion) for this language. Pure data; the engine's workflow interpreter
    executes it identically for every language."""

    @property
    def intents(self) -> dict: ...


# --------------------------------------------------------------------------- #
# Shared value types
# --------------------------------------------------------------------------- #

class KeywordHit:
    """A keyword pre-filter match. Deliberately a tiny concrete value type (not a
    Protocol): it crosses the engine ↔ pack boundary as data."""

    __slots__ = ("intent", "tier", "score")

    def __init__(self, intent: str, tier: str, score: float = 1.0) -> None:
        self.intent = intent
        self.tier = tier
        self.score = score

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"KeywordHit(intent={self.intent!r}, tier={self.tier!r}, score={self.score})"


# The set of component names a pack may declare in its manifest. The loader
# validates declarations against this closed set; the engine resolves each to the
# matching Protocol above.
COMPONENT_NAMES = frozenset(
    {
        "normalizer",
        "tokenizer",
        "keyword_matcher",
        "intent_model",
        "entity_extractor",
        "lexicon",
        "workflow",
        "semantic",  # optional
    }
)

# Components every production pack must declare. `semantic` is intentionally NOT
# here — it is optional and off by default.
REQUIRED_COMPONENTS = frozenset(
    {
        "normalizer",
        "tokenizer",
        "intent_model",
        "entity_extractor",
        "lexicon",
        "workflow",
    }
)
