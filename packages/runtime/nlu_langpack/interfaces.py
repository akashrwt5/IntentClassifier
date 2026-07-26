"""
The Language Pack interface contract — what the engine depends on.

These are the ONLY things the language-agnostic engine knows about. It never
sees "English" or "Japanese"; it sees a `TextNormalizer`, a `Tokenizer`, an
`IntentModel`. Every language provides implementations (or data a generic engine
interpreter consumes) through a `LanguagePack`.

Design choice — structural typing via `typing.Protocol`. The engine programs
against these Protocols, not concrete classes, so a pack satisfies the contract
by shape with no inheritance coupling. That keeps the engine <-> pack boundary a
pure contract.

TWO PROVIDER STYLES, AND WHY THE SPLIT MATTERS
----------------------------------------------
  - LOGIC-bearing components arrive as OBJECTS: `Tokenizer`, `IntentModel`,
    `SemanticClassifier` — each wraps a model or vocabulary.
  - BEHAVIOUR FACTS arrive as DATA TABLES interpreted by a GENERIC engine-side
    interpreter: keyword rules, datetime grammar, lexicons, workflows. The
    Protocols below (`KeywordMatcher`, `EntityExtractor`, `Lexicon`,
    `WorkflowProvider`) are that interpreter's contract; the *tables* live in
    the pack, the *interpretation* lives in the engine.

The second style is what makes a new language authorable rather than
programmable, and it is precisely what today's engine lacks — English datetime
grammar, carrier phrases and connectors are still code (Review-F5 blocker B10,
charter step A7).

This module imports neither the engine nor any model runtime, so both sides can
depend on it without a cycle. `tests/test_package_boundaries.py` enforces that.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

__all__ = [
    "TextNormalizer", "Tokenizer", "KeywordMatcher", "KeywordHit", "IntentModel",
    "SemanticClassifier", "EntityExtractor", "Lexicon", "WorkflowProvider",
    "COMPONENT_NAMES", "REQUIRED_COMPONENTS",
]


# --------------------------------------------------------------------------- #
# Text pipeline
# --------------------------------------------------------------------------- #

@runtime_checkable
class TextNormalizer(Protocol):
    """Language-specific normalization (casing, Unicode form, digit and
    diacritic policy), driven by the pack's normalizer rules."""

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

class KeywordHit(Protocol):
    """One keyword-rule match: the intent, and the tier that matched (which
    determines the calibrated confidence the engine assigns)."""

    intent: str
    tier: str


@runtime_checkable
class KeywordMatcher(Protocol):
    """Stage-1 deterministic pre-filter. Interprets the pack's keyword rule
    table (portable-regex subset, tiers, guards). Top match or None."""

    def match(self, text: str) -> Optional[KeywordHit]: ...


@runtime_checkable
class IntentModel(Protocol):
    """Stage-2 statistical intent classifier for one language (TF-IDF + LR
    exported to ONNX, temperature-calibrated). Returns ranked (intent, score).

    The calibration temperature belongs to the (model, featurizer) PAIR and
    therefore travels with the model inside the pack — never with the language.
    Applying one featurizer's temperature to another's logits is Review-F5
    blocker B8.
    """

    def classify(self, text: str) -> list[tuple[str, float]]: ...


@runtime_checkable
class SemanticClassifier(Protocol):
    """Optional Stage-3 semantic rescue (MiniLM encoder + LR head with a learned
    out-of-scope class). Present ONLY when the pack declares it AND the runtime
    enables it — off by default."""

    def classify(self, text: str) -> tuple[str, float]: ...


# --------------------------------------------------------------------------- #
# Entities
# --------------------------------------------------------------------------- #

@runtime_checkable
class EntityExtractor(Protocol):
    """Entity/slot resolution. Enum entities come from the pack's synonym
    tables; system entities (datetime, number) are interpreted from the pack's
    grammar tables by the generic engine-side interpreter."""

    def extract(self, entity_type: str, text: str) -> Optional[str]: ...

    def extract_datetime(self, text: str,
                         now: Optional[float] = None) -> Optional[str]: ...

    def is_open(self, entity_type: str) -> bool: ...


# --------------------------------------------------------------------------- #
# Language behaviour facts (data the engine interprets)
# --------------------------------------------------------------------------- #

@runtime_checkable
class Lexicon(Protocol):
    """Per-language word tables the engine's generic logic consumes: yes/no,
    uncertainty markers, idioms, carrier phrases, negation cues.

    These replace constants currently hardcoded in the engine. `negation_cues`
    is deliberately distinct from `negatives`: the former are grammatical
    negators preceding a command ("ne", "nicht", "ikke"), the latter is the
    yes/no ANSWER vocabulary used for confirmations ("non merci", "nej tak").
    Conflating them loses French "ne ... pas" entirely.
    """

    def affirmatives(self) -> frozenset[str]: ...

    def negatives(self) -> frozenset[str]: ...

    def negation_cues(self) -> tuple[str, ...]: ...

    def carriers(self) -> tuple[str, ...]: ...

    def idioms(self) -> tuple[str, ...]: ...


@runtime_checkable
class WorkflowProvider(Protocol):
    """The compiled workflow/schema tables: intents, slots, prompts, guards,
    policy. Structural facts are language-neutral; the surface strings are not."""

    def schema(self) -> dict: ...


# --------------------------------------------------------------------------- #
# Component vocabulary — closed set, so a typo cannot pass silently
# --------------------------------------------------------------------------- #

COMPONENT_NAMES: frozenset[str] = frozenset({
    "normalizer", "tokenizer", "keyword_matcher", "intent_model",
    "entity_extractor", "lexicon", "workflow", "semantic",
})

# Everything except `semantic`, which is an opt-in plugin stage.
REQUIRED_COMPONENTS: frozenset[str] = COMPONENT_NAMES - {"semantic"}
