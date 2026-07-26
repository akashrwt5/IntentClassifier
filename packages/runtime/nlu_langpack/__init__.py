"""
nlu_langpack — the Language Pack contract.

The locked boundary between the language-agnostic engine and everything
language-specific.

  from nlu_langpack import load_pack
  pack = load_pack("bundles/en")                        # semantic OFF by default
  pack = load_pack("bundles/en", enable_semantic=True)

The container is a single-language `spec/bundle/3.0` bundle — ADR-005 Part 11:
a per-language bundle is a packaging profile of that format, not a new one.
There is deliberately no second manifest and no second trust model; signing and
checksums stay with `nlu_compiler.verify` / `BundleManager`.

This package imports neither the engine nor any model runtime, so both sides can
depend on it without a cycle.
"""
from .errors import (
    LangPackError, PackCompatibilityError, PackLanguageError,
    PackManifestError, PackResourceError,
)
from .interfaces import (
    COMPONENT_NAMES, REQUIRED_COMPONENTS, EntityExtractor, IntentModel,
    KeywordMatcher, Lexicon, SemanticClassifier, TextNormalizer, Tokenizer,
    WorkflowProvider,
)
from .loader import load_pack
from .manifest import BundleManifest, EngineCompat
from .pack import LanguagePack
from .version import (
    RUNTIME_CONTRACT_VERSION, SUPPORTED_FEATURES, check_compatibility,
)

__all__ = [
    "load_pack", "LanguagePack", "BundleManifest", "EngineCompat",
    "RUNTIME_CONTRACT_VERSION", "SUPPORTED_FEATURES", "check_compatibility",
    "COMPONENT_NAMES", "REQUIRED_COMPONENTS",
    "TextNormalizer", "Tokenizer", "KeywordMatcher", "IntentModel",
    "SemanticClassifier", "EntityExtractor", "Lexicon", "WorkflowProvider",
    "LangPackError", "PackManifestError", "PackCompatibilityError",
    "PackResourceError", "PackLanguageError",
]
