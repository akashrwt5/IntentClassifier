"""
nlu_langpack — the Language Pack contract for the on-device NLU platform.

A Language Pack is the unit that makes the engine understand ONE language. The
engine is a language-agnostic execution framework; everything language-specific
(models, tables, lexicons, grammars, prompts, config) ships inside a pack. Adding
a language means adding a pack, never editing engine code.

This package defines and locks that contract:

- `interfaces`  — the component Protocols the engine depends on (structural).
- `manifest`    — `pack.json` parsing + validation.
- `pack`        — the `LanguagePack` container the loader returns.
- `loader`      — `load_pack()`: validate, check compatibility, gate the semantic
                  plugin off-by-default, and construct a `LanguagePack`.

Public API:

    from nlu_langpack import load_pack, LanguagePack, RUNTIME_CONTRACT_VERSION
    pack = load_pack("packs/en")                 # semantic OFF by default
    pack = load_pack("packs/en", enable_semantic=True)
"""

from .version import RUNTIME_CONTRACT_VERSION
from .errors import (
    PackError,
    PackManifestError,
    PackCompatibilityError,
    PackResourceError,
)
from .manifest import PackManifest
from .pack import LanguagePack
from .loader import load_pack

__all__ = [
    "RUNTIME_CONTRACT_VERSION",
    "PackError",
    "PackManifestError",
    "PackCompatibilityError",
    "PackResourceError",
    "PackManifest",
    "LanguagePack",
    "load_pack",
]
