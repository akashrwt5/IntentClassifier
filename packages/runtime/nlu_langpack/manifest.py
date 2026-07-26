"""
`bundle.json` — the manifest of a single-language `spec/bundle/3.0` bundle.

WHY NOT `pack.json`
-------------------
The reference branch invented a second manifest (`packs/<lang>/pack.json`) with
its own versioning and no integrity story. This repo does not need one:
ADR-005 Part 11 already states that **a downloadable per-language bundle is a
packaging profile of `spec/bundle/3.0` — a bundle with one language — not a new
format.** `bundle.json` already carries `engine_compat`, `languages`, `models`,
`channel`, `checksums_root` and `signature_info`, and the compiler already signs
and verifies it. Adding `pack.json` would mean two containers, two version axes
and two trust models.

So this module parses the manifest the compiler already produces. Signature and
checksum verification stay where they already work — `nlu_compiler.verify` and
`BundleManager` — and are NOT reimplemented here.

Parsing is deliberately strict on the fields the loader depends on: a typo in a
REQUIRED field must fail at load, not surface as a missing capability mid-turn.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import PackManifestError

__all__ = ["EngineCompat", "BundleManifest"]

_CHANNELS = {"dev", "beta", "production"}


@dataclass(frozen=True)
class EngineCompat:
    """The runtime-contract range this bundle declares (integers, per
    bundle.schema.json). `max_tested` is advisory — see version.py."""

    min_runtime_contract: int
    max_tested_runtime_contract: int


@dataclass(frozen=True)
class BundleManifest:
    bundle_id: str
    format_version: str
    content_version: int
    engine_compat: EngineCompat
    languages: dict[str, Any]
    models: dict[str, Any]
    channel: str
    created_at: str
    required_runtime_features: frozenset[str] = frozenset()
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @staticmethod
    def parse(data: dict[str, Any]) -> "BundleManifest":
        if not isinstance(data, dict):
            raise PackManifestError("bundle.json must be a JSON object")

        def req(key: str, typ: type) -> Any:
            if key not in data:
                raise PackManifestError(f"bundle.json missing required field: {key!r}")
            val = data[key]
            if not isinstance(val, typ) or isinstance(val, bool) and typ is not bool:
                raise PackManifestError(
                    f"bundle.json field {key!r} must be {typ.__name__}, "
                    f"got {type(val).__name__}")
            return val

        compat_raw = req("engine_compat", dict)
        for k in ("min_runtime_contract", "max_tested_runtime_contract"):
            if not isinstance(compat_raw.get(k), int) or isinstance(compat_raw.get(k), bool):
                raise PackManifestError(
                    f"bundle.json engine_compat.{k} must be an integer")
        compat = EngineCompat(compat_raw["min_runtime_contract"],
                              compat_raw["max_tested_runtime_contract"])
        if compat.min_runtime_contract > compat.max_tested_runtime_contract:
            raise PackManifestError(
                f"bundle.json engine_compat is inverted: min "
                f"{compat.min_runtime_contract} > max_tested "
                f"{compat.max_tested_runtime_contract}")

        languages = req("languages", dict)
        if not languages:
            raise PackManifestError("bundle.json declares no languages")

        channel = req("channel", str)
        if channel not in _CHANNELS:
            raise PackManifestError(
                f"bundle.json channel {channel!r} invalid; allowed: {sorted(_CHANNELS)}")

        return BundleManifest(
            bundle_id=req("bundle_id", str),
            format_version=req("format_version", str),
            content_version=req("content_version", int),
            engine_compat=compat,
            languages=languages,
            models=req("models", dict),
            channel=channel,
            created_at=req("created_at", str),
            required_runtime_features=frozenset(
                str(f) for f in data.get("required_runtime_features", [])),
            raw=data,
        )

    # -- language selection ------------------------------------------------ #

    def language_status(self, language: str) -> str:
        """Completeness status for one language ('full' | 'partial...').

        `languages` values may be a bare string or an object carrying a status
        field; both shapes appear across the golden bundles, so accept either
        rather than making the loader brittle about it.
        """
        entry = self.languages[language]
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            return str(entry.get("status", "full"))
        return "full"

    def single_language(self) -> str:
        """The language of a per-language bundle.

        A packaging-profile bundle declares exactly one. A multi-language bundle
        is valid under the format but ambiguous here, so the caller must say
        which language it wants.
        """
        if len(self.languages) != 1:
            raise PackManifestError(
                f"bundle declares {len(self.languages)} languages "
                f"({sorted(self.languages)}); pass `language=` to choose one")
        return next(iter(self.languages))
