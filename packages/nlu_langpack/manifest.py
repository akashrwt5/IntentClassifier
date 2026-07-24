"""
`pack.json` — the Language Pack manifest.

Read first, trusted after (future) signature verification. It carries structural
and versioning facts only; runtime knobs live in `config.json` (see `pack.py`).

The manifest is deliberately small and closed: unknown REQUIRED-level typos must
not pass silently, so parsing validates required fields and types explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import PackManifestError
from .interfaces import COMPONENT_NAMES, REQUIRED_COMPONENTS


@dataclass(frozen=True)
class EngineCompat:
    """The contract range this pack was built for."""

    min_runtime_contract: str
    max_runtime_contract: str


@dataclass(frozen=True)
class PackManifest:
    pack_id: str
    language: str
    format_version: str
    content_version: int
    engine_compat: EngineCompat
    components: frozenset[str]
    models: dict[str, Any]
    channel: str
    created_at: str
    semantic_declared: bool = False
    required_features: frozenset[str] = field(default_factory=frozenset)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    # -- parsing ----------------------------------------------------------- #

    @staticmethod
    def parse(data: dict[str, Any]) -> "PackManifest":
        if not isinstance(data, dict):
            raise PackManifestError("pack.json must be a JSON object")

        def req(key: str, typ: type) -> Any:
            if key not in data:
                raise PackManifestError(f"pack.json missing required field: {key!r}")
            val = data[key]
            if not isinstance(val, typ):
                raise PackManifestError(
                    f"pack.json field {key!r} must be {typ.__name__}, got {type(val).__name__}"
                )
            return val

        pack_id = req("pack_id", str)
        language = req("language", str)
        format_version = req("format_version", str)
        content_version = req("content_version", int)
        channel = req("channel", str)
        created_at = req("created_at", str)

        compat_raw = req("engine_compat", dict)
        for k in ("min_runtime_contract", "max_runtime_contract"):
            if k not in compat_raw or not isinstance(compat_raw[k], str):
                raise PackManifestError(f"pack.json engine_compat missing/invalid {k!r}")
        engine_compat = EngineCompat(
            min_runtime_contract=compat_raw["min_runtime_contract"],
            max_runtime_contract=compat_raw["max_runtime_contract"],
        )

        components_list = req("components", list)
        components = frozenset(str(c) for c in components_list)
        unknown = components - COMPONENT_NAMES
        if unknown:
            raise PackManifestError(
                f"pack.json declares unknown component(s): {sorted(unknown)}; "
                f"allowed: {sorted(COMPONENT_NAMES)}"
            )
        missing_required = REQUIRED_COMPONENTS - components
        if missing_required:
            raise PackManifestError(
                f"pack.json missing required component(s): {sorted(missing_required)}"
            )

        models = req("models", dict)

        # `semantic` is an optional object; presence + declared flag are what the
        # loader consults. Absent → not declared → stage cannot be enabled.
        semantic_declared = False
        sem = data.get("semantic")
        if isinstance(sem, dict):
            semantic_declared = bool(sem.get("declared", False))
        if semantic_declared and "semantic" not in components:
            raise PackManifestError(
                "pack.json declares semantic model but 'semantic' is not in components"
            )

        channel_allowed = {"dev", "beta", "production"}
        if channel not in channel_allowed:
            raise PackManifestError(
                f"pack.json channel {channel!r} invalid; allowed: {sorted(channel_allowed)}"
            )

        required_features = frozenset(
            str(f) for f in data.get("required_features", [])
        )

        return PackManifest(
            pack_id=pack_id,
            language=language,
            format_version=format_version,
            content_version=content_version,
            engine_compat=engine_compat,
            components=components,
            models=models,
            channel=channel,
            created_at=created_at,
            semantic_declared=semantic_declared,
            required_features=required_features,
            raw=data,
        )
