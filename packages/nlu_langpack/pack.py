"""
`LanguagePack` — the loaded, validated container the engine consumes.

The engine receives a `LanguagePack` and runs the SAME pipeline regardless of the
pack's language. This object exposes three things the engine needs:

  1. `config`      — runtime knobs (thresholds, policy constants, stage list).
  2. `resources`   — loaded data tables (lexicons, keywords, workflows, entities,
                     datetime grammar). The engine's generic interpreters consume
                     these; the pack owns the data, the engine owns the logic.
  3. `model_paths` — resolved artifact paths (intent model, labels, calibration,
                     tokenizer vocab, and — only if enabled — the semantic model).

Component OBJECTS (the `interfaces.py` Protocols) are constructed by the engine
from the above during Phase-1 step 2 (engine migration). This container is the
stable handoff point; it does not itself run inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .manifest import PackManifest


@dataclass
class LanguagePack:
    root: Path
    manifest: PackManifest
    config: dict[str, Any]
    resources: dict[str, Any] = field(default_factory=dict)
    model_paths: dict[str, Path] = field(default_factory=dict)
    semantic_enabled: bool = False
    issues: list[str] = field(default_factory=list)

    # -- convenience accessors -------------------------------------------- #

    @property
    def language(self) -> str:
        return self.manifest.language

    @property
    def pack_id(self) -> str:
        return self.manifest.pack_id

    def has_component(self, name: str) -> bool:
        return name in self.manifest.components

    @property
    def semantic_available(self) -> bool:
        """True only when the pack declares a semantic stage AND it is enabled.
        The engine treats False identically to 'stage absent' — degrade cleanly."""
        return self.manifest.semantic_declared and self.semantic_enabled

    @property
    def stages(self) -> list[str]:
        """Ordered cascade stages for this pack. Semantic is appended only when
        actually available, so a disabled pack yields a shorter cascade."""
        base = list(self.config.get("stages", ["keyword", "intent_model"]))
        if self.semantic_available and "semantic" not in base:
            base.append("semantic")
        if not self.semantic_available and "semantic" in base:
            base = [s for s in base if s != "semantic"]
        return base

    def config_value(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"LanguagePack(pack_id={self.pack_id!r}, language={self.language!r}, "
            f"stages={self.stages}, semantic_available={self.semantic_available}, "
            f"issues={len(self.issues)})"
        )
