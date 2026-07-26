"""`LanguagePack` — what `load_pack()` hands the engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .manifest import BundleManifest

__all__ = ["LanguagePack"]


@dataclass(frozen=True)
class LanguagePack:
    """One language's behaviour, resolved and validated.

    The engine reads `resources` (data tables it interprets) and `model_paths`
    (artifacts it hands to an InferenceBackend). It never inspects `language` to
    choose a code path — that string is for telemetry and logging only. If the
    engine ever branches on it, the neutrality guard has failed.
    """

    root: Path
    language: str
    manifest: BundleManifest
    config: dict[str, Any]
    resources: dict[str, Any]
    model_paths: dict[str, Path]
    semantic_available: bool
    issues: list[str] = field(default_factory=list)

    @property
    def channel(self) -> str:
        return self.manifest.channel

    @property
    def stages(self) -> tuple[str, ...]:
        """Cascade stages active for this pack, in order."""
        stages = ["keyword", "intent_model"]
        if self.semantic_available:
            stages.append("semantic")
        return tuple(stages)

    def __repr__(self) -> str:
        return (f"LanguagePack(language={self.language!r}, "
                f"channel={self.channel!r}, stages={self.stages}, "
                f"issues={len(self.issues)})")
