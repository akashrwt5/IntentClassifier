"""
`load_pack()` — validate a Language Pack and hand the engine a `LanguagePack`.

Load sequence (fail loudly, early):
  1. Read `pack.json`; parse + validate the manifest (`manifest.PackManifest`).
  2. Compatibility gate: this runtime's contract version must fall within the
     pack's declared `engine_compat` range; required features must be supported.
  3. Load `config.json` (runtime knobs; defaults filled if absent).
  4. Resolve + presence-check declared resource tables and model artifacts.
     Missing artifacts are a hard error on a `production` pack, a warning on
     `dev`/`beta` (so skeleton/dev packs load for iteration).
  5. Resolve the semantic stage's enabled state — OFF BY DEFAULT — and, when off,
     do not resolve/require its model at all.

This module contains NO language-specific logic and NO `if language == …`
branches: a pack for any language code loads through exactly this path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from .errors import PackCompatibilityError, PackManifestError, PackResourceError
from .manifest import PackManifest
from .pack import LanguagePack
from .version import RUNTIME_CONTRACT_VERSION, parse_version

# Runtime features this contract version supports. A pack requiring anything
# outside this set is refused (capability-by-declaration, not by crash).
SUPPORTED_FEATURES: frozenset[str] = frozenset()

# Resource tables the engine's interpreters read, mapped to their in-pack path.
# Presence is validated; content is parsed for JSON tables so a corrupt table
# fails at load, not mid-conversation.
# Keyed by the closed component names in interfaces.COMPONENT_NAMES. Components
# whose resources are directories/binaries (tokenizer vocab, entity + datetime
# tables) are validated in the engine-migration step, not here.
_RESOURCE_FILES: dict[str, str] = {
    "lexicon": "lexicons.json",
    "keyword_matcher": "keywords.json",
    "workflow": "schema.json",
    "normalizer": "normalizer.json",
}


def _resolve_semantic_enabled(
    override: Optional[bool], config: dict[str, Any]
) -> tuple[bool, str]:
    """Resolve whether the semantic stage is enabled, and WHERE the decision came
    from (so the loader can be strict about an explicit request vs. lenient about
    a broad switch).

    Precedence (first wins): constructor arg → NLU_ENABLE_SEMANTIC env →
    config `semantic_enabled` → default False. Returns (enabled, source) where
    source ∈ {"arg", "env", "config", "default"}.
    """
    if override is not None:
        return bool(override), "arg"
    env = os.environ.get("NLU_ENABLE_SEMANTIC")
    if env is not None and env.strip() != "":
        return env.strip().lower() in ("1", "true", "yes", "on"), "env"
    if "semantic_enabled" in config:
        return bool(config["semantic_enabled"]), "config"
    return False, "default"


def _check_compatibility(manifest: PackManifest) -> None:
    runtime = parse_version(RUNTIME_CONTRACT_VERSION)
    lo = parse_version(manifest.engine_compat.min_runtime_contract)
    hi = parse_version(manifest.engine_compat.max_runtime_contract)
    if not (lo <= runtime <= hi):
        raise PackCompatibilityError(
            f"pack {manifest.pack_id!r} needs runtime contract in "
            f"[{manifest.engine_compat.min_runtime_contract}, "
            f"{manifest.engine_compat.max_runtime_contract}]; "
            f"this runtime is {RUNTIME_CONTRACT_VERSION}"
        )
    unsupported = manifest.required_features - SUPPORTED_FEATURES
    if unsupported:
        raise PackCompatibilityError(
            f"pack {manifest.pack_id!r} requires unsupported feature(s): "
            f"{sorted(unsupported)}"
        )


def load_pack(
    path: str | Path,
    *,
    enable_semantic: Optional[bool] = None,
    require_models: Optional[bool] = None,
) -> LanguagePack:
    """Load, validate, and return a `LanguagePack`.

    Args:
      path: pack directory (contains `pack.json`).
      enable_semantic: force the semantic stage on/off, overriding env + config.
      require_models: if True, a missing model artifact is a hard error; if False,
        a warning. Default: True for `production` packs, else False.
    """
    root = Path(path)
    if not root.is_dir():
        raise PackResourceError(f"pack path is not a directory: {root}")

    manifest_path = root / "pack.json"
    if not manifest_path.is_file():
        raise PackManifestError(f"no pack.json in {root}")
    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise PackManifestError(f"pack.json is not valid JSON: {e}") from e

    manifest = PackManifest.parse(manifest_data)
    _check_compatibility(manifest)

    # config.json — optional file; sensible defaults when absent.
    config: dict[str, Any] = {}
    config_path = root / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise PackResourceError(f"config.json is not valid JSON: {e}") from e
    if not isinstance(config, dict):
        raise PackResourceError("config.json must be a JSON object")

    issues: list[str] = []
    if require_models is None:
        require_models = manifest.channel == "production"

    # -- resource tables --------------------------------------------------- #
    resources: dict[str, Any] = {}
    for component, rel in _RESOURCE_FILES.items():
        if component not in manifest.components:
            continue
        fpath = root / rel
        if not fpath.is_file():
            msg = f"declared component {component!r} missing resource file {rel!r}"
            if manifest.channel == "production":
                raise PackResourceError(msg)
            issues.append(msg)
            continue
        try:
            resources[component] = json.loads(fpath.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise PackResourceError(f"{rel} is not valid JSON: {e}") from e

    # -- datetime grammar table (part of entity_extractor) ----------------- #
    # The datetime vocabulary lives at <pack>/datetime/grammar.json. Loaded into
    # resources["datetime"] so the engine can inject it into the entity extractor.
    if "entity_extractor" in manifest.components:
        dt_path = root / "datetime" / "grammar.json"
        if dt_path.is_file():
            try:
                resources["datetime"] = json.loads(dt_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                raise PackResourceError(f"datetime/grammar.json is not valid JSON: {e}") from e
        elif manifest.channel == "production":
            raise PackResourceError("entity_extractor declared but datetime/grammar.json missing")
        else:
            issues.append("datetime/grammar.json missing (entity_extractor declared)")

    # -- model artifacts (paths only; not loaded here) --------------------- #
    model_paths: dict[str, Path] = {}

    def _resolve_artifact(kind: str, rel: Optional[str]) -> None:
        if not rel:
            return
        p = root / rel
        model_paths[kind] = p
        if not p.exists():
            msg = f"declared model {kind!r} missing artifact {rel!r}"
            if require_models:
                raise PackResourceError(msg)
            issues.append(msg)

    intent = manifest.models.get("intent", {}) if isinstance(manifest.models, dict) else {}
    if isinstance(intent, dict):
        _resolve_artifact("intent_model", intent.get("artifact"))
        _resolve_artifact("intent_labels", intent.get("labels"))
        _resolve_artifact("intent_weights", intent.get("weights"))
        _resolve_artifact("intent_calibration", intent.get("calibration"))

    # -- semantic stage (OFF by default; not resolved when off) ------------ #
    semantic_enabled, source = _resolve_semantic_enabled(enable_semantic, config)
    if semantic_enabled and not manifest.semantic_declared:
        # Enabling a stage this pack does not ship. An EXPLICIT per-pack request
        # (constructor arg) is a clear misconfiguration → hard error. A broad
        # switch (env/config) should not crash packs that simply lack the stage
        # → warn and leave it unavailable.
        if source == "arg":
            raise PackCompatibilityError(
                f"semantic stage requested (enable_semantic=True) but pack "
                f"{manifest.pack_id!r} declares none"
            )
        issues.append(
            f"semantic enabled via {source} but pack declares no semantic stage; "
            f"stage left unavailable"
        )
        semantic_enabled = False
    if manifest.semantic_declared and semantic_enabled:
        sem = manifest.models.get("semantic", {}) if isinstance(manifest.models, dict) else {}
        if isinstance(sem, dict):
            _resolve_artifact("semantic_encoder", sem.get("encoder"))
            _resolve_artifact("semantic_head", sem.get("head"))
            _resolve_artifact("semantic_vocab", sem.get("vocab"))
    # When disabled: semantic artifacts are intentionally NOT resolved or required.

    return LanguagePack(
        root=root,
        manifest=manifest,
        config=config,
        resources=resources,
        model_paths=model_paths,
        semantic_enabled=semantic_enabled,
        issues=issues,
    )
