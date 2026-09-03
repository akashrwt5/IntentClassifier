"""
`load_pack()` — validate a single-language bundle and hand the engine a
`LanguagePack`.

Load sequence, failing loudly and early:
  1. Read `bundle.json`; parse + validate the manifest.
  2. Compatibility gate against the runtime contract version.
  3. Select the language and check its completeness for the channel.
  4. Load `config.json` (runtime knobs) if present.
  5. Resolve + presence-check resource tables and model artifacts. Missing
     artifacts are a hard error on `production`, an issue on `dev`/`beta`, so
     skeleton bundles stay loadable while iterating.
  6. Resolve the semantic stage — OFF BY DEFAULT — and, when off, do not
     resolve or require its model at all.

This module contains NO language-specific logic and NO `if language == …`
branches: a bundle for any language code loads through exactly this path.

Signature and checksum verification are deliberately NOT reimplemented here.
`nlu_compiler.verify` and `BundleManager` already do that with a 3-gate
verifier, tamper detection and downgrade refusal; a second, weaker trust check
in the loader would be worse than none.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from .errors import PackLanguageError, PackManifestError, PackResourceError
from .manifest import BundleManifest
from .pack import LanguagePack
from .version import check_compatibility

__all__ = ["load_pack"]

# Per-language resource tables, as laid out by spec/bundle/3.0 (ADR-005 Part 2)
# and verified against the golden bundles in spec/examples/3.0/.
# `{lang}` is substituted with the selected language.
_RESOURCE_FILES: dict[str, str] = {
    "lexicon":         "lexicons/{lang}.json",
    "keyword_matcher": "keywords/{lang}.json",
}

# Language-neutral runtime tables — one copy per bundle, not per language.
#
# `routing` is NOT here. It was, and it was `strict`-required, which means the
# moment the compiler stopped fabricating `runtime/routing.json` this loader
# would have refused every new pack over a table nothing ever read. The spec
# still defines the section; when a pack genuinely carries one it can come back
# — with a consumer.
_RUNTIME_TABLES = ("cascade", "policies", "plan_facts")

# Optional per-language tables. Absent from today's golden bundles: the datetime
# grammar arrives with the A7 eviction, and normalizer rules are not yet
# extracted. Their absence is recorded as an issue, never an error, so this
# loader does not have to change when they land.
_OPTIONAL_FILES: dict[str, str] = {
    "datetime":   "system/datetime/{lang}.json",
    "normalizer": "normalizer/{lang}.json",
}

# Model kinds keyed by language inside `manifest.models`. The value under a
# language (or "shared") is an object carrying `artifact`, not a bare path.
_SHARED_MODEL_KEY = "shared"

# Sibling artifacts that live beside a model but are not declared in the
# manifest's model spec. `calibration.json` matters most: the temperature is a
# property of the (model, featurizer) pair and must travel with the model, which
# is what Review-F5 blocker B8 got wrong.
_MODEL_SIBLINGS = ("calibration.json", "labels.json")


def _resolve_semantic_enabled(override: Optional[bool],
                              config: dict[str, Any]) -> tuple[bool, str]:
    """Resolve the semantic stage's enabled state and WHERE it was decided.

    Precedence: constructor arg -> NLU_ENABLE_SEMANTIC env -> config
    `semantic_enabled` -> default False. The source matters: an explicit arg
    requesting a stage the bundle lacks is a misconfiguration worth crashing
    over, while a broad env/config switch should not break bundles that simply
    have no semantic head.
    """
    if override is not None:
        return bool(override), "arg"
    env = os.environ.get("NLU_ENABLE_SEMANTIC")
    if env is not None and env.strip() != "":
        return env.strip().lower() in ("1", "true", "yes", "on"), "env"
    if "semantic_enabled" in config:
        return bool(config["semantic_enabled"]), "config"
    return False, "default"


def load_pack(bundle_dir: str | Path, *, language: Optional[str] = None,
              enable_semantic: Optional[bool] = None) -> LanguagePack:
    """Load one language out of an unpacked `spec/bundle/3.0` bundle."""
    root = Path(bundle_dir)
    manifest_path = root / "bundle.json"
    if not manifest_path.exists():
        raise PackManifestError(f"no bundle.json at {root}")

    try:
        manifest = BundleManifest.parse(json.loads(
            manifest_path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise PackManifestError(f"bundle.json is not valid JSON: {exc}") from exc

    issues: list[str] = list(check_compatibility(
        manifest.engine_compat.min_runtime_contract,
        manifest.engine_compat.max_tested_runtime_contract,
        manifest.required_runtime_features,
    ))

    lang = language or manifest.single_language()
    if lang not in manifest.languages:
        raise PackLanguageError(
            f"bundle declares {sorted(manifest.languages)}; {lang!r} is not among them")

    status = manifest.language_status(lang)
    if status != "full":
        message = (f"language {lang!r} is {status!r} in bundle {manifest.bundle_id!r}")
        if manifest.channel == "production":
            # Mirrors compiler stage 9: partial completeness never ships.
            raise PackLanguageError(message + " — not allowed on channel=production")
        issues.append(message)

    config_path = root / "config.json"
    config: dict[str, Any] = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PackResourceError(f"config.json is not valid JSON: {exc}") from exc

    strict = manifest.channel == "production"

    resources: dict[str, Any] = {}

    def _read(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PackResourceError(
                f"{path.relative_to(root)} is not valid JSON: {exc}") from exc

    # Required per-language tables.
    for name, tpl in _RESOURCE_FILES.items():
        path = root / tpl.format(lang=lang)
        if not path.exists():
            msg = f"resource {name!r} missing for language {lang!r}: {tpl.format(lang=lang)}"
            if strict:
                raise PackResourceError(msg)
            issues.append(msg)
            continue
        resources[name] = _read(path)

    # Optional per-language tables (see _OPTIONAL_FILES).
    for name, tpl in _OPTIONAL_FILES.items():
        path = root / tpl.format(lang=lang)
        if path.exists():
            resources[name] = _read(path)
        else:
            issues.append(f"optional resource {name!r} not present for {lang!r}")

    # Language-neutral runtime tables.
    for name in _RUNTIME_TABLES:
        path = root / "runtime" / f"{name}.json"
        if not path.exists():
            msg = f"runtime table missing: runtime/{name}.json"
            if strict:
                raise PackResourceError(msg)
            issues.append(msg)
            continue
        resources[name] = _read(path)

    # Capabilities: workflows and entities are partitioned per capability
    # (ADR-002 A4), so assemble them into one view keyed by capability id.
    caps: dict[str, dict[str, Any]] = {}
    cap_root = root / "capabilities"
    if cap_root.is_dir():
        for cap_dir in sorted(p for p in cap_root.iterdir() if p.is_dir()):
            entry: dict[str, Any] = {}
            for part in ("capability", "workflows", "entities"):
                part_path = cap_dir / f"{part}.json"
                if part_path.exists():
                    entry[part] = _read(part_path)
            responses = cap_dir / "responses" / f"{lang}.json"
            if responses.exists():
                entry["responses"] = _read(responses)
            elif strict:
                raise PackResourceError(
                    f"capability {cap_dir.name!r} has no responses for {lang!r}")
            else:
                issues.append(
                    f"capability {cap_dir.name!r} has no responses for {lang!r}")
            caps[cap_dir.name] = entry
    if caps:
        resources["capabilities"] = caps
    elif strict:
        raise PackResourceError("bundle declares no capabilities")

    # `semantic_head` is the manifest's name for the optional stage-3 model.
    semantic_declared = "semantic_head" in manifest.models
    want_semantic, source = _resolve_semantic_enabled(enable_semantic, config)
    if want_semantic and not semantic_declared:
        if source == "arg":
            raise PackResourceError(
                f"semantic stage explicitly requested but bundle "
                f"{manifest.bundle_id!r} declares no semantic model")
        issues.append("semantic stage requested but this bundle declares none; "
                      "stage left unavailable")
        want_semantic = False

    # manifest.models is {kind: {lang|"shared": {artifact, format, ...}}}.
    model_paths: dict[str, Path] = {}
    for kind, per_key in manifest.models.items():
        # A disabled semantic stage is not resolved at all, so it costs nothing
        # and cannot be influenced by a stale artifact.
        if kind == "semantic_head" and not want_semantic:
            continue
        if not isinstance(per_key, dict):
            issues.append(f"models.{kind} is not an object; ignored")
            continue
        for key, spec in per_key.items():
            if key not in (lang, _SHARED_MODEL_KEY):
                continue  # another language's artifact
            if not isinstance(spec, dict) or not isinstance(spec.get("artifact"), str):
                issues.append(f"models.{kind}.{key} declares no artifact path")
                continue
            candidate = root / spec["artifact"]
            if not candidate.exists():
                msg = f"model artifact missing: {kind}.{key} -> {spec['artifact']}"
                if strict:
                    raise PackResourceError(msg)
                issues.append(msg)
                continue
            model_paths[f"{kind}.{key}"] = candidate
            # Undeclared siblings that ship beside the artifact.
            for sibling in _MODEL_SIBLINGS:
                sib = candidate.parent / sibling
                if sib.exists():
                    model_paths[f"{kind}.{key}.{sib.stem}"] = sib

    semantic_available = want_semantic and any(
        k.startswith("semantic_head.") for k in model_paths)

    return LanguagePack(
        root=root,
        language=lang,
        manifest=manifest,
        config=config,
        resources=resources,
        model_paths=model_paths,
        semantic_available=semantic_available,
        issues=issues,
    )
