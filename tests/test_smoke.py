"""Lightweight smoke tests — fast, dependency-light guards that the repo's
core artifacts are present and internally consistent. These do not exercise the
model; they catch structural breakage (missing config, malformed calibration,
un-importable leaf modules) in a couple of milliseconds.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_key_project_files_exist():
    for rel in [
        "README.md",
        "CLAUDE.md",
        "CONTRIBUTING.md",
        "pyproject.toml",
        "Makefile",
        "requirements.txt",
        "config/calibration.json",
        "data/nlu_schema.json",
    ]:
        assert (REPO_ROOT / rel).exists(), f"missing expected file: {rel}"


def test_calibration_config_is_valid():
    cfg = json.loads((REPO_ROOT / "config" / "calibration.json").read_text())
    assert set(["en", "fr", "de", "da"]).issubset(cfg.keys())
    for lang, entry in cfg.items():
        assert isinstance(entry["temperature"], (int, float)), lang
        assert entry["temperature"] > 0, lang
        assert 0.0 <= entry["conf_threshold"] <= 1.0, lang


def _load_leaf_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_entity_extractor_loads_and_runs():
    """entities.py is self-contained (no heavy deps) and must import + run.

    Uses the real API: EntityExtractor(entities_path, ...) with
    extract_datetime(text) / extract(entity, text).
    """
    mod = _load_leaf_module("nlu_entities_smoke", "scripts/nlu/entities.py")
    assert hasattr(mod, "EntityExtractor")
    extractor = mod.EntityExtractor()  # default (English) lexicon
    assert callable(extractor.extract_datetime)
    # Should not raise on a trivial utterance; a plain phrase yields no datetime.
    # extract_datetime returns a tuple whose first element is the parsed value
    # (None when nothing matches).
    result = extractor.extract_datetime("hello there")
    assert isinstance(result, tuple)
    assert result[0] is None


@pytest.mark.integration
def test_nlu_package_importable_if_deps_present():
    """Full engine import — skipped gracefully if runtime deps are absent."""
    try:
        import nlu  # noqa: F401  (path wired by conftest.py)
    except ImportError as exc:
        pytest.skip(f"runtime dependency missing, skipping full-engine import: {exc}")
    assert hasattr(nlu, "NLUEngine")
