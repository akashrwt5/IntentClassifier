"""Semantic rescue — ONE plug-and-play flag.

Resolution order: constructor param → NLU_SEMANTIC_RESCUE env → schema
'semantic_rescue_enabled' (overlay-overridable per language) → default True.
Disabled means the artifacts are never even loaded.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages" / "runtime"))

MODEL = REPO_ROOT / "multilingual" / "models" / "en" / "en_intent_model.onnx"

pytestmark = pytest.mark.skipif(not MODEL.exists(),
                                reason="trained artifacts not present")


def _engine(**kw):
    pytest.importorskip("onnxruntime")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from nlu_engine import NLUEngine

        return NLUEngine(model_name="en", language="en", **kw)


def test_param_off_never_loads_semantic():
    eng = _engine(semantic_enabled=False)
    assert eng.semantic_enabled is False and eng.semantic is None


def test_param_on_loads_semantic_when_artifacts_exist():
    if not (REPO_ROOT / "models" / "semantic_head.npz").exists():
        pytest.skip("semantic head not present")
    eng = _engine(semantic_enabled=True)
    assert eng.semantic_enabled is True and eng.semantic is not None


def test_env_kill_switch(monkeypatch):
    monkeypatch.setenv("NLU_SEMANTIC_RESCUE", "0")
    eng = _engine()
    assert eng.semantic_enabled is False and eng.semantic is None


def test_env_beats_schema_but_param_beats_env(monkeypatch):
    monkeypatch.setenv("NLU_SEMANTIC_RESCUE", "0")
    eng = _engine(semantic_enabled=True)
    assert eng.semantic_enabled is True


def test_schema_flag_is_the_default_source():
    eng = _engine()
    assert eng.semantic_enabled == bool(
        eng.schema.get("semantic_rescue_enabled", True))
