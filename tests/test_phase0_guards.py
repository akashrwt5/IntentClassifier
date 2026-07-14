"""Phase-0 safety guards (Review-F5 Appendix A #1, #5).

Fast, dependency-light tests that the quarantine and startup guards actually
guard: auto_label.py must refuse to run, and the NLU engine must reject the
placeholder GenAI URL instead of silently shipping it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_auto_label_is_quarantined():
    """auto_label.py exits nonzero with the quarantine banner and touches nothing."""
    training = REPO_ROOT / "data" / "01_source_base_training_data.csv"
    before = training.read_bytes() if training.exists() else None
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "auto_label.py")],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode != 0, "quarantined script must refuse to run"
    assert "QUARANTINED" in (proc.stdout + proc.stderr)
    after = training.read_bytes() if training.exists() else None
    assert before == after, "quarantined script must not modify training data"


def _engine_module():
    try:
        from nlu_engine import engine  # path wired by conftest.py
    except ImportError as exc:
        pytest.skip(f"runtime dependency missing: {exc}")
    return engine


def test_genai_guard_rejects_placeholder():
    engine = _engine_module()
    with pytest.raises(RuntimeError, match="placeholder GenAI URL"):
        engine.NLUEngine._resolve_genai_url(engine.DEFAULT_GENAI_URL)


def test_genai_guard_disables_when_unconfigured():
    engine = _engine_module()
    assert engine.NLUEngine._resolve_genai_url(None) is None


def test_genai_guard_passes_real_url_through():
    engine = _engine_module()
    url = "https://genai.example-real-endpoint.com/chat?query="
    assert engine.NLUEngine._resolve_genai_url(url) == url


# --- unknown-data privacy enforcement (Appendix A #10, ND-5) -----------------

def _unknown_log():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "unknown_log_test", REPO_ROOT / "apps" / "cli" / "unknown_log.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_unknown_default_writes_counters_never_text(tmp_path, monkeypatch):
    ul = _unknown_log()
    monkeypatch.delenv(ul.RAW_OPT_IN_ENV, raising=False)
    counters, raw = tmp_path / "counters.csv", tmp_path / "raw.csv"
    assert ul.record_unknown("private medical utterance", 0.42, counters, raw) == "counter"
    assert ul.record_unknown("another one", 0.48, counters, raw) == "counter"
    assert not raw.exists(), "raw text must not be written without opt-in"
    content = counters.read_text()
    assert "private medical utterance" not in content
    assert "0.4-0.5" in content
    # both turns landed in the same day+bucket counter
    assert any(line.endswith(",2") for line in content.strip().splitlines())


def test_unknown_raw_requires_explicit_opt_in(tmp_path, monkeypatch):
    ul = _unknown_log()
    monkeypatch.setenv(ul.RAW_OPT_IN_ENV, "1")
    counters, raw = tmp_path / "counters.csv", tmp_path / "raw.csv"
    assert ul.record_unknown("consented text", 0.3, counters, raw) == "raw"
    assert "consented text" in raw.read_text()
    assert not counters.exists()


def test_confidence_bucket_bounds():
    ul = _unknown_log()
    assert ul.confidence_bucket(0.0) == "0.0-0.1"
    assert ul.confidence_bucket(0.55) == "0.5-0.6"
    assert ul.confidence_bucket(1.0) == "0.9-1.0"
