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
        from nlu import engine  # path wired by conftest.py
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
