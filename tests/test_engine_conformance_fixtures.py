"""Self-conformance: the committed executable-spec fixtures match the live engine.

These are the SAME files native iOS/Android CI replays. If this test fails,
someone changed dialogue behavior without regenerating fixtures — which is
exactly the review moment the mechanism exists to force (regenerate via
`python -m nlu_training.generate_conformance_fixtures`, review the diff,
commit both; the diff IS the cross-platform behavior change).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("yaml")
pytest.importorskip("onnxruntime")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages" / "buildtime"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "runtime"))

FIXTURE_DIR = REPO_ROOT / "tests" / "parity" / "engine_conformance"
MODEL = REPO_ROOT / "models" / "intent" / "en" / "model.onnx"


@pytest.mark.skipif(not MODEL.exists(), reason="trained artifacts not present")
def test_conversation_fixtures_match_live_engine():
    """Committed conversation traces must match the live engine.

    SKIPPED while any fixture language lacks a model. The fixtures were recorded
    when fr/de/da models lived in the retired `multilingual/models/` tree; those
    languages now have no model until their datasets are supplied and trained
    (owner decision: each Language Pack brings its own data). The generator
    builds an engine per language up front, so it cannot produce a partial
    replay — filtering after the fact is too late.

    This is deliberately a SKIP, not a narrowed assertion: silently comparing
    only English would look like full parity coverage while three languages went
    unchecked. It re-arms by itself once datasets/<lang>/ is trained.
    """
    from nlu_training.generate_conformance_fixtures import conversations_fixture
    from nlu_engine.model_paths import available_languages

    committed = json.loads((FIXTURE_DIR / "conversations.json").read_text())
    needed = {v.get("lang") for v in committed["scripts"].values() if v.get("lang")}
    missing = sorted(needed - set(available_languages()))
    if missing:
        pytest.skip(f"fixtures cover {sorted(needed)}; no model for {missing}. "
                    f"Train with: python -m nlu_training.train --lang <lang>")

    assert conversations_fixture() == committed["scripts"], (
        "engine behavior changed — regenerate the conformance fixtures and "
        "review the diff (python -m nlu_training.generate_conformance_fixtures)")


def test_datetime_clock_grid_matches_live_extractor():
    from nlu_training.generate_conformance_fixtures import (
        datetime_clock_grid_fixture)

    committed = json.loads((FIXTURE_DIR / "datetime_clock_grid.json").read_text())
    live = datetime_clock_grid_fixture()
    assert live == committed["rows"], (
        "datetime grammar behavior changed across the clock grid — "
        "regenerate fixtures and review the diff")
