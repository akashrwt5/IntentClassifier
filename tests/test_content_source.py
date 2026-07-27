"""Capability source tree ⇄ compiled schema drift guard (ADR-002 A4).

content/capabilities/ is the editable source; content/nlu_schema.json (+
localization overlays) is the compiled form the engine loads. This test IS
the CI rule that they never disagree: edit source → `python -m
nlu_compiler.content_source assemble` → commit both.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("yaml", reason="dev dependency (pyyaml)")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages" / "buildtime"))

from nlu_compiler.content_source import LANGS, assemble, intent_to_capability  # noqa: E402

CONTENT = REPO_ROOT / "content"


def test_source_tree_matches_compiled_schema():
    built = assemble(write=False)
    current = json.loads((REPO_ROOT / "language_packs" / "en" / "nlu_schema.json").read_text())
    assert built["schema"] == current, (
        "content/capabilities/ and nlu_schema.json disagree — run "
        "`python -m nlu_compiler.content_source assemble` and commit both")
    for lang in LANGS:
        tracked = json.loads(
            (REPO_ROOT / "language_packs" / lang / "nlu_schema.json").read_text())
        assert built["overlays"][lang] == tracked, f"{lang} overlay drift"


def test_every_intent_has_exactly_one_owner():
    owner = intent_to_capability()
    schema = json.loads((REPO_ROOT / "language_packs" / "en" / "nlu_schema.json").read_text())
    assert set(schema["intents"]) == set(owner), "intent set ≠ capability map"


def test_capability_tree_shape():
    caps = sorted(p.name for p in (CONTENT / "capabilities").iterdir() if p.is_dir())
    assert len(caps) == 12, caps
    total = 0
    for cap in caps:
        base = CONTENT / "capabilities" / cap
        assert (base / "capability.yaml").exists()
        import yaml

        manifest = yaml.safe_load((base / "capability.yaml").read_text())
        intents = list((base / "intents").glob("*.yaml"))
        assert len(intents) == len(manifest["intents"]), cap
        total += len(intents)
    assert total == 57
