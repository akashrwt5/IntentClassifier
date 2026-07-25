"""
Neutrality + Language-Pack contract tests (the core of the PR CI guard).

These lock in the requirements the whole restructure delivers:
  - a bare NLUEngine() defaults to the English pack;
  - the semantic stage is OFF by default (opt-in plugin);
  - a fake/hostile language pack runs the FULL pipeline with zero engine edits;
  - the engine contains no `if language` branches.
"""

import importlib.util
import json
import logging
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "packages"))
logging.disable(logging.CRITICAL)

from nlu import NLUEngine  # noqa: E402
from nlu_langpack import load_pack  # noqa: E402


def test_en_pack_loads_and_is_default():
    eng = NLUEngine()  # no language passed → English by default
    assert eng.language == "en"
    assert len(eng.intents) >= 50


def test_semantic_off_by_default_but_enableable():
    assert NLUEngine().semantic is None
    assert NLUEngine().semantic_enabled is False
    # The pack declares the stage, so it can be turned on explicitly.
    assert NLUEngine(enable_semantic=True).semantic is not None


def test_en_pack_validates():
    pack = load_pack(str(_ROOT / "packs" / "en"))
    assert pack.language == "en"
    assert pack.semantic_available is False  # declared, off by default
    assert not pack.issues


def test_hostile_language_pack_runs_end_to_end(tmp_path):
    """A fake 'zz' pack (English tables relabelled) must run the full pipeline
    with NO engine code change — the proof the engine is language-agnostic."""
    zz = tmp_path / "zz"
    shutil.copytree(_ROOT / "packs" / "en", zz)
    manifest = json.loads((zz / "pack.json").read_text())
    manifest["language"] = "zz"
    manifest["pack_id"] = "zz@test"
    (zz / "pack.json").write_text(json.dumps(manifest))

    eng = NLUEngine(pack=str(zz))
    assert eng.language == "zz"
    r = eng.handle("s", "turn up the volume")
    assert r.type == "FULFILL" and r.intent == "Cmd.VolumeIncrease"


def test_engine_has_no_language_branches():
    """Run the standalone CI guard as a test too, so it fails PRs locally."""
    path = _ROOT / "scripts" / "ci" / "check_language_neutral.py"
    spec = importlib.util.spec_from_file_location("check_language_neutral", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main() == 0, "engine contains language-specific branches"


def _clone_en_pack(tmp_path, lang="zz", **patches):
    """Copy packs/en to a temp pack, relabel it, and apply JSON patches:
    patches={"lexicons.json": {...}} is merged into that file."""
    dst = tmp_path / lang
    shutil.copytree(_ROOT / "packs" / "en", dst)
    manifest = json.loads((dst / "pack.json").read_text())
    manifest["language"] = lang
    manifest["pack_id"] = f"{lang}@test"
    (dst / "pack.json").write_text(json.dumps(manifest))
    for fname, patch in patches.items():
        p = dst / fname
        data = json.loads(p.read_text())
        data.update(patch)
        p.write_text(json.dumps(data))
    return dst


# --------------------------- pack-driven negation ---------------------------
# A negation cue list hardcoded in the engine silently makes negation a no-op
# for every non-English pack. These lock it to the pack.

def test_negation_suppresses_a_keyword_hit():
    clf = NLUEngine().classifier
    assert clf._keyword_match("please translate this menu")[0] == "Cmd.TranslationStart"
    # Same rule, negated → must not fire.
    assert clf._keyword_match("i don't want to translate anything")[0] is None


def test_negation_cues_come_from_the_pack(tmp_path):
    """A pack that declares its OWN negation cues is honoured, and the English
    cues do not leak in. This is what makes adding a language pack-only."""
    zz = _clone_en_pack(tmp_path, "zz",
                        **{"lexicons.json": {"negations": ["nixnix"]}})
    clf = NLUEngine(pack=str(zz)).classifier
    assert clf._negations == ("nixnix",)
    # The pack's own cue suppresses.
    assert clf._keyword_match("nixnix translate this")[0] is None
    # An English cue is NOT a negation for this pack, so the rule still fires.
    assert clf._keyword_match("i don't want to translate anything")[0] == "Cmd.TranslationStart"


# ------------------- pack-driven non-interrupting actions -------------------

def test_help_turn_does_not_abandon_a_slot_flow():
    eng = NLUEngine()
    eng.reset("nz1")
    eng.handle("nz1", "set a reminder")
    r = eng.handle("nz1", "ask about the translate feature")
    assert r.interrupted_intent is None, r.interrupted_intent


def test_non_interrupting_actions_come_from_the_pack(tmp_path):
    """Dropping the declaration restores interrupting behaviour — proving the
    guard is pack data, not a rule baked into the engine."""
    zz = _clone_en_pack(tmp_path, "zz",
                        **{"config.json": {"policy": {"non_interrupting_actions": []}}})
    eng = NLUEngine(pack=str(zz))
    assert eng.NON_INTERRUPTING_ACTIONS == ()
    eng.reset("nz2")
    eng.handle("nz2", "set a reminder")
    r = eng.handle("nz2", "ask about the translate feature")
    assert r.interrupted_intent is not None
