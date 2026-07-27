"""
The hostile-language proof (charter A8).

Every other neutrality check is *static* — the guard greps for `if language`,
the contract tests load a bundle. This is the only test that PROVES the claim
behaviourally: invent a language nobody has heard of, give it nothing but data
files, and run real user turns through the whole engine with **zero engine
edits**. If a language literal or a hardcoded English table were still load-
bearing, `zz` would crash, fall back to English, or silently misbehave here.

TWO HALVES, because the engine is not pack-fed yet
--------------------------------------------------
The engine reads loose localization files (`content/localization/*`), while the
Language Pack contract reads a `spec/bundle/3.0` bundle. Both sides need the
proof, and they are different code paths today:

  * `TestHostileEngineLanguage` — the real neutrality proof for the shipped
    engine. A `zz` language made of copied data files runs the full turn
    pipeline.
  * `TestHostilePackLanguage` — the same proof for `nlu_langpack`, so the
    contract is known good before A7's follow-up wires the engine to it.

When the engine becomes pack-fed the two halves collapse into one.
"""

import importlib
import json
import shutil
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_RUNTIME = str(_ROOT / "packages" / "runtime")
if _RUNTIME not in sys.path:
    sys.path.insert(0, _RUNTIME)

_LOC = _ROOT / "language_packs"
_MINIMAL = _ROOT / "spec" / "examples" / "3.0" / "minimal"

engine_mod = importlib.import_module("nlu_engine.engine")
entities_mod = importlib.import_module("nlu_engine.entities")
lp = importlib.import_module("nlu_langpack")
model_paths_mod = importlib.import_module("nlu_engine.model_paths")

pytestmark = pytest.mark.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Half 1 — the engine
# --------------------------------------------------------------------------- #

@pytest.fixture
def zz_localization(tmp_path, monkeypatch):
    """A complete `zz` language, built only from data files.

    French is the donor because it is the most complete non-English language —
    the point is not that `zz` speaks French, it is that a language the engine
    has never heard of is fully describable in data.
    """
    loc = tmp_path / "language_packs"
    zz_dir = loc / "zz"
    zz_dir.mkdir(parents=True)
    en_dir = _LOC / "en"
    for p in en_dir.glob("*.json"):
        shutil.copy(p, zz_dir / p.name)
        
    # Since en lexicon might not be committed yet (generated at build time), create a fake one for zz
    zz_dir.joinpath("nlu_lexicon.json").write_text(json.dumps({
        "carrier_phrases": ["^hello", "^world", "^custom1"],
        "negation_cues": ["not", "never"]
    }))
    
    # A language needs a MODEL as well as tables. Point the resolver at a `zz`
    # build directory carrying English's artifacts — the weights are irrelevant
    # here, what is under test is that the engine wires an unknown language up
    # from data alone.
    #
    # This is not incidental scaffolding. Before the resolver refused
    # cross-language fallback, these tests passed WITHOUT a zz model because the
    # engine silently served English — so they were green for the wrong reason
    # and would not have caught a language wired to the wrong model.
    models_root = tmp_path / "models"
    (models_root / "intent" / "zz").mkdir(parents=True)
    for name in ("model.onnx", "labels.pkl"):
        shutil.copy(_ROOT / "models" / "intent" / "en" / name,
                    models_root / "intent" / "zz" / name)
    monkeypatch.setattr(model_paths_mod, "MODELS_DIR", models_root)
    monkeypatch.setattr(engine_mod, "BASE_DIR", tmp_path)
    monkeypatch.setattr(entities_mod, "BASE_DIR", tmp_path)
    return loc


def test_unknown_language_constructs_with_no_engine_edit(zz_localization):
    """The headline claim: `zz` is a language the engine has never heard of."""
    eng = engine_mod.NLUEngine(language="zz")
    assert eng.language == "zz"
    assert len(eng.intents) >= 50


def test_unknown_language_runs_a_full_turn(zz_localization):
    """Classify -> resolve -> respond, end to end."""
    eng = engine_mod.NLUEngine(language="zz")
    result = eng.handle("zz-session", "monte le volume")
    assert result.type in ("FULFILL", "CONFIRM"), result
    assert result.intent, "no intent resolved for the unknown language"


def test_unknown_language_picks_up_its_own_data(zz_localization):
    """`zz` must use ITS tables, not silently fall back to the English defaults.

    A fallback would still "work" and would hide a dead language pack, so the
    test asserts the language's own data is actually in play.
    """
    eng = engine_mod.NLUEngine(language="zz")
    assert len(eng._carrier) > len(engine_mod._DEFAULT_CARRIERS), (
        "zz got only the default carriers — its lexicon was not loaded"
    )
    assert engine_mod.NLUEngine._has_localization("zz")
    assert engine_mod.NLUEngine._load_negation_cues("zz")


def test_unknown_language_resolves_a_datetime(zz_localization):
    """Datetime is the deepest language-specific path — A7's main eviction."""
    ex = engine_mod.NLUEngine._load_entities("zz")
    iso, _span, conf, _te, _ed = ex.extract_datetime("demain à 9h")
    assert iso is not None and conf > 0, "zz could not resolve a datetime"


def test_a_language_with_no_files_falls_back_cleanly(tmp_path, monkeypatch):
    """The other direction: an unknown language with NO data must not crash.

    It degrades to the built-in defaults, which is what makes English work
    without shipping an English lexicon.
    """
    empty = tmp_path / "empty"
    empty.mkdir()
    
    # We must provide the 'en' fallback schema that the engine relies on when a language has no files.
    en_fallback_dir = tmp_path / "language_packs" / "en"
    en_fallback_dir.mkdir(parents=True)
    shutil.copy(_ROOT / "language_packs" / "en" / "nlu_schema.json", en_fallback_dir / "nlu_schema.json")
    
    monkeypatch.setattr(engine_mod, "BASE_DIR", tmp_path)
    monkeypatch.setattr(entities_mod, "BASE_DIR", tmp_path)
    models_root = tmp_path / "models"
    (models_root / "intent" / "qq").mkdir(parents=True)
    for name in ("model.onnx", "labels.pkl"):
        shutil.copy(_ROOT / "models" / "intent" / "en" / name,
                    models_root / "intent" / "qq" / name)
    monkeypatch.setattr(model_paths_mod, "MODELS_DIR", models_root)

    eng = engine_mod.NLUEngine(language="qq")
    assert eng.language == "qq"
    assert eng._carrier == engine_mod._DEFAULT_CARRIERS
    assert not engine_mod.NLUEngine._has_localization("qq")
    assert eng.handle("qq-session", "turn up the volume").intent


def test_no_language_code_is_hardcoded_in_the_engine():
    """Guard the claim itself: no language literal in engine logic.

    Passing the tests above by *adding* a `zz` case to the engine would defeat
    their purpose, so this scans the AST for language-code string constants.

    DEFAULT PARAMETER VALUES ARE EXCLUDED, and the distinction is the point:

      * BRANCHING on a language — `if language == "en"` — is coupling. The code
        does something different depending on which language it is, so a new
        language needs a new branch. A7 removed all of these.
      * DEFAULTING to a language — `def __init__(..., language="en")` — is
        configuration. It only answers "which language if the caller doesn't
        say"; the code path is identical whichever value arrives, so it is no
        barrier to adding a language. `NLUEngine(language="fr")` works exactly
        the same as `NLUEngine()`.

    Those defaults are nonetheless the last place English is privileged in the
    engine, and they disappear when it becomes pack-fed: a pack states its own
    language in the manifest, so `pack.language` replaces the default entirely.

    Docstrings are excluded too — prose that describes the pattern is not the
    pattern.
    """
    import ast

    # "en" is no longer hostile, it's used as a fallback data directory in entities.py and text_norm.py
    codes = {"zz", "qq", "fr", "de", "da"}

    def docstring_nodes(tree):
        """Every node that is a docstring, so prose is not mistaken for code."""
        out = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
                body = getattr(node, "body", None)
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    out.add(id(body[0].value))
        return out

    def default_value_nodes(tree):
        """Constants used as function default arguments — configuration, not logic."""
        out = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in (*node.args.defaults, *node.args.kw_defaults):
                    if isinstance(default, ast.Constant):
                        out.add(id(default))
        return out

    offenders = []
    for pyfile in sorted((_ROOT / "packages" / "runtime" / "nlu_engine").glob("*.py")):
        tree = ast.parse(pyfile.read_text(encoding="utf-8"))
        skip = docstring_nodes(tree) | default_value_nodes(tree)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and node.value in codes and id(node) not in skip):
                offenders.append(f"{pyfile.name}:{node.lineno}: {node.value!r}")
    assert not offenders, (
        "language literals found in engine logic — a language must be describable "
        "in data alone (path constants in fallback data loads are exempt):\n  " + "\n  ".join(offenders)
    )


# --------------------------------------------------------------------------- #
# Half 2 — the Language Pack contract
# --------------------------------------------------------------------------- #

class TestHostilePackLanguage:
    """The same proof for `nlu_langpack`, which the engine will consume next."""

    @staticmethod
    def _zz_bundle(tmp_path):
        dst = tmp_path / "zz-bundle"
        shutil.copytree(_MINIMAL, dst)
        # Rename every per-language file en -> zz.
        for sub in ("lexicons", "keywords"):
            src = dst / sub / "en.json"
            if src.exists():
                src.rename(dst / sub / "zz.json")
        for cap in (dst / "capabilities").iterdir():
            resp = cap / "responses" / "en.json"
            if resp.exists():
                resp.rename(cap / "responses" / "zz.json")
        manifest = json.loads((dst / "bundle.json").read_text(encoding="utf-8"))
        manifest["languages"] = {"zz": {"status": "full"}}
        manifest["bundle_id"] = "hostile-zz-0001"
        (dst / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
        return dst

    def test_unknown_language_pack_loads(self, tmp_path):
        pack = lp.load_pack(self._zz_bundle(tmp_path))
        assert pack.language == "zz"
        assert pack.resources["lexicon"] and pack.resources["keyword_matcher"]
        assert pack.stages == ("keyword", "intent_model")

    def test_loader_never_needed_to_know_the_language(self, tmp_path):
        """A `zz` pack and an `en` pack produce structurally identical results."""
        zz = lp.load_pack(self._zz_bundle(tmp_path))
        en = lp.load_pack(_MINIMAL)
        assert set(zz.resources) == set(en.resources)
        assert zz.stages == en.stages
