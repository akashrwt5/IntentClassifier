"""The content->bundle compiler: does the pack describe THIS product?

`release-pack.yml` used to build from `spec/examples/3.0/minimal` — a golden test
fixture with ONE capability and TWO intents (`audio.volume.mute`,
`audio.volume.set`). The published pack-en-v1.0.0 therefore carried the fixture's
capabilities, keywords, lexicons and routing with our 57-class model dropped in.
It validated only because `labels.json` still held the fixture's two labels.

These tests assert the compiled bundle is real: 12 capabilities, 57 intents, our
fitted thresholds, and a tensor contract matching the actual model. The strongest
one is at the bottom — the pack must load through the Language Pack contract.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for p in ("packages/buildtime", "packages/runtime"):
    if str(_ROOT / p) not in sys.path:
        sys.path.insert(0, str(_ROOT / p))

SCHEMA = json.loads((_ROOT / "language_packs" / "en" / "nlu_schema.json").read_text(encoding="utf-8"))
MODEL_DIR = _ROOT / "models" / "intent" / "en"
_HAVE_MODEL = (MODEL_DIR / "model.onnx").exists() and (MODEL_DIR / "labels.pkl").exists()

pytestmark = pytest.mark.skipif(
    not _HAVE_MODEL, reason="trained English artifacts absent")


@pytest.fixture(scope="module")
def report_card(tmp_path_factory):
    """A real evaluation. The compiler refuses to build without one (ADR-005
    stage 13: a bundle cannot be built from an evaluation that didn't run)."""
    out = tmp_path_factory.mktemp("rc") / "report_card.json"
    proc = subprocess.run(
        [sys.executable, "-m", "nlu_training.evaluate", "--langs", "en",
         "--out", str(out)],
        cwd=_ROOT, capture_output=True, text=True,
        env={**__import__("os").environ,
             "PYTHONPATH": f"{_ROOT/'packages/buildtime'}:{_ROOT/'packages/runtime'}"})
    if not out.exists():
        pytest.skip(f"evaluate did not produce a report card: {proc.stderr[-400:]}")
    return out


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, report_card):
    from nlu_compiler.content_bundle import compile_bundle
    out = tmp_path_factory.mktemp("bundle") / "bundle-en"
    rc = compile_bundle("en", out, MODEL_DIR, report_card, "1.0.0", "dev", False)
    assert rc == 0, "compile_bundle failed"
    return out


def _j(bundle: Path, rel: str):
    return json.loads((bundle / rel).read_text(encoding="utf-8"))


# ------------------------- it is the real content ---------------------------

def test_carries_every_capability_and_intent(bundle):
    caps = sorted(p.parent.name for p in bundle.glob("capabilities/*/capability.json"))
    content_caps = sorted(d.name for d in (_ROOT / "content" / "capabilities").iterdir()
                          if d.is_dir())
    assert caps == content_caps, "compiled capabilities differ from content/"
    assert len(caps) >= 12, f"only {len(caps)} capabilities — fixture-sized"

    mapped = _j(bundle, "runtime/plan_facts.json")["intents"]
    assert set(mapped) == set(SCHEMA["intents"]), (
        "plan_facts must map every intent in the schema to a capability")
    assert len(mapped) == 57


def test_label_space_and_tensor_contract_agree_with_the_model(bundle):
    """The published pack declared `dim: 2` semantics beside a 57-class graph."""
    labels = _j(bundle, "models/intent/en/labels.json")
    tfidf = next(s for s in _j(bundle, "runtime/cascade.json")["stages"]
                 if s["id"] == "tfidf")
    assert len(labels) == 57
    assert tfidf["output"]["dim"] == len(labels), (
        "cascade output dim must equal the label count — stage 8 probes the "
        "model via ORT and compares")


def test_policies_carry_our_fitted_thresholds_not_the_fixtures(bundle):
    """Carrying the fixture would have shipped interrupt: 0.75 — the stale value
    replaced when interrupt_threshold was fitted out-of-fold."""
    th = _j(bundle, "runtime/policies.json")["thresholds"]
    assert th["confidence"] == SCHEMA["confidence_threshold"]
    assert th["interrupt"] == SCHEMA["interrupt_threshold"]
    assert th["semantic"] == SCHEMA["semantic_threshold"]


def test_confirmation_policy_matches_the_authored_followups(bundle):
    """`always` for a declared followup, `never` otherwise — and no third state.

    This asserted that `when_ambiguous` matched the 14-intent uncertainty-gate
    list. That gate was removed (docs/confirm-gate-diagnosis.md): it sat above
    the fire threshold and turned commands that would have fired into questions.
    Confirmation is now something an intent DECLARES, so it is unconditional.
    """
    conf = _j(bundle, "runtime/policies.json")["confirmation"]
    authored = {i for i, c in SCHEMA["intents"].items() if c.get("followup")}
    assert set(conf) == set(SCHEMA["intents"]), "every intent needs a policy"
    assert {i for i, v in conf.items() if v == "always"} == authored
    assert "when_ambiguous" not in set(conf.values()), (
        "the confidence band is back in the bundle — a runtime would need "
        "`uncertain_confirm_below` to interpret it, and there is none")
    assert all(v in ("always", "never") for v in conf.values())


def test_routing_ladder_uses_our_fire_threshold(bundle):
    """routing.json is a carried template, but this one value must be re-derived
    or the pack escalates at a confidence the engine never uses."""
    ladder = _j(bundle, "runtime/routing.json")["ladder"]
    step = next(s for s in ladder if "below_confidence" in (s.get("when") or {}))
    assert step["when"]["below_confidence"] == SCHEMA["confidence_threshold"]


# --------------------------- shape translation -----------------------------

def test_identifiers_are_translated_to_the_spec_grammar(bundle):
    """content/ predates the spec: `sys.date-time` has a hyphen (illegal in a
    stableId) and slots are CamelCase (the spec requires ^[a-z][a-z0-9_]*$)."""
    ents = _j(bundle, "entities/shared/content.json")["entities"]
    assert "sys.date_time" in ents and "sys.date-time" not in ents
    assert all("-" not in e for e in ents)

    wf = _j(bundle, "capabilities/device.memory/workflows.json")
    slots = wf["intents"]["device.memory.change"]["slots"]
    assert [s["name"] for s in slots] == ["memory_name"], "slot name not snake_cased"


def test_prompt_text_moves_into_responses_and_is_referenced_by_key(bundle):
    """The spec keeps TEXT out of logic: workflows hold response KEYS."""
    wf = _j(bundle, "capabilities/device.memory/workflows.json")
    slot = wf["intents"]["device.memory.change"]["slots"][0]
    responses = _j(bundle, "capabilities/device.memory/responses/en.json")
    assert slot["prompt"] in responses, "slot prompt key has no response"
    assert " " not in slot["prompt"], "the prompt is literal text, not a key"
    assert responses[slot["prompt"]].endswith("?")


def test_every_response_key_referenced_actually_exists(bundle):
    """A dangling key is a silent blank prompt on device."""
    missing = []
    for wf_path in bundle.glob("capabilities/*/workflows.json"):
        cap = wf_path.parent.name
        responses = _j(bundle, f"capabilities/{cap}/responses/en.json")
        for intent, cfg in _j(bundle, f"capabilities/{cap}/workflows.json")["intents"].items():
            keys = [cfg["completion"]["response"]]
            keys += [s["prompt"] for s in cfg.get("slots", [])]
            if "confirmation" in cfg:
                keys.append(cfg["confirmation"]["prompt"])
            missing += [(intent, k) for k in keys if k not in responses]
    assert not missing, f"response keys with no text: {missing[:5]}"


def test_every_completion_action_is_declared_by_its_capability(bundle):
    """Cross-capability action references are a compile error (ADR-002 A9)."""
    bad = []
    for wf_path in bundle.glob("capabilities/*/workflows.json"):
        cap = wf_path.parent.name
        declared = {a["key"] for a in
                    _j(bundle, f"capabilities/{cap}/capability.json")["actions"]}
        for intent, cfg in _j(bundle, f"capabilities/{cap}/workflows.json")["intents"].items():
            if cfg["completion"]["action"] not in declared:
                bad.append((cap, intent, cfg["completion"]["action"]))
    assert not bad, f"completion actions not declared by their capability: {bad[:5]}"


def test_shipped_patterns_conform_to_the_portable_regex_subset(bundle):
    """A pattern outside the subset may behave differently in Swift or Kotlin."""
    from nlu_compiler.portable_regex import check_pattern
    bad = []
    for rule in _j(bundle, "keywords/en.json")["rules"]:
        for pat in [rule["pattern"], *rule.get("guards", [])]:
            errs = check_pattern(pat)
            if errs:
                bad.append((pat, errs))
    for pat in _j(bundle, "lexicons/en.json").get("carriers", []):
        if check_pattern(pat):
            bad.append((pat, check_pattern(pat)))
    assert not bad, f"non-portable patterns shipped: {bad[:3]}"


# ---------------------------- refusals & gates ------------------------------

def test_refuses_to_build_without_a_real_report_card(tmp_path):
    """A stub would compile into a signed artifact asserting unmeasured metrics."""
    from nlu_compiler.content_bundle import compile_bundle
    with pytest.raises(SystemExit):
        compile_bundle("en", tmp_path / "b", MODEL_DIR, None, "1.0.0", "dev", False)


def test_the_compiled_bundle_passes_the_compiler_and_verifies(bundle, tmp_path):
    """The whole point: compile -> validate -> sign -> verify, no fixture."""
    nlu = tmp_path / "pack-en-v1.0.0.nlu"
    build = subprocess.run(
        [sys.executable, "-m", "nlu_compiler.build", str(bundle),
         "--out", str(nlu), "--channel", "dev"],
        cwd=_ROOT, capture_output=True, text=True,
        env={**__import__("os").environ,
             "PYTHONPATH": str(_ROOT / "packages" / "buildtime")})
    assert build.returncode == 0, build.stdout + build.stderr
    assert "ERROR" not in build.stdout

    verify = subprocess.run(
        [sys.executable, "-m", "nlu_compiler.verify", str(nlu)],
        cwd=_ROOT, capture_output=True, text=True,
        env={**__import__("os").environ,
             "PYTHONPATH": str(_ROOT / "packages" / "buildtime")})
    assert verify.returncode == 0, verify.stdout + verify.stderr
    assert "VERIFIED" in verify.stdout


def test_the_compiled_pack_loads_through_the_language_pack_contract(bundle):
    """A bundle that validates but cannot be LOADED is not a language pack."""
    import nlu_langpack
    pack = nlu_langpack.load_pack(bundle)
    assert pack.language == "en"
    assert pack.channel == "dev"
