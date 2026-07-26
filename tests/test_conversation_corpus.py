"""Golden conversation corpus runner (roadmap Phase 5: grow to 200+ scripts).

Each tests/conversations/*.yaml script is a multi-turn conversation replayed
through the real engine. Assertions per turn:

  expect:
    type: FULFILL|PROMPT|CONFIRM|FALLBACK   result type
    intent: x / not_intent: x               exact / forbidden intent
    intent_prefix: help.                    intent family
    action: x (null = no action)            exact action
    not_action: x / not_action_prefix: v.   forbidden action (family)

Script options: `lang` (engine model+language), `force_confirm: true`
(lowers nothing — raises the confirm bar to 1.01 so the uncertainty gate
always engages; used to test the gate flow deterministically).

Semantic rescue is disabled (stale local artifacts — known-issues.md).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml", reason="dev dependency (pyyaml)")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages" / "runtime"))

CORPUS = sorted((REPO_ROOT / "tests" / "conversations").glob("*.yaml"))
MODELS = REPO_ROOT / "models" / "intent"

_ENGINES: dict[str, object] = {}


def _engine(lang: str):
    if not (MODELS / lang / f"{lang}_intent_model.onnx").exists():
        pytest.skip(f"trained artifacts for '{lang}' not present")
    pytest.importorskip("onnxruntime")
    if lang not in _ENGINES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from nlu_engine import NLUEngine

            eng = NLUEngine(model_name=lang, language=lang, semantic_enabled=False)
        _ENGINES[lang] = eng
    return _ENGINES[lang]


def _check(expect: dict, r, where: str):
    got = (f"type={r.type} intent={r.intent} action={r.action} "
           f"conf={r.confidence:.2f} msg={r.message!r}")
    if "type" in expect:
        assert r.type == expect["type"], f"{where}: want type {expect['type']}; got {got}"
    if "intent" in expect:
        assert r.intent == expect["intent"], f"{where}: want intent {expect['intent']}; got {got}"
    if "not_intent" in expect:
        assert r.intent != expect["not_intent"], f"{where}: forbidden intent fired; got {got}"
    if "intent_prefix" in expect:
        assert (r.intent or "").startswith(expect["intent_prefix"]), f"{where}: got {got}"
    if "action" in expect:
        assert r.action == expect["action"], f"{where}: want action {expect['action']}; got {got}"
    if "not_action" in expect:
        assert r.action != expect["not_action"], f"{where}: forbidden action fired; got {got}"
    if "not_action_prefix" in expect:
        assert not (r.action or "").startswith(expect["not_action_prefix"]), \
            f"{where}: forbidden action family; got {got}"


@pytest.mark.parametrize("script_path", CORPUS, ids=lambda p: p.stem)
def test_conversation(script_path: Path, monkeypatch):
    script = yaml.safe_load(script_path.read_text(encoding="utf-8"))
    engine = _engine(script.get("lang", "en"))
    if script.get("force_confirm"):
        monkeypatch.setattr(engine, "_confirm_below", 1.01)
    session = f"corpus-{script_path.stem}"
    engine.reset(session)
    for i, turn in enumerate(script["turns"]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = engine.handle(session, turn["user"])
        _check(turn.get("expect", {}), r, f"{script_path.stem} turn {i + 1} ({turn['user']!r})")
