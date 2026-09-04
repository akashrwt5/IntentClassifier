"""A confirmation branch states everything its outcome needs, in one place.

For a while it took three files to describe what "yes" does to a send:

    content YAML                 action + text          (reference engine only)
    legacy_label_map.json        the compound label     (reference engine only)
    workflows.json               neither                (the device)

The device therefore had to INFER the outcome, and VoiceAIKit's guess was
`completion.action` for yes and an empty string for no. That guess was correct
until content authored a decline with its own action, at which point the same
utterance on the same pack fired `message.cancel` in Python and nothing on iOS.
The label had no route to a device at all without a fourth artifact.

`workflows.confirmation.yes/no` now carries the action, the response key and the
optional host label, so both runtimes read the same three facts from the same
place. These tests hold that shut from both ends: the bundle must carry them,
and the engine must report what they say.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_BUNDLE = _ROOT / "dist" / "bundle-en"
_RUNTIME = _ROOT / "packages" / "runtime"
if str(_RUNTIME) not in sys.path:
    sys.path.insert(0, str(_RUNTIME))

SCHEMA = json.loads(
    (_ROOT / "language_packs" / "en" / "nlu_schema.json").read_text(encoding="utf-8")
)

# Every intent content gives a followup. Derived, so a second gated intent is
# covered the day it is authored rather than the day someone remembers.
CONFIRMED = sorted(i for i, cfg in SCHEMA["intents"].items() if cfg.get("followup"))


def test_at_least_one_intent_is_confirmed():
    """The premise. Without it every parametrised test below vacuously passes."""
    assert CONFIRMED, "no intent authors a followup — these tests would prove nothing"


@pytest.mark.parametrize("intent", CONFIRMED)
@pytest.mark.parametrize("polarity", ["yes", "no"])
def test_content_authors_an_action_for_each_branch(intent, polarity):
    branch = SCHEMA["intents"][intent]["followup"][polarity]
    assert branch.get("action"), (
        f"{intent}.followup.{polarity} has no action, so the client must invent one — "
        f"which is how the two runtimes diverged"
    )


@pytest.mark.parametrize("intent", CONFIRMED)
def test_the_branches_do_different_things(intent):
    fu = SCHEMA["intents"][intent]["followup"]
    assert fu["yes"]["action"] != fu["no"]["action"], (
        f"{intent}: accepting and declining fire the same action, so the answer " f"does not matter"
    )


@pytest.mark.parametrize("intent", CONFIRMED)
def test_a_label_if_present_is_not_a_model_label(intent):
    """`label` is a dialogue-act name, and the head can never emit one.

    If a compound label ever appeared in the trained label space, the engine must
    stop synthesising it — two paths producing one string with different
    confidences is worse than either alone.
    """
    labels_path = _BUNDLE / "models" / "intent" / "en" / "labels.json"
    if not labels_path.exists():
        pytest.skip("built bundle required")
    trained = set(json.loads(labels_path.read_text(encoding="utf-8")))
    fu = SCHEMA["intents"][intent]["followup"]
    for polarity in ("yes", "no"):
        label = fu[polarity].get("label")
        if label is None:
            continue
        assert label not in trained, f"{label} is a classifier label"
        assert label not in SCHEMA["intents"], f"{label} is an intent id"


# --------------------------------------------------------------------------- #
# The bundle
# --------------------------------------------------------------------------- #

pytest_bundle = pytest.mark.skipif(
    not _BUNDLE.exists(), reason="built bundle (dist/bundle-en) required"
)


@pytest_bundle
@pytest.mark.parametrize("intent", CONFIRMED)
def test_the_bundle_carries_the_branches(intent):
    cap = SCHEMA["intents"][intent].get("capability") or _capability_of(intent)
    wf = json.loads(
        (_BUNDLE / "capabilities" / cap / "workflows.json").read_text(encoding="utf-8")
    )["intents"][intent]
    conf = wf.get("confirmation")
    assert conf, f"{intent} asks in content but the bundle's workflow does not"

    responses = json.loads(
        (_BUNDLE / "capabilities" / cap / "responses" / "en.json").read_text(encoding="utf-8")
    )
    actions = {
        a["key"]
        for a in json.loads(
            (_BUNDLE / "capabilities" / cap / "capability.json").read_text(encoding="utf-8")
        )["actions"]
    }

    for polarity in ("yes", "no"):
        branch = conf.get(polarity)
        assert branch, (
            f"{intent}: the bundle states that it asks but not what {polarity!r} does — "
            f"a client can only guess, and two clients guess differently"
        )
        assert branch["action"] in actions, (
            f"{branch['action']} is not declared in capability.json, so it reaches a "
            f"device as a key no capability owns"
        )
        assert responses.get(branch["response"]), f"{branch['response']} resolves to no text"
        authored = SCHEMA["intents"][intent]["followup"][polarity]
        assert branch["action"] == authored["action"]
        assert responses[branch["response"]] == authored["fulfillment"]
        assert branch.get("label") == authored.get("label")


@pytest_bundle
def test_the_superseded_artifacts_are_gone():
    for stale in ("runtime/legacy_labels.json", "runtime/confirmation_labels.json"):
        assert not (_BUNDLE / stale).exists(), (
            f"{stale} is back. A confirmation outcome is one fact; stating half of it "
            f"in a second artifact is what these branches replaced."
        )


def _capability_of(intent: str) -> str:
    for cap_dir in (_BUNDLE / "capabilities").iterdir():
        wf = cap_dir / "workflows.json"
        if wf.exists() and intent in json.loads(wf.read_text(encoding="utf-8"))["intents"]:
            return cap_dir.name
    raise AssertionError(f"{intent} is in no capability")


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #


@pytest_bundle
@pytest.mark.parametrize(
    "reply,polarity",
    [
        ("yes", "yes"),
        ("alright", "yes"),
        ("go ahead", "yes"),
        ("no", "no"),
        ("leave message", "no"),
        ("changed my mind", "no"),
    ],
)
def test_the_engine_reports_what_the_branch_says(reply, polarity):
    """End to end, with no shim and no environment flag.

    `label_compat` used to do this, opt-in behind `NLU_LEGACY_LABELS=1`, which
    meant the reference engine's DEFAULT output disagreed with what every device
    would report. Reading the label off the branch removes the flag from the
    question entirely.
    """
    from nlu_engine import NLUEngine
    from nlu_langpack import load_pack

    engine = NLUEngine(pack=load_pack(str(_BUNDLE)))
    session = f"branch-{reply.replace(' ', '-')}"

    asked = engine.handle(session, "send a message to mom")
    assert asked.type == "CONFIRM", f"no confirmation to resolve: {asked.type}"

    resolved = engine.handle(session, reply)
    branch = SCHEMA["intents"]["Cmd.SendMessage"]["followup"][polarity]

    assert resolved.type == "FULFILL"
    assert resolved.action == branch["action"], f"{reply!r} fired the wrong branch"
    assert resolved.message == branch["fulfillment"]
    assert resolved.intent == (branch.get("label") or "Cmd.SendMessage")
