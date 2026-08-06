"""The decision ladder — one threshold, two outcomes, no confidence-driven asking.

    conf >= confidence_threshold   ->  the intent fires
    conf <  confidence_threshold   ->  the fallback intent

That is the whole routing contract. It replaces `test_confirm_gate.py`, which
tested a mechanism that no longer exists: `uncertain_confirm`, a confidence band
from 0.55 to 0.91 over a hand-curated 14-intent list.

WHY THE BAND WAS REMOVED RATHER THAN RETUNED
--------------------------------------------
It sat ABOVE the fire threshold, so it converted commands that would have fired
into questions. On the honest holdout it produced 103 friction turns against 16
useful catches — 85% of every confirmation a user saw was asked about a CORRECT
prediction, and "increase volume" was held for confirmation while the model
scored it 0.9992.

The product never asked for it either. Dialogflow, which this replaces, matches
an intent above its ML classification threshold and falls back below it, full
stop; confirmation there is authored dialogue via follow-up intents. This
codebase's own `legacy_label_map.json` records exactly one such dialogue —
`Cmd.SendMessage - yes/no`.

Note the old suite contained `test_band_sits_above_the_fire_threshold`, which
asserted `below_confidence > confidence_threshold`. It pinned the defect as an
invariant. Its replacement here asserts the opposite.

See docs/confirm-gate-diagnosis.md and docs/confirm-gate-remediation-plan.md.
"""

import csv
import json
import sys
import warnings
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in ("packages/buildtime", "packages/runtime"):
    if str(_ROOT / _p) not in sys.path:
        sys.path.insert(0, str(_ROOT / _p))

SCHEMA = json.loads((_ROOT / "language_packs" / "en" / "nlu_schema.json")
                    .read_text(encoding="utf-8"))
_MODEL = _ROOT / "models" / "intent" / "en" / "model.onnx"
_HOLDOUT = _ROOT / "language_packs" / "en" / "holdout_honest.csv"
LANGS = ("fr", "de", "da")

# Raised from 5 when the confirmation band was removed. The band was catching 16
# wrong predictions on this holdout; without it they reach the action. That was a
# deliberate trade — it also removed 103 friction turns and let 100+ more correct
# commands run without an extra turn — and the number is recorded here rather
# than quietly absorbed, because it is the cost side of that decision.
#
# Lower it by improving the MODEL or the fire threshold, never by reintroducing a
# confidence-triggered ask.
WRONG_ACTION_BUDGET = 13


# ------------------------------ ladder shape --------------------------------

def test_there_is_exactly_one_fire_threshold():
    assert "confidence_threshold" in SCHEMA
    assert "slot_confidence_threshold" not in SCHEMA, (
        "a second threshold is back. Slot-bearing intents used to fire at 0.50 "
        "on the reasoning that a prompt resolves ambiguity first — but a flow "
        "whose slots are all filled by the classifying utterance completes "
        "immediately, so the lower bar applied to a live action.")


def test_no_confidence_band_survives_in_the_schema():
    uc = SCHEMA.get("uncertain_confirm", {})
    for dead in ("below_confidence", "confirm_floor", "intents"):
        assert dead not in uc, (
            f"uncertain_confirm.{dead} is back. Confirmation must be authored "
            f"per intent as a `followup`, never triggered by a confidence band.")


def test_the_only_declared_confirmations_are_authored_followups():
    """A `followup` is a product decision; a band is a classifier artifact."""
    authored = {i for i, cfg in SCHEMA["intents"].items() if cfg.get("followup")}
    assert authored == {"messaging.message.send"}, (
        f"authored confirmations are {sorted(authored)}. Adding one is a "
        f"deliberate product call — sending a message is the one irreversible, "
        f"externally-visible action in this taxonomy.")


@pytest.mark.parametrize("field", ["context", "prompt", "yes", "no"])
def test_the_authored_followup_is_complete(field):
    fu = SCHEMA["intents"]["messaging.message.send"]["followup"]
    assert fu.get(field) is not None, f"followup is missing {field!r}"


@pytest.mark.parametrize("lang", LANGS)
def test_authored_followups_are_localized(lang):
    path = _ROOT / "language_packs" / lang / "nlu_schema.json"
    if not path.exists():
        pytest.skip(f"no localization for {lang}")
    loc = json.loads(path.read_text(encoding="utf-8")).get("intents", {})
    send = loc.get("messaging.message.send", {})
    assert send.get("confirm_prompt") or send.get("followup"), (
        f"{lang} has no localized send-confirmation prompt")


# ------------------------------ behaviour -----------------------------------

@pytest.fixture(scope="module")
def engine():
    pytest.importorskip("onnxruntime")
    if not _MODEL.exists():
        pytest.skip("trained English model absent")
    from nlu_engine import NLUEngine
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return NLUEngine(model_name="en", language="en", semantic_enabled=False)


def test_a_confident_command_never_asks(engine):
    engine.reset("ladder-1")
    r = engine.handle("ladder-1", "increase volume")
    assert r.type == "FULFILL"
    assert r.confidence >= SCHEMA["confidence_threshold"]


def test_a_low_confidence_turn_falls_back_rather_than_asking(engine):
    """Below the threshold there is no middle tier — it is the fallback intent.

    "can you to us number one hits" classified as device.memory.change at 0.519
    and "one" filled its memory slot, so the flow completed on entry and the
    hearing-aid program changed. It reported confidence 1.0, the slot-fill
    certainty rather than the intent's, which is what hid it.
    """
    engine.reset("ladder-2")
    r = engine.handle("ladder-2", "can you to us number one hits")
    assert r.type == "FALLBACK", (
        f"returned {r.type}/{r.intent} at {r.confidence}; a sub-threshold "
        f"slot-bearing intent must not complete on entry")


def test_an_authored_followup_confirms_regardless_of_confidence(engine):
    """Determinism is the point — the app contract cannot depend on the model.

    `Cmd.SendMessage - yes` used to be reachable only when the classifier
    happened to land inside the old uncertainty band, so the dialogue act
    appeared for "send a message" and vanished for "send a message to john".
    """
    engine.reset("send-1")
    first = engine.handle("send-1", "send a message")
    assert first.type == "CONFIRM"

    engine.reset("send-2")
    confident = engine.handle("send-2", "send a message to john")
    assert confident.type == "CONFIRM", (
        "a high-confidence send skipped its authored confirmation")


def test_yes_completes_the_authored_flow_and_no_cancels_it(engine):
    engine.reset("send-yes")
    engine.handle("send-yes", "send a message")
    done = engine.handle("send-yes", "yes")
    assert done.type == "FULFILL" and done.action

    engine.reset("send-no")
    engine.handle("send-no", "send a message")
    stopped = engine.handle("send-no", "no")
    assert stopped.type == "FULFILL" and not stopped.action, (
        "declining a send must not carry an action")


def test_a_different_command_escapes_the_confirmation(engine):
    """A user who changes their mind must not be held in the confirmation.

    `_handle_confirmation` re-asked on any non-yes/no reply AND re-set its own
    context, so the lifespan never counted down: "send a message" followed by
    "increase volume" asked about sending a message forever. The branch was
    unreachable while no intent declared a `followup`; giving one to
    messaging.message.send made it live.

    A confident different command now interrupts, exactly as it does mid
    slot-filling.
    """
    engine.reset("escape")
    assert engine.handle("escape", "send a message").type == "CONFIRM"
    r = engine.handle("escape", "increase volume")
    assert r.type == "FULFILL" and r.intent == "device.volume.increase", (r.type, r.intent)
    assert r.interrupted_intent == "messaging.message.send"


def test_repeated_non_answers_route_out_instead_of_holding(engine):
    """A user who cannot be understood is let go, not asked forever."""
    engine.reset("stuck")
    assert engine.handle("stuck", "send a message").type == "CONFIRM"
    seen = [engine.handle("stuck", m).type for m in ("hmm", "uhh", "errr")]
    assert seen[-1] == "FALLBACK", (
        f"still confirming after {len(seen)} non-answers: {seen}")


def test_the_legacy_compound_labels_still_resolve(engine, monkeypatch):
    """The app contract this followup exists to keep alive."""
    from nlu_engine import label_compat
    monkeypatch.setenv("NLU_LEGACY_LABELS", "1")
    label_compat._load.cache_clear()
    try:
        engine.reset("legacy")
        engine.handle("legacy", "send a message")
        assert engine.handle("legacy", "yes").intent == "Cmd.SendMessage - yes"
        engine.reset("legacy2")
        engine.handle("legacy2", "send a message")
        assert engine.handle("legacy2", "no").intent == "Cmd.SendMessage - no"
    finally:
        label_compat._load.cache_clear()


# ------------------------------- the budget ---------------------------------

@pytest.mark.skipif(not (_MODEL.exists() and _HOLDOUT.exists()),
                    reason="trained English artifacts or honest holdout absent")
def test_wrong_action_budget_is_met_on_the_honest_holdout():
    """Replays the frozen honest holdout through the full engine.

    This is the number Review-F5 blocker B1 tracks. It is one side of a
    two-sided trade — see `test_friction_budget_is_met_on_the_honest_holdout`
    below, which exists because budgeting only this side is how the ladder
    reached 85% friction with a green suite.
    """
    pytest.importorskip("onnxruntime")
    from nlu_engine import NLUEngine
    from nlu_training.wrong_action_harness import is_actionable, is_read_only

    with _HOLDOUT.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eng = NLUEngine(model_name="en", language="en", semantic_enabled=False)
        wrong = []
        for i, row in enumerate(rows):
            r = eng.handle(f"budget-{i}", row["text"])
            intent = r.intent or ""
            if (r.type == "FULFILL" and intent != row["intent"]
                    and is_actionable(intent) and not is_read_only(intent)):
                wrong.append((row["text"], row["intent"], intent))

    assert len(wrong) <= WRONG_ACTION_BUDGET, (
        f"{len(wrong)} wrong actions on {len(rows)} honest turns, budget is "
        f"{WRONG_ACTION_BUDGET}:\n" +
        "\n".join(f"  {t!r}: {truth} -> {fired}" for t, truth, fired in wrong))


# A confirmation shown for a CORRECT prediction is pure cost: an extra turn for
# a command that was already understood. With the band gone the only remaining
# source is the authored send-message followup, so this should stay small. If it
# climbs, a confidence-driven ask has crept back in.
FRICTION_BUDGET = 40


@pytest.mark.skipif(not (_MODEL.exists() and _HOLDOUT.exists()),
                    reason="trained English artifacts or honest holdout absent")
def test_friction_budget_is_met_on_the_honest_holdout():
    pytest.importorskip("onnxruntime")
    from nlu_engine import NLUEngine

    with _HOLDOUT.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eng = NLUEngine(model_name="en", language="en", semantic_enabled=False)
        friction = [row["text"] for i, row in enumerate(rows)
                    if (r := eng.handle(f"friction-{i}", row["text"])).type == "CONFIRM"
                    and r.intent == row["intent"]]

    assert len(friction) <= FRICTION_BUDGET, (
        f"{len(friction)} correct predictions were confirmed instead of fired "
        f"(budget {FRICTION_BUDGET}). Examples: {friction[:5]}")
