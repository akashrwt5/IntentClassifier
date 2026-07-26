"""The uncertainty-confirmation gate — coverage, prompts, and the budget.

The gate is what converts a wrong prediction into a question instead of a wrong
action. Three things have to hold together for that to work, and each has
already failed once:

  * **Coverage.** Every state-changing intent must be gated. Before this,
    `messaging.message.listen` was not, and it accounted for 3 of the 11 wrong
    actions on the honest holdout — a message read aloud unbidden.
  * **Prompts.** A gated intent with no `confirm_prompt` falls back to a generic
    "should I go ahead with that?", which tells a user nothing about what is
    about to happen. Worse than not gating.
  * **The band.** `below_confidence` was 0.80, fit against a temperature that
    turned out to be wrong (blocker B8). It is now fit out-of-fold by
    `nlu_training.fit_confirm_gate` and must not drift silently.
"""

import csv
import json
import sys
import warnings
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for p in ("packages/buildtime", "packages/runtime"):
    if str(_ROOT / p) not in sys.path:
        sys.path.insert(0, str(_ROOT / p))

SCHEMA = json.loads((_ROOT / "content" / "nlu_schema.json").read_text(encoding="utf-8"))
UC = SCHEMA["uncertain_confirm"]
GATED = set(UC["intents"])
LANGS = ("fr", "de", "da")

# Fit out-of-fold on datasets/en/train.csv at T=0.657336 by
# `python -m nlu_training.fit_confirm_gate --lang en --max-friction 0.15`:
# catches 80.7% of wrong state-changing predictions for 14.1% friction on
# correct ones. Change it there, not here.
EXPECTED_BELOW = 0.9
WRONG_ACTION_BUDGET = 5


def _read_only(label: str) -> bool:
    return label == "device.status.battery" or label.rsplit(".", 1)[-1] == "query"


def state_changing() -> set[str]:
    return {i for i in SCHEMA["intents"]
            if not i.startswith(("help.", "sys.")) and not _read_only(i)}


# ------------------------------- coverage ----------------------------------

def test_every_state_changing_intent_is_gated():
    missing = sorted(state_changing() - GATED)
    assert not missing, (
        f"state-changing intents outside the confirmation gate: {missing}. "
        f"An ungated intent fires on a wrong prediction with no way for the "
        f"user to stop it.")


def test_gate_does_not_cover_read_only_intents():
    """Read-only intents change nothing; confirming them is pure friction."""
    wrong = sorted(i for i in GATED if i.startswith(("help.", "sys.")) or _read_only(i))
    assert not wrong, f"read-only intents should not be gated: {wrong}"


# -------------------------------- prompts ----------------------------------

def test_every_gated_intent_has_an_english_prompt():
    missing = sorted(i for i in GATED
                     if not SCHEMA["intents"].get(i, {}).get("confirm_prompt"))
    assert not missing, (
        f"gated intents with no confirm_prompt: {missing}. They would fall back "
        f"to the generic prompt, which does not say what is about to happen.")


@pytest.mark.parametrize("lang", LANGS)
def test_every_gated_intent_has_a_localized_prompt(lang):
    path = _ROOT / "content" / "localization" / f"nlu_schema.{lang}.json"
    if not path.exists():
        pytest.skip(f"no localization for {lang}")
    loc = json.loads(path.read_text(encoding="utf-8")).get("intents", {})
    missing = sorted(i for i in GATED if not loc.get(i, {}).get("confirm_prompt"))
    assert not missing, f"gated intents with no {lang} confirm_prompt: {missing}"


# --------------------------------- band ------------------------------------

def test_confirmation_band_matches_the_fitted_value():
    assert UC["below_confidence"] == EXPECTED_BELOW, (
        f"below_confidence is {UC['below_confidence']}, expected "
        f"{EXPECTED_BELOW} from the out-of-fold fit. Re-fit with "
        f"`python -m nlu_training.fit_confirm_gate --lang en` and update "
        f"EXPECTED_BELOW deliberately — never tune it against the holdout.")


def test_band_sits_above_the_fire_threshold():
    """A band below the fire threshold would gate nothing that fires."""
    assert UC["below_confidence"] > SCHEMA["confidence_threshold"]
    assert UC["confirm_floor"] < SCHEMA["confidence_threshold"]


# ------------------------- the budget, end to end ---------------------------

_MODEL = _ROOT / "models" / "intent" / "en" / "model.onnx"
_HOLDOUT = _ROOT / "datasets" / "en" / "holdout_honest.csv"


@pytest.mark.skipif(not (_MODEL.exists() and _HOLDOUT.exists()),
                    reason="trained English artifacts or honest holdout absent")
def test_wrong_action_budget_is_met_on_the_honest_holdout():
    """The measurement the whole gate exists to satisfy.

    Replays the frozen honest holdout through the full engine, semantic rescue
    off. This is the number Review-F5 blocker B1 tracks; it was 11 before the
    gate was extended to every state-changing intent.
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
