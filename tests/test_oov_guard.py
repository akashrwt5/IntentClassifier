"""The out-of-vocabulary guard — the words the model cannot see.

A TF-IDF vector is a fixed set of slots, one per known term. A token outside
the vocabulary is not weighed and dismissed; there is nowhere to put it, so the
sentence arrives without it:

    'turn off'          -> 3 non-zero features
    'turn off toshiba'  -> 3 non-zero features, cosine 1.000000, bit-identical

The model is not wrong about "turn off toshiba" — it is never asked. Confidence
is honest about the input it was handed, which is why no threshold, training row
or hyperparameter separates these two. `help me find a paper` reduces to
`help me find`, where `Help_FindMyHearingAids` is the correct reading.

And the word that puts an utterance out of scope is almost always rare and
specific — a brand, an object, a topic — exactly what a finite vocabulary lacks.
Its absence is evidence. The guard stops discarding it.

Measured on holdout_honest.csv: out-of-scope utterances reaching a device action
fall from 10 (5.1%) to 6 (3.1%).

See docs/PRODUCTION_TRACKER.md B1.
"""

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

pytestmark = pytest.mark.skipif(not _MODEL.exists(),
                                reason="trained English model absent")


@pytest.fixture(scope="module")
def engine():
    pytest.importorskip("onnxruntime")
    from nlu_engine import NLUEngine
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return NLUEngine(model_name="en", language="en", semantic_enabled=False)


# ------------------------------ the plumbing --------------------------------

def test_the_ratio_is_configured_and_content_owned():
    r = SCHEMA.get("oov_reject_ratio")
    assert r is not None, "oov_reject_ratio missing; the guard is disabled"
    assert 0.0 < r <= 1.0
    b = SCHEMA.get("oov_bypass_confidence")
    assert b is not None and 0.0 < b <= 1.01, (
        "oov_bypass_confidence missing; without it the guard refuses entity "
        "values — 'send a message to john' is 25% out-of-vocabulary and real")
    assert "oov_reject_ratio" not in (
        _ROOT / "packages" / "runtime" / "nlu_engine" / "engine.py"
    ).read_text(encoding="utf-8").split("self.schema.get")[0], \
        "the ratio must come from the pack, not a literal in the engine"


def test_the_vocabulary_comes_from_the_shipped_graph(engine):
    """Read from the model that classifies, so the two cannot disagree.

    A vocabulary file beside the model would be one more pair able to drift —
    the failure this codebase has already paid for with the device/server
    temperature (B8) and the device/server training subset.
    """
    vocab = engine.classifier.backend.unigram_vocabulary()
    assert len(vocab) > 100, f"vocabulary looks wrong: {len(vocab)} terms"
    for known in ("volume", "hearing", "reminder"):
        assert known in vocab


def test_a_backend_without_a_vocabulary_disables_the_guard(engine):
    """Never reject everything because a backend could not answer."""
    class Mute:
        def tfidf_logits(self, text):
            raise NotImplementedError

    clf = engine.classifier
    original, clf.backend = clf.backend, Mute()
    try:
        assert clf.oov_ratio("something entirely unknown zzzz") == 0.0
    finally:
        clf.backend = original


# ------------------------------- the ratio ----------------------------------

@pytest.mark.parametrize("text", [
    "increase volume", "decrease volume", "mute", "turn up the volume",
    "what's my battery level", "start streaming",
])
def test_in_domain_commands_have_no_unknown_words(engine, text):
    assert engine.classifier.oov_ratio(text) == 0.0, (
        f"{text!r} contains words the model cannot see; either the vocabulary "
        f"regressed or this is no longer an in-domain command")


def test_an_out_of_domain_object_registers(engine):
    assert engine.classifier.oov_ratio("help me find a paper") > 0.0


@pytest.mark.parametrize("text", [
    "send a message to john", "stream from netflix",
])
def test_entity_values_are_not_refused(engine, text):
    """Unknown words that are the VALUE the command operates on.

    A contact name, a brand, a free-text reminder topic can never all be in a
    finite vocabulary, so these carry a real out-of-vocabulary share and are
    real commands. The ratio alone cannot separate them from a foreign topic;
    the confidence can, which is what `oov_bypass_confidence` is for.
    """
    assert engine.classifier.oov_ratio(text) > 0.0, f"fixture drift: {text!r}"
    engine.reset("entity")
    r = engine.handle("entity", text)
    assert r.type != "FALLBACK", (
        f"{text!r} refused at {r.confidence:.3f}; the guard is eating entity values")


# ------------------------------ the behaviour -------------------------------

def test_the_reported_case_no_longer_fires(engine):
    """'help me find a paper' -> Help_FindMyHearingAids.

    The model sees `help me find`; 'paper' appears zero times in train.csv, so
    it has no slot at all and cannot be added by min_df, by the per-intent cap,
    or by any threshold.
    """
    engine.reset("oov-1")
    r = engine.handle("oov-1", "help me find a paper")
    assert r.type == "FALLBACK", (
        f"returned {r.type}/{r.intent} at {r.confidence:.3f}")


def test_the_guard_can_only_withhold_an_action_never_cause_one(engine):
    """It runs before the fire test, so its only power is to refuse."""
    engine.reset("oov-2")
    r = engine.handle("oov-2", "increase volume")
    assert r.type == "FULFILL" and r.intent == "Cmd.VolumeIncrease"


def test_head_commands_are_untouched_by_the_guard(engine):
    """The guard must not tax the commands users actually say."""
    for text, intent in (("increase volume", "Cmd.VolumeIncrease"),
                         ("decrease volume", "Cmd.VolumeDecrease"),
                         ("mute", "Cmd.VolumeMute"),
                         ("volume up", "Cmd.VolumeIncrease")):
        engine.reset("oov-head")
        r = engine.handle("oov-head", text)
        assert r.type == "FULFILL" and r.intent == intent, (text, r.type, r.intent)


# ------------------------------- the budget ---------------------------------

_HOLDOUT = _ROOT / "language_packs" / "en" / "holdout_honest.csv"

# Out-of-scope utterances that still reach a device action, on the frozen
# holdout. This is the B1 blocker's metric. It was 19 (9.7%) before the
# per-intent cap was removed and 10 (5.1%) before this guard.
#
# Lower it with data or vocabulary coverage, not by tightening the ratio until
# real commands stop working — `lost` is the other side of that trade and this
# file deliberately does not budget only one side of it.
OOS_ACTION_BUDGET = 6


@pytest.mark.skipif(not _HOLDOUT.exists(), reason="honest holdout absent")
def test_out_of_scope_action_budget_is_met(engine):
    import csv

    with _HOLDOUT.open(encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["intent"] == "Default Fallback Intent"]

    leaked = []
    for i, row in enumerate(rows):
        engine.reset(f"oos-{i}")
        r = engine.handle(f"oos-{i}", row["text"])
        if r.type == "FULFILL":
            leaked.append((row["text"], r.intent))

    assert len(leaked) <= OOS_ACTION_BUDGET, (
        f"{len(leaked)}/{len(rows)} out-of-scope utterances fired an action "
        f"(budget {OOS_ACTION_BUDGET}):\n"
        + "\n".join(f"  {t!r} -> {i}" for t, i in leaked))
