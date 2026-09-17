"""The featurizer normalisation is a CONTRACT, not a helper.

`nlu_engine.text_norm.normalize_text` decides the vocabulary every Stage-2
artifact is fitted on — the ONNX graph, the pruned device weights, the CoreML
linear head and the TFLite head all descend from one `TfidfVectorizer` fitted on
its output. A change here that no test notices is a silent train/inference
divergence: the model keeps predicting *something*, no assertion fails, and the
accuracy moves.

These tests pin the transform itself. They do not need a trained model.
"""

import json
from pathlib import Path

import pytest

from nlu_engine.text_norm import normalize_text

REPO = Path(__file__).resolve().parents[1]
PACK = REPO / "language_packs" / "en"


# --------------------------------------------------------------------------- #
# contractions — the original reason this module exists
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw, expected", [
    ("what's up", "what is up"),
    ("What's Up", "what is up"),
    ("don't mute it", "do not mute it"),
    ("can't hear you", "cannot hear you"),
    ("mom's reminder", "moms reminder"),      # possessive: apostrophe dropped
    ("turn up the volume", "turn up the volume"),
])
def test_contraction_expansion(raw, expected):
    assert normalize_text(raw) == expected


def test_no_apostrophe_survives():
    """skl2onnx's tokenizer does not share Python's \\w semantics around the
    apostrophe, so an apostrophe reaching the vectorizer means the pickle and
    the exported ONNX disagree. None may survive."""
    for raw in ("what's up", "o'clock", "mom's", "y'all", "it’s late"):
        assert "'" not in normalize_text(raw)
        assert "’" not in normalize_text(raw)


# --------------------------------------------------------------------------- #
# plural folding
# --------------------------------------------------------------------------- #

def test_folding_is_off_when_the_table_is_empty():
    assert normalize_text("change programs", lemmas={}) == "change programs"


def test_folding_applies_the_table():
    t = {"programs": "program", "memories": "memory"}
    assert normalize_text("change programs", lemmas=t) == "change program"
    assert normalize_text("switch my memories", lemmas=t) == "switch my memory"


def test_folding_is_idempotent():
    """The map points at forms that are themselves unfoldable, so a second pass
    is a no-op. If that ever stops holding, a re-normalised corpus would drift
    away from the vocabulary it was fitted on."""
    t = {"programs": "program", "memories": "memory"}
    once = normalize_text("change programs and memories", lemmas=t)
    assert normalize_text(once, lemmas=t) == once


def test_folding_respects_word_boundaries():
    """A substring match would rewrite the inside of longer words."""
    t = {"programs": "program"}
    assert normalize_text("deprogramsettings", lemmas=t) == "deprogramsettings"


def test_folding_runs_after_contraction_expansion():
    """Order matters: expansion introduces words the table may need to see, and
    the table must not be applied to a form that still carries an apostrophe."""
    t = {"programs": "program"}
    assert normalize_text("what's my programs", lemmas=t) == "what is my program"


def test_longest_key_wins():
    """Built longest-first, so a key that prefixes another cannot shadow it."""
    t = {"program": "prog", "programs": "program"}
    assert normalize_text("my programs", lemmas=t) == "my program"


# --------------------------------------------------------------------------- #
# the shipped pack tables
# --------------------------------------------------------------------------- #

def test_pack_tables_are_flat_maps():
    """`evaluate.py` once read `.get("contractions", {})` on this file and so ran
    with expansion silently disabled. The shape is flat; keep it that way."""
    conts = json.loads((PACK / "contractions.json").read_text(encoding="utf-8"))
    assert conts and all(isinstance(v, str) for v in conts.values())
    lemmas_path = PACK / "lemmas.json"
    if lemmas_path.exists():
        lemmas = json.loads(lemmas_path.read_text(encoding="utf-8"))
        assert all(isinstance(v, str) for v in lemmas.values())


def test_shipped_lemmas_never_fold_onto_themselves():
    lemmas_path = PACK / "lemmas.json"
    if not lemmas_path.exists():
        pytest.skip("no lemma table for en")
    lemmas = json.loads(lemmas_path.read_text(encoding="utf-8"))
    assert not [k for k, v in lemmas.items() if k == v]


def test_shipped_lemmas_have_no_chains():
    """A value that is itself a key would make the transform order-dependent and
    break idempotency."""
    lemmas_path = PACK / "lemmas.json"
    if not lemmas_path.exists():
        pytest.skip("no lemma table for en")
    lemmas = json.loads(lemmas_path.read_text(encoding="utf-8"))
    assert not [v for v in lemmas.values() if v in lemmas]

