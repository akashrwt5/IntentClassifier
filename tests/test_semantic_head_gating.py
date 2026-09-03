"""The semantic head ships only when the stage that uses it is on.

Regression test for a build that went red on CI and could not go red locally,
which is the only reason it got as far as CI.

`compile_models` used to decide whether to ship a stage-3 head by asking "does
`semantic_head.json` exist beside the trained model?". It does not exist in a
fresh clone and it DOES exist on a build machine, so the file was copied for the
first time in CI and the bundle validator saw it for the first time there. It
failed three ways at once:

    stage 1  SCHEMA_INVALID        'embedder' unexpected / 'embedder_id' required
    stage 8  EMBEDDER_ID_MISMATCH  head 'onnx' vs manifest 'minilm-l6-v2'
    stage 8  HEAD_LABEL_MISMATCH   57 labels; stage 8 wants the intents plus OOS

Every one of those is the validator being right. `train_semantic_head` writes
`embedder: "onnx"` to record WHICH EMBED PATH BUILT THE HEAD -- its own comment
says so -- and that is not what `embedder_id` means; the schema defines it as
the tie to "the exact encoder+vocab pair it was trained against", whose mismatch
is the silent-wrong-vector-space bug class. And the head carries the 57
classifier labels with no OOS class, which is not the label set a stage-3 head
is specified to have.

So the head as trained today is not shippable, and the stage that would consume
it is off. The question the compiler must ask is whether the STAGE is enabled,
not whether a file happens to be lying around.

Deliberately builds its own stub model directory instead of using the repo's
trained artifacts, so it runs in a fresh clone too. A test that can only fail on
the machine where the bug already shipped is not a regression test.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in ("packages/buildtime", "packages/runtime"):
    if str(_ROOT / _p) not in sys.path:
        sys.path.insert(0, str(_ROOT / _p))

SCHEMA = json.loads(
    (_ROOT / "language_packs" / "en" / "nlu_schema.json").read_text(encoding="utf-8"))


@pytest.fixture
def stub_model_dir(tmp_path: Path) -> Path:
    """A models/ tree shaped like a real training output, with a semantic head."""
    joblib = pytest.importorskip("joblib")

    intent_dir = tmp_path / "models" / "intent" / "en"
    intent_dir.mkdir(parents=True)
    joblib.dump(sorted(SCHEMA["intents"]), intent_dir / "labels.pkl")
    (intent_dir / "model.onnx").write_bytes(b"stub onnx")
    (intent_dir / "calibration.json").write_text(json.dumps({
        "temperature": 0.671457, "conf_threshold": 0.7,
        "temperature_coreml": 0.822109, "temperature_coreml_full": 0.54399,
    }), encoding="utf-8")

    # Exactly the shape train_semantic_head writes: `embedder`, not `embedder_id`,
    # and no OOS class. Both are why this head cannot ship as it stands.
    (tmp_path / "models" / "semantic_head.json").write_text(json.dumps({
        "embedder": "onnx",
        "labels": sorted(SCHEMA["intents"]),
        "weights": [[0.0]] * len(SCHEMA["intents"]),
        "bias": [0.0] * len(SCHEMA["intents"]),
    }), encoding="utf-8")
    pkg = tmp_path / "models" / "SemanticHead.mlpackage"
    pkg.mkdir()
    (pkg / "stub").write_text("x", encoding="utf-8")

    return intent_dir


def _compile(model_dir: Path, *, semantic_enabled: bool):
    from nlu_compiler import content_bundle

    schema = dict(SCHEMA)
    schema["semantic_rescue_enabled"] = semantic_enabled
    out = Path(tempfile.mkdtemp())
    try:
        _, copied, _, artifact, coreml = content_bundle.compile_models(
            "en", model_dir, out, schema)
        return {
            "copied": [p for p in copied if "semantic_head" in p],
            "artifact": artifact,
            "coreml": coreml,
            "head_on_disk": (out / "models/semantic_head/shared/head.json").exists(),
        }
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_disabled_stage_ships_no_semantic_head(stub_model_dir):
    """Today's content. The head file is RIGHT THERE and must still not ship."""
    result = _compile(stub_model_dir, semantic_enabled=False)

    assert result["copied"] == [], (
        "a semantic head was shipped for a disabled stage -- this is the CI "
        "failure: a signed artifact that fails its own contract in three places, "
        "for a code path that never executes")
    assert result["artifact"] is None
    assert result["coreml"] is None
    assert not result["head_on_disk"]


def test_enabled_stage_ships_the_head_it_declares(stub_model_dir):
    """The other direction: when the stage is on, what is declared is shipped.

    Guards the fix against being 'never ship a semantic head', which would pass
    the test above and quietly make the stage unshippable forever.
    """
    result = _compile(stub_model_dir, semantic_enabled=True)

    assert result["artifact"] == "models/semantic_head/shared/head.json"
    assert result["head_on_disk"], "declared an artifact it did not ship"
    assert result["coreml"] == "models/semantic_head/shared/SemanticHead.mlpackage"


def test_content_keeps_the_semantic_stage_disabled():
    """The premise the tests above rest on, asserted rather than assumed.

    If someone enables the stage, this fails and points at the three things that
    must be true first -- a real `embedder_id` naming the encoder the head was
    trained against, an OOS class in its label set, and a head whose labels match
    the bundle's intents. All three are fixes to the TRAINER, not to the schema.
    """
    assert SCHEMA.get("semantic_rescue_enabled") is False, (
        "semantic rescue was enabled; before a head can ship it needs a real "
        "embedder_id, an OOS class, and labels matching the bundle's intents")
