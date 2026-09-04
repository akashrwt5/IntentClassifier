"""The pack ships the confirmation labels, and nothing around them.

`runtime/legacy_labels.json` carried two things: a modern -> legacy `map` and
`confirm_compound`. The map was IDENTITY for all 57 intents — it renamed nothing
— and the completeness gate that justified keeping it complete
(`test_every_trained_intent_has_a_legacy_mapping`) reads the repo-side SOURCE,
not the pack. So every device received 57 lookups that return their own input,
to satisfy a build-time check that never opened the file. 3004 bytes for two
strings that do real work.

Those two strings still cannot come from anywhere else. The classifier's label
space does not contain `Cmd.SendMessage - yes` and cannot: polarity is decided
after classification. Neither runtime is allowed to compose an intent label. So
the pair stays content-owned — under its own name, in its own artifact, with the
dead half left behind in the repo where the gate lives.

These tests fail if either half of that decision is undone: the map coming back
into the bundle, or the compound labels leaving it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_BUNDLE = _ROOT / "dist" / "bundle-en"
_ARTIFACT = _BUNDLE / "runtime" / "confirmation_labels.json"
_SOURCE = _ROOT / "packages" / "runtime" / "nlu_engine" / "legacy_label_map.json"

pytestmark = pytest.mark.skipif(
    not _BUNDLE.exists(), reason="built bundle (dist/bundle-en) required")


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert _ARTIFACT.exists(), (
        "runtime/confirmation_labels.json is not in the bundle. Without it a "
        "resolved confirmation reports the plain intent, and a host that keys "
        "off the compound label sees an intent it has no branch for.")
    return json.loads(_ARTIFACT.read_text(encoding="utf-8"))


def test_the_superseded_artifact_is_gone():
    stale = _BUNDLE / "runtime" / "legacy_labels.json"
    assert not stale.exists(), (
        "runtime/legacy_labels.json is back in the bundle. Its `map` is identity "
        "for every intent, so it is 2.6 KB of no-op lookups on every device; the "
        "gate that wants a complete map reads the repo source instead.")


def test_the_artifact_carries_only_the_confirmation_labels(artifact):
    assert set(artifact) <= {"$comment", "generated_from", "confirm_compound"}, (
        f"unexpected keys: {sorted(set(artifact) - {'$comment', 'generated_from', 'confirm_compound'})}")
    assert "map" not in artifact, (
        "the identity map followed the compound labels into their new home — the "
        "rename would then have moved the dead weight rather than dropped it")


def test_both_polarities_are_named(artifact):
    compound = artifact["confirm_compound"]
    assert compound, "an artifact with an empty map should not be written at all"
    for intent, forms in compound.items():
        assert set(forms) == {"yes", "no"}, (
            f"{intent} names {sorted(forms)}; a half-named pair makes one branch "
            f"report a compound label and the other a plain intent, for the same turn")
        assert forms["yes"] != forms["no"], f"{intent}: both polarities share a label"


def test_the_compound_labels_are_not_model_labels(artifact):
    """The reason this file has to exist at all.

    If the head could emit these, the engine must not also synthesise them — two
    paths would produce the same string with different confidences. It cannot:
    polarity is not known until after the classifier has spoken.
    """
    labels = set(json.loads(
        (_BUNDLE / "models" / "intent" / "en" / "labels.json").read_text(encoding="utf-8")))
    for intent, forms in artifact["confirm_compound"].items():
        assert intent in labels, (
            f"{intent} is not a trained label, so no confirmation on it can ever resolve")
        for polarity, label in forms.items():
            assert label not in labels, (
                f"{label} IS a classifier label — the engine must not invent it too")


def test_the_artifact_is_projected_from_the_shared_source(artifact):
    """Both runtimes read the same strings.

    The reference engine reads the repo source directly and the device reads this
    projection. Recording the source path in the artifact is not decoration: it
    is how someone who finds a wrong label on a device knows which file to edit.
    """
    assert artifact.get("generated_from") == \
        "packages/runtime/nlu_engine/legacy_label_map.json"
    source = json.loads(_SOURCE.read_text(encoding="utf-8"))
    assert artifact["confirm_compound"] == source["confirm_compound"], (
        "the bundle and the reference engine disagree on the compound labels — "
        "the device and the Python engine would name the same turn differently")


def test_the_source_keeps_the_map_the_gate_needs():
    """The other half of the decision: dropped from the PACK, not from the repo.

    `test_every_trained_intent_has_a_legacy_mapping` reads this file. If the map
    were deleted outright, that gate would pass vacuously and a new intent could
    reach the app under whatever name the trainer happened to give it.
    """
    source = json.loads(_SOURCE.read_text(encoding="utf-8"))
    assert source.get("map"), (
        "the legacy map is gone from the repo source too — the completeness gate "
        "now has nothing to check")
