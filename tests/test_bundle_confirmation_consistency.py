"""The bundle states confirmation twice — the two statements must agree.

A compiled bundle expresses "does this intent confirm?" in two places, and a
native client may read either:

  * `runtime/policies.json`  -> `confirmation: {intent: "always" | "never"}`
  * `capabilities/<cap>/workflows.json` -> `confirmation: {required: bool, ...}`

`required` was hardcoded `False` and documented as "ask only when the
uncertainty gate says so". Once that gate was removed, `False` stopped meaning
"conditionally" and started meaning "never" — so `messaging.message.send`
claimed `always` in policies and `required: false` in its workflow. A client
reading workflows would have skipped the send confirmation entirely, silently,
while the reference engine asked every time.

Nothing compared the two, so nothing failed. This does.

See docs/confirm-gate-diagnosis.md.
"""

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in ("packages/buildtime", "packages/runtime"):
    if str(_ROOT / _p) not in sys.path:
        sys.path.insert(0, str(_ROOT / _p))

_BUNDLE = _ROOT / "dist" / "bundle-en"

pytestmark = pytest.mark.skipif(
    not (_BUNDLE / "runtime" / "policies.json").exists(),
    reason="built bundle (dist/bundle-en) required")


def _policies():
    return json.loads((_BUNDLE / "runtime" / "policies.json")
                      .read_text(encoding="utf-8"))["confirmation"]


def _workflow_confirmations():
    out = {}
    for path in sorted((_BUNDLE / "capabilities").glob("*/workflows.json")):
        for intent, wf in json.loads(path.read_text(encoding="utf-8"))["intents"].items():
            if wf.get("confirmation") is not None:
                out[intent] = wf["confirmation"]["required"]
    return out


def test_workflows_and_policies_agree_on_every_intent():
    policies = _policies()
    mismatches = [
        f"  {intent}: workflows.required={required}, policies={policies.get(intent)!r}"
        for intent, required in _workflow_confirmations().items()
        if required != (policies.get(intent) == "always")
    ]
    assert not mismatches, (
        "a client reading workflows/ would confirm differently from one reading "
        "policies/:\n" + "\n".join(mismatches))


def test_confirmation_matches_the_schemas_authored_followups():
    """Both bundle statements must trace back to the source of truth."""
    schema = json.loads((_ROOT / "language_packs" / "en" / "nlu_schema.json")
                        .read_text(encoding="utf-8"))
    authored = {i for i, cfg in schema["intents"].items() if cfg.get("followup")}
    policies = _policies()

    assert {i for i, v in policies.items() if v == "always"} == authored
    assert {i for i, r in _workflow_confirmations().items() if r} == authored


def test_no_confidence_conditional_confirmation_reaches_the_bundle():
    """`when_ambiguous` and the band that interpreted it are both gone.

    Emitting one without the other leaves a runtime knowing WHICH intents were
    conditional but not WHEN — so it confirms always or never, both wrong.
    """
    policies_doc = json.loads((_BUNDLE / "runtime" / "policies.json")
                              .read_text(encoding="utf-8"))
    assert "when_ambiguous" not in set(policies_doc["confirmation"].values())
    for dead in ("uncertain_confirm_below", "uncertain_confirm_floor"):
        assert dead not in policies_doc["thresholds"]


def test_the_oov_guard_ships_as_a_PAIR():
    """Half of this guard is worse than none of it.

    `oov_reject` alone refuses entity values, which are out-of-vocabulary BY
    NATURE — a contact name, a brand, a free-text reminder topic can never all
    be in a finite vocabulary. "send a message to john" is 25% unknown and
    entirely real; only `oov_bypass` separates it from "help me find a paper",
    which is 25% unknown and out of scope.

    Shipping one half of a pair is how the device and server temperatures
    diverged (B8). This asserts the two travel together, in both the bundle and
    the schema they are compiled from.
    """
    thresholds = json.loads((_BUNDLE / "runtime" / "policies.json")
                            .read_text(encoding="utf-8"))["thresholds"]
    assert ("oov_reject" in thresholds) == ("oov_bypass" in thresholds), (
        f"the out-of-vocabulary guard shipped incomplete: {sorted(thresholds)}")

    schema = json.loads((_ROOT / "language_packs" / "en" / "nlu_schema.json")
                        .read_text(encoding="utf-8"))
    if "oov_reject" in thresholds:
        assert thresholds["oov_reject"] == schema["oov_reject_ratio"]
        assert thresholds["oov_bypass"] == schema["oov_bypass_confidence"]


def test_the_agreement_threshold_reaches_the_bundle():
    """A client without it rejects corroborated commands the engine fires.

    Measured on the honest holdout: 4 turns where a keyword rule and the model
    name the same intent but the top class sits between the agreement bar and
    the fire threshold ("turn the volume up on the hearing aid" at 0.627).
    """
    thresholds = json.loads((_BUNDLE / "runtime" / "policies.json")
                            .read_text(encoding="utf-8"))["thresholds"]
    schema = json.loads((_ROOT / "language_packs" / "en" / "nlu_schema.json")
                        .read_text(encoding="utf-8"))
    assert thresholds.get("agreement") == schema["agreement_threshold"]
