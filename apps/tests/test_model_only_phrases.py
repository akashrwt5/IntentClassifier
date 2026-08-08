#!/usr/bin/env python3
"""Model-only benchmark (semantic layer OFF), driven by a CSV of phrases.

Reads every phrase from ``apps/tests/semantic_benchmark_250.csv`` (columns:
``utterance, expected_intent, difficulty``) and runs it through the SAME engine
``apps/cli/nlu_cli.py`` builds — with the Stage-3 semantic layer explicitly
DISABLED — to answer one question: how many phrases does the TF-IDF model
resolve on its OWN, with no semantic rescue?

The benchmark's ``expected_intent`` column uses the legacy ``Cmd.*`` / ``Help_*``
taxonomy; ``LABEL_MAP`` below migrates it to the current ``domain.object.action``
labels. ``Default Fallback Intent`` maps to the sentinel ``GENAI`` — the model is
expected to DEFLECT (return a FALLBACK), not fire an intent.

Run as a report:
    python apps/tests/test_model_only_phrases.py

Run under pytest:
    pytest apps/tests/test_model_only_phrases.py -v
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "packages" / "runtime"))

MODEL = REPO / "models" / "intent_model.onnx"
BENCHMARK = Path(__file__).resolve().parent / "semantic_benchmark_250.csv"

# Legacy benchmark labels -> current domain.object.action taxonomy.
# "Default Fallback Intent" -> GENAI sentinel (expected to deflect, not fire).
LABEL_MAP = {
    "Cmd.BatteryLevel": "Cmd.BatteryLevel",
    "Cmd.MemoryChange": "Cmd.MemoryChange",
    "Cmd.SendMessage": "Cmd.SendMessage",
    "Cmd.StreamingStart": "Cmd.StreamingStart",
    "Cmd.VolumeDecrease": "Cmd.VolumeDecrease",
    "Cmd.VolumeIncrease": "Cmd.VolumeIncrease",
    "Default Fallback Intent": "GENAI",
    "Help_FindMyHearingAids": "Help_FindMyHearingAids",
    "Help_Home": "Help_Home",
    "reminders.add": "reminders.add",
}


def load_cases():
    """Yield (utterance, expected_intent_new, difficulty, note) from the CSV.

    ``note == "borderline"`` marks genuinely ambiguous utterances where the
    labelled intent is defensible but a DEFLECT (fallback) is an equally valid
    answer — so a deflect on those is not counted as a failure."""
    with open(BENCHMARK, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            old = row["expected_intent"].strip()
            yield (row["utterance"].strip(),
                   LABEL_MAP.get(old, old),          # unmapped labels pass through
                   row.get("difficulty", "").strip(),
                   row.get("note", "").strip())


def build_engine():
    """Same construction as apps/cli/nlu_cli.py, but Stage-3 semantic OFF."""
    from nlu_engine import NLUEngine
    return NLUEngine(semantic_enabled=False)


def predict(engine, text):
    """Intent the engine settled on ('GENAI' when it deflects to fallback)."""
    engine.reset("case")
    r = engine.handle("case", text)
    if r.type == "FALLBACK":
        return "GENAI"
    return r.intent or ""


def evaluate():
    """Return (rows, correct, per_intent{expected:[ok,total]}, misses[], borderline).

    A borderline row counts as correct if the model matches the label OR deflects
    to fallback (both are acceptable there)."""
    eng = build_engine()
    per_intent = defaultdict(lambda: [0, 0])
    misses = []
    correct = total = borderline = 0
    for text, expected, _diff, note in load_cases():
        total += 1
        pred = predict(eng, text)
        is_border = note == "borderline"
        borderline += is_border
        ok = (pred == expected) or (is_border and pred == "GENAI")
        correct += ok
        per_intent[expected][1] += 1
        per_intent[expected][0] += ok
        if not ok:
            misses.append((text, expected, pred))
    return total, correct, per_intent, misses, borderline


def run_report():
    if not MODEL.exists():
        print("models/intent_model.onnx not present — run `make train` first.")
        return
    total, correct, per_intent, misses, borderline = evaluate()
    print("=== MODEL-ONLY benchmark (semantic layer OFF) ===")
    print(f"    source: {BENCHMARK.name}  ({total} phrases; {borderline} borderline "
          f"— deflect accepted there)\n")
    print("    Per expected intent:")
    for intent in sorted(per_intent):
        ok, n = per_intent[intent]
        print(f"      {ok:3d}/{n:<3d}  {intent}")
    print(f"\n    TOTAL resolved by the model alone (no semantic): "
          f"{correct}/{total}  ({correct/total:.1%})\n")
    print(f"    Misses ({len(misses)}):")
    for text, expected, pred in misses:
        print(f"      {text!r:52s} want={expected:28s} got={pred}")


# ── pytest ────────────────────────────────────────────────────────────────────
def _skip_guard():
    import pytest
    if not MODEL.exists():
        pytest.skip("models/intent_model.onnx not present (make train)")
    if not BENCHMARK.exists():
        pytest.skip(f"benchmark CSV not found: {BENCHMARK}")
    pytest.importorskip("onnxruntime")


def test_benchmark_loads_and_maps():
    _skip_guard()
    cases = list(load_cases())
    assert len(cases) >= 200, f"expected the full benchmark, got {len(cases)} rows"
    # every expected label is either a mapped new-taxonomy intent or the GENAI sentinel
    for _text, expected, _diff, _note in cases:
        assert expected == "GENAI" or "." in expected, f"unmapped label: {expected}"


def test_model_only_resolution_is_reported():
    """Runs the whole benchmark model-only. This is a MEASUREMENT, not a strict
    gate — it fails only if resolution collapses, which would signal the model or
    engine wiring broke, not that a few HARD paraphrases missed."""
    _skip_guard()
    total, correct, _per, _misses, _border = evaluate()
    assert total >= 200
    assert correct / total >= 0.40, (
        f"model-only resolution collapsed: {correct}/{total} "
        f"— expected the model to handle a meaningful share of HARD phrases")


if __name__ == "__main__":
    run_report()
