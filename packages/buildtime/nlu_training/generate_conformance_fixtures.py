#!/usr/bin/env python3
"""Cross-platform conformance fixtures — the Python engine as executable spec.

With the shared Rust core removed by owner directive (ADR-011), THIS is the
drift-prevention mechanism: the Python engine's decisions over a fixed
corpus become golden fixtures that the native iOS and Android engines must
reproduce in their CI. Two sections attack the two known drift surfaces:

1. `conversations` — every tests/conversations/*.yaml script replayed
   multi-turn: full decision trace (type/intent/action/confidence/message
   keys) per turn, INCLUDING session-state effects (slot flows, confirms).
2. `datetime_clock_grid` — the datetime parity CSVs re-evaluated at THREE
   reference instants (midnight / midday / late evening, different
   weekdays), because parking/anchoring logic depends on wall-clock state —
   the exact gap ADR-001 said single-clock fixture CSVs leave open.

Regenerate after any dialogue/policy/model change:
    PYTHONPATH=packages/buildtime:packages/runtime \\
        python -m nlu_training.generate_conformance_fixtures
Output: tests/parity/engine_conformance/*.json (committed; native CI loads
them; a diff in review IS the cross-platform behavior change).
"""

from __future__ import annotations

import csv
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "tests" / "parity" / "engine_conformance"
CORPUS = REPO / "tests" / "conversations"
DT_FIXTURES = REPO / "tests" / "datetime_parity"

# Three reference instants exercising day-boundary + anchoring behavior:
# Tue 00:00 (every today-time is future), Wed 12:00 (afternoon rollovers),
# Sun 22:30 (late evening → most times roll to tomorrow/next week).
CLOCK_GRID = [
    "2026-06-30T00:00:00+00:00",
    "2026-07-01T12:00:00+00:00",
    "2026-07-05T22:30:00+00:00",
]


def _engine(lang: str):
    sys.path.insert(0, str(REPO / "packages" / "runtime"))
    from nlu_engine import NLUEngine

    eng = NLUEngine(model_name=lang, language=lang, semantic_enabled=False)
    return eng


def conversations_fixture() -> dict:
    import yaml

    engines: dict = {}
    scripts = {}
    for path in sorted(CORPUS.glob("*.yaml")):
        script = yaml.safe_load(path.read_text(encoding="utf-8"))
        lang = script.get("lang", "en")
        if lang not in engines:
            engines[lang] = _engine(lang)
        eng = engines[lang]
        if script.get("force_confirm"):
            continue  # test-only override, not shipping behavior — skip
        session = f"fixture-{path.stem}"
        eng.reset(session)
        turns = []
        for turn in script["turns"]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = eng.handle(session, turn["user"])
            turns.append({"user": turn["user"], "type": r.type,
                          "intent": r.intent, "action": r.action,
                          "confidence": round(r.confidence, 4),
                          "complete": r.complete})
        scripts[path.stem] = {"lang": lang, "turns": turns}
    return scripts


def datetime_clock_grid_fixture() -> list[dict]:
    from nlu_engine.entities import EntityExtractor

    loc = REPO / "language_packs"
    extractors = {}
    for lang in ("fr", "de", "da"):
        path = loc / lang / "nlu_entities.json"
        if path.exists():
            extractors[lang] = EntityExtractor(entities_path=path, language=lang)
    rows = []
    for lang, ex in extractors.items():
        with open(DT_FIXTURES / f"nlu_datetime_parity_{lang}.csv",
                  encoding="utf-8") as f:
            utterances = [r["utterance"] for r in csv.DictReader(f)]
        for now_iso in CLOCK_GRID:
            now = datetime.fromisoformat(now_iso).astimezone(timezone.utc)
            for utt in utterances:
                iso, _span, conf, time_explicit, explicit_day = \
                    ex.extract_datetime(utt, now=now)
                rows.append({"lang": lang, "now": now_iso, "utterance": utt,
                             "iso": iso, "confidence": round(conf, 4),
                             "time_explicit": bool(time_explicit),
                             "explicit_day": bool(explicit_day)})
    return rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    convs = conversations_fixture()
    (OUT_DIR / "conversations.json").write_text(
        json.dumps({"generator": "python-engine (executable spec, ADR-011)",
                    "semantic_enabled": False, "scripts": convs},
                   indent=2, ensure_ascii=False) + "\n")
    print(f"conversations.json: {len(convs)} scripts")

    grid = datetime_clock_grid_fixture()
    (OUT_DIR / "datetime_clock_grid.json").write_text(
        json.dumps({"generator": "python-engine EntityExtractor",
                    "clock_grid": CLOCK_GRID, "rows": grid},
                   indent=2, ensure_ascii=False) + "\n")
    print(f"datetime_clock_grid.json: {len(grid)} rows "
          f"({len(CLOCK_GRID)} clocks × parity utterances × 3 langs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
