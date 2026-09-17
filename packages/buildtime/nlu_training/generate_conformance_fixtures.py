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


MODELS = REPO / "models" / "intent"


def _has_model(lang: str) -> bool:
    """A language this checkout can actually build an engine for.

    Only `en` is trained in this repository — de/fr/da have no train.csv and no
    nlu_entities.json here at all — so regenerating blind either crashes or,
    worse, writes a file missing everything it could not produce.
    """
    return (MODELS / lang / "model.onnx").exists()


def _existing(name: str) -> dict:
    """The committed fixture, or {} if there is none."""
    path = OUT_DIR / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def _engine(lang: str):
    sys.path.insert(0, str(REPO / "packages" / "runtime"))
    from nlu_engine import NLUEngine

    eng = NLUEngine(model_name=lang, language=lang, semantic_enabled=False)
    return eng


def conversations_fixture(langs: set[str] | None = None) -> tuple[dict, dict]:
    """Replay the corpus; return (scripts, skipped_by_lang).

    A language is regenerated only when this checkout can build an engine for
    it AND `langs` allows it. Everything else keeps whatever the committed
    fixture already held — deleting a golden entry because the machine could
    not rebuild it is how parity coverage disappears without anyone noticing.
    """
    import yaml

    previous = (_existing("conversations.json").get("scripts") or {})
    engines: dict = {}
    scripts: dict = {}
    skipped: dict = {}
    for path in sorted(CORPUS.glob("*.yaml")):
        script = yaml.safe_load(path.read_text(encoding="utf-8"))
        lang = script.get("lang", "en")

        wanted = langs is None or lang in langs
        if not wanted or not _has_model(lang):
            reason = ("not requested" if not wanted
                      else "no trained model in this checkout")
            skipped.setdefault(lang, []).append(path.stem)
            if path.stem in previous:
                scripts[path.stem] = previous[path.stem]   # keep the golden entry
            continue

        if lang not in engines:
            engines[lang] = _engine(lang)
        eng = engines[lang]
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
    return scripts, skipped


def datetime_clock_grid_fixture(langs: set[str] | None = None) -> tuple[list[dict], dict]:
    """Return (rows, skipped_by_lang), keeping committed rows for any language
    this checkout cannot rebuild.

    `language_packs/{fr,de,da}/nlu_entities.json` do not exist here, so without
    the merge a regeneration silently empties all 75 committed rows.
    """
    from nlu_engine.entities import EntityExtractor

    previous = _existing("datetime_clock_grid.json").get("rows") or []
    loc = REPO / "language_packs"
    extractors = {}
    skipped: dict = {}
    for lang in ("fr", "de", "da"):
        path = loc / lang / "nlu_entities.json"
        wanted = langs is None or lang in langs
        if wanted and path.exists():
            extractors[lang] = EntityExtractor(entities_path=path, language=lang)
        else:
            skipped[lang] = ("not requested" if not wanted
                             else "no nlu_entities.json in this checkout")
    rows = [r for r in previous if r.get("lang") in skipped]
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
    return rows, skipped


def _report(what: str, skipped: dict) -> None:
    """Say out loud what was NOT regenerated.

    A generator that quietly reproduces stale entries is indistinguishable from
    one that refreshed them, and the difference is the whole value of a golden
    file. If this prints, the fixture is part current and part inherited.
    """
    if not skipped:
        return
    print(f"  {what}: KEPT COMMITTED ENTRIES for "
          f"{', '.join(sorted(skipped))} — not regenerated here")
    for lang in sorted(skipped):
        detail = skipped[lang]
        if isinstance(detail, list):
            detail = f"{len(detail)} script(s): no trained model in this checkout"
        print(f"    {lang}: {detail}")


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="nlu_training.generate_conformance_fixtures",
                                 description=__doc__)
    ap.add_argument("--langs", nargs="+", default=None,
                    help="languages to REGENERATE (default: every language this "
                         "checkout has a model for). Any other language keeps "
                         "its committed entries rather than losing them.")
    args = ap.parse_args(argv)
    langs = set(args.langs) if args.langs else None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    convs, conv_skipped = conversations_fixture(langs)
    (OUT_DIR / "conversations.json").write_text(
        json.dumps({"generator": "python-engine (executable spec, ADR-011)",
                    "semantic_enabled": False, "scripts": convs},
                   indent=2, ensure_ascii=False) + "\n")
    print(f"conversations.json: {len(convs)} scripts")
    _report("conversations.json", conv_skipped)

    grid, grid_skipped = datetime_clock_grid_fixture(langs)
    (OUT_DIR / "datetime_clock_grid.json").write_text(
        json.dumps({"generator": "python-engine EntityExtractor",
                    "clock_grid": CLOCK_GRID, "rows": grid},
                   indent=2, ensure_ascii=False) + "\n")
    print(f"datetime_clock_grid.json: {len(grid)} rows")
    _report("datetime_clock_grid.json", grid_skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
