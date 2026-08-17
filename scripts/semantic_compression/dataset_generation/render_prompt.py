"""Render the exact generator prompt for one intent, and ingest a pasted reply.

Evaluating a prompt by paying for an API run is the slow, expensive way round.
The prompt can be judged for free by pasting it into any chat UI and reading
what comes back — and because that reply can be fed straight into
``stage1_report.py``, the manual path produces the same numbers as the real one
rather than a vague impression.

Two modes:

``--manual``
    Emits the prompt with an explicit output-format instruction appended. The
    API path gets its schema enforced by structured output; a chat window does
    not, so the format has to be stated in words instead.

``--import FILE``
    Reads the JSON the model returned, validates it against the real schema,
    and writes it into the Stage 1 store. ``stage1_report.py`` then scores it
    exactly as it would score a paid run.

Usage::

    python render_prompt.py --intent Cmd.VolumeIncrease --manual > prompt.txt
    # paste prompt.txt into a chat UI, save the JSON reply as reply.json
    python render_prompt.py --intent Cmd.VolumeIncrease --import reply.json
    python stage1_report.py --only Cmd.VolumeIncrease
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

import generator as gen
from schemas import batch_model_for
from seed_loader import GeneratorConfig, SeedCorpusError, load_config, load_seed_corpus

DEFAULT_CONFIG = Path(__file__).with_name("generator_config.yaml")

MANUAL_FORMAT_NOTE = """

# Output format
Reply with JSON ONLY — no commentary, no markdown fence. A single object:

{{
  "utterances": [
    {{"utterance": "...", "intent": "{intent}", "type": "...",
     "difficulty": "...", "source": "..."}}
  ]
}}

`type` must be exactly one of: ExplicitCommand, ImplicitCommand, Observation,
ObservationPlusCommand, Question, Negation, Conversation, Fallback.
`difficulty` must be exactly one of: Easy, Medium, Hard.
`source` must be exactly one of: LLM-Generated, ASR-Simulated, Human-Seed.
"""


def _specs(config: GeneratorConfig) -> dict[str, dict[str, Any]]:
    path = config.output_file("intent_specs")
    if not path.is_file():
        raise SeedCorpusError(f"{path.name} not found -- run bootstrap_specs.py first")
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {s["name"]: s for s in document.get("intents", [])}


def render(config: GeneratorConfig, intent: str, *, batch_size: int, manual: bool) -> str:
    specs = _specs(config)
    if intent not in specs:
        raise SeedCorpusError(f"No specification for {intent!r}")
    corpus = load_seed_corpus(config)
    spec = specs[intent]
    gen_cfg = config.raw.get("generation", {})

    payload = {
        "intent": intent,
        "family": spec.get("intent_family", "Unassigned"),
        "business_description": spec["business_description"],
        "trigger_conditions": gen._bullets(spec["trigger_conditions"]),
        "do_not_trigger": gen._bullets(spec["do_not_trigger"]),
        "boundary_cases": gen._bullets(spec["boundary_cases"]),
        "neighbor_intents": ", ".join(spec["neighbor_intents"]),
        "positive_example": spec.get("positive_example", ""),
        "hard_negative_example": spec.get("hard_negative_example", ""),
        "slot_block": gen._slot_block(config, intent),
        # Same helper the real generator uses, so the manual path measures the
        # same prompt and honours the same privacy suppression.
        "seed_block": gen._evidence_block(
            config,
            corpus.intents.get(intent, []),
            intent,
            int(gen_cfg.get("seed_reference_count", 8)),
        ),
        "avoid_block": "",
        "batch_size": batch_size,
        "composition_block": gen._composition_block(config, intent, batch_size),
        "correction": "",
    }

    text = (
        "=================== SYSTEM MESSAGE ===================\n"
        + gen.SYSTEM_PROMPT
        + "\n=================== USER MESSAGE =====================\n"
        + gen.HUMAN_PROMPT.format(**payload)
    )
    if manual:
        text += MANUAL_FORMAT_NOTE.format(intent=intent)
    return text


def ingest(config: GeneratorConfig, intent: str, path: Path) -> int:
    """Validate a pasted reply and write it into the Stage 1 store."""
    raw = path.read_text(encoding="utf-8").strip()

    # Chat UIs habitually wrap JSON in a markdown fence; strip it rather than
    # making the user clean up by hand.
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    data = json.loads(raw)
    if isinstance(data, list):
        data = {"utterances": data}

    try:
        # Same per-intent schema the paid path uses, so a pasted reply that
        # puts a type value in the intent field fails here too.
        batch = batch_model_for(intent).model_validate(data)
    except ValidationError as exc:
        # A chat UI enforces no schema, so a dropped or misplaced field is
        # routine on this path. Report it against the offending utterance
        # rather than as a traceback -- this tool exists to be usable by hand,
        # and a stack trace at the moment of failure defeats that.
        print(f"{exc.error_count()} schema problem(s) in the reply:\n")
        items = data.get("utterances", []) if isinstance(data, dict) else []
        for error in exc.errors():
            location = error["loc"]
            index = location[1] if len(location) > 1 and isinstance(location[1], int) else None
            field = location[-1]
            where = f"item {index + 1}" if index is not None else "batch"
            print(f"  · {where} field {field!r}: {error['msg']}")
            if index is not None and index < len(items):
                print(f"      {str(items[index].get('utterance', ''))[:60]!r}")
        print("\nFix those entries in the reply file and import again.")
        raise SeedCorpusError("reply does not match the metadata schema") from exc
    specs = _specs(config)
    family = specs.get(intent, {}).get("intent_family", "Unassigned")

    batch_errors, row_errors = gen.validate_batch(
        batch,
        intent=intent,
        fallback_intent=config.fallback_intent,
        accepted_keys=set(),
        max_words=int(config.raw.get("generation", {}).get("max_utterance_words", 20)),
        expected_size=len(batch.utterances),
        cancellation_capability=gen._cancellation_capability(config, intent),
    )
    faults = [message for messages in row_errors.values() for message in messages]
    if batch_errors or faults:
        print(
            f"{len(batch_errors)} batch-level and {len(faults)} row-level problem(s). "
            f"The real pipeline would retry the batch for the former and drop "
            f"{len(row_errors)} of {len(batch.utterances)} utterances for the latter:"
        )
        for error in batch_errors + faults:
            print(f"  · {error}")
        print()

    store = gen.Stage1Store(config.checkpoint_dir)
    rows = [
        {
            "utterance": item.utterance,
            "intent": intent,
            "intent_family": family,
            "type": item.type.value,
            "difficulty": item.difficulty.value,
            "source": item.source.value,
        }
        for item in batch.utterances
    ]
    store.append(intent, rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render or ingest a generator prompt by hand.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--intent", required=True)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--manual", action="store_true", help="Append the output-format note.")
    parser.add_argument("--import", dest="import_path", type=Path, default=None)
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        if args.import_path:
            count = ingest(config, args.intent, args.import_path)
            print(f"Imported {count} utterances for {args.intent}.")
            print(f"Now run:  python stage1_report.py --only {args.intent}")
        else:
            print(render(config, args.intent, batch_size=args.batch_size, manual=args.manual))
    except SeedCorpusError as exc:
        print(f"ERROR: {exc}")
        return 1
    except json.JSONDecodeError as exc:
        print(f"ERROR: reply is not valid JSON: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
