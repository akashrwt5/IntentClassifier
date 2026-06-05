# On-Device NLU Engine — Dialogflow Replacement

This document describes the conversational NLU engine that replaces Dialogflow
entirely, running offline on-device. It covers the architecture, the data
contracts, and how to port the logic to Android/iOS.

## Why replace Dialogflow

Dialogflow gave us five things. We now own all five, offline:

| Dialogflow feature | Our replacement |
|---|---|
| Intent detection | TF-IDF + LogisticRegression → ONNX (`models/intent_model.onnx`) |
| Entity extraction | `scripts/nlu/entities.py` (enum + system date-time/number) |
| Slot filling | `scripts/nlu/engine.py` slot-filling state machine |
| Contexts & follow-ups | `scripts/nlu/context.py` + engine confirmation flow |
| Session management | `scripts/nlu/context.py` `SessionStore` |

No network calls. No per-request billing. No vendor lock-in.

## Architecture

```
                      ┌─────────────────────────────────────────────┐
   user utterance ───▶│                 NLUEngine                    │
   (STT text)         │  scripts/nlu/engine.py                       │
                      │                                              │
                      │  1. CONFIRM?  active yes/no context ─────────┼──▶ fire yes/no action
                      │  2. SLOTS?    intent mid-collection ─────────┼──▶ prompt for next slot
                      │  3. CLASSIFY  fresh turn                      │
                      │       │                                      │
                      │       ▼                                      │
                      │  IntentClassifier ──▶ EntityExtractor ──▶ … │
                      │  (classifier.py)      (entities.py)          │
                      └──────────────────────┬───────────────────────┘
                                             ▼
                                      NLUResult (uniform)
```

### Three modes, evaluated in priority order, per turn

1. **CONFIRMATION** — a yes/no follow-up context is active (e.g. "Send this
   message?"). The utterance is mapped to yes / no / unclear.
2. **SLOT FILLING** — an intent is mid-collection. Fill the awaited slot from
   this utterance; if required slots remain, prompt for the next.
3. **CLASSIFY** — fresh turn: classify intent, extract entities present
   up-front, then fulfil immediately (no slots), start slot filling, or open a
   confirmation follow-up.

## Declarative config (the portable contract)

All conversational behaviour lives in JSON, not code — so the same config drives
the Python reference engine and the mobile ports.

### `data/nlu_schema.json`
- `intents.<NAME>.slots[]` — `{name, entity, required, prompt}`
- `intents.<NAME>.action` — app action id to invoke on fulfilment
- `intents.<NAME>.fulfillment` — assistant confirmation message
- `intents.<NAME>.followup` — yes/no confirmation flow (context, prompt, yes/no branches)
- `affirmative` / `negative` — yes/no vocabularies for confirmations

### `data/nlu_entities.json`
- `memory` (38 presets, fuzzy), `recurrence` (21 values), `remind` (open-ended topic)
- `sys.date-time`, `sys.number-integer` — handled by rule-based parsers

Both files are generated from the Dialogflow export, then hand-reviewable.

## Response contract — `NLUResult`

```python
{
  "type": "FULFILL" | "PROMPT" | "CONFIRM" | "FALLBACK",
  "intent": "REMINDER",
  "action": "reminders.add",          # app action to run on FULFILL
  "parameters": {"name": "...", "date-time": "..."},
  "message": "Reminder created.",     # what the assistant says next
  "confidence": 0.91,
  "complete": true,                   # intent fully resolved
  "url": "https://genai…"             # only on FALLBACK
}
```

How the app reacts to each `type`:
- **FULFILL** — run `action` with `parameters`, speak `message`.
- **PROMPT** — speak `message`, wait for the next utterance (slot filling continues).
- **CONFIRM** — speak `message`, expect a yes/no next.
- **FALLBACK** — hand `url` to the GenAI assistant.

## Usage

```python
from nlu import NLUEngine
engine = NLUEngine()

engine.handle("user-123", "set a reminder")
#   PROMPT  "What do you want to be reminded?"
engine.handle("user-123", "take medication")
#   PROMPT  "When should I remind you?"
engine.handle("user-123", "at 5 pm")
#   FULFILL reminders.add {name: "Take Medication", date-time: "...T17:00"}
```

Interactive demo:

```bash
python scripts/nlu_cli.py
```

Tests:

```bash
python scripts/test_nlu.py
```

## Porting to Android / iOS

The Python is the **reference implementation**. To port:

1. **Model** — ship `intent_model.onnx` + `intent_labels.pkl`, run with ONNX
   Runtime Mobile. Replicate the keyword pre-filter in `classifier.py`.
2. **Config** — bundle `nlu_schema.json` + `nlu_entities.json` as assets.
3. **Entities** — port `entities.py`. The date-time parser is deliberately pure
   regex/arithmetic so it translates directly to Kotlin/Swift. Levenshtein for
   fuzzy enum matching is ~15 lines.
4. **Engine + context** — port the three-mode dispatch and the in-memory
   `SessionStore` (a map keyed by session id).

No component depends on Python-only libraries in its reference path (the
optional `dateparser` is server-side enhancement only).

## Slot-filling intents (from the Dialogflow export)

| Intent | Required slots | Optional | Follow-up |
|---|---|---|---|
| `REMINDER` | `name` (@remind), `date-time` (@sys.date-time) | `recurrence` (@recurrence) | — |
| `MEMORY` | `MemoryName` (@memory) | — | — |
| `PUSH_TO_TALK` | — | — | yes/no confirm send |

All other intents are fire-and-forget (volume, translate, transcribe,
telehear, selfcheck, battery, find-my-phone, listen-message, activity,
notifications, help). Out-of-scope and low-confidence inputs route to GenAI.
