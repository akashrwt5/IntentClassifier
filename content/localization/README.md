# NLU localization data (draft — for native review)

Per-language localization artifacts for the multilingual NLU. The multilingual classifier supports
four languages — **en, fr, de, da** (see `scripts/nlu_cli_multilingual.py --model`). English is the
canonical source (`../nlu_entities.json`, `../nlu_schema.json`); this directory holds the **fr / de /
da** drafts. They are **machine-drafted and NOT yet native-reviewed** — treat every string as a
proposal, not ground truth.

These files are the **single source of truth shared with the STT app** (mirrored at
`STT/docs/localization-drafts/`). Keep the two copies byte-identical when either changes.

## File families (one set per language)

### `nlu_entities.<lang>.json` — enum synonyms
Same shape as the canonical `../nlu_entities.json`.
- **English canonical keys are preserved** (`"Car"`, `"Take Medication"`, `"Daily"`, …) — only the
  *synonym lists* are localized. This keeps slot→action mapping and server parity intact.
- **English synonyms are retained** alongside the localized ones (code-switching robustness), then
  the target-language synonyms are appended.
- Coverage matches canonical exactly: memory 38, recurrence 21, remind 6.
- `sys.date-time` / `sys.number-integer` are grammar (the lexicon), not synonym data — left as
  system stubs here.

### `nlu_schema.<lang>.json` — localization **overlay** (not a full schema)
A patch over the canonical `../nlu_schema.json`, carrying only the translatable strings:
- `intents.<Name>.fulfillment` and `intents.<Name>.slots[].prompt` for every intent (same keys,
  same order as canonical).
- `affirmative` / `negative` confirmation words.
- Brand/feature proper nouns (IntelliVoice, Thrive Score, Edge/Mask Mode, HearShare, WiCROS,
  SelfCheck, Hearing Care Anywhere, Translate, Transcribe, Fall Alert) are intentionally **not**
  translated.
- **Not localized here (pending, language-specific grammar):** `keyword_triggers` and
  `back_reference` regexes. Those need per-language authoring, tracked in the plan doc.

### `nlu_lexicon.<lang>.json` — date/time grammar + yes/no + carrier phrases
The language-specific grammar the date/number parser consumes:
- `grammar` — `time_format` (all four use 24h), decimal-hour idioms, and `notes` flagging parser
  traps.
- `weekdays` (7), `months` (12), `day_anchors`, `time_of_day` (with default hours),
  `numbers_0_to_31`, `ordinals_1_to_31`, `relative_units`, `relative_markers`.
- `affirmative` / `negative` / `uncertain` and `carrier_phrases` (regexes that strip reminder
  preambles like "rappelle-moi de …" / "erinnere mich daran …" / "mind mig om at …").

## ⚠️ Known parser traps flagged by the drafts (must be honored in the date engine)
- **German `halb drei` = 02:30** and **Danish `halv tre` = 02:30** — "half" counts DOWN to the named
  hour. A naive `X:30` reading lands every such reminder an hour late.
- **French `moins le quart`** subtracts from the *next* hour.
- German regional `dreiviertel drei` (= 2:45) / `viertel drei` (= 2:15) — included but dialect-gated.

## Top native-review items
- **fr**: "programme" vs "mémoire" for hearing-aid memory; `moins le quart` semantics; carrier
  `de|d'` over-stripping.
- **de**: `halb drei` mapping; Saxon/Austrian `dreiviertel`/`viertel` forms; generous synonyms
  (Speech→"Gespräch"/"Sprache", Noise→"Geräusch"); Austrian months Jänner/Feber.
- **da**: `halv tre` mapping; bare article "en"/"et" colliding with number 1; "jo"/"næ" as
  yes/no particles; "program" vs "hukommelse" for memory.

Nothing here is wired into training or the build — wiring is the separate, approval-gated work in
`MULTILINGUAL_NLU_LOCALIZATION_PLAN.md`.
