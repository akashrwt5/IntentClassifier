# Adding a New Language to the NLU Pipeline

This guide covers everything needed to add a new language to the multilingual NLU
system — from content authoring through training, calibration, and iOS deployment.

The architecture is intentionally data-driven: **no Swift code changes are ever
needed**, and Python requires only 1–3 line additions. The real work is authoring
accurate JSON content files and supplying training utterances.

---

## Table of Contents

1. [Overview](#overview)
2. [Step-by-step checklist](#step-by-step-checklist)
3. [File reference](#file-reference)
   - [nlu_lexicon.\<lang\>.json](#nlu_lexiconlangjson)
   - [nlu_entities.\<lang\>.json](#nlu_entitieslangjson)
   - [nlu_schema.\<lang\>.json](#nlu_schemalangjson)
   - [Training CSV](#training-csv)
4. [Training and calibration](#training-and-calibration)
5. [iOS deployment](#ios-deployment)
6. [Testing a new language](#testing-a-new-language)
7. [Deliberately deferred capabilities](#deliberately-deferred-capabilities)
8. [Known parser traps](#known-parser-traps)

---

## Overview

### What the system does per language

The NLU pipeline has four stages. Each stage has a different language footprint:

| Stage | What it does | Language surface |
|-------|-------------|-----------------|
| **0 — Keyword triggers** | Regex short-circuit for high-priority intents | Per-language regexes in `nlu_schema.<lang>.json` |
| **1 — Entity extraction** | Slot values (datetime, enum entities) | Fully lexicon-driven via `nlu_lexicon` + `nlu_entities` |
| **2 — TF-IDF + LogReg** | Intent classification | Per-language trained model from CSV |
| **3 — Semantic rescue** | Embedding-based fallback for low-confidence | English-only currently (see Deferred section) |

### Architecture principle

`NLUEngine` (both Swift and Python) accepts a `language` parameter and loads
everything else from JSON files. There are no `if language == "fr"` branches in
engine code. Adding a language is entirely a content + training problem.

---

## Step-by-step checklist

```
Content authoring (requires native speaker):
  [ ] 1.  Author nlu_lexicon.<lang>.json
  [ ] 2.  Author nlu_entities.<lang>.json
  [ ] 3.  Author nlu_schema.<lang>.json
  [ ] 4.  Author keyword_triggers in nlu_schema.<lang>.json (optional but strongly recommended)
  [ ] 5.  Prepare training CSV (multilingual/data/<lang>.csv)

Python code (1–3 line changes):
  [ ] 6.  Register language in multilingual/train_multilingual.py
  [ ] 7.  Add to calibration sweep list in scripts/calibrate_languages.py
  [ ] 8.  Add to CLI choices in scripts/nlu_cli_multilingual.py

Training and calibration:
  [ ] 9.  Run training:     python multilingual/train_multilingual.py --language <lang>
  [ ] 10. Run calibration:  python scripts/calibrate_languages.py
  [ ] 11. Add calibration entry to config/calibration.json
  [ ] 12. Write 9+ golden datetime fixtures to tests/datetime_parity/nlu_datetime_parity_<lang>.csv
  [ ] 13. Run parity test:  python -m pytest tests/test_datetime_parity.py

iOS deployment:
  [ ] 14. Copy 3 JSON files to STT/STT/STT/Resources/Localization/
  [ ] 15. Update STT/STT/STT/Resources/Multilingual/calibration.json
  [ ] 16. Confirm Xcode Build Phase copies the new files (Copy Bundle Resources)
```

---

## File reference

All localization files live in `data/localization/`. Each language needs exactly
three files. The English canonical files (`data/nlu_schema.json`,
`data/nlu_entities.json`) are the source of truth — the language files only patch
or extend them.

---

### `nlu_lexicon.<lang>.json`

**The hardest file. Requires native-speaker knowledge of date/time grammar.**

This file drives the entire datetime and number parser in both Python and Swift.
The Python and Swift parsers are byte-identical mirrors — any entry in this file
affects both platforms simultaneously.

#### Required sections

```jsonc
{
  "_meta": { "language": "<lang>", "status": "draft-for-native-review" },

  "grammar": {
    // "12h" or "24h" — most European languages use 24h
    "time_format": "24h",
    // Idioms that don't decompose to hour + minute cleanly.
    // Each entry: {"phrase": "...", "minutes": <offset>} or {"phrase": "...", "hour": N, "minutes": 0}
    // Example (French): {"phrase": "et demie", "minutes": 30}
    // CRITICAL: German/Danish "halb/halv" count DOWN — see Known parser traps section.
    "decimal_hour_idioms": [],
    // Word that joins hours and minutes in spoken form. French: "et", German: "Uhr", Danish: "og"
    "conjunction": "and",
    // Free-text notes flagging traps for future maintainers
    "notes": ""
  },

  // 7 entries, keys must be English day names exactly
  "weekdays": {
    "Monday": ["<synonym1>", "<synonym2>"],
    "Tuesday": [...], "Wednesday": [...], "Thursday": [...],
    "Friday": [...], "Saturday": [...], "Sunday": [...]
  },

  // 12 entries, keys must be English month names exactly
  "months": {
    "January": [...], "February": [...], "March": [...],
    "April": [...], "May": [...], "June": [...],
    "July": [...], "August": [...], "September": [...],
    "October": [...], "November": [...], "December": [...]
  },

  // Fixed keys, all required
  "day_anchors": {
    "today": [...],
    "tomorrow": [...],
    "day_after_tomorrow": [...],
    "yesterday": [...],
    "next_week": [...],
    "next_month": [...],
    "next_year": [...],
    "this_weekend": [...]
  },

  // Fixed keys: morning/noon/afternoon/evening/night/midnight
  // "hour" is the default hour used when no explicit time is given ("Monday morning" → 08:00)
  "time_of_day": {
    "morning":   { "names": [...], "hour": 8  },
    "noon":      { "names": [...], "hour": 12 },
    "afternoon": { "names": [...], "hour": 14 },
    "evening":   { "names": [...], "hour": 18 },
    "night":     { "names": [...], "hour": 21 },
    "midnight":  { "names": [...], "hour": 0  }
  },

  // 0–31 as digit strings → array of spoken forms (cardinal)
  "numbers_0_to_31": {
    "0": [...], "1": [...], ..., "31": [...]
  },

  // 1–31 as digit strings → array of spoken ordinal forms ("first", "second", ...)
  "ordinals_1_to_31": {
    "1": [...], "2": [...], ..., "31": [...]
  },

  // Fixed canonical keys; list every surface form including abbreviations
  "relative_units": {
    "minute": [...], "hour": [...], "day": [...],
    "week": [...], "month": [...], "year": [...]
  },

  // "in" = future offset ("in 5 minutes"), "at" = clock time ("at 3pm"), "on" = date ("on Monday")
  "relative_markers": {
    "in": [...],
    "at": [...],
    "on": [...]
  },

  // Suffixes that follow the hour number in clock expressions.
  // French uses "h"/"heure"/"heures". German/Danish have no explicit suffix needed.
  // Omit or leave empty if the language doesn't use clock suffixes.
  "clock_hour_markers": [],

  // Polarity words — these also appear in nlu_schema overlay but the lexicon copy
  // is what the datetime/carrier parser uses.
  "affirmative": ["yes", "ok", ...],
  "negative": ["no", "cancel", ...],
  "uncertain": ["maybe", "not sure", ...],

  // Regex patterns that strip reminder-command preambles.
  // Each is anchored at ^ (start of utterance) or $ (end), and strips the verb phrase
  // so the remaining text is the reminder topic or time expression.
  // Keep English defaults; prepend language-specific patterns.
  // Example (French): "^rappelle[\\s-]moi\\s+(de\\s+|d')?\\s*"
  "carrier_phrases": [
    "^remind\\s+me\\b[\\s,]*(to\\s+|about\\s+|that\\s+)?",
    "^set(?:\\s+up)?\\s+(?:an?\\s+)?(?:reminder|alarm)\\b\\s*(?:to|about|for\\s+(?!\\d))?\\s*",
    "^add\\s+(?:a\\s+)?reminder\\b\\s*(?:to|about|for\\s+(?!\\d))?\\s*",
    "\\s+(?:please|pls)\\s*$"
  ]
}
```

#### Tips

- **Longer synonyms must come first within each array** — the parser matches
  greedily and longer entries shadow shorter ones. The Python and Swift loaders
  both sort by (length DESC, value ASC) before building lookup tables, so order
  in the file doesn't matter as long as you include all forms.
- **Include common abbreviations and accent variants** (e.g., `fevrier` alongside
  `février`) — ASR transcription drops accents inconsistently.
- **Decimal-hour idioms must be exhaustive** — if a language has "quarter past",
  "half past", and "quarter to" idioms, all three must appear. Missing one means
  those expressions parse as Unknown.
- **`clock_hour_markers`** — only needed for languages that attach a suffix to the
  hour digit in clock expressions (French "15h30", "neuf heures"). Most Germanic
  languages don't need this.

---

### `nlu_entities.<lang>.json`

**Translate enum synonym lists. Shape is identical to `data/nlu_entities.json`.**

```jsonc
{
  "memory": {
    "type": "enum",
    "fuzzy": true,
    "values": {
      // Key is the CANONICAL English value — never translate the key.
      // English synonyms are RETAINED first for code-switching robustness,
      // then language synonyms are appended.
      "Car": ["car", "vehicle", "driving", "<lang-synonym1>", "<lang-synonym2>"],
      "Restaurant": ["restaurant", "dining", "eating out", "<lang-synonym1>"],
      // ... all 38 memory values ...
    }
  },
  "recurrence": {
    "type": "enum",
    "fuzzy": false,
    "values": {
      "Daily": ["daily", "every day", "<lang-synonym>"],
      // ... all 21 recurrence values ...
    }
  },
  "remind": {
    "type": "enum",
    "fuzzy": true,
    "values": {
      "Appointment": ["appointment", "meeting", "<lang-synonym>"],
      // ... all 6 remind values ...
    }
  },
  // Leave system entities as stubs — they are handled by the lexicon/datetime parser
  "sys.date-time": { "type": "system" },
  "sys.number-integer": { "type": "system" }
}
```

Coverage must match the canonical file exactly: **38 memory values, 21 recurrence
values, 6 remind values**. Omitting a value means that slot can never be filled in
that language.

---

### `nlu_schema.<lang>.json`

**A patch overlay — not a full schema copy. Only translatable strings.**

```jsonc
{
  "_meta": {
    "language": "<lang>",
    "status": "draft-for-native-review",
    "kind": "localization-overlay"
  },
  "version": 2,
  "confidence_threshold": 0.7,

  "intents": {
    // Only the fields that need translation. Omit intents with no user-visible text.
    "Cmd.VolumeIncrease": { "fulfillment": "<translated text>" },
    "Cmd.MemoryChange": {
      "fulfillment": "<translated text>",
      "slots": [
        // "name" must match the canonical slot name exactly
        { "name": "MemoryName", "prompt": "<translated prompt>" }
      ]
    },
    "reminders.add": {
      "fulfillment": "<translated text>",
      "slots": [
        { "name": "recurrence", "prompt": "" },
        { "name": "name", "prompt": "<What do you want to be reminded about?>" },
        { "name": "date-time", "prompt": "<When should I remind you?>" }
      ]
    }
    // ... repeat for every intent with user-visible strings ...
  },

  // Yes/no polarity words — must include English defaults for code-switching
  "affirmative": ["yes", "ok", "okay", "<lang-yes>", "<lang-ok>"],
  "negative": ["no", "cancel", "<lang-no>", "<lang-cancel>"],

  // Language-specific intent triggers (regex). These fire BEFORE TF-IDF (Stage 0)
  // and are the primary reliability lever for intent-critical utterances.
  // Strongly recommended for reminders.add and any intent that uses distinctive
  // command verbs in the target language.
  "keyword_triggers": [
    {
      "intent": "reminders.add",
      "regex": "<regex matching all ways to say 'set a reminder' in <lang>>"
    }
  ]
}
```

**Do NOT translate:**
- Brand/feature names: IntelliVoice, Thrive Score, Edge Mode, Mask Mode, HearShare,
  WiCROS, SelfCheck, Hearing Care Anywhere, Translate, Transcribe, Fall Alert.
- Canonical intent keys (`Cmd.VolumeIncrease`, `reminders.add`, etc.).
- Slot names (`MemoryName`, `name`, `date-time`, etc.).

**Schema overlay merge rules (automatic — no code change needed):**
- `intents[*].fulfillment` — overlay value replaces canonical
- `intents[*].slots[*].prompt` — matched by slot `name`, overlay value replaces canonical
- `affirmative` / `negative` — overlay list replaces canonical list entirely
- `keyword_triggers` — overlay entries are **appended** to (not replacing) canonical list
- Everything else (entity types, required flags, thresholds, actions) comes from canonical

---

### Training CSV

File: `multilingual/data/<lang>.csv`

```
text,intent
<utterance in target language>,<intent name>
```

- Intent names must match the canonical English intent names exactly
  (e.g., `Cmd.VolumeIncrease`, `reminders.add`, `Default Fallback Intent`).
- Aim for **100–300 utterances per intent** for good generalization. The training
  script caps at 500 per intent before combining with other languages.
- Include natural variation: different phrasings, with and without politeness
  markers, with and without explicit slot values.
- Machine-translated + human-reviewed is a practical approach; pure machine
  translation without review produces ~5–8% accuracy drop.
- Include code-switching examples (mixing target language with English) — real ASR
  output for hearing-aid users is often mixed.

**Quality bar:** Training accuracy gate is 75% by default (configurable via
`--min-accuracy`). The existing languages land at:

| Language | Macro F1 (holdout) | Temperature |
|----------|--------------------|-------------|
| en | 0.90 | 0.621 |
| fr | 0.84 | 0.670 |
| de | 0.83 | 0.678 |
| da | 0.74 | 0.816 |

Lower-resource languages (fewer training utterances) naturally have higher
temperature values (wider softmax) and lower F1.

---

## Training and calibration

### 1. Register the language

In `multilingual/train_multilingual.py`, add one line to the `LANGUAGES` dict (around line 106):

```python
LANGUAGES = {
    "en": DATA_DIR / "en.csv",
    "fr": DATA_DIR / "fr.csv",
    "de": DATA_DIR / "de.csv",
    "da": DATA_DIR / "da.csv",
    "xx": DATA_DIR / "xx.csv",   # ← add this
}
```

In `scripts/calibrate_languages.py` (around line 43):

```python
LANGUAGES = ["en", "fr", "de", "da", "xx"]   # ← append
```

In `scripts/nlu_cli_multilingual.py` (around lines 66 and 74):

```python
choices=["en", "fr", "de", "da", "xx", "multilingual"]   # ← append
```

### 2. Train

```bash
# Train only the new language
python multilingual/train_multilingual.py --language xx

# Or retrain everything including the combined multilingual model
python multilingual/train_multilingual.py --all
```

Outputs written to `multilingual/models/xx/`:
```
xx_intent_model.onnx
xx_intent_labels.pkl
xx_intent_labels.json
xx_intent_pipeline.pkl
xx_intent_classifier_weights.json
manifest.json
```

Test split written to `multilingual/test/xx_holdout.csv`.

### 3. Calibrate

```bash
python scripts/calibrate_languages.py
```

Then add the output entry to `config/calibration.json`:

```jsonc
"xx": {
  "temperature": <output from calibration>,
  "conf_threshold": 0.6,
  "conf_gap_threshold": 0.2,
  "macro_f1_holdout": <from training output>,
  "ece": <from calibration output>
}
```

---

## iOS deployment

No Swift code changes are needed. The steps are:

1. **Copy the three JSON files** to `STT/STT/STT/Resources/Localization/`:
   ```
   nlu_lexicon.xx.json    (identical to data/localization/nlu_lexicon.xx.json)
   nlu_entities.xx.json   (identical to data/localization/nlu_entities.xx.json)
   nlu_schema.xx.json     (identical to data/localization/nlu_schema.xx.json)
   ```
   These must be byte-identical to the Python-side files — they are the single
   source of truth.

2. **Update calibration** in `STT/STT/STT/Resources/Multilingual/calibration.json`
   with the same entry added to `config/calibration.json` in step 3 above.

3. **Confirm Xcode Build Phase** — open the `STT.xcodeproj`, go to the app target →
   Build Phases → Copy Bundle Resources, and verify all three new JSON files appear.
   Xcode does not auto-add files dropped into the filesystem.

4. **No Core ML model** is needed for launch. The iOS fallback path (`TFIDFLogisticScorer.swift`)
   loads `xx_intent_classifier_weights.json` directly and runs TF-IDF + LogReg in
   pure Swift. Core ML conversion is a performance optimization, not a correctness requirement.
   See [Deliberately deferred capabilities](#deliberately-deferred-capabilities) for details.

---

## Testing a new language

### 1. Interactive end-to-end test (Python)

```bash
python scripts/nlu_cli_multilingual.py --model xx --language xx
```

Try:
- A plain intent with no slots ("volume up" equivalent)
- A reminder with an explicit time ("remind me to call mom tomorrow at 9am" equivalent)
- A reminder with only a relative time ("remind me in 10 minutes" equivalent)
- The `reminders.add` keyword trigger phrase

### 2. Datetime parity test

Create `tests/datetime_parity/nlu_datetime_parity_xx.csv` with at least 9 golden
rows covering:

```
text,expected_iso,description
<tomorrow equivalent> at 9,<ISO>,tomorrow morning explicit
<in N minutes equivalent>,<ISO>,relative future
<day-of-week> <time-of-day>,<ISO>,weekday + time-of-day
<month day> at <time>,<ISO>,explicit date
<decimal-hour idiom>,<ISO>,language-specific idiom
```

Then run:

```bash
python -m pytest tests/test_datetime_parity.py -v
```

This gate ensures Python and Swift parse every golden row identically.

### 3. Holdout accuracy check

```bash
python multilingual/train_multilingual.py --language xx
```

The training script prints macro F1 on the held-out split. Target ≥ 0.75 before shipping.

---

## Deliberately deferred capabilities

These capabilities exist in the system but are English-only or not yet wired for
new languages. Each is a self-contained future work item.

---

### 1. Semantic rescue (Stage 3) — English only

**What it is:** After TF-IDF fails to reach the confidence threshold, the system
embeds the utterance using a MiniLM sentence embedding model, then applies a
trained logistic head to predict intent from the embedding. This catches utterances
that are semantically on-topic but use unusual wording.

**Current state:** The MiniLM model (`models/minilm-l6-v2.onnx`) is an English
embedding model. It produces embeddings for any language, but they are optimized
for English intent space, so rescue accuracy drops significantly for non-English
utterances.

**Impact:** Non-English utterances that fall below TF-IDF confidence threshold go
directly to GenAI instead of being rescued. For well-trained languages (fr, de) this
is infrequent but non-zero.

**How to fix (future):** Replace `minilm-l6-v2.onnx` with a multilingual sentence
embedding model (e.g., `paraphrase-multilingual-MiniLM-L12-v2`) and retrain the
logistic head on multilingual intent embeddings. The `SemanticFallback` class in
`scripts/nlu/engine.py` and `MultilingualIntentClassifierService.swift` are already
wired for this — only the model artifact changes.

**Workaround for new languages:** Well-authored keyword triggers (Stage 0) and good
training data (Stage 2) reduce the frequency of reaching Stage 3. Priority intents
(`reminders.add`, `Cmd.MemoryChange`) should always have keyword triggers authored.

---

### 2. Per-language Core ML model export

**What it is:** The iOS production path is `IntentClassifier.mlpackage` — a Core ML
model for fast on-device inference. Only the combined English model has been exported
to `.mlpackage` format.

**Current state:** New languages run on the fallback path: `TFIDFLogisticScorer.swift`
loads `xx_intent_classifier_weights.json` and runs TF-IDF + LogReg in pure Swift.
This is correct and complete; it is slower than Core ML but the difference is
imperceptible for intent classification (< 20ms vs. < 5ms).

**Impact:** Performance only — no correctness difference. The JSON weights path
is already proven in production for all four existing languages.

**How to fix (future):** After training generates `xx_intent_pipeline.pkl`, run
`multilingual/export_coreml.py --language xx` to produce
`multilingual/models/xx/xx_IntentClassifier.mlpackage`. Then add it to the Xcode
bundle. See `multilingual/COREML_EXPORT_IMPLEMENTATION_PROMPT.md` for the full
implementation plan.

---

### 3. Back-reference regexes — English only

**What it is:** Some intents have `back_reference` regex fields in `nlu_schema.json`
that fire when a user says "change it back" or "the previous one" — short follow-up
utterances that reference a prior conversation turn. These are English regex patterns.

**Current state:** The patterns are not localized. Non-English users must state the
full slot value rather than using back-reference shorthand.

**Impact:** Minor UX regression for multi-turn conversations. The system still works;
users just can't say "das vorherige" (the previous one in German) to reference a
prior memory — they must say the full memory name.

**How to fix (future):** Add a `back_reference` array to `nlu_schema.<lang>.json`
with language-specific patterns. The schema merge logic in `LocalizationLoader.swift`
and `engine.py` (`_load_schema`) already merges this key if present.

---

### 4. Keyword triggers — not automatically localized

**What it is:** `keyword_triggers` in `nlu_schema.json` are regex rules that fire
before TF-IDF (Stage 0) and hard-map an utterance to an intent with confidence 1.0.
They are the reliability mechanism for intent-critical utterances that might
otherwise score below threshold.

**Current state:** English keyword triggers are in `data/nlu_schema.json`. French
triggers were manually authored and added to `data/localization/nlu_schema.fr.json`.
German and Danish triggers have not been authored.

**Impact:** Without triggers, German/Danish `reminders.add` utterances rely on TF-IDF
alone. If the utterance uses an unusual phrasing (e.g., "Kannst du mich erinnern…")
and TF-IDF confidence is below threshold, it falls to GenAI.

**How to fix:** Add a `keyword_triggers` array to `nlu_schema.<lang>.json` with
patterns covering all common command verbs for high-priority intents. The French
implementation (`nlu_schema.fr.json`) is a good reference. This is a content task,
not a code task — one or two regexes per intent covering 80% of phrasings is
sufficient.

---

## Known parser traps

These are documented in `data/localization/README.md` and must be honored in the
lexicon for the affected languages. They are NOT bugs — the lexicon drives the
parser, so the lexicon must reflect the language's actual semantics.

| Language | Trap | Correct behavior |
|----------|------|-----------------|
| **de** | `halb drei` | Means 02:30 (half BEFORE three), NOT 03:30. `decimal_hour_idioms` entry must use `{"phrase": "halb", "minutes": -30}` not `+30`. |
| **da** | `halv tre` | Same as German — half BEFORE the named hour. |
| **de** | `dreiviertel drei` | Means 02:45 (three-quarters before three). Regional/dialect — include but gate behind dialect flag if needed. |
| **de** | `viertel drei` | Means 02:15 (quarter past two, Austrian/Saxon). Different from Standard German `viertel nach zwei`. |
| **fr** | `moins le quart` | Subtracts from the NEXT hour. `huit heures moins le quart` = 07:45, not 08:45. |
| **fr** | `midi` / `minuit` | Hard-coded as 12:00 and 00:00 in `decimal_hour_idioms`. Do not add to `time_of_day` as a floating anchor. |
| **All** | Accents in ASR | ASR transcription drops accents inconsistently. Include both accented and unaccented forms for all words (e.g., `février` and `fevrier`, `ü` and `u`). |
