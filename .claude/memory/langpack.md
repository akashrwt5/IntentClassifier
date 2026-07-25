# Memory: Language Pack Architecture

> Single responsibility: the language-neutral engine boundary — the pack
> contract, the `packs/` layout, and the rules CI enforces. This is the
> defining architecture of this branch. Plan: `docs/Review-F5/IMPLEMENTATION-PLAN.md`
> §9; live status: `docs/Review-F5/IMPLEMENTATION-PROGRESS.md`.

## The contract

`packages/nlu_langpack/` is the **locked boundary** between the
language-agnostic engine and everything language-specific.

| Module | Role |
|---|---|
| `interfaces.py` | Engine-facing component Protocols (the contract) |
| `version.py` | `RUNTIME_CONTRACT_VERSION = "1.0"` + compatibility gate |
| `manifest.py` | `pack.json` parsing + validation |
| `pack.py` | `LanguagePack` container returned to the engine |
| `loader.py` | `load_pack()`: validate → compat gate → config → resources → semantic gate |
| `errors.py` | Loud, specific pack error taxonomy |

`packages/` must be on `sys.path` — the engine inserts it itself
(`NLUEngine._load_pack`), and `tests/` does it in module scope. From a shell use
`PYTHONPATH=packages`.

```python
from nlu_langpack import load_pack
pack = load_pack("packs/en")                    # semantic OFF by default
pack = load_pack("packs/en", enable_semantic=True)
pack.language, pack.stages, pack.semantic_available
# packs/en -> ('en', ['keyword', 'intent_model'], False)
```

`NLUEngine.__init__(..., language="en", pack=None)` substitutes
`packs/<language>` when `pack` is None — that, and nothing else, is why a bare
`NLUEngine()` is English. `pack` also accepts a `LanguagePack` instance or a
directory path.

## Pack layout (`packs/en/` is the reference)

```
packs/en/
  pack.json              # id, language, format_version, engine_compat, components, models
  config.json            # thresholds, stages, policy
  schema.json            # 59 intents + 32 keyword_triggers + affirmative/negative
  keywords.json          # keyword rules (evicted from the engine verbatim)
  lexicons.json          # carriers, yes/no, idioms, connectors
  normalizer.json
  entities/enums.json
  datetime/grammar.json  # THE ENTIRE datetime vocabulary (see below)
  intent_model/          # model.onnx, labels.pkl, labels.json, weights.json, calibration.json
  semantic/              # minilm-l6-v2.onnx, minilm-vocab.txt, semantic_head.npz
```

Policy knobs: `confidence_threshold` **0.75**, `slot_confidence_threshold` 0.60,
`semantic_enabled` **false**, `semantic_threshold` **0.40**, `stages`
`["keyword","intent_model"]`, `interrupt_threshold` 0.75, `agreement_threshold`
0.50, `max_slot_attempts` 3, `context_ttl_seconds` 90, `session_ttl_seconds` 600,
`non_interrupting_actions` `["help."]`.

> **Which file wins:** the engine reads `confidence_threshold`,
> `slot_confidence_threshold` and `semantic_threshold` from **`schema.json`**.
> `config.json` mirrors them for readability and those copies are **ignored** —
> editing them has no effect. `tests/test_calibration.py` fails if the two
> diverge, because they silently had: `semantic_threshold` was 0.55 in
> `config.json` while the engine used **0.40** from `schema.json`. The policy
> block (`interrupt_threshold` etc.) *is* read from `config.json`.

`non_interrupting_actions` lists **action-ID prefixes** that may never abandon
an in-progress slot flow, however confident the classifier is (a help question
scores ~0.99, so this cannot be a threshold). Matched on the action ID — a
stable identifier, never display text — so a new pack inherits it for free.

`lexicons.json` keys: `affirmative`, `negative`, `uncertainty`, `no_idioms`,
`carriers`, `leading_connectors`, `negations`. `negations` feeds the keyword
negation guard for **both** `contains` and `regex` rules.

## Neutrality rules (CI-enforced — do not break)

**The contract: adding a language = authoring `packs/<lang>/` + training data.
Nothing else.** If a task makes you edit `scripts/nlu/` to support a language,
the gap belongs in the pack.

1. **Zero `if language ==` / `!=` in `scripts/nlu/`.** `grep` must return no
   code matches. Guard: `scripts/ci/check_language_neutral.py` (ignores
   comments; verified against decoy comment lines).
1b. **No hardcoded match vocabulary in `scripts/nlu/`.** Same guard, second
   check. A module-level collection of natural-language phrases must be named
   `_DEFAULT_*` — the convention for a fallback table a pack overrides
   (`_DEFAULT_DT_GRAMMAR`, `_DEFAULT_NEGATIONS`). Canonical role keys are fine
   (`_WD_ORDER` "Monday", `_ANCHOR_OFFSET` "day_after_tomorrow", tier names) —
   those are stable identifiers a pack maps its own words onto, not text matched
   against speech. This check exists because `_NEGATIONS`, an English-only tuple
   in `classifier.py`, made negation suppression a silent no-op for every
   non-English pack and check 1 could not see it.
2. **Zero English datetime words in regex literals.** Day anchors, periods,
   weekdays, word-numbers, am/pm (incl. dotted `a.m`), clock idioms (half past /
   quarter to / N past M / N to M), relative markers/units/quantifiers, and the
   topic-strip function words all come from the pack's `datetime/grammar.json`.
   The only English left is the consolidated `_DEFAULT_DT_GRAMMAR` fallback
   **table** (data, overridden by the pack) plus schema role-keys.
3. **A hostile `zz` pack must run end-to-end with zero engine edits.** This is
   the real neutrality test — `tests/test_neutrality.py`.
4. **Semantic is a pack-declared, off-by-default stage.** Precedence:
   arg → env → config → default(False). Enabling semantic on a pack that
   declares none: arg = **hard error**; env/config = **warning**, stage left
   unavailable (a broad switch must not crash packs lacking the stage).

## Behaviour-preservation baselines

Any engine change must hold these (they are how the eviction was proven safe):

- Classifier parity **37/37**; full `handle()` parity 37/37 with semantic on.
- English datetime golden corpus **77/77** on *both* the default extractor and
  the pack-fed engine (`tests/datetime_parity/nlu_datetime_parity_en_golden.json`,
  enforced by `tests/test_datetime_parity_en.py`, 154 assertions).
- Topic-strip **20/20** both paths.
- fr/de/da datetime parity unchanged.
- Suite at exactly the 8 tracked pre-existing failures — no more.
- Hostile `zz` pack parses "half past 9 tomorrow".

## Status (plan §9)

Steps **1–6 all complete**: contract + loader, language eviction, keyword rules
+ datetime grammar into pack tables, semantic pack-declared/off-by-default,
`pr.yml` neutrality CI, and automated train-gate-release.

**Next checkpoint — requires explicit go-ahead** (it is the first step that
moves existing engine code):
1. Big-bang move `scripts/nlu` → `packages/nlu_engine` (wholesale; update
   importers/tests). No dual-run.
2. `RUNTIME_CONTRACT_VERSION`-aware engine constructor taking a `LanguagePack`.
3. Evict any residual `engine.py` language constants into `lexicons.json`.
4. Wire semantic fully as a pack-declared plugin.

## Adding a language

The pack layout is the whole story: author `packs/<lang>/` with the same files,
no engine edits. Walkthrough: `docs/Review-F5/LOCAL-TESTING-AND-NEW-LANGUAGE-GUIDE.md`
and `docs/adding-a-new-language.md`. See also the `/add-language` command.

## Related memory

Architecture -> `architecture.md` · Inference -> `inference.md` ·
Decisions -> `decisions.md` (ADR-008 pack contract, ADR-009 big-bang move) ·
Roadmap -> `roadmap.md`.
