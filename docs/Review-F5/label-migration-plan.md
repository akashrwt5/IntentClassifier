# Label-Space Cleanup & Taxonomy Migration Plan (ND-3 proposal)

Status: **PROPOSAL — awaiting owner approval.** No label, data, or model
change happens until approved. Per roadmap §2.3/R6, ADR-002 A7.

## Recorded baseline (regen run, 2026-07-14, 59-label space)

| Model | Holdout acc | Macro-F1 |
|---|---|---|
| en | 0.900 | 0.89 |
| fr | 0.852 | 0.84 |
| de | 0.833 | 0.82 |
| da | 0.760 (gate-failing) | 0.73 |
| multilingual | 0.845 | 0.82 |
| multilingual_small | 0.848 | 0.83 |

Any post-migration retrain must hold these within noise (±0.01) except da
(tracked separately). The unified evaluate JSON becomes the comparison
artifact.

## Change 1 — remove the two dialogue-act labels

`Cmd.SendMessage - yes` and `Cmd.SendMessage - no` are **conversation states,
not intents**: they exist only because Dialogflow modeled confirmation as
classification. The NLU engine already resolves confirmations first via
yes/no lexicons before classifying (engine turn priority), so these labels
are (a) redundant and (b) harmful — they steal probability mass from
`Cmd.SendMessage` and can fire outside any confirmation context.

Plan: drop both labels and their training rows (rows re-labeled into the
affirmative/negative lexicon fixtures + confirmation regression tests, not
discarded). 59 → 57 classes.

## Change 2 — Default Fallback Intent (decision needed)

Options:

- **(A) Keep as an explicit trained OOS class, renamed `sys.oos.fallback`**
  (recommended). The semantic head already models OOS explicitly; keeping a
  trained TF-IDF fallback class preserves current OOS recall behavior and is
  the ADR-005 golden-bundle convention (`sys.oos`).
- (B) Remove it and rely purely on confidence thresholds. Cleaner label
  space, but changes OOS recall characteristics — higher risk to the
  wrong-action budget.

> **Amendment (2026-07-14, supersedes the family table below where they
> differ):** the authoritative new names live in
> `docs/Review-F5/capability-map.json`, which aligns intent domains with the
> ratified ADR-002 §A3 capability decomposition (e.g. `device.volume.increase`
> not `audio.volume.increase`; `activity.steps.query` not
> `health.activity.step`). The map is machine-checked to cover all 59 current
> labels exactly. Everything else in this plan (baseline, gates, execution
> steps, risks, the A/B decision) is unchanged.

## Change 3 — taxonomy migration to domain.object.action

Mechanical rules, then the full map is generated as
`data/label_migration_map.json` (old → new, exhaustive, checked by CI so no
label is silently dropped):

| Family (rule) | Examples |
|---|---|
| `Cmd.Volume*` → `audio.volume.<action>` | Cmd.VolumeIncrease → audio.volume.increase · Mute/Unmute/Decrease likewise |
| `Cmd.MemoryChange` → `audio.memory.change` | |
| `Cmd.Streaming*` → `audio.streaming.<action>` | start/stop |
| `reminders.*` → `reminder.task.<action>` | reminders.add → reminder.task.create · complete |
| `Cmd.SendMessage` → `message.text.send`; `Cmd.ListenMessage` → `message.text.listen` | |
| `Cmd.FindMyPhone` → `phone.locator.find` | |
| `Cmd.BatteryLevel` → `device.battery.query` | |
| `Cmd.TranslationStart` → `assist.translate.start`; `Cmd.TranscribeStart` → `assist.transcribe.start` | |
| `Cmd.Activity*` → `health.activity.<metric>` | run/walk/step/stand/cycle/aerobics/exercise/calories |
| `Help_*` (33 labels) → `help.<topic>.show` | Help_Battery → help.battery.show · Help_FindMyHearingAids → help.find_hearing_aids.show … |
| `Default Fallback Intent` → `sys.oos.fallback` (per Change 2A) | |

Notes: renames only — **no merges, no splits**, so per-class metrics are
directly comparable pre/post. Intent *count* changes only via Change 1 (−2).

## Execution steps (after approval)

1. Emit `data/label_migration_map.json` + a migration script that rewrites
   training/holdout/OOS CSVs and `nlu_schema.json` keys (values untouched).
2. Move `- yes`/`- no` rows into confirmation-lexicon fixtures; add
   confirmation regression tests (yes/no during and outside a confirmation).
3. Retrain all languages; run full gates: holdout `--strict`, OOS recall,
   per-intent F1 floors, wrong-action budget, calibration ECE; compare to the
   baseline table above via the evaluate JSON.
4. Regenerate keyword pre-filter mapping (schema-driven — no code change).
5. **Coordinated app-side change:** iOS consumes `labels.json` + golden
   fixtures; ship the migration map to the STT repo and regenerate fixtures
   in the same release window (ADR-002 A7: `superseded_by` covers stragglers).
6. Update memory (`datasets.md`, `decisions.md`) + spec golden bundles remain
   the naming reference.

## Risks

- Metric noise from retrain randomness — mitigated by fixed seeds (already in
  trainers) and the ±0.01 hold rule.
- iOS desync if fixture regeneration lags — mitigated by shipping map +
  fixtures atomically; parity CI (once ND-7 unblocks) guards it.
- Danish is excluded from the hold rule but included in the rename (it must
  not diverge structurally).

**Decision requested:** approve Changes 1+3, and pick option A or B for
Change 2.
