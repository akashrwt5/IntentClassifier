# Host label contract (`runtime/confirmation_labels.json`)

The trained model and the bundle's `labels.json` speak one taxonomy, and today
that taxonomy is already the one the app speaks (`Cmd.VolumeIncrease`, `Help_*`).
Two things still do not line up on their own, and they are handled in two
different places — which is the point of this document.

| Fact | Where it is declared | Where it is applied | Ships to devices? |
|---|---|---|---|
| modern label → legacy app label (`map`) | `packages/runtime/nlu_engine/legacy_label_map.json` | reference engine only, behind `NLU_LEGACY_LABELS=1` | **no** |
| resolved confirmation → host label (`confirm_compound`) | same file, projected into the bundle | every client, at its output boundary | **yes** |

## Why the map does not ship

`map` is IDENTITY for all 57 intents: the model already emits the names the app
speaks, so it renames nothing. It is kept, and kept complete, because it is the
place where "what does the app call this intent?" is declared — a new intent with
no answer fails `test_every_trained_intent_has_a_legacy_mapping` rather than
reaching the app under whatever the trainer happened to name it.

But that gate reads the **repo source**, never the bundle. Shipping the map put
57 lookups that return their own input on every device, 2.6 KB, to satisfy a
build-time check that never opened them. So the build projects only the half that
carries information. Re-add a device-side map when a host actually needs a
rename; that is a different artifact and a different decision.

## Why the confirmation labels do

A confirmation turn carries two facts — which intent was confirmed, and how it
resolved — and this host encodes that pair as a single name:
`Cmd.SendMessage - yes`. Those strings cannot come from anywhere else:

- **Not from the classifier.** The head has 57 classes and neither compound is
  among them. Neither could be: polarity is not decided until after the
  classifier has spoken.
- **Not from a workflow.** `capabilities/*/workflows.json` is host-neutral dialog
  structure, its intent object is `additionalProperties: false`, and this string
  is a host naming contract with a different lifetime — it is meant to be deleted
  when the app migrates.
- **Not composed by a client.** Neither runtime may interpret or build an intent
  label; that invariant is what lets the taxonomy change without touching engine
  code.

So they are content, in their own artifact, deletable in one step.

## The file

`runtime/confirmation_labels.json`
(schema: `spec/bundle/3.0/confirmation_labels.schema.json`):

```json
{
  "generated_from": "packages/runtime/nlu_engine/legacy_label_map.json",
  "confirm_compound": {
    "Cmd.SendMessage": { "yes": "Cmd.SendMessage - yes", "no": "Cmd.SendMessage - no" }
  }
}
```

Optional in both directions. No `confirm_compound` in the source writes no file,
and a client that finds no file reports the plain intent — the host can still
tell the branches apart by the completion action (`message.send` vs
`message.cancel`). Omit it entirely once the app consumes plain intent labels.

Both polarities are required per intent. Naming only one leaves the two branches
of the same turn reporting different *shapes* of label, which is worse than
naming neither.

## The algorithm (apply at the engine's output boundary)

When a turn **resolves a yes/no confirmation** — your engine knows the polarity
and which intent was under confirmation, so carry both, even when the surfaced
intent is `sys.confirm.cancelled` — and `confirm_compound` has that intent,
override the reported intent with the matching `yes`/`no` string.

```
func reported(intent, polarity):
    return confirm_compound[intent]?[polarity] ?? intent
```

That is the entire device-side contract. Everything internal to the engine —
classification, slots, confirmation logic, telemetry — stays in the modern label
space; only the outward string changes, and only on this one kind of turn.

## Where each runtime does it

- **Python (reference):** `packages/runtime/nlu_engine/label_compat.py`, applied
  on the way out of `handle()`. It reads the source file directly, so it also
  applies `map`. **Opt-in — `NLU_LEGACY_LABELS=1`, off by default**, so internal
  callers and the test suite are never silently relabeled.
- **Swift:** `NLUEngine.reported(_:polarity:)`, applied in `handleConfirmation`
  only. Fed from `ResolvedPack.confirmationLabels`, which `BundleDataLoader`
  reads from `runtime/confirmation_labels.json` — that path only. The pre-split
  `runtime/legacy_labels.json` is NOT tried as a fallback: no pack carrying it
  reached a device, so reading it would be code for a case that does not exist.
  A pack that ever needs this artifact somewhere else declares its own paths in
  `bundle.json`, the way `ModelSpec` does; a list of guessed filenames in the
  loader is not the mechanism.

The two defaults differ: Swift applies the labels whenever the pack ships them,
Python only under the env flag. Align them before recording any parity fixture
that covers a resolution turn.

## Reference & conformance

- **Conformance vectors:** `tests/fixtures/legacy_label_parity_en.csv` — input →
  expected output, including the two-turn send-confirmation compound. A client is
  correct when it reproduces these strings. Regenerate with
  `python scripts/gen_legacy_label_fixtures.py`.
- **Artifact guards:** `tests/test_confirmation_labels_artifact.py` asserts what
  the bundle ships and what it must not ship back.
- **Single source of truth:** `packages/runtime/nlu_engine/legacy_label_map.json`.
  `content_bundle.compile_confirmation_labels` projects `confirm_compound` from
  it into every bundle, so the reference engine and every client read the same
  strings.
