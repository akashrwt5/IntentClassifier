# Legacy label contract (`runtime/legacy_labels.json`)

The trained model and the bundle's `labels.json` speak the modern
`domain.object.action` taxonomy (ND-3). Some app builds still consume the legacy
Dialogflow label space (`Cmd.VolumeIncrease`, `Help_*`, `Cmd.SendMessage - yes`).
Rather than change the model, or ship a per-platform adapter, the bundle carries
**one JSON map** and each native client translates at its own engine-output
boundary. iOS and Android implement the identical ~10-line algorithm below.

This is opt-in: if `runtime/legacy_labels.json` is absent from the bundle, the
app already consumes modern labels and no translation is needed.

## The file

`runtime/legacy_labels.json` (schema: `spec/bundle/3.0/legacy_labels.schema.json`):

```json
{
  "map": { "device.volume.increase": "Cmd.VolumeIncrease", "help.volume.show": "Help_Volume", "…": "…" },
  "confirm_compound": {
    "messaging.message.send": { "yes": "Cmd.SendMessage - yes", "no": "Cmd.SendMessage - no" }
  }
}
```

- **`map`** — modern label → legacy app label. Any label not present is passed
  through unchanged (a new intent degrades to its modern name, never dropped).
- **`confirm_compound`** — for intents whose legacy contract encoded the yes/no
  *dialogue act* as a distinct label. Keyed by the intent that was under
  confirmation. These are **not** model labels — the classifier never emits them;
  the client reconstructs them from a resolved confirmation outcome.

## The algorithm (apply at the engine's output boundary)

For every result the engine hands to the app:

1. Translate each label-bearing field — `intent`, `interruptedIntent`,
   `tfidfIntent` — through `map`, passthrough if absent.
2. If this turn **resolved a yes/no confirmation** (your engine knows the
   polarity and which intent was being confirmed — carry both, even when the
   surfaced intent is `sys.confirm.cancelled`), and `confirm_compound` has that
   intent, **override** `intent` with the matching `yes`/`no` string.

Pseudocode:

```
func toLegacy(result):
    result.intent            = map[result.intent]            ?? result.intent
    result.interruptedIntent = map[result.interruptedIntent] ?? result.interruptedIntent
    result.tfidfIntent       = map[result.tfidfIntent]       ?? result.tfidfIntent

    if result.confirmPolarity != null and result.confirmedIntent != null:
        compound = confirm_compound[result.confirmedIntent]
        if compound != null:
            result.intent = compound[result.confirmPolarity]   // "yes" | "no"
    return result
```

That is the entire contract. Everything internal to the engine — classification,
slots, confirmation logic, telemetry — stays in the modern label space; only the
outward strings change.

## Reference & conformance

- **Executable spec:** the Python reference engine implements exactly this in
  `packages/runtime/nlu_engine/label_compat.py` (enabled by `NLU_LEGACY_LABELS=1`).
- **Conformance vectors:** `tests/fixtures/legacy_label_parity_en.csv` lists
  input → expected legacy output (including the two-turn send-confirmation
  compound). A client is correct when it reproduces these strings. Regenerate
  with `python scripts/gen_legacy_label_fixtures.py`.
- **Single source of truth:** `packages/runtime/nlu_engine/legacy_label_map.json`.
  The build (`content_bundle.compile_legacy_labels`) copies it verbatim into
  every bundle as `runtime/legacy_labels.json`, so the reference engine and all
  clients read the same table.
