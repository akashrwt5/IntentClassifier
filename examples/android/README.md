# Android reference client

A working Kotlin port of the on-device NLU turn, reading a **format-3.0 pack**.
It exists to answer one question concretely: *what does Android read instead of
`nlu_schema.json`, and what does it do with it?*

This is sample code, not a shipping module — no Gradle wiring, no DI, no
coroutines. The parts that are load-bearing are the ones with comments
explaining what breaks without them.

---

## What Android reads

`nlu_schema.json` is still in the pack, for the reference engine. **Do not read
it from Android.** It is the compiler's input shape: one blob mixing platform
config with content, with no per-capability versioning. Format 3.0 ships it
decomposed, and the decomposed files are the client contract.

| Purpose | File |
|---|---|
| Thresholds, limits, per-intent confirmation | `runtime/policies.json` |
| Ordered keyword pre-filter | `keywords/<lang>.json` |
| Yes/no words, contractions | `lexicons/<lang>.json` |
| Help-marker + polarity redirects | `runtime/guards.json` |
| Intent → capability (availability) | `runtime/plan_facts.json` |
| Which stages are on | `runtime/cascade.json` |
| Action keys (→ `NLUActionKey.kt`) | `capabilities/<cap>/capability.json` |
| Slots, confirmation, completion action | `capabilities/<cap>/workflows.json` |
| User-visible strings | `capabilities/<cap>/responses/<lang>.json` |
| Model | `models/intent/<lang>/intent_classifier_weights_full.json` |

The split is what makes a new language a content change: only
`responses/<lang>.json`, `keywords/<lang>.json` and `lexicons/<lang>.json` move.
No client logic changes.

---

## Files

```
TextNormalizer.kt         lowercase, expand contractions, drop apostrophes
TfidfIntentClassifier.kt  sublinear TF-IDF + LR + temperature-scaled softmax, OOV ratio
KeywordMatcher.kt         ordered regex rules, first match wins, no confidence
Guards.kt                 help-marker and polarity redirects
NluBundle.kt              loads all of the above from the pack
NluEngine.kt              the decision ladder
GoldenParityTest.kt       proves this Kotlin computes what the Python computes
```

---

## The ladder, whole

```
conf >= fireBar   ->  the intent fires
conf <  fireBar   ->  Default Fallback Intent, app routes to GenAI
```

There is no third tier and **no confidence band that asks "just to be sure"**.
That mechanism existed, sat *above* the fire threshold, and converted commands
that would have fired into questions: on the honest holdout, 103 friction turns
against 16 useful catches — 85% of every confirmation a user saw was asked about
a **correct** prediction, and `increase volume` was held for confirmation while
the model scored it 0.9992.

Confirmation still exists, but only where a human **authored** it per intent in
`workflows.json`. In this taxonomy that is exactly one intent —
`Cmd.SendMessage`, the single irreversible externally-visible action. Do not
reintroduce a confidence-triggered ask on the client.

`fireBar` has one exception: when the keyword rule and the model independently
name the **same** intent, the bar drops from `thresholds.confidence` (0.70) to
`thresholds.agreement` (0.50). Two independent recognisers agreeing is stronger
evidence than either alone.

---

## Three ways to get this wrong

### 1. Mixing a temperature with the wrong weights

The pack carries three, and each is correct for its own head:

| File | Vocab | T |
|---|---|---|
| `models/intent/en/calibration.json` | ONNX / server | 0.671 |
| `intent_classifier_weights.json` | 1592 (pruned) | 0.822 |
| `intent_classifier_weights_full.json` | 5896 (full) | 0.544 |

Pair the wrong one and everything compiles, runs, and picks the **right intent** —
temperature is rank-preserving, it never changes the argmax. Only every
confidence is wrong, and confidence is what the entire ladder is made of. This
shipped once (blocker B8). `NluBundle.loadWeights` takes one path and returns one
matched set for exactly this reason, and `GoldenParityTest` asserts the pairing.

### 2. Shipping the pruned head

Use **`_full.json`**. It costs 2 MB more and buys two things:

- out-of-scope utterances reaching a device action: **15.4% → 8.7%**
- the OOV guard keeps working

That second one is sharp. `thresholds.oov_reject = 0.25` was fitted against a
**1472-unigram** vocabulary. The full head has exactly that many unigrams; the
pruned head has **653**. Measured on the honest holdout:

| Head | unigrams | guard fires on real commands |
|---|---|---|
| full | 1472 | 34 / 1275 (2.7%) |
| pruned | 653 | **158 / 1275 (12.4%)** |

Ship the pruned head with the shipped threshold and you refuse one real command
in eight. Nothing would report it.

### 3. Skipping normalization

The vocabulary was built from normalized text, so `what's my battery` was fitted
as `what is my battery`. Hand the raw form to the tokenizer and it splits as
`["what", "s", "my", ...]` — `"s"` has no slot, and neither do the bigrams
`"what s"` / `"s my"`. Measured on the honest holdout (full head):

| | accuracy |
|---|---|
| raw lowercase | 0.9184 |
| normalized first | 0.9204 |

Nine predictions differ. `what's my battery level` goes 0.9966 → 0.9999. Small,
free, and it grows with the share of contracted speech — ASR contracts far more
than written training data does.

---

## Parity, and its limits

`golden_vectors.json` is generated from the shipped weights by
`scripts/gen_android_golden.py`, which is a transcription of
`nlu_export/export_ios_weights.py::_device_logits` — the function the temperature
was fitted on. Regenerate it after every retrain:

```bash
PYTHONPATH=packages/buildtime:packages/runtime python scripts/gen_android_golden.py
```

This port was verified by transcribing the Kotlin back into Python and replaying
the full 1470-row honest holdout against the live `NLUEngine`:

```
kotlin port == engine       1443 / 1470   (98.16%)
end-to-end accuracy kotlin  0.9109
end-to-end accuracy engine  0.9068
```

**The 27 disagreements are not port bugs.** Every one has the same shape: the
device head (T=0.544) is slightly more confident than the ONNX head (T=0.671),
so a turn sitting near 0.70 falls on opposite sides of the bar.

```
"music memory"                     kotlin FULFILL/Cmd.MemoryChange @0.770
                                   engine FALLBACK               @0.694
"what does smallest state in new zealand"
                                   kotlin FULFILL/Help_WhatsNew  @0.979
                                   engine FALLBACK               @0.934
```

That second one is the direction to worry about: an out-of-scope query the
server deflects and the device fires. **Device and server are two different
models and will not agree at the margin.** Treat any measurement taken on one as
not transferable to the other — and note that `wrong_action_harness` currently
measures the ONNX path only, so the device path has no equivalent number yet.

---

## Not covered here

- **Signature verification.** Verify `checksums_root` and the manifest before
  `NluBundle.load`. A loader that skips it makes the signing decoration.
- **Slot flows and entity resolution.** `resumePending` is a sketch; the real
  datetime/entity grammar lives in `lexicons/<lang>.json` and is substantial.
- **Semantic stage.** `runtime/cascade.json` reports it disabled; there is no
  MiniLM path in this sample.
- **Telemetry.** Emit the domain, never the intent — `Help_Tinnitus` and
  `Help_FallAlert` are health-adjacent and this product's users are a clinical
  population. See `nlu_engine/telemetry.py::domain_of`.
- **Startup cost.** `org.json` over a 2.78 MB weights file is not free. Measure
  it; if it matters, convert `coef`/`idf` to a flat binary at build time and
  memory-map it. Do not "optimize" by pruning the vocabulary — see §2.
