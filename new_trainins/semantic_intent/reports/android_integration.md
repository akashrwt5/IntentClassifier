# Android Offline Integration Contract

Phase 27 of the plan. This describes what the app receives and what it must do
with it. No network call exists anywhere in this path.

## Runtime chain

```text
Microphone
   |
Offline STT --------------------------------.
   |                                         |
Text                                   utterance confidence
   |                                         |
Normalizer          <- must match            |   the only gate signal that
   |                   common.py::normalize  |   does not read the words.
WordPiece tokenizer <- onnx/tokenizer/vocab  |   Do not discard it.
   |                                         |
ONNX model          <- encoder + pooling + classifier + temperature + softmax
   |                                         |
Safety gate <-------------------------------'
   |   thresholds from onnx/runtime_config.json
   |
Intent  or  Fallback
   |
Kotlin command handler
   |
Hearing aid / Android API
```

## What is inside the ONNX graph, and what is not

| stage | where it lives | why |
|---|---|---|
| tokenization | Kotlin, from `vocab.txt` | keeps the graph portable; ORT-Extensions is optional, not required |
| encoder + mean pooling + L2 norm | ONNX | one graph, no glue code to drift |
| classifier head | ONNX | fused as a Gemm |
| temperature scaling | **ONNX** | if calibration lives in app code it will eventually drift out of sync with the weights, and then the number the gate reads is not the number that was validated |
| confidence / margin thresholds | `runtime_config.json` | product can retune the safety trade-off without re-exporting a model |

## Model inputs and outputs

```text
input   input_ids            int64   [batch, seq]   padded to max_len (64)
input   attention_mask       int64   [batch, seq]
output  probs                float32 [batch, 57]    already calibrated
output  whitened_embedding   float32 [batch, dim]   for the OOD score
```

**Opset is 18, not 17.** The exporter cannot down-convert this graph and says
so; the file stays at opset 18. ONNX Runtime Mobile in the app must support
opset 18 or the model will not load at all. Check this before anything else —
it is a hard load failure, not a quality regression.

Label order is `runtime_config.json -> labels`. Never hardcode it in Kotlin;
read it from the file that shipped with the weights.

## Normalization must match training

The Kotlin normalizer has to reproduce `scripts/common.py::normalize`:
NFKC, lowercase, trim, expand the contraction table, strip punctuation except
apostrophes, collapse whitespace. A mismatch here is silent: nothing crashes,
accuracy just quietly drops on contractions and punctuation.

If the selected encoder carries a prefix (E5 uses `query: `),
`runtime_config.json -> tokenizer.prefix` holds it and it must be prepended
**before** tokenization.

## The safety gate

```kotlin
data class IntentResult(
    val intent: String,
    val confidence: Float,
    val top2Intent: String,
    val top2Score: Float,
    val margin: Float,
    val accepted: Boolean,
    val reason: String,
)

fun decide(
    probs: FloatArray,
    cfg: RuntimeConfig,
    asrConfidence: Float? = null,   // signal 6 — see below
): IntentResult {
    val order = probs.indices.sortedByDescending { probs[it] }
    val i1 = order[0]; val i2 = order[1]
    val c1 = probs[i1]; val c2 = probs[i2]
    val margin = c1 - c2

    // OOD score: min euclidean distance to the stored whitened centroids.
    // The whitening is already inside the graph, so this is the whole thing.
    val oodScore = cfg.ood.whitenedCentroids.minOf { c ->
        var acc = 0f
        for (k in whitenedEmbedding.indices) {
            val d = whitenedEmbedding[k] - c[k]; acc += d * d
        }
        kotlin.math.sqrt(acc)
    }

    // Per-intent threshold: muting an aid and nudging the volume are not the
    // same bet. cfg.gate.riskOf lists only the high-risk intents.
    val tier = cfg.gate.riskOf[cfg.labels[i1]] ?: "normal"
    val confThreshold = cfg.gate.confByRisk[tier] ?: cfg.gate.confThreshold

    // Corrective phrasing: the sentence rejects one option and asks for
    // another ("not edge mode, i meant mask mode"). Measured accuracy on this
    // shape is 0.48-0.74, well under the precision this gate promises, so it
    // is refused rather than guessed. Run it on the RAW lowercased text, not
    // the normalizer's output — the comma is the structural signal and the
    // normalizer strips punctuation.
    val corrective = cfg.gate.rejectCorrective &&
        Regex(cfg.gate.correctivePattern).containsMatchIn(rawText.lowercase())

    // Recognizer confidence, checked FIRST. Every other branch reasons about
    // what the words mean; this one asks whether they were heard, and whether
    // the person was talking to the device at all. Null until
    // fit_asr_threshold.py has been run on recordings from this hardware —
    // and null is inert, so shipping before that changes nothing.
    val asrTooLow = cfg.asr?.enabled == true &&
        asrConfidence != null && asrConfidence < cfg.asr.minConfidence

    val (accepted, reason) = when {
        asrTooLow ->
            false to "recognizer was unsure it heard this"
        cfg.labels[i1] == cfg.gate.rejectLabel ->
            false to "classified as unsupported"
        corrective ->
            false to "corrective phrasing — ask the user to repeat"
        cfg.oodThreshold != null && oodScore > cfg.oodThreshold ->
            false to "unlike anything in training"
        c1 < confThreshold ->
            false to "below calibrated confidence threshold ($tier risk)"
        margin < cfg.gate.marginThreshold ->
            false to "top-1/top-2 margin too small"
        else -> true to "above calibrated threshold"
    }
    return IntentResult(cfg.labels[i1], c1, cfg.labels[i2], c2,
                        margin, accepted, reason)
}
```

Six reasons to refuse. Four of them read the same softmax; only the OOD score
and the recognizer confidence are independent of it:

- **recognizer confidence** catches audio the ASR was unsure it heard, and
  speech that was never aimed at the device. It is the only signal that does
  not look at the words. Inert until fitted — see below
- **reject label** catches requests the model recognises as unsupported
- **confidence** catches requests it half-recognises, at a threshold that
  depends on how much damage that intent does when it fires wrongly
- **OOD score** catches input unlike anything in training. This is the only
  signal computed before the classifier. A softmax compares the 57 known
  classes against each other — it answers "which of these", never "is it any of
  these" — so it can be completely confident about input that resembles nothing
  it was trained on
- **margin** catches the dangerous case confidence alone misses:

```text
Cmd.VolumeIncrease = 0.51
Cmd.VolumeDecrease = 0.48
margin             = 0.03
```

Top-1 confidence of 0.51 looks like a decision. It is a coin flip between two
opposite commands, and a hearing aid should ask again rather than guess.

## Only ACCEPT triggers hardware

```text
accepted = true   -> execute the command
accepted = false  -> fallback: ask the user to repeat, or say it is unsupported
```

### Risk tiers

Five intents carry a stricter threshold because a mistake there is one the user
may not be able to notice or undo:

| intent | why |
|---|---|
| `Cmd.VolumeMute` | removes the channel the user would use to notice the error |
| `Cmd.SendMessage` | reaches another person |
| `reminders.complete` | clears something the user may be relying on |
| `Cmd.StreamingStop` | cuts audio they were actively listening to |
| `Cmd.MemoryChange` | silently changes the sound profile |

`Cmd.VolumeUnmute` is deliberately NOT in this list: it is the recovery action.
Firing it wrongly is loud, obvious and instantly reversible — the opposite of
the failure mode we are guarding against.

### Signal 6: wiring up recognizer confidence

The remaining false executions are mostly ASR fragments: text like
"and push it down for dramatics" that the recognizer produced while the user
was talking about something else. That string is ordinary English and it really
does sit near the volume-down region, so it is not out-of-distribution and no
text-only signal separates it (measured OOD AUROC on STT noise: ~0.70, versus
~0.92 on genuine OOD).

The signal that does separate it is the recognizer's own confidence, and the
gate now has a branch for it. Two things are needed on the app side:

**1. Stop discarding the score.** Read the per-utterance confidence, not a
per-word average — per-word averages are usually flat:

```kotlin
override fun onResults(results: Bundle) {
    val text  = results.getStringArrayList(
        SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull() ?: return
    val conf  = results.getFloatArray(
        SpeechRecognizer.CONFIDENCE_SCORES)?.firstOrNull()
    // conf may be null: not every engine or locale populates it.
    // Null flows through as "no opinion" and the branch stays inert.
    val result = decide(runModel(text), cfg, conf)
}
```

**2. Fit the threshold on your hardware.** There is deliberately no default.
Confidence scales are not comparable between recognizers — Android returns
0..1, Whisper returns a negative `avg_logprob`, Vosk returns a per-word mean —
so a number that rejects 5% of good input on one stack rejects 60% on another.
Record about 120 utterances following `reports/asr_confidence_protocol.md`
(roughly an hour), then:

```bash
python scripts/fit_asr_threshold.py --data data/asr_samples.csv \
       --apply models/final_student_256/onnx
```

That writes an `asr` block into `runtime_config.json`. The model file is not
touched — this is a threshold, not a weight, and like the other thresholds it
lives outside the graph because it is a property of your audio stack rather
than of the classifier. Refit when the recognizer, its version, or the
microphone changes.

If the fitter reports AUROC below 0.60 it will refuse to write a threshold.
That is an answer: the score you exported carries no information about whether
the person was addressing the device.

**The shortcut.** Push-to-talk or a wake word removes this failure mode
entirely, by only ever classifying audio the user meant as a command — no
recording, no fitting, no threshold to maintain. If the product can have a
button, signal 6 is not worth the hour.

Never execute on a rejected result, and never "downgrade" a rejected result to
a best guess. For volume this feels harmless; for mute, program switching and
message sending it is not.

## Threading and lifecycle

- Load the ORT session once and keep it; session creation is the expensive part
- Run inference off the main thread
- Reuse a single input buffer sized `[1, 64]`; there is no batching at runtime
- Expect single-sentence latency in the low tens of milliseconds on a mid-range
  phone for a 6-layer 384-d encoder; measure on the real target rather than
  trusting the desktop numbers in `reports/benchmark.md`

## Shipping checklist

- [ ] `intent_int8.onnx` passes `scripts/parity_test.py` against Python, including **gate agreement**, not only max-abs-delta
- [ ] `vocab.txt` in the APK is byte-identical to the one used at export
- [ ] Kotlin normalizer output matches `normalize()` on a shared fixture file
- [ ] `runtime_config.json` labels match the training label order
- [ ] airplane mode test: full pipeline works with no network
- [ ] ONNX Runtime Mobile build supports **opset 18**
- [ ] high-risk intents use `confByRisk["high"]`, and the list is read from
      `runtime_config.json` rather than hardcoded in Kotlin
- [ ] OOD score matches Python on a shared fixture (the graph does the
      whitening; Kotlin must only do the distance)
- [ ] corrective regex runs on RAW lowercased text, never on normalizer output
- [ ] rejected results provably cannot reach the command handler
- [ ] decided whether the product uses push-to-talk. If yes, signal 6 is
      unnecessary; if no, `asr` is fitted on THIS hardware and a null
      confidence has been tested to leave behaviour unchanged
