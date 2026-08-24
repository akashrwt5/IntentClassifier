# Android Offline Integration Contract

Phase 27 of the plan. This describes what the app receives and what it must do
with it. No network call exists anywhere in this path.

## Runtime chain

```text
Microphone
   |
Offline STT
   |
Text
   |
Normalizer          <- must match scripts/common.py::normalize
   |
WordPiece tokenizer <- driven by onnx/tokenizer/vocab.txt
   |
ONNX model          <- encoder + pooling + classifier + temperature + softmax
   |
Safety gate         <- thresholds from onnx/runtime_config.json
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
input   input_ids       int64 [batch, seq]      padded to max_len (64)
input   attention_mask  int64 [batch, seq]
output  probs           float32 [batch, 57]     already calibrated
```

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

fun decide(probs: FloatArray, cfg: RuntimeConfig): IntentResult {
    val order = probs.indices.sortedByDescending { probs[it] }
    val i1 = order[0]; val i2 = order[1]
    val c1 = probs[i1]; val c2 = probs[i2]
    val margin = c1 - c2

    val (accepted, reason) = when {
        cfg.labels[i1] == cfg.gate.rejectLabel ->
            false to "classified as unsupported"
        c1 < cfg.gate.confThreshold ->
            false to "below calibrated confidence threshold"
        margin < cfg.gate.marginThreshold ->
            false to "top-1/top-2 margin too small"
        else -> true to "above calibrated threshold"
    }
    return IntentResult(cfg.labels[i1], c1, cfg.labels[i2], c2,
                        margin, accepted, reason)
}
```

Three independent reasons to refuse, and they are not redundant:

- **reject label** catches requests the model recognises as unsupported
- **confidence** catches requests it half-recognises
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
- [ ] rejected results provably cannot reach the command handler
