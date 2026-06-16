# CoreML Conversion Guide

**Branch:** `feature/Adv2/AddSemanticUnderstanding-4-adding-coreML-supports`  
**Purpose:** Convert the 3-stage NLU pipeline to CoreML for on-device iOS inference.

---

## Prerequisites

- macOS 13+ (Ventura or later)
- Python 3.9+
- All model artifacts present in `models/`

```bash
pip install coremltools onnxruntime numpy
```

Verify:
```bash
python -c "import coremltools; print(coremltools.__version__)"  # expect 7.x+
python -c "import onnxruntime; print(onnxruntime.__version__)"
```

---

## Step 1 — Ensure source models are present

```bash
ls models/
# Required before running export:
#   intent_classifier_weights.json   (Stage 2 source)
#   semantic_head.json               (Stage 3a source)
#   minilm-vocab.txt                 (Swift tokeniser)
#   minilm-l6-v2.onnx               (Stage 3b source)
#   intent_classifier_weights.json   (Swift Stage 2 fallback)
```

If `minilm-l6-v2.onnx` is missing:
```bash
python scripts/download_minilm.py
```

If `intent_classifier_weights.json` is missing:
```bash
python scripts/export_ios_weights.py
```

---

## Step 2 — Inspect ONNX models (no conversion yet)

```bash
python scripts/export_coreml.py --inspect-only
```

This prints the MiniLM ONNX input/output names and runs a dummy inference to
show the **actual output shape**. Read the output carefully:

```
MiniLM ONNX introspection
  Declared inputs:
    input_ids             shape=[1, 'sequence']  type=int64
    attention_mask        shape=[1, 'sequence']  type=int64
    token_type_ids        shape=[1, 'sequence']  type=int64
  Declared outputs:
    last_hidden_state     shape=[1, 'sequence', 384]  type=float
  Actual output shapes (seq_len=5 dummy run):
    last_hidden_state     shape=(1, 5, 384)  dtype=float32
  ✅ Output is token-level [batch, seq, dim].
     SemanticEmbedder.swift mean-pool logic is CORRECT.
```

**If it says "Output is already POOLED"**, you need to update
`SemanticEmbedder.swift` — remove the mean-pool loop in `meanPoolAndNormalize()`
and just L2-normalise the single [384] vector directly.

**The primary output name** is what `SemanticEmbedder.swift` reads:
```swift
output.featureValue(for: "last_hidden_state")?.multiArrayValue
```
If the ONNX output has a different name, update that string in Swift.

---

## Step 3 — Run Stage 2 + 3a conversion first (fast, no MiniLM)

```bash
python scripts/export_coreml.py --skip-minilm
```

Expected output:
```
Stage 2: IntentClassifier
  Classes  : 51
  Features : <n_features from weights JSON>
  Saved models/IntentClassifier.mlpackage
  input : tfidf_vector
  output: classProbability
  output: label

Stage 3a: SemanticHead
  Classes  : 51
  Emb dim  : 384
  Saved models/SemanticHead.mlpackage
  input : embedding
  output: classProbability
  output: label

Validation
  IntentClassifier: loaded
  SemanticHead: loaded
  IntentClassifier test inference (dummy zero vector):
    Top prediction : '<some label>'  (value is meaningless)
    Inference OK
```

If this step fails, see Troubleshooting below.

---

## Step 4 — Run MiniLM conversion (slow, ~60–120 s)

```bash
python scripts/export_coreml.py
```

This adds Stage 3b on top of what Step 3 produced. The MiniLM model is
FLOAT16 precision and supports variable sequence lengths 1..64 (matching
the `max_len=64` limit in `SemanticEmbedder.swift`).

Expected size of `MiniLMEmbedder.mlpackage`: approximately 12–15 MB on disk
(the ONNX INT8 model is 22.8 MB; FP16 CoreML is typically smaller).

---

## Step 4b — (Optional) INT8 quantization to halve MiniLM size

The FP16 `MiniLMEmbedder.mlpackage` is ~45 MB. If that is too large for the app
bundle, you can quantize the weights to INT8 (~22 MB). This is a **size**
optimization, **not** a speed one — on the Apple Neural Engine, FP16 is the
native compute format, so INT8 latency is roughly the same.

INT8 perturbs the embeddings slightly, and the semantic head was trained on
FP16/FP32 embeddings — so borderline rescues near the 0.55 threshold can flip.
**Never ship INT8 without measuring.**

```bash
# 1. Produce both FP16 and INT8 variants:
python scripts/export_coreml.py --quantize
#    → models/MiniLMEmbedder.mlpackage       (FP16, ~45 MB, kept)
#    → models/MiniLMEmbedder_int8.mlpackage  (INT8, ~22 MB)

# 2. Measure the accuracy delta on the real holdout + OOS data:
python scripts/compare_coreml_quant.py
```

`compare_coreml_quant.py` runs both models through `semantic_holdout_100.csv`
(in-scope) and `semantic_oos.csv` (out-of-scope), and reports (numbers below
are illustrative — your run prints the real measured values):

```
  Metric                              FP16      INT8           Δ
  in-scope accuracy (%)               88.0      87.0        -1.0
  OOS rejection (%)                   94.3      94.3        +0.0

  embedding cosine FP16↔INT8: mean=0.9991  min=0.9920
  decision flips: 2 / 257

  VERDICT: INT8 acceptable (accuracy Δ -1.0%, OOS Δ +0.0%).
```

**Ship bar:** accuracy delta < 1% AND OOS rejection not worse. If the verdict is
"keep FP16", the size win isn't worth the regression — bundle the FP16 model.

If you ship INT8, rename `MiniLMEmbedder_int8.mlpackage` → `MiniLMEmbedder.mlpackage`
when copying to the STT repo (the Swift code looks for `MiniLMEmbedder`); the
FP16↔INT8 choice is invisible to the iOS app.

---

## Step 5 — Copy artifacts to the STT iOS repo

Files to copy into `STT/STT/STT/Resources/`:

| File | Required | Purpose |
|---|---|---|
| `IntentClassifier.mlpackage` | **Yes** | Stage 2 CoreML (primary) |
| `SemanticHead.mlpackage` | **Yes** | Stage 3a CoreML |
| `MiniLMEmbedder.mlpackage` | **Yes** | Stage 3b CoreML |
| `intent_classifier_weights.json` | Yes (already there) | Stage 2 Swift fallback |
| `semantic_head.json` | Yes (already there) | Stage 3a Swift fallback |
| `minilm-vocab.txt` | **Yes** | Stage 3b Swift BERT tokeniser |

```bash
cp models/IntentClassifier.mlpackage   ../STT/STT/STT/Resources/
cp models/SemanticHead.mlpackage       ../STT/STT/STT/Resources/
cp models/MiniLMEmbedder.mlpackage     ../STT/STT/STT/Resources/
cp models/minilm-vocab.txt             ../STT/STT/STT/Resources/
cp models/semantic_head.json           ../STT/STT/STT/Resources/
cp models/intent_classifier_weights.json ../STT/STT/STT/Resources/
```

---

## Step 6 — Xcode integration

1. In Xcode, the `Resources` folder is a synchronized root group — any files
   copied into it appear in the project automatically.
2. **Important:** Xcode compiles `.mlpackage` files to `.mlmodelc` during the
   build phase. At **runtime** on a device or simulator, the bundle contains
   `.mlmodelc`, not `.mlpackage`.
3. In `IntentClassifierService.swift` and `SemanticEmbedder.swift`, both
   extensions are tried:
   ```swift
   Bundle.main.url(forResource: "MiniLMEmbedder", withExtension: "mlmodelc")
       ?? Bundle.main.url(forResource: "MiniLMEmbedder", withExtension: "mlpackage")
   ```
   This covers both the compiled (device/simulator) and raw (direct bundle)
   cases.

---

## Step 7 — Update Swift files for new CoreML contracts

Two places in the STT repo need updating after this conversion:

### A. `IntentClassifierService.swift` — Stage 2 input name

The CoreML model input name is `"tfidf_vector"` (not `"text"`).
The `coreMLClassify()` function should pass the TF-IDF float vector:

```swift
private func coreMLClassify(_ text: String, model: MLModel) -> (String, Double)? {
    // Build TF-IDF vector in Swift (existing logic)
    let vec = tfidfVector(for: text)
    guard let array = try? MLMultiArray(shape: [vec.count as NSNumber], dataType: .float32)
    else { return nil }
    for (i, v) in vec.enumerated() { array[i] = NSNumber(value: Float(v)) }

    guard
        let input  = try? MLDictionaryFeatureProvider(
            dictionary: ["tfidf_vector": MLFeatureValue(multiArray: array)]
        ),
        let output = try? model.prediction(from: input),
        let probs  = output.featureValue(for: "classProbability")?.dictionaryValue
                     as? [String: Double]
    else { return nil }

    guard let best = probs.max(by: { $0.value < $1.value }) else { return nil }
    return (best.key, best.value)
}
```

### B. `SemanticEmbedder.swift` — bundle extension + output name

Use `mlmodelc` extension first (compiled Xcode build), fall back to `mlpackage`:
```swift
let modelURL = Bundle.main.url(forResource: "MiniLMEmbedder", withExtension: "mlmodelc")
            ?? Bundle.main.url(forResource: "MiniLMEmbedder", withExtension: "mlpackage")
```

Verify the output feature name matches what introspection printed in Step 2.

### C. `SemanticClassifier.swift` — use CoreML SemanticHead instead of JSON

Once `SemanticHead.mlpackage` is bundled, update to use it:
```swift
// Load SemanticHead.mlpackage (CoreML)
let modelURL = Bundle.main.url(forResource: "SemanticHead", withExtension: "mlmodelc")
            ?? Bundle.main.url(forResource: "SemanticHead", withExtension: "mlpackage")
// Then call model.prediction() with {"embedding": MLMultiArray(...)}
// Output: classProbability dict — same contract as IntentClassifier
```

### D. Fix semantic threshold bypass (NLUEngine.swift)

When `semanticRescue == true`, the result already passed Stage 3's 0.55 threshold.
It must NOT be re-checked against Stage 2's 0.70 threshold:

```swift
private func handleNewIntent(_ text: String) async -> NLUResponse {
    session.decrementContexts()
    let result = await classifier.classifyAsync(text)

    // Semantic-rescued results bypass the TF-IDF threshold — they already
    // passed Stage 3's 0.55 gate inside classifyAsync().
    let isUnknown = result.label == "Default Fallback Intent"
    let belowThreshold = !result.semanticRescue && result.confidence < schema.confidenceThreshold
    if isUnknown || belowThreshold {
        return .fallback(url: classifier.genaiURL(for: text), confidence: result.confidence)
    }
    // ... rest unchanged
}
```

---

## Troubleshooting

### `NeuralNetworkBuilder` deprecation warning
Expected in coremltools 7+. The builder still works for simple linear models.
The warning can be suppressed but the output is valid.

### `ct.convert()` fails on MiniLM INT8 quantization ops
The INT8-quantized ONNX model uses `QLinearMatMul` and `QLinearConv` ops that
some coremltools versions cannot convert.

Solution: download the FP32 version:
```bash
python scripts/download_minilm.py --fp32
# Then re-run export_coreml.py
```
FP32 CoreML will be ~45 MB vs ~15 MB for FP16. Acceptable for on-device use.

### `RangeDim` causes shape error during conversion
Some older coremltools versions don't handle `RangeDim` with ONNX dynamic axes.
Try pinning a fixed sequence length:
```python
# Replace RangeDim with a fixed shape, e.g. 32
ct.TensorType(name="input_ids", shape=(1, 32), dtype=np.int32)
```
Then pad all sequences to length 32 in `SemanticEmbedder.swift`.

### `classProbability` output missing in IntentClassifier test inference
The `classifier` mode in `NeuralNetworkBuilder` should add this automatically.
If missing, your coremltools version may not support it — try:
```bash
pip install --upgrade coremltools
```

### Models not loading in Xcode / nil from Bundle.main.url
Xcode compiles `.mlpackage` → `.mlmodelc` at build time. Both extensions are
tried in the Swift code. If both return nil:
- Confirm the `.mlpackage` is in the synchronized folder (should auto-include)
- Clean build folder (`⇧⌘K`) and rebuild
- Check the Xcode build log for CoreML compilation errors
