# Branch Overview — `feature/Adv2/AddSemanticUnderstanding-4-adding-coreML-supports-2`

## Cut from

`feature/Adv2/AddSemanticUnderstanding-4-adding-coreML-supports` (at `e8fec52`)

## Purpose

Execute the **on-device memory optimization** work: reduce the iOS RAM footprint
of the Stage 3b MiniLM-L6-v2 semantic encoder (`MiniLMEmbedder`), reported at
**>100 MB**, **without losing classification quality**.

This branch deliberately starts with the **zero-accuracy-loss runtime fixes**
(Tier 0 + Tier 1 of the plan) *before* any model-architecture change. Full
strategy and reasoning: [`docs/on-device-memory-optimization-plan.md`](./on-device-memory-optimization-plan.md).

## Root cause (working hypothesis)

`MiniLMEmbedder` is **~45 MB FP16 on disk but uses >100 MB RAM**. That ~55 MB gap
is *runtime overhead*, not weights. Most likely cause, confirmed by reading the
iOS code (`SemanticEmbedder.swift`):

- the encoder is exported with a **flexible `RangeDim(1, 64)`** sequence shape
  (`export_coreml.py:342`), and Swift feeds a **variable `[1, n]`** input that is
  never padded;
- it is loaded with **`config.computeUnits = .all`**.

That combination keeps the model **off the Apple Neural Engine**, runs it on
GPU/CPU, and reserves memory for the worst-case (64-token) length. So weight-level
work (quantization, distillation) may not move the RAM number until this is fixed.

## Work in this branch

> Some changes land in **this repo** (Python export/training) and some in the
> separate **STT iOS repo** (Swift). The "Where" column says which.

### Tier 0 — Diagnose (no code change, do first)

| # | Task | Where | Status |
|---|------|-------|--------|
| 1 | Measure peak RAM at 4 checkpoints (baseline / models loaded / encoder loaded / during inference) | STT — Xcode Instruments | Planned |
| 2 | Confirm which compute unit runs the encoder (Xcode → `.mlpackage` → Performance tab) | STT — Xcode | Planned |

### Tier 1 — Runtime / config fixes (zero accuracy loss)

| # | Task | Where | Status |
|---|------|-------|--------|
| 3 | Replace `RangeDim(1, 64)` with a fixed (or enumerated) sequence shape in the CoreML export | `scripts/export_coreml.py` (this repo) | Planned |
| 4 | Pad `input_ids`/`mask`/`token_type_ids` to the fixed length; `mask = 0` on pads (mean-pool already skips them) | `SemanticEmbedder.swift` (STT) | Planned |
| 5 | Reduce `max_len` 64 → 32, keeping Python and Swift tokenisers in sync | `nlu/semantic.py`, `train_semantic_head.py` (this repo) + `SemanticEmbedder.swift` (STT) | Planned |
| 6 | Set `computeUnits = .cpuAndNeuralEngine`; A/B-measure vs `.cpuOnly` and `.all` | `SemanticEmbedder.swift` (STT) | Planned |
| 7 | Lazy-load the encoder on first semantic rescue and release it after idle (it fires on ~10% of turns) | `IntentClassifierService.swift` (STT) | Planned |
| 8 | Re-run the quality gate to prove accuracy is unchanged | `scripts/compare_coreml_quant.py` (this repo) | Planned |

## What is NOT changed here

- **No model-architecture change** — weights stay bit-for-bit identical, so
  classification accuracy is unchanged. (Distillation / static embeddings are
  Tier 3 / Tier 4 of the plan → a **separate future branch**, not this one.)
- The cascade algorithm (keyword → TF-IDF → semantic rescue) is untouched.
- The semantic head is **not** retrained — there is no encoder swap on this
  branch, so the embedding space is the same.
- Ship bar is unchanged: **in-scope accuracy Δ ≥ −1% AND OOS rejection not
  worse** (`compare_coreml_quant.py`).

## Done log

_Update as work lands — newest first._

- _(nothing yet — branch just created; scope documented)_

## See also

- Full optimization plan: [`docs/on-device-memory-optimization-plan.md`](./on-device-memory-optimization-plan.md)
- iOS integration steps: [`docs/coreml-conversion-guide.md`](./coreml-conversion-guide.md)
- Encoder/model rationale: [`docs/semantic-understanding-plan.md`](./semantic-understanding-plan.md)
