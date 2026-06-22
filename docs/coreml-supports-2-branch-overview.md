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
| 3 | Replace `RangeDim(1, 64)` with a fixed sequence shape (default 32) in the CoreML export | `scripts/export_coreml.py` (this repo) | **Done** — `--seq-len`, default 32 |
| 4 | Pad `input_ids`/`mask`/`token_type_ids` to the fixed length; `mask = 0` on pads (mean-pool already skips them) | `SemanticEmbedder.swift` (STT) | Planned |
| 5 | Set Swift `maxLen` 64 → 32 to match the fixed CoreML shape (Python `max_len` intentionally left at 64 — see note) | `SemanticEmbedder.swift` (STT) | Planned |
| 6 | Set `computeUnits = .cpuAndNeuralEngine`; A/B-measure vs `.cpuOnly` and `.all` | `SemanticEmbedder.swift` (STT) | Planned |
| 7 | Lazy-load the encoder on first semantic rescue and release it after idle (it fires on ~10% of turns) | `IntentClassifierService.swift` (STT) | Planned |
| 8 | Re-run the quality gate to prove accuracy is unchanged | `scripts/compare_coreml_quant.py` (this repo) | Planned |

> **Note — Python `max_len` and the second export script.** The Python tokenisers
> (`nlu/semantic.py`, `train_semantic_head.py`) keep `max_len=64` on purpose: the
> semantic head was trained on ONNX embeddings at 64, and since every real
> utterance is ≤ 33 tokens, a fixed-32 CoreML model produces identical mean-pooled
> embeddings (pads are masked out) — so no retrain is needed and accuracy is
> unchanged. The CoreML-*consuming* tooling — `compare_coreml_quant.py`,
> `train_semantic_head_coreml.py`, and the alternate `export_onnx_to_coreml.py`
> producer — has been updated to pad to the fixed length (they previously fed
> variable-length input and would crash against the fixed-shape model).

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

- **Fixed the actual NaN root cause: FP16 attention-mask overflow.** A freshly
  exported model (native `nn.LayerNorm`, no custom ANE code) was still 100% NaN
  on every input — ruling out the earlier custom-LayerNorm theory. Real cause:
  recent `transformers` builds the additive attention mask as
  `(1 - mask) * torch.finfo(dtype).min` (-3.4e38 for FP32). coremltools casts
  that to `-inf` in FP16, so at every **unmasked** token `0 * -inf = NaN`, which
  softmax then spreads across the whole output (12288/12288 NaN). It surfaced
  after an unpinned `pip install transformers` pulled a version that switched the
  mask sentinel from a finite `-10000.0` to `finfo.min`. Fix in
  `export_coreml.py`: load with `attn_implementation="eager"` and override
  `get_extended_attention_mask` to use a finite, FP16-safe `-1e4`. Masking is
  unchanged (`exp(-1e4) ≈ 0`) and the mean-pool already skips pads, so embeddings
  and accuracy are identical.

- **NaN diagnostic + guards.** Head training aborted with an opaque sklearn
  `Input X contains NaN` because the CoreML encoder emitted non-finite values.
  Added `scripts/diagnose_embedder_nan.py` to isolate the source (FP16 base vs
  palettized int8, raw output before mean-pool, optional full-training-set scan),
  a loud pre-`fit` NaN guard in `train_semantic_head_coreml.py` that names the
  offending phrases and points at the diagnostic, and a finite-output check in
  `export_coreml.py`'s smoke test that now exercises **both** the FP16 and int8
  packages (the old test checked only shape, only FP16, so a NaN-filled model
  passed silently).

- **Repo cleanup.** Removed `.venv-dfcompare/` (17 184 files) and `.idea/` (9)
  that were committed in `294211c2`, deleted the broken `scripts/export-coreML-ANE.py`
  (its custom `ANELayerNorm` computed variance as `(x*x).mean()` in FP16, which
  overflows BERT's hidden-state magnitudes → Inf → NaN), and expanded
  `.gitignore` to cover venvs, IDE dirs, `.mlpackage` bundles, and model binaries.

- **Toolchain consistency for the fixed shape.** Updated the three other scripts
  that talk to the CoreML encoder so they pad to the fixed length (default 32)
  instead of feeding variable-length input: `compare_coreml_quant.py` and
  `train_semantic_head_coreml.py` (consumers — would otherwise crash on the shape
  mismatch once the model is regenerated) and `export_onnx_to_coreml.py` (the
  alternate ONNX→CoreML producer — same `RangeDim`→fixed change + padded sanity
  check). Padding with `mask=0` keeps embeddings identical to the old path and is
  still valid against a dynamic model.

- **`export_coreml.py` — fixed-shape CoreML export (Task #3).** Replaced the
  flexible `RangeDim(1, 64)` sequence axis with a **fixed `(1, 32)`** shape via a
  new `--seq-len` flag (default 32; `--seq-len 0` restores the legacy dynamic
  shape). Length 32 was chosen empirically — across every file in `data/` the max
  WordPiece length is 33 tokens (one garbled training sample), p99 ≤ 20, p95 ≤ 15,
  and all eval phrases ≤ 14. Updated the conversion validation smoke test and
  `docs/coreml-conversion-guide.md` to match. **Code change only** — the
  `.mlpackage` is regenerated on macOS (`python scripts/export_coreml.py`) and
  must ship **together with** the Swift padding change (Task #4), or on-device
  prediction will fail on the shape mismatch.

## See also

- Full optimization plan: [`docs/on-device-memory-optimization-plan.md`](./on-device-memory-optimization-plan.md)
- iOS integration steps: [`docs/coreml-conversion-guide.md`](./coreml-conversion-guide.md)
- Encoder/model rationale: [`docs/semantic-understanding-plan.md`](./semantic-understanding-plan.md)
