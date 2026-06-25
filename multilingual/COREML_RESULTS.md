# CoreML ↔ ONNX Results — Multilingual Intent Classifiers

**Headline:** All 6 models Tier-A **PASS** on this Linux runner. The exported FP16
`mlprogram` packages carry the trained weights exactly to float16 (spec-weights
vs JSON ≤ 6.2e-3, i.e. within the float16 half-ULP bound), and the NumPy device
reference reproduces the ONNX raw logits to **≤ 2.5e-6** over the **full
holdouts (~15,500 utterances)**. An FP16 weight-rounding simulation of the
CoreML matmul predicts **0 argmax flips, 0/30 gate disagreements, and identical
holdout accuracy** vs ONNX on every model; the on-device (real Core ML runtime)
confirmation of those same numbers is **macOS-pending** (this is a Linux
routine). **Total FP16 package size for all six models: 3.93 MB** (FP32 fallback:
7.84 MB).

Build + verify (reproduce):

```bash
python multilingual/export_coreml_multilingual.py --all --fp16 --fp32
python multilingual/test/test_coreml_multilingual.py --full        # Tier-A
python multilingual/test/test_coreml_multilingual.py --runtime     # Tier-B (macOS)
```

Each package: format **mlprogram**, precision **FP16** (FP32 optional fallback),
fixed input `tfidf_vector` shape **(1, V)**, single output **`logits`** (no baked
T=1 `classProbability`). Metadata embeds `temperature`, `conf_threshold`,
`conf_gap_threshold`, `normalize`, `vocab_size`, `class_labels`, and the
`softmax(logits / T)` Swift contract.

## Shipped temperature per model

| Model | T (from weights JSON) |
|---|---|
| en | 0.621397 |
| fr | 0.669939 |
| de | 0.677689 |
| da | 0.815632 |
| multilingual | 0.718213 |
| multilingual_small | 0.755432 |

Device confidence contract: **read `logits`, compute `softmax(logits / T)` in
Swift, gate at 0.70 on that.** A missing `temperature` key ⇒ T = 1.0 (plain
softmax).

---

## Table 1 — Packaging & numeric equivalence (Tier-A, Linux-verified)

Verified on this Linux runner with **no Core ML runtime** (spec inspection +
blob weight extraction + NumPy↔ONNX). "spec-weights vs JSON max abs Δ" is the
largest per-coefficient difference between the weights read back out of the saved
`.mlpackage` and the trained JSON `coef`/`intercept` — bounded by float16's
half-ULP (`2⁻¹¹·max|coef|`). "NumPy-ref vs ONNX max abs Δ" is over each model's
full holdout.

| Model | V | FP16 size | FP32 size | spec-weights vs JSON max abs Δ | NumPy-ref vs ONNX-logits max abs Δ | Tier-A |
|---|---|---|---|---|---|---|
| en | 3843 | 447.0 KB | 889.6 KB | 3.59e-03 | 2.17e-06 | **PASS** |
| fr | 4198 | 488.0 KB | 971.5 KB | 3.86e-03 | 2.21e-06 | **PASS** |
| de | 4629 | 537.6 KB | 1070.8 KB | 3.89e-03 | 2.29e-06 | **PASS** |
| da | 3839 | 446.6 KB | 888.7 KB | 3.82e-03 | 1.84e-06 | **PASS** |
| multilingual | 14195 | 1640.0 KB | 3275.5 KB | 6.18e-03 | 2.25e-06 | **PASS** |
| multilingual_small | 4017 | 467.1 KB | 929.7 KB | 5.84e-03 | 2.46e-06 | **PASS** |
| **total** | — | **3.93 MB** | **7.84 MB** | **≤ 6.18e-03** | **≤ 2.46e-06** | **6/6 PASS** |

The NumPy reference is the same math as `scripts/test_ios_conformance.py::
_ios_predict` (the on-device scorer). Because it matches ONNX to ~2e-6 and the
package weights match the JSON to float16, **CoreML ≡ ONNX transitively**.

---

## Table 2 — ONNX vs CoreML accuracy & gate parity (runtime)

The **CoreML-runtime** cells require `import coremltools` + a real `.mlpackage`
`predict()`, which is **macOS-only** — so on this Linux routine they are
`macOS-pending` (run `--runtime` on a Mac to fill them).

To give real, decision-relevant numbers now, the table below reports an **FP16
weight-rounding simulation** of what the Core ML matmul computes: the trained
`coef`/`intercept` and the L2-normalised input vector are rounded to float16
(fp32 accumulation, as BNNS/ANE do), then compared to ONNX (fp32). This isolates
the dominant float16-storage effect and closely predicts the on-device result.
All numbers are over the **full holdout** per model (n shown), with gate parity
over the **30 conformance utterances**.

| Model | n | ONNX acc | FP16-sim acc | acc Δ | max logit Δ | max top-1 conf Δ | argmax flips | gate disagree /30 | agreement % |
|---|---|---|---|---|---|---|---|---|---|
| en | 1493 | 0.8975 | 0.8975 | 0.0000 | 4.17e-03 | 1.19e-03 | 0 | 0 | 100% |
| fr | 1392 | 0.8491 | 0.8491 | 0.0000 | 4.20e-03 | 7.75e-04 | 0 | 0 | 100% |
| de | 1413 | 0.8316 | 0.8316 | 0.0000 | 4.28e-03 | 1.21e-03 | 0 | 0 | 100% |
| da | 1362 | 0.7599 | 0.7599 | 0.0000 | 4.64e-03 | 9.03e-04 | 0 | 0 | 100% |
| multilingual | 4910 | 0.8422 | 0.8422 | 0.0000 | 5.42e-03 | 1.21e-03 | 0 | 0 | 100% |
| multilingual_small | 4910 | 0.8481 | 0.8481 | 0.0000 | 6.79e-03 | 1.43e-03 | 0 | 0 | 100% |
| **all** | **15480** | — | — | **0.0000** | **≤ 6.79e-03** | **≤ 1.43e-03** | **0** | **0/180** | **100%** |

**Real Core ML runtime (Tier-B) status: `macOS-pending`.** Run on a Mac:

```bash
python multilingual/test/test_coreml_multilingual.py --runtime
```

The Tier-B harness asserts CoreML accuracy within ε of ONNX and **0/30** gate
disagreements per model. Given the FP16 simulation above (which uses the *same*
float16 weights the package stores and which already shows 0 flips / 0 gate
disagreements / identical accuracy), the on-device result is expected to match;
Tier-B records the exact runtime deltas on Apple hardware.

This corroborates the BACKGROUND measurement that FP16 is decision-safe: max
logit Δ small, **0 argmax flips, 0 gate flips** on every model.

---

## ANE eligibility (S5 — best-effort, non-blocking)

**Status: `macOS-pending` for the live compute-plan; expected outcome documented
below.** The compute-plan / Instruments checks need a real Core ML runtime
(`MLComputePlanProxy`), which is macOS-only — on this Linux runner
`coremltools.libcoremlpython` is absent, so the plan cannot be queried here.

**Expected reality (from the measured facts).** Each model is a single dense
matrix–vector product `(59 × V) · (V × 1)`. Real work is tiny: the input is
~0.07–0.26 % dense (≈4–10 non-zero TF-IDF features), so the *useful* arithmetic
is on the order of a few hundred MACs (≈ `59 × non-zeros`). Even re-expressed as
the dense matmul the ANE would actually run (`59 × V` MACs, e.g. ~838 k for
`multilingual`), this is far below the FLOP volume needed to amortise the ANE's
fixed data-staging / dispatch cost. The Core ML runtime will therefore very
likely **schedule these ops on CPU (BNNS)** even with `ComputeUnit.ALL`. That is
expected and fine: at this size CPU is lower-latency and lower-energy than paying
ANE staging overhead, and the whole op costs microseconds either way.

We have already maximised *eligibility* (residency is the runtime's choice, not
ours): **FP16 precision + fixed `(1, V)` input shape + `mlprogram` format**, with
no flexible/RangeDim axes (a dynamic shape would disqualify the ANE outright).

**How to actually check on Apple hardware:**

1. **Compute plan (programmatic, macOS + coremltools):**
   ```python
   import coremltools as ct
   plan = ct.models.compute_plan.MLComputePlan.load_from_path(
       "multilingual/models/en/IntentClassifier_en.mlpackage",
       compute_units=ct.ComputeUnit.ALL,
   )
   prog = plan.model_structure.program
   for op in prog.functions["main"].block.operations:
       usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
       if usage:
           print(op.operator_name,
                 "preferred:", type(usage.preferred_compute_device).__name__,
                 "supported:", [type(d).__name__ for d in usage.supported_compute_devices])
   ```
   Records each op's preferred + supported devices (`MLCPUComputeDevice` /
   `MLGPUComputeDevice` / `MLNeuralEngineComputeDevice`). Expect the `linear`/
   `matmul` op to report **preferred = CPU** for these sizes.

2. **Instruments (observed placement on a device):** Xcode ▸ Open Developer Tool
   ▸ Instruments ▸ **Core ML** template; run the app on a physical iPhone,
   trigger Stage-2 inference, and read the per-layer compute-unit assignment in
   the Core ML track (the "Compute" lane shows CPU/GPU/ANE per op). Xcode's model
   **Performance** report (open the `.mlpackage`, Performance tab, pick a device)
   gives the same prediction offline.

**The one lever that helps most (follow-up, not implemented here):** **vocab
pruning** — shrink `V` the way production did (14195 → 1370). It cuts package
size and the dense-matmul width regardless of where the op runs, and is the
single biggest efficiency win. It changes the trained model, so it is flagged as
a follow-up and intentionally **not** done in this packaging task.
