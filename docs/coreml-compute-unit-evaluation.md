# CoreML compute-unit evaluation: why the intent head runs on CPU

**Status:** measured evidence for ADR-017. **Date:** 2026-07-31.
**Question answered:** should the on-device intent classifier run on the Apple
Neural Engine (ANE) or the CPU, and should we ship `mlprogram` or `neuralnetwork`?

**Answer:** CPU, `neuralnetwork`, pre-compiled `.mlmodelc`, `computeUnits = .cpuOnly`.
ANE is slower, uses more app RAM, and is non-deterministic for this model.

---

## 0. TL;DR — the four numbers that decide it

| | ANE (`mlprogram` FP16, `.all`) | CPU (`neuralnetwork`, `.cpuOnly`) | ratio |
|---|---|---|---|
| Latency for a real voice command (after idle) | **21–34 ms** | **0.4–1.1 ms** | ~30× worse |
| Model load / OTA hot-swap | **98 ms** | **15 ms** | 6.5× worse |
| App memory (`phys_footprint`) | **10.22 MB** | **3.69 MB** | 2.8× worse |
| Logits reproducible across runs | **no** | yes | — |

The ANE loses on every axis that matters for a sporadically-invoked voice assistant.

---

## 1. Test rig and provenance

| | |
|---|---|
| Host | Apple M2 Max, 16-core ANE, macOS 26.5 (build `25F80`) |
| Toolchain | Xcode 26.5 (17F42), coremltools 9.0, Swift 6.3.2 |
| Weights | `models/intent/en/intent_classifier_weights.json` (pruned, 1317 feat) and `intent_classifier_weights_full.json` (full, 4718 feat), 57 classes |
| Eval set | `language_packs/en/holdout_honest.csv`, n = 1470 |
| Gate | `conf_threshold = 0.70`; T = 0.791486 (pruned) / 0.640042 (full) |

**Architecture A** = today's production path: `coremltools.models.neural_network.NeuralNetworkBuilder`,
format `neuralnetwork`, FP32, iOS 11+, no torch.
**Architecture B** = the multilingual-branch prototype: `mlprogram`, FP16, iOS 15+.
B was rebuilt here with the **MIL builder** (`mb.program` + `mb.linear`), *not*
`torch.jit.trace` — see §8, the two are equivalent and MIL needs no torch.

**Not measured (no physical iPhone available):** mJ/inference, P-core vs E-core
attribution, thermal behaviour over sustained load. See §10 for how much this
changes the conclusion (answer: it does not).

Harness: `bench.swift` (cold/warm latency), `bench2.swift` (load sequencing +
CoreML API behaviour), `bench3.swift` (residency after idle), `bench4.swift`
(memory footprint), `build_variants.py`, `compute_plan.py`, `fp16_eval.py`.

---

## 2. The ANE-eligibility question — the common misconception

Measured with `MLComputePlan`, `computeUnits = .all`:

| Model | Preferred device | Supported devices |
|---|---|---|
| A `neuralnetwork` pruned (1317) | **CPU** | CPU, GPU, ANE×16 |
| A `neuralnetwork` full (4718) | **CPU** | CPU, GPU, ANE×16 |
| A `neuralnetwork` FP32-IO variant | **CPU** | CPU, GPU, ANE×16 |
| B `mlprogram` **FP16** pruned (1317) | **CPU** | CPU, GPU, ANE×16 |
| B `mlprogram` **FP16** full (4718) | **ANE** | CPU, GPU, ANE×16 |
| B `mlprogram` **FP32** pruned | CPU | CPU, GPU — **no ANE** |
| B `mlprogram` **FP32** full | GPU | CPU, GPU — **no ANE** |

Three corrections to the folklore:

1. **Architecture A is already ANE-eligible.** Seeing "Neural Engine: 0" in Xcode
   is CoreML's *cost decision*, not a capability limit of the `neuralnetwork`
   format. The planner picks CPU because CPU is cheaper for this graph.
2. **ANE eligibility comes from FP16, not from `mlprogram`.** An FP32 ML Program
   gets no ANE at all.
3. **Even the FP16 ML Program falls back to CPU at 1317 features.** ANE only wins
   the planner's cost model at 4718 — and then loses on the clock (§3).

Per-op cost weights for B FP16 full: `cast` 0.005, `linear` 0.995, `cast` 0.000.
The model is one matmul; there is nothing to parallelise across 16 ANE cores.

---

## 3. Latency

### 3.1 Warm throughput (median of 300 back-to-back predictions, held instance)

| Variant | ComputeUnits | median (ms) | p95 (ms) |
|---|---|---|---|
| A nn pruned | `.all` (→CPU) | **0.0342** | 0.0469 |
| A nn pruned | `.cpuOnly` | 0.0359 | 0.0410 |
| A nn full | `.all` (→CPU) | **0.0400** | 0.0520 |
| A nn full | `.cpuOnly` | 0.0420 | 0.0520 |
| B fp16 pruned | `.all` (→CPU) | 0.0623 | 0.1011 |
| B fp16 pruned | `.cpuOnly` | 0.0769 | 0.1340 |
| B fp16 full | `.cpuOnly` | 0.0720 | 0.1169 |
| B fp16 full | `.all` (→**ANE**) | **0.2251** | 0.3018 |
| B fp16 full | `.cpuAndNeuralEngine` | 0.2222 | 0.2961 |

ANE is **3.1× slower** than the identical ML Program on CPU (0.225 vs 0.072 ms)
and **5.6× slower** than the legacy neural network (0.225 vs 0.040 ms). This
reproduces the original field observation of 0.04 ms vs 0.14 ms.

### 3.2 Latency after idle — the number that describes real usage

`bench3.swift`. The `MLModel` instance is loaded **once and held for the entire
run** — never released — then predictions are separated by an idle gap.

| Idle before the call | B FP16 `.all` (ANE) | B FP16 `.cpuOnly` | A nn `.cpuOnly` |
|---|---|---|---|
| back-to-back (warm) | 0.19 – 0.36 | 0.068 | 0.042 |
| 0.5 s | 0.64 / 1.24 | — | — |
| 1 s | 0.90 | 0.35 | 0.47 |
| 2 s | 1.05 / 1.08 | — | — |
| 5 s | 2.02 / **27.84** / **33.57** | 0.73 | 0.58 |
| 10 s | **21.54** / **27.14** | — | — |
| 15 s | **25.40** | 1.12 | 0.60 |
| 30 s | **22.08** | 0.40 | 1.05 |

(all values ms; multiple values = separate repeat runs)

**Past roughly 5 seconds of idle, the first ANE prediction costs 21–34 ms** even
though the model object was never released. This is the ANE being power-gated and
its execution context torn down; the next call must re-wake the block and re-DMA
the weights. The CPU paths, in the same harness, same process, same idle gaps,
stay at 0.4–1.1 ms — which isolates the effect to the ANE rather than to OS
scheduling.

**This is the decisive result for a voice assistant.** Commands arrive sporadically
with seconds-to-minutes of silence between them, so *every real invocation lands in
the post-idle regime*. The 0.14 ms ANE median only exists inside a tight benchmark
loop that never occurs in production. Effective per-command cost: **ANE ≈ 21–34 ms,
CPU ≈ 0.4–1.1 ms.**

It also inverts the battery argument. Power-gating exists precisely *because*
keeping the ANE live costs energy; waking a gated block on every command costs more
than 40 µs of E-core matmul, not less.

---

## 4. Load cost and the OTA hot-swap budget

`bench2.swift` — four loads in one process, each from a **different freshly-copied
path**, simulating an OTA bundle unpacked into `Documents/packs/<version>/`.

| Load # | A `.mlmodelc` (load / first predict) | B FP16 `.mlmodelc` (load / first predict) |
|---|---|---|
| 1 | 73.26 / 3.726 ms | 144.74 / 5.574 ms |
| 2 | 18.47 / 2.180 ms | 101.30 / 0.761 ms |
| 3 | 14.57 / 0.224 ms | 98.76 / 0.668 ms |
| 4 | 15.29 / 0.211 ms | 97.85 / 0.655 ms |

Two conclusions:

1. **Loads 2–4 are flat, at different paths each time ⇒ a pre-compiled `.mlmodelc`
   triggers no per-model on-device compilation.** The load-1 premium is
   process-level CoreML/ANE framework initialisation, not model compilation. The
   feared "ANE recompiles on every OTA" penalty does not exist for `.mlmodelc`.
2. **Steady-state hot-swap: A ≈ 15 ms, B ≈ 98 ms** — 6.5× worse, on every swap,
   permanently. It is not a one-time cache fill.

Against the **< 500 ms hot-swap NFR** (HLD §6.4), full cold totals from `bench.swift`:

| Variant | ComputeUnits | compile | load | first predict | cold total |
|---|---|---|---|---|---|
| A nn full `.mlmodelc` | `.cpuOnly` | 0 | 15.6 | 1.63 | **17.2 ms** |
| A nn full `.mlmodelc` | `.all` | 0 | 93.7 | 2.85 | 96.5 ms |
| A nn full `.mlpackage` | `.all` | 32.8 | 61.7 | 0.29 | 94.8 ms |
| B fp16 full `.mlmodelc` | `.all` | 0 | 137.0 | 5.25 | 142.2 ms |
| B fp16 full `.mlpackage` | `.all` | 58.5 | 107.7 | 4.81 | **171.0 ms** |

A on `.cpuOnly` uses **3.4 %** of the 500 ms budget. B's worst case uses 34 % on an
M2 Max; on an A15/A17 that headroom shrinks considerably.

**Pinning `.cpuOnly` is a free win:** it cuts Arch A's cold load from 93.7 ms to
15.6 ms, because `.all` pays ANE/GPU service initialisation and then executes on
CPU anyway.

---

## 5. Memory — ANE does *not* move cost off the app

`bench4.swift`, reporting `task_vm_info.phys_footprint` — exactly the counter iOS
jetsam uses to decide which app to kill.

| Config | baseline | + model load | + 1st predict | + 500 predicts | **total** |
|---|---|---|---|---|---|
| A nn full, `.cpuOnly` | 1.56 | +1.61 | +0.30 | +0.22 | **3.69 MB** |
| A nn full, `.all` | 1.56 | +3.58 | +0.11 | +0.12 | 5.38 MB |
| A nn full, `.cpuAndNeuralEngine` | 1.53 | +3.53 | +0.17 | +0.11 | 5.34 MB |
| B fp16 full, `.cpuOnly` | 1.52 | +5.78 | +0.19 | +0.30 | 7.78 MB |
| B fp16 full, `.cpuAndNeuralEngine` | 1.52 | +6.22 | +0.28 | +0.27 | 8.28 MB |
| B fp16 full, `.all` | 1.52 | +8.16 | +0.33 | +0.22 | **10.22 MB** |

Apple Silicon is a unified-memory architecture: the weights live in the same DRAM
regardless of compute unit, and CoreML's ANE buffers are IOSurface allocations
**charged to the requesting process**. Enabling the ANE therefore *increases* the
app's footprint — the ANE client stack is additionally mapped into the process.

Worst config vs best: **10.22 MB vs 3.69 MB — going to ANE costs 6.5 MB more app
RAM, not less.** Nothing is "handled separately by the OS."

---

## 6. Pre-compiled `.mlmodelc` over OTA

### 6.1 Is it portable across devices? — Yes

`xcrun coremlcompiler compile` output for Architecture A contains:

```
IntentClassifier.mlmodelc/
├── model.espresso.net       # JSON graph (inner_product + softmax)
├── model.espresso.shape
├── model.espresso.weights
├── coremldata.bin
├── metadata.json            # availability: iOS 11.0, specificationVersion 1
├── model/coremldata.bin
├── analytics/coremldata.bin
└── neural_network_optionals/coremldata.bin
```

The compute-target string is `generic`. There is **no `.hwx` ANE microcode** in the
bundle — that is generated on-device by ANECompilerService. So the artifact is
device-agnostic Espresso IR: a macOS CI runner can compile it and any iPhone can
load it. Compilation on the runner takes 0.09–0.15 s.

**Caveat:** format compatibility follows the Xcode that compiled it. Pin the CI
Xcode version and treat an Xcode bump as a bundle-format change requiring
re-validation on the oldest supported iOS.

### 6.2 CoreML API behaviour — measured, not assumed

| Call | Result |
|---|---|
| `MLModel(contentsOf: <.mlmodelc>)` | works, no compile step |
| `MLModel(contentsOf: <.mlpackage>)` | **FAILS** — *"Unable to load model … Compile the model with Xcode or `MLModel.compileModel(at:)`"* |
| `MLModel.compileModel(at: <.mlmodelc>)` | **FAILS** — *"A valid manifest does not exist"* |
| same `.mlmodelc` under `.cpuOnly` / `.cpuAndNeuralEngine` / `.all` | all load, **no recompilation** |

`MLModel(contentsOf:)` does not "skip" compilation — it *requires* an already-compiled
model. The Swift loader must therefore dispatch on file extension (as
`VoiceIntentKit` already does); it must never be "simplified" to always call
`compileModel`.

Changing `computeUnits` does **not** invalidate the compiled artifact. It selects a
backend at load time; the load-time difference (15 ms vs 94 ms) is service
initialisation, not recompilation.

### 6.3 The ANE compile cache — why you can never rely on it

```
~/Library/Caches/com.apple.e5rt.e5bundlecache/<OS_BUILD>/<SHA256_of_model>/
                                              ^25F80    ^content hash
```

- Keyed by **content hash** → every new OTA model version is a cache miss *by
  construction*; path-independence does not help.
- Keyed by **OS build number** → an iOS update invalidates every entry.
- Lives under `Caches` → purgeable under storage pressure, excluded from backup.

**Never budget hot-swap latency assuming a warm ANE cache.**

### 6.4 Size impact

`du -sk`:

| Model | `.mlpackage` | `.mlmodelc` | Δ | compile time |
|---|---|---|---|---|
| A nn pruned | 300 KB | 324 KB | +24 KB (+8 %) | 0.09 s |
| A nn full | 1060 KB | 1080 KB | +20 KB (+2 %) | 0.11 s |
| B fp16 pruned | 156 KB | 164 KB | +8 KB | 0.12 s |
| B fp16 full | 536 KB | 544 KB | +8 KB | 0.15 s |
| B fp32 pruned | 304 KB | 312 KB | +8 KB | 0.12 s |
| B fp32 full | 1060 KB | 1068 KB | +8 KB | 0.12 s |

Compiled output is only 2–8 % larger than source. Shipping **both** formats roughly
doubles the CoreML slice — about 1.4 MB of the current 3.2 MB `pack-en-v1.0.0.nlu`.
Since `MLModel(contentsOf:)` cannot consume a `.mlpackage` anyway, and the genuine
last-resort fallback is the pure-Swift `intent_classifier_weights.json` already in
the bundle, the `.mlpackage` earns little as shipped ballast.

---

## 7. FP16 accuracy — safe, but the stated KPI is wrong

`fp16_eval.py`, n = 1470, every variant executed through the **real CoreML runtime**;
reference is float64 NumPy over the same device-equivalent TF-IDF vectors.

### Pruned head (1317 features, T = 0.791486)

| Variant | acc | argmax flips | gate crossings @0.70 | max abs Δlogit | ECE | NLL |
|---|---|---|---|---|---|---|
| float64 reference | 88.57 % | — | — | — | 0.0202 | 0.4477 |
| A nn FP32 `.cpuOnly` | 88.57 % | 0 | 0 | 2.225e-06 | 0.0202 | 0.4477 |
| A nn FP32 `.all` | 88.57 % | 0 | 0 | 2.225e-06 | 0.0202 | 0.4477 |
| B FP32 `.cpuOnly` | 88.57 % | 0 | 0 | 1.405e-06 | 0.0202 | 0.4477 |
| B FP16 `.cpuOnly` | 88.57 % | 0 | 0 | 1.327e-02 | 0.0201 | 0.4478 |
| B FP16 `.all` | 88.57 % | 0 | 0 | 1.327e-02 | 0.0201 | 0.4478 |

### Full head (4718 features, T = 0.640042)

| Variant | acc | argmax flips | gate crossings @0.70 | max abs Δlogit | max abs Δconf | ECE | NLL |
|---|---|---|---|---|---|---|---|
| float64 reference | 90.20 % | — | — | — | — | 0.0192 | 0.3714 |
| A nn FP32 `.cpuOnly` | 90.20 % | 0 | 0 | 1.690e-06 | 4.49e-07 | 0.0192 | 0.3714 |
| B FP32 `.cpuOnly` | 90.20 % | 0 | 0 | 1.222e-06 | 3.25e-07 | 0.0192 | 0.3714 |
| B FP16 `.cpuOnly` | 90.20 % | 0 | 0 | 1.057e-02 | 2.27e-03 | 0.0205 | 0.3714 |
| B FP16 `.all` (ANE) | 90.20 % | 0 | 0 | 6.376e-03 | 2.33e-03 | 0.0198 | 0.3714 |
| B FP16 `.cpuAndNeuralEngine` | 90.20 % | 0 | 0 | 6.376e-03 | 2.33e-03 | 0.0198 | 0.3714 |

Three findings:

1. **FP16 is numerically safe.** Zero argmax flips and zero 0.70-gate crossings at
   both vocabulary sizes. FP16 is *not* the reason to avoid Architecture B.
2. **The KPI "accuracy must remain > 95 % on holdout" is already failing** — the
   model sits at 88.57 % (pruned) / 90.20 % (full) on the honest holdout. The FP16
   question passes only because Δ = 0. The gate should be restated as a *delta*
   gate against the FP32 reference (Δacc ≤ 0.1 pp, 0 argmax flips, 0 gate
   crossings), which is what it was evidently meant to express.
3. **ANE and CPU return different logits from the same FP16 model** (max abs Δlogit
   6.376e-03 vs 1.057e-02; ECE 0.0198 vs 0.0205). Under `.computeUnits = .all` the
   backend is CoreML's choice and can vary with device, thermal state and ANE
   contention from other apps. **The same shipped model is therefore not
   bit-reproducible run-to-run.** With a 0.70 confidence gate and canary cohorts
   whose metrics are compared against each other (HLD §5.6), that is
   non-reproducibility we cannot debug in support triage. Architecture A on
   `.cpuOnly` is deterministic to 1.7e-06 of the float64 reference.

---

## 8. MIL builder vs PyTorch tracing — torch is not required

Architecture B was rebuilt here with the coremltools MIL builder and **no torch
installed**:

```python
@mb.program(input_specs=[mb.TensorSpec(shape=(1, n_features), dtype=types.fp32)])
def prog(tfidf_vector):
    return mb.linear(x=tfidf_vector, weight=coef, bias=intercept, name="logits")

mlmodel = ct.convert(prog, convert_to="mlprogram",
                     minimum_deployment_target=ct.target.iOS15,
                     compute_precision=ct.precision.FLOAT16,
                     inputs=[ct.TensorType(name="tfidf_vector", shape=(1, F), dtype=np.float32)],
                     outputs=[ct.TensorType(name="logits", dtype=np.float32)])
```

The resulting compute plan is **identical** to the torch-traced route:
`cast` / `linear` / `cast`, all `supported = [CPU, GPU, ANE×16]`, `preferred = ANE`
for the full-vocab FP16 build. ANE eligibility derives from the MIL op set + FP16
precision + deployment target — not from the frontend. `torch.jit.trace` merely
*produces* MIL; MIL is what CoreML consumes.

Requirements if an ML Program is ever needed: `compute_precision=FLOAT16` (FP32 gets
no ANE) and explicit `ct.TensorType(dtype=np.float32)` on inputs/outputs so the
Swift boundary stays FP32.

**Consequence:** should we ever ship an ML Program, torch never needs to be a build
dependency — consistent with ADR-002.

---

## 9. Secondary finding: Architecture A ships `DOUBLE` I/O

| Model | input | output |
|---|---|---|
| A `neuralnetwork` | `tfidf_vector` MultiArray **DOUBLE** [4718] | `logits` MultiArray **DOUBLE** [57] |
| B `mlprogram` | `tfidf_vector` FLOAT32 [1, 4718] | `logits` FLOAT32 [1, 57] |

`NeuralNetworkBuilder` defaults the interface to `DOUBLE`, so Swift must fill a
37.7 KB Float64 buffer per call (4718 × 8 B) and CoreML casts every element.
Flipping the spec to `FLOAT32` halves the buffer and removes the cast.

Measured model-side latency difference is **noise** (`A_nn_full` 0.0452 ms vs
`Aprime_nn_f32io_full` 0.0439 ms; pruned 0.0374 vs 0.0378 ms), and the compute plan
is unchanged. The win is on the Swift buffer-fill side, which this harness does not
time. **Low priority, low risk, not a blocker.**

---

## 10. What was not measured, and why it does not change the conclusion

No physical iPhone was available, so there are **no mJ/inference figures, no
P-core/E-core attribution, and no thermal data**. All numbers above are M2 Max /
macOS 26.5. Absolute milliseconds will differ on A15/A17.

The conclusion is nonetheless robust:

- The ANE penalty is a **fixed-overhead** story (dispatch, IOSurface allocation,
  power-gate wake, weight DMA) amortised over 270 k MACs ≈ 0.5 MFLOP. Fixed
  overhead gets **worse** on a smaller ANE, not better.
- iPhone ANE power-gating is **more** aggressive than a mains-powered Mac's, so the
  §3.2 post-idle penalty should grow on-device, not shrink.
- `phys_footprint` accounting (§5) is the same mechanism on iOS.
- The load-cost (§4) and reproducibility (§7) arguments are architectural and
  device-independent.

An energy study would *validate* a decision already determined by wall-clock,
memory and determinism — it would not change it. `bench3.swift` and `bench4.swift`
run unmodified in an iOS XCTest target if device confirmation is wanted later.

---

## 11. Decisions that follow

1. **Keep Architecture A** (`NeuralNetworkBuilder`, `neuralnetwork`, FP32). Do not
   migrate the intent head to ML Program. 5.6× faster warm, ~30× faster on the real
   post-idle path, 6.5× cheaper to load, 2.8× less app RAM, deterministic to
   1.7e-06, iOS 11+ instead of iOS 15+, no torch.
2. **Set `configuration.computeUnits = .cpuOnly`** in `VoiceIntentKit`. Cuts cold
   load 93.7 → 15.6 ms and makes the logits reproducible. `.all` currently pays for
   ANE/GPU service init and executes on CPU regardless.
3. **Hold one `MLModel` instance for the app's lifetime.** Loading is a function of
   object lifetime, not of compute unit; neither backend reloads per prediction.
4. **Ship `.mlmodelc` as the primary artifact.** Keep the `MLModel.compileModel`
   branch in Swift as defence against older bundles; consider dropping the
   `.mlpackage` from the shipped `.nlu` (~1.4 MB saved) since the real fallback is
   the pure-Swift weights JSON.
5. **Pin the CI Xcode version** that runs `xcrun coremlcompiler` and record it in
   `bundle.json`; smoke-test on the oldest supported iOS.
6. **Restate the FP16 KPI as a delta gate** vs the FP32 reference. The current
   ">95 % absolute" is not met by the model today.
7. **Revisit for the semantic layer only.** The ANE conclusion is specific to a
   single 57-class linear head fired sporadically. The 23 MB MiniLM embedder
   (ADR-007, currently disabled) is a legitimate ANE candidate: enough compute to
   amortise dispatch, and it already exports as FP16 `mlprogram` with a fixed
   sequence length for exactly that reason.

---

## 12. When to re-open this decision

**This document records measurements, not a preference.** "CPU over ANE" is the
answer for *one 57-class linear head fired once per voice command* — it is not a
general position on the Apple Neural Engine, and it should not be applied by
analogy to any other model in this system.

Re-open the question, and re-run the harness, if **any** of the following becomes
true:

| Trigger | Why it changes the answer |
|---|---|
| The head stops being a single linear op — extra layers, a small transformer, semantic rescue (ADR-007) re-enabled | ANE overhead is fixed; more real compute amortises it |
| Invocation becomes continuous or always-on rather than one-shot per command | §3.2's power-gating penalty only bites at low duty cycle |
| Feature dimension grows ~10× | The planner already flipped CPU→ANE preference between 1317 and 4718 (§2); at some size CPU stops being cheapest |
| Apple ships a documented prewarm/residency API, or changes ANE gating policy | §3.2 is entirely a consequence of current gating behaviour |
| CPU inference stops being comfortably sub-millisecond under the latency budget | The margin that makes this decision easy disappears |
| **Anyone re-runs `bench3.swift` / `bench4.swift` on real iPhone hardware and gets materially different numbers** | §10 flags this as the largest untested assumption |

The correct way to challenge this decision is to re-run the harness and post
numbers — not to argue from first principles in either direction. That applies
equally to arguments *for* the ANE and arguments *against* it.

Models in this system that are **not** covered by this decision and remain open
ANE candidates: the 23 MB MiniLM embedder (ADR-007), which already exports as FP16
`mlprogram` with a fixed sequence length specifically to stay ANE-resident, and any
future streaming or always-on component.

---

## Related

ADR-017 (this decision) · ADR-006 (FP16 `mlprogram` — qualified by this evaluation
for the intent head) · ADR-014 (Fat Bundle) · ADR-016 (pruned + full heads) ·
`.claude/memory/mobile.md` · `docs/coreml-conversion-guide.md`
