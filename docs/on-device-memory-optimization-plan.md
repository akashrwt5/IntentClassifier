# On-Device Memory Optimization & Model Distillation Plan

**Status:** Proposal / planning
**Author:** NLU / ML
**Scope:** Reduce the iOS RAM footprint of the Stage 3b semantic encoder
(`MiniLMEmbedder`) **without materially losing classification quality.**
**Primary ask:** a distillation recipe (teacher = `all-MiniLM-L6-v2` or
`e5-small-v2` → smaller student). **Secondary ask:** every other lever we have
for controlling model size and memory.

---

## 0. TL;DR — the advisor's bottom line

> **Do not start a multi-week distillation project until you have proven the
> weights are the problem.** Your FP16 encoder is ~45 MB on disk but is
> reportedly consuming >100 MB of RAM. That ~55 MB delta is *runtime overhead*,
> and it is very likely fixable in **days with zero accuracy loss** — before we
> touch the model architecture at all.

Recommended sequence (each tier is a gate — only proceed if the previous tier
didn't get you under budget):

| Tier | Lever | Effort | Quality risk | Expected RAM win | Verdict |
|---|---|---|---|---|---|
| **0** | **Diagnose where the 100 MB goes** | hours | none | — | **Do first, always** |
| **1** | Runtime/config fixes (compute units, fixed shape, lazy load + unload, `max_len` 64→32) | 1–3 days | **none** | Often **the whole fix** | **Do first** |
| **2** | Stronger compression (4-bit palettization, per-grouped LUT, vocab pruning, Matryoshka dims) | 2–5 days | small, measurable | 2–3× on weights | Do if Tier 1 insufficient |
| **3** | **Distillation to a 2–4 layer student** (your main ask) | 1–3 weeks | small–moderate | ~1.3–1.5× weights + lower activations | Do if you need a genuinely smaller model |
| **4** | **Static embeddings (Model2Vec-style)** — drop the transformer entirely | 3–7 days | moderate (domain-dependent) | **10×+, near-zero runtime RAM** | **Highest upside — run as a parallel experiment** |

My recommendation: **run Tier 0 + Tier 1 immediately**, run **Tier 4 (static
embeddings) as a cheap parallel spike**, and treat **Tier 3 (transformer
distillation)** as the fallback if static embeddings drop too much OOS-rejection
quality. Reasoning is in §3–§7.

---

## 1. Current architecture (what we're optimizing)

The NLU pipeline is a staged cascade (see `docs/semantic-understanding-plan.md`,
`docs/coreml-conversion-guide.md`):

```
Stage 1  keyword pre-filter            (cheap, always on)
Stage 2  TF-IDF + LogReg               IntentClassifier.mlpackage   (tiny)
Stage 3  SEMANTIC RESCUE — only fires when Stage 2 confidence < threshold (~10% of turns)
   3a    SemanticHead (LogReg, 60×384) SemanticHead.mlpackage       (~90 KB)
   3b    MiniLM-L6-v2 sentence encoder MiniLMEmbedder.mlpackage     ← THE MEMORY HOG
```

Stage 3b facts (from `scripts/export_coreml.py`, `scripts/nlu/semantic.py`,
`models/`):

- Encoder: `sentence-transformers/all-MiniLM-L6-v2` — **6 layers, 384 hidden,
  ~22.7M params, BERT WordPiece vocab of 30 522 tokens**.
- iOS runtime: **CoreML** (`MiniLMEmbedder.mlmodelc`), FP16, `RangeDim`
  sequence axis **1..64**, `iOS16` target, output `last_hidden_state [1,seq,384]`,
  mean-pooled + L2-normalised in `SemanticEmbedder.swift`.
- On-disk: **FP16 ≈ 45 MB**, **6-bit palettized ≈ 22 MB** (already implemented in
  `quantize_minilm_int8()`).
- The head (3a) is trained SetFit-style on a **frozen** encoder
  (`scripts/train_semantic_head.py`) over ~8 000 in-domain phrases + a curated
  out-of-scope (OOS) class. **The encoder is never fine-tuned.**

### 1.1 Where the parameters actually live (this drives everything)

| Component | Params | Share | Shrinks if we cut **layers**? | Shrinks if we go **static**? |
|---|---:|---:|:---:|:---:|
| Token embedding table (30 522 × 384) | ~11.7M | **~52%** | ❌ no | ✅ (and can be pruned) |
| Positional + type embeddings | ~0.2M | ~1% | ❌ | ✅ removed |
| 6 × transformer layers (~1.77M each) | ~10.6M | **~47%** | ✅ yes | ✅ removed entirely |
| **Total** | **~22.7M** | 100% | | |

**Critical consequence:** *more than half the weights are the token-embedding
lookup table, which layer-distillation cannot touch.* Cutting 6→2 layers only
takes FP16 from ~45 MB to ~31 MB — there is a hard floor unless we also attack
the embedding table (vocab pruning / factorization) or remove the transformer
(static embeddings). **This is the single most important fact in this document**
and is why Tier 4 has so much more upside than Tier 3 on *weights*.

---

## 2. Tier 0 — Diagnose before you optimize (do this first)

We are optimizing a number we have not yet attributed. Spend a few hours here; it
will redirect the entire project.

**Instrument peak RAM in the iOS app** around the semantic path:

1. Use Xcode **Instruments → Allocations / VM Tracker**, or log
   `task_vm_info.phys_footprint` before/after each step.
2. Measure four checkpoints:
   - app baseline (no NLU models loaded),
   - after `IntentClassifier` + `SemanticHead` load,
   - after `MiniLMEmbedder` **loads**,
   - during/after a `MiniLMEmbedder` **inference** at `seq_len = 64`.
3. Repeat with `MLModelConfiguration.computeUnits` set to `.all`,
   `.cpuAndNeuralEngine`, and `.cpuOnly`.

**What the answer tells you:**

| Observation | Likely cause | Go to |
|---|---|---|
| Big jump on **load**, before any inference | Weights duplicated across compute units; flexible-shape model can't go ANE-resident and is staged on GPU+CPU | Tier 1 |
| Big jump only **during inference** | Activation buffers / worst-case `seq=64` allocation | Tier 1 (`max_len`) |
| Footprint stays high **after** the ~10% semantic turn ends | Model kept resident; never unloaded | Tier 1 (lazy load/unload) |
| Footprint ≈ weights and still too big | Genuinely a weights problem | Tier 2 → 3 → 4 |

> Until Tier 0 is done, every estimate below the "weights" line is a hypothesis.

---

## 3. Tier 1 — Runtime & config fixes (no quality loss)

These change *how* the existing model runs. **None of them alter a single weight,
so accuracy is mathematically unchanged.** In practice this tier resolves most
"CoreML uses way more RAM than the file size" reports.

1. **Pin the compute unit.** `.all` lets CoreML place the graph on ANE *and* GPU
   *and* CPU, which can duplicate weight buffers. Try
   `config.computeUnits = .cpuAndNeuralEngine` (and compare `.cpuOnly`). For a
   22M-param encoder run ~10% of the time, CPU-only is often the smallest- and
   simplest-footprint option, and latency is still single-digit ms.

2. **Kill the flexible `RangeDim` shape.** Today the model is converted with
   `seq = RangeDim(1, 64)` (`export_coreml.py:342`). Flexible-shape models
   frequently **cannot be ANE-resident** and force a GPU/CPU path that
   pre-allocates for the *maximum* length. Replace with either:
   - a **single fixed length** (`shape = (1, 32)`) and pad in Swift, or
   - a **small enumerated set** of buckets (e.g. 8/16/32) via
     `ct.EnumeratedShapes`.
   This is usually both a memory **and** latency win. (See the
   `EnumeratedShapes` / fixed-shape note already in
   `docs/coreml-conversion-guide.md` Troubleshooting.)

3. **Cut `max_len` 64 → 32.** Your utterances are short hearing-aid commands;
   inspect the token-length distribution of `data/01_source_base_training_data.csv` — almost
   everything will be < 24 WordPiece tokens. Halving the sequence axis halves the
   attention/activation working set. Change `max_len` in `SemanticEmbedder.swift`
   (and keep `scripts/nlu/semantic.py:_tokenise` / `train_semantic_head.py` in
   sync so train-time and run-time match).

4. **Lazy-load and release.** Stage 3b fires on ~10% of turns. Don't hold the
   encoder resident for the other 90%:
   - instantiate `MiniLMEmbedder` on first semantic rescue,
   - release it (drop the `MLModel` reference) after an idle timeout.
   Peak RAM still spikes during a rescue, but **steady-state** drops to ~0 for the
   encoder. If your concern is sustained footprint, this alone may close the case.

5. **Load order / single instance.** Ensure exactly one `MLModel` instance per
   model is cached app-wide (no per-call re-instantiation, no accidental two
   copies from the `mlmodelc`/`mlpackage` fallback in
   `coreml-conversion-guide.md` §6).

**Exit criterion:** re-run Tier 0 measurement. If you're under budget, **stop
here** — you've spent days, not weeks, and lost zero accuracy.

---

## 4. Tier 2 — Stronger compression (small, measured quality loss)

If Tier 1 isn't enough and the weights themselves must shrink, compress before
you re-architect. All of these are post-training and reuse your existing eval
harness (`scripts/compare_coreml_quant.py`).

1. **Go from 6-bit to 4-bit palettization.** `quantize_minilm_int8()` already does
   6-bit k-means palettization (45 → ~22 MB). Try `nbits=4`
   (`cto.OpPalettizerConfig(mode="kmeans", nbits=4)`), optionally with
   **per-grouped-channel LUTs** (`granularity="per_grouped_channel",
   group_size=...`) to recover accuracy. 4-bit ≈ ~12–14 MB weights.
   **Always** gate with `compare_coreml_quant.py` (ship bar in §8).

2. **Joint compression (palettize + activation/int8) and `OpLinearQuantizer`.**
   coremltools `optimize.coreml` supports combining weight palettization with
   8-bit activation quantization. More aggressive; measure carefully — the
   existing code comments (`export_coreml.py:387`) document that *linear* INT8
   produced `QLinearMatMul` ops the Metal backend couldn't compile, which is why
   palettization was chosen. Stay on the palettization path unless you re-verify
   Metal compilation.

3. **Attack the embedding table directly (the 52%).**
   - **Vocabulary pruning:** your domain touches a small slice of the 30 522
     WordPiece tokens. Build the set of tokens that actually occur across
     `data/*.csv` (+ keep all single-character/sub-word pieces so nothing becomes
     truly OOV), remap IDs, and slice the embedding matrix. Dropping to ~6–8k
     tokens removes ~9M params (~18 MB FP16) **with no change to the transformer
     math** for in-vocab tokens. This is the highest-leverage, lowest-risk
     structural change available and is under-used for closed domains.
   - **Embedding factorization (ALBERT-style):** factor 30 522×384 into
     30 522×128 · 128×384. Needs a short fine-tune/distill to recover; folds
     naturally into Tier 3.

4. **Matryoshka / dimensionality reduction (384 → 256 or 128).** Shrinks the head
   (3a) and any stored vectors, and slightly the encoder output. Two routes:
   - learned **Matryoshka** objective during distillation (Tier 3), or
   - a cheap **PCA projection** fit on teacher embeddings, applied after pooling.
   Re-train the head on the reduced dim (`train_semantic_head.py` already embeds
   then fits — just change the dim).

**Exit criterion:** under budget with `compare_coreml_quant.py` verdict =
acceptable → ship. Otherwise → Tier 3 / Tier 4.

---

## 5. Tier 3 — Distillation (the main ask)

This is the requested deliverable: distill the encoder into a smaller student.
The crucial framing for *our* system:

> **We do not need a general-purpose encoder.** We need embeddings that keep our
> 60 intents linearly separable and keep near-domain OOS *rejectable*. That makes
> **task-specific distillation** dramatically more tractable than the
> general-purpose distillation that produced MiniLM in the first place.

### 5.1 Teacher selection — E5-small vs MiniLM-L6-v2 (your question)

The teacher runs **only offline, at training time**. Its size/latency are
**free** — they never ship. Therefore: **distill from the strongest teacher you
can run offline, not from what you currently deploy.**

| Candidate teacher | MTEB | Layers | Symmetric? | Notes for us |
|---|---:|---:|:---:|---|
| `all-MiniLM-L6-v2` (current) | 56.3 | 6 | ✅ | Self-distill to fewer layers; **caps student ≤ 56.3** |
| `e5-small-v2` | 62.1 | 12 | ⚠️ needs `query:`/`passage:` prefixes | Higher ceiling, but its asymmetric prompting is friction for our symmetric single-utterance use |
| **`bge-small-en-v1.5`** | **62.2** | 12 | ✅ no prefix | **Recommended teacher** — strongest symmetric small model, drop-in |
| `bge-base` / `e5-base` / `gte-base` | ~64–65 | 12 | ✅/⚠️ | Even higher ceiling if offline compute allows; bigger = slower labeling only |

**Recommendation:** teacher = **`bge-small-en-v1.5`** (or `e5-small-v2` if you
prefer, accepting the prefix handling). Distilling from a 62-MTEB teacher into a
small student can yield a student **better than today's deployed MiniLM-L6
(56.3)** while being smaller — you raise the ceiling and shrink the model at the
same time.

> ⚠️ **Consequence of changing teacher:** the student's embedding space is new, so
> you **must** retrain the semantic head (Stage 3a) on student embeddings and
> **re-tune the 0.55 rescue threshold**. Both are already automated
> (`train_semantic_head.py` re-embeds and refits; the rejection-curve printout
> gives the new threshold). Budget for it; it's cheap.

### 5.2 Student architecture — three options

| Option | Definition | Weights (FP16) | Pros | Cons |
|---|---|---:|---|---|
| **A. Truncated MiniLM** | 6→**3** (or →2) layers, 384 hidden, init from teacher's first N layers | ~35 MB (→2: ~31 MB) | Lowest risk, reuses entire toolchain unchanged | Embedding table floor; modest weight win |
| **B. Slim student** | 4 layers + hidden 384→256 + factorized/pruned embeddings | ~12–18 MB | Real win, attacks the 52% table | More tuning; needs head dim change |
| **C. Static (Model2Vec)** | **No transformer** — distill to token→vector table + mean-pool | ~8–30 MB **and ~0 runtime RAM** | Biggest memory + latency win | Loses contextual word sense; see §6 |

Start with **A** (fastest path, proves the pipeline), measure, then decide
whether **B** or **C** is worth the extra effort. **C is covered separately in
§6 because it's strong enough to run in parallel now.**

### 5.3 Distillation data (this is where projects succeed or fail)

Embedding-matching needs **far more sentences than head-training does**. Fitting
a student on only the ~8 000 in-domain phrases will overfit the manifold and
**collapse OOS rejection**. The beauty of distillation: the teacher labels
*unlabeled* text for free, so data is cheap.

Assemble a distillation corpus of **100k–500k sentences**:

- **In-domain core:** all `data/01_source_base_training_data.csv` phrases (~8k).
- **In-domain augmentation:** LLM paraphrases of each phrase (Claude — see
  `docs/claude-api` references in repo tooling), back-translation
  (en→de/fr/es→en), and EDA (synonym swap, insert/delete). Target ~5–10×.
- **Near-domain negatives / OOS:** expand `data/semantic_oos.csv` heavily — this
  is what preserves the *rejection* behaviour. Include general-knowledge,
  other-app, and "almost a command" phrasings.
- **General robustness:** a broad open corpus slice (e.g. paraphrase datasets,
  generic web sentences) so the student doesn't forget how to embed anything
  outside the 60 intents — important for OOS detection.

**Leakage guard:** exclude every phrase in `data/semantic_holdout_100.csv` and
`data/semantic_benchmark_250.csv` from the distillation corpus
(`train_semantic_head.py` already does this for the head; mirror it here).

### 5.4 Distillation objective

Combine three losses (all on L2-normalised, mean-pooled sentence embeddings):

1. **Embedding match (primary):**
   `L_embed = 1 − cos(student(x), teacher(x))` (or MSE). This transfers the
   teacher's geometry.
2. **Task term (keeps us discriminative where it matters):**
   `L_task = CE(head(student(x)), intent_label)` on the labeled in-domain subset,
   with the explicit OOS class — same setup as `train_semantic_head.py`. Can be a
   jointly-trained linear head or periodic refit.
3. **Relational/distribution match (preserves separation & rejection):**
   `L_rel = KL(sim_student ‖ sim_teacher)` over pairwise cosine-similarity
   matrices within a batch (a.k.a. similarity/relational distillation). This is
   what keeps near-domain OOS *far* from intents.

Optional **Matryoshka** wrapper on `L_embed` if you also want 384→256/128.

`L = α·L_embed + β·L_task + γ·L_rel` (start α=1, β=0.5, γ=0.5; tune on holdout).

### 5.5 Training recipe (concrete defaults)

- **Init:** copy teacher's token-embedding matrix + first N transformer layers
  into the student (huge convergence speedup vs random init).
- **Framework:** `sentence-transformers` supports distillation directly
  (`losses.MSELoss` for embedding match) — reuse it; it's already an optional dep
  in `requirements.txt` and the runtime in `semantic.py`.
- **Optimizer:** AdamW, LR 5e-5 (3e-5–1e-4), linear warmup 10%, weight decay 0.01.
- **Schedule:** 3–10 epochs over the distillation corpus, batch 128–256, fp16/bf16
  mixed precision.
- **Hardware/time:** single mid-range GPU, a few hours. This is not a large run.
- **Checkpoint selection:** pick by **downstream** metric (head accuracy +
  OOS rejection on holdout), **not** by distillation loss.

### 5.6 Export & deploy (reuse existing toolchain)

1. Save student as a HuggingFace model dir.
2. Point `MINILM_HF_NAME` in `scripts/export_coreml.py:68` at the student path,
   re-run `python scripts/export_coreml.py` → new `MiniLMEmbedder.mlpackage`.
   - **Use a fixed/enumerated shape here (Tier 1 item 2), not `RangeDim`** — bake
     the memory win into the export.
3. `python scripts/export_coreml.py --quantize` to palettize (4-bit per §4).
4. **Retrain the head** on student embeddings:
   `python scripts/train_semantic_head.py` (it re-embeds with the new encoder and
   refits; also regenerates the manifest).
5. **Re-tune the threshold** from the rejection-curve printout; update
   `DEFAULT_THRESHOLD` in `scripts/nlu/semantic.py` if it moved.
6. Regenerate `models/minilm-vocab.txt` if you pruned the vocab, and keep the
   Swift WordPiece tokeniser in sync.
7. Validate: `python scripts/compare_coreml_quant.py` (ONNX vs FP16 vs palettized).

---

## 6. Tier 4 — Static embeddings (Model2Vec): the dark horse

For a **closed-domain command classifier**, the strongest memory play is to
remove the transformer entirely. **Model2Vec**-style static distillation turns a
sentence-transformer into a **token → vector lookup table**; at inference you
embed a sentence by averaging its token vectors. No attention, no layers.

Why it fits us unusually well:
- Our utterances are short, fixed-domain commands where word *order/context*
  matters less than vocabulary presence — exactly where static embeddings lose
  the least.
- **Runtime RAM ≈ the table size**, with **no activation buffers, no ANE/GPU
  staging, no compute-unit duplication** — it sidesteps the entire ~55 MB
  overhead that is likely the actual problem.
- Latency drops from ~8 ms to **microseconds**.
- A vocab-pruned static table for our domain can be **single-digit MB**.

Costs/risks:
- Loses contextual sense (e.g. polysemy) — must be validated on **OOS rejection**,
  where the quality drop usually shows up first.
- It's still distillation, so the §5 data and §8 eval protocol apply.

**Plan:** distill `bge-small-en-v1.5` → static model with Model2Vec, prune the
table to the domain vocab, retrain the head (`train_semantic_head.py`), tune the
threshold, and run the §8 ship bar. This is a **3–7 day spike** and, if it
passes the OOS bar, it is the best outcome on every axis (RAM, latency, size,
battery). **Run it in parallel with Tier 1.**

---

## 7. Decision matrix — what I'd actually choose

| If Tier 0 shows… | …then the right lever is | Why |
|---|---|---|
| Overhead/config dominates (likely) | **Tier 1** (fixed shape, compute unit, lazy load, `max_len`) | Free, days, zero quality loss |
| Weights dominate, want minimal risk | **Tier 2** (4-bit palettize + vocab prune) | Post-training, measured, no retrain of encoder |
| Want a smaller *model* with headroom to spare | **Tier 3** (distill from `bge-small`, 6→3 layers) | Can end up smaller **and** more accurate than today |
| Want the maximum win and OOS holds up | **Tier 4** (static / Model2Vec) | 10×+ RAM, removes the transformer and its overhead |

**My recommendation, concretely:** ship Tier 1 now; in parallel run the Tier 4
static-embedding spike; keep Tier 3 transformer distillation as the fallback if
static embeddings miss the OOS bar; use Tier 2 (vocab pruning + 4-bit) on
whichever encoder you keep.

---

## 8. Evaluation protocol & ship bar (non-negotiable)

Every candidate — quantized, distilled, or static — passes through the **same**
gate, extending the existing `scripts/compare_coreml_quant.py` philosophy.

**Metrics (vs the current FP16 MiniLM baseline):**

1. **In-scope accuracy** on `data/semantic_holdout_100.csv`.
2. **OOS rejection rate** on `data/semantic_oos.csv` (the one most likely to
   regress — watch it closely).
3. Second opinion on `data/semantic_benchmark_250.csv`.
4. **Embedding fidelity** vs teacher (mean/min cosine) and **decision-flip count**
   (already reported by `compare_coreml_quant.py`).
5. **On-device peak RAM** (Tier 0 method) — the metric we're actually optimizing.
6. **Latency** on a real device.

**Ship bar (reuse the existing convention):**

> Accept iff **in-scope accuracy Δ ≥ −1.0%** *and* **OOS rejection not worse**,
> *and* peak RAM under budget. Otherwise keep the previous model — a size/RAM win
> is not worth a rejection regression in a hearing-aid product.

---

## 9. Risks & rollback

| Risk | Mitigation |
|---|---|
| **OOS rejection collapses** after distillation (student over-smooths near-domain negatives) | Heavy OOS in distill corpus (§5.3) + relational loss (§5.4); gate on OOS rejection (§8) |
| Threshold drift after encoder swap | Re-tune from `train_semantic_head.py` rejection curve; never reuse old 0.55 blindly |
| Train/run skew (tokeniser, `max_len`, pooling) | Keep `semantic.py`, `train_semantic_head.py`, and Swift `SemanticEmbedder` in lockstep; `compare_coreml_quant.py` exists to catch exactly this |
| CoreML compile failures on new ops (seen before with linear INT8 → `QLinearMatMul`) | Stay on palettization path; re-verify Metal compile per `export_coreml.py:387` notes |
| Vocab pruning causes OOV on novel input | Keep all sub-word/character pieces so any word still decomposes; validate on holdout |
| Distillation under-converges | Init from teacher layers; select checkpoint by downstream metric, not loss |

**Rollback:** keep the current FP16 `MiniLMEmbedder.mlpackage` as the shipped
default; introduce the new encoder behind a build flag / staged rollout; the
JSON/Swift fallbacks already let the app degrade gracefully if a model fails to
load.

---

## 10. Recommended milestones

1. **M0 (this week):** Tier 0 diagnosis — attribute the 100 MB. *Deliverable:* a
   table of the four RAM checkpoints × three compute-unit settings.
2. **M1:** Tier 1 runtime fixes (fixed shape + compute unit + `max_len 32` +
   lazy load). Re-measure. *Likely closes the gap.*
3. **M2 (parallel):** Tier 4 static-embedding spike from `bge-small-en-v1.5`;
   run the §8 ship bar. Decision: static viable or not?
4. **M3 (if needed):** Tier 3 transformer distillation (6→3 from `bge-small`),
   retrain head, re-tune threshold, ship-bar.
5. **M4:** Tier 2 compression (4-bit palettize + vocab prune) on the chosen
   encoder; final ship-bar; staged rollout.

---

## Appendix A — Parameter math (why the table dominates)

```
token embeddings : 30 522 × 384            = 11 720 448   (~52%)
position + type  : (512 + 2) × 384         =    197 376
per layer        : 4·(384×384) attn
                   + (384×1536 + 1536×384) ffn ≈ 1.77M
6 layers         : 6 × 1.77M               ≈ 10.6M        (~47%)
total            ≈ 22.7M params  →  FP16 ≈ 45 MB on disk
```

Cutting 6→2 layers saves ~7M params → ~31 MB FP16: a floor set by the embedding
table. To go below it you must prune/factorize the table or drop the transformer
(Tier 4).

## Appendix B — File / command index

| Purpose | File |
|---|---|
| Encoder runtime (Python ref) | `scripts/nlu/semantic.py` |
| Head training (re-embeds + refits) | `scripts/train_semantic_head.py` |
| CoreML export + palettization | `scripts/export_coreml.py` |
| Three-way quality gate | `scripts/compare_coreml_quant.py` |
| iOS integration steps | `docs/coreml-conversion-guide.md` |
| Model/encoder rationale | `docs/semantic-understanding-plan.md` |
| Eval data | `data/semantic_holdout_100.csv`, `data/semantic_oos.csv`, `data/semantic_benchmark_250.csv` |

```bash
# Re-export after swapping the encoder (point MINILM_HF_NAME at the student first)
python scripts/export_coreml.py            # FP16 CoreML
python scripts/export_coreml.py --quantize # + palettized variant
python scripts/train_semantic_head.py      # retrain head on new embeddings
python scripts/compare_coreml_quant.py     # ship-bar gate
```
