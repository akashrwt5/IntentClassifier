# CoreML Export (FP16) for the Multilingual Intent Models — Autonomous Routine Prompt (resumable)

Use this as the PROMPT of a recurring scheduled Routine (claude.ai/code/routines).
It is written to be **idempotent and resumable**: every run picks up from the last
commit, so a skipped run (usage limits, timeout) loses no work — the next run
continues from the checklist state recorded in git.

## Routine settings to use alongside this prompt

- **Repository:** this repo.
- **Trigger:** Schedule → hourly (or nightly).
- **Permissions:** leave *Allow unrestricted branch pushes* **OFF**. The work
  branch below is `claude/`-prefixed, which routines may push to by default. This
  keeps the routine structurally unable to touch `main` or any `feature/*` branch.
- **Environment:** Default (Trusted network access covers PyPI for
  `coremltools`, `onnxruntime`, `scikit-learn`, `scipy`, `numpy`, `torch`).
- **Model:** the most capable available.
- **Work branch (fixed, do not change):** `claude/coreml-export`. It must be the
  same every run so progress accumulates and resumes; never let a run create a
  fresh auto-named branch.

---

## ROLE

You are a **Principal AI/ML Engineer specializing in CoreML and on-device iOS
inference** — probability calibration, ONNX↔CoreML parity, the Apple Neural
Engine (ANE), and the realities of how Core ML schedules work on iPhone. You
write production code that survives senior review: **no data leakage, no
unvalidated assumptions, every claim backed by a measurement.** When the data
contradicts a convenient assumption, you report the data. You do not ship a
calibration or parity claim you have not numerically verified.

---

## OPERATING MODE — read first, every run

You run autonomously on a recurring schedule. Each run is a fresh clone of the
default branch. You MUST:

1. Switch to the fixed work branch (it carries this spec + any prior progress):
   ```
   git fetch origin
   git checkout claude/coreml-export 2>/dev/null || git checkout -b claude/coreml-export origin/claude/coreml-export 2>/dev/null || git checkout -b claude/coreml-export
   git pull --ff-only origin claude/coreml-export || true
   ```
   This branch is `claude/`-prefixed so the routine can push with unrestricted
   pushes OFF. **NEVER** create a new auto-named branch — always reuse this exact
   name, or resume across runs will break.
2. Read this spec and the supporting analysis:
   - `multilingual/TEMPERATURE_SCALING_RESULTS.md` (the calibration the device must reproduce),
   - `scripts/export_coreml.py` (the **existing** production CoreML export you are extending + fixing),
   - `scripts/test_ios_conformance.py` (the hand-rolled device scorer `_ios_predict` — the numeric reference).
3. **Determine progress from git + the checklist below.** Inspect the code and
   recent commits to see which steps are already done. Do NOT redo completed steps.
4. Do the next ONE incomplete step, then **commit and push immediately** with
   `git config user.email noreply@anthropic.com` set. Small, frequent commits are
   required so progress survives across runs.
5. If all steps are complete and all validation gates pass, do nothing except
   confirm green status — do not make cosmetic changes. This is the terminal state.
6. If you get blocked (a gate genuinely cannot pass, a step needs macOS this Linux
   runner does not have, an ambiguous product decision), commit what you have, then
   write/update a short `STATUS.md` on the branch describing exactly where you
   stopped and why, and push it. **Tier-B (real Core ML runtime) tests requiring
   macOS are an expected partial-block on a Linux runner — that is not a failure;
   record it and move on.** Do not guess on a product decision.

---

## BACKGROUND — measured facts (do not re-derive; these are ground truth)

These were measured directly on the shipped artifacts. Treat them as established.

- **Why CoreML and not ONNX Runtime on device:** ONNX Runtime adds a 3rd-party
  binary to the iOS app (~2-5 MB ORT-Mobile/minimal, ~10-20 MB full). Core ML is
  part of iOS (zero added binary), and the app **already** links Core ML for the
  Stage-3 MiniLM transformer. So Stage-2 reuses an existing inference surface at
  zero marginal binary cost. **This binary-size argument is the reason CoreML is
  chosen — not performance.** Do not relitigate it.
- **The model is a sparse linear classifier.** Each model is one matrix–vector
  product `(K=59 × V) · (V × 1) → 59 logits`, then softmax. The TF-IDF input is
  ~0.07-0.26% dense (≈4-10 non-zero features per utterance). Re-expressed as a
  **dense** CoreML matmul it does 388×-1484× redundant arithmetic vs the sparse
  Swift gather. This is a deliberate, accepted trade (it costs microseconds /
  microjoules / tens of KB, not memory budget). **Do not try to "fix" it with a
  sparse op — CoreML cannot express sparse inputs.** TF-IDF stays in Swift; CoreML
  is a linear-layer + label container only.
- **Per-model dimensions** (multilingual models ship FULL vocab; production is pruned):

  | Model | K×V | weights FP32 / FP16 | per-call input | redundancy |
  |---|---|---|---|---|
  | production (en, pruned) | 59×1370 | 316 KB / 158 KB | 5.4 KB | 388× |
  | en | 59×3843 | 886 KB / 443 KB | 15.0 KB | 439× |
  | fr | 59×4198 | 968 KB / 484 KB | 16.4 KB | 404× |
  | de | 59×4629 | 1067 KB / 533 KB | 18.1 KB | 540× |
  | da | 59×3839 | 885 KB / 442 KB | 15.0 KB | 458× |
  | multilingual | 59×14195 | 3272 KB / 1636 KB | 55.4 KB | 1484× |
  | multilingual_small | 59×4017 | 926 KB / 463 KB | 15.7 KB | 680× |

- **FP16 is decision-safe.** Rounding every coefficient through float16 and
  re-scoring the 30-utterance conformance set: max logit delta ≤ 3.5e-3, max top-1
  confidence delta ≤ 8.2e-4, **0/30 argmax flips, 0/30 0.70-gate flips on every
  model.** Therefore **FP16 is the required precision** (halves package size and
  matmul bandwidth at no decision cost). FP32 is an optional fallback artifact only.
- **Calibration is Swift-side, on `logits`.** The CoreML graph emits raw `logits`
  plus a baked `classProbability = softmax(logits)` at **T=1**. The device contract
  is: **read `logits`, compute `softmax(logits / T)` in Swift, gate on that.**
  `classProbability` is a T=1 trap — never consume it for the 0.70 gate.
- **Temperature-scaling gap in the current export:** `scripts/export_coreml.py`
  still documents **isotonic** calibration (lines ~27-45, 199, 222) and **never
  reads or embeds `temperature`**. The graph is calibration-agnostic (it only emits
  logits), so it is not *broken*, but the docs are stale and the package carries no
  `T`. Part of this task is to fix that.

---

## RESUMABLE CHECKLIST (track via commits)

Work top to bottom. Each item is a commit boundary. Do the next incomplete one.

- [ ] **S1 — New exporter `multilingual/export_coreml_multilingual.py`.**
  For each of the six models (`en, fr, de, da, multilingual, multilingual_small`),
  build a CoreML package from `multilingual/models/<m>/<m>_intent_classifier_weights.json`:
  - **Format `mlprogram`, precision FP16** (`compute_precision=ct.precision.FLOAT16`).
    Preferred build path: construct a tiny `torch.nn.Linear(V, 59)`, load `coef`
    into `.weight` and `intercept` into `.bias`, `torch.jit.trace`, then
    `ct.convert(..., convert_to="mlprogram", compute_units=ct.ComputeUnit.ALL,
    inputs=[ct.TensorType(name="tfidf_vector", shape=(1, V), dtype=np.float32)])`.
    If `torch` is unavailable, fall back to `NeuralNetworkBuilder` +
    `quantization_utils.quantize_weights(m, nbits=16)` and record the format
    limitation in the commit message.
  - **Fixed input shape `(1, V)`** — no flexible/RangeDim shapes (they disqualify ANE).
  - **Outputs:** expose `logits` (primary). If you also emit `classProbability`,
    document it as T=1 / not-for-gating.
  - **Embed metadata** (`user_defined_metadata`): `temperature`, `conf_threshold`,
    `conf_gap_threshold`, `normalize`, `vocab_size`, and a short note documenting the
    `softmax(logits / T)` Swift contract. Also keep `--fp32` to emit an optional
    FP32 fallback artifact.
  - Save to `multilingual/models/<m>/IntentClassifier_<m>.mlpackage`; refresh
    `manifest.json` hashes. CLI: `--model en|all`, `--fp16` (default), `--fp32`.

- [ ] **S2 — Fix the production exporter `scripts/export_coreml.py`.**
  Remove all isotonic references in docstrings + `output_description`; read
  `temperature` from `models/intent_classifier_weights.json` and embed it (plus
  thresholds) in the package metadata; document the `softmax(logits / T)` contract.
  Keep the `logits` output and graph shape unchanged. Optionally add the FP16/
  `mlprogram` path so production matches the multilingual exporter. Do NOT change
  the numeric behavior of the exposed logits.

- [ ] **S3 — Tier-A test (platform-independent) `multilingual/test/test_coreml_multilingual.py`.**
  Runs anywhere (incl. this Linux runner) — proves numeric equivalence WITHOUT the
  Core ML runtime:
  1. Load each `.mlpackage` **spec**; assert input dim == V with **fixed** shape,
     output names present, class labels embedded, and `temperature` metadata present
     and == the JSON value.
  2. Extract the weight matrix + bias **from the CoreML spec** and assert they equal
     the JSON `coef`/`intercept` within FP16 tolerance (proves the export carried the
     right numbers).
  3. Run a NumPy reference scorer (identical math to `_ios_predict`: sublinear TF-IDF
     over the pruned vocab → L2-normalize → `coef·x + intercept`) over each model's
     holdout (`multilingual/test/<m>_holdout.csv`) and assert it matches the **ONNX
     raw logits** within tolerance. Transitively this proves CoreML==ONNX.

- [ ] **S4 — Tier-B test (macOS-only, auto-skip elsewhere).**
  Guard with `platform.system() == "Darwin"`; on Linux **skip with a clear printed
  reason** (do not fail). On macOS:
  - `mlmodel.predict()` over each holdout; compare CoreML `logits` vs ONNX `logits`
    vs the NumPy reference (report max abs delta; tolerance accounts for FP16 ≈ 1e-2 on probs).
  - Report **CoreML holdout accuracy** and assert it is within ε of ONNX accuracy.
  - **Threshold parity** on the 30 conformance utterances: apply `softmax(logits/T)`,
    assert **0 disagreements** at the 0.70 gate vs ONNX.

- [ ] **S5 — ONNX vs CoreML comparison report `multilingual/COREML_RESULTS.md`.**
  Per model, a table: vocab V, package size FP16 (and FP32 if emitted), ONNX accuracy
  vs CoreML accuracy (Tier-B, or "macOS-pending" if not yet run), max logit delta,
  max top-1 confidence delta, threshold disagreements / 30, and the agreement %.
  Include the Tier-A numeric-equivalence result (which IS available on Linux) so the
  doc is meaningful even before a macOS run. State explicitly which rows are Linux-
  verified (Tier-A) vs macOS-pending (Tier-B).

- [ ] **S6 — [BEST-EFFORT, NON-BLOCKING] ANE eligibility investigation.**
  Goal: determine whether these FP16 `mlprogram` models can run on the Apple Neural
  Engine (saves CPU/battery on the phone). This is *not* a gate — report findings,
  do not block on it.
  - Document the expected reality from the measured facts: a tiny sparse linear op
    (~531 MACs of real work) will very likely be scheduled on **CPU/BNNS** even with
    `ComputeUnit.ALL`, because there is no dense FLOP volume to amortize ANE's data-
    staging cost, and ANE would run the dense (not sparse) matmul. FP16 + fixed shape
    + `mlprogram` already maximize *eligibility*; residency is the runtime's choice.
  - Provide a **procedure to actually check** on Apple hardware:
    (a) `MLModel.get_compute_plan()` / `coremltools.models.compute_plan` (macOS) to
    report each op's preferred + supported compute devices (ANE/GPU/CPU);
    (b) the Xcode **Core ML Instruments / performance report** procedure to observe
    actual per-layer placement on a device.
  - If running on macOS, run the compute-plan check and record which compute unit each
    op is assigned. If on Linux, write the procedure + expected outcome into
    `COREML_RESULTS.md` under an "ANE eligibility" section and mark it macOS-pending.
  - Note the one lever that would *most* help ANE/CPU efficiency regardless: **vocab
    pruning** (shrinks V like production's 14195→1370). Flag it as a follow-up, do not
    implement unless asked (it changes the model).

- [ ] **S7 — Validation (definition of done).** See the gates below. When all pass
  (Tier-A everywhere; Tier-B on macOS or explicitly macOS-pending), and
  `COREML_RESULTS.md` is committed, this is the terminal state.

---

## VALIDATION — definition of done

- **FP16 packages exist** for all six multilingual models (+ production via S2),
  `mlprogram`, **fixed `(1, V)`** input, `logits` output, `temperature` in metadata.
- **Tier-A (Linux-runnable) passes for every model:** spec shape/labels/metadata
  correct; spec weights == JSON `coef`/`intercept` within FP16 tol; NumPy reference
  == ONNX raw logits within tol. This is the mandatory gate that runs on this routine.
- **Tier-B (macOS) passes OR is explicitly recorded as macOS-pending in `STATUS.md`/
  `COREML_RESULTS.md`** — never silently skipped. When run: CoreML accuracy within ε
  of ONNX; **0/30 threshold disagreements** at the 0.70 gate on the conformance set.
- **ONNX vs CoreML comparison table** committed in `COREML_RESULTS.md` (per model:
  accuracy, max logit delta, max confidence delta, threshold disagreements, sizes).
- **Production `export_coreml.py` de-isotonic'd** and temperature-aware (S2).
- **ANE finding documented** (S6) — eligible/likely-CPU, with the on-device check
  procedure. Non-blocking.
- Backward compat preserved: a missing `temperature` key ⇒ T = 1.0 (plain softmax).

---

## CONSTRAINTS

- Do NOT touch the tokenizer or `multilingual/text_norm.py`. TF-IDF stays in Swift;
  CoreML receives the pre-computed L2-normalized vector.
- Device contract is **read `logits`, apply `softmax(logits / T)` in Swift** — never
  gate on the baked T=1 `classProbability`.
- FP16 is the required/primary precision; FP32 is an optional fallback artifact.
- Fixed input shape only — no flexible shapes (ANE eligibility + simplicity).
- Do not modify the trained weights, the ONNX models, the temperature values, or any
  temperature-scaling artifact. This task only adds CoreML packaging + tests.
- Every commit: `git config user.email noreply@anthropic.com && git config user.name Claude`.
- Push to `claude/coreml-export` only. Never push to `main` or any `feature/*` branch.

---

## REPRODUCE

```bash
# Build FP16 CoreML packages for all six multilingual models
python multilingual/export_coreml_multilingual.py --all --fp16

# Fix + rebuild the production package (temperature-aware, de-isotonic'd)
python scripts/export_coreml.py            # or the FP16 path if added

# Tier-A: numeric equivalence vs ONNX — runs on Linux/CI/macOS
python multilingual/test/test_coreml_multilingual.py

# Tier-B: real Core ML runtime accuracy + threshold parity — macOS only (auto-skips elsewhere)
python multilingual/test/test_coreml_multilingual.py --runtime

# Existing parity reference (ONNX vs hand-rolled device scorer), all six models
python scripts/test_ios_conformance.py --model all
```
