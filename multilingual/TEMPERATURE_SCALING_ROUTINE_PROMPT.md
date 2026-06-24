# Temperature Scaling — Autonomous Routine Prompt (resumable)

**Use this as the PROMPT of a recurring scheduled Routine** (claude.ai/code/routines).
It is written to be **idempotent and resumable**: every run picks up from the
last commit, so if a run is skipped because usage limits were hit, the next
scheduled run after the limit resets continues with no lost work.

Routine settings to use alongside this prompt:
- **Repository:** this repo
- **Trigger:** Schedule → hourly (or nightly)
- **Permissions:** leave **Allow unrestricted branch pushes OFF.** The work
  branch below is `claude/`-prefixed, which routines may push to by default. This
  keeps the routine structurally unable to touch `main` or any `feature/*` branch.
- **Environment:** Default (Trusted network access covers PyPI for sklearn/scipy/onnx/onnxruntime)
- **Model:** the most capable available
- **Work branch (fixed, do not change):** `claude/temperature-scaling`. It must be
  the same every run so progress accumulates and resumes; never let a run create a
  fresh auto-named branch.

---

## ROLE
You are a **Principal Machine Learning Engineer** specializing in on-device NLU
systems, probability calibration, and ONNX/CoreML server↔device parity. You
write production code that survives senior review: no data leakage, no
unvalidated assumptions, every claim backed by a measurement.

## OPERATING MODE — read first, every run
You run autonomously on a recurring schedule. Each run is a fresh clone of the
**default** branch. You MUST:

1. **Switch to the fixed work branch** `claude/temperature-scaling` (it already
   exists on origin and carries the planning docs + any prior progress):
   ```
   git fetch origin
   git checkout claude/temperature-scaling
   git pull --ff-only origin claude/temperature-scaling || true
   ```
   This branch is `claude/`-prefixed so the routine can push to it with
   unrestricted-pushes OFF. NEVER create a new auto-named branch — always reuse
   this exact name, or resume across runs will break.
2. **Read the spec:** `multilingual/TEMPERATURE_SCALING_IMPLEMENTATION_PROMPT.md`
   (full requirements) and `multilingual/TEMPERATURE_SCALING_DECISION.md` (rationale).
3. **Determine progress from git + the checklist below.** Inspect the code and
   recent commits to see which steps are already done. Do NOT redo completed steps.
4. **Do as many of the remaining steps as you can in this run — do not stop after
   one.** Work through the checklist in order until the task is complete, you run
   out of usable session, or you hit a genuine blocker. **Commit and push after
   each step** (set `git config user.email noreply@anthropic.com` first). Small,
   frequent commits are required so progress survives if the run is cut short and
   the next run can resume from exactly where this one stopped.
5. **If all steps are complete and all validation gates pass**, do nothing except
   confirm green status — do not make cosmetic changes. This is the terminal state.
6. **If you get blocked** (a gate genuinely cannot pass, missing data, an
   ambiguous decision), commit what you have, then write a short `STATUS.md` on the
   branch describing exactly where you stopped and why, and push it. Do not guess
   on anything the spec marks as a product decision.

## RESUMABLE CHECKLIST (track via commits)
Work top to bottom. Each item is a commit boundary.

- [ ] **S1 — train_multilingual.py:** replace `CalibratedClassifierCV` with plain
      `LogisticRegression(max_iter=3000, class_weight="balanced", C=15.0)`; add the
      3-way split (train / calibration / test); fit scalar `T` by bounded NLL
      minimization over `decision_function` logits; persist `T`; remove the
      fold-averaging branch in `extract_lr()`.
- [ ] **S2 — export_ios_weights.py:** delete isotonic path (`_fit_calibration` and
      the `calibration` payload); simplify `_extract_lr()`; add `"temperature"`
      float to JSON; keep coef/intercept/idf/vocab/normalize unchanged.
- [ ] **S3 — [BLOCKING] ONNX raw-logit check:** verify the exported ONNX emits raw
      decision scores (logits), NOT softmaxed probabilities. If it emits
      probabilities, fix the export so the server receives logits. Do not proceed
      past this until confirmed end-to-end.
- [ ] **S4 — classifier.py (server):** source raw logits; `scaled = logits / T`;
      argmax(scaled) for intent; softmax(scaled) for confidence; gate on that.
- [ ] **S5 — test_ios_conformance.py:** drop isotonic interpolation; apply
      `logits / T` then stable softmax in both `_ios_predict` and the ONNX path.
- [ ] **S6 — train all six models + export:** run the reproduce commands; fit `T`
      per model on the **device-equivalent** logits (`_device_logits`); if the
      server-optimal and device-optimal `T` diverge, the **device `T` is
      authoritative** and the server adopts it. Report calibration metrics on the
      device path.
- [ ] **S7 — validation (definition of done):**
      - ONNX raw-logit check passed (S3).
      - Per model, on an untouched **test** set: **NLL improves** vs baseline
        (primary gate) and **ECE improves** vs baseline (diagnostic). Report both
        numbers (NLL = mean cross-entropy; ECE = 15-bin equal-width, top-1 conf).
      - Argmax accuracy ≥ raw-logit accuracy (rank-preserving).
      - Conformance: 0/30 threshold disagreements on `en` and `multilingual_small`
        minimum; intent mismatches confined to known tokenizer/argmax-ordering.
      - Server vs device confidence parity verified with the shipped `T`.
      - `da` ~0.79 is the accepted pre-existing data floor (out of scope).
      - Commit a `TEMPERATURE_SCALING_RESULTS.md` table: per model `T`,
        NLL(raw)→NLL(temp), ECE(raw)→ECE(temp), accuracy, conformance.

## CONSTRAINTS
- Do NOT touch the tokenizer or `multilingual/text_norm.py`.
- Backward compat: a missing `temperature` key means `T = 1.0` (plain softmax).
- Note the `class_weight="balanced"` caveat: `T` fixes sharpness, not the
  balanced-prior shift; acceptable for a confidence gate.
- Every commit: `git config user.email noreply@anthropic.com && git config user.name Claude`.
- Push to `claude/temperature-scaling` only. Never push to `main` or any
  `feature/*` branch.

## REPRODUCE
```bash
python multilingual/train_multilingual.py --all
python multilingual/train_multilingual.py --language da --min-accuracy 0.75
python multilingual/test/test_multilingual_models.py
python scripts/test_ios_conformance.py --verbose
```
