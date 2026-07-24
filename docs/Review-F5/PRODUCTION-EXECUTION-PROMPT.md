# Production Execution Prompt
## Standing charter for continuously driving IntentClassifier → the on-device conversational AI platform described in `docs/Review-F5/`

> **How to use this file.** Paste it (or run `/execute-production-roadmap`) at the
> start of a Claude coding session in this repo. It is a *standing charter*, not a
> one-shot task: run the loop in §6 repeatedly until every exit gate in §8 is green
> or you hit an approval gate in §7. Do one small, verified, reversible increment
> per iteration. Never break the shipping system.

---

## 0. Mission — what "done" means

Turn the current working *pipeline* into the *platform* the Review-F5 documents
specify: one signed versioned **NLU Bundle** as the deployment unit, a
**configuration-driven** intent/dialogue model, **capability**-organized features,
a **conversation orchestrator/planner**, GenAI as a bounded fallback capability,
a real **test tree + CI + MLOps**, and a path to **runtime unification + Android**
— reached in small increments that keep the existing iOS/Python system working
and never regress model quality or the medical wrong-action budget.

You are done for now when the §8 exit gates are green. You are *never* done by
taking a shortcut through an approval gate (§7) or a STOP rule (§4).

---

## 1. Source of truth & precedence

Read the *specific* file the current task needs — do not reload everything. Prefer
the **Code Graph Memory MCP** for code navigation and **Context7** for library
APIs (see `CLAUDE.md`).

1. **Law — the ADRs** (`docs/Review-F5/adr-00{1..5}-*.md`). Architecture decisions.
   On conflict: a later ADR refines an earlier one; an ADR overrides the roadmap.
   - ADR-001 shared runtime strategy (Option B now; Rust core only at the trigger)
   - ADR-002 capability & action-execution / SDK architecture
   - ADR-003 conversation orchestration (Orchestrator / Planner / Dialogue / Policy)
   - ADR-004 GenAI routing & cloud escalation (`assist.cloud` capability)
   - ADR-005 NLU Bundle specification & compiler (the cross-team contract)
2. **Plan — the roadmap** (`docs/Review-F5/production-architecture-review-and-roadmap.md`):
   Phases 0–5 (§19), gaps G1–G7, risks RK1–RK10, and Appendix A (the 10 P0/P1 defects).
3. **Current state — memory** (`.claude/memory/*.md`). Keep it in sync every iteration.
4. **Repo rules** (`CLAUDE.md`, `CONTRIBUTING.md`).

> The ADRs are **Proposed**, not Accepted. Treat their *no-regret preparatory work*
> as actionable now; treat anything they mark "Board ratifies", or that is
> irreversible/safety-critical, as an approval gate (§7).

---

## 2. Non-negotiable operating principles

- **Never rewrite working code unnecessarily.** Small, localized, reversible diffs.
  Every change stays green through `make check` and the relevant gates (§9).
- **Preserve behavior and cross-platform parity.** Any change to shared logic must
  keep the frozen Python oracle's outputs identical on the replay corpus, or the
  difference must be explicitly intended and documented.
- **Medical-safety first.** Never regress the wrong-action budget (≤5 global, and
  moving toward per-domain budgets). High-cost actions stay confirmation-gated.
- **Keep docs and memory in sync in the same change** (roadmap.md, decisions.md,
  known-issues.md). Stale docs are a defect.
- **One logical change per commit**, conventional commits, on `feature/production-work`.
  Prefer small PRs that pass `pr.yml` CI.
- **Config over code, facts over behavior** (ADR-001.1 four-question test): if two
  platforms must agree on it, it belongs in the shared layer / bundle as data.

---

## 3. Working method each iteration

Use the task list (TaskCreate/TaskUpdate) so progress is visible. For anything
non-trivial, end with a verification step. Prefer the repo's specialized agents:
`architect` (design/ADR reading), `ml-engineer` (data/training/eval), `python-engineer`
(refactor/typing), `mobile-ml-engineer` (export/parity), `reviewer` (pre-merge),
`tester` (coverage), `documentation` (memory/docs).

---

## 4. STOP rules — things the ADRs forbid; never do these

- **No Rust/C++ now.** ADR-001: do Option B (config-compiled logic) now; build the
  shared Rust core *only* at the Android-multi-turn-dialogue trigger, and only after
  ratification. Do not start it autonomously.
- **Do not ship Danish.** Machine-translated; macro-F1 ≈ 0.745. Keep it behind a flag
  until it passes on a native-authored holdout (ADR/roadmap §9.3).
- **Do not run or trust `auto_label.py`.** It writes a retired label taxonomy and
  would poison training data. Quarantine/rewrite only (Appendix A #1).
- **Do not invent a DSL** or put executable logic in the bundle. New behavior = a new
  versioned runtime vocabulary behind a `required_runtime_features` flag (ADR-005 §12).
- **Do not remove unavailable intents from the classifier label space** (ADR-002 A6);
  route recognized-but-unavailable to the capability's `unavailable_response`.
- **GenAI is narration-only.** No path from a cloud response to the Action Dispatcher;
  escalation sends the *current utterance only*, never context/history/device state
  (ADR-004).
- **Runtimes gate on exactly three things** — format version, engine compat/features,
  signature. Never branch on compiler/content version (ADR-005 §8).
- **Do not delete data, models, or files that aren't obviously junk** without approval.

---

## 5. Backlog — phased, derived from roadmap §19 + ADR action items

Reconcile against the repo first (§11): several Phase-0 items are already done. For
each task keep the ADR/section citation so the "why" is traceable.

### Phase 0 — Stop the bleeding (no-regret; partially DONE)
- [x] Lock dependencies (`requirements.lock`, bounds) · [x] `pyproject.toml` · [x] PR CI
  (`ci.yml`) · [x] fix French datetime defects · [~] repo hygiene.
- [ ] Fix/merge the root `README.md` (documents a 3-generations-old system) — Appendix A #2.
- [ ] Quarantine `auto_label.py` (make it refuse to run; add a banner) — Appendix A #1, RK5.
- [ ] Remove `Engage.zip`, `checkpoints/`, skeleton `multilingual_intent/` dirs (keep any
  needed corpora in DVC) — Appendix A #9. **(deletion → approval gate §7)**
- [ ] Add `CODEOWNERS` separating `data/`/content from `scripts/nlu/` — roadmap §2.2.
- [ ] Add a startup guard rejecting the placeholder `DEFAULT_GENAI_URL` — Appendix A #5, RK1(sec).
- [ ] Document the `unknown_data.csv` privacy stance (default = counters; raw text opt-in
  only) — Appendix A #10, RK6.
- [ ] Repair the iOS parity CI (needs `INTENTCLASSIFIER_PAT`) — Appendix A #4, ADR-001 AI#2.
  **(secret provisioning → approval gate)**

### Phase 1 — Platform foundations (**blocks everything**) — EXIT GATE at §8.1
- [ ] Restructure into `packages/` + `apps/cli` + `spec/` per roadmap §13 / ADR-005 §13.
  Mechanical, behavior-preserving; parity replay proves it. **(large restructure → approval gate)**
- [ ] Label-space cleanup: remove `Cmd.SendMessage - yes/- no` classifier labels (→ yes/no
  lexicons + confirmation tests); decide the `Default Fallback Intent` TF-IDF class; fix the
  intent taxonomy to `domain.object.action` with a migration map — roadmap §2.3, R6; ADR-002 A7.
  **(changes shipped label set/model → approval gate)**
- [ ] Author `spec/bundle/3.0/` JSON Schemas + portable-regex subset + two golden bundles —
  ADR-005 AI#1,2.
- [ ] Build the **bundle compiler** stages 1–10 as a *shared validator library* first, then
  11–15 (packaging/signing) — ADR-005 §5, AI#3. **(signing → approval gate §7)**
- [ ] Repartition `content/` into capabilities (§A3 map); add `capability.yaml` to compiler
  inputs; codegen action-key constants/param structs + completeness gate — ADR-002 AI#3,4,6.
- [ ] Migrate current artifacts into the source layout as the compiler's first input —
  ADR-005 AI#4.
- [ ] Convert trainers into a `nlu_training` package (`datasets/gates/calibrate/export`),
  one unified `evaluate` JSON report — roadmap §9.2.
- [ ] Wire DVC (datasets) + MLflow (local, file backend) — roadmap §16.
- [ ] Real `tests/` pytest tree (unit/component/golden/parity/perf) — roadmap §15.
- [ ] Define `InferenceBackend` + session-state + `notifyExecution` + availability-snapshot
  contracts in prose/spec (no Rust) — ADR-001 AI#3, ADR-002 AI#5.
- [ ] Freeze ADR-003 Part-12 interface contracts; wire import-linter/package rules into CI —
  ADR-003 AI#6,7.

### Phase 2 — Runtime unification + Android (**RATIFICATION-GATED**, §7)
Rust `nlu-core` + UniFFI shells; Android first NLU = the shared core. Do **not** start
before ADR-001 is Accepted *and* the Android-multi-turn trigger fires. Preparatory,
allowed now: the config-driven tables (Option B) and the interface contracts above.

### Phase 3 — Lifecycle & observability
OTA channel + remote config + staged rollout + rollback; Ed25519 bundle signing + pinned
keys + rehearsed rotation runbook (ADR-005 §11); telemetry event schema + on-device
aggregation + per-`bundle_id`/per-domain dashboards; `assist.cloud` GenAI capability with
consent flow (ADR-004). **(signing keys, consent/legal → approval gates)**

### Phase 4 — Training Studio (Tauri) — after the compiler API is stable (Phase 1)
Dataset Manager + Intent Designer + Testing Console first (unblocks linguists for Danish),
then Training Dashboard + Model Management + Flow Designer — roadmap §12.

### Phase 5 — Language & workflow scale-out (ongoing)
Native-data program for Danish/next languages; migrate all 59 intents to full workflow
schema (confirmations for high-cost intents, universal verbs, timeout prompts); grow the
conversation golden corpus to 200+ scripts.

---

## 6. The continuous loop

Repeat until §8 is green or only approval-gated / blocked work remains:

1. **Sync.** Read `.claude/memory/roadmap.md` + `EXECUTION_STATUS.md`; reconcile with the
   repo (`git log`, `make check`, latest evaluate report). Update status if reality differs.
2. **Pick.** Choose the highest-priority **unblocked, non-approval-gated** task from §5,
   respecting dependencies (Phase 1 blocks Phases 2–5).
3. **Plan.** Record the task; state the approach and the ADR/section it satisfies. If it is
   major, irreversible, safety-, data-, or security-critical → **do not implement**; move it
   to the "Needs decision" queue (§10) and pick another task.
4. **Implement.** Smallest correct diff. Keep the shipping system working (feature flags,
   dual-path, or oracle parity). Add/extend tests alongside the change.
5. **Verify.** Run the applicable gates (§9). Everything must be green. For parity-sensitive
   changes, replay the corpus against the frozen Python oracle and diff (empty or explained).
6. **Record.** Update the matching memory file + `EXECUTION_STATUS.md`; conventional commit;
   open/annotate the PR. Note any newly-discovered work as backlog items.
7. **Repeat.** If blocked, log why in the "Needs decision" queue and take the next unblocked task.

Each iteration ends with a 2–3 line progress note (what changed, gate status, what's next).

---

## 7. Approval gates — pause and ask a human (do NOT proceed autonomously)

Present a short options summary and wait for a decision on:

- **Ratifying any ADR** or doing anything an ADR marks "Board ratifies" (ADR-001 AI#1,
  ADR-002 AI#1,2, ADR-003 AI#1, ADR-005 AI#1).
- **Building the Rust core** / triggering the Android-dialogue migration (ADR-001).
- **Signing keys / KMS / trust root / cert rotation** and anything security-critical
  (ADR-005 §11).
- **The repo-wide restructure into `packages/`/`spec/`** (large, even if mechanical) —
  propose the move plan and parity strategy first.
- **Changing the shipped label taxonomy, training data, calibration, or thresholds**
  (re-trains models; affects metrics/safety).
- **Anything touching the medical wrong-action path, privacy, or consent**
  (`unknown_data`, GenAI egress, session persistence encryption).
- **Deleting files** that are not obviously generated junk; **provisioning CI secrets**.

---

## 8. Definition of done (exit gates)

Stop the autonomous loop when these hold. Phases 4–5 are ongoing by design.

### 8.1 Phase 1 exit gate (the primary target)
- One command builds a **signed** (dev-key acceptable pre-Phase-3) bundle from a clean
  checkout; the golden example bundles load in the Python runtime in CI.
- `pr.yml` CI is green on **every PR** (lint, unit, component, compiler-validation, leakage
  guard); the shared validator is the single validation path.
- Label space cleaned (no dialogue-act labels), taxonomy migrated with a migration map,
  metrics **not regressed** vs the recorded baseline.
- `content/` repartitioned into capabilities; one unified `evaluate` JSON report is the CI
  gate artifact; datasets under DVC; experiments in MLflow.

### 8.2 Appendix-A defect list cleared
All 10 items resolved or explicitly deferred-with-owner in `known-issues.md`.

### 8.3 Overall v1 platform (later)
Phase 3 shipped: a bundle update reaches a 5% cohort and rolls back **without an app
release**; telemetry dashboards live; Danish only ships once it passes its native holdout.

---

## 9. Verification — what "green" means

- `make check` (Ruff + MyPy-advisory + pytest) green; `make format` (darker) clean on
  changed lines.
- **No quality regression:** holdout `--strict`, OOS recall, per-intent F1 floors, and the
  wrong-action budget (moving to per-domain) all hold vs the recorded baseline — the single
  `evaluate` JSON is the artifact.
- **Parity:** datetime parity CSVs, ONNX↔iOS conformance, and CoreML Tier-A/B pass for any
  touched surface; new shared logic is replayed against the frozen oracle.
- **Data changes** run the holdout-leakage guard.
- **Bundle builds** are byte-identical from identical inputs (once the compiler exists).

---

## 10. Progress tracking & reporting

- Maintain **`docs/Review-F5/EXECUTION_STATUS.md`**: per-task state, phase % complete, the
  **"Needs decision" queue** (approval-gated/blocked items with the exact question), and the
  last green commit hash.
- Update the matching **`.claude/memory/`** file every iteration — `roadmap.md` (progress),
  `decisions.md` (append an ADR entry when one is ratified), `known-issues.md` (open/close).
- Surface newly-discovered defects as backlog items immediately; never silently absorb them.

---

## 11. First actions (bootstrap this session)

1. **Reconcile current state** against Phase 0/1: already done — dependency lock + bounds,
   `pyproject.toml`, PR `ci.yml`, French datetime fix, memory/agents scaffolding. Confirm via
   `git log` and `make check`, then create/refresh `EXECUTION_STATUS.md`.
2. **Queue the approval-gated items** (§7) into "Needs decision" with crisp questions
   (ADR ratification, packages/ restructure, label-taxonomy change, signing, secrets, deletions).
3. **Start the loop (§6)** on the top unblocked, non-gated task — likely: quarantine
   `auto_label.py`, fix the root README, add `CODEOWNERS`, add the GenAI-URL startup guard,
   document the `unknown_data` privacy stance, then begin `spec/bundle/3.0/` schemas.

> Reminder: the goal is a production-grade platform reached *safely and incrementally*.
> When in doubt between "make progress" and "respect a gate," respect the gate and ask.
