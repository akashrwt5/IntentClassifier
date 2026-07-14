# EXECUTION STATUS — Review-F5 Production Roadmap

> Living tracker for the production execution charter. Updated every
> iteration. Source of truth for per-task state, the Needs-decision queue,
> and the last green commit.

- **Branch:** `feature/production-work`
- **Last green commit:** `febc3658` (ruff clean; pytest 33/33; mypy advisory — v2.3.0 crashes with an internal error in this env, tracked as tooling note)
- **Iteration date:** 2026-07-14

## Phase 0 — Stop the bleeding

| Task | Ref | State |
|---|---|---|
| Dependency lock (requirements.lock, bounds) | roadmap §19 | ✅ done (pre-existing) |
| pyproject.toml | roadmap §19 | ✅ done (pre-existing) |
| PR CI (ci.yml) | roadmap §19 | ✅ done (pre-existing) |
| French datetime defects | Appendix A | ✅ done (pre-existing, parity 25/25) |
| Quarantine auto_label.py (refuses to run + banner) | A#1, RK5 | ✅ done (this iteration) |
| Fix root README.md (was 3 generations stale) | A#2 | ✅ done (this iteration) |
| CODEOWNERS (data/content vs scripts/nlu) | roadmap §2.2 | ✅ done — team handles are placeholders (see ND-6) |
| Startup guard: reject placeholder DEFAULT_GENAI_URL | A#5, RK1 | ✅ done (this iteration; + tests) |
| Document unknown_data.csv privacy stance | A#10, RK6 | ✅ documented (`docs/privacy-unknown-data.md`); enforcement gated (ND-5) |
| Remove Engage.zip, checkpoints/, skeleton dirs | A#9 | ⏸ gated — deletion approval (ND-4) |
| Repair iOS parity CI (INTENTCLASSIFIER_PAT) | A#4 | ⏸ gated — secret provisioning (ND-7) |

## Phase 1 — Platform foundations (primary target, exit gate §8.1)

| Task | Ref | State |
|---|---|---|
| spec/bundle/3.0 JSON Schemas + portable-regex subset + 2 golden bundles | ADR-005 AI#1,2 | ⬜ next up (schema authoring is unblocked; ratification of ADR-005 itself is ND-1) |
| Bundle compiler stages 1–10 as shared validator library | ADR-005 §5 | ⬜ blocked by schemas |
| Bundle compiler stages 11–15 (packaging/signing) | ADR-005 §5 | ⏸ signing gated (ND-8) |
| packages/ + apps/cli + spec/ restructure | roadmap §13 | ⏸ gated — large restructure (ND-2) |
| Label-space cleanup (dialogue-act labels, taxonomy migration) | roadmap §2.3, ADR-002 A7 | ⏸ gated — changes shipped label set (ND-3) |
| Repartition content/ into capabilities + capability.yaml | ADR-002 AI#3,4,6 | ⬜ unblocked |
| nlu_training package + unified evaluate JSON | roadmap §9.2 | ⬜ unblocked |
| DVC (datasets) + MLflow (local file backend) | roadmap §16 | ⬜ unblocked |
| Real tests/ pytest tree (unit/component/golden/parity/perf) | roadmap §15 | 🟡 started (tests/ exists: smoke, datetime parity, phase-0 guards) |
| InferenceBackend / session-state / notifyExecution contracts (prose spec) | ADR-001 AI#3, ADR-002 AI#5 | ⬜ unblocked |
| Freeze ADR-003 Part-12 interface contracts; import-linter in CI | ADR-003 AI#6,7 | ⬜ unblocked |

## Phases 2–5

- **Phase 2 (Rust core / Android):** ⛔ do not start — ratification-gated (ND-1) + trigger not fired. Allowed prep: Option-B config tables, interface contracts.
- **Phase 3 (OTA, signing, telemetry, assist.cloud):** blocked on Phase 1; keys/consent gated (ND-8, ND-9).
- **Phase 4 (Training Studio):** blocked on stable compiler API.
- **Phase 5 (language scale-out):** ongoing; Danish stays flag-gated until native holdout passes.

## Appendix-A defect scoreboard

| # | Defect | State |
|---|---|---|
| 1 | auto_label.py poisons training data | ✅ quarantined |
| 2 | Root README 3 generations stale | ✅ rewritten |
| 3 | (see roadmap Appendix A) | ⬜ to reconcile next iteration |
| 4 | iOS parity CI broken (PAT) | ⏸ ND-7 |
| 5 | Placeholder DEFAULT_GENAI_URL | ✅ startup guard + tests |
| 6–8 | (see roadmap Appendix A) | ⬜ to reconcile next iteration |
| 9 | Engage.zip / checkpoints / skeleton dirs | ⏸ ND-4 |
| 10 | unknown_data.csv privacy | ✅ documented / ⏸ enforcement ND-5 |

## Needs decision — approval-gated queue (charter §7)

| ID | Question | Blocking |
|---|---|---|
| ND-1 | **Ratify ADRs 001–005?** They are Proposed; "Board ratifies" items (shared-runtime strategy, capability SDK, orchestration, GenAI routing, bundle spec) need explicit acceptance. | Phase 2 entirely; parts of 1 & 3 |
| ND-2 | **Approve the repo-wide restructure** into `packages/` + `apps/cli` + `spec/` (mechanical, parity-replay-proven)? A move plan + parity strategy will be proposed first. | Phase 1 structure |
| ND-3 | **Approve label-space cleanup**: remove `Cmd.SendMessage - yes/- no` classifier labels, decide Default-Fallback TF-IDF class, migrate taxonomy to `domain.object.action` with migration map? Re-trains models; affects metrics/safety. | Phase 1 exit gate |
| ND-4 | **Approve deletion** of `Engage.zip`, `checkpoints/`, skeleton `multilingual_intent/` dirs (corpora preserved in DVC first)? | A#9 |
| ND-5 | **Approve privacy enforcement change**: `predict.py` unknown-data logging → counters by default, raw text behind explicit opt-in consent flag; set retention window (legal input needed). | A#10 close-out |
| ND-6 | **Provide real GitHub team handles** for CODEOWNERS (placeholders `@intentclassifier/*` in place). | CODEOWNERS effectiveness |
| ND-7 | **Provision `INTENTCLASSIFIER_PAT`** secret so iOS parity CI runs. | A#4 |
| ND-8 | **Signing keys / KMS / trust root** decisions for bundle signing (Ed25519, rotation runbook). | Compiler stages 11–15, Phase 3 |
| ND-9 | **GenAI consent flow + legal review** for assist.cloud capability. | Phase 3 |

## Iteration log

- **2026-07-14** — Bootstrap iteration. Reconciled Phase 0; quarantined
  auto_label.py; rewrote README; added CODEOWNERS; GenAI placeholder-URL
  startup guard (+ CLI None-handling + tests/test_phase0_guards.py); privacy
  stance documented; memory synced; this tracker created. Next: verify gates,
  commit; then start spec/bundle/3.0 schema authoring (no-regret portion).
