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
| Remove Engage.zip, checkpoints/, skeleton dirs | A#9 | ✅ closed — items already absent (ND-4 approved, nothing to delete) |
| Repair iOS parity CI (INTENTCLASSIFIER_PAT) | A#4 | ⏸ gated — secret provisioning (ND-7) |

## Phase 1 — Platform foundations (primary target, exit gate §8.1)

| Task | Ref | State |
|---|---|---|
| spec/bundle/3.0 JSON Schemas + portable-regex subset + 2 golden bundles | ADR-005 AI#1,2 | ✅ done — 16 schemas, portable-regex.md + corpus, minimal+full golden bundles, conformance tests green (tests/test_bundle_spec.py). Format 3.0 declared initial versioned format (AI#1). |
| Bundle compiler stages 1–10 as shared validator library | ADR-005 §5 | ✅ v0 done — `packages/buildtime/nlu_compiler/` (stages 1–5, 8–10 validation halves; transforms/ingestion arrive with the full compiler). CLI `python -m nlu_compiler <dir>`. 13 tests incl. mutation coverage of every Part-6 failure class. |
| Bundle compiler stages 11–15 (packaging/signing) | ADR-005 §5 | ⏸ signing gated (ND-8) |
| packages/ + apps/cli + spec/ restructure | roadmap §13 | ✅ done — M0–M4 executed with byte-identical oracle parity at every phase (see ND-2 row) |
| Label-space cleanup (dialogue-act labels, taxonomy migration) | roadmap §2.3, ADR-002 A7 | ✅ done — executed under ND-3 approval (see ND-3 row); migration map is CI-checkable via datasets/label_migration_map.json |
| Repartition content/ into capabilities + capability.yaml | ADR-002 AI#3,4,6 | 🟡 map materialized — `docs/Review-F5/capability-map.json` (12 capabilities, machine-checked over all 59 intents; supersedes ND-3 plan's naming sketch). Physical repartition follows ND-2 M3 + ND-3 approval. |
| nlu_training package + unified evaluate JSON | roadmap §9.2 | ✅ v0 done — `packages/buildtime/nlu_training` (`python -m nlu_training`): one report_card-schema-valid JSON (per-lang F1/acc/ECE, wrong-action proxy count + per-domain, OOS recall, gates). Reproduces the recorded baseline exactly. |
| DVC (datasets) + MLflow (local file backend) | roadmap §16 | ⬜ unblocked |
| Real tests/ pytest tree (unit/component/golden/parity/perf) | roadmap §15 | 🟡 started (tests/ exists: smoke, datetime parity, phase-0 guards) |
| InferenceBackend / session-state / notifyExecution contracts (prose spec) | ADR-001 AI#3, ADR-002 AI#5 | ✅ done — `spec/contracts/runtime-contract-v1.md` (the engine_compat anchor: 5 seams, session blob schema, outcome taxonomy, snapshot push, ordering rules, Python conformance-gap table) |
| Freeze ADR-003 Part-12 interface contracts; import-linter in CI | ADR-003 AI#6,7 | ✅ done — contracts frozen via ratification + runtime-contract-v1.md; boundary rules CI-enforced by tests/test_package_boundaries.py (AST-based: runtime↛buildtime, compiler↛engine, content/spec code-free, engine dependency-lean) |

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
| 9 | Engage.zip / checkpoints / skeleton dirs | ✅ already absent (ND-4) |
| 10 | unknown_data.csv privacy | ✅ documented + enforced (ND-5) |

## Needs decision — approval-gated queue (charter §7)

| ID | Question | Blocking |
|---|---|---|
| ND-1 | ✅ **RESOLVED 2026-07-14 — Ratified.** ADRs 001–005 accepted as written (owner decision via decision prompt). Status lines updated in the ADR files; recorded as ADR-010 in `.claude/memory/decisions.md`. Phase 2 stays trigger-gated. | — |
| ND-2 | ✅ **APPROVED 2026-07-14 — executing.** M0 done (oracle frozen). M1 done (engine → packages/runtime/nlu_engine + shim). M2 done (trainers/data-scripts → packages/buildtime/nlu_training; exports → nlu_export; CLIs + unknown_log → apps/cli; Makefile targets updated; dev/debug/test scripts stay in scripts/ for M4). M3 done (content/ + datasets/ split). M4 done (docs/config/agents path sweep, pre-commit filter, bundles/ gitignored). **Restructure complete.** Open sub-items: (a) retire the scripts/nlu shim after one green cycle incl. STT-repo check, (b) DVC wiring (toolchain + remote decision), (c) dev/debug scripts remain in scripts/ by design. Every phase: 60/60 tests + evaluate byte-identical to oracle. | — |
| ND-3 | ✅ **EXECUTED 2026-07-14 (option A).** 59 → 57 labels, domain.object.action across 12 capabilities; 102k dataset rows + schema + overlays migrated; 1,741 yes/no rows → confirmation_fixtures.csv; Cmd.Health stragglers dissolved; all languages retrained + recalibrated; da re-exported at 0.70 gate (still flag-gated). Gates: accuracy held ±0.010 all langs; OOS recall improved .51→.59; de macro-F1 −0.030 (explained, tracked in known-issues). New baseline: tests/parity/oracle_post_migration. **Owner to-do: ship datasets/label_migration_map.json to the STT repo + regenerate iOS fixtures; regenerate local semantic artifacts.** | — |
| ND-4 | ✅ **RESOLVED 2026-07-14 — already clean.** Deletion approved, but `Engage.zip`, `checkpoints/`, and skeleton `multilingual_intent/` dirs no longer exist in the working tree or git index (removed in an earlier cleanup). A#9 closed, nothing to delete. | — |
| ND-5 | ✅ **RESOLVED 2026-07-14 — implemented.** `scripts/unknown_log.py`: counters by default (`data/unknown_counters.csv`), raw text only behind `NLU_COLLECT_RAW_UNKNOWN`; `predict.py` wired through it + its placeholder GenAI URL removed. Tests added. Open sub-item: retention window for consented raw text (legal). | — |
| ND-6 | ✅ **RESOLVED 2026-07-14 — keep placeholders.** Owner decision: `@intentclassifier/*` placeholders stay until the GitHub org/teams exist; rules inert but documented. | — |
| ND-7 | ⏸ **DEFERRED (owner decision 2026-07-14).** iOS parity CI stays skipped until `INTENTCLASSIFIER_PAT` is provisioned. A#4 remains open with owner. | A#4 |
| ND-8 | **Signing keys / KMS / trust root** decisions for bundle signing (Ed25519, rotation runbook). | Compiler stages 11–15, Phase 3 |
| ND-9 | **GenAI consent flow + legal review** for assist.cloud capability. | Phase 3 |

## Newly discovered backlog

- ✅ **ND-10 RESOLVED 2026-07-14 (untrack after regen check — executed).**
  Regen check via `train_multilingual.py --all`: en/fr/de/multilingual/
  multilingual_small all pass gates and re-export → untracked (30 files) +
  gitignored. **Exception: `da/` stays tracked** — Danish fails the 0.80
  accuracy gate (0.760) so its artifacts are NOT regenerable from a clean
  checkout; untracking them would lose the only copy. Revisit when the
  Danish native-data program lifts it past the gate (Phase 5).

## Newly discovered — SAFETY finding (2026-07-14, harness)

**System-level wrong-action budget is NOT met:** 99 wrong actions on the
shipped-language holdout replay (≈2–2.7% of turns) vs the ≤5 budget.
Dominant causes: device.volume polarity confusions and help→command
misfires; the CONFIRM gate never fires (no confirmation-required intents
in the schema). Mitigations are policy/safety changes → **ND-11 in the
decision queue**. Full data: `tests/parity/oracle_post_migration/
wrong_action_system_report.json`.

| ID | Question | Blocking |
|---|---|---|
| ND-11 | ✅ **(a)+(b) APPROVED & EXECUTED 2026-07-14.** Polarity guards (4 rules × 4 languages) + uncertainty-confirmation gate (<0.80, volume/streaming, localized ask-first prompts). Result: shipped wrong actions **99 → 73** (−26%); 30 wrong guesses intercepted by CONFIRM; friction 2–5% of turns. Budget (≤5) still not met — residual is non-gated domains + high-confidence confusions. **ND-11b follow-up decision:** extend gate to activity/translation/transcription/find? raise confirm bar? re-measure with semantic after artifact regen? | Wrong-action budget |

## Iteration log

- **2026-07-14 (i)** — Engine-in-the-loop wrong-action harness built + run
  (semantic off — stale local artifacts). System-level: en 39 / fr 32 /
  de 28 / da 33; CONFIRM layer confirmed unused. Safety finding recorded
  as ND-11 (owner decision needed on mitigations). Harness + report
  committed as the budget-measurement baseline.
- **2026-07-14 (h)** — ND-2 M0–M4 + ND-3 executed (see ND rows). Then:
  official wrong-action definition wired into evaluate (predicted-actionable
  + confident; raw-classifier upper bound 244/4-langs — budget gate needs
  the engine-in-the-loop harness, tracked); package-boundary rules enforced
  in CI (AST test). Post-migration baseline refreshed with the new fields.
  Next: engine-in-the-loop wrong-action harness, DVC/MLflow (owner input),
  compiler build stages.
- **2026-07-14 (g)** — Capability map materialized (12 capabilities ×
  59 intents, machine-checked; label plan amended to align intent domains
  with A3). Unified evaluate shipped: report_card-schema-valid JSON,
  baseline-exact (en .899/.893, fr .852/.840, de .833/.821, da .760/.728).
  Finding: v0 wrong-action proxy (confident misclassifications) = 323 — the
  official ≤5 budget needs the curated actionable-command suite; da OOS
  recall 0.51 is weak (both noted in known-issues). Next: DVC/MLflow wiring
  or ND-2/ND-3 execution on approval.
- **2026-07-14 (f)** — CI now runs the spec-conformance gate (validator CLI
  over both golden bundles + jsonschema in CI deps). Runtime Interface
  Contract v1 authored (spec/contracts/) — the number engine_compat gates
  against; includes the Python engine's conformance-gap table as Phase-1/2
  work items. Next: capability repartition map (proposal), nlu_training/
  evaluate JSON, DVC/MLflow.
- **2026-07-14 (e)** — Shared validator library v0 shipped
  (packages/buildtime/nlu_compiler: stages 1–5, 8–10 + CLI + 13 tests, all
  Part-6 failure classes mutation-tested). ND-3 label-migration plan
  delivered (59-label inventory, baseline metrics recorded from the live
  regen run). Next: capability repartition proposal or nlu_training
  package; awaiting ND-2/ND-3 approvals for the gated work.
- **2026-07-14 (d)** — Phase 1 started. Authored spec/bundle/3.0 (16 JSON
  Schemas + defs), portable-regex.md with normative corpus, golden bundles
  minimal+full, conformance tests (8 passing; seed stages 1–3/8/9 checks).
  jsonschema added to dev deps. ND-2 move plan delivered for review.
  Next: shared validator library (compiler stages 1–10), ND-3 label plan.
- **2026-07-14 (c)** — Second decision round: ND-5 implemented
  (unknown_log.py counters-by-default + opt-in raw, predict.py wired +
  legacy placeholder GenAI URL removed, 6 new tests); ND-6 closed
  (placeholders intentional); ND-7 deferred (owner); ND-10 executed —
  30 artifact files untracked after live regen check, Danish kept tracked
  (fails 0.80 gate at 0.760, not regenerable). Next: ND-2 move plan,
  ND-3 label plan, spec/bundle/3.0 schemas.
- **2026-07-14 (b)** — Decision round: ADRs 001–005 ratified (ND-1) and
  recorded (ADR files + decisions.md ADR-010); ND-2/ND-3 set to plan-first;
  ND-4 closed — approved junk already absent from repo (A#9 done). Discovered
  tracked model artifacts (see backlog). Next: draft the ND-2 move plan and
  ND-3 label-migration plan; start spec/bundle/3.0 schemas (now unblocked).
- **2026-07-14** — Bootstrap iteration. Reconciled Phase 0; quarantined
  auto_label.py; rewrote README; added CODEOWNERS; GenAI placeholder-URL
  startup guard (+ CLI None-handling + tests/test_phase0_guards.py); privacy
  stance documented; memory synced; this tracker created. Next: verify gates,
  commit; then start spec/bundle/3.0 schema authoring (no-regret portion).
