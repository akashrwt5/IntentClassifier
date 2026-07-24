# Production Readiness Review — IntentClassifier (Review-F5)

**Reviewer role:** Principal engineer (on-device conversational AI / NLU platforms —
Alexa/Siri/Dialogflow class systems).
**Date:** 2026-07-24
**Branch reviewed:** `feature/production-work` @ `06d625c`
**Scope:** The whole repository, judged against the Review-F5 charter, ADRs 001–005,
the roadmap, and the §8 exit gates in `PRODUCTION-EXECUTION-PROMPT.md`.
**Method:** Independent verification — I re-ran the gates in a clean environment rather
than trusting `EXECUTION_STATUS.md`. Evidence is cited inline.

---

## 1. Verdict — Go / No-Go

**NO-GO for a production ship today. GO to proceed toward one — the platform is close,
and the remaining blockers are well understood.**

This is not a struggling project. The *engineering* is genuinely strong and the
Phase-1 platform exit gate (§8.1) is substantially met: a signed (dev-key) bundle
builds from a clean checkout, the shared validator is the single validation path, the
`packages/ + apps/ + spec/` restructure landed with byte-identical oracle parity, the
label taxonomy was migrated to `domain.object.action`, and the shipped-language models
are well-calibrated. If the bar were "is the platform architecture real and working,"
the answer would be yes.

But this is a **medical-adjacent product** (a hearing-aid assistant), and the charter's
first non-negotiable principle is *medical-safety-first: never regress the wrong-action
budget (≤5)*. That budget is **violated by roughly 8×**, and the failure mode is the
dangerous one — high-confidence wrong actions the current safety gates cannot catch.
That single fact is the reason for the No-Go, independent of everything else.

Two categories of blocker stand between the current state and production:

| # | Blocker | Type | Owner |
|---|---|---|---|
| **B1** | **Wrong-action safety budget not met** — 41 wrong actions on shipped langs vs a budget of 5; residue is high-confidence | **Hard engineering + data** | ML/eng |
| **B2** | **Reproducibility/CI gap** — datasets on a *local* DVC remote; CI never pulls them, so the safety & quality gates silently **skip** in CI | **Engineering** | eng |
| **B3** | Production bundle signing keys / KMS (ND-8) | Owner / security | owner |
| **B4** | GenAI (`assist.cloud`) consent + legal sign-off (ND-9) | Owner / legal | owner |
| **B5** | Phase-3 infra hosting — registry, remote config, telemetry backend (ND-12) | Owner / infra | owner |
| **B6** | iOS parity CI secret + branch protection on `main` (ND-7, A#6) | Owner / GitHub | owner |
| **B7** | 5 German intents ship with zero real German training data (ND-13); Danish machine-translated | Content / linguistics | owner + linguists |

B1 and B2 are the two I would not ship without. B3–B7 are real, but they are owner
business/legal/infra decisions that are already engineered *up to* the decision point —
the code is waiting behind them.

---

## 2. What "production ready" means here (the bar I graded against)

The Review-F5 charter defines three exit-gate tiers. My grading:

| Gate | What it requires | Status |
|---|---|---|
| **§8.1 Phase-1 platform** | Signed bundle from clean checkout; golden bundles load in CI; single validation path; label space cleaned; taxonomy migrated; metrics not regressed; content in capabilities; DVC + MLflow | 🟡 **Mostly green** — build/validator/restructure/taxonomy done; *metrics-not-regressed is not actually gated in CI* (B2) |
| **§8.2 Appendix-A defects** | All 10 P0/P1 defects resolved or deferred-with-owner | 🟢 **Essentially cleared** — 8 done, 2 deferred-with-owner (iOS PAT, duplicate semantic trees) |
| **§8.3 v1 platform** | Phase 3: OTA to a 5% cohort + rollback without an app release; telemetry dashboards live; Danish only ships on native holdout; **wrong-action budget held** | 🔴 **Red** — Phase-3 server side unbuilt (owner-gated); Danish not shippable; **budget violated** |

So: the platform *foundation* is production-grade; the *product* is not yet shippable.

---

## 3. Independent verification — what I actually ran

I did not take the status doc's "150/150" on faith. In a clean Linux environment the
repo's own `.venv` is unusable (built on macOS), so I installed the locked runtime deps
plus the dev tooling and ran the gates myself.

**Test suite (`pytest`):** **90 passed, 60 skipped, 0 failed.**
The "150/150" in the status doc is real *only on a machine that has the trained model
artifacts*. Every one of the **60 skips is a model-dependent test** — the wrong-action
mitigations, the unified `evaluate`, and the semantic-flag suites — skipped with
`trained artifacts not present (make train-multilingual)`. **The safety and quality
tests do not run from a clean checkout.** (See §5, B2 — this is the same reason they do
not run in CI.)

Two things worth flagging from that run:
- Three dev dependencies are needed by the compiler/signing tests but are **not in the
  lockfile or `pyproject`**: `referencing`, `cryptography`, and `jsonschema>=4.18`. CI
  installs `jsonschema` explicitly in its step but not `referencing`/`cryptography`;
  the bundle-build/lifecycle tests would error in CI exactly as they did for me until
  those are added. This is a small but real reproducibility defect.

**Compiler / validator (`python -m nlu_compiler`):** both golden bundles
(`spec/examples/3.0/{minimal,full}`) validate **clean — 0 errors, 0 warnings.** The
shared-validator-as-single-path claim holds.

**Bundle build + sign + verify (dev key):** `tests/test_bundle_build.py` — **9/9 pass**,
including tamper-detection, downgrade refusal, and a production-runtime-refuses-dev-key
test. The signing pipeline is real and correctly fails closed.

**Datasets:** managed by DVC, and the DVC remote is a **local filesystem path**
(`.dvc/config` → `url = ../../dvc-store`). The dataset CSVs are **not in the working
tree** and are **not retrievable by CI or by anyone cloning fresh** — only on the
owner's machine. This is the root cause of B2.

---

## 4. Safety audit — the hard blocker (B1)

**Source of truth:** `tests/parity/oracle_post_migration/wrong_action_system_report.json`
(the committed engine-in-the-loop replay; I could not regenerate it here because the
datasets are DVC-local, so I read the artifact directly).

```
budget_met = False
wrong_actions_shipped_langs = 41   (en 11, fr 16, de 14)   |   da 11 (waived)
budget = 5
```

The team has done real, competent safety work to get here — the trajectory is
99 → 73 → 65 → 55 → 46 → 41. The engine has a genuine, config-driven defense stack:
polarity guards, the ND-14 help-marker guard, and an uncertainty-confirmation gate.
This is the right architecture. **The problem is that it has hit the ceiling of what
confidence-gating can do**, because the residual errors fire at **high confidence
(0.87–1.00)**:

- **Polarity confusions inside `device.volume`** — "lower how loud it is" → *increase*
  (0.90); "hush it up" → *mute* (0.90). A hearing-aid user asking to lower the volume
  and getting it muted (or raised) is a direct comfort/safety event.
- **Out-of-scope → confident action** — "iphone" → `find.phone.locate` (0.999);
  "sende en besked til nogen" → `messaging.message.send` (1.00). Garbage in, confident
  action out.
- **Help-question → state-changing action** — "help for translate" →
  `translation.session.start` (0.90); "stream from an accessory mic…" →
  `streaming.session.start` (0.90). Asking *how* a feature works starts *using* it.

Because these fire above the 0.80 confirm threshold, the existing gate is blind to them
by construction. The `wrong_action_per_domain` in the evaluate baseline confirms the
concentration: `device` dominates (113 of the 223 raw-classifier wrong actions).

**This is the No-Go.** The wrong-action budget is the one line the charter says never to
cross, and it is crossed by ~8×. It cannot be closed by tuning a threshold; it needs
(a) targeted data/model work on the confusable clusters (volume polarity, OOS
rejection, help-vs-action), and (b) a policy decision to **confirmation-gate high-cost
state-changing intents regardless of confidence** — the one lever that catches
high-confidence errors. Note the charter itself flags "the CONFIRM gate never fires (no
confirmation-required intents in the schema)" as a root cause; the mitigation exists but
has not been extended to unconditional confirmation on the highest-cost actions.

I would also push back on one framing in the status trail: the budget is discussed
per-language and "shipped-lang total," but the charter's ≤5 is effectively a *global*
budget moving to *per-domain*. Even the best single language (en, 11) is 2× over. There
is no reading of the number under which it currently passes.

---

## 5. Reproducibility & CI audit (B2)

`ci.yml` is well-constructed: it installs from `requirements.lock` (good), runs Ruff
(blocking), darker on changed lines, MyPy (advisory per ADR-008), `pytest -ra`, and a
standalone bundle-spec conformance CLI pass over the golden bundles. It runs on pushes to
`feature/production-work` + `main` and on all PRs.

The problem is what CI **cannot** see:

- **No `dvc pull` step.** Datasets never arrive, so the model-dependent tests
  (wrong-action budget, unified `evaluate`, semantic) **skip in CI exactly as they did
  for me** — and `pytest -ra` reports skips without failing. **Green CI therefore does
  not mean the safety or quality gates passed; it means they didn't run.** The §8.1
  requirement that "the unified `evaluate` JSON is the CI gate artifact" and
  "metrics not regressed" is **not actually enforced.**
- The DVC remote is a **local path** (`../../dvc-store`), so even adding `dvc pull` would
  not help CI until a real shared remote (S3/GCS/etc.) is provisioned.
- **`main` has no branch protection** (Appendix A #6 — a GitHub setting only the owner
  can flip). Nothing mechanically prevents an unreviewed push to `main`.
- The **CoreML macOS workflow only triggers on branch `claude/coreml-export`**, not on
  `main`/PRs, and the **iOS XCTest parity job is disabled** pending `INTENTCLASSIFIER_PAT`
  (ND-7). So cross-platform parity — the thing that matters most for an on-device
  product — is **not gating merges**.

Net: the CI is honest about lint/format/unit/compiler correctness, but it currently
provides *false assurance* on model quality, safety, and device parity. Closing B2 is
what turns "green CI" into a signal you can actually ship on.

---

## 6. Component-by-component audit

### 6.1 Runtime engine (`packages/runtime/nlu_engine/`) — 🟢 strong
A 934-line orchestration engine with clear layering (confirmation context → interruption
→ guards → new-intent handling), config-driven thresholds, a startup guard that rejects
the placeholder `DEFAULT_GENAI_URL` (Appendix A #5 — verified by tests), and the safety
guard stack from §4. Well-structured, allocation-conscious hot path, importable leaf
modules for light tests. The engine is also positioned as the *executable spec* for the
native implementations via `runtime-contract-v1.md`. No concerns with the code itself —
the concern is the model behavior flowing through it (§4).

### 6.2 Bundle spec + compiler (`spec/bundle/3.0/`, `packages/buildtime/nlu_compiler/`) — 🟢 strong
16 JSON Schemas (draft 2020-12), a portable-regex subset with a normative corpus, two
golden bundles, and a shared validator library that is the single validation path for
CLI + tests + CI. Compiler stages 1–15 including deterministic packaging and dev-key
Ed25519 sign/verify with a 3-gate verifier. Validates clean; fails closed on tamper.
This is textbook "config over code, facts over behavior" (ADR-001.1). **Gap:** production
signing (ND-8/B3) — today only the dev key exists and production runtimes correctly
reject it, so no real release can be signed yet.

### 6.3 Training + evaluation (`packages/buildtime/nlu_training/`) — 🟢 solid, 🟡 gated in CI
One unified, schema-valid `evaluate` JSON (per-lang F1/acc/ECE, OOS recall, wrong-action
counts, per-domain breakdown, gate booleans). Recorded baseline:

| Lang | macro-F1 | holdout acc | ECE | Ship status |
|---|---|---|---|---|
| en | 0.896 | 0.907 | 0.018 | shippable quality |
| fr | 0.853 | 0.866 | 0.022 | shippable quality |
| de | 0.821 | 0.845 | 0.022 | above floor; 5 zero-data intents (ND-13) |
| da | 0.789 | 0.812 | 0.032 | **flag-gated — machine-translated, native holdout not met** |

Calibration is genuinely good (ECE ≤ 0.032 everywhere) — the temperature-scaling work
paid off. **OOS recall is 0.68**, i.e. ~1 in 3 out-of-scope utterances still leak into
an intent; this directly feeds the OOS→confident-action wrong actions in §4 and is the
highest-leverage quality fix. MLflow tracking exists but is a **local file backend only**.

### 6.4 Mobile / parity (ONNX / CoreML / iOS) — 🟡 built, not gating
Static batch-1 ONNX contract with temperature in metadata; CoreML FP16/FP32 export;
Tier-A numeric (Linux) + Tier-B runtime + ANE op-placement (macOS) tests; iOS XCTest
parity lives in the separate `akashrwt5/STT` repo. The methodology is right and the
acceptance bar (Δacc ≈ 0, 0 gate disagreements) is correct. **Gap:** none of it gates
`main`/PRs (§5), and the iOS handoff (ship `label_migration_map.json`, regenerate golden
fixtures) is an open owner action after the taxonomy migration.

### 6.5 Phase-3 lifecycle (OTA / telemetry / GenAI) — 🟡 client done, server absent
Two-slot `BundleManager` with downgrade protection and disaster fallback (10 tests),
telemetry events with no-raw-text-by-construction, and a draft key-rotation runbook. The
*client* half is done and tested. The *server* half (bundle registry, remote config,
telemetry ingestion, staged-rollout tripwires, dashboards) is entirely gated on the ND-12
hosting decision (B5). `assist.cloud` GenAI is a bounded, narration-only fallback that is
a no-op until configured, gated on ND-9 consent/legal (B4).

### 6.6 Repo hygiene / governance — 🟢 much improved
Restructure to `packages/apps/spec` complete with parity; `auto_label.py` quarantined;
README rewritten; CODEOWNERS present (placeholder handles); privacy stance for
`unknown_data` documented and enforced (counters by default). Residual: the duplicate
`scripts/SemanticSupport` vs `multilingual/SemanticSupport` trees still exist
(Appendix A #9, deferred); a couple of legacy tracked artifacts (`models/*.onnx`,
Danish) remain by deliberate exception.

---

## 7. Prioritized path to production

Ordered by what actually gates a ship. P0 = blocks production; P1 = required for a
trustworthy release; P2 = follow-on.

### P0 — must close before any production ship
1. **Meet the wrong-action budget (B1).** Two workstreams in parallel:
   - *Policy:* extend the confirmation gate to **unconditionally** confirm the
     highest-cost state-changing intents (volume mute, message send, session start/stop,
     find-phone) **regardless of confidence** — this is the only lever that catches the
     high-confidence residue. Re-measure. (Owner-approval gate per charter §7 — it
     changes UX friction; put it in front of the owner with the friction numbers.)
   - *Data/model:* targeted work on the three confusable clusters — `device.volume`
     polarity, OOS rejection (lift OOS recall from 0.68), and help-vs-action separation.
     Re-train, re-calibrate, re-replay. Define **per-domain budgets** so `device` (the
     hot domain) gets its own hard ceiling.
   - Wire the wrong-action replay as a **blocking CI gate** once B2 is fixed, so the
     budget can never silently regress again.
2. **Make the safety/quality gates actually run (B2).**
   - Stand up a **real shared DVC remote** (S3/GCS) and add `dvc pull` to CI.
   - Add `referencing` + `cryptography` (and pin `jsonschema>=4.18`) to the lockfile/deps.
   - Make the model-dependent tests **fail (not skip)** in the CI context that is
     supposed to enforce them, or run a dedicated nightly "full-artifact" job that is
     required before release tagging.
   - Enable **branch protection on `main`** (A#6).

### P1 — required for a trustworthy first release
3. **ND-8 — production signing keys / KMS** (B3). Recommendation already written
   (`open-decisions-brief.md`): Cloud KMS, non-exportable, signing gated on green eval +
   one human approval; rehearse rotation with dev keys first. Owner decision.
4. **Gate device parity on releases** (B6). Provision `INTENTCLASSIFIER_PAT`; make the
   CoreML/iOS parity jobs required for release (not just a feature branch). Complete the
   iOS fixture + `label_migration_map.json` handoff to the STT repo.
5. **Language ship-gating** (B7). Keep Danish behind its flag until the native-authored
   holdout passes; author real German data for the 5 zero-data intents (ND-13) or hold
   those intents behind availability routing. Do not let machine-translated coverage
   reach a shipped label.

### P2 — follow-on for the v1 platform (§8.3)
6. **ND-12 infra** (B5) → build the Phase-3 server half: bundle registry + remote config
   + telemetry ingestion, staged rollout with tripwires, per-`bundle_id`/per-domain
   dashboards. Prove a 5% cohort update + rollback **without an app release**.
7. **ND-9 GenAI consent/legal** (B4) → ship `assist.cloud` opt-in, off by default.
8. Consolidate the duplicate `SemanticSupport` trees; regenerate the multilingual
   semantic head with post-migration labels (currently stale/git-tracked); promote
   MLflow/DVC off local backends.

---

## 8. Bottom line

The team built a real platform, not a demo: a signed bundle format with a single-path
validator, a clean capability-partitioned content source, calibrated multilingual models,
a config-driven safety engine, and a device-parity methodology. Measured against the
Phase-1 platform gate, this is close to done and the work quality is high.

It is **not production-ready** for two reasons that both come down to *trust*: the
medical wrong-action budget is violated ~8× by high-confidence errors the current gates
cannot catch (B1), and the CI that would catch a regression currently **skips the very
tests that matter** because the data isn't reachable (B2). Fix those two — plus the owner
decisions on signing, consent, infra, and device-parity CI that are already engineered
up to the decision point — and this becomes a defensible production ship.

My recommendation: **treat B1 and B2 as a single hardening milestone, gate the release on
a green wrong-action replay in CI, and put the ND-8/9/12 owner decisions in front of the
owner now** so they resolve in parallel rather than on the critical path.

---

### Appendix — evidence index
- Test run: `pytest` → 90 passed / 60 skipped / 0 failed (skips all model-dependent).
- Compiler: `python -m nlu_compiler spec/examples/3.0/{minimal,full}` → 0 errors/0 warnings.
- Bundle build/sign: `tests/test_bundle_build.py` → 9/9.
- Safety: `tests/parity/oracle_post_migration/wrong_action_system_report.json`
  (`budget_met=false`, shipped-lang 41 vs 5).
- Quality: `tests/parity/oracle_post_migration/evaluate_report.json` (per-lang F1/acc/ECE,
  OOS recall 0.68, per-domain wrong actions device=113).
- CI: `.github/workflows/ci.yml` (no `dvc pull`), `coreml-macos.yml` (feature-branch only).
- Data: `.dvc/config` (`localstore` → `../../dvc-store`, local path).
- Open decisions: `docs/Review-F5/open-decisions-brief.md`, `EXECUTION_STATUS.md` (ND-7..13).
