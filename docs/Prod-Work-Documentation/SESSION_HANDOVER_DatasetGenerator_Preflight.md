# SESSION HANDOVER — Super Dataset Generator, Pre-flight Complete

**Date: 2026-08-20. Branch: `feature/Akash/semantic_with_new_csv-DataGeneration-Automation`, HEAD `c63f7f0`.**
**Everything this session produced is UNCOMMITTED in the working tree as of writing — see §6 step 0.**

Read the prior handover's rules first if you have it; they are restated in §1 because they are
binding. This document contains only facts verified in-session (files read, runs measured,
outputs inspected row by row). Anything not verified is explicitly marked as such.

---

## 1. Rules of engagement (from Akash — enforced in prior sessions)

1. **Ask before making changes.** Surface bugs as findings, not fixes-in-flight. Never
   fix-and-commit without asking. Akash approved every change listed in §3 individually.
2. **Present measurements as findings, then stop.** Do not chain a measurement into a
   remediation in the same turn without approval.
3. **Do not switch models, providers, prompts, or quotas to "improve" a result** without
   approval. Every paid run is a measurement; unilateral changes invalidate the lineage.
4. **Change one thing per paid run.** Two variables in one run = neither attributable.
5. **Never quote seed text anywhere** (chat, commits, configs, docs). Refer by intent and
   count. The seed corpus contains raw production ASR transcripts with PII; the repo is
   public. The PII decision is CLOSED — do not re-raise it.
6. **Akash commits.** You prepare. Do not run paid generation without his explicit go.
7. Akash writes in Hindi/Hinglish; answer in the language he uses.

## 2. System facts (verified against source this session)

- Taxonomy: 60 intents = 24 `Cmd.*` + 33 `Help*` + 3 other. All 60 specs live in
  `authored_specs.yaml` (hand-authored; provenance: 59 assistant-session + 1 human) and are
  merged by `bootstrap_specs.py` into `intent_specs.yaml`, which `generator.py._load_specs`
  reads at generation time. The `intent_specs.yaml` header's own note: requires human review
  before Stage 1. **That review (Akash's sign-off) has still not happened.**
- Stage 1 temperatures: top-level `llm.temperature: 0.1` is for SPEC AUTHORING;
  `generation.llm_overrides.temperature: 0.9` is what Stage 1 generation actually runs at
  (verified in `llm_client.py`, which layers `generation.llm_overrides` per stage). The
  prior handover stated this backwards.
- Quota profiles (`generation.quotas`): prefix-assigned. `Cmd.*` → `command`
  (min_short 0.28 @ ≤4 words — pre-existing, measured, untouched), `Help*` → `help`
  (new, see §3), `Default Fallback Intent` and `reminders.*` → `open` (no quotas, by design).
- Budgets: default 120/intent; overrides: Fallback 800, MemoryChange 300, reminders.add 200,
  VolumeIncrease/VolumeDecrease/EdgeModeIncrease 180, EdgeModeDecrease 160. Total 8,360 rows,
  ~348 calls. Dry run for one 120-intent plans 5 calls at batch 25.
- Deployed training data (`language_packs/en/train.csv`, 8,430 rows, 57 labels) is
  permutation-heavy (e.g. Cmd.MemoryChange: 1,601 rows, 339 distinct 3-word openings) and
  lacks the 3 EdgeMode intents and Help_Activity while still carrying
  Help_HearingCareAnywhereConnect — the runtime label delta is still outstanding.
- "Can-I" evidence from train.csv: utterances opening "can you" label as commands 839:251;
  "can I" labels 88 Help vs 10 Cmd, and never against a device-state command. This grounds
  the agent-based rule now in the Help_Volume spec.
- `Help_Volume` raw seed file: 122 unique lines; 2.5% ≤4 words (3 distinct phrasings),
  25.4% ≤7 words, mean 9.1 words. This grounds the help profile's short threshold.

## 3. What changed this session (all approved by Akash, all verified)

**Spec fixes** (in `authored_specs.yaml`, regenerated into `intent_specs.yaml` via
`bootstrap_specs.py`, verified present in the regenerated file):
1. Help_Volume boundary rewritten as an AGENT-based rule ("Can YOU make it louder" = command;
   "Can I make it louder" = Help; a "can I" whose outcome only the assistant delivers is
   still a command), with the train.csv evidence cited in the spec.
2. Aerobics↔Exercise tiebreak for bare "activity goal" queries, stated identically in both
   specs (bare "activity goal" → Aerobics; exercise/workout words → Exercise).
3. Cmd.VolumeMute power-off split: direct power-off request → Fallback; how-to → Help_Volume.
4. Help_Activity now names all eight activities (Stand and Calories were missing).
5. Help_Volume ↔ Help_SelfCheck mutual cross-reference (volume-framed symptom vs fault claim).
6. Fallback spec citation fixed: Section 6 (not 7) precedence rules.

**Code changes** (both compile; command-profile prompt verified byte-identical to all
previously paid runs):
7. `generator.py`: per-profile `short_max_words` (default 4) in the composition block;
   `_quota_profile` docstring updated.
8. `stage1_report.py`: compliance check reads the same per-profile threshold (was hardcoded ≤4).

**Config changes** (`generator_config.yaml`):
9. `help` profile: `min_short: 0.28`, `short_max_words: 7`, `types.Question: [0.88, null]`,
   `asr_simulated: [0.08, 0.16]` — every number from measurement (see §4).
10. Fallback seed-replacement block: environment-name trap bullet (descriptive mentions of
    restaurant/church/car/outdoors/meeting with no switch request), then a SCOPE GUARD
    listing the product's own capabilities with "never generate requests for these", then a
    tense clarifier (past-tense / other-person mentions only; first-person present-tense
    arrival statements belong to Cmd.MemoryChange).

## 4. Paid measurements this session (5 + 5 + 2 + 2 = 14 calls)

**Help_Volume baseline (120 rows, empty help profile), report `stage1_help_volume.md`:**
type 98.3% Question with zero "can you" rows (the feared command-shaped contamination of
Help intents did NOT occur); diversity gain +0.218; 0% near-dupes; saturation levelling off
at 120 (budget confirmed); BUT length badly skewed: mean 14.8 words, 68% at 13–20 words,
0% ≤4 words, vs real usage mean 8.5 and 4% at 13+.

**Help_Volume verify (120 rows, new help profile), report `stage1_help_volume_v2.md`:**
≤7-word share 1% → 25% (matches the seeds' 25.4%); mean 14.8 → 12.4; 100% Question (closed
the ObservationPlusCommand length-cap loophole — v1's only 2 non-Question rows were 22+
words); ASR 10% (in band); Hard difficulty 11% → 27.5%; rejections 27 → 3; near-dupes
0% → 1.7% (2 rows; Stage 2's job). Compliance "UNDER" at 30 short rows vs 34 asked is the
known floors-mostly-honoured behaviour. 13–20w tail remains 52% — ACCEPTED deliberately
(quotas are not meant to chase seed proportions; hard/long material is over-weighted by
design). **The help profile is verified and governs all 33 Help intents (47% of the corpus).**

**Fallback smoke v1 (50 rows, 2 batches), report `stage1_fallback_smoke.md`:**
the never-tested prose seed-replacement block WORKS mechanically (50/50 valid, difficulty
exactly on target). Environment traps present (~32% of rows). Found 2 mislabels: a
"remind me to…" row (reminders.add) and a "text Sarah that…" row (Cmd.SendMessage) — the
prompt never said which capabilities are in scope. Also 2 rows echoed the bullet's example
sentences near-verbatim. Both problems fixed in §3 item 10.
NOTE: the report's negative diversity gain (−0.013) for Fallback is a metric artifact —
Fallback seeds are already maximally diverse (0.969), so "gain" has no headroom. Expected;
do not "fix".

**Fallback smoke v2 (49 rows, 2 batches, after scope guard), report
`stage1_fallback_smoke_v2.md`:** all 49 rows inspected individually — zero in-scope
commands; every request-shaped row targets another product/assistant by name. Environment
traps ~29%, varied shapes, no example echo. 2 rows sat on the Cmd.MemoryChange carve-out
line (present-tense environment statements) → tense clarifier added afterwards (§3 item 10);
that clarifier is NOT yet verified by a paid run — Akash chose to verify it in the full
run's report and human sample instead of a third smoke.

## 5. Known open issues (verified, not yet fixed)

- `stage1_report.py` Rejections section ignores `--only` and reads the whole shared
  `rejections.jsonl`, so stale entries from other intents appear in scoped reports.
  Cosmetic; fix offered, not yet requested.
- 61 asymmetric `neighbor_intents` links. Agreed direction (prior session): treat the graph
  as undirected in Stage 3 code; do not hand-edit specs.
- Runtime label-map delta (add 3 EdgeMode + Help_Activity, remove
  Help_HearingCareAnywhereConnect) — outside this pipeline, blocks drop-in deployment.
- `reminders.add` seed file (medication reminders) not yet PII-screened.
- ASR-Simulated came back 0/49 in Fallback smoke v2 (3/50 in v1) — `open` profile has no
  ASR quota; check the share in the full run's report.
- Untracked junk in the working tree noted for housekeeping: `COMMIT_MSG.tmp`,
  `classifier_head.pkl`, `datasets/evaluation_errors.csv`, `.python-version`.

## 6. THE PLAN — in order, with gates

**Step 0 — commit the pre-flight (Akash runs; nothing is committed yet).**
`make format && make check` first (repo standard; the pre-commit darker hook has previously
reformatted files mid-commit — if git misbehaves in a sandbox, Akash commits from his Mac).

**Step 1 — Akash's spec sign-off (HIS action, the last blocker).**
The mechanical checks are done; he is confirming decisions, not hunting bugs. Items to
skim: the six §3 spec fixes, the Aerobics/Exercise tiebreak direction, and the
`Help_Volume` FAR rows noted in §4.

**Step 2 — full Stage 1 run (~348 calls, $6–12; needs Akash's explicit go).**
```
python generator.py --dry-run          # expect ~348 calls, 60 intents
python generator.py                    # resumes if interrupted; per-intent checkpoints
python stage1_report.py --markdown stage1_full.md
```
Fallback and Help_Volume already hold smoke/verify rows; decide with Akash whether to
`--force` those two for a clean lineage (recommended for Fallback: its 49 rows predate
nothing important) or keep and top up. **Do not change ANY config between now and this run.**

**Quality gates on `stage1_full.md`** (numbers from the verified runs above):
- Section 0: coverage 100% for all 60 intents; any intent short of budget → rerun resumes it.
- Section 1: diversity gain clearly positive for every intent EXCEPT Fallback (≈0 expected).
- Section 0b: help intents ≥ roughly 25% at ≤7 words, Question ≥88%; command intents meet
  their existing floors. "UNDER" by 10–15% of the ask = known behaviour, not failure.
- Section 2: near-dupes per intent ≤ ~2% (Stage 2 cleans the rest); vocab novelty ≥ 0.75.
- Section 4: FAR rows are review candidates for dev_hard, not deletions; NEAR is noise.
- Human sample: 30–50 rows chosen from the Hard slice + highest-FAR intents, checked for
  invented capabilities (no metric catches those) and, for Fallback specifically: zero
  in-scope commands, environment mentions past-tense/other-person only.

**Step 3 — Stage 2, dedup (not written).** Constraints already decided and recorded in the
config comments: WITHIN-INTENT ONLY (cross-intent similarity is Stage 3 signal, not noise);
judge model `sentence-transformers/all-mpnet-base-v2` (independent of the bge-small
distillation stack); `near_duplicate_threshold: 0.92` is an UNCALIBRATED placeholder —
calibrate against the 47 known exact within-intent duplicates from `seed_audit.py` before
trusting it, and record the separation achieved. Runs on Akash's Mac (HuggingFace is
blocked in sandboxes).

**Step 4 — Stage 3, hard negatives (not written).** Sources: each intent's
`neighbor_intents` treated as UNDIRECTED, `command_help_pairs` (all 24 pairs, the 0.822
boundary), and the command-vs-observation boundary. Include MINIMAL PAIRS for the
token-decided boundaries: can-you/can-I (volume), step/walk, phone/aids, recovery,
mute/minimum, less/off, tinnitus-noise/aid-volume, and environment-mention with/without
request (the MemoryChange carve-out). `hard_negatives_per_intent: 40` is the configured
budget. Design review with Akash BEFORE any paid Stage 3 call.

**Step 5 — validation harness + export (Phases 4–5, not written).** `validate_specs.py`
(re-check hand-edited specs without an API call), then export `super_dataset_rich.csv`,
`train.csv`, `dev_hard.csv` (dev_hard = Hard + compound slices, excluded from train).

**Step 6 — sealed Tier-2 holdout (can start NOW, in parallel; not blocked by anything).**
Must come from OUTSIDE the generator's prompt lineage: human-authored, or a different
model with a different prompt. `dev_hard.csv` is in-distribution and cannot serve. Report
Tier 1 and Tier 2 separately, never averaged — the gap is the generalisation signal.

**Step 7 — the decision experiment.** `EXPERIMENT_generated_vs_real.md` (train TF-IDF+LogReg
and MiniLM+head on generated data, test on the 68 real held-out seeds; decision rules
pre-written in that file). Akash deferred it deliberately — ASK before starting. It is the
cheapest direct answer to "is generated data actually better", and worth proposing again
once the full run lands.

## 7. When to STOP and come back to Akash

- Before ANY paid API call, config value change, spec edit, or commit — every time.
- Any batch failure streak, provider/model anomaly, or temperature guard trip.
- Any finding that suggests changing quotas, prompts, budgets, or the taxonomy — present
  the measurement and wait.
- Anything touching the Fallback seed file, `reminders.add` seeds, or PII in any form.
- If a report number contradicts this document, trust the report, say so, and stop.

## 8. When to come back to the previous session's agent (or reproduce its checks)

The analysis pattern used throughout: stage the report + the per-intent `.jsonl` +
`rejections.jsonl`, compute length/type/opening distributions, read every Fallback row
individually, and compare against the real corpus (`train.csv` / seed files, counts only,
never quoting). Reproduce that on `stage1_full.md` before declaring the run good.
