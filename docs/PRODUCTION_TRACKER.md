# Production readiness tracker

**Verdict: not production-ready.** English is close on the *engine* axis and far
on the *coverage* axis; the other three shipped languages do not exist in this
tree at all.

Reviewed against what an on-device intent system for a **medical-adjacent
wearable** has to clear — not against a general chatbot bar. Two things make
this domain stricter than Alexa/Siri: a wrong action changes the state of a
device on someone's ear, and the user is hearing-impaired, so every recovery
turn is more expensive than usual.

Measured on this tree at commit `b1a7bdd3`. Figures are English, semantic rescue
off (it does not load — see B2).

---

## Scorecard

| Area | State | Note |
|---|---|---|
| Decision ladder | 🟢 | One threshold, no invented confidences. Just rebuilt. |
| Calibration methodology | 🟢 | OOF, eval sets excluded, provenance, ECE 0.0183. Genuinely good. |
| Leakage discipline | 🟢 | Guarded holdout, leakage mask in every fitter. |
| Bundle format + signing | 🟢 | Spec'd, validated, signed, verified, codegen for both clients. |
| English accuracy | 🟡 | 90.4% acc / 0.896 macro-F1. Head intents fine, `help.*` weak. |
| Out-of-scope rejection | 🔴 | **9.7% of OOS utterances fire a device action.** |
| Wrong-action rate | 🔴 | 36–39 on 1470 turns (~2.5%). No per-intent cost model. |
| Multilingual (fr/de/da) | 🔴 | **Packs empty. de/da have no model. fr has no calibration.** |
| Semantic rescue stage | 🔴 | Dead. Bundle advertises an embedder it does not ship. |
| ASR robustness | 🔴 | **No noise/misrecognition testing of any kind.** |
| On-device budgets | 🔴 | No measured latency, memory or battery numbers in the repo. |
| Build/CI health | 🔴 | `make check` broken (BUG-009). 14 open bugs, 6 High. |
| Observability | 🟡 | Telemetry schema exists; no dashboards, alerts or rollback trigger. |
| Privacy | 🟢 | Raw-utterance logging opt-in and off by default; no text in results. |

---

## Blockers — must close before any production rollout

### B1 — Out-of-scope utterances fire real actions

195 OOS turns on the honest holdout. **19 (9.7%) reach a device action.**

```
'turn off toshiba'                    -> device.volume.mute
'amazon screen is dark'               -> help.home.show
'when does fall start'                -> help.fall_alert.show
'help me find a paper'                -> help.find_my_hearing_aids.show
```

Report-card OOS recall is **0.708**. For a device that mutes or changes a
hearing program, ~1 in 10 unrelated sentences causing an action is the single
most user-visible safety gap. "turn off toshiba" muting a hearing aid is the
example to put in front of the owner.

The confirmation band used to absorb some of these. It was removed for good
reasons (see `confirm-gate-diagnosis.md`), which makes fixing rejection properly
the replacement, not an optional follow-up.

**Exit:** OOS recall ≥ 0.90 and OOS→action ≤ 1% on a frozen holdout.

### B2 — The semantic rescue stage does not exist at runtime

`NLUEngine.semantic is None` — the MiniLM artifacts do not load. Meanwhile
`bundle.json` declares an `embedder_id` and ships no embedder or vocab
(BUG-013, BUG-014, both High).

Consequences that compound: every rejection figure quoted anywhere in this repo
is a **worst case**; the fire threshold cannot be fitted honestly; and the one
component that would fix B1 — a head with a *learned* out-of-scope class rather
than a threshold — is the one that is switched off.

**Exit:** stage loads and is measured, or is removed from the bundle contract
and the docs. Not left half-declared.

### B3 — Only English exists

```
language_packs/{fr,de,da}/   EMPTY
models/intent/de, da/        EMPTY
models/intent/fr/            model + labels, NO calibration
```

`CLAUDE.md` and the capability manifests declare en/fr/de/da. A French pack with
no `calibration.json` runs at T=1.0, so its confidences are uncalibrated and the
0.70 threshold means something different there than in English.

**Exit:** each shipped language has a pack, a fitted calibration, its own
holdout, and its own report card — or is removed from the declared set.

### B4 — Build and evaluation integrity

- **BUG-009 (High)** — 4 `make` targets point at a deleted tree; `make check` is
  broken. A green CI that does not run the gate is worse than a red one.
- **BUG-012 (High)** — two eval fixtures sit on the dead `Cmd.*` taxonomy and
  score **0% silently**.
- **New, found today** — `make calibrate` with an incomplete model set
  **overwrites `config/calibration.json` and drops every language's values**,
  leaving only `_deprecated`. Tracked file, silent loss.

**Exit:** `make check` green from clean checkout; no evaluation artifact can
score 0% without failing.

---

## Major gaps — not blockers, but production debt

### G1 — The model is the ceiling

TF-IDF + LogisticRegression over 57 classes. Per-intent recall on the holdout:

```
help.home.show                    47.4%   (9/19)
help.battery.show                 50.0%   (4/8)
help.clean_care.show              60.0%   (6/10)
help.transcribe.show              66.7%   (8/12)
reminders.task.complete           72.7%   (8/11)
```

These are semantically separable and hard only for bag-of-words. The system
compensates with 28 regexes, which is why the confidence path grew a second
scale in the first place. The strongest recognizer in the repo (MiniLM, already
CoreML-exported) is relegated to a disabled fallback — the cascade is inverted.

Fixing B2 is the prerequisite for even measuring the alternative.

### G2 — Wrong actions are budgeted globally, not by cost

36–39 wrong actions per 1470 turns. `device` domain accounts for 22 of 39.

There is one global threshold for an intent set that spans `volume.increase`
(reversible in one word) and `message.send` (irreversible, externally visible).
A single number cannot serve both. The cost model is designed
(`fit_decision_ladder.py` takes explicit costs) but the ratios are unset,
because they are a product and clinical judgement.

**Needed from the owner:** the relative cost of a wrong action, a confirmation,
and a rejection. Everything downstream is arithmetic once that exists.

### G3 — No ASR robustness testing

This is a **speech** system and there is not one test covering
misrecognition, truncation, disfluency, accent-driven substitution or noise.
The holdout is clean text. Real input is ASR output.

`'can you to us number one hits'` is in the holdout — clearly an ASR mangling —
so the data has some, incidentally, not systematically.

**Needed:** a perturbation harness (phonetic substitution, word drop, filler
insertion) and a separate accuracy gate on it.

### G4 — No on-device budgets

Nothing in the repo records latency, memory or battery on target hardware. What
exists is desktop ONNX (p50 0.098 ms) and CoreML parity tests. Bundle is 2.7 MB,
of which **56% is never read by any mobile client** (BUG-019), and a Python
pickle is shipped to mobile (BUG-020).

**Needed:** measured p50/p95/p99 and peak RSS on the actual device, with a
declared budget to regress against.

### G5 — One unfitted constant remains

`CONTESTED_CONFIDENCE = 0.60`. Provisional, documented as such, and
`fit_decision_ladder.py` exists to fit it. Fit it before shipping.

### G6 — Rollback and kill-switches unproven

`NLU_SEMANTIC_RESCUE` and `NLU_LEGACY_LABELS` exist. There is no documented
procedure for: bad bundle in the field, staged rollout, per-cohort holdback, or
what metric triggers a rollback. Bundles are signed and versioned — the
machinery is there, the operational contract is not.

---

## What is already strong

Worth saying plainly, because the list above is long:

- **Calibration is done properly.** Out-of-fold, evaluation sets excluded,
  provenance recorded, server/device temperatures deliberately separated. Better
  than most production systems.
- **Leakage is taken seriously.** A guarded holdout, a leakage mask inside every
  fitter, and an explicit refusal to tune against the honest holdout (B9).
- **The bundle format is real engineering.** Versioned, schema-validated,
  signed, verified, with Swift/Kotlin codegen and golden examples as the
  cross-team integration test.
- **Failure modes are documented rather than hidden.** The bug tracker, the ADRs
  and the `_note` fields carry the reasoning, including the mistakes.
- **The decision path is now single-scale**, with tests that fail if a constant
  reappears in the confidence field.

---

## Suggested gate before rollout

Ship nothing to real users until all of these hold on a frozen holdout:

| Gate | Target | Today |
|---|---|---|
| OOS → action | ≤ 1% | **9.7%** ❌ |
| OOS recall | ≥ 0.90 | **0.708** ❌ |
| Wrong actions | ≤ 1% of turns | ~2.5% ❌ |
| Head-command accuracy | ≥ 99% | ✅ (test exists) |
| Per-language calibration | all shipped langs | **en only** ❌ |
| `make check` from clean checkout | green | **broken** ❌ |
| Device p95 latency | declared budget | **unmeasured** ❌ |
| ASR-perturbation accuracy | declared budget | **no harness** ❌ |
| Rollback drill | performed | **not done** ❌ |

---

## Order I would work in

1. **B4** — fix the build and the silent-0% fixtures. Everything else is
   measured through this, so it comes first.
2. **B2** — revive or remove the semantic stage. It gates honest rejection
   numbers and it is the most likely fix for B1.
3. **B1** — out-of-scope rejection. Highest user-visible risk.
4. **G2** — get the cost ratios from product, then fit the threshold per intent.
5. **G3** — ASR perturbation harness.
6. **B3** — decide: build the other three languages properly, or drop them from
   the declared set. Do not ship a language with an uncalibrated model.
7. **G4/G6** — device budgets and the rollback drill.
8. **G1** — revisit the cascade once B2 makes the comparison possible.

---

## What I could not assess

Stated so this document is not read as more complete than it is:

- **Real device performance.** No target hardware here; all latency figures are
  desktop ONNX.
- **The iOS and Android runtimes.** Separate repos. The bundle contract is
  reviewable from here; their implementations are not. Worth checking whether
  they replicate the `KEYWORD_CONFIDENCE` bug — the bundle ships `tier`, and
  `keywords.schema.json` says tier 2 requires classifier agreement, which the
  Python engine ignored until this week.
- **Real user traffic.** Every figure is a curated holdout. The gap between
  holdout and live speech is exactly what G3 exists to shrink, and it will still
  be a gap.
- **Clinical/regulatory posture.** Whether this device class carries obligations
  that change the wrong-action bar is not an engineering question and is not
  answered here.
