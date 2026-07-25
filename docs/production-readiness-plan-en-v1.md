# Production Readiness Plan — English v1

**Target branch:** `claude/claude-setup-architecture-ebqobs`
**Source of borrowed assets:** `feature/production-work` @ `93b18886`
**Date:** 2026-07-25
**Scope:** ship English. Preserve the property that adding a language is
**authoring `packs/<lang>/` + training data — nothing else.**

---

## 0. The one-paragraph summary

`feature/production-work` is architecturally ahead of us and verification-wise
behind us. It has built — and tested — the exact safety machinery we need
(uncertainty confirmation, polarity guards, help-marker guard, a system-level
wrong-action harness), but **62 of its 127 tests skip in CI** because the model
they need is gitignored and CI never trains, and its own committed report shows
**41 wrong actions against a budget of 5**. We have the better *language*
architecture (self-contained packs vs. merge-overlays) and a suite that actually
runs. The plan is therefore: **borrow their safety behaviours, re-home them in
our packs, and never inherit their verification gap.**

---

## 1. What we borrow, and what we deliberately do not

### 1.1 Borrow — behaviours (high value, low risk)

| Asset | What it is | Why we want it |
|---|---|---|
| **Uncertainty confirmation** (ND-11a) | 11 state-changing intents gated below a confidence floor; ask instead of firing | Turns a silent wrong action into a question. The single biggest safety win available. |
| **Per-intent confirm prompts** | `"Just to be sure — mute your hearing aids?"` | Far better UX than a generic prompt; costs nothing but content. |
| **Polarity guards** (ND-11b) | Regex → `blocked_intent` → `redirect_intent` triples | Directly attacks our hardest failure class: up/down, mute/unmute minimal pairs (0.97 cosine). |
| **Help-marker guard** (ND-14) | `markers` regex + `pairs` map (action intent → read-only `help.*` sibling) | Kills 4 of our 6 confident wrong actions. This is the fix for "how do I change memories" firing a memory change. |
| **Wrong-action harness** | System-level replay measuring the *whole* defence stack | We currently measure the classifier. This measures what the user experiences. |

### 1.2 Borrow — engineering hygiene

| Asset | Why |
|---|---|
| `pyproject.toml` + `requirements.lock` | We have neither. Today our code is not installable and deps are unpinned — flagged in the Phase-0 review. |
| Ruff-enforced CI + darker format-on-touch | Lets us enforce style on new code without reformatting history. |
| Their **counting rules** for wrong actions | `FULFILL`≠truth = charged. `CONFIRM`≠truth = *gated*, reported separately, **not** charged. `PROMPT`/`FALLBACK` = no action. This is the right accounting. |

### 1.3 Do NOT borrow

| Asset | Why not |
|---|---|
| `content/` + `content/localization/*.json` **overlay merge** model | **This is the important one.** Their per-language config is a *merge* of a base schema plus overlays. Ours is a *replacement*: each pack is self-contained. Merge semantics mean adding a language can silently inherit or fight English defaults. Our model is strictly better for the stated goal — keep it. |
| The `.nlu` bundle compiler + `spec/` | Excellent, but it is Phase-3 release infrastructure. Not needed to ship English v1. |
| DVC + MLflow | Same. Valuable at scale; overhead now. |
| The 57-intent `domain.object.action` migration | A breaking change to our label space, our packs, and our trained model. Do it later, deliberately, or not at all. |
| Their CI shape | `pytest -ra` with no training step is precisely the defect that makes their safety tests vacuous. |

---

## 2. The architectural principle we must not break

Everything borrowed lands as **pack data**, never as engine logic.

```
BORROWED BEHAVIOUR          WHERE IT LIVES ON OUR BRANCH
uncertainty confirmation →  packs/<lang>/config.json  (policy block)
per-intent prompts       →  packs/<lang>/schema.json  (intents[].confirm_prompt)
polarity guards          →  packs/<lang>/schema.json  (polarity_guards[])
help-marker guard        →  packs/<lang>/schema.json  (help_marker_guard{})
risk tiers               →  action-ID prefixes (already how non_interrupting_actions works)
```

The engine gains **generic mechanisms**; every word, pattern and threshold comes
from the pack. `scripts/ci/check_language_neutral.py` already fails the build on
a hardcoded English phrase in `scripts/nlu/` — it will catch any regression here
automatically.

**Acceptance test for the whole plan:** a hostile `zz` pack must exercise
confirmation, polarity guards and the help-marker guard **with zero engine
edits**. If it can't, we've built it wrong.

---

## 3. Phased plan

Each phase has an explicit gate. Do not start a phase until the previous gate is
green.

### Phase A — Make measurement trustworthy *(prerequisite for everything)*

Nothing below can be evaluated until we can measure the whole system.

| # | Task | Detail |
|---|---|---|
| A1 | **Merge the temperature-scaling branch** | `claude/…-Temperaturescaling-fixes` — out-of-fold calibration (T 0.796→0.648, ECE 0.0817→0.0349), leakage guard, provenance. Delivered 280→292/341 at equal safety. |
| A2 | **Port the wrong-action harness** | System-level replay with their counting rules. This becomes the release metric, replacing classifier accuracy. |
| A3 | **Fix `train.py`'s leakage guard** | It compares raw strings, so 3 rows differing by a trailing `?` leak into training. Normalise (case + punctuation), then remove the 3 rows from `01_source_base_training_data.csv`. |
| A4 | **Add `pyproject.toml` + lockfile** | Make the package installable and deploys reproducible. |

**Gate A:** the harness produces a wrong-action number for English, end-to-end,
reproducibly, on a model trained from clean data.

---

### Phase B — Safety mechanisms *(the core of v1)*

| # | Task | Pack data added |
|---|---|---|
| B1 | **Help-marker guard** | `schema.json: help_marker_guard {markers, pairs}` — maps each state-changing intent to its read-only `help.*` sibling. Redirects rather than fires. |
| B2 | **Risk-tiered confirmation** | `config.json: policy.confirm` — `{act_threshold, confirm_floor}` per action-prefix tier. Below `act`, ask; below `confirm_floor`, defer. |
| B3 | **Per-intent confirm prompts** | `schema.json: intents[].confirm_prompt` |
| B4 | **Polarity guards** | `schema.json: polarity_guards[]` — English needs few (keyword rules cover most); the mechanism must exist before `packs/fr/`. |
| B5 | **Return top-2 from `classify()`** | Prerequisite for disambiguation. Currently the runner-up is computed and discarded. |
| B6 | **Reply resolution** | Order: match an offered option → yes/no lexicon → uncertain → new command → timeout. Measured: 9/10 natural replies (`"quieter"`, `"softer"`, `"less"`) self-resolve by re-classification with **no new vocabulary**. |

**Design note on B2.** Their design uses one global floor (`below_confidence:
0.80`). Our measurement says add a **lower** bound too: below ~0.50 the top-1 is
right only 48% and even top-2 misses 16% — asking there wastes a turn. Between
0.50 and the act threshold, the right answer is in the top two **97–100%** of the
time, which is exactly where asking pays.

**Design note on B6.** Prefer **"Louder or quieter?"** over **"Turn it up?"**. In
the 0.50–0.75 band a yes/no confirmation dead-ends ~30% of the time; a two-option
question resolves ~97% in one turn. This is the piece `production-work` does not
have, and our data says it matters.

**Gate B:** wrong actions on the English harness **≤ 2** (from 5), with no
regression in delivered accuracy, and the hostile `zz` pack exercising all four
guards with zero engine edits.

---

### Phase C — Ship-readiness

| # | Task |
|---|---|
| C1 | **CI that trains before it tests** — and fails on skip when `CI=true`. Never inherit their gap. |
| C2 | **Fix the session-store leak** — unbounded growth retaining `{'name': 'take my heart medication'}` in RAM. Memory *and* privacy. |
| C3 | **Resolve the GenAI endpoint** — `genai_url` is still the placeholder `https://genai.yourcompany.com/chat?query=`, with no guard. Configure it or make an absent value a hard startup error. |
| C4 | **Golden conversation corpus** — multi-turn replay fixtures, the spec for the eventual Swift/Kotlin port. |
| C5 | **Pack release metadata** — `pack_id` is still `en@0.1.0-skeleton`, `channel: dev`. |

**Gate C:** CI trains, runs everything, and fails on skip. No placeholder
endpoints. Bounded memory.

---

### Phase D — Data quality *(parallel with B/C)*

From `docs/dataset-audit-2026-07-25.md`. Three criticals:

- **ISS-01** — `Cmd.MemoryChange` outweighs `Help_MemoryOptions` **19:1**.
  Measured: capping to 200 gives **+1.9 points** and cuts wrong actions ~28→22.
- **ISS-02** — **43%** of `Help_*` data is imperative-framed, destroying the one
  signal separating a question from a command. This is *why* a rules-only
  help-marker guard catches only 1 of 6 on its own.
- **ISS-03** — no laterality entity, while **253 utterances** say left/right ear.
  A missing capability, not a quality issue.

**Note the interaction:** B1 (help-marker guard) and ISS-02 (re-annotation) attack
the same failure from opposite ends. Do both; neither alone is sufficient.

---

## 4. Definition of done — English v1

| Criterion | Target | Today |
|---|---|---|
| Wrong actions (system harness, English) | **≤ 2** | 5 (classifier-level) |
| Delivered accuracy, holdout | ≥ 85% | 82.1% (92 after A1) |
| ECE | ≤ 0.05 | 0.0817 → 0.0349 after A1 |
| Tests skipped in CI | **0** | 0 (keep it that way) |
| Placeholder endpoints shipped | 0 | 1 (`genai_url`) |
| Unbounded memory growth | none | present |
| Package installable | yes | no |
| Hostile `zz` pack exercises every guard | yes | n/a |

---

## 5. The "add a new language" contract

The plan is only successful if this stays true. After Phase B, adding French is:

1. Author `packs/fr/` — `pack.json`, `config.json` (thresholds, policy, confirm
   tiers), `schema.json` (intents, keyword triggers, **polarity guards**,
   **help-marker pairs**, confirm prompts), `lexicons.json` (incl. `negations`,
   `affirmative`, `negative`), `datetime/grammar.json`, `entities/enums.json`.
2. Add French training data; train; `fit_calibration.py --lang fr --write`.
3. Run the harness for `fr`. Ship when it clears the same gate English did.

**Zero edits to `scripts/nlu/`.** CI enforces it.

The two French clock-idiom tests currently deselected (`et demie`,
`moins le quart`) are the **ready-made acceptance test**: if authoring `packs/fr/`
makes them pass without touching the engine, the contract is proven. If it
doesn't, we've found the real limit — cheaper to discover with one language than
after three are built on the assumption.

---

## 6. Sequencing and risk

```
A1 ─┬─> A2 ──> [Gate A] ──> B1..B6 ──> [Gate B] ──> C1..C5 ──> [Gate C] ──> ship EN
A3 ─┤                            ▲
A4 ─┘                    D1..D3 ─┘  (parallel; B and D reinforce each other)
```

**Biggest risk: measuring the wrong thing.** We have spent real effort tuning
scalars against a 341-row holdout where ±3 is noise. Phase A exists to stop that.
Once the harness runs, **the wrong-action count is the number that gates release**
— not accuracy.

**Second risk: scope creep from `production-work`.** The bundle compiler, DVC and
MLflow are genuinely good and genuinely not needed to ship English. Resist.

**Third risk: the native port.** None of this addresses that iOS/Android run
Swift/Kotlin, and only the *classifier* has cross-platform conformance today. The
dialogue layer — confirmation, slots, datetime — has no parity harness on either
branch. C4 (golden conversation corpus) is the first step, and it should be
written **before** the port, as its specification.

---

## 7. What we are explicitly not doing in v1

- The 57-intent taxonomy migration
- The `.nlu` bundle compiler, signing, staged rollout
- DVC / MLflow
- French, German, Danish (mechanism ready; packs not authored)
- Replacing TF-IDF with embeddings — its ceiling is a property of the
  representation, but that is a v2 conversation
- Semantic rescue on by default — it is worth +12.6 points at no wrong-action
  cost, but the on-device memory question is unanswered

---

## Appendix — measurements this plan rests on

All from this branch unless stated.

| Measurement | Value |
|---|---|
| Classifier top-1 (holdout) | 312/341 = 91.5% |
| Engine delivered (semantic off) | 280/341 = 82.1% |
| Correct answers discarded by the gate | 35 (10.3% of traffic) |
| Confident wrong actions | 6 (5 mutating) |
| ...of which the semantic head already answers correctly | **5 of 6** |
| Right answer in top-2, confidence 0.50–0.90 | **97–100%** |
| Natural replies self-resolving by re-classification | 9/10 |
| ECE at shipped T=0.796 | 0.0817 |
| ECE at out-of-fold T=0.648 | 0.0349 |
| `production-work` tests skipped in CI | **62 of 127** |
| `production-work` wrong actions vs budget | **41 vs 5** |

**Caveats.** The 341-row holdout is small — treat any difference under ~3
utterances as noise. `production-work`'s 41 is *their* committed number, not one I
reproduced; mitigations landed after that report may have moved it, and it should
be regenerated before being used to make a decision.
