# NLU Dataset Architecture: The "Super Dataset" Strategy (v3)

This document outlines the architectural blueprint for upgrading the legacy Dialogflow dataset into a production-grade embedding dataset for our 10MB Edge Semantic Model.

> [!NOTE]
> **v3 changelog — what moved, and why this revision exists.**
> v2 was headed "Final" and the implementation then moved without it. Every change below was verified against the code and data on 2026-08-23; none of it is a plan, it is a correction.
>
> | § | Change |
> |---|---|
> | 2 | Resolved taxonomy corrected **57 → 60**, with the runtime-label-map delta stated |
> | 2 | Spec authoring is no longer LLM-bootstrapped; `hand_authored_specs.yaml` → `authored_specs.yaml`, all 60 authored |
> | 5 | Stages 2 and 3 marked **not implemented**; their config keys are dead |
> | 7 | `Intent Family` recorded as a prompt input, not a row field |
> | 8 | Tier 1 renamed `dev_synth_hard.csv` — the name `dev_hard.csv` is taken by a different instrument |
> | 8 | New: the taxonomy and the evaluation instruments do not yet agree |
> | 2 | `reminders.add` screened and cleared — it is not a blocked file, and never was |
>
> A document describing a system nobody built is worse than no document, because it is read as a status report. Where something is unbuilt this revision says so in the section that specifies it, rather than in a backlog somewhere else.

## 1. Core Shift: From Permutations to Semantic Diversity
The existing seed data relies heavily on syntactic permutations. The new strategy targets **semantic coverage** over raw phrase count.

### Linguistic Diversity (Beyond Slang)
The LLM generator will synthesize diverse linguistic variations to mimic real-world chaos:
- Polite and indirect requests
- Short, abrupt commands
- Elderly speech patterns
- ASR (Speech Recognition) transcription errors
- Partial sentences and filled pauses (e.g., *"uh", "hmm"*)
- **Compound / Mixed-Signal Utterances:** (e.g., *"It's quieter today, make it louder."*)

## 2. Intent Specification (New)
Before generation begins, **every intent** must have a formal specification. This removes reliance on the LLM inferring business boundaries, and ensures labeling stays consistent across multiple generation runs.

### Template
```text
Intent Name:
Cmd.VolumeIncrease

Business Description:
Increase the hearing aid volume.

Trigger Conditions:
- User requests louder hearing.
- User asks to increase hearing aid volume.

Do Not Trigger:
- Pure observations.
- Questions asking for information only.
- TV or phone volume requests.
- Streaming-related requests.

Boundary / Uncertain Cases:
- If an utterance is ambiguous between this intent and Fallback, default to Fallback.
- Ambiguity must never be resolved by inventing a business capability not listed in "Trigger Conditions."

Neighbor Intents:
- Cmd.VolumeDecrease
- Cmd.StreamingStart
- Cmd.StreamingStop

Positive Example:
"Make it louder."

Hard Negative Example:
"The volume sounds lower today."
```
Every intent in the taxonomy must have one of these specs authored **before** it is passed to the generator. The specs, not the generator's inference, are the source of truth for labeling.

### Taxonomy Resolution (Prerequisite to Spec Authoring)
Specs can only be authored once we know what the taxonomy *is*. An audit of the legacy export (`seed_audit.py`, report: `seed_audit_report.md`) found that the 63 seed files are not 63 intents, and that several pairs are not separable at the seed level at all. The rules live in `generator_config.yaml` under `taxonomy:` so they stay auditable rather than buried in code.

| Rule | Files | Rationale |
|---|---|---|
| Exclude | `_EntityMemory`, `_EntityMemoryFromUsersays`, `_EntityRecurrence`, `_EntityRemind` | Dialogflow **entity value lists**, not intents. Contents are bare slot values (`Normal`, `Restaurant`, `Outdoors`). Their contents survive as `taxonomy.slot_vocabularies`, which the generator injects as slot fillers rather than as labels. |
| Exclude | `_ConfirmYes`, `_ConfirmNo` | Dialog-flow control turns, not user intents. They belong to a confirmation state machine the edge classifier does not own. |
| Merge | `Help.Activity` → `Help_Activity` | Dot-variant from an older export; 25 of its 29 unique phrases already appear in the target. Phrases are unioned. |
| Drop | `Cmd.Health` | Rollup **parent**, not a sibling: 155 of its 160 unique utterances are drawn verbatim from the `Cmd.Activity*` children. Five utterances exist only here and are dropped with it; they are listed in `drop_intents` so the loss is recoverable. |
| Drop | `Help_HearingCareAnywhereConnect` | Disabled in Dialogflow, and a 100% subset of `Help_RemoteProgramming` — zero distinguishing utterances. |
| Add source | `Cmd.MemoryChange` workbook | A second seed source (`additional_seed_sources`) with templates collapsed, because the Dialogflow export alone under-represents the product's highest-volume command. |

**Resolved taxonomy: 60 intents.**

> **Corrected in v3.** v2 recorded 57 and that number was wrong by the time it was written. It is worth being precise about *which* 57, because two different sets of that size are in play and confusing them is how the error survived: the **shipping runtime label map** (`packages/runtime/nlu_engine/legacy_label_map.json`) holds 57, and the **derived taxonomy** holds 60. They are not the same 57 — the derived set both adds and removes.

### Runtime Label-Map Delta

`seed_audit.py` §4b reports this difference on every run. Each item is a decision already taken, none of them yet reflected in the app.

| Direction | Intent | Status |
|---|---|---|
| ADD | `Cmd.EdgeModeIncrease`, `Cmd.EdgeModeDecrease`, `Cmd.EdgeModeDeactivate` | Genuinely new capability. 193 seed utterances exist, 0 rows in shipping training data, no runtime label. |
| ADD | `Help_Activity` | Production folded this into `Help_Health`. Kept separate here: configuring a goal and locating a screen are different tasks. |
| REMOVE | `Help_HearingCareAnywhereConnect` | Deprecated above, but still shipped with 55 rows. Finish the deprecation. |

57 + 4 − 1 = 60.

> **Until this delta lands, a model trained on the 60 is not drop-in compatible with the current app.** This is a release blocker, not a data-cleaning chore, and it is recorded here rather than only in code comments because it changes what "done" means for the whole pipeline.

### The Taxonomy and the Evaluation Instruments Do Not Yet Agree

This is the consequence of the delta that is easiest to miss, because it degrades a *measurement* rather than breaking a build.

Every English evaluation instrument (`INSTRUMENTS.md`) was built on the 57-intent runtime taxonomy. The primary decision instrument, `dev_hard.csv`, is frozen for P2–P8 by design — a ruler that moves mid-plan retires every number measured on it. But measured against the 60:

- `dev_hard` contains **zero rows** for `Cmd.EdgeModeIncrease`, `Cmd.EdgeModeDecrease`, `Cmd.EdgeModeDeactivate` and `Help_Activity`. Four of the sixty intents cannot be scored at all.
- `dev_hard` contains **7 rows labelled `Help_HearingCareAnywhereConnect`**, an intent this taxonomy drops. A model that correctly follows the taxonomy is marked wrong on every one of them.

So the frozen ruler carries a small, known, one-directional bias against the very change the Super Dataset exists to make. It is small — 7 rows in 813 is under a point, well inside the instrument's 0.038 MDE — but it is a bias rather than noise, and an unrecorded bias is how a confident wrong number gets shipped.

**Required before the Super Dataset is scored:** decide explicitly between excluding the 7 rows from scoring (and re-stating the P1 baseline of 0.8327 alongside the excluded figure) or accepting the penalty as recorded. Either is defensible. Choosing neither is not.

> **Why this is an architectural concern, not a data-cleaning chore.** Two intents whose seed utterances are identical do not present a *hard* classification problem; they present an **unlabelable** one. Generating synthetic data for both teaches the model to split an inseparable region of the embedding space, which surfaces as unstable confidence — and unstable confidence on command intents is exactly what inflates False-Accept Rate (Section 8). No amount of downstream generation quality repairs a taxonomy defect, so collision detection runs *before* Stage 0.

Any intent added to the taxonomy in future must clear the same bar: it needs at least one utterance that no other intent claims.

### Data Handling Constraint: Fallback Seeds Contain Customer PII
The `Default Fallback Intent` seed file is not a curated OOS list — it is 613 raw production ASR transcripts, including speech captured ambiently around the wearer. It contains a street address, a phone number, benefits and health details, personal names, and fragments of private conversation.

#### `reminders.add`: screened and cleared (v3)

An earlier note recorded `reminders.add` as "similarly contains medication reminders" and treated it as a second blocked file. **That was never verified, and it does not hold.** The claim appears to have come from the intent's *purpose* — in production a user's reminders may well be medical — rather than from the *seed file's contents*, which are authored test phrases.

Screened before the decision, reporting counts only and quoting nothing:

| | `reminders.add` |
|---|---|
| Lines / unique | 508 / 255 (50% duplicate) |
| Mean / max words | 3.4 / 14 |
| Phone-like digit runs, emails, addresses, postcodes | **0** |
| Dates, clock times, dosages | **0** |
| Medication or clinical vocabulary | **0** |
| Capitalised tokens that could be personal names | **0** |

Its associated slot vocabulary is equally thin: `_EntityRemind` is 13 lines, 7 unique, 1.5 words on average, no digits and no capitalised tokens. The `slot_vocabularies` entry itself is two inline values already committed to this public repository.

The shape is the decisive evidence, not the keyword screen. Set beside other authored intent files, `reminders.add` is unremarkable:

| File | Lines | Unique | Dup | Mean words |
|---|---:|---:|---:|---:|
| `reminders.add` | 508 | 255 | 50% | 3.4 |
| `Cmd.EdgeModeIncrease` | 300 | 153 | 49% | 3.0 |
| `Help_Pairing` | 285 | 138 | 52% | 4.6 |
| `Cmd.VolumeIncrease` | 141 | 69 | 51% | 2.7 |

A 50% duplicate rate at 3.4 words is the signature of template permutations. Raw ASR capture does not look like this — it is long, near-unique per line, and structurally messy, which is exactly how the Fallback file reads and why that one is blocked.

> **Decision (Akash, 2026-08-23): `reminders.add` seeds go to the API like any other intent.** It is not added to `privacy.no_seed_block_intents`, and it needs no `seed_block_replacement`.
>
> What the screen establishes and what it does not: pattern matching cannot prove absence, and a clean screen is not a guarantee. What carries the decision is that this file has the structure of authored test phrases rather than captured speech, and the risk it was blocked for was a property attributed to it without anyone checking.

> The general rule is unchanged and still binds: **any intent whose seeds are drawn from production transcripts rather than authored test phrases must be screened before it reaches an external API.** `Default Fallback Intent` remains the only intent that fails that test. The lesson `reminders.add` adds is the reverse one — an unverified block is also a defect, because it is indistinguishable from a verified one and it stops work for no reason.

**Therefore `Default Fallback Intent` is hand-authored and never sent to a third-party LLM API** (`taxonomy.hand_authored_intents` in `generator_config.yaml`; spec in `authored_specs.yaml`). Hand-authored specs are validated by exactly the same rules as generated ones.

This costs nothing in quality. Fallback is the one intent defined by *exclusion* rather than by evidence — "everything the assistant must not act on" — so it is better authored from Sections 6 and 7 of this document than inferred from seeds. Note also that the sampler makes naive exposure worse than it appears: farthest-point selection actively seeks unusual outliers, and PII-bearing lines are precisely those outliers.

> Any future intent whose seeds are drawn from production transcripts rather than authored test phrases must be screened the same way before it reaches an external API.

### Spec Authoring: All 60 Are Authored, Not Bootstrapped (changed in v3)

v2 specified an LLM bootstrapper (`bootstrap_specs.py` → `intent_specs.yaml`) with hand-authoring reserved for `Default Fallback Intent` on privacy grounds. That is no longer what happens. **All 60 specs are listed in `taxonomy.hand_authored_intents` and live in `authored_specs.yaml`.**

The reason is a limitation of per-intent generation rather than a privacy one, and it generalises: a per-intent API call cannot make sibling boundaries mutually consistent, because it never sees the sibling it must avoid overlapping with. Specs were therefore drafted family by family, so that `Cmd.VolumeIncrease`'s "Do Not Trigger" and `Cmd.StreamingStart`'s "Trigger Conditions" are written against each other. This is the same argument Section 2 already makes about collision detection, applied one level earlier.

`bootstrap_specs.py` and `intent_specs.yaml` are retained: removing a name from `hand_authored_intents` hands that intent back to the bootstrapper.

> [!WARNING]
> **Stage 0's own gate is not cleared.** `intent_specs.yaml` still carries `REQUIRES HUMAN REVIEW before Stage 1 generation`, and `generator_config.yaml` repeats it for all 60. Stage 1 has nevertheless produced corpus output for 16 intents and pilot output for 12. Cheap pilots against unreviewed specs are a reasonable way to test the machinery. Paying for a full run, or training on the result, is not — the specs are the source of truth for every downstream label, so a spec defect is not a data-quality problem that later stages can filter. Review is a prerequisite to the full run, and this document treats it as one.

### Seed Evidence Selection
The bootstrapper and the Stage 1 generator both show the LLM a **maximum-diversity sample** of each intent's seeds (greedy farthest-point traversal over token-set Jaccard distance), not the first *N* lines. The legacy export is permutation-heavy, so the head of a file is a run of near-identical siblings — the least informative evidence available for inferring a boundary. Measured on `Cmd.EdgeModeIncrease`, head-slicing returns ten variants of *"activate edge mode …"*; diverse sampling surfaces *"Focus on him"*, *"He is not clear"*, *"Can't follow her, its too windy"* — the implicit-command phrasings that actually define the intent.

## 3. The Generator Role (LLM System Persona)
To achieve semantic diversity without hallucination or label leakage, the automated generation pipeline will prompt the LLM using the following comprehensive system prompt. This acts as the "Master Blueprint" for the LLM.

```markdown
# Role
You are a **Principal Conversational AI Architect**, **Senior Machine Learning Engineer**, and **Dataset Engineer** with more than **15 years of experience** building production-grade Conversational AI systems deployed to millions of users.
Your areas of expertise include: Conversational AI, Intent Classification, Sentence Embeddings, Contrastive Learning, On-device AI, Edge Machine Learning, ASR, Hearing Aid Voice Assistants, Synthetic Dataset Generation, and Production ML Pipelines.
You specialize in designing datasets for **small embedding models (approximately 10MB)** that run entirely on-device.

Your objective is **not** to generate random paraphrases. Your objective is to engineer a production-quality semantic dataset that maximizes: Intent Classification Accuracy, Semantic Diversity, Robustness, Generalization, and Edge-device Performance while minimizing: Intent Overlap, Dataset Bias, Label Leakage, False Positives/Negatives, and Ambiguous Samples. Always think like an **ML Engineer** optimizing a production model — not like a chatbot generating random text.

# Dataset Generation Philosophy
The generated dataset should prioritize:
- Semantic diversity over syntactic diversity (Meaning over wording)
- Real-world user behavior over textbook grammar
- Production robustness over dataset size
Avoid generating simple word substitutions (e.g., "Increase volume", "Raise volume", "Turn volume up"). These teach almost nothing new. Instead prefer: *"Can you make it easier to hear?"*, *"I'm struggling to hear people"*, *"Could you speak louder?"*. These introduce genuinely different semantic representations.

# Dataset Generation Rules
For every intent, follow the exact workflow below:
**Step 0 — Load Intent Specification**: Before generating anything, load the intent's formal specification (Business Description, Trigger Conditions, Do Not Trigger, Boundary Cases, Neighbor Intents). Treat this spec as the sole source of truth.
**Step 1 — Understand the Intent**: Understand the business meaning, user expectations, and boundaries as defined in the Intent Specification.
**Step 2 — Identify Neighboring Intents**: Use the "Neighbor Intents" field from the specification to determine likely misclassifications for Hard Negative generation.
**Step 3 — Generate Positive Examples**: Generate natural utterances covering different speaking styles consistent with the Trigger Conditions.
**Step 4 — Generate Linguistic Diversity**:
- *Polite Requests:* "Would you...", "Can you please..."
- *Short Commands:* "Louder", "Mute"
- *Elderly Speech:* "I can't hear very well", "Could you make it louder dear?"
- *Partial Sentences / Filled Pauses:* "Little louder...", "Uh...", "Hmm..."
- *Compound / Mixed-Signal:* "It's quieter today, make it louder."
- *ASR Errors:* Simulate realistic speech recognition mistakes (e.g., "Turn it louder", "Raise hearing"). Do not intentionally generate nonsense.
**Step 5 — Generate Hard Negatives**: Generate difficult boundary examples from Neighbor Intents and the "Do Not Trigger" list (e.g., Anchor: *Increase volume* -> Hard Negative: *Start streaming audio*). Generate hard negatives between Commands and Observations (e.g., Anchor: *Turn it up* -> Hard Negative: *The volume seems lower than usual* [Fallback]).
**Step 6 — Generate OOS Examples**: Generate realistic Out-of-Scope utterances (Greetings, Weather, Smart Home).
**Step 7 — Handle Ambiguity & Conflicts (Precedence Rules)**:
- *Compound Observations:* If an utterance contains both an observation and an actionable command, the command always determines the intent. Pure observations map to `Default/Fallback`.
- *Conflicting Commands:* When multiple contradictory commands appear, the **last explicit command takes precedence**.
- *Negations:* If an utterance only cancels an action, classify it as `Fallback`. If it provides a replacement, classify by the replacement.
- *Unresolved Boundary Cases:* Defer to the intent's own "Boundary / Uncertain Cases" field; if still unresolved, default to `Fallback`.
**Step 8 — Metadata Generation**: Generate rich metadata (Utterance, Intent, Intent Family, Type, Difficulty, Source). The `Type` field MUST use the strict enumeration and definitions below:
  - **ExplicitCommand:** Direct request for action. *Note: Grammatical questions requesting an action ("Can you...") are commands.*
  - **ImplicitCommand:** Indirect request expressing a personal need implying an action.
  - **Observation:** Factual statement describing a state without requesting a change.
  - **ObservationPlusCommand:** Observation followed by request.
  - **Question:** Request for information ONLY.
  - **Negation:** Instruction cancelling an action.
  - **Conversation:** Casual speech.
  - **Fallback:** Out-of-scope or unsupported utterances.
**Step 9 — Validation**: Validate duplicate detection, semantic duplicates, intent balance, metadata completeness, and consistency against the Generation Constraints. Reject failures.
**Step 10 — Export**: Output a rich internal metadata dataset and a clean CSV for training.

# Generation Constraints
The generator must NEVER:
- Invent unsupported product features or capabilities not listed in the Intent Specification.
- Change the business meaning of an intent.
- Mix multiple intents in one utterance unless explicitly generating a compound utterance per Step 7.
- Produce contradictory labels.
- Generate duplicate or near-duplicate (semantic) utterances.
- Generate unrealistic or unnatural speech.
- Overuse simple template/word-substitution variations.
- Exceed realistic voice-command length (~20 words), unless explicitly generating a compound case.
- Violate the Step 7 precedence rules while generating hard negatives, OOS, or any other category.

# Output Requirements
1. Load the Intent Specification first.
2. Preserve intent boundaries exactly as specified.
3. Generate semantically diverse positives.
4. Generate realistic linguistic diversity.
5. Generate challenging Hard Negatives.
6. Generate balanced OOS examples.
7. Apply ambiguity/conflict precedence rules consistently.
8. Generate rich, strictly-typed metadata.
9. Validate the generated data against constraints.
10. Export both the metadata-rich dataset and the clean training dataset.
```

> [!CAUTION]
> **`prompt.txt` is not the prompt, and two intended rules were lost in it.**
>
> The blueprint above is the specification. The prompt actually sent is `SYSTEM_PROMPT` in `generator.py` (used at the single call site). `prompt.txt` is a *rendered artifact* produced by `render_prompt.py --manual` for pasting into a chat UI — `render_prompt.py` reads `gen.SYSTEM_PROMPT`, so the file is downstream of the real prompt and nothing reads it back.
>
> It has since been hand-edited, and has therefore diverged in **both** directions:
>
> | | `generator.py` (live) | `prompt.txt` (dead) |
> |---|---|---|
> | `Easy` definition | corrected, structural | retired "high lexical similarity to seed data" |
> | Precedence rule: Help vs Command | **absent** | present |
> | "Difficulty is NOT length" instruction | **absent** | present |
>
> Verified across every commit that touched either file: the Help-versus-Command rule and the anti-length instruction have **never** existed in `generator.py`, in `generator_config.yaml`, or anywhere else. They exist only in `prompt.txt`. Commit `fd6fa0f` is titled "give the generator two boundaries it did not have"; those two boundaries went into the dead file, and the generator still does not have them.
>
> Two consequences worth stating where the prompt is specified rather than in a backlog:
>
> 1. The anti-length instruction was recorded as "measured and did nothing." It was never sent, so it was never measured. Section 5's length tilt is therefore **unexplained by that experiment**, not resisted by it.
> 2. The boundary improvement attributed to the Help-versus-Command rule cannot be attributed to it. The `help` quota profile requires ≥88% `Question`, which is the mechanism actually keeping command-shaped rows out of Help intents.
>
> **Never hand-edit `prompt.txt`.** Change `SYSTEM_PROMPT` and re-render. Better: treat `prompt.txt` as build output and keep it out of version control, so it cannot be edited by mistake in the first place.

## 4. Dynamic Scaling & Intent Families
Intents will be grouped logically to provide context during generation.
*Example Families:*
- **Audio Control:** `Cmd.VolumeIncrease`, `Cmd.VolumeDecrease`, `Cmd.VolumeMute`, `Cmd.VolumeUnmute`
- **Streaming:** `Cmd.StreamingStart`, `Cmd.StreamingStop`

Complex intents will receive larger synthetic budgets to ensure sufficient semantic coverage, while simple intents will receive proportionally less to prevent dataset imbalance.

> **Scope note:** Intent Families exist only to improve synthetic dataset generation and Hard Negative selection. They are not runtime intent labels.

## 5. Multi-Stage Dataset Generation Pipeline (Positives vs Hard Negatives)
To ensure high-quality separation, dataset generation strictly follows a staged pipeline:

| Stage | What it does | Status (2026-08-23) |
|---|---|:-:|
| **0 — Specification** | Author the Intent Specification for every intent (Section 2). Precedes all generation. | Built; **review gate not cleared** |
| **1 — Positives** | Generate diverse positive utterances across all intents, bound by each intent's Trigger Conditions. | Built and running |
| **2 — Clustering** | Deduplicate and cluster the semantic space. | **Not implemented** |
| **3 — Hard Negatives** | Generate Hard Negatives *only* against the finalized positive space, sampled from Neighbor Intents and "Do Not Trigger" cases, and crucially between **Commands** and **Observations**. | **Not implemented** |

### What Stages 2 and 3 Actually Do Today

Recorded plainly because the configuration currently implies otherwise, and a config key that looks live is worse than an absent one — it answers "is this handled?" with a yes.

**Stage 2.** There is no clustering stage. Deduplication happens inline during Stage 1, and `deduplication.scope` is `within_intent`. Nothing measures semantic collision **between** intents in generated data. Section 2 argues at length that an unlabelable region of embedding space is an architectural defect rather than a data-cleaning chore, and that collision detection must therefore run before Stage 0 — `seed_audit.py` does exactly that, for *seeds*. The equivalent check on the generated corpus, which is 20× larger and where the collisions would be newly manufactured rather than inherited, does not exist. This is the single largest gap in the pipeline.

**Stage 3.** There is no hard-negative stage. `generation.hard_negatives_per_intent: 40` and `generation.oos_ratio: 0.15` are present in `generator_config.yaml` and are **referenced by no code**. What exists instead is one `hard_negative_example` string per spec, pasted into the Stage 1 prompt as context. That is useful, and it is not the same thing: a single illustrative example steers generation, whereas Stage 3 was specified to produce a labelled population of boundary rows sampled against the finalised positive space. The Command-versus-Observation negatives that Section 8 names as a high-risk category are, at present, whatever Stage 1 happened to emit.

> **Either build them or delete the keys.** Leaving `hard_negatives_per_intent: 40` in a file that nothing reads is how the next reader concludes hard negatives are handled and stops looking.

## 6. Structured Ambiguity & Out-of-Scope (OOS)
We will enforce a strict "Command vs Observation" mapping policy.

### Compound & Conflict Precedence Rules
> 1. **Compound Observations:** If an utterance contains both an observation and an actionable command, the command always determines the intent. Pure observations map to Fallback.
> 2. **Conflicting Commands:** When multiple contradictory commands appear in the same utterance, the **last explicit command takes precedence**.
> 3. **Negations:** An utterance cancelling an action without providing a replacement maps to Fallback.
> 4. **Unresolved Cases:** If still ambiguous after rules 1–3, defer to the intent's Boundary/Uncertain Cases field, then default to Fallback.

The Fallback/OOS dataset will be deliberately balanced across categories:
- Device observations (e.g., *"It sounds quiet"*)
- General conversation & Greetings
- Irrelevant commands (Weather, Music, Television, Smart Home)

## 7. Metadata, Validation, & Clean Exports

### Rich Metadata
Every generated utterance will include rich metadata tracking its linguistic "DNA".
*Metadata Schema:* `[Utterance, Intent, Type, Difficulty, Source]`

> **`Intent Family` is an input, not a field (corrected in v3).** `schemas.py` does not carry it on `GeneratedUtterance`. The family is supplied *to* the prompt (`generator_config.yaml → families`) to shape generation and neighbour selection, and is recoverable for any row by looking the intent up in that same config. Asking the model to echo it back would add a field that can disagree with the config, which is a defect surface with no corresponding gain. Section 4's scope note already says families are not runtime labels; they are not row labels either.
*Supported Type values:* `ExplicitCommand, ImplicitCommand, Observation, ObservationPlusCommand, Question, Negation, Conversation, Fallback`
*Difficulty:* `[Easy, Medium, Hard]`
*Source:* `[LLM-Generated, ASR-Simulated, Human-Seed]`

#### Difficulty Definitions
To keep difficulty assignment consistent across generation runs:
- **Easy:** Short, single-clause, direct commands. No observation clause, no hedging. *(Defined structurally on purpose. The previous wording, "high lexical similarity to the seed data", contradicted both Section 1 and the generator prompt's own instruction not to reword the seeds — and measurably produced verbatim seed reproductions.)*
- **Medium:** Indirect requests. Polite phrasing. Mild ASR variations.
- **Hard:** Compound utterances. Heavy ASR corruption. Long conversational phrasing. Boundary cases (as defined in each intent's specification).

### Clean Pipeline Consumption
Rich metadata is extremely valuable for evaluation, but the final training pipeline should remain simple.
**Workflow:** `Intent Specifications -> Raw Generated Dataset -> Metadata Rich Dataset -> Validation -> Clean Training Dataset (train.csv)`

### Automated Quality Validation
Before acceptance, the dataset will pass through an automated validation suite:
- Duplicate utterance / semantic duplicate detection
- Intent balance validation
- Metadata completeness checks
- Intent label consistency validation
- Generation Constraints compliance (Section 3)

#### Validation Failure Policy
If a generated sample fails validation:
1. Reject the sample.
2. Regenerate only the failed sample (not the full batch).
3. Repeat until validation succeeds or the retry limit is reached.
4. Log all rejected samples for offline inspection.

> Future dataset iterations may incorporate production feedback and real-world telemetry. The continuous improvement process is documented separately as part of the Model Lifecycle.

## 8. Strict Holdout Evaluation
Evaluation is **two-tier**. These are different instruments and must not be conflated.

| | `dev_synth_hard.csv` | `holdout_eval.csv` |
|---|---|---|
| **Source** | Carved from the generated dataset | Independent of the generator |
| **Distribution** | In-distribution | Out-of-distribution |
| **Purpose** | Day-to-day iteration, error analysis | Final release gate |
| **Visible during iteration?** | Yes | **No** — opened only at the gate |
| **Size** | Large | Small is fine |
| **Status** | Not yet built | **Not yet built** |

**Tier 1 — `dev_synth_hard.csv` (in-distribution).** The `Hard` and compound `Type` slices are isolated out of the generated dataset and excluded from `train.csv`. This is a legitimate, useful iteration signal, and it is what Phase 5 of the roadmap produces.

> [!IMPORTANT]
> **Renamed in v3, because `dev_hard.csv` already exists and is something else.**
>
> P1 built a file called `dev_hard.csv` (`split_dev_sets.py`, `language_packs/en/dev_hard.csv`, 813 rows). It is **not** this instrument. It is the partition of `holdout_honest.csv` that has no near-duplicate in `train.csv` — real held-out speech, frozen for P2–P8, and the primary decision instrument for the entire compression plan at a baseline of 0.8327.
>
> The two files share nothing but a name:
>
> | | P1's `dev_hard.csv` (exists) | This document's Tier 1 (does not) |
> |---|---|---|
> | Rows come from | Real deployed speech | The generation LLM |
> | Selected by | No near-duplicate in `train.csv` | `Type` is Hard or compound |
> | Role | Frozen gate for P2–P8 | Day-to-day iteration signal |
>
> Building Tier 1 under the name `dev_hard.csv` would overwrite the plan's frozen ruler with synthetic data, and it would do so silently — same filename, same directory, plausible row count, and every downstream number quietly retired. Tier 1 is therefore `dev_synth_hard.csv` from here on. The prefix is the point: it says out loud that the rows came from the generator.
>
> A corollary that P1 already fixed and this document should state: **the Super Dataset enters training only.** `check_instruments.py` fails the build on any training row that near-duplicates a `dev_hard` row, which makes the landing a filter over new rows rather than a rebuild of the instrument.

**Tier 2 — `holdout_eval.csv` (sealed).** A genuinely isolated set that the generation LLM has **never** seen and that shares no prompt lineage with the training data — human-authored, or produced from a separate model and prompt, then frozen.

**Tier 2 has not been started.** No `holdout_eval.csv` exists anywhere in the repository. It is worth being blunt about this because Section 8 is the section that decides whether anything ships, and because Tier 2 has never been blocked by the generator — it can be sourced in parallel and always could have been.

> **Why one tier is not enough.** A holdout carved from the same generator, the same prompt and the same model as the training data is not independent of it. It inherits the generator's blind spots, so the model is graded on precisely the cases its teacher was already good at, and the resulting score is optimistically biased. That bias concentrates exactly where it hurts most: FAR is a measure of the *unanticipated* — utterances that should have gone to Fallback but did not — and a synthetic holdout cannot contain what its generator failed to anticipate. Tier 1 answers "did this change help?"; only Tier 2 answers "is this safe to ship?"

Both tiers must explicitly cover these high-risk categories:
- Compound Observation + Command
- Negated Commands
- Multi-clause Requests
- Observation followed by Action
- Conflicting Commands
- ASR-heavy utterances
- Very short commands
- Extremely polite requests

### Evaluation Metrics
- **Overall accuracy and per-intent F1** across each evaluation set. Report the two tiers separately and never average them; a gap between them *is* the generalization signal.
- **Confusion matrix** broken down by `Type` (from Section 7 metadata) to isolate failure modes — e.g. accuracy specifically on `ObservationPlusCommand` vs pure `Observation`.
- **False-Accept Rate (FAR):** Rate at which Fallback/OOS utterances are misclassified as an actionable command. This is the priority metric — weighted more heavily than overall accuracy, since an unintended device action (like blasting the volume) is costlier than an unnecessary Fallback for hearing-aid users.
- **False-Reject Rate (FRR):** Rate at which genuine commands are misclassified as Fallback.

---

> [!IMPORTANT]
> **Implementation Next Steps (v3).**
> Ordered so that each item is measurable when it is reached. Items 1–2 need no API spend.
>
> 1. ~~Author Intent Specifications (Section 2) for every intent in the taxonomy.~~ Done — all 60 authored in `authored_specs.yaml`. **Human review is still outstanding and gates the full run.**
> 2. ~~Construct the Python-based Generator tool.~~ Stage 1 done. **Stages 2 and 3 are not built** (Section 5).
> 3. Source the sealed Tier-2 `holdout_eval.csv` (Section 8) from outside the generator's prompt lineage. Not blocked by anything; start it in parallel.
> 4. Resolve the 60-versus-57 instrument gap (Section 2) before any Super Dataset result is reported.
> 5. Build Stage 2 cross-intent collision detection over the generated corpus, or record explicitly that generated collisions go unmeasured.
> 6. Build Stage 3, or delete `hard_negatives_per_intent` and `oos_ratio` so the config stops implying it exists.
> 7. Land the runtime label-map delta (Section 2). Until it lands nothing here is deployable, however good the dataset is.
