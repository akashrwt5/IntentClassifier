# NLU Dataset Architecture: The "Super Dataset" Strategy (v2 — Final)

This document outlines the final architectural blueprint for upgrading the legacy Dialogflow dataset into a production-grade embedding dataset for our 10MB Edge Semantic Model.

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
| Exclude | `_EntityMemory`, `_EntityRecurrence`, `_EntityRemind` | Dialogflow **entity value lists**, not intents. Contents are bare slot values (`Normal`, `Restaurant`, `Outdoors`). |
| Merge | `Help.Activity` → `Help_Activity` | Dot-variant from an older export; 25 of its 29 unique phrases already appear in the target. Phrases are unioned. |
| Drop | `Cmd.Health` | Rollup **parent**, not a sibling: 155 of its 160 unique utterances are drawn verbatim from the `Cmd.Activity*` children. |
| Drop | `Help_HearingCareAnywhereConnect` | Disabled in Dialogflow, and a 100% subset of `Help_RemoteProgramming` — zero distinguishing utterances. |

**Resolved taxonomy: 57 intents.**

> **Why this is an architectural concern, not a data-cleaning chore.** Two intents whose seed utterances are identical do not present a *hard* classification problem; they present an **unlabelable** one. Generating synthetic data for both teaches the model to split an inseparable region of the embedding space, which surfaces as unstable confidence — and unstable confidence on command intents is exactly what inflates False-Accept Rate (Section 8). No amount of downstream generation quality repairs a taxonomy defect, so collision detection runs *before* Stage 0.

Any intent added to the taxonomy in future must clear the same bar: it needs at least one utterance that no other intent claims.

### Data Handling Constraint: Fallback Seeds Contain Customer PII
The `Default Fallback Intent` seed file is not a curated OOS list — it is 613 raw production ASR transcripts, including speech captured ambiently around the wearer. It contains a street address, a phone number, benefits and health details, personal names, and fragments of private conversation. `reminders.add` similarly contains medication reminders.

**Therefore `Default Fallback Intent` is hand-authored and never sent to a third-party LLM API** (`taxonomy.hand_authored_intents` in `generator_config.yaml`; spec in `hand_authored_specs.yaml`). Hand-authored specs are validated by exactly the same rules as generated ones.

This costs nothing in quality. Fallback is the one intent defined by *exclusion* rather than by evidence — "everything the assistant must not act on" — so it is better authored from Sections 6 and 7 of this document than inferred from seeds. Note also that the sampler makes naive exposure worse than it appears: farthest-point selection actively seeks unusual outliers, and PII-bearing lines are precisely those outliers.

> Any future intent whose seeds are drawn from production transcripts rather than authored test phrases must be screened the same way before it reaches an external API.

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

## 4. Dynamic Scaling & Intent Families
Intents will be grouped logically to provide context during generation.
*Example Families:*
- **Audio Control:** `Cmd.VolumeIncrease`, `Cmd.VolumeDecrease`, `Cmd.VolumeMute`, `Cmd.VolumeUnmute`
- **Streaming:** `Cmd.StreamingStart`, `Cmd.StreamingStop`

Complex intents will receive larger synthetic budgets to ensure sufficient semantic coverage, while simple intents will receive proportionally less to prevent dataset imbalance.

> **Scope note:** Intent Families exist only to improve synthetic dataset generation and Hard Negative selection. They are not runtime intent labels.

## 5. Multi-Stage Dataset Generation Pipeline (Positives vs Hard Negatives)
To ensure high-quality separation, dataset generation will strictly follow a staged pipeline:
1. **Stage 0 (Specification):** Author the Intent Specification for every intent (Section 2). This precedes all generation.
2. **Stage 1 (Positives):** Generate diverse positive utterances across all intents, bound by each intent's Trigger Conditions.
3. **Stage 2 (Clustering):** Deduplicate and cluster the semantic space.
4. **Stage 3 (Hard Negatives):** Generate Hard Negatives *only* against the finalized positive space. Hard negatives must be sampled from Neighbor Intents and "Do Not Trigger" cases, and crucially, between **Commands** and **Observations**.

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
*Metadata Schema:* `[Utterance, Intent, Intent Family, Type, Difficulty, Source]`
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

| | `dev_hard.csv` | `holdout_eval.csv` |
|---|---|---|
| **Source** | Carved from the generated dataset | Independent of the generator |
| **Distribution** | In-distribution | Out-of-distribution |
| **Purpose** | Day-to-day iteration, error analysis | Final release gate |
| **Visible during iteration?** | Yes | **No** — opened only at the gate |
| **Size** | Large | Small is fine |

**Tier 1 — `dev_hard.csv` (in-distribution).** The `Hard` and compound `Type` slices are isolated out of the generated dataset and excluded from `train.csv`. This is a legitimate, useful iteration signal, and it is what Phase 5 of the roadmap produces.

**Tier 2 — `holdout_eval.csv` (sealed).** A genuinely isolated set that the generation LLM has **never** seen and that shares no prompt lineage with the training data — human-authored, or produced from a separate model and prompt, then frozen.

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
> **Implementation Next Steps:**
> The architectural blueprint is now fully locked. The next engineering task is to:
> 1. ~~Author Intent Specifications (Section 2) for every intent in the taxonomy.~~ Bootstrapped by `bootstrap_specs.py` into `intent_specs.yaml` — **requires human review** before Stage 1.
> 2. Construct the Python-based Generator tool that implements this exact pipeline.
> 3. Source the sealed Tier-2 `holdout_eval.csv` (Section 8) from outside the generator's prompt lineage. This is not blocked by the generator and can proceed in parallel.
