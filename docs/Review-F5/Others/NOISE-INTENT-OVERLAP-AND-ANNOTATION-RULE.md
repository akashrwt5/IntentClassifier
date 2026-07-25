# The "noise" intent overlap — analysis & annotation rubric

**Problem:** the word *noise* (and loud / unbearable / deafening / environment) is
smeared across **15 intents** in training. Neither the model nor a human annotator
can cleanly separate them, so the holdout contains contested labels that make the
accuracy number partly meaningless. This doc shows the overlap and proposes one
consistent rule to re-annotate against.

---

## 1. How "noise"-words are distributed in TRAINING (495 rows)

| Intent | rows | What these examples actually are |
|---|---|---|
| Cmd.MemoryChange | 178 | names an environment/preset: "change to **restaurant** memory", "crowd memory" |
| Help_Tinnitus | 105 | the tinnitus masker: "white **noise** masker", "noise stimulus" |
| Help_Volume | 60 | *how-to* about loudness: "**how do i** turn down the loudness in one aid?" |
| Cmd.VolumeDecrease | 54 | direct command: "**turn down** the loudness", "min loudness" |
| Cmd.VolumeIncrease | 34 | direct command (the opposite) |
| Default Fallback | 33 | genuinely out of scope |
| Help_EdgeMode | 15 | always names the feature: "reduce noise **with edge mode**" |
| Help_Customize | 7 | generic how-to: "**how do i** reduce background noise" |
| (9 more intents) | ~10 | long tail — 1–3 rows each |

**The smell:** "noise" has no clean owner. The same concept (a noisy environment)
legitimately appears under MemoryChange, Tinnitus, EdgeMode, Customize, and the
Volume intents.

## 2. The holdout is internally inconsistent (the real bug)

These 12 holdout phrases all describe the *same situation* — "it's loud / noisy" —
but are labeled five different ways, and two near-identical phrases get **different**
labels:

| Current gold label | Phrase |
|---|---|
| Cmd.MemoryChange | adjust for this loud room |
| Cmd.MemoryChange | this environment needs a different setup |
| **Cmd.VolumeDecrease** | **this place is deafening** |
| Cmd.VolumeDecrease | the noise is unbearable right now |
| Cmd.VolumeDecrease | the loudness is giving me a headache |
| Cmd.VolumeIncrease | the television is blaring and i still miss the words |
| Cmd.VolumeIncrease | add some more loudness please |
| Cmd.VolumeMute | kill the noise completely |
| Help_Customize | reduce the background noise level |
| Help_EdgeMode | too much wind noise outside |
| Help_EdgeMode | reduce wind interference with edge mode |
| Help_Volume | help with adjusting loudness |

> "adjust for this loud room" → **MemoryChange**, but "this place is deafening" →
> **VolumeDecrease**. Same meaning, different label. That inconsistency — not the
> model — is what's driving the confusion and the unstable score.

## 3. The distinguishing principle (derived from the clean examples)

Read against the training data, the intents *do* separate cleanly on two axes.
The contested rows are the ones that don't state an action or a feature.

**Axis 1 — Command vs Help (is the user acting, or asking?)**
- Imperative / symptom ("turn it down", "it's too loud") → a `Cmd.*` action.
- Interrogative ("**how do i** …", "what is …") → a `Help_*` topic.

**Axis 2 — within Command, what did they name?**
- A direct intensity word ("loud / turn down / quieter") → **Cmd.VolumeDecrease**.
- A place / preset ("restaurant", "crowd", "this environment", "program for …")
  → **Cmd.MemoryChange**.

**Axis 3 — within Help, what feature did they name?**
- Says "edge mode" → **Help_EdgeMode**.
- Says masker / white noise / tinnitus → **Help_Tinnitus**.
- Generic "how do i reduce background noise" → **Help_Customize**.
- "how do i change the loudness" → **Help_Volume**.

## 4. Proposed annotation rubric (apply top-to-bottom, first match wins)

1. Contains a **feature name** ("edge mode", "tinnitus/masker", a named program) →
   that feature's intent (EdgeMode / Tinnitus / MemoryChange).
2. Phrased as a **question** ("how do i…", "what is…", "can i…") → the matching
   `Help_*` topic (Help_Volume, Help_Customize, …), **never** a `Cmd.*`.
3. **Imperative volume command** ("turn it down/up", "louder/quieter", "mute") →
   the matching `Cmd.Volume*`.
4. Names an **environment/preset** without an intensity verb ("loud room",
   "restaurant", "this environment") → **Cmd.MemoryChange**.
5. **Bare symptom, no action, no feature** ("the noise is unbearable", "this place
   is deafening") → **pick ONE canonical intent and apply it to ALL of them.**
   Recommended: `Cmd.VolumeDecrease` (the safest immediate action), *or* introduce
   a dedicated `Cmd.NoiseReduce` command if the product wants noise-management
   distinct from volume. Do not split identical symptoms across labels.

**Key product decision to make:** should a *symptom* ("it's too noisy") trigger an
**action** (lower volume / activate noise management) or route to a **help topic**
(explain Edge Mode)? A symptom expects an action — so it should map to a `Cmd.*`,
not a `Help_*`. That single ruling resolves most of the 12 rows.

> **Interim decision (accepted):** the app does not yet support activating Edge
> Mode by voice, so noise-symptom utterances map to **`Help_EdgeMode`** for now.
> Adding a `Cmd.EdgeModeActivate` command is tracked as **FW-001** in
> `FUTURE-WORK.md` — revisit these labels once the capability ships.

## 5. Recommended next steps

1. Adjudicate the 12 rows in `noise_holdout_adjudication.csv` using the rubric —
   one owner, one pass, consistent.
2. If "symptom → Edge Mode" is desired, Edge Mode needs **symptom-style training
   examples** (today it has zero — every row literally says "edge mode"), and it
   should probably be a *command* variant, not the help topic.
3. Collapse or clarify the overlap: decide the single owner for "environmental
   noise" vs "tinnitus masking" vs "volume", and write these rules into the
   annotation guide so new data stays consistent.
4. Only then re-measure. Tuning the model against contested labels just overfits
   to one annotator's coin-flip.
