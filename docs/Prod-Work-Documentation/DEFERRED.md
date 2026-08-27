# Deferred and open — Stage 0 spec review

Things that were consciously put down, and things nobody has answered yet. They
live here because a decision deferred in conversation is a decision lost: it
leaves no trace in the specs, no flag in `SPEC_REVIEW.md`, and no row in any
report. The per-family sign-off table in `SPEC_REVIEW.md` records what has been
*reviewed*; this file records what has been *skipped*, and why.

Every item states what closing it needs. An item with no closing condition is a
wish, not a task.

Last updated 2026-08-27, after the `Cmd.*` review, the HelpAudio,
HelpAppSettings, HelpHealth, HelpDeviceCare and HelpConnectivity families,
`Default Fallback Intent`, and an audit of everything above.

---

## Where the review actually stands

| | Count | State |
|---|---:|---|
| `Cmd.*` reviewed | 19 of 24 | AudioControl 4, Streaming 2, Messaging 2, Memories 1, DeviceStatus 1, DeviceLocate 1, ActivityTracking 8 |
| `Cmd.*` deferred | 5 | EdgeMode 3, SpeechServices 2 |
| `Help*` deferred | 1 | `Help_EdgeMode`, grouped with the EdgeMode commands |
| `Help*` reviewed | 25 of 33 | **HelpAudio** 4 of 5 (`Help_EdgeMode` deferred), **HelpAppSettings**, **HelpHealth**, **HelpDeviceCare** 6 of 6 each, **HelpConnectivity** 3 of 3. Two others were *read as counterparts* only |
| Other | 1 of 3 | `Default Fallback Intent` reviewed. `reminders.add` and `reminders.complete` still not |

`Default Fallback Intent` has now been read end to end (2026-08-26). It had been
edited in almost every round without ever being reviewed itself — the most rules
of any intent and the least scrutiny, which was the wrong way round. Its 23 rules
held up; what did not was its neighbour list. See D7.

**No sign-off box in `SPEC_REVIEW.md` is ticked, and `intent_specs.yaml` still
carries `REQUIRES HUMAN REVIEW`. Both are correct: the review is not finished.**

---

## A. Deferred intent reviews

### A1. EdgeMode — `Cmd.EdgeModeIncrease`, `Cmd.EdgeModeDecrease`, `Cmd.EdgeModeDeactivate`, `Help_EdgeMode`

Deferred by decision. Not skipped quietly — but it is the *least* safe family to
leave, because `Cmd.EdgeModeIncrease` has now been edited **five** times during
other families' reviews without ever being reviewed itself:

- the no-cue complaint rule (AudioControl round)
- the environment-cue-with-vague-remedy trigger (AudioControl round)
- the presenter/named-source boundary against `Cmd.StreamingStart` (Streaming round)
- the "environment named as the cause of difficulty" rule (Memories round)
- two memory-name collision rules, `Speech` and `Noise` (`Help_Tinnitus` round)

Five edits into a spec nobody has read is now the largest single risk carried by
this deferral. Each edit was correct in the context that produced it; none was
checked against the other four, or against `Cmd.EdgeModeDecrease`, which must
share the same trigger surface in mirror image.

`Cmd.EdgeModeDecrease` and `Cmd.EdgeModeDeactivate` have not been touched or read
at all.

**`Help_EdgeMode` deferred with them (Akash, 2026-08-26), when the HelpAudio
family came up.** It is the Help counterpart of all three commands and its whole
job is the boundary between explaining Edge Mode and doing it, so reading it
without them would be reading one side of a contract. Its deployed rows are
0.0% command-shaped across 56, so there is no evidence of leakage in the
direction that would make this urgent.

The cost of that grouping: two other HelpAudio specs point at this family and are
now accepted unverified. `Help_IntelliVoice` and `Help_MaskMode` each route
"improve clarity now" to `Cmd.EdgeModeIncrease` and each defer Edge Mode
questions to `Help_EdgeMode`, and none of those four cross-references has been
checked from the other side. Same shape as the `Cmd.ListenMessage` →
`Cmd.TranscribeStart` dependency in A2.

**Three edits are queued against this family and deliberately not applied**
(Akash, 2026-08-26), so that `Cmd.EdgeModeIncrease`'s blind-edit count stays at
five. All three belong to this family's own round:

1. **A disclaimer naming the two features it does not own.** `Cmd.EdgeModeIncrease`
   claims "activate, add or increase Edge Mode *or adaptive tuning*" and "asks
   for voices to be made clearer", which makes it the nearest attractor for a
   direct request naming either feature — and acting on one applies Edge Mode, a
   device action the user did not ask for. Both Help specs state their side; this
   one is silent.

   **Corrected 2026-08-27.** This item originally read "Searched all 60 specs …
   **No `Cmd.*` spec names either**". That was true when written and D6 made it
   false three rounds later, by adding *"Naming a MODE is not a request to change
   program. \"Mask Mode\" is the subject of Help_MaskMode"* to
   `Cmd.MemoryChange`'s boundary cases. "Mask Mode" now appears in four specs —
   `Help_MaskMode`, `Help_Tinnitus`, `Help_MemoryOptions` and `Cmd.MemoryChange`.
   `Cmd.EdgeModeIncrease` still names neither feature, which is the part that
   matters here, but the absolute claim was left to rot for three rounds. A
   finding that quotes a search result has to be re-run before it is relied on.
2. **The neighbour link.** `Help_IntelliVoice` and `Help_MaskMode` each name
   `Cmd.EdgeModeIncrease` in `do_not_trigger` but neither lists it in
   `neighbor_intents`. That is the prose-only downgrade this review has now
   flagged three times: the boundary is documented but never reaches the prompt
   as a confusion and is never sampled for hard negatives. It cannot be fixed
   from the Help side alone, because the check requires the link to be mutual.
   `spec_review.py` cannot see it either — the pair sits below the 0.20 TF-IDF
   reporting threshold.
3. **`Cmd.EdgeModeDeactivate` and `Help_MaskMode` now claim overlapping ground
   with nothing on either side naming the other.** Surfaced by the
   `Help_MaskMode` round, and caused by it: correcting that spec's description to
   "how to switch it on or off" moved the pair from **0.1491 to 0.2381**, into
   `SPEC_REVIEW.md` Section 2b. The overlap is real rather than a threshold
   artefact — `Cmd.EdgeModeDeactivate` triggers on "turn off, stop, end, cancel
   or deactivate", `Help_MaskMode` on "how to enable, disable or find", and
   *"turn off mask mode"* is a sentence both shapes fit. It names Edge Mode
   explicitly, which is what keeps them apart today, but the generator sees one
   spec at a time. Naming it from the `Help_MaskMode` side alone would only move
   the pair to Section 2c, so it waits for a fix on both sides.

**To close:** read all four specs against each other; confirm the five edits
above still hold when the family is read as a whole; apply the three queued
edits; and check the four cross-references from `Help_IntelliVoice` and
`Help_MaskMode` from this side.

### A2. SpeechServices — `Cmd.TranscribeStart`, `Cmd.TranslationStart`

Deferred by decision. Neither spec has been read.

One known dependency: `Cmd.ListenMessage`'s boundary rests on
`Cmd.TranscribeStart` ("requests to start live transcription ... are
Cmd.TranscribeStart"). That rule was accepted during the Messaging review
without reading the intent it points at.

**To close:** read both specs, plus `Help_Transcribe` and `Help_Translate`; check
the `Cmd.ListenMessage` cross-reference from the other side.

---

## B. Open capability questions

These are the questions only the product can answer, and they have found a real
defect four times out of four: `max volume`, `connect the remote microphone`,
`find my hearing aids`, and the memory-name collisions. Treat an unanswered one
as an unverified capability claim, not as a detail.

### B1. Does `Cmd.FindMyPhone` actually locate the phone?

`Help_FindMyHearingAids` was confirmed to only *explain* how to find an aid,
which is why it stays a Help intent and now carries an explicit carve-out. The
mirror question was never asked: do the aids ring or locate the **phone**, or do
they only explain how?

**If they only explain, `Cmd.FindMyPhone` is mislabelled the same way** — a
Command intent for an action the product does not perform.

**To close:** one answer. If it only explains, the intent moves to Help, which is
a runtime label-map change.

### B2. Does the product read incoming **texts** aloud?

`Cmd.ListenMessage` claims it:

    - User asks to have incoming texts or messages read aloud.

Confirmed: voice and push-to-talk messages play, and the product plays the most
recent message. Text messages were never confirmed.

**To close:** one answer. If texts are not read aloud, that trigger line is an
invented capability and must go.

---

## C. Taxonomy gaps

Two real user questions have no intent that answers them. In both cases the
routing is now stated explicitly so nothing is *silently* misrouted — but a user
asking a reasonable question gets a clarification instead of an answer.

Both were found the same way: a Cmd spec routed the how-to somewhere, and the
intent it named turned out not to claim it. That one-way-route shape has now
appeared four times in this review. It is worth checking for directly rather than
waiting to trip over it.

### C1. No Help intent for messaging or push-to-talk

Decided during the Messaging review: how-to questions about messaging
("how do I send a voice message?") go to `Default Fallback Intent`, because no
`Help_PushToTalk` or `Help_Messaging` intent exists. `Help_VoiceAssistant`
explains the assistant itself, not this feature.

Stated in four places: `Cmd.SendMessage`, `Cmd.ListenMessage`,
`Help_VoiceAssistant` and `Default Fallback Intent`.

**To close:** either add the intent — taxonomy 60 → 61, runtime label map,
config, and no seed data to draw on — or accept the gap deliberately and record
that decision here.

### C2. No intent for powering the aids on or off

Found during the `Help_Volume` review. `Cmd.VolumeMute` had been routing
"how do I turn my aids off?" to `Help_Volume`, which never claimed it — powering
a device off is not volume control, and `Help_Volume`'s business description does
not cover it. Searched the whole taxonomy: **no intent covers powering the aids
on or off**, including every member of the HelpDeviceCare family.

Decided: both the direct request and the how-to question are
`Default Fallback Intent`, matching C1. Powering off was already recorded as not
a supported *voice* capability; what was missing is that nothing *explains* it
either.

Stated in three places: `Cmd.VolumeMute`, `Help_Volume` and
`Default Fallback Intent`.

**To close:** same choice as C1 — add an intent, or record the gap as accepted.

---

## D. Decided, and deliberately left alone

Recorded so nobody "fixes" them later without knowing they were considered.

### D1. `Help_Tinnitus` no longer lists `Cmd.VolumeIncrease` as a neighbour

Raised during the neighbour-restoration round and left as-is by decision. Not a
`command_help_pairs` contract pair. Worth revisiting only if tinnitus-masker
loudness questions turn up misclassified as volume commands.

### D2. Seven neighbour relationships dropped and never restored

The neighbour rework removed 46 links and restored 13. Seven of the removed ones
are gone from `neighbor_intents` **and** unmentioned in any rule field, so
nothing anywhere connects them:

| From | To |
|---|---|
| `Cmd.BatteryLevel` | `Help_SelfCheck` |
| `Help_FallAlert` | `Help_DeviceSettings` |
| `Help_IntelliVoice` | `Help_Customize` |
| `Help_MaskMode` | `Help_Customize` |
| `Help_RemoteProgramming` | `Help_DeviceSettings` |
| `Help_Tinnitus` | `Help_Customize` |
| `Help_Translate` | `Help_VoiceAssistant` |

An eighth, `Help_Tinnitus` → `Cmd.VolumeIncrease`, was on this list and is
recorded separately as D1 because it was explicitly decided rather than
overlooked. The seven above were never ruled on — they are Help-to-Help, which is
why they were not urgent, but "not urgent" was never "reviewed".

**Separately**, two links removed in the same rework but *outside* this table —
they kept their prose boundary, so they were never fully dropped — were restored
during the `Help_Volume` review: `Help_Volume` ↔ `Help_SelfCheck` and
`Help_Volume` ↔ `Help_Tinnitus` are mutual neighbours again. The boundary was
written in prose while the sampling relationship had been deleted, which is the
exact trade the F-section warns about. `Help_Volume` ↔ `Help_Customize` was
considered and left out: its boundary (equalizer and sound tuning versus
loudness) is coarse enough not to need hard negatives.

**To close:** the seven fall out naturally when the Help families are reviewed.
Check this table then rather than in isolation.

### D3. Should "tie goes to inaction" be risk-weighted?

`Default Fallback Intent` states the principle absolutely. Two specs already
depart from it in the same direction and for what looks like a good reason:

    bare "play it"     -> Cmd.ListenMessage    (seed evidence)
    bare "turn it off" -> Default Fallback Intent
    bare "turn it on"  -> Default Fallback Intent

Playing a message is reversible; muting the aids or ending a stream changes the
device. The asymmetry looks correct, but it is nowhere stated, so today it reads
as two specs disagreeing.

A **third** departure surfaced in the HelpHealth review, and this one is
deliberate and written down. `Help_FallAlert` says that where an utterance is
ambiguous between it and a general help intent, prefer it — "an unanswered safety
question is costlier than a redundant explanation". That is risk weighting stated
outright, on a different axis (Help versus Help, so no device action and no FAR
cost) and in the opposite direction to inaction.

Three specs now weight risk against the absolute principle, in two directions and
for good reasons each time. The principle as written admits none of them.

**To close:** one line in `Default Fallback Intent`'s boundary cases making the
risk weighting explicit, or a decision that the principle is absolute and
`Cmd.ListenMessage` should change. Raised, not answered.

### D4. Medical and clinical questions are Fallback, by decision

Found during the `Help_Tinnitus` review. That spec carried a trigger claiming
questions about what the ringing in the ears *is* — a question about the
condition, not about the product's masker feature. Nothing in the taxonomy
answers a clinical question, and answering one is not something this product
should attempt.

Decided (Akash, 2026-08-26): the trigger is removed; questions about a medical
or clinical condition — what tinnitus is, why the ears ring, whether hearing loss
can be cured — are `Default Fallback Intent`. `Help_Tinnitus` explains the masker
FEATURE only.

Stated in two places: `Help_Tinnitus.do_not_trigger` and
`Default Fallback Intent.trigger_conditions`.

This is deliberately **not** filed as a taxonomy gap like C1 and C2. Those are
supported features with no intent to explain them. This is a class of question
the product should decline, so Fallback is the right answer rather than a
missing one.

### D5. `Help_Tinnitus` needs no command carve-out

`Help_FindMyHearingAids` required an explicit carve-out because 40.6% of its
deployed rows are command-shaped — users tell it to do the thing rather than ask
how. The same check was run on `Help_Tinnitus` before assuming it needed one:
**0.0% command-shaped across 191 rows.** It is a Help intent in phrasing as well
as in name, so no carve-out was added.

Recorded because the absence of a carve-out here is a measured result, not an
oversight. The 40.6%/0.0% check is worth running on each Help intent as its
family comes up.

### D6. Naming a mode is not a request to change program

Decided by Akash, 2026-08-26, during the `Help_MaskMode` round.

    "switch to Mask", "change my memory to Mask"   ->  Cmd.MemoryChange
    anything saying "Mask Mode", on/off verb or not ->  Help_MaskMode

The reasoning is his: "Personal Mode on kar do" would not be read as a request to
load the Personal memory either. A memory switch needs the **bare memory name
with an explicit change verb**; the phrase "<name> Mode" names a feature.

This changed a REASON, not an outcome. `Help_MaskMode` already kept those
utterances, but it justified doing so with "there is no Cmd intent for Mask
Mode" — which stopped being true the moment `Mask` turned out to be a memory,
because `Cmd.MemoryChange` is exactly that intent. A spec that reaches the right
answer through a false premise will reach a wrong one as soon as the premise is
leaned on again, so the premise was replaced rather than left alone.

Stated in three places now: `Help_MaskMode`'s boundary cases and its Mask-memory
exclusion, and `Cmd.MemoryChange`'s boundary cases.

Two pieces of evidence pointed opposite ways and the product call settled it. For
Help: the deployed model already routes the four command-shaped `Help_MaskMode`
rows to Help. For Command: `Cmd.MemoryChange`'s own seed file contains **no row
naming mask at all**, which would be odd if users did ask to switch to it.

**Nothing to close.** Recorded so nobody restores the old reasoning.

### D7. `Default Fallback Intent` reviewed — the rules were right, the neighbours were not

Read end to end on 2026-08-26. Its 10 triggers, 7 exclusions and 6 boundary cases
survived; two findings were raised against them and **one of the two was mine and
wrong**.

**Withdrawn — `Section 6` is a real reference.** The trigger *"where the intent
remains genuinely ambiguous after the Section 6 precedence rules have been
applied"* was flagged as a dead pointer, because `SYSTEM_PROMPT` numbers its
precedence rules 1–5 and contains no "Section 6". It refers to the **blueprint**,
not the prompt: `nlu_super_dataset_architecture.md` §6 is "Structured Ambiguity &
Out-of-Scope (OOS)", whose subsection is "Compound & Conflict Precedence Rules".
`generator.py` uses the same convention twice in its own comments, one of them
saying so outright — "stated the way the blueprint states it". The line is
correct and was left alone. Akash pushed back before it was changed.

**The neighbour list is where the real defect was.** 59 specs list Fallback as a
neighbour; Fallback lists 6. The asymmetry check exempts it by design, because it
cannot carry 59 — but the exemption meant nobody ever asked *which* 6, and the
answer had drifted from what the spec argues about.

`Cmd.MemoryChange` was **not** among them, while the listening-environment
exception — Fallback's single most intricate rule, stated across three fields —
is entirely about it. Added. One-sided, since `Cmd.MemoryChange` already lists
Fallback.

The other three were first read as stale and that reading was also wrong; naming
an intent and sharing its subject are not the same test:

| neighbour | in Fallback's rules | from its own side | verdict |
|---|---|---|---|
| `Cmd.StreamingStart` | "TV, phone, smart-home, music services"; "Turn the TV up" | names Fallback 3× | real, kept |
| `Help_Volume` | trigger 8 and exclusion 2, the power on/off gap | names Fallback for that gap | real, kept |
| `reminders.add` | nothing | nothing | link real, rule missing — below |

**The reminders boundary existed on one side only.** `reminders.add` says "the
subject never changes the intent" and its own positive example has a shopping
subject, while Fallback's trigger 1 claims shopping outright. Both specs claimed
"remind me to buy milk" and only one of them knew it.

Decided (Akash, 2026-08-26): **a reminder is a reminder whatever it is about.** An
exclusion naming `reminders.add` now sits on the Fallback side, so trigger 1
cannot swallow reminders. `reminders.add` unchanged — it was already right, and it
is still unreviewed.

**Nothing to close.** Recorded so the withdrawn finding is not re-raised and the
neighbour reasoning is not re-litigated.


### D8. `Help_Home` is the Home screen, not the help space's catch-all

Decided by Akash, 2026-08-26, during the HelpAppSettings review.

`Help_Home` carried a fifth trigger — *"User asks a broad orientation question
about how to use the app or the hearing aids"* — supported by two boundary cases
that made it the general orientation fallback **within** the help space, ahead of
`Default Fallback Intent`.

**The seed evidence does not support it.** 16 of its 23 seeds name the home or
main screen outright, and not one is a general how-do-I-use-the-app question.

**And it collided with three intents at once**, measured on the deployed rows:

| phrasing | `Help_Home` | `Help_WhatsNew` | `Help_DemoMode` |
|---|---:|---:|---:|
| quick start | 0 of 104 | **7 of 68** | 0 of 44 |
| overview / summary | 4 | **22** | 0 |
| getting started | 2 | 1 | 0 |
| how to use the app | 2 | 0 | **3 of 44** |
| home / main screen | **19** | 0 | 0 |

`Help_WhatsNew` owns quick start and overview outright; `Help_DemoMode` owns
using the app without aids; `Help_Health` owns finding a health figure in the
app. All three were claimed in passing by that one trigger, and **none of the
three collisions was guarded** — the two specs that did name each other did so on
a different axis entirely (what's new, and the Home screen), so they read as
guarded while the real overlap sat open.

Trigger removed. Both dependent boundary cases rewritten. A broad question naming
no screen and no feature is `Default Fallback Intent`, which is what that intent
is for. The `Help_WhatsNew` exclusion now names quick start, overview and
getting-started explicitly.

This also closed `Help_Health` ↔ `Help_Home`, the long-running Section 2a item —
see E1.

**Nothing to close.**

### D9. `Help_DeviceSettings` ↔ `Help_Customize` had the boundary but not the link

Found in the same round. Both specs name the other in `do_not_trigger` — device
level preferences versus per-memory sound shaping, which is this family's
most-used boundary — and neither listed the other in `neighbor_intents`.

Because the absence was symmetric it passed the one-directional-link check, and
at **0.16** the pair sits below `spec_review.py`'s 0.20 reporting threshold, so
nothing flagged it. The boundary was documented and never sampled as a hard
negative. Made mutual.

**Nothing to close** for this pair. The wider pattern is not closed — see E4.

### D10. The assistant does not report a heart rate

Decided by Akash, 2026-08-26, in the HelpHealth review. `Help_HeartRate` carried
the trigger *"User asks for their current heart rate"* — a request for a VALUE,
in a Help intent — justified by a boundary case reading *"No Cmd intent exists
for reading heart rate, so a direct request for the current value also resolves
here rather than to Fallback."*

The justification was backwards. There is no Cmd intent because **the product
does not report the value**; it shows where the reading is found. Written as it
was, the spec asserted a capability the product does not have — the same defect
B2 tracks for reading texts aloud.

The trigger is gone and the business description now says outright that the
assistant explains where the reading is shown and does not report it. The
utterance still lands here, and the boundary case now gives the correct reason:
`Help_FindMyHearingAids`' reason. The assistant cannot do the thing, so
explaining IS the action, and precedence rule 4's clause — *"when the requested
action IS explaining, it is the Help intent"* — covers the direct phrasing as
well as the how-to. 5.9% of this intent's deployed rows are command-shaped, and
that is correct rather than a leak.

**Nothing to close.**

### D11. Interpreting a health value is a health question, not a product question

Decided by Akash, 2026-08-26. `Help_HeartRateRecovery` claimed two things beyond
explaining its measurement — *"whether their heart rate recovery is good, or what
a normal value is"* and *"how to improve their heart rate recovery"* — and its
business description promised *"what a good value looks like"*.

D4 had already decided that a medical or clinical question is
`Default Fallback Intent`, because the product explains its own features and not
the user's health. Nothing connected the two rules, so the taxonomy declined to
say what tinnitus is while offering to interpret a cardiac measurement and advise
on improving it.

Both triggers removed, the business description corrected, the boundary case
rewritten, and an exclusion added naming Fallback. **D4's Fallback trigger is
widened to match** — it covered a medical *condition*, and now also covers
whether a measured health value is good or normal and how to improve one.
Explaining what the measurement means stays here; interpreting the user's own
result does not.

**A correction, made 2026-08-27.** The widened wording said *"whether a measured
health value is good or normal, or how to improve one"*. That reaches the app's
own wellness scores, and `Help_ThriveScore` already claims *"how to improve or
increase a score"* — so two live specs claimed the same utterance, neither named
the other, and generation would have produced the same sentence under two labels.

Nothing would have caught it. Dedup is inline and scoped `within_intent`; Stage 2,
which exists to find cross-intent collisions, is not built; and the pair scores
**0.039** against `spec_review.py`'s 0.20 threshold. It was found only because
Akash asked what the Fallback edit had done to generation.

Narrowed to a **clinical reading** — heart rate and the like — with the sentence
"An app score is not a clinical reading" added so the distinction is explicit.
D11's decision is unchanged; only its reach is. `Help_ThriveScore` was not
touched, and is in E7's unsupported set in any case.

**Nothing to close.** The strict reading was chosen over two softer options.

---

### D12. `Help_SelfCheck` carries fault reports, and generation must match that

Akash, 2026-08-27. The app **does** have a self-check feature, but only
`Help_SelfCheck` is supported — no voice command runs it and none is planned.

That makes it the second-most command-shaped intent in the taxonomy. Of its 110
deployed rows, **27 (24.5%) are command-shaped**, behind only
`Help_FindMyHearingAids` at 40.6%. Opening those 27 by shape shows they are not a
linter artefact: 7 carry an action verb (`check`, `test`, `fix`) and **20 are
genuine fault reports** — their vocabulary is *not*, *work*, *problem*,
*doesn't*, *left*, *right*, *respond*, *diagnose*.

The spec already listed fault reports as triggers, so the routing was right. What
was missing was the reason and the generation instruction. Added as a boundary
case, and worded for THIS case rather than copied — `Help_FindMyHearingAids` says
the assistant *cannot* do the thing, which is not true here. Here the app can, but
no voice command drives it, so explaining is still the action the assistant
performs and precedence rule 4 covers the direct phrasing.

The generation half matters as much: **a generated command-shaped rate near zero
is a defect, not a success.** Without that line, generation produces how-to
phrasing only and the corpus misses a quarter of how users actually speak to this
intent.

**Nothing to close.**

### D13. Two more boundaries that existed in prose but not as links

Same shape as D9, found in the HelpDeviceCare review.

`Help_CleanCare` ↔ `Help_SelfCheck` — wax-attributed trouble versus unexplained
device failure. This family's subtlest distinction, stated in `do_not_trigger` on
**both** sides, with neither listing the other as a neighbour. Symmetric absence,
so the one-directional check passed; below the 0.20 threshold, so no tier reported
it. Made mutual.

`Help_WiCROS` → `Help_Volume` was one-way. `Help_WiCROS` sends general volume
questions to `Help_Volume`, and `Help_Volume` never mentioned CROS, WiCROS or the
balance control at all — so the route had no destination that knew about it.
`Help_Volume` now names the balance control and yields it back, and the pair is
mutual. Balance and volume both change loudness, which is exactly why the boundary
needs stating.

`Help_WiCROS` → `Help_ChangingMemories` is the same shape and was left alone —
`Help_ChangingMemories` has not been reviewed. Logged in E4.

**Nothing to close** for these two pairs.

### D14. The whole HelpConnectivity family was written as how-to and is not spoken that way

Reviewed 2026-08-27. Every trigger in all three specs is a question form — 4 of 6,
4 of 7 and 3 of 5 literally begin *"User asks how…"* — while the deployed speech
is substantially direct requests.

| intent | deployed | command-shaped | what the flagged rows are |
|---|---:|---:|---|
| `Help_Pairing` | 224 | **66 (29.5%)** | 47 carry `connect`, `pair`, `sync`, `link`, `unpair`, `disconnect`; the other 19 still name pairing |
| `Help_RemoteProgramming` | 127 | 16 (12.6%) | 11 are stated needs — `need`, `adjustment`, `talk`, `audiologist` |
| `Help_HearShare` | 72 | 6 (8.3%) | `accept`, `invitation` |

`Help_Pairing` is the third most command-shaped intent in the taxonomy and is
**more command-shaped than help-shaped** — 29.5% against 24.1%. As checked in the
`Help_IntelliVoice` and `Help_SelfCheck` rounds, these are not linter artefacts.

**For `Help_Pairing` the routing was already decided, on the wrong spec.**
`Cmd.StreamingStart` — reviewed in the Streaming round — says outright
*"Requests to pair hearing aids to a phone over Bluetooth, which are
Help_Pairing"*, and distinguishes it from the accessory case, where *"connecting
is done by hand, not by voice — a direct request is Default Fallback Intent"*.
So the taxonomy knew that a direct pairing request belongs here, and the intent
that receives them did not. The one-way route again, this time on the highest
-volume instance of it.

Akash, 2026-08-27, for the other two: the assistant does **not** submit an
adjustment request and does **not** accept a HearShare invitation. It explains
how, for both.

All three now carry the same carve-out, worded per case, and each states its
measured rate so generation reproduces it. Without that line the generator writes
how-to phrasing only, and for `Help_Pairing` that would miss the larger half of
how users actually speak to it.

**Nothing to close.**

### D15. Three prose-only links in HelpConnectivity, all fixable within reach

`Help_Pairing` ↔ `Help_HearShare`, `Help_HearShare` ↔ `Help_Health` and
`Help_RemoteProgramming` ↔ `Help_Customize` each had a boundary in prose and no
neighbour link. All four partners are reviewed, so all three were made mutual
rather than logged.

`Help_HearShare` and `Help_RemoteProgramming` had only two and three neighbours
respectively — the thinnest in the taxonomy after `Help_WiCROS`, and both sit on
the sharing-versus-clinician boundary, which is a real confusion.

**Nothing to close.**

## E. Not started

### E1. All 33 `Help*` intents

Four were read as counterparts during `Cmd.*` reviews — `Help_Battery`,
`Help_FindMyHearingAids`, `Help_ChangingMemories`, `Help_VoiceAssistant` — and
two of those were edited. None was reviewed in its own right, and reading a spec
against one partner is not the same as reading it against its own family.

Remaining: HelpSpeechServices 2 (deferred with A2), HelpFind 1, plus
`Help_VoiceAssistant`, `Help_ChangingMemories`, `Help_MemoryOptions` and
`Help_Reminder` from mixed families. Done: HelpAppSettings, HelpHealth and
HelpDeviceCare 6 of 6 each, HelpConnectivity 3 of 3, HelpAudio 4 of 5.

**One finding is already open against two of them.** `spec_review.py` Section 2a
— the highest-priority tier, where neither spec names the other and neither so
much as mentions its subject — now reports exactly one pair:

    Help_Health <-> Help_Home    score 0.20

Both are "explain a screen" intents and the shared vocabulary is real: *screen*,
*app*, *where*, *what*, *how*. The risk case is a question that names a health
figure without naming the Health screen — "where do I see my hearing in the
app?" — which `Help_Home` claims as a broad orientation question and
`Help_Health` claims as "where in the app a particular health figure can be
seen". `Help_Home` does yield generically ("questions about a specific feature
belong to that feature's intent") but never names `Help_Health`, and the
generator sees one spec at a time, so it cannot resolve "that feature's intent"
to a sibling it has not been shown. `Help_Health` has no reciprocal line at all.

Two honest qualifications, both verified rather than assumed:

- The pair is **not** a regression from the `Help_Tinnitus` round. Neither spec
  was edited. The score moved 0.1937 → 0.2007 because adding text elsewhere in
  the corpus shifted the IDF weights, and 0.20 is the reporting threshold. It was
  always a borderline pair; it is now merely visible.
- It is Help-to-Help, so the cost of getting it wrong is a wrong explanation,
  not a device action. Zero False-Accept-Rate impact. That is why it is recorded
  here rather than fixed in the middle of another family's round.

It twice drifted across the 0.20 reporting threshold on edits made to other
specs entirely — 0.1937, then 0.2007, then 0.1999 — appearing and disappearing
from Section 2a without either spec being touched. Worth remembering as what a
hard cut on a continuous score does at its boundary.

**CLOSED 2026-08-26**, in the HelpAppSettings round, on both sides and with a
neighbour link, so it cannot drift back. `Help_Home` excludes questions about
where a health figure can be seen; `Help_Health` excludes Home-screen and broad
app questions. The larger cause was removed at the same time — see D8.

### E2. `reminders.add` and `reminders.complete`

Never reviewed as specs. `Default Fallback Intent` was the third and has now been
done — see D7.

Both were read *around* during the Fallback review, which surfaced two things
worth carrying into their own round rather than acting on now:

- `reminders.add`'s boundary case — "the subject never changes the intent" — is
  now relied on by a Fallback exclusion. It was right, but it is still a rule in
  an unreviewed spec that another spec has started leaning on.
- `reminders.complete` routes delete, cancel and postpone to Fallback as
  unsupported actions. Fallback covers that only through the generic "requests
  for a capability the product does not have"; it never names reminders. Another
  one-way route, not yet checked from the Fallback side.

---

### E3. The memory-name collision set was never derived from the entity list

Seven guards exist, each saying a word is "both a memory name and this intent's
subject", so an explicit request to change memory or program is
`Cmd.MemoryChange` and anything else is the owning intent:

| Intent | Memory name |
|---|---|
| `Cmd.EdgeModeIncrease` | `Speech`, `Noise` |
| `Cmd.StreamingStart` | `Television` |
| `Cmd.VolumeMute` | `Mute` |
| `Help_MaskMode` | `Mask` |
| `Help_Pairing` | `Telephone` |
| `Help_Tinnitus` | `Tinnitus` |

The `Help_Tinnitus` round recorded this as a finished set — "seven memory names,
seven guards". **That claim was not verified and should not have been made.** The
seven are the ones that happened to come up in conversation; nobody compared them
against the memory names the runtime actually recognises.

That list exists: `language_packs/en/nlu_entities.json`, under `memory.values`,
38 entries. Scanned against every spec's own subject — business description plus
trigger conditions — it gives **30 name/intent overlaps, 17 of them unguarded**.

Most of the 17 are ordinary English rather than defects: `Work` matches
`Help_SelfCheck` only through "working", `Speech` matches `Help_Transcribe`
through speech-to-text. A handful look real:

    Home       -> Help_Home              the Home SCREEN vs the Home memory
    Mute       -> Help_Volume            Help_Volume explains muting; Mute is a memory
    Noise      -> Help_EdgeMode          noise reduction vs the Noise memory
    Quiet      -> Cmd.VolumeIncrease     "make it quieter" vs switching to Quiet
    Telephone  -> Cmd.FindMyPhone        the phone vs the Telephone memory
    Meeting    -> Cmd.TranscribeStart    transcribing a meeting vs the Meeting memory

`Help_Volume` and `Cmd.VolumeIncrease` are already reviewed and signed nothing;
if `Mute` and `Quiet` turn out to be real, two closed rounds reopen.

**To close:** run the scan, decide each of the 17 individually, and either add a
guard or record why the overlap is harmless. Only then is the set complete.

---

### E4. Prose-only neighbour links are a taxonomy-wide pattern, counted only here

Counted properly for the first time in the HelpAppSettings review, and again
across HelpHealth. **Both counts were far too small.** An audit on 2026-08-27
scanned all 60 specs rather than the `Help*` ones and found **59 prose-only
routes across 25 specs**, against the 14 rows tabled below. Each row is a boundary a spec states in prose while the sampling link it
needs does not exist:

| Intent | Named in a rule, but not a neighbour | Neighbour, but named in no rule |
|---|---|---|
| `Help_DeviceSettings` | `Cmd.StreamingStart`, `Help_ChangingMemories`, `Help_Pairing` | `Help_WiCROS` |
| `Help_DemoMode` | `Help_Pairing` | `Help_Home` |
| `Help_Customize` | — | `Cmd.MemoryChange`, `Help_AppSettings` |
| `Help_AppSettings` | — | `Help_DemoMode` |
| `Help_WhatsNew` | — | `Help_DemoMode` |
| `Help_Home` | — | `Help_Reminder`, `Help_VoiceAssistant` |
| `Help_Activity` | `Help_HeartRate` | `Help_ThriveScore` |
| `Help_Health` | — | `Cmd.ActivityStep`, `Help_FallAlert` |
| `Help_ThriveScore` | — | `Help_Activity`, `Help_HeartRateRecovery` |
| `Help_FallAlert` | `Help_FindMyHearingAids` | — |
| `Help_InsertDevice` | `Help_FindMyHearingAids` | — |
| `Help_WiCROS` | `Help_ChangingMemories` | `Help_DeviceSettings` |
| `Help_SelfCheck` | — | `Help_Battery`, `Help_FallAlert`, `Help_InsertDevice` |
| `Help_Accessories` | — | `Help_Volume` |

`Help_DeviceSettings` is the worst of them: it names six intents in its rules and
lists four as neighbours, and only one of those four appears in any rule.

Only `Help_DeviceSettings` ↔ `Help_Customize` was fixed (D9), because both specs
sat in the same family. The HelpHealth rows were counted and left alone for the
same reason the rest were. The rest reach into `Help_Pairing`, `Help_ChangingMemories`
and `Cmd.StreamingStart`, which have not been reviewed, and fixing a link
requires editing both sides.

Neither direction is automatically wrong. A neighbour named in no rule may be a
sensible confusion nobody wrote down; a rule with no link may be a boundary too
coarse to need hard negatives. What is wrong is that no one has decided.

**To close:** run the same count across all 60 specs, decide each row, and either
add the link, add the rule, or record why neither is needed. Same shape as E3 —
a systematic pass, not a per-family patch.

### E5. Five generic activity queries were dropped with `Cmd.Health` and never revisited

Not a defect — a documented decision with a documented loose end. `Cmd.Health` is
dropped in `generator_config.yaml` because it is a rollup PARENT: 155 of its 160
unique utterances appear verbatim under the `Cmd.Activity*` children, so keeping
it would train the classifier to split identical text between a parent and a
child. That reasoning is sound and is written down.

The config also records what the decision costs, and that part is not recorded
anywhere a reviewer would look:

> 5 utterances exist only under `Cmd.Health` and are dropped with it … Revisit if
> generic activity queries need coverage.

All five are duration or generic-progress questions. Checked how exposed that
leaves the taxonomy: duration phrasing does appear in the deployed
`Cmd.Activity*` rows — 40 rows across six of the eight — so a duration question
that NAMES its activity is covered. What has no home is a duration question that
names an activity the product does not track, or none at all.

Two related things confirmed while checking, both fine: `Cmd.Health`'s seed file
contains **zero** heart-rate rows, so dropping it does not affect
`Help_HeartRate`; and `Help.Activity`, the dot-variant file, is merged into
`Help_Activity` by config rather than orphaned.

**To close:** decide whether a generic activity query needs an intent, or record
that unnamed-activity questions are `Default Fallback Intent` and say so in a
spec. Today no spec mentions the case at all.

### E6. `Help_Activity` has no deployed rows and cannot be evaluated

`Help_Activity` is one of the four intents in the 60-vs-57 runtime delta, and the
only one of the four that is not an EdgeMode command. It has **0 rows in
`train.csv`** and 0 in `dev_hard`, against 26 seed utterances.

It was reviewed here on its spec and its seeds alone. Every other intent in this
family could be checked against deployed speech; this one could not, so the
command-shaped measurement that has caught three defects so far is simply
unavailable for it.

This is the instrument gap the architecture doc already records, seen from the
review side rather than the evaluation side.

**To close:** it closes when the instrument gap does — see F.

### E7. Unsupported intents — decided to drop, NOT yet applied

Akash, 2026-08-27. `Help_HeartRate` and `Help_HeartRateRecovery` are disabled in
Dialogflow; `Help_ThriveScore` will not be supported either. Decision is to drop
all three from the taxonomy the way `Help_HearingCareAnywhereConnect` already is,
taking it **60 → 57**. Nothing has been applied. This entry is the plan and the
cost, so the decision can be made against both.

That 57 is **not** the runtime label map's 57. All three are still in
`nlu_schema.json`, so the drop widens the delta rather than closing it.

**`Help_HearingCareAnywhereConnect` needs nothing.** It is already dropped in
`generator_config.yaml` and has never been in the 60. No round has touched it.

**The other two are a different case, and this is the part worth reading before
applying.** `Help_HearingCareAnywhereConnect` was gone from the runtime as well.
These two are **still in the shipping label map** — both appear in
`language_packs/en/nlu_schema.json` — and they still carry deployed data.

| | `train.csv` | `dev_hard.csv` | seeds | in `nlu_schema.json` |
|---|---:|---:|---:|:-:|
| `Help_HeartRate` | 51 | 7 | 19 | yes |
| `Help_HeartRateRecovery` | 46 | 7 | 22 | yes |
| `Help_ThriveScore` | 105 | 13 | 65 | yes |

**The cost lands on the measurement, and it stops being small.** The architecture
doc records that `dev_hard` carries 7 rows of `Help_HearingCareAnywhereConnect`
which this taxonomy drops, so a model following the taxonomy is marked wrong on
each — a bias, not noise, pointing against the change the Super Dataset exists to
make. It calls that 7 of 813 "under a point and well inside the 0.038 MDE".

These three add **27 more rows of exactly that kind**:

    7 of 813   =  0.9%   today            well inside the 0.038 MDE
    34 of 813  =  4.2%   after the drop   LARGER than the MDE

That crosses the line the doc drew. At 4.2% the instrument gap is no longer a
footnote on a Super Dataset result — it is bigger than the effect the experiment
is powered to detect, and it points one way. The doc already requires an explicit
decision before any result is reported (exclude those rows and re-state the
0.8327 baseline, or accept and record). **After this drop that decision cannot be
deferred at all.**

**What applying it takes:**

    generator_config.yaml   add all three to taxonomy.drop_intents with a reason
                            remove all three from families.HelpHealth  (6 -> 3)
                            remove all three from taxonomy.hand_authored_intents
    authored_specs.yaml     delete the three spec blocks
    intent_specs.yaml       regenerate  (60 -> 57)
    Help_Health             remove 3 neighbours, rewrite 2 do_not_trigger and
                            1 boundary case that name them
    Help_Activity           remove 1 neighbour, rewrite 1 do_not_trigger
    Default Fallback Intent state the unsupported set explicitly -- see below
    verify_round.py         checks 50-58 assert edits to two of these specs and
                            would need removing with them

HelpHealth is left with `Help_Activity`, `Help_Health` and `Help_FallAlert`.

Their seed files stay in the folder untouched, as
`Help_HearingCareAnywhereConnect`'s does.

**Fallback must be updated in the SAME change, and not before it.** With the three
gone, nothing in the taxonomy covers heart rate, heart rate recovery or the Thrive
scores. Those questions reach `Default Fallback Intent` — correct once the
features are unsupported, but it has to be *stated in a spec* rather than happening
by absence, exactly as C1 and C2 state the messaging and power-on/off gaps.

The ordering matters and is not cosmetic. Writing those subjects into Fallback
while the three intents are still live would make Fallback and each of them claim
the same utterances — a new cross-intent collision, and the same mistake D11
already made once (below). Drop first, or drop and state in one change; never
state first.

`Help_Health` and `Help_Activity` currently route these subjects to the three by
name. After the drop those routes have no destination, which is the one-way-route
shape this review has now found five times — so they must be rewritten in the
same change, not left to be discovered.

**Already done and deliberately kept** (Akash, 2026-08-27): the HelpHealth round
edited both specs before the disable was known — D10 and D11. Those edits stand.
They removed a claimed capability the product does not have and stopped the
taxonomy offering health advice, so the specs are more truthful than they were; if
the intents are ever re-enabled the correct spec is waiting. No further review
work will be done on them.

**To close:** apply the plan above, or record that the two stay in the taxonomy
while disabled at runtime and say why.

### E8. Defects the audit found INSIDE families already signed as reviewed

Found 2026-08-27 by auditing this review's own output, not by the review itself.
Each sits in a family the standings table calls done, so each is a hole in a
sign-off rather than a new area of work.

**`Help_Activity` versus the eight `Cmd.Activity*` intents — three specs, one
utterance, no guard.** (HelpHealth, marked 6 of 6.)

    Help_Activity trigger  "...or where to see distance for a tracked activity"
    all 8 Cmd.Activity*    "locating the screen is Help_Health"
    Help_Health trigger    "User asks where in the app a particular health figure can be seen"

`Cmd.ActivityCycle`'s own `hard_negative_example` is *"Where can I see the
distance that I biked?"* — an utterance its spec sends to `Help_Health` and
`Help_Activity` claims outright. `Help_Activity` then contradicts its own trigger
in its boundary cases — *"Where the question is about finding a screen rather
than changing a goal, prefer Help_Health"*.

**`Help_SelfCheck` contradicts itself.** (HelpDeviceCare, 6 of 6.)

    trigger   "User reports that an aid does not work, IS FAINT, or has a problem."
    boundary  "A complaint that sound is too quiet is a volume request, not a fault report."

"My left one is faint" carries no request and no not-working claim. The trigger
takes it; the boundary case refuses it.

**`Help_Volume` ↔ `Help_Pairing` both claim no-sound-from-the-phone.** Both
reviewed. Neither names the other and they are not neighbours.

    Help_Volume   "...such as getting no sound or the volume dropping to nothing"
    Help_Pairing  "User reports that audio is not coming through the aids from the phone"

**Seven of the eight `Cmd.Activity*` intents route to `Help_Health` with no
neighbour link** — a family marked 8 of 8 reviewed, and the largest single block
of the E4 pattern.

**To close:** reopen HelpHealth, HelpDeviceCare and the ActivityTracking `Cmd.*`
family for these specific items. They are not full re-reviews.

### E9. `generator_config.yaml` and the specs disagree, in two places

**`command_help_pairs` declares a messaging pair that C1 says does not exist.**

    generator_config.yaml   Cmd.SendMessage:   Help_VoiceAssistant
                            Cmd.ListenMessage: Help_VoiceAssistant

against `Cmd.SendMessage`, `Cmd.ListenMessage` and `Help_VoiceAssistant`, which
all state that the taxonomy has no messaging Help intent and send how-to
questions to `Default Fallback Intent`. C1 records that decision as *"stated in
four places"* and never checked the config, which asserts the opposite. Worse,
`verify_round.py` asserts all 23 pairs are mutual neighbours, so the contradiction
is held in place by a passing test.

`Cmd.FindMyPhone: Help_FindMyHearingAids` was removed from this table earlier in
the review for being a false pair; nobody then re-read the rest of the table.

**Provenance counts are stale.** `authored_specs.yaml`'s header says *"The
remaining 56 were drafted in an assistant session"* and `generator_config.yaml`
says *"All 57 are currently listed"*, while both files hold **60**.
`intent_specs.yaml`'s `meta` is the only one right, at 59 assistant-session plus 1
human.

**To close:** decide whether the messaging pairs stay (and C1 is wrong) or go (and
the config is wrong); correct the two counts either way.

### E10. 81 passing checks coexisted with 15 real defects

The audit that produced E8, E9 and the corrections in D-section is itself the
finding. `verify_round.py` had 76 checks, all green, while the specs carried two
false rankings, a false absolute claim, a self-contradicting route, an unguarded
collision, and five stale numbers in this file.

Every one of those checks asserted **that an intended edit had been made**. None
asserted **that what was written was true**. Those are different questions, and
only the second one protects the generated corpus.

Two checks now close part of that gap — 74 re-derives every command-shaped
percentage a spec asserts, from `train.csv`, and fails naming the intent and both
numbers; 71 forbids any spec claiming its deployed rows are "entirely" one shape.
Both were mutation-tested.

What is still unchecked, and is how E8 was found rather than caught: no test
verifies that an intent named as a destination actually claims the subject sent
to it, or that two specs do not claim the same subject.

**To close:** add those two checks. They are the whole one-way-route and
collision bug class, which is the class this review has hit most.

## F. Outside the spec review

These are pipeline-level and already documented in
`docs/Prod-Work-Documentation/nlu_super_dataset_architecture.md`. Listed here
only so that a reader of this file is not misled into thinking Stage 0 is the
last thing standing.

- **Stage 2 (cross-intent collision) is not built.** Deduplication is inline and
  scoped `within_intent`; nothing measures collision between intents in
  generated data.
- **Stage 3 (hard negatives) is not built.** `hard_negatives_per_intent: 40` and
  `oos_ratio: 0.15` are in `generator_config.yaml` and referenced by no code.
- **Tier-2 sealed holdout does not exist.** Never blocked by the generator.
- **The 60-vs-57 instrument gap.** `dev_hard` cannot score four intents and
  carries seven rows of one this taxonomy drops.
- **Length/difficulty tilt.** The generator over-produces long rows; the fix was
  never actually tested, because the instruction meant to fix it was written into
  `prompt.txt` rather than `SYSTEM_PROMPT` and was never sent. A `--pilot` run
  with the corrected prompt has not yet been done.
- **`spec_review.py` is structurally blind to `Default Fallback Intent`.** Its
  pairwise scores against the other 59 intents sit far below the 0.20 reporting
  threshold, because its vocabulary is a grab-bag — weather, sport, television,
  greetings, ASR noise, medical — so TF-IDF cosine dilutes no matter what the
  spec claims. Measured before the D11 correction its highest score was **0.099**,
  half the threshold, against 35 of 60 intents that reach it with some partner.

  **The number moved, and how it moved is the point.** The D11 correction added
  the words "a clinical reading such as a heart rate" to a Fallback trigger, and
  that alone took Fallback ↔ `Help_HeartRate` from below 0.1 to **0.183** —
  within 0.017 of being reported. So "Fallback can never reach the threshold" was
  too strong: three added words nearly got it there. What holds is the weaker and
  more useful statement, that Fallback's score is driven by how much unusual
  vocabulary a single edit adds rather than by how much ground the spec claims,
  so the tool reports it late or not at all.

  That is the wrong intent to be blind to. Fallback is named by 59 specs, is
  edited in almost every round, and is where every boundary decision lands. The
  D11 collision sat at 0.039 and would never have been reported; the pair D11's
  fix then created sat at 0.183 and was not reported either, and was found by
  audit rather than by the tool. Any change to Fallback has to be checked by
  reading.
- **`spec_review.py` reads specs, never seeds.** Found during the `Help_MaskMode`
  round. `Help_Tinnitus` and `Help_MaskMode` overlap heavily in seed vocabulary —
  50 of 130 tinnitus rows carry `masker`/`masking`, against 18 of 18 mask-mode
  rows carrying `mask` — while the two specs score **0.056** against each other
  and never approach any reporting tier. The tool cannot see a collision that
  lives in the data rather than in the prose, and nothing else currently looks.
  Worth knowing before a clean `SPEC_REVIEW.md` is read as "no collisions".
- **Seed files are UTF-16 (61 of 68) and must be read through `seed_loader.py`.**
  A direct `read_text()` on `Help_MaskMode.txt` yields 37 junk lines averaging
  3.2 words where the file holds 18 real ones averaging 5.5, and every count
  derived from it is then wrong in a way that still looks plausible.
  `seed_loader.decode_seed_file` already tries `utf-8-sig`, `utf-16`,
  `utf-16-le`, `utf-16-be` in order, so the pipeline is correct — ad-hoc analysis
  scripts are where this bites.
- **`boundary_lint.py` over-counts its own baseline.** Found during the
  `Help_IntelliVoice` review. `HELP_VERB` matches `guide me` but not a bare
  `guide`, so a polite request for a user guide is scored command-shaped rather
  than explain-request. Across the deployed data this is **18 of the 279
  command-shaped flags on `Help*` intents in the taxonomy — 6.5%** — but it concentrates badly:
  5 of `Help_IntelliVoice`'s 6 flags are this one artefact, which is what made
  that intent briefly look like it carried command traffic when its rows are
  entirely question-shaped.

  It matters because the linter's gate is *relative*: generated rows are
  compared against the deployed rate per intent, so an inflated baseline raises
  the bar a real regression has to clear before it fails. The fix is one word in
  `HELP_VERB`. Left alone here on the one-thing-per-change rule; it is a change
  to an instrument, not to a spec, and it should be made and re-baselined on its
  own.

---

## What this file is not

It is not a backlog to be groomed, and it is not a substitute for the sign-off
table. It is a record of the specific things this review chose not to look at, so
that the next person can tell "we decided against it" from "nobody got to it" —
and so that a green `SPEC_REVIEW.md` is never mistaken for a finished review.

Add to it whenever something is deferred. Delete an item only when it is closed,
and say in the commit message how it was closed.
