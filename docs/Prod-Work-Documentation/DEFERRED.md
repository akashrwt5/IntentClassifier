# Deferred and open — Stage 0 spec review

Things that were consciously put down, and things nobody has answered yet. They
live here because a decision deferred in conversation is a decision lost: it
leaves no trace in the specs, no flag in `SPEC_REVIEW.md`, and no row in any
report. The per-family sign-off table in `SPEC_REVIEW.md` records what has been
*reviewed*; this file records what has been *skipped*, and why.

Every item states what closing it needs. An item with no closing condition is a
wish, not a task.

Last updated 2026-08-30, after the `Cmd.*` review, the HelpAudio,
all Help families, `Default Fallback Intent`, the Reminders intents, an
audit of everything above, and the EdgeMode and SpeechServices families.
**60 of 60 reviewed. Every intent in the taxonomy has now been read.**

**Signed off 2026-08-30 by Akash Rawat, all 18 families.** Stage 0 is closed.
E7 applied the same day, taking the taxonomy **60 → 57**. See D19 for what
sign-off does and does not claim and D20 for the drop.

---

## Where the review actually stands

| | Count | State |
|---|---:|---|
| `Cmd.*` reviewed | 24 of 24 | AudioControl 4, Streaming 2, Messaging 2, Memories 1, DeviceStatus 1, DeviceLocate 1, ActivityTracking 8, EdgeMode 3, SpeechServices 2 |
| `Cmd.*` deferred | 0 | — |
| `Help*` deferred | 0 | — |
| `Help*` reviewed | 33 of 33 | Every Help family |
| Other | 3 of 3 | `Default Fallback Intent`, `reminders.add`, `reminders.complete` |

`Default Fallback Intent` has now been read end to end (2026-08-26). It had been
edited in almost every round without ever being reviewed itself — the most rules
of any intent and the least scrutiny, which was the wrong way round. Its 23 rules
held up; what did not was its neighbour list. See D7.

**All 18 sign-off boxes are ticked and `intent_specs.yaml` carries a
`meta.sign_off` record instead of `REQUIRES HUMAN REVIEW`.** The ticks live in
`spec_review.py`'s `SIGNED_OFF`, not in the generated Markdown — see D19.

---

## A. Deferred intent reviews

### A1. EdgeMode — `Cmd.EdgeModeIncrease`, `Cmd.EdgeModeDecrease`, `Cmd.EdgeModeDeactivate`, `Help_EdgeMode` — CLOSED

**Closed 2026-08-28. See D17 for what the family review found and decided.** Two
of the three queued edits below were applied; edit 2 was declined by Akash and is
now a standing gap, recorded in D17 and pinned by check 98. The text below is
left as written so the reasoning that produced the deferral stays readable.

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

**To close:** ~~read all four specs against each other; confirm the five edits
above still hold when the family is read as a whole; apply the three queued
edits; and check the four cross-references from `Help_IntelliVoice` and
`Help_MaskMode` from this side.~~ Done 2026-08-28, except edit 2 — declined, see
D17.

### A2. SpeechServices — `Cmd.TranscribeStart`, `Cmd.TranslationStart` — CLOSED

**Closed 2026-08-28. See D18.** The known dependency below was checked from the
destination and holds. The text is left as written.

Deferred by decision. Neither spec has been read.

One known dependency: `Cmd.ListenMessage`'s boundary rests on
`Cmd.TranscribeStart` ("requests to start live transcription ... are
Cmd.TranscribeStart"). That rule was accepted during the Messaging review
without reading the intent it points at.

**To close:** ~~read both specs, plus `Help_Transcribe` and `Help_Translate`; check
the `Cmd.ListenMessage` cross-reference from the other side.~~ Done 2026-08-28.

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

### D16. The last seven intents

Reviewed 2026-08-27, closing every family the deferrals do not hold.

**`Help_MemoryOptions` gets the carve-out its speech demands** (Akash). 15 of its
83 deployed rows are command-shaped — 18.1%, fourth highest of any Help intent —
and 8 of the 15 carry `add` or `create`. Checked before writing it: **no `Cmd.*`
intent creates a memory**; `Cmd.MemoryChange` only switches between memories that
already exist. So explaining is the action, and a direct request to add or save a
memory belongs here. The spec half-knew this already — its boundary case said *"a
request to add a named mode as a saved memory resolves here"* and its own
hard-negative example is a bare imperative — but it never said why, and never
told generation to reproduce the rate. Both now stated. That is the fifth intent
in this review to need it, after `Help_FindMyHearingAids`, `Help_SelfCheck`,
`Help_Pairing` and `Help_HearShare`.

**`Help_ChangingMemories` ↔ `Help_MemoryOptions` on memory availability.** One
claimed *"whether they have a particular memory available"*, the other *"an
expected memory is missing from their list"*, and neither exclusion covered the
other. Split along the line the two specs already use everywhere else — asking
whether a memory is available precedes a SWITCH and stays with
`Help_ChangingMemories`; reporting that one has gone is MANAGEMENT and is
`Help_MemoryOptions`. Stated on both sides.

**`Help_VoiceAssistant`'s last vague cross-reference** — *"transcription or
translation, which have their own intents"* — now names `Help_Transcribe` and
`Help_Translate`. Its C1 rules were read closely and hold; the whole
messaging-has-no-Help-intent decision rests on this spec and it says so correctly
in three places.

**`reminders.complete` had a one-way route.** It sends delete, cancel and postpone
to Fallback as unsupported actions, and Fallback covered them only through the
generic *"a capability the product does not have"*, never naming reminders. Now
stated on the Fallback side, closing the second of E2's two carry-forwards. The
first — `reminders.add`'s *"the subject never changes the intent"*, which a
Fallback exclusion now leans on — was read and is correct.

**Two prose-only links made mutual**: `Help_FindMyHearingAids` ↔ `Help_Pairing`
(lost versus unconnected, the distinction that spec calls out itself) and
`Help_ChangingMemories` ↔ `Help_Customize`.

`Help_FindMyHearingAids` needed nothing else. Its 40.6% carve-out is the model the
other four were written from, and it reads correctly.

**Nothing to close.**

### D17. The EdgeMode family

Reviewed 2026-08-28, closing A1. All four specs read together, the five blind
edits re-checked against the family as a whole, and the four cross-references
from `Help_IntelliVoice` and `Help_MaskMode` checked from this side.

**This family has no deployed rows to check against.** All three `Cmd.EdgeMode*`
intents are in the runtime delta (57 vs 57, overlapping on 53) — they are absent from
`language_packs/en/nlu_schema.json`, so the shipped model cannot emit them and
`dev_hard` holds 0 rows for all three (`Help_EdgeMode` has 9). The
command-shaped-rate check that caught three defects in other families cannot run
here. Everything below rests on the seed corpus instead, which is smaller and is
label-noise from the same source.

**A one-way route, and the checker could not see it.** `Cmd.EdgeModeDeactivate`
sent *"requests to switch to a different saved memory or program"* to
`Help_ChangingMemories`, whose own first exclusion says *"any actual request to
switch memory now, which is Cmd.MemoryChange"*. The destination denies the
subject. Section 7 passed it because the two share `switch`, `memory` and
`program` — the destination is the right SUBJECT and the wrong SIDE of the
Command/Help split, which is a semantic distinction and outside what word overlap
can see. Now routed to `Cmd.MemoryChange`, with the Help intent named for the
how-to reading.

**"Put it back to normal" — two intents claimed it, Fallback gets it** (Akash).
`Cmd.EdgeModeDeactivate` triggered on *"return to normal listening"*,
`Cmd.MemoryChange` on *"return to normal, default or automatic"*, neither named
the other and they were not neighbours. This was `SPEC_REVIEW.md` Section 8's
pinned `Cmd.EdgeModeDeactivate vs Cmd.MemoryChange` flag.

The evidence is thin in every direction and worth recording rather than hiding:
0 of `Cmd.EdgeModeDeactivate`'s 16 seeds and 0 of `Cmd.MemoryChange`'s 58 carry
`normal`, `default`, `automatic` or `everyday`; 5 of Fallback's 613 do. In
`dev_hard`, exactly 1 row carries a bare `normal` with no memory word and it is
labelled `Cmd.MemoryChange`. In the generated master file, bare-normal rows are
spread across five intents. **Nothing settles it**, which is itself the argument
Akash took: two intents can each perform the action, nothing in the words
chooses, so the tie goes to inaction. Bare *"put it back to normal"* is Fallback;
a named memory is `Cmd.MemoryChange`; Edge Mode named or in context is
`Cmd.EdgeModeDeactivate`. Stated in all three specs. The two commands now name
each other, which closes the Section 8 flag — 7 collisions to 6.

**Three specs disagreed on a bare environment observation — Fallback wins**
(Akash). `Cmd.EdgeModeIncrease` said *"pure observations about the environment
with no implied request ('it's noisy in here') remain Fallback"*, while
Fallback's own exclusion sends *"an observation naming a LISTENING ENVIRONMENT
for which the product has a memory"* to `Cmd.MemoryChange` — and `Noise`,
`Restaurant`, `Crowd` and `Outdoors` are all real memory names in
`nlu_entities.json`. So Fallback's rule as written claimed the very sentence
`Cmd.EdgeModeIncrease` was handing back to it.

Resolved by narrowing the exception to a named PLACE: the memories are
environment-scoped, so *"I'm outdoors now"* is an implicit switch, but a remark
on how the sound is where the user already is names no place and asks for
nothing. Stated identically in `Cmd.EdgeModeIncrease`, `Cmd.MemoryChange` and
`Default Fallback Intent`.

**The seed corpus disagrees with this decision and that is on the record.** By
shape and keyword — `boundary_lint` neutral, an environment word, no request verb
and no difficulty word — the bare-observation shape sits 14 rows in
`Cmd.EdgeModeIncrease`, 6 in `Default Fallback Intent` and 0 in
`Cmd.MemoryChange`. The deployed labellers put most of them in the command. This
is a keyword-and-shape probe rather than a reading, so it is evidence and not
proof, but it points the other way from the decision and should be revisited if
`Cmd.EdgeModeIncrease` recall comes out low.

**A mirror-image gap between the two direction intents.**
`Cmd.EdgeModeDecrease` triggers on *"decrease the comfort or communication aspect
of Edge Mode"*; `Cmd.EdgeModeIncrease` never named that axis at all, while 20 of
its 152 seeds carry `comfort` or `communication` and 8 of those carry an increase
word. Added. This is exactly the asymmetry A1 warned about — five edits made to
`Cmd.EdgeModeIncrease` from other families' rounds, none of them checked against
`Cmd.EdgeModeDecrease`, which has to share the same trigger surface reversed.

**Queued edit 1 applied, edit 3 applied, edit 2 DECLINED** (Akash).

- Edit 1. `Cmd.EdgeModeIncrease` now names `Help_IntelliVoice` and
  `Help_MaskMode` and says why acting on either would apply Edge Mode instead.
- Edit 3. `Cmd.EdgeModeDeactivate` ↔ `Help_MaskMode` named on both sides and made
  mutual. This was the last pair standing in `SPEC_REVIEW.md` Section 2b, which
  is now empty — 2a, 2b and 2c are all clear for the first time.
- Edit 2. **Declined.** `Help_IntelliVoice` and `Help_MaskMode` still name
  `Cmd.EdgeModeIncrease` in `do_not_trigger` without listing it in
  `neighbor_intents`. **Consequence, stated plainly: that boundary never renders
  into the Stage 1 prompt as a "most likely confusion" and is never sampled for
  hard negatives.** It is prose the generator sees for one spec at a time and
  nothing else. Pinned by check 98 so it stays a decision rather than becoming an
  omission nobody noticed. `spec_review.py` cannot see it either — the pair sits
  below the 0.20 TF-IDF threshold.

**`Help_EdgeMode` needed nothing.** All 19 of its seeds are help-shaped, so unlike
the five Help intents that got a command carve-out this one has no leakage to
document — the three `Cmd.EdgeMode*` intents exist and absorb the direct
requests. Its *"where the reading is balanced, prefer this intent over the
command"* tiebreak is the FAR-safe direction and was left alone. Check 101 pins
that it was not touched.

**The new checks caught this round's own edits, twice.** Adding the Fallback
carve-outs produced a new Section 7 route (`Cmd.MemoryChange` →
`Cmd.EdgeModeIncrease`, a bare pointer sentence carrying no subject) and a new
Section 8 collision (`Default Fallback Intent` vs `Help_ChangingMemories`, over
`return to the normal memory`). The second was a real gap — a how-to question
about going back to normal is the Help intent and neither spec said so — and is
now stated on the Fallback side. The first was rewritten so the sentence naming
`Cmd.EdgeModeIncrease` carries the subject it is talking about instead of being a
bare cross-reference. Fourth time this session a check has caught the reviewer
rather than the corpus.

**Nothing to close.** The remaining EdgeMode item is edit 2, which is a decision,
not an open question.

### D18. The SpeechServices family

Reviewed 2026-08-28, closing A2 and with it the whole taxonomy. All four specs
read against each other, and `Cmd.ListenMessage`'s cross-reference checked from
the destination.

**A2's known dependency holds.** `Cmd.ListenMessage` sends *"requests to start
live transcription of a conversation"* to `Cmd.TranscribeStart`, and that intent's
first two triggers claim exactly that. Nothing to fix. This is the first deferred
cross-reference in the review that turned out to be correct as accepted.

**The first question-shaped carve-out, and the first on a `Cmd.` intent**
(Akash). Five Help intents have needed the opposite — a spec that read as
question-only while its deployed speech carried commands. `Cmd.TranslationStart`
is that failure in mirror:

    Cmd.TranslationStart   11 of 63 deployed rows   17.5% question-shaped
    Cmd.TranscribeStart     1 of 50                  2.0%
    Cmd.ListenMessage       7 of 85                  8.2%
    Cmd.SendMessage         9 of 154                 5.8%

17.5% is the highest of any action command in the taxonomy. (`Cmd.ActivityStep`
and the other status queries run higher still, but a status query is a question
by nature; this one is an action.) The spec already carried the rule — *"a
question form does not make an utterance a Question type"* — and its own
`positive_example` is a question. What it never did was tell GENERATION that, so
generated data would have been imperatives only and the model would have learnt
that a question is never this intent. The measured rate and the
reproduce-it-or-it-is-a-defect instruction are now stated, in the same wording the
five Help carve-outs use.

**The sample is small and that is written into the spec.** 11 rows. The direction
is not in doubt but the number is approximate. Check 74 now re-derives
question-shaped percentages as well as command-shaped ones, so a stale rate fails
the run instead of reaching a generation prompt.

**The `Meeting` memory name, from E3's own shortlist** (Akash). E3 flagged
`Meeting -> Cmd.TranscribeStart` as one of the handful of overlaps that "look
real", and the deployed data agrees: 48 of `Cmd.MemoryChange`'s 1,601 rows name a
meeting against 5 of `Cmd.TranscribeStart`'s 50. The memory reading is the common
one, and *"transcribe the meeting"* is this intent's own subject. Guarded in the
wording the other six already use, mirrored on the `Cmd.MemoryChange` side, and
made a mutual neighbour pair. **One of E3's 17; sixteen remain.** Guard count
7 → 8.

**`Cmd.TranscribeStart` was named by two specs and named neither back.**
`Cmd.SendMessage` (*"recording to send differs from recording to transcribe"*) and
`Cmd.ListenMessage` both state the boundary; this intent said nothing, while
`record` appears in 11 of its 50 deployed rows against 7 of `Cmd.SendMessage`'s
154. Both claim the word. Now stated from this side, with the discriminator that
`Cmd.SendMessage` already uses — whether another person receives it.

**The stop-transcription gap: left alone, by decision** (Akash — *"stop ki need
nahi hai"*). `Cmd.TranscribeStart` sends stop requests to Fallback with *"no stop
intent exists in this taxonomy"*, and Fallback covers them only through the
generic *"a capability the product does not have"*, never naming transcription —
the same shape D16 closed for `reminders.complete`. Streaming has
`Cmd.StreamingStop`; transcription has no counterpart. **This is recorded as
decided, not as missed:** the capability is not wanted, so no `Cmd.TranscribeStop`
is proposed and the route is left generic. The seed evidence for it is one
mislabelled utterance — 1 of 16 seeds and 1 of 50 deployed rows carry a stop word,
which the spec's own boundary case already calls out and which this round
re-verified as still true rather than stale.

`Cmd.TranslationStart`'s *"translate content that is not speech"* route to
Fallback is the same shape and is left the same way, for the same reason.

**`Help_Transcribe` and `Help_Translate` needed nothing.** Both are 0.0%
command-shaped across 70 and 66 deployed rows — the cleanest Command/Help
separation in the taxonomy — so neither needs the carve-out five other Help
intents did. `Help_Translate`'s *"how to talk to someone speaking another
language"* trigger was checked against `Cmd.TranslationStart` because it looked
like a collision: 30 of 63 `Cmd.TranslationStart` rows name a language against 5
of `Help_Translate`'s 66, and both specs already state the naming-a-language
discriminator. It holds. Check 106 asserts both specs were left alone rather than
edited for symmetry.

**SECTION 7 WAS BLIND TO THE MOST COMMON DESTINATION IN THE TAXONOMY**

`INTENT_IN_TEXT` matched `Cmd.X`, `Help_X` and `Default Fallback Intent`, but not
the bare spelling `Fallback` — which **23 routing sentences across 14 specs**
actually use. Every one of those routes was invisible to the check written
specifically to catch routes with no destination. The matcher now normalises the
short form.

Fixing it surfaced exactly one real route, in a family signed off three rounds
ago: `Cmd.FindMyPhone` sends *"requests to find any other object"* to Fallback,
which never claimed locating anything. Closed on the Fallback side, and
`Cmd.FindMyPhone`'s two bare-`Fallback` sentences rewritten to the full name.

**And the fix immediately caught this round's own edit.** The new Fallback trigger
collided with `Cmd.FindMyPhone` in Section 8 on `locate` and `phone`, because
`subject_collisions` clears a pair only when one spec NAMES the other and
`Cmd.FindMyPhone` used the short spelling. The rename closed it. Fifth time this
session a check has caught the reviewer rather than the corpus.

**What the fix does not catch, stated so nobody misreads a clean Section 7.**
Neither the stop-transcribing route nor the non-speech-translate route flags even
with the matcher repaired, because their subject words overlap Fallback's very
broad vocabulary. Still a floor, not a gate.

**Nothing to close.**

### D19. Sign-off

**2026-08-30, Akash Rawat, all 18 families.** 60 of 60 intents read against their
siblings across 18 rounds, recorded D1-D18.

**What it claims and what it does not.** Not that the specs are perfect. That
every intent has been read against the ones it can be confused with, and that
what is still open is written down here rather than unknown. Sections A through F
of this file are the list of what is known-open; E3, E4, E7 and the eleven
Section 7/8 flags are all still open and none of them blocks the gate.

**A tick had nowhere to live that survived.** `SPEC_REVIEW.md` is regenerated on
every run — its own first line says so — and the boxes were hardcoded `☐` at
`spec_review.py:869`. Ticking them by hand would have been erased the next time
anyone ran the tool, silently. The same trap sat under the `meta` note:
`bootstrap_specs.py` WRITES that block, so editing `intent_specs.yaml` alone
would have restored `REQUIRES HUMAN REVIEW` on the next regeneration.

Sign-off therefore lives in four places, all of them source rather than output:

    spec_review.py       SIGNED_OFF -- family, who, date; renders the ticks
    bootstrap_specs.py   the meta.sign_off block it writes into intent_specs
    intent_specs.yaml    meta.sign_off, matching what bootstrap would emit
    authored_specs.yaml  the header no longer says review is pending

Checks 34, 35, 35b and 35c were FLIPPED rather than deleted. They used to assert
the gate was shut — 18 unticked boxes, `REQUIRES HUMAN REVIEW` present. They now
assert it is open and properly recorded, so the state cannot drift back
unnoticed in either direction.

**A drift found while wiring it.** `Cmd.VolumeIncrease`'s `authored_by` said
`assistant-session, pending human review` in `authored_specs.yaml` and
`assistant-session (claude-opus-5), pending human review` in `intent_specs.yaml`.
Nothing noticed, because the drift guard covers the seven CONTENT fields only and
provenance is not one of them. `bootstrap_specs.py` reads `authored_by` to build
`_provenance`, so a regeneration would have silently changed the provenance tally
from 59/1 to 58/1/1. Corrected, and check 35d now guards `authored_by` against
`_provenance.model` on all 60.

The stale `, pending human review` was dropped from all 59 at the same time. It
was a review STATE wedged into an authorship field; the state now lives in
`meta.sign_off` where it can be verified.

**Sign-off does not authorise a paid run.** Three things still stand in front of
Stage 1 and none is a spec defect:

1. ~~**E7 is decided and not applied.**~~ **Applied the same day — see D20.**
   Written before D20 existed and left standing so the sequence stays readable.
2. **749 stale rows would carry through.** Sixteen checkpoint files date from
   2026-08-17 and 2026-08-23 — before the prompt fix of 2026-08-28 and before 18
   rounds of spec edits. Without `--force` the plan is 317 calls rather than 348,
   and `Cmd.VolumeIncrease` (180 rows) and `Help_Volume` (120 rows) get ZERO new
   calls, shipping entirely pre-fix output. Both of those specs were edited after
   those rows were written.
3. **The corrected prompt has never been run.** Last generation 2026-08-23;
   `prompt.txt` fixed 2026-08-28. No pilot has been run against it.

**Nothing to close.**

### D20. E7 applied — the taxonomy is 57

**2026-08-30.** `Help_HeartRate`, `Help_HeartRateRecovery` and `Help_ThriveScore`
dropped, as decided on 2026-08-27. E7 is the plan and the cost; this is what was
actually done and what it moved.

**E7's checklist was followed and was not quite complete.** It was written before
D11, D16, D17 and D18 all edited `Default Fallback Intent`, so a fresh scan was
run rather than trusting it. The scan found one thing the list did not name:

    Default Fallback Intent   trigger  "An app score is not a clinical reading."

That sentence was added by D11's correction and exists ONLY to keep app scores
with `Help_ThriveScore`. Left in place with the intent gone, it would have carved
scores out of Fallback's clinical rule while nothing else claimed them — scores
would have belonged nowhere. Removed in the same change. Check 112 pins it.

**What moved:**

    intent_specs / authored_specs   60 -> 57 specs, 129 comments preserved
    generator_config.yaml           3 added to drop_intents with reasons,
                                    removed from families.HelpHealth (6 -> 3)
                                    and from hand_authored_intents (60 -> 57)
    length_targets.yaml             3 entries removed
    neighbour links                 338 -> 321
    Fallback                        1 exclusion removed, 1 trigger added,
                                    the app-score carve-out removed
    Help_Health                     2 exclusions merged into 1, 1 boundary case
                                    rewritten, 3 neighbours dropped
    Help_Activity                   1 exclusion rewritten, 1 neighbour dropped
    Stage 1 plan                    8,360 -> 8,000 rows, 348 -> 333 calls forced

**Fallback CLAIMS them; it is not merely where they land.** The ordering E7
insisted on was kept — the three were dropped and Fallback updated in the same
change, never stated first. The new trigger names all three subjects and says why
they resolve there, the same standard C1 and C2 set for messaging and for
powering the aids on or off.

**Nine checks retired rather than deleted quietly.** 50-55, 56c, 56d, 57 and 73
asserted D10 and D11 edits to two specs that no longer exist. The comment in
`verify_round.py` says which round wrote them and why they went. What survived is
the Fallback widening those rounds produced — checks 56 and 56b — because it
outlives the intents that prompted it.

**A Section 8 flag closed as a side effect, and it is worth naming.**
`Default Fallback Intent vs Help_Health` cleared, not because the collision was
argued away but because `Help_Health`'s rewritten exclusion now names
`Default Fallback Intent` outright and `subject_collisions` skips a pair once one
side names the other. Section 8 goes 6 → 4. That is a real fix, but it is worth
recording that the mechanism was incidental.

**THE MEASUREMENT COST IS NOW LIVE, AND IT IS THE ONE THING E7 SAID NOT TO
FORGET.** All three are still in `language_packs/en/nlu_schema.json` and still
carry deployed data, so the drop WIDENS the runtime delta (57 vs 57, overlapping on 53) rather than
closing it:

    7 of 813   =  0.9%   before   well inside the 0.038 MDE
    34 of 813  =  4.2%   now      LARGER than the MDE

`dev_hard` now contains 34 rows for intents this taxonomy does not have. A model
following the taxonomy is marked wrong on every one, and the bias points against
the change the Super Dataset exists to demonstrate. The architecture doc already
requires an explicit decision before any result is reported — exclude those rows
and re-state the 0.8327 baseline, or accept and record it. **After this drop that
decision cannot be deferred.** It is not a Stage 1 blocker; it is a
report-the-result blocker, and it is now the largest open item in this file.

**Sign-off was given for 60 intents, hours before the taxonomy became 57.**
`meta.sign_off.scope` says so rather than implying the 57 were signed as such.
Three specs — Fallback, `Help_Health`, `Help_Activity` — were edited after
sign-off, all three as the direct consequence of the E7 decision of 2026-08-27
rather than as new review findings. Recorded here so nobody later reads the
sign-off as covering text it did not.

**Nothing to close.**

### D21. A review of this session's own changes, and what it found

Run 2026-08-30 after the EdgeMode, SpeechServices, sign-off and E7 commits, as an
adversarial pass rather than a confirmation one. Fourteen findings, every one
verified against the repo before being accepted. The four that mattered:

**A false superlative reached a spec. Third time in this review.**
`Cmd.TranslationStart` claimed 17.5% question-shaped was *"the highest of any
action command in the taxonomy"*. `Cmd.FindMyPhone` is **20.2%** and is an action
command — it rings the phone. The claim came from an analysis that excluded it
through a hardcoded status-query list the spec never disclosed, which is the same
shape as the ranking errors the audit round found in `Help_Pairing` and
`Help_SelfCheck`. Check 70 guards those two by name and nothing generalised it.

Fixed by dropping the ranking and keeping the comparison that carries the actual
argument — the two halves of one family, eight times apart. Two new guards: check
117 fails any taxonomy-wide superlative about a spec's own rate, and **check 74
now re-derives CITED percentages too**, not just a spec's own. That is the gap
that let this through twice: `Help_FindMyHearingAids at 40.6%` and
`Cmd.TranscribeStart at 2.0%` were never checked by anything.

**`meta.sign_off.scope` had already drifted from the source that writes it.**
`intent_specs.yaml` carried D20's qualifier — *"Three unsupported intents dropped
afterwards"* — and `bootstrap_specs.py` still wrote the unqualified sentence. The
next regeneration would have erased exactly the sentence D20 added so nobody
reads the sign-off as covering 57 intents. **This is the trap D19 documents,
reproduced one field over, by the commit that documented it.** Check 35b was a
presence check (`"sign_off"` appears in the source) and could not see a drifting
value; it now compares all five fields against the literal the source would emit.

**The RUNTIME DELTA block still described the world before E7.** It said the
taxonomy is 60 and the label map 57, and its REMOVE list named one intent. Both
sides are now **57 and still do not match** — they overlap on 53, four names each
side. Equal totals reading as agreement is worse than the old mismatch, so the
block now says so in its first line, and the three dropped intents are on the
REMOVE list with the 4.2% dev_hard cost attached.

**A dangling `- but` mid-sentence in `Help_IntelliVoice`,** left by the audit
round. Pre-existing, but D17 claimed to have checked that spec's cross-references
from the other side and did not notice the wording. Check 5's corruption regex
looks for `[a-z]- [a-z]` and cannot match `. - but`; check 119 now covers it.

**Also fixed:** `spec_review.py` rendered "cannot name 59 intents" (56);
`generator_config.yaml` still said all specs *"need HUMAN REVIEW before Stage 1"*
in the same file as the sign-off that closed it; `generator.py --pilot` help
quoted 348 calls and 8,360 rows (333 and 8,000); and three check labels lied
about what they assert — 84 and 35d said 60, and 58 said *"the surviving
HelpHealth specs untouched"* when D20 had edited two of them in that very commit.
58 now pins the shape D20 left them in.

**Verified clean:** `seed_loader` loads 57 with no warning and files all three
orphaned seed files under `dropped` with reasons; every `.py` compiles and
`bootstrap_specs` imports; `Cmd.EdgeModeDeactivate`'s narrowing orphans 1 of its
16 seeds, and that one names a different mode which the same commit disclaims;
`Cmd.MemoryChange`'s narrowing orphans 0 train rows and the 1 `dev_hard` row D17
already recorded; the nine checks E7 retired all have live successors, so no
coverage was lost; no dropped, duplicated or mangled list items anywhere else in
the diff.

**Two things left open rather than fixed — see E11 and E12.**

### E11. The Fallback privacy block contradicted the taxonomy — CLOSED 2026-08-30

**The fix originally written here was wrong on both counts, and is left visible
rather than quietly replaced.** It said to implement `redistribute_seeds_to` and
point the three dropped intents' 106 seeds at `Default Fallback Intent`. That
would have been:

- **useless** — `Default Fallback Intent` is listed under
  `generation.privacy.no_seed_block_intents`, so `_seed_block_for` returns a
  hand-written substitute and its seed block is **never sent at all**. A larger
  seed pool changes nothing that reaches the API.
- **exactly the thing to stop and ask about** — it would have moved 106 more raw
  production transcripts into the pool that is withheld for PII reasons.

It also said "all five `drop_intents` entries carry" the key. Four do;
`Cmd.Health` never had it and documents its loss with
`orphaned_utterances_note` instead — which is the pattern the others should have
followed.

**Underneath it was a real defect, introduced by D20 and caught by nothing.**
The hand-written substitute block is the ONLY thing telling the model how
Fallback sounds. Its SCOPE GUARD read:

    This product ITSELF supports: ... battery status, find-my-phone, and
    activity/health queries. NEVER generate a request for any of those
    capabilities, however phrased

D20 had just made heart rate, heart rate recovery and the Thrive scores
unsupported and given Fallback a trigger claiming them. So the block steering
Fallback's generation **forbade the utterances Fallback now owns**. Stage 1 would
have produced a Fallback set with none of them, and the low rate would have read
as normal rather than as a suppressed rule.

**Fixed (Akash, 2026-08-30).** The guard now says *"activity tracking, and the
Health screen and its goals"* — both still supported — and carries a named
exception telling the model to write ordinary questions about the three
unsupported subjects, with a note that they need real coverage because the
specification claims them and almost no observed speech does. Written by hand
from the decision, not from the corpus; nothing in it is a transcript or a
paraphrase of one, which is the standard the block's own comment sets.

**`redistribute_seeds_to` removed** from all four entries (Akash). No Python has
ever read it, in any commit, and no document defines it. The `drop_intents`
comment now states that seeds of a dropped intent are discarded, and why
redistribution to the obvious target would not have helped.

**Five new checks.** 120 pins that Fallback is still the one intent whose seed
block is withheld; 121 and 122 pin the corrected guard; **123 is the general
form** — no subject Fallback's own triggers claim may appear in the guard's
supported-and-never-generate list, which is the contradiction nothing could see;
124 fails if the dead key comes back. Mutation-tested: restoring either the old
guard wording or the key fails its check.

**The two remaining dead config keys are NOT closed by this.**
`hard_negatives_per_intent: 40` and `oos_ratio: 0.15` are still read by no code.
They belong to the unbuilt Stage 2 and Stage 3 and are a separate decision.

### E12. Stale build output — CLOSED 2026-08-30

`specs_summary.md` still opened *"60 intents. Provenance: 59 × assistant-session
(claude-opus-5), pending human review"* and carried rows and neighbour-table
entries for all three dropped intents. `seed_audit_report.md` was dated
2026-08-13. Both regenerated.

**Running `bootstrap_specs.py` turned out to be free and safe, which is worth
recording because it was assumed otherwise.** All 57 intents are hand-authored,
so `intents = [i for i in corpus.intent_names if i not in hand_authored_names]`
is empty, no chain is built and no API key is required. Confirmed by `--dry-run`
first — *"57 hand-authored, 0 to generate"* — and the run made **zero LLM
calls**.

**It also proved D19's claim rather than asserting it.** D19 said the sign-off
survives a regeneration; that had only been checked by comparing strings. The
regeneration ran and `meta` came back **identical**, with zero field differences
across all 57 specs. The only change to `intent_specs.yaml` was line wrapping.

**One thing to know before the next surgical edit.** `bootstrap_specs.py` writes
`intent_specs.yaml` at `width=88`; the apply scripts used through this review
dumped it at `width=100000`, one line per item. That is why this commit shows
~1,000 lines of churn in a file whose content did not change. The file is now in
bootstrap's canonical format, and the intended workflow is the one just proven:
**edit `authored_specs.yaml`, then regenerate.** A future script that dumps
`intent_specs.yaml` directly should match `width=88`, or the churn comes back.

**`seed_audit.py` section 4b was also wrong.** It labelled every runtime-only
name *"no seeds, no spec"*, which was true when every such name was seedless and
stopped being true when D20 dropped three intents that keep 19, 22 and 65 seed
files on disk. The label now distinguishes a dropped intent from a seedless one.
Its headline was already right: derived 57, runtime 57, overlapping on 53.

**Nothing to close.**


### D23. The pilot, and the quota that had been overruling six specs

First paid run against the corrected prompt, 2026-08-30. 12 intents, 12 calls,
$0.15, **0 failures and 0 rejections**, 301 rows, all unique, no cross-intent
duplicates.

**E11's fix is confirmed by data rather than by reasoning.** 5 of Fallback's 25
rows carry the heart-rate/Thrive subject D20 gave it. Before the scope-guard fix
that was structurally impossible — the block told the model never to write them.
D17's environment observations came through too, 3 of 25.

**And the pilot found something five rounds of spec work could not.**

    Help_FindMyHearingAids   0 of 26 command-shaped   deployed 40.6%   expected 10.6
    Help_Pairing             0 of 25                  deployed 29.5%   expected  7.4

Not model disobedience. The prompt contained **both instructions at once**:

    line 68   "...40.6% command-shaped, and generation must match that rather
               than producing how-to phrasing only; a generated rate near zero
               is a defect, not a success."
    line 94   "- at least 22 must have type `Question`."

The second comes from the `help` profile's `Question: [0.88, null]`. A 0.88 floor
leaves at most 3 rows of 25 for every other type — a 12% ceiling. Five of the six
carve-out rates sit above it, so they were **arithmetically impossible**. Prose
against a number in the same prompt: the number won.

The quota was raised for a length reason (the model was using
`ObservationPlusCommand`, exempt from the word cap, to write 22+ word rows). The
carve-outs were written three rounds later, in spec prose, with no config change.
Nobody checked the two against each other.

**`boundary_lint` marked both `ok`.** Its pass condition is *"a generated rate at
or below deployed"* — one-sided, built to catch a generator putting TOO MUCH
command phrasing in a Help intent. It cannot fail an intent for producing none of
what it was asked for. So the only instrument that could have seen this was
structurally blind to it, and five rounds of carve-out work were invisible.

**The fix, in three parts.**

1. `_quota_profile` merges a `quotas.per_intent` map on top of the assigned
   profile, `types` key by key. An override states only what differs. Nothing is
   relaxed globally: the other 28 Help intents keep 0.88.
2. Seven per-intent overrides. Each **asks for the thing** rather than merely
   making room — an `ExplicitCommand` floor equal to the rate the spec states,
   with the `Question` floor lowered to fit and three rows of slack. This is the
   config's own lesson, written in its comments twice: *quota what the model
   under-produces*. The loophole the 0.88 floor closed is not reopened:
   `ExplicitCommand` is not in `LONG_FORM_TYPES` and stays capped at 20 words.
3. `boundary_lint` is now **two-sided for carve-out intents only**. Everywhere
   else a low rate is the desired outcome and testing for it would fail correct
   work. Guarded like the upper tail — it only judges when deployed speech
   predicts at least 3 rows, so noise does not decide. Re-run against the same
   pilot data it had passed, it now reports `FAIL (UNDER)` for both.

**A mutation test found a sixth carve-out, and a check that could be walked
through.** The first detector matched the phrase *"near zero is a defect"* and
missed `Help_HearShare` (*"should not drop it to zero"*) and
`Help_RemoteProgramming` (12.6%, which the 12% ceiling also blocked). Detecting
on the STATED RATE instead found all six. Separately, check 129 grepped for
`def p_at_most` — which `def p_at_most_DISABLED` also satisfies. It now calls
the function and asserts the maths: 0 of 26 against 40.6% must be improbable,
and 10 of 26 must not be.

A third mutation exposed a worse habit: a missing override made check 126 raise
`KeyError` and kill the run **before check 125 could report the finding**. A
check that takes the report down with it is worse than no check. Both now
tolerate the absence and let 125 name it.

**Twelve new checks (124b–130).** 127 is the load-bearing one: each
`ExplicitCommand` floor must equal the rate its own spec states, and check 74
re-derives that rate from `train.csv` — so the config cannot drift away from the
specs without something failing.

**NOT YET PROVEN.** The pilot proved the bug; nothing has yet proved the fix.
The rendered prompts now read *"at least 10 must have type `ExplicitCommand`"*
for `Help_FindMyHearingAids` and *"at least 7"* for `Help_Pairing`, and both are
in the pilot set — so a second pilot would settle it for ~$0.15. Until then this
is a corrected instruction, not a measured result.

**Length is the other open result.** 5 of 12 intents failed `length_lint`: four
too long, and `Help_Pairing` too short-share (2 rows at or under 7 words against
8.4 expected). The lint's own note points at where to look first — *"tying Hard
to length is what caused this once already"*. Not touched in this round; one
thing per paid run.


### D24-D26. Three pilots, and what they actually settled

$0.21 across four calls after the first 12-call pilot. Each change was made
alone, measured, and the next decided from the result.

**Pilot 1 (12 intents, $0.15) found the quota conflict** — D23.

**D24, the SHAPE of a direct request.** The quota fix moved the type labels
0 -> 13 and 0 -> 11, and `boundary_lint` still failed both. The model labelled
rows `ExplicitCommand` while writing them as how-to questions. Measured, deployed
speech makes the request a different way, and no spec had ever said so:

    Help_FindMyHearingAids   can-you 18.8%, please 18.3%, imperative 3.1%
    Help_Pairing             can-you 11.6%, please 11.2%, imperative 3.1%
    Help_SelfCheck           please 13.6%, can-you 10.9%, imperative 0%

Real users almost never use a bare imperative. All six carve-outs now carry
their measured breakdown, named with `boundary_lint`'s own pattern names so the
spec and the instrument share a vocabulary, and check 131 re-derives every share
from `train.csv`.

**D25, a sentence that may have primed what it warned against.** D24's line ended
by NAMING the trigger words — *"show me, tell me and help me are
explain-requests rather than commands"*. After it:

    Help_FindMyHearingAids   explain-request  32.0% -> 54.5%   (deployed 12.9%)
    Help_Pairing                              21.7% -> 16.0%   (deployed 25.4%)

It went the wrong way on exactly the intent that was still failing, and the right
way on the one that was recovering. **n is 22 and 25, so this is a signal and not
a proof** — but naming the words was a bad idea regardless. Replaced with a
per-intent measured ceiling, re-derived by check 133; check 134 fails if any spec
names those words again.

**Pilot 3 settled Help_Pairing and did not settle the other.**

    Help_Pairing              cmd-shaped  8.7% -> 20.0% -> 20.8%   deployed 29.5%   ok
    Help_FindMyHearingAids                0.0% ->  0.0% ->  4.0%   deployed 40.6%   FAIL

**D26, and a prediction of mine that was half wrong.** I expected the lint's
`EXPLAIN` short-circuit to be treating generated data unfairly. It is not — it
falls the same way on both sides, so the comparison is sound. What it does do is
hide the shape of the remaining gap:

                                    command-shaped   carries a command pattern
    Help_FindMyHearingAids  deployed        40.6%                       46.4%
    Help_FindMyHearingAids  generated        4.0%                       32.0%
    Help_Pairing            deployed        29.5%                       48.7%
    Help_Pairing            generated       20.8%                       41.7%

**Generation is not producing zero direct requests. It is producing them at about
seven tenths of the deployed rate and putting a help-verb in front**, which
resolves to explain-request before the command patterns are ever tested. "4.0%"
read as a near-total failure; the truth is a phrasing defect on top of a
roughly-right rate.

`boundary_lint` now reports a `Cmd-patterned` column beside the scored one.
**Diagnostic, never scored** — the pass condition stays on the same measure for
both sides, because changing it would compare two different things. When
`Flagged` is near zero and `Cmd-patterned` is not, the defect is phrasing.

**Deliberately stopped here.** A fourth pilot would be a third attempt at the
same move — rewording a spec and hoping. Three data points say prose steers this
model well on `Help_Pairing` and weakly on `Help_FindMyHearingAids`, and the
residual is one intent phrasing direct requests as "can you help me X" instead of
"can you X".

**What this means for the full run, stated so nobody reads it as "all fixed".**
`Help_Pairing`'s result is what the fix looks like when it works, and it is good
enough. `Help_FindMyHearingAids` will ship with roughly 70% of its deployed
direct-request rate, most of it phrased with a help-verb. That is a real
shortfall on one intent, it is measured, and it is written here. It is not a
reason to hold a 333-call run — but a Super Dataset result must not be read as
if this intent matched deployed speech, because it does not.

**Not investigated:** whether the other four carve-outs behave like
`Help_Pairing` or like `Help_FindMyHearingAids`. None is in the pilot set. The
full run is the first time anyone will see them.


### D27. The Help-vs-Command verification spec, checked against the code

An external specification (Akash, 2026-09-08) setting out what the pipeline must
teach about Help versus Command. Reviewed against the executable path rather than
the documentation, as its own section 15 demands. One check built from it.

**Its central assumption does not hold for this product.** The document treats
Help-vs-Command as one global boundary. Deployed speech says it is per intent:

    Help_FindMyHearingAids   40.6% command-shaped
    Help_Pairing             29.5%
    Help_SelfCheck           24.5%
    Help_Tinnitus             0.0%
    Help_Translate            0.0%

A single rule would be wrong at both ends. That is why six intents carry measured
carve-outs and the other 24 do not, and it is the most expensive thing this
review learned.

**Already implemented, with the code that does it.** Agency (its sections 1 and
5) is precedence rule 4 in `generator.py`'s SYSTEM_PROMPT, in those words.
Indirect commands (6) are the `ImplicitCommand` type with a 0.12 floor in the
`command` quota profile -- enforced, not merely described. "Not a keyword list"
(2, 10) is the six carve-outs plus `Help_Volume`'s measured table: `"how do I"`
95 Help to 0 Cmd, `"can you"` 4 to 149, `"can I"` 88 to 10 across the taxonomy --
measured rather than asserted. Fragmentary Help (4.6) is the prompt's FRAGMENT
instruction with per-intent `min_short` targets. `can I` versus `can you` (12.E)
is `boundary_lint`'s `can-i` and `can-you` patterns, and holds in generated data:
2 `can I` rows in the 301-row pilot, both Help.

**Genuinely missing, and it is the document's own centre.** Minimal pairs and
hard negatives (8, 9, 12.A/B) have NO generation mechanism.
`hard_negatives_per_intent: 40` and `oos_ratio: 0.15` are read by no code; Stage
3 does not exist. The prompt sends ONE `hard_negative_example` and a list of
neighbour intent NAMES with no utterances, so the model is never shown a Help and
a Command side by side. `deduplication.scope: within_intent` at least means
minimal pairs would survive if they were ever produced.

**Declined: its section 11 `speech_act` and `agency` fields.** This session
measured the model labelling 13 rows `type: ExplicitCommand` while writing them
as questions. Two more self-declared enums are two more places for the label to
disagree with the text, plus prompt tokens. The property is measured FROM the
text instead -- which is what `boundary_lint` does, identically on both sides.
Its section 13 warning against keyword classifiers does not apply either: these
patterns are a measuring instrument compared against deployed data, not a
labeller.

**Built: section 12.C, marker-free Help.** Measured before building, because a
check that cannot fire is worse than none. 45.9% of the 2,946 deployed Help rows
carry none of `how`, `what`, `where`, `can I`, `is there`; per intent it runs
19.3% to 78.1%. It fires on real pilot data:

    Help_Tinnitus            generated  8.0%  deployed 55.0%   FAIL  p = 0.000
    Help_FindMyHearingAids             40.0%           68.8%   FAIL  p = 0.003
    Help_Pairing                       62.5%           78.1%   ok    p = 0.060
    Help_ChangingMemories              32.0%           47.8%   ok    p = 0.083
    Help_Translate                     48.0%           51.5%   ok    p = 0.439

**And it sees something the rest of the pipeline cannot.** `Help_Tinnitus` is
0.0% command-shaped on BOTH sides and passes every existing check, while
generating Help rows that lean on a question marker 92% of the time against a
real 45%. The two metrics are close to independent: of `Help_Tinnitus`'s 105
marker-free deployed rows, **0** are command-shaped.

Design choices, each measured rather than assumed: per-intent baseline (the
spread is 19.3-78.1%); one-sided on the LOWER tail (producing MORE marker-free
rows is another check's business); the marker list is the specification's five
verbatim, because widening it shrinks the marker-free set and makes the test
quietly stricter than the property it names; CONTAINS rather than starts-with,
because `HELP_SHAPED`'s anchored patterns answer a different question -- "tell me
how to pair them" is an explain-request by surface form and marker-carrying by
this one.

**A check of mine was fooled by a substring for the second time.** Check 140
grepped for `failures += marker_failures`, which `pass  # failures += ...` also
satisfies, so a mutation that unwired the scoring passed. It now calls
`report_markers` on a case whose answer is arithmetic and reads the wiring out of
the parsed syntax tree. The earlier instance was check 129 and `def
p_at_most_DISABLED`. Grepping source text is not a test.

**A review of the check, before it was committed, found four things in it.**

1. **The rendered report hardcoded "45.9% marker-free".** A corpus statistic
   written into generated output is the exact defect this review has now fixed
   six times. Computed at render time instead, with the row count beside it.
2. **Check 138 dissected the compiled regex's pattern string** to prove the
   marker list was unchanged. Fragile surgery on an implementation detail, and
   this file had already been fooled twice by inspecting text rather than running
   the thing. It now asserts behaviour: five markers must be detected, and six
   near-misses (`why`, `which`, `when`, `please`, `tell me`, a bare imperative)
   must not.
3. **The regex assumes apostrophes survive `normalise`.** `"what's"` becomes
   `"what s"` and matches; `"whats"` does not, and such a row would be counted
   marker-FREE, inflating the deployed baseline and making the test stricter than
   the property it names. Measured: **zero in Help**. Not hypothetical though --
   the corpus already holds four, 2 in `Cmd.BatteryLevel` and 2 in Fallback. One
   Help row away from being wrong, so check 138c asserts it rather than noting it.
4. **Two statistical limits are now printed with the result rather than left
   implicit.** At alpha 0.05 over the 30 Help intents a full run judges, roughly
   1.5 failures are expected BY CHANCE from a perfect generator -- so one or two
   FAILs is not evidence, a pattern across related intents is. And the exact
   binomial assumes independent rows, which a batch is not: it is one LLM call
   under composition quotas and an avoid-list, so its rows are negatively
   correlated and the p-values are optimistic. **Both limits apply equally to the
   Help-versus-Command section that has been in this file all along**, and were
   never stated there either.

**Recommended, not done:** the document is the right requirements source for
Stage 3 when it is built. `command_help_pairs` already names the 21 Cmd/Help
pairs the minimal pairs would be drawn from.


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
                                         GUARDED by D18 -- 16 unguarded remain

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

`Help_Activity` is one of the four taxonomy-only intents in the runtime delta
(60-vs-57 when this was written; both sides are 57 since D20, overlapping on 53
with four names on each side), and the
only one of the four that is not an EdgeMode command. It has **0 rows in
`train.csv`** and 0 in `dev_hard`, against 26 seed utterances.

It was reviewed here on its spec and its seeds alone. Every other intent in this
family could be checked against deployed speech; this one could not, so the
command-shaped measurement that has caught three defects so far is simply
unavailable for it.

This is the instrument gap the architecture doc already records, seen from the
review side rather than the evaluation side.

**To close:** it closes when the instrument gap does — see F.

### E7. Unsupported intents — decided to drop, NOT yet applied — APPLIED 2026-08-30, see D20

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

### E8. Defects the audit found INSIDE families already signed as reviewed — CLOSED

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

**CLOSED 2026-08-27.** All four fixed, and the first one resolved the opposite
way to the assumption.

`Help_Activity` versus `Help_Health` on locating activity data — Akash decided it
belongs to **`Help_Activity`**, on the evidence: 5 of `Help_Activity`'s 26 seeds
name distance, and 0 of `Help_Health`'s 23 do. Nine specs were pointing at the
intent with no evidence, including `Help_Activity` itself. All **17**
`Help_Health` references across the eight `Cmd.Activity*` specs now read
`Help_Activity`; `Help_Activity`'s trigger widened from distance to any tracked
activity's data and how a figure is calculated, which also closes the separate
calorie-calculation one-way route; its self-contradicting boundary case replaced;
and `Help_Health` yields it back by name. No new neighbour links were needed —
`Help_Activity` was already a neighbour of all eight — so `Help_Health` stays at
9 rather than going to 16.

`Help_SelfCheck`'s "is faint" trigger removed. Deployed data settles it: one row
of 110 names faint and none names quiet, weak, low or soft, while
`Cmd.VolumeIncrease` has 3 and 23.

`Help_Volume` ↔ `Help_Pairing` guarded on both sides and made mutual. The
collision was in the spec text rather than in the data — 107 of `Help_Pairing`'s
224 rows name a phone or Bluetooth against 1 of `Help_Volume`'s 146 — but
generation writes new rows, and the SOURCE is now stated as the discriminator.

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

### E10. 81 passing checks coexisted with 15 real defects — CLOSED

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

**CLOSED 2026-08-27.** Both added, as `spec_review.py` Sections 7 and 8, so they
appear in the report a human reads and are asserted by `verify_round.py`.

    Section 7  Routes with no destination
               A sends X to B, and B's own text barely mentions X.
    Section 8  Two intents claiming the same utterance
               Two triggers claiming the same ground, neither spec naming the
               other and not mutual neighbours. Only words appearing in six or
               fewer intents' triggers count, so shared hearing-aid vocabulary
               cannot raise a flag by itself.

**What they actually catch, replayed against the commits where each defect was
still live — 2 of the 5 this review found by reading:**

| | defect | |
|---|---|---|
| ✅ | `Cmd.ActivityCycle` → `Help_Health` | Section 7 |
| ✅ | Fallback vs `Help_HeartRate` | Section 8 |
| ❌ | `Help_Health` → `Help_Home` | destination shared enough ordinary vocabulary |
| ❌ | Fallback vs `Help_ThriveScore` | *value* against *score* |
| ❌ | `Help_Volume` vs `Help_Pairing` | *no sound* against *audio not coming through* |

Every miss is semantic rather than lexical. That is the honest limit of word
overlap, and it is why these are a floor rather than a gate. **They do not
replace reading two specs against each other**, and nothing in this review should
be taken to mean a clean Section 7 and 8 means the specs agree.

Calibration is not free either. First integration was quietly weaker than the
prototype it was validated as: `spec_review.py`'s `STOP` set omits ordinary
function words, so the routed subject *"Questions ABOUT where cycling distance is
displayed"* scored an overlap of exactly 1 against `Help_Health` on the word
"about" alone, and the check passed a route that was a real defect. Sections 7
and 8 now use their own vocabulary set.

**11 flags stand today** — 4 routes and 7 collisions — and they are pinned BY NAME
in `verify_round.py`, so a new one fails the run while the known ones do not nag.
They are next round's reading, not defects proven:

    Section 7   Help_AppSettings -> Help_WhatsNew
                Help_HearShare -> Help_RemoteProgramming
                Help_IntelliVoice -> Help_MaskMode
                Help_RemoteProgramming -> Help_Customize

    Section 8   Cmd.EdgeModeDeactivate vs Cmd.MemoryChange   normal, return
                                                             CLOSED by D17
                Cmd.ListenMessage vs reminders.complete      last, latest
                Cmd.VolumeMute vs Help_Tinnitus              off, turned
                Default Fallback Intent vs Help_Health       features, health
                Default Fallback Intent vs Help_HearShare    another, person
                Default Fallback Intent vs Help_ThriveScore  improve, score
                Help_MemoryOptions vs reminders.add          add, create

The `Default Fallback Intent` vs `Help_ThriveScore` row is worth naming: that is
the collision D11's correction was supposed to close. The wording now disclaims
it — *"An app score is not a clinical reading"* — but **the intent is still never
named**, which is the exact standard this review has applied to everyone else.
The check found the reviewer's own half-finished fix.

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
- **The runtime-delta instrument gap.** `dev_hard` cannot score four intents and
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
