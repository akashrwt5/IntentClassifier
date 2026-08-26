# Deferred and open — Stage 0 spec review

Things that were consciously put down, and things nobody has answered yet. They
live here because a decision deferred in conversation is a decision lost: it
leaves no trace in the specs, no flag in `SPEC_REVIEW.md`, and no row in any
report. The per-family sign-off table in `SPEC_REVIEW.md` records what has been
*reviewed*; this file records what has been *skipped*, and why.

Every item states what closing it needs. An item with no closing condition is a
wish, not a task.

Last updated 2026-08-26, after the `Cmd.*` review, the HelpAudio and
HelpAppSettings families, and `Default Fallback Intent`.

---

## Where the review actually stands

| | Count | State |
|---|---:|---|
| `Cmd.*` reviewed | 19 of 24 | AudioControl 4, Streaming 2, Messaging 2, Memories 1, DeviceStatus 1, DeviceLocate 1, ActivityTracking 8 |
| `Cmd.*` deferred | 5 | EdgeMode 3, SpeechServices 2 |
| `Help*` deferred | 1 | `Help_EdgeMode`, grouped with the EdgeMode commands |
| `Help*` reviewed | 10 of 33 | **HelpAudio** 4 of 5 (`Help_EdgeMode` deferred) and **HelpAppSettings** 6 of 6. Four others were *read as counterparts* only |
| Other | 1 of 3 | `Default Fallback Intent` reviewed. `reminders.add` and `reminders.complete` still not |

`Default Fallback Intent` has now been read end to end (2026-08-26). It had been
edited in almost every round without ever being reviewed itself — the most rules
of any intent and the least scrutiny, which was the wrong way round. Its 22 rules
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

1. **A disclaimer naming the two features it does not own.** Searched all 60
   specs: "IntelliVoice" appears in 2 (`Help_IntelliVoice`, `Help_VoiceAssistant`)
   and "Mask Mode" in 2 (`Help_MaskMode`, `Help_MemoryOptions`). **No `Cmd.*`
   spec names either.** `Cmd.EdgeModeIncrease` claims "activate, add or increase
   Edge Mode *or adaptive tuning*" and "asks for voices to be made clearer",
   which makes it the nearest attractor for a direct request naming either
   feature — and acting on one applies Edge Mode, a device action the user did
   not ask for. Both Help specs now state their side; this side is silent.
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

Read end to end on 2026-08-26. Its 10 triggers, 6 exclusions and 6 boundary cases
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

## E. Not started

### E1. All 33 `Help*` intents

Four were read as counterparts during `Cmd.*` reviews — `Help_Battery`,
`Help_FindMyHearingAids`, `Help_ChangingMemories`, `Help_VoiceAssistant` — and
two of those were edited. None was reviewed in its own right, and reading a spec
against one partner is not the same as reading it against its own family.

Remaining, largest first: HelpDeviceCare 6, HelpHealth 6, HelpConnectivity 3,
HelpSpeechServices 2, HelpFind 1. Done: HelpAppSettings 6 of 6, HelpAudio 4 of 5
with `Help_EdgeMode` deferred.

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
trigger conditions — it gives **31 name/intent overlaps, 17 of them unguarded**.

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

The HelpAppSettings review counted them properly for the first time, within one
family. Each row is a boundary a spec states in prose while the sampling link it
needs does not exist:

| Intent | Named in a rule, but not a neighbour | Neighbour, but named in no rule |
|---|---|---|
| `Help_DeviceSettings` | `Cmd.StreamingStart`, `Help_ChangingMemories`, `Help_Pairing` | `Help_WiCROS` |
| `Help_DemoMode` | `Help_Pairing` | `Help_Home` |
| `Help_Customize` | — | `Cmd.MemoryChange`, `Help_AppSettings` |
| `Help_AppSettings` | — | `Help_DemoMode` |
| `Help_WhatsNew` | — | `Help_DemoMode` |
| `Help_Home` | — | `Help_Reminder`, `Help_VoiceAssistant` |

`Help_DeviceSettings` is the worst of them: it names six intents in its rules and
lists four as neighbours, and only one of those four appears in any rule.

Only `Help_DeviceSettings` ↔ `Help_Customize` was fixed (D9), because both specs
sit in this family. The rest reach into `Help_Pairing`, `Help_ChangingMemories`
and `Cmd.StreamingStart`, which have not been reviewed, and fixing a link
requires editing both sides.

Neither direction is automatically wrong. A neighbour named in no rule may be a
sensible confusion nobody wrote down; a rule with no link may be a boundary too
coarse to need hard negatives. What is wrong is that no one has decided.

**To close:** run the same count across all 60 specs, decide each row, and either
add the link, add the rule, or record why neither is needed. Same shape as E3 —
a systematic pass, not a per-family patch.

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
  than explain-request. Across the deployed data this is **18 of the 283
  command-shaped flags on `Help*` intents — 6.4%** — but it concentrates badly:
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
