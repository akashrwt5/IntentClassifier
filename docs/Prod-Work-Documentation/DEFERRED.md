# Deferred and open — Stage 0 spec review

Things that were consciously put down, and things nobody has answered yet. They
live here because a decision deferred in conversation is a decision lost: it
leaves no trace in the specs, no flag in `SPEC_REVIEW.md`, and no row in any
report. The per-family sign-off table in `SPEC_REVIEW.md` records what has been
*reviewed*; this file records what has been *skipped*, and why.

Every item states what closing it needs. An item with no closing condition is a
wish, not a task.

Last updated 2026-08-26, after the `Cmd.*` review, `Help_Volume`,
`Help_Tinnitus` and `Help_IntelliVoice`.

---

## Where the review actually stands

| | Count | State |
|---|---:|---|
| `Cmd.*` reviewed | 19 of 24 | AudioControl 4, Streaming 2, Messaging 2, Memories 1, DeviceStatus 1, DeviceLocate 1, ActivityTracking 8 |
| `Cmd.*` deferred | 5 | EdgeMode 3, SpeechServices 2 |
| `Help*` deferred | 1 | `Help_EdgeMode`, grouped with the EdgeMode commands |
| `Help*` reviewed | 3 of 33 | `Help_Volume`, `Help_Tinnitus`, `Help_IntelliVoice` — all HelpAudio, 3 of that family's 5. Four others were *read as counterparts* only |
| Other | 0 of 3 | `Default Fallback Intent`, `reminders.add`, `reminders.complete` |

`Default Fallback Intent` has been edited repeatedly during the `Cmd.*` review —
it is the destination for most boundary decisions — but it has never been read
end to end as its own spec. It carries more accumulated rules than any other
intent and is the least reviewed. That is the wrong way round.

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

**Two edits are queued against `Cmd.EdgeModeIncrease` and deliberately not
applied** (Akash, 2026-08-26), so that the blind-edit count stays at five. Both
came out of the `Help_IntelliVoice` review and both belong to this family's own
round:

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

**To close:** read all four specs against each other; confirm the five edits
above still hold when the family is read as a whole; apply the two queued edits;
and check the four cross-references from `Help_IntelliVoice` and `Help_MaskMode`
from this side.

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

---

## E. Not started

### E1. All 33 `Help*` intents

Four were read as counterparts during `Cmd.*` reviews — `Help_Battery`,
`Help_FindMyHearingAids`, `Help_ChangingMemories`, `Help_VoiceAssistant` — and
two of those were edited. None was reviewed in its own right, and reading a spec
against one partner is not the same as reading it against its own family.

Largest families first: HelpAppSettings 6, HelpDeviceCare 6, HelpHealth 6,
HelpAudio 5 (2 done), HelpConnectivity 3, HelpSpeechServices 2, HelpFind 1.

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

**To close:** it falls out of the HelpHealth and HelpAppSettings reviews. If the
answer is the obvious one, it is two lines — name the sibling in each spec's
`do_not_trigger` — and the tier empties.

### E2. `Default Fallback Intent`, `reminders.add`, `reminders.complete`

Never reviewed as specs. `Default Fallback Intent` is the priority — see the
note at the top of this file.

---

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
