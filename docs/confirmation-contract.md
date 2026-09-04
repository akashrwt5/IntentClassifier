# Confirmation contract (`workflows.confirmation`)

When an intent asks permission, the turn that answers it carries three facts:
what the answer **does**, what it **says**, and what the host **calls** the
outcome. All three are authored on the branch, compiled into the workflow, and
read by every runtime from there.

```yaml
# content/capabilities/messaging.ptt/intents/Cmd.SendMessage.yaml
followup:
  'yes':
    action: message.send
    fulfillment: Ready to record your message.
    label: Cmd.SendMessage - yes        # optional, compat only
  'no':
    action: message.cancel
    fulfillment: Okay, I won't.
    label: Cmd.SendMessage - no
```

```json
// capabilities/messaging.ptt/workflows.json
"confirmation": {
  "required": true,
  "prompt": "Cmd.SendMessage.confirm",
  "yes": { "action": "message.send",   "response": "Cmd.SendMessage.confirm_yes",
           "label": "Cmd.SendMessage - yes" },
  "no":  { "action": "message.cancel", "response": "Cmd.SendMessage.confirm_no",
           "label": "Cmd.SendMessage - no" }
}
```

## Why it is shaped this way

**The branches used to be inferred.** The bundle said an intent asks, never what
either answer does. VoiceAIKit filled the gap with `completion.action` for yes
and a literal `""` for no. The yes guess is the subtle one: it is right only
while *accepting a confirmation* means the same thing as *never being asked*.
The moment content authored `message.send` and `message.cancel`, the reference
engine fired those and the device fired `message.compose` and nothing — same
pack, same words, two answers, and nothing anywhere went red.

**The label had no route to a device at all.** `Cmd.SendMessage - yes` is not a
model class and cannot become one: polarity is decided *after* classification,
so the head's label space does not contain it. It is not an intent id either.
And neither runtime may compose one, by the invariant that a client never
interprets or builds an intent label. It lived in
`packages/runtime/nlu_engine/legacy_label_map.json` under `confirm_compound`,
which the reference engine read and no device ever saw. Two artifacts were tried
to carry it — `runtime/legacy_labels.json`, then
`runtime/confirmation_labels.json` — and both described *half a turn from a
second file*, which is the defect, not the file name.

Putting the label on the branch collapses that: one turn, one place, and whoever
changes the yes branch cannot forget the other half because there is no other
half.

## Rules

- **Both branches or neither.** A pack that asks a question without stating both
  answers forces every client to invent them, and clients invent different
  things. The schema marks `yes`/`no` optional ONLY so packs signed before they
  existed still verify; a runtime is entitled to refuse such a pack, and
  VoiceAIKit does — `VoiceIntentError.confirmationBranchesMissing`.
- **Branch actions are actions.** They must be declared in the capability's
  `actions`, exactly like `completion.action`. `validator.stage_3_references`
  enforces it; an undeclared one reaches a device as a key no capability owns,
  so the host dispatches on a string nothing answers to.
- **`label` is optional and compat-only.** Absent means report the intent id
  unchanged. Delete the key when the app consumes plain ids; nothing else
  depends on it.
- **`fulfillment` becomes a response key** (`<intent>.confirm_yes` / `_no`), so
  a second language is a second responses file, not a schema change.

## Where each runtime applies it

| | reads | reports |
|---|---|---|
| Python | `nlu_schema.json` → `followup[polarity]` | `branch["label"] or intent` in `_handle_confirmation` |
| Swift | `workflows.confirmation[polarity]` → `FollowupBranch` | `fu.yes.label ?? intent` in `handleConfirmation` |

No environment flag on either side. `label_compat` used to do this for Python
behind `NLU_LEGACY_LABELS=1`, which meant the reference engine's *default*
output disagreed with what every device would report. Reading the label off the
branch removes the flag from the question.

## What `legacy_label_map.json` still does

Only the `map` half: modern label → legacy app label, IDENTITY for all 57
intents today. It renames nothing and does not ship to devices. It is kept, and
kept complete, because it is where "what does the app call this intent?" is
declared — a new intent with no answer fails
`test_every_trained_intent_has_a_legacy_mapping` rather than reaching the app
under whatever the trainer named it. That gate reads this file, never a pack.

## Conformance

- `tests/test_confirmation_branches.py` — content authors both branches, the
  bundle carries them, the engine reports what they say. Parametrised over every
  intent with a followup, so a second gated intent is covered when authored.
- `tests/test_legacy_label_compat.py` — the shim no longer rewrites confirmation
  outcomes, and the source no longer carries `confirm_compound`.
- `Tests/VoiceAIKitTests/ConfirmationAndSlotFlowTests.swift` →
  `ConfirmationBranchTests` — the Swift half of the same three claims.
- `spec/examples/3.0/full` — the canonical complete shape.
