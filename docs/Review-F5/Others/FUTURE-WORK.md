# Future Work / Deferred Items

Backlog of decisions and features intentionally deferred, with enough context to
pick them up later. Newest first.

---

## FW-001 — Add a `Cmd.EdgeModeActivate` (noise-management command) intent

**Status:** Deferred — the app does not currently support activating Edge Mode via
a voice command. For now, noise-complaint utterances map to the **informational**
`Help_EdgeMode` intent (see decision below).

**Context.** During the "noise" intent-overlap review
(`NOISE-INTENT-OVERLAP-AND-ANNOTATION-RULE.md`), several utterances were identified
as *urgent symptoms / imperatives* about background noise:

- "the noise is unbearable right now"
- "kill the noise completely"
- "reduce the background noise level"
- (and environmental cases like "too much wind noise outside")

These are phrased as **actions** ("kill…", "reduce…", "right now"), so ideally they
would trigger the device to *activate* noise management — i.e. a command intent such
as `Cmd.EdgeModeActivate` (or `Cmd.NoiseReduce`) with an `edge_mode.activate` action.

**Current decision (interim).** The app has no Edge-Mode-activation capability yet,
so these utterances are labeled **`Help_EdgeMode`** (the "here is help with Edge Mode"
information response). This is deliberate and acceptable until the command exists.

**What to do when the app supports it.**
1. Add a `Cmd.EdgeModeActivate` intent to `packs/en/schema.json`
   (`action: "edge_mode.activate"`, plus fulfillment text).
2. Wire the `edge_mode.activate` action to the device capability.
3. Re-label the imperative/symptom noise rows from `Help_EdgeMode` →
   `Cmd.EdgeModeActivate` in the training data and in
   `docs/Review-F5/noise_holdout_adjudication.csv` (rows flagged `REVIEW`).
4. Author **symptom-style training examples** for the new command (independently
   written, deduped against the holdout). Note: today `Help_EdgeMode` has zero
   symptom-phrased rows — every example literally contains the words "edge mode" —
   so the model cannot learn symptom→EdgeMode until such data is added.
5. Keep `Help_EdgeMode` for genuine *questions* ("what is edge mode?",
   "how do i use edge mode?").

**Why it matters.** Routing an urgent command ("kill the noise completely") to an
information response is a UX gap: the user wants an action, not an explainer. This is
tracked so the interim `Help_EdgeMode` mapping is revisited once the capability ships.

**Related:** `NOISE-INTENT-OVERLAP-AND-ANNOTATION-RULE.md`,
`noise_holdout_adjudication.csv`.
