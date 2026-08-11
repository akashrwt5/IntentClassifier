package com.example.nlu

/**
 * The turn. Port of `nlu_engine/engine.py::_handle_new_intent`.
 *
 * THE ROUTING CONTRACT, WHOLE:
 *
 *     conf >= fireBar   ->  the intent fires
 *     conf <  fireBar   ->  Default Fallback Intent (the app routes to GenAI)
 *
 * There is no third tier. There is no confidence band that asks the user "just
 * to be sure". That mechanism existed, sat ABOVE the fire threshold, and turned
 * commands that would have fired into questions: on the honest holdout it
 * produced 103 friction turns against 16 useful catches, so 85% of every
 * confirmation a user saw was asked about a CORRECT prediction. "increase
 * volume" was held for confirmation while the model scored it 0.9992.
 *
 * Confirmation still exists — but only where a human AUTHORED it, per intent,
 * in `workflows.json`. In this taxonomy that is exactly one intent,
 * `Cmd.SendMessage`, because sending a message is the single irreversible,
 * externally-visible action. A product decision, not a classifier artifact.
 * Do not reintroduce a confidence-triggered ask on the client side.
 */
class NluEngine(
    private val bundle: NluBundle,
    private val model: TfidfIntentClassifier,
) {

    enum class Type { FULFILL, PROMPT, CONFIRM, FALLBACK }

    data class Result(
        val type: Type,
        val intent: String?,
        val action: String?,
        val confidence: Double,
        val prompt: String? = null,
        /** "keyword" | "tfidf" — telemetry only. */
        val stage: String? = null,
        /** "corroborated" | "contested" | null — telemetry only. */
        val arbitration: String? = null,
    )

    /**
     * Confidence reported when a rule fires but the model's top prediction is a
     * DIFFERENT intent. The rule still wins the label — it is a deliberate
     * product decision — but the disagreement is real evidence and the number
     * must say so. Corroborated predictions measure 99.1% correct on the honest
     * holdout; contested ones ~45%, a coin flip.
     *
     * PROVISIONAL: chosen, not fitted. Keep it in sync with
     * `IntentClassifier.CONTESTED_CONFIDENCE` in the Python until it is fitted
     * out-of-fold on train.csv — never on the holdout.
     */
    private val contestedConfidence = 0.60

    fun handle(session: Session, rawText: String): Result {
        // A live confirmation or slot prompt owns the turn. Omitted here for
        // brevity — see `engine.py::_handle_confirmation` / `_advance_slots`.
        // Both are bounded by bundle.limits.maxSlotAttempts; an unbounded
        // reprompt loop that re-sets its own context is a trap this codebase
        // has already shipped once.
        session.pending?.let { return resumePending(session, it, rawText) }

        // ---- stage 1 + 2: arbitration -------------------------------------
        //
        // The model runs on EVERY turn and is the sole author of the
        // confidence. The rule, when one fires, is the sole author of the
        // label. Separating those two responsibilities is the entire point:
        // a rule is a hand-authored decision about MEANING and cannot produce
        // a probability; only the model produces a number on the scale the
        // thresholds were fitted against.
        val normalized = bundle.normalizer.normalize(rawText)
        val probs = model.distribution(normalized)
        val top = probs.indices.maxBy { probs[it] }
        val modelIntent = model.labels[top]

        val kwIntent = bundle.keywords.match(rawText)   // rules see RAW text

        var intent: String
        var conf: Double
        val stage: String
        var arbitration: String? = null

        if (kwIntent == null) {
            intent = modelIntent; conf = probs[top]; stage = "tfidf"
        } else {
            stage = "keyword"
            intent = kwIntent
            if (modelIntent == kwIntent) {
                arbitration = "corroborated"; conf = probs[top]
            } else {
                arbitration = "contested"; conf = contestedConfidence
            }
        }

        // ---- guards, then RE-READ the confidence ---------------------------
        //
        // This re-read is not tidiness. A guard changes which intent is being
        // reported, and the number it inherited describes the intent that was
        // BLOCKED. "how do i turn up the loudness": the rule proposes
        // Cmd.VolumeIncrease, the model says Help_Volume, the help guard
        // correctly redirects — and the turn then carried the CONTESTED 0.60 of
        // the blocked action, dropping a perfectly good help request under the
        // fire bar and deflecting it to GenAI.
        val known = bundle.knownIntents
        val guarded = bundle.guards.applyHelpGuard(
            rawText, bundle.guards.applyPolarityGuards(rawText, intent, known), known
        )
        if (guarded != intent) {
            model.labels.indexOf(guarded).takeIf { it >= 0 }?.let { conf = probs[it] }
            intent = guarded
        }

        // ---- out-of-vocabulary guard --------------------------------------
        //
        // A confident reading of the words the featurizer CAN see says nothing
        // about the words it cannot. "help me find a paper" reaches the model as
        // `help me find`, because "paper" has no slot at all — so
        // Help_FindMyHearingAids is the correct answer to the question the model
        // was actually asked. The confidence is honest; the input was not.
        //
        // THE TWO THRESHOLDS ARE A PAIR. Ship oovReject without oovBypass and
        // you refuse every command containing an entity value, because entity
        // values are out-of-vocabulary BY NATURE — a contact name, a brand, a
        // free-text reminder topic can never all be in a finite vocabulary:
        //
        //   "send a message to john"   oov 0.25, conf 1.000   <- real command
        //   "help me find a paper"     oov 0.25, conf 0.849   <- out of scope
        //
        // The ratio cannot separate those. The confidence can.
        //
        // Runs AFTER the guards so it sees the intent actually being returned,
        // and BEFORE the fire test so its only power is to withhold an action.
        val reject = bundle.thresholds.oovReject
        val bypass = bundle.thresholds.oovBypass
        if (reject != null && bypass != null &&
            intent != FALLBACK_INTENT && conf < bypass &&
            model.oovRatio(normalized) >= reject
        ) {
            return fallback(conf, stage, arbitration)
        }

        // ---- the fire test -------------------------------------------------
        //
        // ONE threshold for every intent. Slot-bearing intents used to get a
        // lower bar on the reasoning that a prompt resolves ambiguity first —
        // but a flow whose slots are ALL filled by the classifying utterance
        // completes immediately, so the lower bar applied to a live action.
        //
        // The single exception is CORROBORATION: two independent recognisers
        // naming the same intent is stronger evidence than either alone, so the
        // bar drops to `agreement`. "turn it up its too quiet" is the case —
        // rule and model both say VolumeIncrease, but "quiet" splits the mass
        // with VolumeDecrease and leaves the top class at 0.66.
        val fireBar =
            if (arbitration == "corroborated") bundle.thresholds.agreement
            else bundle.thresholds.confidence

        if (intent == FALLBACK_INTENT || conf < fireBar) {
            return fallback(conf, stage, arbitration)
        }

        val wf = bundle.workflows[intent] ?: return fallback(conf, stage, arbitration)
        return fulfill(session, intent, conf, wf, stage, arbitration)
    }

    private fun fulfill(
        session: Session,
        intent: String,
        conf: Double,
        wf: NluBundle.Workflow,
        stage: String,
        arbitration: String?,
    ): Result {
        // Capability availability. This used to be derived from the label —
        // `intent.startsWith(capabilityId)` — which matched `device.volume.mute`
        // to `device.volume` and then silently matched NOTHING once labels
        // became `Cmd.*`, so every capability the app had pushed as unavailable
        // went back to firing. Read the mapping the pack ships instead.
        val capability = bundle.intentCapability[intent]
        if (capability != null && !session.isCapabilityAvailable(capability)) {
            return fallback(conf, stage, arbitration)
        }

        val missing = wf.slots.firstOrNull { it.required && session.slot(it.name) == null }
        if (missing != null) {
            session.pending = Session.Pending.Slot(intent, missing.name)
            return Result(Type.PROMPT, intent, null, conf,
                bundle.response(missing.promptKey), stage, arbitration)
        }

        // Authored confirmation only. Never confidence-driven.
        if (wf.confirmationRequired && !session.confirmed(intent)) {
            session.pending = Session.Pending.Confirm(intent)
            return Result(Type.CONFIRM, intent, null, conf,
                bundle.response(wf.confirmPromptKey), stage, arbitration)
        }

        return Result(Type.FULFILL, intent, wf.action, conf,
            bundle.response(wf.responseKey), stage, arbitration)
    }

    /**
     * Below the bar the result carries the ROUTING DECISION and nothing else —
     * no URL, no text. The app still holds the utterance and builds its own
     * GenAI request; the pack deliberately does not ship an endpoint.
     */
    private fun fallback(conf: Double, stage: String, arbitration: String?) =
        Result(Type.FALLBACK, "GENAI", "genai.fallback", conf, null, stage, arbitration)

    private fun resumePending(session: Session, pending: Session.Pending, text: String): Result {
        val t = bundle.normalizer.normalize(text)
        return when (pending) {
            is Session.Pending.Confirm -> when {
                t in bundle.affirmative -> {
                    session.pending = null
                    session.markConfirmed(pending.intent)
                    val wf = bundle.workflows.getValue(pending.intent)
                    Result(Type.FULFILL, pending.intent, wf.action, 1.0,
                        bundle.response(wf.responseKey))
                }
                t in bundle.negative -> {
                    session.pending = null
                    // Declining must not carry an action.
                    Result(Type.FULFILL, pending.intent, null, 1.0)
                }
                else -> {
                    // Neither yes nor no. A user who changed their mind must be
                    // able to leave — re-asking forever, while re-setting the
                    // context each time, is an infinite loop this codebase
                    // shipped once. Bounded, and a confident new command wins.
                    session.confirmAttempts += 1
                    if (session.confirmAttempts >= bundle.limits.maxSlotAttempts) {
                        session.pending = null
                        session.confirmAttempts = 0
                        handle(session, text)
                    } else {
                        val wf = bundle.workflows.getValue(pending.intent)
                        Result(Type.CONFIRM, pending.intent, null, 1.0,
                            bundle.response(wf.confirmPromptKey))
                    }
                }
            }
            is Session.Pending.Slot -> {
                session.setSlot(pending.slotName, text)
                session.pending = null
                val wf = bundle.workflows.getValue(pending.intent)
                fulfill(session, pending.intent, 1.0, wf, "slot_fill", null)
            }
        }
    }

    companion object {
        const val FALLBACK_INTENT = "Default Fallback Intent"
    }
}

/** Per-conversation state. Real apps expire this after `limits.sessionTimeoutS`. */
class Session(
    private val availableCapabilities: Set<String>? = null,
) {
    sealed interface Pending {
        data class Confirm(val intent: String) : Pending
        data class Slot(val intent: String, val slotName: String) : Pending
    }

    var pending: Pending? = null
    var confirmAttempts: Int = 0
    private val slots = HashMap<String, String>()
    private val confirmedIntents = HashSet<String>()

    fun slot(name: String): String? = slots[name]
    fun setSlot(name: String, value: String) { slots[name] = value }
    fun confirmed(intent: String) = intent in confirmedIntents
    fun markConfirmed(intent: String) { confirmedIntents.add(intent) }

    /** null = the host has not told us; assume available. */
    fun isCapabilityAvailable(id: String) = availableCapabilities?.contains(id) ?: true

    fun reset() {
        pending = null; confirmAttempts = 0; slots.clear(); confirmedIntents.clear()
    }
}
