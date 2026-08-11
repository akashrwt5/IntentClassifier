package com.example.nlu

/**
 * Post-prediction redirects, from `runtime/guards.json`.
 *
 * Both guards can only change WHICH intent is reported. Neither invents one and
 * neither raises confidence. When either fires, the caller must RE-READ the
 * confidence for the intent it now holds — see [NluEngine] for why.
 */
class Guards(
    helpMarkers: String?,
    private val helpPairs: Map<String, String>,
    polarity: List<PolarityRule>,
) {

    data class PolarityRule(val pattern: String, val blocked: String, val redirect: String)

    private val markers: Regex? = helpMarkers?.let { Regex(it) }
    private val polarityRules: List<Triple<Regex, String, String>> =
        polarity.map { Triple(Regex(it.pattern), it.blocked, it.redirect) }

    /**
     * Asking HOW to use a feature must never TRIGGER it.
     *
     * "how do i turn up the volume" is a help request that every signal in the
     * system reads as a volume command: the keyword rule fires on "turn up", and
     * the model has seen far more commands than questions. This is the safety
     * rule that separates them — a state-changing action with a paired `Help_*`
     * sibling, in an utterance carrying explicit question markers, redirects to
     * the sibling.
     *
     * Read-only queries are deliberately NOT paired: asking "how do I check my
     * battery" and checking it are close enough that the redirect costs more
     * than it saves.
     */
    fun applyHelpGuard(rawText: String, intent: String, known: Set<String>): String {
        val rx = markers ?: return intent
        val sibling = helpPairs[intent] ?: return intent
        if (!rx.containsMatchIn(rawText.lowercase())) return intent
        if (sibling !in known) return intent   // capability not in this bundle
        return sibling
    }

    /**
     * Redirect a prediction the utterance's own polarity words contradict.
     *
     * Fires only when EXACTLY ONE rule matches AND the opposite cue is absent.
     * "lower how LOUD it is" carries both a decrease cue and an increase cue;
     * the model already resolves it correctly, so a guard firing on "loud" alone
     * would flip a right answer into a wrong one. When a mirror rule also
     * matches, the polarity signal is contradictory and the guard abstains.
     *
     * The shipped English bundle has zero polarity rules, so this is currently
     * inert — it is here because a pack may add them without a client change.
     */
    fun applyPolarityGuards(rawText: String, intent: String, known: Set<String>): String {
        if (polarityRules.isEmpty()) return intent
        val low = rawText.lowercase()
        val hits = ArrayList<String>(2)
        for ((rx, blocked, redirect) in polarityRules) {
            if (blocked != intent || !rx.containsMatchIn(low)) continue
            val oppositePresent = polarityRules.any { (rx2, b2, r2) ->
                b2 == redirect && r2 == intent && rx2.containsMatchIn(low)
            }
            if (!oppositePresent) hits.add(redirect)
        }
        return if (hits.size == 1 && hits[0] in known) hits[0] else intent
    }
}
