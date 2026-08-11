package com.example.nlu

/**
 * Stage 1: the hand-authored rule pre-filter, from `keywords/<lang>.json`.
 *
 * The compiler has already flattened the schema's `exact` / `contains` / `regex`
 * forms into ONE ordered regex list, so a client only has to walk it in order.
 * That order is load-bearing — first match wins — which is exactly why keywords
 * ship as a single top-level file and are not split per capability.
 *
 *   tier 1  anchored (`^mute$`)       — was an `exact` trigger
 *   tier 2  free regex                — was a `regex` trigger
 *   guards  exclusion patterns        — a hit is suppressed if any guard matches
 *
 * A RULE PRODUCES NO CONFIDENCE. It is deterministic; it cannot express a
 * probability. This class returns an intent or null, and nothing else, on
 * purpose. The engine that used to attach a constant here (`regex` = 0.75) was
 * comparing a made-up number against thresholds fitted on calibrated softmax
 * probabilities — two incompatible scales in one field, which is how "increase
 * volume" ended up asking the user for confirmation at a model confidence of
 * 0.9992.
 *
 * Rules match RAW text, not normalized text: the authors wrote them against
 * what a user actually says.
 */
class KeywordMatcher(rules: List<Rule>) {

    data class Rule(
        val pattern: String,
        val intent: String,
        val tier: Int,
        val guards: List<String> = emptyList(),
    )

    private class Compiled(
        val regex: Regex,
        val intent: String,
        val tier: Int,
        val guards: List<Regex>,
    )

    private val compiled: List<Compiled> = rules.map {
        Compiled(
            regex = Regex(it.pattern),
            intent = it.intent,
            tier = it.tier,
            guards = it.guards.map(::Regex),
        )
    }

    /** Match tier of the last hit — telemetry and interrupt logic only. */
    var lastTier: Int? = null
        private set

    /** The intent of the first rule that fires, or null. */
    fun match(rawText: String): String? {
        val t = rawText.lowercase().trim()
        lastTier = null
        for (rule in compiled) {
            if (!rule.regex.containsMatchIn(t)) continue
            if (rule.guards.any { it.containsMatchIn(t) }) continue
            lastTier = rule.tier
            return rule.intent
        }
        return null
    }
}
