package com.example.nlu

/**
 * Port of `packages/runtime/nlu_engine/text_norm.py::normalize_text`.
 *
 * WHY THIS EXISTS AT ALL
 * ----------------------
 * The TF-IDF vocabulary was built from NORMALIZED text at training time, so
 * "what's my battery" was fitted as `what is my battery`. A client that hands
 * the raw surface form to the tokenizer splits it as ["what", "s", "my", ...]
 * — "s" is not in the vocabulary, and neither are the bigrams "what s" / "s my".
 * The disambiguating features simply vanish.
 *
 * Measured on the 1470-row honest holdout (full-vocab device head):
 *
 *     raw lowercase only   0.9184   (1350 correct)
 *     normalized first     0.9204   (1353 correct)
 *     predictions differ:  9
 *
 * Three rows is not dramatic, but it is free and it grows with the share of
 * contracted speech — ASR output contracts far more than written training data.
 *
 * THE CONTRACT: this must stay byte-identical in behaviour to the Python and
 * the Swift. It is the one function every platform runs, and a divergence here
 * shows up as a confidence drift nothing tests.
 *
 * The contraction table is DATA, from `lexicons/<lang>.json` — do not inline an
 * English one here. Other languages contract differently (fr "j'ai", da "det's")
 * and hardcoding English is how negation suppression silently became a no-op
 * for three languages in the Python engine.
 */
class TextNormalizer(contractions: Map<String, String>) {

    private val table: Map<String, String> = contractions

    /** Longest-first alternation so a key that prefixes another cannot shadow it. */
    private val pattern: Regex? =
        if (table.isEmpty()) null
        else Regex(
            table.keys
                .sortedByDescending { it.length }
                .joinToString("|", prefix = "\\b(", postfix = ")\\b") { Regex.escape(it) }
        )

    /**
     * lowercase -> unify apostrophes -> expand contractions -> drop residual
     * apostrophes -> collapse whitespace.
     *
     * Idempotent: normalize(normalize(x)) == normalize(x).
     */
    fun normalize(text: String): String {
        var t = text.lowercase()
        for (variant in APOSTROPHE_VARIANTS) t = t.replace(variant, '\'')
        pattern?.let { rx -> t = rx.replace(t) { m -> table[m.value] ?: m.value } }
        t = t.replace("'", "")
        return WHITESPACE.replace(t, " ").trim()
    }

    private companion object {
        /** Curly and modifier apostrophes ASR and iOS keyboards emit. */
        val APOSTROPHE_VARIANTS = charArrayOf('’', 'ʼ', '`')
        val WHITESPACE = Regex("\\s+")
    }
}
