package com.example.nlu

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * The Stage-2 model: TF-IDF + LogisticRegression + temperature-scaled softmax.
 *
 * There is no TFLite artifact in the bundle, so Android computes this directly
 * from `models/intent/<lang>/intent_classifier_weights_full.json`. That is not a
 * workaround — the linear head is ~40 lines and runs in microseconds; a runtime
 * would buy nothing but a dependency.
 *
 * Reference implementation: `nlu_export/export_ios_weights.py::_device_logits`.
 * That function is what the shipped `temperature` was FIT on, so this file
 * matching it is what makes the confidence mean anything.
 *
 *   vec[j] = (1 + ln(count_j)) * idf[j]      sublinear TF
 *   vec   /= ||vec||                          L2 over the PRUNED subspace only
 *   logits = coef · vec + intercept
 *   probs  = softmax(logits / T)
 */
class TfidfIntentClassifier(private val w: Weights) {

    /**
     * Loaded verbatim from one weights JSON. All five arrays are a matched set —
     * see [NluBundle.loadWeights] for why they must never be mixed across files.
     */
    data class Weights(
        val labels: List<String>,
        val vocab: Map<String, Int>,
        val idf: DoubleArray,
        /** [nClasses][nFeatures] */
        val coef: Array<DoubleArray>,
        val intercept: DoubleArray,
        val temperature: Double,
    )

    val labels: List<String> get() = w.labels

    /** Exposed so a test can assert it against the golden file — see B8. */
    val temperature: Double get() = w.temperature

    /** Unigram slots only — what [oovRatio] is allowed to ask about. */
    private val unigrams: Set<String> = w.vocab.keys.filterTo(HashSet()) { !it.contains(' ') }

    /** Calibrated distribution over the full label space. Input must be normalized. */
    fun distribution(normalizedText: String): DoubleArray {
        val counts = HashMap<Int, Int>()
        for (token in tokenize(normalizedText)) {
            val j = w.vocab[token] ?: continue
            counts[j] = (counts[j] ?: 0) + 1
        }

        // Sparse throughout: an utterance touches a handful of the 5896 slots,
        // and materialising a dense vector per turn is pure allocation.
        val vec = HashMap<Int, Double>(counts.size * 2)
        var sumSq = 0.0
        for ((j, c) in counts) {
            val v = (1.0 + ln(c.toDouble())) * w.idf[j]
            vec[j] = v
            sumSq += v * v
        }
        val norm = sqrt(sumSq)
        if (norm > 0.0) for (j in vec.keys) vec[j] = vec[j]!! / norm

        val logits = DoubleArray(w.labels.size) { i ->
            var acc = w.intercept[i]
            val row = w.coef[i]
            for ((j, v) in vec) acc += row[j] * v
            acc
        }
        return softmax(logits, w.temperature)
    }

    /**
     * Share of tokens the featurizer has no slot for.
     *
     * A TF-IDF vector is a fixed set of slots; a token outside the vocabulary is
     * not weighed and dismissed, there is nowhere to put it. "turn off toshiba"
     * and "turn off" produce a bit-identical vector, so no threshold and no
     * amount of training separates them — the model is never asked. The unknown
     * word is itself evidence, and this is what recovers it.
     *
     * Unigrams only: a bigram is absent whenever either half is, so counting
     * bigrams double-charges the same unknown word.
     */
    fun oovRatio(normalizedText: String): Double {
        val tokens = UNIGRAM_TOKEN.findAll(normalizedText.lowercase())
            .map { it.value }.toList()
        if (tokens.isEmpty()) return 0.0
        return tokens.count { it !in unigrams }.toDouble() / tokens.size
    }

    companion object {
        /**
         * Must match `_swift_tokenize`: lowercase, split on non-alphanumerics,
         * unigrams plus ADJACENT bigrams. Single characters are kept.
         *
         * Note this is deliberately NOT sklearn's `\b\w\w+\b`. The exported head
         * is featurized this way and the temperature was fitted on these logits;
         * "improving" the tokenizer here silently decalibrates every confidence.
         */
        fun tokenize(text: String): List<String> {
            val words = text.lowercase().split(NON_ALNUM).filter { it.isNotEmpty() }
            if (words.size < 2) return words
            val out = ArrayList<String>(words.size * 2 - 1)
            out.addAll(words)
            for (i in 0 until words.size - 1) out.add(words[i] + " " + words[i + 1])
            return out
        }

        fun softmax(logits: DoubleArray, temperature: Double): DoubleArray {
            val max = logits.max()
            var sum = 0.0
            val out = DoubleArray(logits.size) { i ->
                val e = exp((logits[i] - max) / temperature)
                sum += e
                e
            }
            for (i in out.indices) out[i] /= sum
            return out
        }

        private val NON_ALNUM = Regex("[^a-z0-9]+")

        /**
         * sklearn's default token pattern — the OOV guard's view of a "word".
         *
         * `(?U)` is not optional. Java's `\w` is ASCII-only by default while
         * Python's `(?u)\w` is Unicode-aware, so without it "café" tokenizes as
         * ["caf"] on Android and ["café"] on the server. The English pack barely
         * notices; fr/de/da would compute a different out-of-vocabulary ratio
         * from the same sentence, and the guard would fire differently per
         * platform with nothing reporting it.
         */
        private val UNIGRAM_TOKEN = Regex("(?U)\\b\\w\\w+\\b")
    }
}
