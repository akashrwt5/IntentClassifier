package com.example.nlu

import org.json.JSONObject
import java.io.InputStream

/**
 * Loader for a format-3.0 pack. This is the file that answers "what does Android
 * read instead of nlu_schema.json?" — the schema is the reference engine's
 * single blob; the pack ships it decomposed, and a client reads the pieces.
 *
 *   runtime/policies.json      thresholds, limits, per-intent confirmation
 *   runtime/guards.json        help-marker + polarity redirects
 *   runtime/plan_facts.json    intent -> capability (for availability checks)
 *   runtime/cascade.json       which stages are enabled
 *   keywords/<lang>.json       ordered rule pre-filter
 *   lexicons/<lang>.json       yes/no words, contractions
 *   capabilities/<cap>/capability.json   action keys  (-> NLUActionKey.kt)
 *   capabilities/<cap>/workflows.json    slots, confirmation, completion action
 *   capabilities/<cap>/responses/<lang>.json   the user-visible strings
 *   models/intent/<lang>/intent_classifier_weights_full.json
 *
 * `nlu_schema.json` is still present in the pack for the reference engine. Do
 * not read it from Android: it is the compiler's input shape, it mixes platform
 * config with content, and it is the file the split above exists to replace.
 */
class NluBundle(
    val thresholds: Thresholds,
    val limits: Limits,
    val confirmation: Map<String, String>,
    val intentCapability: Map<String, String>,
    val workflows: Map<String, Workflow>,
    val responses: Map<String, String>,
    val actionKeys: Set<String>,
    val keywords: KeywordMatcher,
    val guards: Guards,
    val normalizer: TextNormalizer,
    val affirmative: Set<String>,
    val negative: Set<String>,
    val semanticEnabled: Boolean,
) {

    data class Thresholds(
        val confidence: Double,
        val agreement: Double,
        val interrupt: Double,
        val oovReject: Double?,
        val oovBypass: Double?,
    )

    data class Limits(val maxSlotAttempts: Int, val sessionTimeoutS: Int)

    data class Slot(
        val name: String,
        val entity: String,
        val required: Boolean,
        val promptKey: String,
    )

    data class Workflow(
        val action: String?,
        val responseKey: String?,
        val slots: List<Slot>,
        val confirmationRequired: Boolean,
        val confirmPromptKey: String?,
    )

    fun response(key: String?): String? = key?.let { responses[it] }

    /** Every intent this pack can produce. */
    val knownIntents: Set<String> get() = intentCapability.keys

    companion object {

        /**
         * `open(path)` resolves a path INSIDE the unpacked pack — on Android,
         * typically `context.assets::open` after unzipping the .nlu, or a
         * FileInputStream over the extracted directory.
         *
         * Verify the pack's signature before calling this. The manifest and
         * `checksums_root` exist so a client can refuse a tampered pack, and a
         * loader that skips the check makes them decoration.
         */
        fun load(lang: String, open: (String) -> InputStream): NluBundle {
            val policies = readJson(open, "runtime/policies.json")
            val th = policies.getJSONObject("thresholds")
            val lim = policies.getJSONObject("limits")

            val confirmationObj = policies.getJSONObject("confirmation")
            val confirmation = confirmationObj.keys().asSequence()
                .associateWith { confirmationObj.getString(it) }

            val planFacts = readJson(open, "runtime/plan_facts.json").getJSONObject("intents")
            val intentCapability = planFacts.keys().asSequence()
                .associateWith { planFacts.getJSONObject(it).getString("capability") }

            // Walk only the capabilities this pack declares — never a directory
            // listing, which would silently pick up a stale folder.
            val capabilityIds = intentCapability.values.toSortedSet()
            val workflows = HashMap<String, Workflow>()
            val responses = HashMap<String, String>()
            val actionKeys = HashSet<String>()

            for (cap in capabilityIds) {
                readJson(open, "capabilities/$cap/capability.json")
                    .getJSONArray("actions").let { arr ->
                        for (i in 0 until arr.length()) {
                            actionKeys.add(arr.getJSONObject(i).getString("key"))
                        }
                    }

                val wf = readJson(open, "capabilities/$cap/workflows.json").getJSONObject("intents")
                for (intent in wf.keys()) {
                    workflows[intent] = parseWorkflow(wf.getJSONObject(intent))
                }

                val res = readJson(open, "capabilities/$cap/responses/$lang.json")
                for (k in res.keys()) responses[k] = res.getString(k)
            }

            val kwRules = readJson(open, "keywords/$lang.json").getJSONArray("rules").let { arr ->
                (0 until arr.length()).map { i ->
                    val o = arr.getJSONObject(i)
                    KeywordMatcher.Rule(
                        pattern = o.getString("pattern"),
                        intent = o.getString("intent"),
                        tier = o.getInt("tier"),
                        guards = o.optJSONArray("guards")?.let { g ->
                            (0 until g.length()).map(g::getString)
                        } ?: emptyList(),
                    )
                }
            }

            val guardsJson = readJson(open, "runtime/guards.json")
            val helpMarker = guardsJson.optJSONObject("help_marker")
            val pairsObj = helpMarker?.optJSONObject("pairs")
            val polarityArr = guardsJson.optJSONArray("polarity")

            val lex = readJson(open, "lexicons/$lang.json")
            val contractionsObj = lex.getJSONObject("contractions")

            val cascade = readJson(open, "runtime/cascade.json").getJSONArray("stages")
            var semantic = false
            for (i in 0 until cascade.length()) {
                val s = cascade.getJSONObject(i)
                if (s.getString("id") == "semantic") semantic = s.optBoolean("enabled", false)
            }

            return NluBundle(
                thresholds = Thresholds(
                    confidence = th.getDouble("confidence"),
                    agreement = th.getDouble("agreement"),
                    interrupt = th.getDouble("interrupt"),
                    // Half of this pair is worse than none of it — see NluEngine.
                    oovReject = if (th.has("oov_reject")) th.getDouble("oov_reject") else null,
                    oovBypass = if (th.has("oov_bypass")) th.getDouble("oov_bypass") else null,
                ),
                limits = Limits(
                    maxSlotAttempts = lim.getInt("max_slot_attempts"),
                    sessionTimeoutS = lim.getInt("session_timeout_s"),
                ),
                confirmation = confirmation,
                intentCapability = intentCapability,
                workflows = workflows,
                responses = responses,
                actionKeys = actionKeys,
                keywords = KeywordMatcher(kwRules),
                guards = Guards(
                    helpMarkers = helpMarker?.optString("markers")?.ifEmpty { null },
                    helpPairs = pairsObj?.let { p ->
                        p.keys().asSequence().associateWith { p.getString(it) }
                    } ?: emptyMap(),
                    polarity = polarityArr?.let { arr ->
                        (0 until arr.length()).map { i ->
                            val o = arr.getJSONObject(i)
                            Guards.PolarityRule(
                                pattern = o.getString("pattern"),
                                blocked = o.getString("blocked"),
                                redirect = o.getString("redirect"),
                            )
                        }
                    } ?: emptyList(),
                ),
                normalizer = TextNormalizer(
                    contractionsObj.keys().asSequence()
                        .associateWith { contractionsObj.getString(it) }
                ),
                affirmative = stringSet(lex, "affirmative"),
                negative = stringSet(lex, "negative"),
                semanticEnabled = semantic,
            )
        }

        /**
         * The weights and their temperature come from ONE file, always.
         *
         * The pack carries three temperatures, and they are all correct for
         * their own model:
         *
         *   models/intent/en/calibration.json          0.671   ONNX / server head
         *   intent_classifier_weights.json      (1592) 0.822   pruned device head
         *   intent_classifier_weights_full.json (5896) 0.544   full device head
         *
         * Mixing them compiles, runs, and produces plausible numbers — every
         * confidence is simply wrong, and no test can see it, because
         * temperature is rank-preserving: it never changes WHICH intent wins,
         * only how sure the system claims to be. That is blocker B8, and it
         * shipped once already. Hence: one file in, one matched set out.
         *
         * Prefer `_full.json`. It is 2.78 MB against the pruned head's 0.75 MB
         * and cuts out-of-scope utterances reaching a device action from 15.4%
         * to 8.7% — in a hearing-aid app, 2 MB is the cheaper side of that trade.
         */
        fun loadWeights(open: (String) -> InputStream, path: String): TfidfIntentClassifier.Weights {
            val w = readJson(open, path)

            val labelsArr = w.getJSONArray("labels")
            val labels = (0 until labelsArr.length()).map(labelsArr::getString)

            val vocabObj = w.getJSONObject("vocab")
            val vocab = HashMap<String, Int>(vocabObj.length() * 2)
            for (k in vocabObj.keys()) vocab[k] = vocabObj.getInt(k)

            val idfArr = w.getJSONArray("idf")
            val idf = DoubleArray(idfArr.length()) { idfArr.getDouble(it) }

            val coefArr = w.getJSONArray("coef")
            val coef = Array(coefArr.length()) { i ->
                val row = coefArr.getJSONArray(i)
                DoubleArray(row.length()) { j -> row.getDouble(j) }
            }

            val interceptArr = w.getJSONArray("intercept")
            val intercept = DoubleArray(interceptArr.length()) { interceptArr.getDouble(it) }

            require(coef.size == labels.size) { "coef rows ${coef.size} != labels ${labels.size}" }
            require(coef.all { it.size == idf.size }) { "coef width != idf length ${idf.size}" }

            return TfidfIntentClassifier.Weights(
                labels = labels,
                vocab = vocab,
                idf = idf,
                coef = coef,
                intercept = intercept,
                temperature = w.getDouble("temperature"),
            )
        }

        private fun parseWorkflow(o: JSONObject): Workflow {
            val completion = o.optJSONObject("completion")
            val confirm = o.optJSONObject("confirmation")
            val slotsArr = o.optJSONArray("slots")
            return Workflow(
                action = completion?.optString("action")?.ifEmpty { null },
                responseKey = completion?.optString("response")?.ifEmpty { null },
                slots = slotsArr?.let { arr ->
                    (0 until arr.length()).map { i ->
                        val s = arr.getJSONObject(i)
                        Slot(
                            name = s.getString("name"),
                            entity = s.optString("entity"),
                            required = s.optBoolean("required", true),
                            promptKey = s.getString("prompt"),
                        )
                    }
                } ?: emptyList(),
                confirmationRequired = confirm?.optBoolean("required", false) ?: false,
                confirmPromptKey = confirm?.optString("prompt")?.ifEmpty { null },
            )
        }

        private fun stringSet(o: JSONObject, key: String): Set<String> {
            val arr = o.optJSONArray(key) ?: return emptySet()
            return (0 until arr.length()).mapTo(HashSet()) { arr.getString(it) }
        }

        private fun readJson(open: (String) -> InputStream, path: String): JSONObject =
            open(path).use { JSONObject(it.bufferedReader().readText()) }
    }
}
