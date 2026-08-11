package com.example.nlu

import org.json.JSONObject
import java.io.File
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The one test that matters for a port.
 *
 * `golden_vectors.json` is generated from the SHIPPED weights by the Python
 * reference path (`nlu_export/export_ios_weights.py::_device_logits`, the same
 * function the temperature was fitted on). If this Kotlin computes the same
 * numbers, it is the same model. If it does not, the difference is a
 * decalibration that no amount of end-to-end testing would surface — every
 * intent would still be correct, only every confidence would be wrong, and
 * confidence is what the whole decision ladder is made of.
 *
 * Regenerate after any retrain:
 *     PYTHONPATH=packages/buildtime:packages/runtime \
 *         python scripts/gen_android_golden.py
 */
class GoldenParityTest {

    private val bundleDir = File(System.getProperty("nlu.bundle") ?: "../../dist/bundle-en")
    private val open: (String) -> java.io.InputStream = { File(bundleDir, it).inputStream() }

    private val golden = JSONObject(
        File("src/main/assets/golden_vectors.json").readText()
    )
    private val model = TfidfIntentClassifier(
        NluBundle.loadWeights(open, golden.getString("weights"))
    )
    private val bundle = NluBundle.load("en", open)
    private val tolerance = golden.getDouble("tolerance")

    @Test
    fun `the temperature in the golden file is the one in the weights we load`() {
        // The pack ships three temperatures for three different heads. Pairing
        // the wrong one with these coefficients still runs and still picks the
        // right intent — temperature is rank-preserving — so this assertion is
        // the only thing standing between a mismatch and silent decalibration.
        // That is blocker B8, and it shipped once.
        assertEquals(golden.getDouble("temperature"), model.temperature, 1e-9)
    }

    @Test
    fun `normalization matches the Python`() {
        forEachVector { v ->
            assertEquals(
                v.getString("normalized"),
                bundle.normalizer.normalize(v.getString("text")),
                "normalization drift on ${v.getString("text")}",
            )
        }
    }

    @Test
    fun `tokenization matches the Python`() {
        forEachVector { v ->
            val expected = v.getJSONArray("tokens").let { a ->
                (0 until a.length()).map(a::getString)
            }
            val actual = TfidfIntentClassifier.tokenize(v.getString("normalized"))
            assertEquals(expected, actual.take(expected.size),
                "tokenizer drift on ${v.getString("text")}")
        }
    }

    @Test
    fun `top intent and confidence match the Python to 1e-6`() {
        forEachVector { v ->
            val probs = model.distribution(v.getString("normalized"))
            val top = probs.indices.maxBy { probs[it] }
            assertEquals(v.getString("intent"), model.labels[top],
                "intent drift on ${v.getString("text")}")
            val delta = abs(probs[top] - v.getDouble("confidence"))
            assertTrue(delta < tolerance,
                "confidence drift on ${v.getString("text")}: " +
                    "${probs[top]} vs ${v.getDouble("confidence")} (delta $delta)")
        }
    }

    @Test
    fun `out-of-vocabulary ratio matches the Python`() {
        forEachVector { v ->
            val delta = abs(model.oovRatio(v.getString("normalized")) - v.getDouble("oov_ratio"))
            assertTrue(delta < tolerance, "oov drift on ${v.getString("text")}")
        }
    }

    // ---- the contract the ladder depends on ---------------------------------

    @Test
    fun `the out-of-vocabulary guard is loaded as a PAIR or not at all`() {
        // Half of this guard is worse than none of it: `oovReject` alone refuses
        // every command carrying an entity value, because entity values are
        // out-of-vocabulary by nature. "send a message to john" is 25% unknown
        // and entirely real.
        assertEquals(
            bundle.thresholds.oovReject == null,
            bundle.thresholds.oovBypass == null,
            "the OOV guard loaded incomplete — refuse to run rather than run half of it",
        )
    }

    @Test
    fun `confirmation is authored, never confidence-driven`() {
        val alwaysConfirm = bundle.confirmation.filterValues { it == "always" }.keys
        assertEquals(setOf("Cmd.SendMessage"), alwaysConfirm)
        // The two places the pack states this must agree — a client reading
        // workflows must confirm exactly where a client reading policies does.
        val fromWorkflows = bundle.workflows.filterValues { it.confirmationRequired }.keys
        assertEquals(alwaysConfirm, fromWorkflows)
        // And nothing may express a conditional confirmation.
        assertTrue(bundle.confirmation.values.none { it == "when_ambiguous" })
    }

    @Test
    fun `every workflow action exists as a declared action key`() {
        // This is what makes NLUActionKey a real contract rather than a comment:
        // a workflow that completes into an action nobody declared would fail
        // silently at runtime as an unhandled `when` branch.
        val missing = bundle.workflows.values.mapNotNull { it.action }
            .filter { it !in bundle.actionKeys }
        assertTrue(missing.isEmpty(), "workflow actions absent from capability.json: $missing")
    }

    @Test
    fun `every prompt and response key resolves to a string`() {
        val dangling = buildList {
            for ((intent, wf) in bundle.workflows) {
                wf.responseKey?.let { if (bundle.response(it) == null) add("$intent -> $it") }
                if (wf.confirmationRequired) {
                    wf.confirmPromptKey?.let { if (bundle.response(it) == null) add("$intent -> $it") }
                }
                wf.slots.forEach {
                    if (bundle.response(it.promptKey) == null) add("$intent -> ${it.promptKey}")
                }
            }
        }
        assertTrue(dangling.isEmpty(), "unresolved response keys: $dangling")
    }

    private fun forEachVector(block: (JSONObject) -> Unit) {
        val arr = golden.getJSONArray("vectors")
        for (i in 0 until arr.length()) block(arr.getJSONObject(i))
    }
}
