# Memory: Architectural Decisions (ADR log)

> Single responsibility: durable "why" behind non-obvious choices. Append new
> decisions as short ADRs; use the `engineering:architecture` skill for full
> ADRs. Keep entries terse — link to the detailed doc.

## ADR-001 — TF-IDF + LogisticRegression -> ONNX for the core classifier
**Status:** Accepted. **Why:** ~16KB, offline, deterministic, tiny latency on
device; no neural runtime needed. Trade-off: less semantic generalization than
embeddings — mitigated by the semantic-rescue layer.

## ADR-002 — No torch/transformers on the inference path
**Status:** Accepted. **Why:** on-device size/latency. Embeddings run through
ONNX; `torch`/`coremltools` are export-only (macOS/CI).

## ADR-003 — Temperature scaling over isotonic calibration
**Status:** Accepted. **Why:** isotonic calibration was **not rank-preserving**,
causing server↔device argmax disagreements and worse ECE on holdout. Temperature
scaling is rank-preserving by design and empirically better-calibrated. Detail:
`multilingual/TEMPERATURE_SCALING_DECISION.md`,
`multilingual/MODEL_CALIBRATION_DECISION.md`.

## ADR-004 — Per-language single-model calibration (Option B)
**Status:** Accepted. **Why:** best server↔device parity and smallest size vs.
ensemble isotonic. Residual parity issues traced to tokenizer divergence and
argmax-before-vs-after-calibration. Detail: `MODEL_CALIBRATION_DECISION.md` §4-5.

## ADR-005 — Static ONNX shapes, batch size 1
**Status:** Accepted (hard constraint). **Why:** ANE eligibility + deterministic
CoreML conversion. Consequence: embed one sentence at a time everywhere.

## ADR-006 — CoreML FP16 mlprogram, temperature in metadata, logits output
**Status:** Accepted. **Why:** FP16 halves size with acc Δ ≈ 0; shipping logits +
temperature lets the device own `softmax(logits/T)` and keeps parity. Detail:
`multilingual/COREML_RESULTS.md`, `docs/coreml-conversion-guide.md`.

## ADR-007 — Semantic rescue as a separate low-confidence layer
**Status:** Accepted. **Why:** keeps the fast TF-IDF path primary; only pays
embedding cost when confidence/margin gates fail. Threshold via
`nlu_schema.json:semantic_threshold`.

## ADR-008 — Tooling lenient-by-design, tighten incrementally
**Status:** Accepted. **Why:** never-linted code must stay green from day one;
enforce a lean Ruff rule set now; formatting is **format-on-touch** (darker:
Black on changed lines only) so legacy history isn't reformatted (avoids conflicts
across the many in-flight branches); MyPy stays non-blocking with per-module
tightening. Detail: `pyproject.toml`, `CONTRIBUTING.md`, `.pre-commit-config.yaml`.

## ADR-009 — MCP transport: code graph stays local; web uses Context7 only
**Status:** Accepted. **Why:** the code-graph server (CodeGraphContext, `cgc`)
runs as a **local stdio** process and indexes this **proprietary** codebase.
claude.ai web sessions can only reach **remote HTTPS** MCP servers, so putting the
code graph on the web would mean exposing a private code index over the internet
(tunnel or hosted box) — an unacceptable default. Decision: use the code graph
**only on desktop** clients (Claude Code / Desktop in PyCharm); in web sessions
use **Context7** (hosted at `https://mcp.context7.com/mcp`) for library docs only.
Revisit only if a private, authenticated hosting path is explicitly approved.


## ADR-010 — Review-F5 platform ADRs 001–005 ratified
**Status:** Accepted (owner ratification, 2026-07-14). **What:** the five
Review-F5 platform ADRs move Proposed → Accepted as written:
ADR-001 shared-runtime strategy (Option B now; Rust core only at the
Android-multi-turn trigger), ADR-002 capability & action-execution SDK,
ADR-003 conversation orchestration (Orchestrator/Planner/Dialogue/Policy),
ADR-004 GenAI routing & cloud escalation (narration-only, utterance-only
egress), ADR-005 NLU Bundle spec & compiler. **Why:** unblocks gated Phase-1
work (bundle schemas/compiler, capability repartition, contract freezing).
Phase 2 (Rust core / Android) remains **trigger-gated** despite ratification;
signing keys, consent/legal, restructure, and label-space changes remain
separately approval-gated (EXECUTION_STATUS ND-2/3/8/9). Detail:
`docs/Review-F5/adr-00{1..5}-*.md`.


## ADR-011 — No Rust core; Python + native iOS/Android is the strategy
**Status:** Accepted (owner directive, 2026-07-14). **What:** the shared Rust
runtime sketched in ADR-001 Phase 2 is removed from the active roadmap. The
platform remains: Python for training, compiler, tooling, and the reference
NLU engine; native iOS and native Android implementations on device. **Why:**
owner decision — explore a shared core later only if required, on explicit
request; the ADR-001 trigger (Android multi-turn dialogue) is no longer
sufficient on its own. **Consequences:** cross-platform parity continues to
be guaranteed by the contract + fixture approach (runtime-contract-v1.md,
golden bundles, parity CSVs, conformance tests) rather than by a shared
binary. No Rust code exists in the repo; none is planned.


## ADR-012 — Slot filling must never fabricate a value from a non-answer
**Status:** Accepted (2026-07-26). **What:** while a slot is being awaited, the
recogniser produces a value only when the input is genuinely a valid value for
that slot's type; a turn that is not an answer is routed, never coerced. Three
concrete rules: (1) enum fuzzy matching excludes function words — "the" is not a
typo of the memory "three", so an off-topic sentence can no longer fill a slot
via a stopword (`entities.py extract_enum`, `_DEFAULT_FUZZY_STOPWORDS`); (2) the
permissive `dateparser` fallback fires only on text carrying a digit — every
word-based temporal form is already owned by the grammar, so "no" can no longer
be read as November (`entities.py extract_datetime`, section 8); (3) a
slot-filling turn is interpreted in fixed precedence — valid strict answer →
genuine high-confidence interruption → pure cancellation ("no"/"cancel", via
`_is_cancel`) → otherwise a no-match that re-prompts and, after
`MAX_SLOT_ATTEMPTS`, falls back (`engine.py _handle_slot_filling`).
**Why:** two production wrong-actions ("who is the prime minister of india" ->
memory=three; "no" -> a reminder for a date never given) shared one cause — a
lenient matcher inventing a value for a turn that was not an answer. A
confidence threshold cannot separate these (a genuine typo scored 0.70, the
"the"->"three" collision 0.60), so the fix is structural, not a threshold.
**Consequences:** cancellation cues are content-owned (`schema cancel_cues`,
English default in `engine.py`); a word that is itself a real command ("stop" ->
`streaming.session.stop`) interrupts rather than cancels, because interruption
has precedence over the meta layer. Regression fixtures:
`tests/test_slot_value_validation.py` (recogniser level) and
`tests/test_slot_filling_no_answer.py` (engine level). Follow-ups considered but
not taken here: phonetic (metaphone) matching in place of Levenshtein for a
voice/ASR domain; the grammar owning absolute numeric dates so `dateparser` can
be retired; progressive re-prompts on repeated no-match.


## ADR-013 — Apostrophe/tokenizer parity: one surface-form normaliser for train + inference
**Status:** Accepted (2026-07-26). **What:** the English TF-IDF path now applies
a shared `normalize_text` (`packages/runtime/nlu_engine/text_norm.py`) at BOTH
training (`nlu_training/train.py`, before fit + ONNX export) and inference
(`nlu_engine.classifier`, TF-IDF/ONNX path only — the keyword stage still matches
raw text). It expands English contractions ("what's" -> "what is") and strips
residual apostrophes ("mom's" -> "moms"). **Why:** skl2onnx's ONNX tokenizer does
not replicate sklearn's `\b\w\w+\b` behaviour around the apostrophe, so the
exported `model.onnx` and the in-memory `pipeline.pkl` gave DIFFERENT predictions
for any apostrophe input — e.g. "what's up" was `sys.oos.fallback` in the pkl but
`device.volume.increase` in ONNX. Folding the apostrophe out before the vectorizer
makes both tokenisers see an identical surface form, restoring numeric parity.
This mirrors the accent-folding fix already shipped for the multilingual models
(`multilingual/text_norm.py`); the two normalisers should be consolidated.
**Consequences:** (1) verified — ONNX and pkl now agree on all apostrophe inputs;
"what's up"/"can you help me" -> OOS; "what's my battery" -> battery. (2) The
normalisation exposed a genuine hidden holdout leak ("i'm lost in this app" vs the
trained "i am lost in this app"); removed from train per the never-touch-holdout
rule. (3) FOLLOW-UP REQUIRED for on-device parity: the iOS/CoreML exporters
(`nlu_export/export_weights.py`, `export_ios_weights.py`) and the Swift runtime
must apply the SAME normalisation, or iOS will reintroduce the divergence. (4)
Related pre-existing divergence to revisit: server ONNX uses `min_df=2` while the
multilingual and iOS-export recipes use `min_df=1`.


## ADR-014 — CoreML ships INSIDE the signed .nlu (Fat Bundle), as `coreml_artifact`
**Status:** Accepted (2026-07-27). **What:** the release pack carries the CoreML
`.mlpackage` inside the signed bundle rather than as a loose side artifact. It is
written as `models.intent.<lang>.coreml_artifact` — a schema-legal sibling of the
ONNX `artifact` on the intent entry — NOT as a new `models.coreml` STAGE. The
`.mlpackage` directory is copied into the staging tree by
`scripts/ci/assemble_pack.py --coreml`, so its files are checksummed and signed
with the rest of the pack; `release-pack.yml` passes `--coreml` when the
(macOS, `continue-on-error`) export job produced one. **Why:** `bundle.schema.json`
already declares the `coreml_artifact` property and the validator maps `.mlpackage`
files, so a Fat Bundle assembles and verifies today; distributing the iOS model in
the same signed unit as the ONNX avoids a second unsigned channel. This supersedes
the earlier stance (assemble_pack *refused* `--coreml`). **Consequences:** (1)
`models` stays the closed set (intent/embedder/semantic_head);
`test_models_schema_really_forbids_a_coreml_stage` still guards that CoreML never
becomes its own stage. (2) `test_release_pack.py` was updated to the new contract
(`test_coreml_is_packaged_into_the_bundle`, `test_release_job_passes_coreml_to_the_packer`).
(3) OPEN CAVEAT: the `.mlpackage` `nlu_export.export_coreml` currently emits derives
from the repo-committed DEVICE weights, not the ONNX trained in the same run — so the
bundled CoreML model is an iOS convenience artifact, NOT proof of ONNX↔CoreML parity.
Retargeting the exporter is the follow-up that turns this into a real parity gate.


## ADR-015 — TFLite is the linear HEAD, exported from the fitted model (not from ONNX)
**Status:** Accepted (2026-07-27). **What:** the TFLite intent artifact is the
classifier HEAD only — float TF-IDF vector in, logits out — a single Dense layer
seeded directly from the fitted sklearn `LogisticRegression` (`coef_`/`intercept_`
in `pipeline.pkl`). `nlu_export/export_tflite.py` emits `model.tflite` (fp32) and
`model_int8.tflite` (dynamic-range int8) beside `model.onnx`; both ride in the
signed `.nlu` as `models.intent.<lang>.tflite_artifact` / `tflite_int8_artifact`
(the ADR-014 Fat Bundle mechanism). **Why:** the shipping ONNX graph is string-in
and built from ONNX-ML ops (`StringNormalizer`/`Tokenizer`/`TfIdfVectorizer`/
`LinearClassifier`) with no TFLite equivalents, so the full pipeline is not
representable in TFLite — and transcoding a downstream artifact is exactly what
introduces drift. The on-device contract already splits vectorisation from
classification (`intent_classifier_weights.json` ships `vocab`+`idf`; the CoreML
head is "float-vector input"), so TFLite mirrors it: native TF-IDF on-device, the
linear head in the model. Since the head is a linear map, fp32 TFLite is
bit-parity with the ONNX `LinearClassifier` (which `train.py` emits with
`raw_scores=True`) — parity by construction, max |Δlogit| ≈ 1e-6.
**Consequences:** (1) TensorFlow is an EXPORT-ONLY dep (allowlisted in
`test_declared_dependencies.py`), never on the inference path; it runs on Linux so
TFLite export lives in the release-pack `train-gate` job, not a macOS runner.
(2) The exporter self-checks fp32 parity and fails on divergence; int8 keeps argmax
(guarded by `tests/test_tflite_export.py`). (3) TRADE-OFF: Android must run the
same native TF-IDF (vocab+idf + the ADR-013 surface-form normaliser) that iOS does;
a self-contained string-in TFLite was rejected as higher-gap (TF-IDF≠sklearn) and
requiring the Flex delegate. (4) `bundle.schema.json` gained `tflite_artifact` /
`tflite_int8_artifact` as sibling properties of the intent entry; `models` stays
the closed stage set.


## ADR-016 — Two CoreML intent heads (pruned + full vocab); both from the trained pipeline
**Status:** Accepted (2026-07-27). **What:** the release ships TWO CoreML intent
heads, both regenerated from the run's trained `pipeline.pkl` via
`export_ios_weights.py`: the default top-per-class PRUNED head
(`IntentClassifier.mlpackage`, ~1.3k features / ~290 KB — the small on-device
default) and a FULL-vocab head (`IntentClassifier_full.mlpackage`, 4718 features /
~1 MB, matching the ONNX/TFLite feature space), produced with
`--top-per-class 0`. `export_coreml.py` builds the full head whenever
`intent_classifier_weights_full.json` exists; both ride in the signed `.nlu` as
`models.intent.<lang>.coreml_artifact` / `coreml_full_artifact` (ADR-014 Fat
Bundle mechanism). **Why:** the CoreML head is FP32 and its size is set purely by
vocabulary; the pruning was a deliberate on-device size win, but consumers also
want a head that is feature-for-feature comparable to ONNX/TFLite. Shipping both
keeps the small default and offers the parity head without forcing a size
regression. Regenerating from `pipeline.pkl` (uploaded from train-gate) also
closes the old "CoreML derives from committed device weights" caveat — the heads
now come from THIS run's model. **Consequences:** (1) `bundle.schema.json` gained
`coreml_full_artifact`; `models` stays the closed stage set (the schema-forbids
test still holds). (2) T is refit per variant on its own device-equivalent logits
(pruned ≠ full); a stale T mis-tunes the 0.70 gate. (3) The full head requires the
Android/iOS runtime to build the full 4718-dim TF-IDF vector (same native TF-IDF
counterpart as ADR-015). (4) The macOS `coreml-export` job stays
`continue-on-error` (no Core ML runtime off-Apple-silicon); CoreML `.mlpackage`
writing needs macOS `libmodelpackage`, so the heads are built there, not on Linux.


## Related memory

Training/calibration -> `training.md` · Mobile -> `mobile.md` · Roadmap ->
`roadmap.md`.
