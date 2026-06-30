# Plan Prompt — Multilingual Semantic Rescue for IntentClassifier

Use this as the opening prompt for an agent session tasked with **planning**
(not yet implementing) the multilingual semantic rescue upgrade.

---

## ROLE

You are a **Principal ML Engineer** with deep expertise in sentence embeddings,
multilingual NLP, and on-device inference. You design production-grade ML
pipelines that ship to real devices — you think about artifact sizes, latency
budgets, fallback paths, and testability before you write a single line of code.

Your job in this session is to produce a **detailed implementation plan** — a
written document committed to the repository. You are NOT expected to implement
anything beyond the plan itself. When you finish, a separate implementation agent
will execute the plan step by step.

**Planning agent rules:**
- Read before you write. Explore the codebase thoroughly before proposing any change.
- Every claim in the plan must be backed by a file path and line number you have
  verified by reading.
- Flag every assumption you cannot verify from the code as an open question that
  the implementation agent must resolve before proceeding.
- Do not invent abstractions. Propose the smallest change that achieves the goal.
- Quantify everything you can: artifact sizes, expected latency, accuracy targets,
  number of files that change.

---

## CONTEXT — read before exploring

### Repository

`akashrwt5/IntentClassifier` — a Python NLU pipeline for a hearing-aid companion
app. The app classifies voice utterances into intents (volume up/down, set reminder,
change memory program, etc.) and extracts slot values.

**Working branch:** `claude/multilingual-nlu-status-check-s7ggcw/MulitlingualSemanticSupport`
Create this branch off `claude/multilingual-nlu-status-check-s7ggcw` if it does not yet exist:
```bash
git fetch origin
git checkout claude/multilingual-nlu-status-check-s7ggcw
git checkout -b claude/multilingual-nlu-status-check-s7ggcw/MulitlingualSemanticSupport
```
Always develop and commit on this branch. Do not push to main or to the parent
branch without explicit permission.

### The four-stage NLU pipeline

```
Stage 0  Keyword triggers   — regex short-circuit, fires first, confidence = 1.0
Stage 1  Entity extraction  — slot values (datetime, enum entities)
Stage 2  TF-IDF + LogReg    — primary intent classifier
Stage 3  Semantic rescue    — embedding-based fallback when Stage 2 is uncertain
```

Stage 3 activates only when TF-IDF confidence falls below `confidence_threshold`
(currently 0.6 per language, from `config/calibration.json`). It embeds the
utterance, applies a trained logistic head, and either rescues the intent or lets
it fall to GenAI.

### The problem to solve

**Stage 3 is English-only.** The current semantic artifacts are:

```
models/minilm-l6-v2.onnx        — INT8 MiniLM-L6-v2 English sentence encoder (~23 MB)
models/minilm-vocab.txt          — WordPiece vocabulary for the runtime tokeniser
models/semantic_head.npz         — logistic head weights trained on English embeddings
```

`SemanticFallback` in `scripts/nlu/semantic.py` uses this encoder. The custom
WordPiece tokeniser in the same file (`_tokenise`, `_wordpiece`) works character
by character from the English vocabulary — it will tokenise French/German/Danish
text but with high UNK rates, degrading embedding quality significantly.

When `NLUEngine` is instantiated with `language="fr"` (or `de`, `da`), it loads
the same English `SemanticFallback`. Low-confidence French utterances that reach
Stage 3 get embeddings computed from a mismatched tokeniser, the head was trained
on English embedding geometry, and rescue accuracy is materially worse than for
English — the system effectively has no semantic safety net for non-English users.

### Critical architecture constraint — DO NOT TOUCH EXISTING SEMANTIC CODE

**The existing semantic implementation is off-limits.** Do not propose any changes to:
- `scripts/nlu/semantic.py`
- `scripts/nlu/engine.py`
- `scripts/train_semantic_head.py`
- `scripts/build_semantic_index.py`
- `scripts/test_semantic.py`
- `scripts/debug_semantic_scores.py`
- `models/semantic_head.npz`
- `models/minilm-l6-v2.onnx`
- `models/minilm-vocab.txt`

These files and artifacts serve the English production path and must remain
completely untouched. The English Stage 3 pipeline must continue to work exactly
as it does today, with zero modifications.

**All multilingual semantic code lives in a new, self-contained module:**
```
multilingual/SemanticSupport/
```

The existing code is reference material only. Read it to understand patterns,
class shapes, artifact formats, and the head `.npz` schema — then write your own
parallel implementation from scratch in the new folder. The implementation agent
will write new code that mirrors the design of the existing code without touching
the original files.

### What already exists (read as reference — do not modify)

| File | Read for | Do not modify |
|------|----------|---------------|
| `scripts/nlu/semantic.py` | `SemanticFallback` class design, ONNX path, tokeniser pattern, `.npz` head format | ✗ |
| `scripts/nlu/engine.py` lines 220–245 | `_load_semantic` method — how the engine instantiates Stage 3 | ✗ |
| `scripts/nlu/engine.py` lines 510–540 | Stage 3 rescue logic, `AGREEMENT_THRESHOLD`, `semantic_threshold` | ✗ |
| `scripts/train_semantic_head.py` | How the head is trained, what it outputs, `Default Fallback Intent` handling | ✗ |
| `multilingual/train_multilingual.py` | LANGUAGES dict (~line 106), data pipeline pattern | ✗ |
| `config/calibration.json` | Per-language thresholds and temperatures | ✗ |
| `data/localization/` | Language file structure | ✗ |
| `multilingual/models/` | Existing per-language TF-IDF artifact layout | ✗ |
| `multilingual/data/` | Per-language training CSVs (training data for the new head) | read only |

Read `docs/adding-a-new-language.md`, section "Deliberately deferred capabilities",
for the deferred capability this plan addresses.

### Key constraints

1. **Latency budget:** The current English MiniLM ONNX path takes ~8ms embed +
   <1ms head. The multilingual encoder must fit within ~15ms embed on a server CPU.
   The plan must include benchmark numbers for each timing component; estimates
   flagged as such are acceptable, but guesses without justification are not.
2. **Artifact size:** The plan must state FP32 size, INT8 size, peak memory usage
   during inference, and the quantisation accuracy trade-off (macro F1 delta FP32 vs.
   INT8). Justify the chosen format.
3. **Offline-first — no runtime downloads.** All model and tokenizer artifacts must
   exist locally before the application starts. The runtime must load only local files.
   Nothing may be fetched from the network at inference time, ever.
4. **Reproducible builds.** Running `scripts/SemanticSupport/download_models.py`
   multiple times must produce byte-identical artifacts. After the initial setup the
   repository must be fully usable offline.
5. **Zero changes to the English path.** English users must continue hitting the
   existing `SemanticFallback` in `scripts/nlu/semantic.py` unchanged. The new
   multilingual module is completely additive. The English production path must remain
   byte-for-byte compatible before and after this change.
6. **Core ML forward-compatibility.** Although iOS is out of scope for this plan,
   the selected ONNX graph and any preprocessing steps must remain compatible with
   future Core ML conversion. Avoid runtime dependencies or graph ops that would make
   Core ML deployment significantly harder.
7. **iOS is otherwise out of scope for this plan.** This plan covers the Python
   server pipeline only.
8. **No dependency on sentence-transformers at runtime.** The production path must
   use the ONNX encoder to avoid a heavy dependency. `sentence-transformers` is
   acceptable only for training and embedding-generation scripts.

---

## WHAT THE PLAN MUST COVER

Produce a plan document at `docs/MULTILINGUAL_SEMANTICS_IMPLEMENTATION.md` that
addresses each of the following sections. Sections may be in any order that makes
logical sense.

### 1. Encoder

**The encoder is fixed: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.**
Do not evaluate alternative encoders unless a blocking technical issue is encountered
(e.g., the model cannot be exported to ONNX, INT8 quantisation degrades macro F1 by
more than 3 points, or a hard dependency conflict arises). If a blocking issue occurs,
document it explicitly and escalate before switching models.

The plan must document the following properties of this encoder:
- Hugging Face model ID and the `optimum-cli` or `torch.onnx.export` command used to
  produce the ONNX graph
- FP32 ONNX file size and INT8 quantised file size (use `onnxruntime.quantization`)
- Embedding dimension: **384** (same as English MiniLM-L6-v2 — confirm this)
- Peak memory during ONNX inference (measure with `tracemalloc` or `psutil`)
- Estimated CPU inference latency per utterance (benchmark with 100 warm runs)
- Vocabulary coverage for French, German, and Danish (tokenisation UNK rate on a
  100-sentence sample per language — must be below 1%)
- Quantisation accuracy trade-off: macro F1 delta between FP32 and INT8 on the
  held-out multilingual split (acceptable if delta < 1.5 points)

### 2. Head retraining strategy

A new logistic head must be trained on multilingual embeddings from the chosen
encoder. The head must be separate from the English `semantic_head.npz` and live
in `multilingual/SemanticSupport/models/`.

The plan must specify:
- What training data to use — the existing `multilingual/data/` CSVs (fr, de, da,
  en) are the natural source; verify row counts are sufficient per intent
- Whether to train one shared head across all languages or per-language heads, and why
- How to preserve the `Default Fallback Intent` class as an explicit out-of-scope
  class (mirror the existing approach from `train_semantic_head.py`)
- What holdout split to use and what minimum macro F1 to require per language
  before the head is considered shippable

### 3. New module layout: `multilingual/SemanticSupport/`

**All new code goes here.** The plan must specify the complete folder structure and
the purpose of every file, for example:

```
multilingual/SemanticSupport/
  __init__.py
  semantic.py          — MultilingualSemanticFallback class (mirrors SemanticFallback design)
  tokeniser.py         — Multilingual tokeniser for the encoder's vocab format
  models/
    paraphrase-multilingual-MiniLM-L12-v2.onnx        — FP32 ONNX encoder
    paraphrase-multilingual-MiniLM-L12-v2.int8.onnx   — INT8 quantised encoder (if accuracy acceptable)
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    vocab.txt                                          — if applicable for the tokeniser
    semantic_head_multilingual.npz                     — logistic head trained on multilingual embeddings
    manifest.json                                      — SHA-256 checksums of all artifacts above
scripts/SemanticSupport/
  download_models.py              — downloads encoder + tokenizer, exports ONNX, quantises, writes manifest
  train_multilingual_semantic_head.py  — trains the logistic head on multilingual embeddings
  test_multilingual_semantic.py   — validates per-language accuracy + smoke tests
  debug_multilingual_semantic.py  — diagnostic tool (mirrors debug_semantic_scores.py)
```

The plan must justify this layout and note any deviations from the example above.

### 4. `MultilingualSemanticFallback` class design

Describe the design of the new class in `multilingual/SemanticSupport/semantic.py`.
It mirrors the design of `SemanticFallback` in `scripts/nlu/semantic.py` but is
written independently — use the existing class as a reference, not a base class.

The plan must specify:
- Constructor signature — what parameters it takes (head path, encoder path, threshold)
- How it loads and warms up the ONNX encoder session
- How the tokeniser is selected for the chosen multilingual encoder (the existing
  WordPiece tokeniser may need adaptation for a different vocabulary format —
  describe what changes are required)
- The `classify(text) -> (intent, confidence)` interface (must be identical to
  the existing `SemanticFallback.classify` so `engine.py` can call either)
- Whether `is_available()` should surface which languages are covered

### 5. Engine integration — the only touch point on existing code

The only connection between the new module and the existing engine is in
`scripts/nlu/engine.py`'s `_load_semantic` method (lines ~220–239). The plan must
describe the smallest possible change to this method:
- How it detects that `language != "en"` and loads `MultilingualSemanticFallback`
  from `multilingual/SemanticSupport/semantic.py` instead of the English `SemanticFallback`
- How it falls back gracefully if the multilingual artifacts are missing (log a
  warning and leave `self.semantic = None`, exactly as the English path does today)
- What import pattern to use (inline import inside `_load_semantic` to avoid
  circular deps, mirroring the existing `from .semantic import SemanticFallback`)
- Whether `AGREEMENT_THRESHOLD` (0.50) and `semantic_threshold` generalise to
  non-English languages or need per-language values in `config/calibration.json`
- What logging changes are needed so telemetry can distinguish English vs.
  multilingual rescue

Note: `_load_semantic` is the **only method in `engine.py` that may be touched**.
The Stage 3 rescue logic itself (lines 512–535) must not change.

### 6. Artifact layout and size budget

State the complete on-disk layout of all new artifacts after the plan is
implemented. English artifacts are shown for reference — they must remain
unchanged:

```
models/                                              ← English artifacts — DO NOT TOUCH
  semantic_head.npz
  minilm-l6-v2.onnx
  minilm-vocab.txt

multilingual/SemanticSupport/models/                 ← All new artifacts go here
  paraphrase-multilingual-MiniLM-L12-v2.onnx        ← FP32 encoder
  paraphrase-multilingual-MiniLM-L12-v2.int8.onnx   ← INT8 quantised encoder (if shipped)
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  vocab.txt                                          ← if applicable
  semantic_head_multilingual.npz                     ← logistic head
  manifest.json                                      ← SHA-256 of every file above
```

The plan must include a table with measured or estimated values for each artifact:

| File | FP32 size | INT8 size | Memory at inference | Embed latency (CPU) | Embed dimension |
|------|-----------|-----------|--------------------|--------------------|-----------------|
| `paraphrase-multilingual-MiniLM-L12-v2.onnx` | ? MB | — | ? MB | ? ms | 384 |
| `paraphrase-multilingual-MiniLM-L12-v2.int8.onnx` | — | ? MB | ? MB | ? ms | 384 |
| `semantic_head_multilingual.npz` | ? KB | — | negligible | <1 ms | — |

Flag any value that is estimated rather than measured. Also state:
- Which format (FP32 or INT8) is recommended for production and why
- Quantisation trade-off: macro F1 delta FP32 vs. INT8 on held-out multilingual split
- How `manifest.json` is structured (JSON dict of `{filename: sha256_hex}`) and
  which script generates it (`download_models.py`)

### 7. Scripts: download, train, test

Describe each new script under `scripts/SemanticSupport/`. For each script the plan
must state the exact command to run it and list every output artifact it produces.

**`download_models.py`** ← the required first-run setup script
- Downloads `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` from
  Hugging Face (model weights + all tokenizer files)
- Exports to ONNX via `optimum-cli export onnx` or equivalent
- Produces INT8 quantised ONNX via `onnxruntime.quantization.quantize_dynamic`
- Stores all artifacts under `multilingual/SemanticSupport/models/`
- Generates `manifest.json` with SHA-256 checksums of every artifact
- **Idempotent:** if all artifacts are present and all checksums match, exits
  immediately without re-downloading or re-generating anything
- **No network access at runtime** — this script is the only place network I/O
  is permitted; once it completes the system is fully offline

Command and outputs:
```
python scripts/SemanticSupport/download_models.py

Outputs:
  multilingual/SemanticSupport/models/paraphrase-multilingual-MiniLM-L12-v2.onnx
  multilingual/SemanticSupport/models/paraphrase-multilingual-MiniLM-L12-v2.int8.onnx
  multilingual/SemanticSupport/models/tokenizer.json
  multilingual/SemanticSupport/models/tokenizer_config.json
  multilingual/SemanticSupport/models/special_tokens_map.json
  multilingual/SemanticSupport/models/vocab.txt          (if applicable)
  multilingual/SemanticSupport/models/manifest.json
```

**`train_multilingual_semantic_head.py`**
- Requires: artifacts from `download_models.py` must already exist
- Input: multilingual ONNX encoder + `multilingual/data/*.csv` training files
- Generates embeddings for all utterances across all languages using the ONNX encoder
- Trains a sklearn LogisticRegression head (mirroring `train_semantic_head.py` pattern)
- Preserves `Default Fallback Intent` as an explicit out-of-scope class
- Gates on minimum accuracy before writing output — exits non-zero if gate fails
- Output: `multilingual/SemanticSupport/models/semantic_head_multilingual.npz`

Command and outputs:
```
python scripts/SemanticSupport/train_multilingual_semantic_head.py

Outputs:
  multilingual/SemanticSupport/models/semantic_head_multilingual.npz
  (updates manifest.json with new checksum)
```

**`test_multilingual_semantic.py`**
- Loads `MultilingualSemanticFallback` from `multilingual/SemanticSupport/semantic.py`
- Reports per-language macro F1 on a held-out split
- Runs the four end-to-end smoke tests listed in section 8
- Exits non-zero if any language falls below minimum F1 threshold

Command and outputs:
```
python scripts/SemanticSupport/test_multilingual_semantic.py

Outputs: per-language F1 report, pass/fail for each smoke test, exit code
```

**`debug_multilingual_semantic.py`**
- Mirrors `scripts/debug_semantic_scores.py`
- Accepts `--text` and `--language` flags
- Prints: tokenisation result, UNK rate, embedding norm, top-5 intent scores with
  probabilities, and whether rescue would fire at current threshold

Command:
```
python scripts/SemanticSupport/debug_multilingual_semantic.py --text "c'est trop fort" --language fr
```

### 8. Benchmarks

The implementation plan must specify how each of the following will be measured and
what the acceptable upper bound is. Benchmarks must be run on CPU (no GPU assumed
for server deployment).

| Component | How to measure | Target upper bound |
|-----------|---------------|-------------------|
| Model load time | Time from `ort.InferenceSession(path)` to warm-up complete | < 3 s |
| Tokenisation time | Time to tokenise a 10-word utterance (100 warm runs, median) | < 0.5 ms |
| Embedding generation | Time from tokenised input to L2-normalised 384-dim vector (100 warm runs, median) | < 15 ms |
| Logistic head inference | Time from embedding vector to (intent, confidence) tuple | < 1 ms |
| End-to-end Stage 3 latency | Time from raw text string to rescue result, including all steps | < 20 ms |
| Peak memory during inference | `tracemalloc` or `psutil.Process().memory_info().rss` delta during embed | < 300 MB |

The plan must include the measurement script (inline in `test_multilingual_semantic.py`
or a dedicated `--benchmark` flag) and the actual measured values for the chosen
encoder. If measurement is not possible during planning, provide estimates flagged as
such with the methodology (e.g., "estimated from parameter count × dimension ratio
relative to English MiniLM measured at 8ms").

### 9. Testing gates

The plan must specify the tests that constitute a complete, shippable
implementation:

**Accuracy gate (per language):**
- Minimum macro F1 on held-out split: fr ≥ 0.80, de ≥ 0.78, da ≥ 0.75
- Rationale: slightly below TF-IDF holdout F1 is acceptable because the semantic
  head only fires on low-confidence TF-IDF cases

**Regression gate (English path unchanged):**
- Run the existing `scripts/test_semantic.py` before and after — output must be
  byte-identical
- Confirms that `_load_semantic` change does not affect the English code path

**Smoke tests (end-to-end, `language=<lang>`):**

| Utterance | Language | Expected rescue intent |
|-----------|----------|----------------------|
| "je n'entends pas bien dans un endroit bruyant" | fr | Cmd.MemoryChange |
| "c'est trop fort" | fr | Cmd.VolumeDecrease |
| "ich höre schlecht in lauten Umgebungen" | de | Cmd.MemoryChange |
| "det er for højt" | da | Cmd.VolumeDecrease |

These must pass via `NLUEngine(language=<lang>)` end-to-end, not just via direct
`MultilingualSemanticFallback.classify()` calls.

### 10. Migration and graceful degradation

- What happens when multilingual artifacts are absent (fresh checkout, partial
  install)? The plan must describe the fallback: log a warning at startup and set
  `self.semantic = None` for non-English engines, identical to how the English
  path degrades today.
- Is there a config flag to enable/disable multilingual semantic rescue per
  language without deleting artifacts? Propose one if needed.
- How does the plan handle the case where only some language artifacts are present
  (e.g., fr + de built but da not yet)?

### 11. Open questions

List every question the implementation agent must resolve before starting:

```
Q1. [topic] — <what needs to be decided and what information is needed>
```

---

## DELIVERABLE

Commit `docs/MULTILINGUAL_SEMANTICS_IMPLEMENTATION.md` to branch
`claude/multilingual-nlu-status-check-s7ggcw/MulitlingualSemanticSupport`
with commit message:

```
docs: multilingual semantic rescue implementation plan
```

The document must be self-contained — an engineer who has never seen this codebase
should be able to pick it up and implement without asking clarifying questions
(other than the open questions you explicitly flag).

The implementation plan must include a **"Setup and execution"** section that shows
the exact commands to run in order, the output artifacts each command produces, and
what a successful run looks like:

```
# Step 1 — download and export model artifacts (one-time, requires network)
python scripts/SemanticSupport/download_models.py
# Produces: multilingual/SemanticSupport/models/{encoder.onnx, encoder.int8.onnx,
#            tokenizer.json, tokenizer_config.json, special_tokens_map.json,
#            vocab.txt, manifest.json}

# Step 2 — train the logistic rescue head on multilingual embeddings
python scripts/SemanticSupport/train_multilingual_semantic_head.py
# Produces: multilingual/SemanticSupport/models/semantic_head_multilingual.npz
#           (manifest.json updated with new checksum)

# Step 3 — validate accuracy and run smoke tests
python scripts/SemanticSupport/test_multilingual_semantic.py
# Prints: per-language macro F1, benchmark timings, smoke test pass/fail
# Exits 0 on success, non-zero if any gate fails
```

After step 1, the repository must be fully usable offline.

---

## EXPLORATION CHECKLIST

Complete every item by reading the actual files before writing the plan:

```
[ ] Read scripts/nlu/semantic.py in full
    — understand SemanticFallback class, ONNX path, _tokenise/_wordpiece, .npz head format
[ ] Read scripts/nlu/engine.py lines 220–245 (_load_semantic)
    — note exactly how it instantiates SemanticFallback and falls back on failure
[ ] Read scripts/nlu/engine.py lines 510–540
    — note AGREEMENT_THRESHOLD, semantic_threshold, and rescue acceptance logic
[ ] Read scripts/train_semantic_head.py in full
    — understand training data format, head output schema, Default Fallback handling
[ ] Check if scripts/download_minilm.py exists; read it if so
    — for the download/export pattern to mirror
[ ] Read multilingual/train_multilingual.py lines 100–120
    — note LANGUAGES dict entries and data file paths
[ ] Read config/calibration.json
    — note per-language conf_threshold values
[ ] Run: ls -lh multilingual/data/
    — confirm which language CSVs exist and their sizes
[ ] Run: ls -lh models/
    — note every existing semantic artifact and its size
[ ] Run: ls -lh multilingual/models/
    — check if any per-language semantic artifacts already exist
[ ] Read docs/adding-a-new-language.md §Deliberately deferred capabilities
```

Only after completing this checklist should you begin writing the plan.
