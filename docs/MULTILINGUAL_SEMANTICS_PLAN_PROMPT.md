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
   <1ms head. Any multilingual encoder must fit within ~15ms on a server CPU.
   Measure; don't guess.
2. **Artifact size budget:** The English MiniLM is ~23 MB (INT8 quantised). A
   multilingual encoder will be larger — the plan must state the size and justify
   why it is acceptable, or propose quantisation.
3. **Zero changes to the English path.** English users must continue hitting the
   existing `SemanticFallback` in `scripts/nlu/semantic.py` unchanged. The new
   multilingual module must be completely additive.
4. **iOS is out of scope for this plan.** This plan covers the Python server
   pipeline only.
5. **No dependency on sentence-transformers at runtime.** The production path must
   use an ONNX encoder to avoid a heavy dependency. `sentence-transformers` is
   acceptable for training/embedding-generation only.

---

## WHAT THE PLAN MUST COVER

Produce a plan document at `docs/MULTILINGUAL_SEMANTICS_IMPLEMENTATION.md` that
addresses each of the following sections. Sections may be in any order that makes
logical sense.

### 1. Encoder selection

Choose a multilingual sentence embedding model. Candidates to evaluate:

- `paraphrase-multilingual-MiniLM-L12-v2` (Sentence Transformers, ~50 MB INT8)
- `paraphrase-multilingual-mpnet-base-v2` (Sentence Transformers, ~420 MB FP32)
- `multilingual-e5-small` (Microsoft, ~100 MB FP32)
- `LaBSE` (Google, ~360 MB FP32)

For each candidate, the plan must state:
- Whether it has an available ONNX export or can be exported with `optimum-cli`
- INT8 quantisation viability and expected size after quantisation
- Embedding dimension and whether it differs from MiniLM-L6-v2's 384 dimensions
- Estimated CPU inference latency (measure or estimate from parameter count — flag as estimate)
- Whether the vocabulary natively covers French, German, and Danish
- A final recommendation with justification

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
  tokeniser.py         — Multilingual tokeniser for the chosen encoder's vocab format
  models/
    <encoder>.onnx     — multilingual ONNX encoder (downloaded/exported by setup script)
    <encoder>-vocab.*  — vocab/tokenizer file for the encoder
    semantic_head_multilingual.npz  — logistic head trained on multilingual embeddings
    manifest.json      — SHA-256 checksums of all artifacts
scripts/SemanticSupport/
  download_multilingual_encoder.py  — downloads and ONNX-exports the encoder
  train_multilingual_semantic_head.py  — trains the head on multilingual embeddings
  test_multilingual_semantic.py     — validates head accuracy per language
  debug_multilingual_semantic.py    — diagnostic tool (mirrors debug_semantic_scores.py)
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

### 6. Artifact layout

State the complete on-disk layout of all new artifacts after the plan is
implemented. English artifacts are shown for reference — they must remain
unchanged:

```
models/                                      ← English artifacts — DO NOT TOUCH
  semantic_head.npz
  minilm-l6-v2.onnx
  minilm-vocab.txt

multilingual/SemanticSupport/models/         ← All new artifacts go here
  <encoder-name>.onnx                        ← multilingual encoder
  <encoder-name>-vocab.<ext>                 ← vocab/tokenizer file
  semantic_head_multilingual.npz             ← new logistic head
  manifest.json                              ← SHA-256 checksums
```

State:
- Exact file names (based on the chosen encoder)
- Estimated sizes (FP32 and INT8 if quantised)
- What `manifest.json` contains and how it is generated

### 7. Scripts: download, train, test

Describe each new script under `scripts/SemanticSupport/`:

**`download_multilingual_encoder.py`**
- What it downloads and from where
- How it exports to ONNX (via `optimum-cli` or `torch.onnx.export`)
- Whether it applies INT8 quantisation and how
- Output file names and expected sizes
- Idempotent: skip download if artifact already present and manifest checksum matches

**`train_multilingual_semantic_head.py`**
- Input: the multilingual encoder ONNX + `multilingual/data/*.csv` training files
- How it generates embeddings for all utterances across all languages
- How it trains the logistic head (same sklearn LogisticRegression pattern as the
  existing `train_semantic_head.py`)
- How it handles `Default Fallback Intent` as an explicit out-of-scope class
- Output: `multilingual/SemanticSupport/models/semantic_head_multilingual.npz`
- Minimum accuracy gate before writing the artifact

**`test_multilingual_semantic.py`**
- Loads `MultilingualSemanticFallback`
- Runs the four smoke-test utterances listed in section 8
- Reports per-language accuracy on a held-out split
- Exits non-zero if any language falls below the minimum F1 threshold

**`debug_multilingual_semantic.py`**
- Mirrors `scripts/debug_semantic_scores.py`
- Accepts `--text` and `--language` flags
- Prints embedding norm, top-5 intent scores, and whether rescue would fire

### 8. Testing gates

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

### 9. Migration and graceful degradation

- What happens when multilingual artifacts are absent (fresh checkout, partial
  install)? The plan must describe the fallback: log a warning at startup and set
  `self.semantic = None` for non-English engines, identical to how the English
  path degrades today.
- Is there a config flag to enable/disable multilingual semantic rescue per
  language without deleting artifacts? Propose one if needed.
- How does the plan handle the case where only some language artifacts are present
  (e.g., fr + de built but da not yet)?

### 10. Open questions

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
