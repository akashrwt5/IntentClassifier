# CLAUDE.md

Operating guide for Claude in this repo. Intentionally small — **project
knowledge lives in `.claude/memory/`, not here.** Read the memory file relevant
to your task instead of scanning the repo.

## What this is (one line)

On-device, multilingual (en/fr/de/da) intent classifier for a hearing-aid app —
a lightweight offline replacement for Dialogflow. Details: `memory/architecture.md`.

**This branch's defining work:** the **Language Pack** architecture (Review-F5) —
a language-neutral engine driven entirely by swappable data packs.
See `memory/langpack.md` before touching `scripts/nlu/` or `packs/`.

## Repository rules

- **Never rewrite working code unnecessarily.** Prefer small, localized,
  incremental changes. Explain reasoning before any behavior change; get approval
  before significant architectural changes.
- **The engine stays language-neutral.** No `if language ==` / `!=` branches and
  no English words in regex literals inside `scripts/nlu/`. CI enforces this
  (`scripts/ci/check_language_neutral.py`). Language-specific data belongs in a
  pack under `packs/<lang>/`.
- **On-device constraints are first-class.** The CoreML/ANE model needs a fixed
  `(1, V)` shape and the MiniLM embedder runs one sentence at a time (the
  classifier ONNX itself is dynamic-batch — see the table in
  `memory/architecture.md`). Preserve numeric parity + calibration when touching
  export/quantization; verify against `multilingual/test/` fixtures.
- **This is a medical-context app — the raw utterance never leaks.** It is never
  embedded in an `NLUResult`; raw-utterance logging is opt-in and off by default
  (`NLU_LOG_UTTERANCES`). Widening what leaves the device is an architectural
  decision, not a quick edit. Detail: `memory/inference.md`.
- **Keep docs in sync with code.** When behavior changes, update the relevant
  `memory/` file (and any affected `docs/`) in the same change.
- **Do not commit generated artifacts.** `*.onnx`, `*.pkl`, `*.mlpackage` under
  `models/` and `dist/` are gitignored and regenerated. (The artifacts committed
  inside `packs/en/` are the exception — they are the reference pack's payload.)

## Coding standards

- Python 3.10. **No linter/formatter/type-checker is configured on this branch**
  (no `pyproject.toml`, no `.pre-commit-config.yaml`) — match the style of the
  surrounding file rather than reformatting it. Do not bulk-reformat.
- Add type hints to new/modified code. Keep the inference hot path
  allocation-light. Do not introduce torch/transformers on the inference path.

## Workflow

```bash
pip install -r requirements.txt

python scripts/train.py                  # train English core -> models/
python scripts/predict.py                # interactive classifier CLI
python scripts/nlu_cli.py                # full NLU engine CLI (pack-driven)
python multilingual/train_multilingual.py
python scripts/calibrate_languages.py    # -> config/calibration.json
python scripts/test_holdout.py --strict
pytest                                   # see note below
```

`pytest` has **8 tracked pre-existing failures**. CI runs the suite with those
deselected — the authoritative list lives in `.github/workflows/pr.yml`, and the
rationale in `docs/Review-F5/TO-REMOVE.md`. Any *new* failure is a real
regression. Do not weaken an assertion to make one pass.

1. Branch from `claude/claude-setup-architecture-ebqobs` (active branch).
2. Make the change + update the matching `memory/` file.
3. Run the affected tests before committing.
4. Conventional commits (`feat:`/`fix:`/`docs:`/`refactor:`/`test:`/`chore:`).

## Tooling / MCP policy (read before scanning files)

- **Code Graph Memory MCP is the primary source of code understanding.** Use it
  for dependency analysis, call graphs, symbol lookup, architecture navigation,
  impact analysis, and tracing execution flow. **Do not scan the whole repo**
  when the Code Graph can answer the question.
- **Context7 MCP for external library docs** (scikit-learn, ONNX Runtime,
  coremltools, TensorFlow Lite, NumPy, skl2onnx, scipy). Prefer it over recalling
  possibly-outdated API details.
- **`.claude/memory/` for project knowledge.** Load only the file(s) your task
  needs. Fall back to reading source only when the Code Graph is insufficient.

## Agents (`.claude/agents/`)

Dormant, single-responsibility subagents. Each costs **zero tokens until
invoked**, runs **on demand in its own context window** (not in the background,
not all at once), loads only its relevant memory, and returns a summary.

| Agent | Responsibility | Loads memory | Access |
|---|---|---|---|
| `architect` | System design, trade-offs, ADRs | architecture, langpack, decisions, roadmap | read-only |
| `ml-engineer` | Data, training, eval, calibration | datasets, training | read/write + shell |
| `python-engineer` | Refactor, typing, performance | architecture, inference, langpack | read/write + shell |
| `mobile-ml-engineer` | ONNX/CoreML/TFLite, quantization | mobile, decisions | read/write + shell |
| `reviewer` | Bugs, security, perf, tech debt | known-issues, architecture, langpack | read-only |
| `tester` | Unit/integration/regression tests | inference, known-issues | read/write + shell |
| `documentation` | README, memory, ADRs, guides | (the file being edited) | read/write |

## Memory index (`.claude/memory/`)

- `architecture.md` — components, turn flow, constraints.
- `langpack.md` — the Language Pack contract, `packs/` layout, neutrality rules.
- `datasets.md` — data lineage, holdouts, add-a-phrase workflow.
- `training.md` — training + calibration pipelines.
- `inference.md` — classifier + NLU engine runtime path.
- `mobile.md` — ONNX/CoreML/TFLite export, quantization, parity.
- `roadmap.md` — done / in-progress / planned.
- `decisions.md` — ADR log (the durable "why").
- `known-issues.md` — active bugs, gaps, gotchas.

## Slash commands (`.claude/commands/`)

`sync-memory`, `train-eval`, `export-mobile`, `add-language`, `review-change`,
`execute-plan`.

## Git

Active branch: `claude/claude-setup-architecture-ebqobs`. Keep changes here.
Do not push to another branch without explicit permission.
