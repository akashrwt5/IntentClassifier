# CLAUDE.md

Operating guide for Claude in this repo. Intentionally small — **project
knowledge lives in `.claude/memory/`, not here.** Read the memory file relevant
to your task instead of scanning the repo.

## What this is (one line)

On-device, multilingual (en/fr/de/da) intent classifier for a hearing-aid app —
a lightweight offline replacement for Dialogflow. Details: `memory/architecture.md`.

## Repository rules

- **Never rewrite working code unnecessarily.** Prefer small, localized,
  incremental changes. Explain reasoning before any behavior change; get approval
  before significant architectural changes.
- **On-device constraints are first-class.** ONNX graph is static batch size 1
  (embed one sentence at a time). Preserve numeric parity + calibration when
  touching export/quantization; verify against `multilingual/test/` fixtures.
- **Keep docs in sync with code.** When behavior changes, update the relevant
  `memory/` file (and any affected `docs/`) in the same change.
- **Do not commit generated artifacts.** `*.onnx`, `*.pkl`, `*.mlpackage` are
  gitignored and regenerated (`make train`, `make export-coreml`).
- Optimize for mobile ML deployment (ONNX / CoreML / TFLite).

## Coding standards

- Python 3.10, line length 100, Black-formatted, Ruff-linted, MyPy-checked.
  Config in `pyproject.toml`; hooks in `.pre-commit-config.yaml`.
- Tooling is **lenient by design** so never-linted code stays green; tighten
  rule families / strictness module-by-module (see `memory/decisions.md` ADR-008).
- Add type hints to new/modified code. Keep the inference hot path allocation-light.

## Workflow

```bash
make install-dev     # deps + ruff/black/darker/mypy/pytest/pre-commit
make format          # ruff --fix + darker (formats CHANGED lines only)
make check           # lint + typecheck + test
make train | predict | nlu | calibrate | export-coreml
```

1. Branch from `feature/production-work` (active branch — keep changes here).
2. Make the change + update the matching `memory/` file.
3. `make format && make check` before committing.
4. Conventional commits (`feat:`/`fix:`/`docs:`/`refactor:`/`test:`/`chore:`).

Full pipeline reference: `docs/pipelines.md`.

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
| `architect` | System design, trade-offs, ADRs | architecture, decisions, roadmap | read-only |
| `ml-engineer` | Data, training, eval, calibration | datasets, training | read/write + shell |
| `python-engineer` | Refactor, typing, performance | architecture, inference | read/write + shell |
| `mobile-ml-engineer` | ONNX/CoreML/TFLite, quantization | mobile, decisions | read/write + shell |
| `reviewer` | Bugs, security, perf, tech debt | known-issues, architecture | read-only |
| `tester` | Unit/integration/regression tests | inference, known-issues | read/write + shell |
| `documentation` | README, memory, ADRs, guides | (the file being edited) | read/write |

## Memory index (`.claude/memory/`)

- `architecture.md` — components, turn flow, constraints.
- `datasets.md` — data lineage, holdouts, add-a-phrase workflow.
- `training.md` — training + calibration pipelines.
- `inference.md` — classifier + NLU engine runtime path.
- `mobile.md` — ONNX/CoreML/TFLite export, quantization, parity.
- `roadmap.md` — done / in-progress / planned.
- `decisions.md` — ADR log (the durable "why").
- `known-issues.md` — active bugs, gaps, gotchas.

## Slash commands (`.claude/commands/`)

`sync-memory`, `train-eval`, `export-mobile`, `add-language`, `review-change`.

## Git

Active branch: `feature/production-work`. Keep changes on this branch.
