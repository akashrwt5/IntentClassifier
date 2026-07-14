# `.claude/` — Claude workspace configuration

This directory keeps repository knowledge **modular** so Claude loads only what a
task needs, minimizing token usage.

## Layout

- `memory/` — the living knowledge base. One responsibility per file
  (architecture, datasets, training, inference, mobile, roadmap, decisions,
  known-issues). Agents and commands load only the files they need. Keep these
  in sync with the code (use `/sync-memory`).
- `agents/` — single-responsibility subagents. Dormant config; **zero tokens
  until invoked**, run on demand in an isolated context window, each loads only
  its relevant memory. See the table in `../CLAUDE.md`.
- `commands/` — slash commands: `/sync-memory`, `/train-eval`, `/export-mobile`,
  `/add-language`, `/review-change`.

## MCP servers (see `../.mcp.json`)

- **Code Graph Memory** — *primary source of code understanding*. Use it for
  dependency analysis, call graphs, symbol lookup, architecture navigation,
  impact analysis, and tracing execution flow **instead of scanning the repo**.
- **Context7** — up-to-date external library docs (scikit-learn, ONNX Runtime,
  coremltools, TensorFlow Lite, NumPy, skl2onnx, scipy). Prefer it over recalling
  possibly-stale APIs.

> The launch commands in `.mcp.json` are best-effort. Adjust `command`/`args`
> to match your actually-installed servers (e.g. a local Code Graph Memory
> binary or a different package name). A server that fails to start is ignored;
> the others still load.

## Principle

`CLAUDE.md` stays small (rules, standards, workflow, links). Project knowledge
lives here in `memory/`, not in `CLAUDE.md`.
