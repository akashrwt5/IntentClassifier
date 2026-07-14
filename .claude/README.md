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

> `.mcp.json` uses **CodeGraphContext** (`cgc mcp start`) for the code graph and
> **Context7** (`npx @upstash/context7-mcp`). Install first: `pip install
> codegraphcontext` then `cgc mcp setup` (see docs). These are **local stdio**
> servers — they work with desktop clients (Claude Code / Claude Desktop /
> Cowork desktop) that can spawn a local process; a browser-only web session
> cannot. A server that fails to start is ignored; the others still load.

## Principle

`CLAUDE.md` stays small (rules, standards, workflow, links). Project knowledge
lives here in `memory/`, not in `CLAUDE.md`.

> **Web sessions:** add Context7 as a remote connector in claude.ai
> (Settings -> Connectors -> Add custom connector -> `https://mcp.context7.com/mcp`).
> The code graph stays **desktop-only** by policy — proprietary code is not
> exposed over the internet (see `memory/decisions.md` ADR-009).
