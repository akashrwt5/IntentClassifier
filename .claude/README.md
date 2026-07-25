# `.claude/` — Claude workspace configuration

This directory keeps repository knowledge **modular** so Claude loads only what a
task needs, minimizing token usage.

## Layout

- `memory/` — the living knowledge base. One responsibility per file
  (architecture, langpack, datasets, training, inference, mobile, roadmap,
  decisions, known-issues). Agents and commands load only the files they need.
  Keep these in sync with the code (use `/sync-memory`).
- `agents/` — single-responsibility subagents. Dormant config; **zero tokens
  until invoked**, run on demand in an isolated context window, each loads only
  its relevant memory. See the table in `../CLAUDE.md`.
- `commands/` — slash commands: `/sync-memory`, `/train-eval`, `/export-mobile`,
  `/add-language`, `/review-change`, `/execute-plan`.

## Principle

`CLAUDE.md` stays small (rules, standards, workflow, links). Project knowledge
lives here in `memory/`, not in `CLAUDE.md`. The point is that a task about
CoreML export reads `mobile.md` and nothing else — it never pays for the
training or dataset knowledge it doesn't need.

## Branch-accuracy rule

This setup is written for **this** branch. Its architecture is the Language Pack
line (`packs/`, `packages/nlu_langpack/`, a language-neutral `scripts/nlu/`).

`feature/production-work` is a **different architecture** — 57-intent
`domain.object.action` label space, `packages/runtime/nlu_engine/`, `apps/`,
`datasets/`, `content/`, `spec/`, a Makefile and `pyproject.toml`. This branch is
2 ahead / 57 behind it. **Never copy a fact, path, or command from there into
this memory.** Verify against the working tree instead.

## MCP servers (see `../.mcp.json`)

- **Code Graph Memory** — *primary source of code understanding*. Use it for
  dependency analysis, call graphs, symbol lookup, architecture navigation,
  impact analysis, and tracing execution flow **instead of scanning the repo**.
- **Context7** — up-to-date external library docs (scikit-learn, ONNX Runtime,
  coremltools, NumPy, skl2onnx, scipy). Prefer it over recalling possibly-stale
  APIs.

> `.mcp.json` uses **CodeGraphContext** (`cgc mcp start`) for the code graph and
> **Context7** (`npx @upstash/context7-mcp`). Install first: `pip install
> codegraphcontext` then `cgc mcp setup`. These are **local stdio** servers —
> they work with clients that can spawn a local process (Claude Code CLI /
> Desktop); a browser-only web session cannot. A server that fails to start is
> ignored; the others still load. `.cgcignore` keeps binaries and caches out of
> the index.

> **Web sessions:** add Context7 as a remote connector in claude.ai
> (Settings → Connectors → Add custom connector → `https://mcp.context7.com/mcp`).
> The code graph stays **desktop-only** by policy — proprietary code is not
> exposed over the internet (see `memory/decisions.md` ADR-013).
