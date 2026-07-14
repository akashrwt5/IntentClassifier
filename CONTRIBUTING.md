# Contributing

Thanks for working on the Intent Classifier. This guide keeps changes safe,
consistent, and easy to review.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
make install-dev      # runtime deps + ruff/black/mypy/pytest/pre-commit
pre-commit install    # (make install-dev already does this)
```

Python 3.10 is the target runtime.

## Development workflow

1. Branch from `feature/production-work` (the active integration branch).
2. Make **small, localized, incremental** changes. Do not rewrite working code
   without a measured reason and prior agreement.
3. Run the quality gates before committing:
   ```bash
   make format      # auto-fix with ruff + black
   make check       # lint + typecheck + test
   ```
4. Commit with a clear, conventional message (`feat:`, `fix:`, `docs:`,
   `refactor:`, `test:`, `chore:`). Pre-commit runs automatically.
5. Open a PR; the `reviewer` agent / a human reviews before merge.

## Code style

- Formatting: **Black** (line length 100), applied **format-on-touch** via
  **darker** — pre-commit and CI format/check only the lines you changed, so
  touching a legacy file never reformats the whole thing. `make format` does this
  locally; `make format-all` is the deliberate one-time full-repo pass.
  Linting/imports: **Ruff**. Type-checking: **MyPy** (lenient; tightening module-by-module).
- Add type hints to new and modified functions.
- Keep the inference hot path allocation-light; the model runs on-device.

## Testing

- Add tests under `tests/` (pytest). Prefer proper pytest tests over new
  loose `scripts/test_*.py` files.
- Markers: `slow`, `integration`, `coreml` (macOS/Core ML-only). Keep the
  Linux suite green.
- Do not weaken assertions to make a test pass. Cover edge cases and failure modes.
  ```bash
  make test          # or: make test-cov
  ```

## Models, data, and mobile export

- Trained artifacts (`*.onnx`, `*.pkl`, `*.mlpackage`) are **gitignored** and
  regenerated locally or in CI — do not commit them.
- The ONNX graph uses **static batch size 1**; embed one sentence at a time.
- When touching export/quantization, verify numeric parity and the confidence
  gate against the golden fixtures in `multilingual/test/` (Tier-A on Linux;
  Tier-B / ANE run on the Apple-Silicon macOS CI).
- Keep `requirements.txt` and the `[project].dependencies` in `pyproject.toml`
  in sync.

## Documentation

Update the relevant doc in the same change when behavior changes. Pipeline
reference: `docs/pipelines.md`. Repo guide: `CLAUDE.md`.

## Getting help

The `.claude/agents/` subagents (architect, ml-engineer, python-engineer,
mobile-ml-engineer, reviewer, tester, documentation) are scoped helpers Claude
can invoke on demand for their respective domains.
