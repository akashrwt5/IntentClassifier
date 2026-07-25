#!/bin/bash
# SessionStart hook — prepare a fresh cloud container so tests/linters just run.
#
# Only for Claude Code on the web: a fresh container has no Python deps and, more
# subtly, no en_US.UTF-8 locale. Local machines and GitHub CI already have both
# (pr.yml pip-installs, and ubuntu-latest ships the locale), so this exits early
# everywhere else.
#
# Idempotent and non-interactive: every step is guarded, so a warm container
# re-runs it in well under a second.
set -uo pipefail

[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0

cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}" || exit 0

# 1. Locale. onnxruntime's StringNormalizer op (used by the TF-IDF intent model)
#    asks the OS for en_US.UTF-8 and HARD-FAILS if it is absent. The error reads
#    like a corrupt model — "Exception during initialization: string_normalizer"
#    — so without this a session can burn real time re-exporting a fine model.
if ! locale -a 2>/dev/null | grep -qiE '^en_US\.utf-?8$'; then
    echo "[session-start] generating en_US.UTF-8 locale"
    if command -v locale-gen >/dev/null 2>&1; then
        locale-gen en_US.UTF-8 >/dev/null 2>&1
    elif command -v localedef >/dev/null 2>&1; then
        localedef -i en_US -f UTF-8 en_US.UTF-8 >/dev/null 2>&1
    fi
fi

# 2. Dependencies. Runtime deps from requirements.txt; pytest + ruff to match
#    what .github/workflows/pr.yml runs, so local results mirror CI.
if ! python -c "import numpy, sklearn, onnxruntime, pandas, pytest, ruff" >/dev/null 2>&1; then
    echo "[session-start] installing Python dependencies (~1 min on a cold container)"
    python -m pip install -q --upgrade pip >/dev/null 2>&1
    python -m pip install -q -r requirements.txt >/dev/null 2>&1
    python -m pip install -q pytest ruff >/dev/null 2>&1
fi

# 3. `packages/` holds the nlu_langpack contract and is not installed as a
#    package, so imports need it on the path (tests insert it themselves).
#    Absolute, because the path must survive a change of working directory —
#    and appended only once, since SessionStart also fires on resume/clear/
#    compact and a repeated append would build up "packages:packages:packages".
if [ -n "${CLAUDE_ENV_FILE:-}" ] && ! grep -q 'PYTHONPATH.*/packages' "$CLAUDE_ENV_FILE" 2>/dev/null; then
    echo "export PYTHONPATH=\"\${PYTHONPATH:+\$PYTHONPATH:}$PWD/packages\"" >> "$CLAUDE_ENV_FILE"
fi

# 4. Report readiness. A hook that quietly half-works is worse than one that
#    says what is missing, so surface the real state rather than exiting 0 blind.
python - <<'PY'
import sys
sys.path[:0] = ["scripts", "packages"]
ok = True
try:
    from nlu import NLUEngine  # noqa: F401
    print("[session-start] engine imports OK")
except Exception as e:
    ok = False
    print(f"[session-start] WARNING engine import failed: {type(e).__name__}: {e}")
try:
    from nlu_langpack import load_pack
    p = load_pack("packs/en")
    print(f"[session-start] pack OK: {p.language} {p.stages} semantic={p.semantic_available}")
except Exception as e:
    ok = False
    print(f"[session-start] WARNING pack load failed: {type(e).__name__}: {e}")
print("[session-start] ready" if ok else "[session-start] ready with warnings (see above)")
PY

# Never block the session: a degraded environment is reported, not fatal.
exit 0
