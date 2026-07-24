#!/usr/bin/env bash
#
# Project cleanup — removes build cruft, caches, and sandbox leftovers so the
# repo is clean. SAFE by default; locates the repo root itself.
#
# Usage:
#   bash scripts/cleanup.sh              # remove safe cruft (pycache, caches, dist, sandbox junk, .DS_Store)
#   bash scripts/cleanup.sh --dry-run    # show what WOULD be removed; delete nothing
#   bash scripts/cleanup.sh --scaffold   # also remove the empty packages/{runtime,buildtime}
#   bash scripts/cleanup.sh --repo-hygiene  # also remove Engage.zip + checkpoints/ (review Appendix A)
#   bash scripts/cleanup.sh --superseded # also remove Restructuring/ (old PoC) + multilingual_intent/ (empty)
#   bash scripts/cleanup.sh --all        # safe cruft + scaffold + repo-hygiene + superseded
#
# NOT removed by any flag: models/ (build needs it) and multilingual/ (618M fr/de/da
# work) — delete multilingual/ by hand only if you are dropping multilingual.
#
# Tracked files are removed with `git rm` (staged for commit); untracked ones with
# `rm`. Nothing here affects the running system — it is pure housekeeping. Review
# with `git status` afterwards and commit.
set -euo pipefail

DRY=0; SCAFFOLD=0; HYGIENE=0; SUPERSEDED=0
for a in "${@:-}"; do
  case "$a" in
    "") ;;
    --dry-run) DRY=1 ;;
    --scaffold) SCAFFOLD=1 ;;
    --repo-hygiene) HYGIENE=1 ;;
    --superseded) SUPERSEDED=1 ;;
    --all) SCAFFOLD=1; HYGIENE=1; SUPERSEDED=1 ;;
    -h|--help) sed -n '2,22p' "$0" | sed 's/^#\s\{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $a (try --help)"; exit 2 ;;
  esac
done

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || { cd "$(dirname "$0")/.." && pwd; })"
cd "$ROOT"
echo "repo: $ROOT"
[ "$DRY" = 1 ] && echo "(dry-run — nothing will be deleted)"

REMOVED=0
rm_path() {
  local p="$1"
  [ -e "$p" ] || return 0
  echo "  remove: $p"
  [ "$DRY" = 1 ] && return 0
  if git ls-files --error-unmatch "$p" >/dev/null 2>&1; then
    git rm -rf --quiet --ignore-unmatch "$p" >/dev/null 2>&1 || rm -rf "$p"
  else
    rm -rf "$p"
  fi
  REMOVED=$((REMOVED + 1))
}

echo "== safe cruft =="
rm_path "__sandbox_junk"
rm_path "dist"
rm_path ".pytest_cache"
rm_path ".ruff_cache"
for lock in .git/index.lock.stale .git/index.lock.bak .git/index.lock.bak2; do
  rm_path "$lock"
done
# __pycache__ dirs, *.pyc, and .DS_Store — repo-wide, but skip heavy/vendored
# trees (.git, virtualenvs, node_modules) for speed and safety.
PRUNE=( -path ./.git -o -path './.venv*' -o -path ./node_modules )
while IFS= read -r d; do rm_path "$d"; done < <(find . \( "${PRUNE[@]}" \) -prune -o -name __pycache__ -type d -print)
while IFS= read -r f; do rm_path "$f"; done < <(find . \( "${PRUNE[@]}" \) -prune -o -name '*.pyc' -type f -print)
while IFS= read -r f; do rm_path "$f"; done < <(find . \( "${PRUNE[@]}" \) -prune -o -name .DS_Store -type f -print)

if [ "$SCAFFOLD" = 1 ]; then
  echo "== empty pre-existing scaffold =="
  for d in packages/runtime packages/buildtime; do
    if [ -d "$d" ]; then
      # SAFETY: only remove if there is no real source under it (just __pycache__).
      if [ -z "$(find "$d" -type f ! -name '*.pyc' ! -name '.DS_Store' -print -quit)" ]; then
        rm_path "$d"
      else
        echo "  SKIP (contains source, not empty): $d"
      fi
    fi
  done
fi

if [ "$HYGIENE" = 1 ]; then
  echo "== repo hygiene (flagged in the production review, Appendix A) =="
  rm_path "Engage.zip"
  rm_path "checkpoints"
fi

if [ "$SUPERSEDED" = 1 ]; then
  echo "== superseded / dead folders (nothing imports these) =="
  rm_path "Restructuring"       # early proof-of-concept, replaced by packages/nlu_langpack + packs/ + scripts/nlu
  rm_path "multilingual_intent" # empty skeleton dirs
  # NOTE: models/ is KEPT (build pipeline needs it). multilingual/ is NOT removed
  # here (618M, holds the fr/de/da training + French data) — remove it yourself
  # only if you are not pursuing multilingual packs:  rm -rf multilingual/
fi

echo "done."
[ "$DRY" = 1 ] || echo "removed ${REMOVED} item(s). Review with 'git status' and commit."
