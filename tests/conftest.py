"""Shared pytest configuration.

Puts the ``scripts/`` directory on ``sys.path`` so the legacy ``nlu`` shim and the ``nlu_engine`` package (packages/runtime — ND-2 M1) can be
imported in tests that need the full engine. Individual tests
that only need a single leaf module (e.g. ``entities.py``) may still load it
directly to avoid pulling in heavier optional dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
RUNTIME_DIR = REPO_ROOT / "packages" / "runtime"

for _d in (SCRIPTS_DIR, RUNTIME_DIR):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))
