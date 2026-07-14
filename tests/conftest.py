"""Shared pytest configuration.

Puts the ``scripts/`` directory on ``sys.path`` so the ``nlu`` package can be
imported as ``import nlu`` in tests that need the full engine. Individual tests
that only need a single leaf module (e.g. ``entities.py``) may still load it
directly to avoid pulling in heavier optional dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
