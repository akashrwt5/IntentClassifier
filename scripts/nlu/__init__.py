"""DEPRECATED SHIM — the NLU engine moved to ``packages/runtime/nlu_engine``.

Kept for one release cycle so legacy ``import nlu`` / ``from nlu.entities
import …`` call sites keep working during the ND-2 restructure (move plan
M1). New code imports ``nlu_engine`` with ``packages/runtime`` on sys.path.

This shim aliases the real package in ``sys.modules``, so both names refer
to the SAME module objects — no double-loading, no state divergence.
"""

import sys
import warnings
from pathlib import Path

warnings.warn(
    "'nlu' (scripts/nlu) is deprecated; use 'nlu_engine' from packages/runtime "
    "(ND-2 restructure, docs/Review-F5/restructure-move-plan.md)",
    DeprecationWarning,
    stacklevel=2,
)

_runtime_dir = str(Path(__file__).resolve().parents[2] / "packages" / "runtime")
if _runtime_dir not in sys.path:
    sys.path.insert(0, _runtime_dir)

import importlib  # noqa: E402

import nlu_engine as _nlu_engine  # noqa: E402

# Alias the package AND its submodules so 'nlu.X' and 'nlu_engine.X' are the
# same module objects (no double-loading, no state divergence).
sys.modules[__name__] = _nlu_engine
for _sub in ("engine", "classifier", "entities", "context", "semantic", "manifest"):
    sys.modules[f"{__name__}.{_sub}"] = importlib.import_module(f"nlu_engine.{_sub}")
