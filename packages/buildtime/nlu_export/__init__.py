"""nlu_export — model export / packaging scripts (moved from scripts/ in ND-2 M2)."""

# --- runtime-package bootstrap ------------------------------------------------
# Identical to `nlu_training/__init__.py`, and here for the same reason it is
# there: an exporter must featurise text with the exact function inference
# applies (`nlu_engine.text_norm.normalize_text`), or the fitted temperature and
# the shipped vocabulary describe two different featurizers — blocker B8.
#
# That bootstrap was added to `nlu_training` when CI (which sets only
# `PYTHONPATH=packages/buildtime`) died with `ModuleNotFoundError: No module
# named 'nlu_engine'` AFTER training had already succeeded. `nlu_export` was
# left out, so the same failure came back the moment an exporter took the same
# import: `export_tflite` inside a function, `export_ios_weights` at module
# level — one step in the Linux job, one in the macOS job, the same error.
#
# Fixing it in the two workflow steps alone would leave the THIRD caller to
# rediscover it. This is the same one-line-per-package cure, finally applied to
# both packages that need it.
#
# Deliberately no submodule import below: this package's modules pull in
# tensorflow / coremltools, and importing `nlu_export` must stay cheap.
import sys as _sys
from pathlib import Path as _Path

_RUNTIME = _Path(__file__).resolve().parents[2] / "runtime"
if _RUNTIME.is_dir() and str(_RUNTIME) not in _sys.path:
    _sys.path.insert(0, str(_RUNTIME))
