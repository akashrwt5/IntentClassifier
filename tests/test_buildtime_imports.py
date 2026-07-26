"""Every build-time entry point must import with only `packages/buildtime` on the path.

CI runs these as `PYTHONPATH=packages/buildtime python -m nlu_training.<mod>`.
They legitimately import from the RUNTIME package — the trainer and every
threshold fitter must featurise text with the exact function inference applies
(`nlu_engine.text_norm.normalize_text`), or the fitted values describe a
featurizer the shipped model does not use, which is blocker B8.

That cross-package import has to work without the caller adding
`packages/runtime`. It used to depend on each module inserting the path itself:
`train.py` did, the fitters did not, and a release run died with
`ModuleNotFoundError: No module named 'nlu_engine'` at the calibration step —
*after* training had already succeeded, so the failure appeared two steps away
from its cause. The bootstrap now lives once in `nlu_training/__init__.py`.

This test runs each import in a SUBPROCESS with a scrubbed environment, because
the pytest process already has both packages on `sys.path` and would pass
regardless — which is exactly why the local suite stayed green while CI failed.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]

# Modules CI invokes directly, plus the shared helpers they pull in.
BUILDTIME_MODULES = [
    "nlu_training.train",
    "nlu_training.evaluate",
    "nlu_training.fit_calibration",
    "nlu_training.fit_confirm_gate",
    "nlu_training.fit_slot_thresholds",
    "nlu_training.wrong_action_harness",
    "nlu_training.leakage",
    "nlu_compiler.content_source",
    "nlu_compiler.build",
]


def _import_in_isolation(module: str, pythonpath: str):
    env = dict(os.environ)
    env["PYTHONPATH"] = pythonpath
    env.pop("PYTHONHOME", None)
    return subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=_ROOT, env=env, capture_output=True, text=True, timeout=300)


@pytest.mark.parametrize("module", BUILDTIME_MODULES)
def test_imports_with_only_buildtime_on_the_path(module):
    r = _import_in_isolation(module, "packages/buildtime")
    assert r.returncode == 0, (
        f"`PYTHONPATH=packages/buildtime python -c 'import {module}'` failed — "
        f"this is exactly how CI invokes it:\n{r.stderr.strip()[-1500:]}")


def test_the_bootstrap_is_what_makes_it_work():
    """Mutation check: without the bootstrap the runtime import must fail.

    If `nlu_engine` were importable for some unrelated reason (installed as a
    package, a stray .pth), the test above would pass vacuously and the guard
    would be worthless.
    """
    r = _import_in_isolation("nlu_engine.text_norm", "packages/buildtime")
    assert r.returncode != 0, (
        "nlu_engine imports with only packages/buildtime on the path even "
        "WITHOUT nlu_training's bootstrap, so these tests cannot detect a "
        "missing bootstrap. Check for an installed copy of the runtime package.")
    assert "nlu_engine" in r.stderr
