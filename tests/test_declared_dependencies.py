"""Every third-party module the build imports must be in requirements.lock.

CI installs `requirements.lock` and nothing else (plus a short ad-hoc line for
the bundle-spec extras). A package that is imported but not locked therefore
works on every developer machine — where it arrived as some other tool's
transitive dependency — and fails only on a clean runner.

That has now happened three times in this pipeline, each time deep in a run:

  * pyyaml       the compiler reads content/*.yaml; it was declared only in the
                 dev/test extras, so the release job died with
                 ModuleNotFoundError AFTER training had already succeeded.
  * jsonschema / referencing / cryptography
                 undeclared, so bundle tests ERRORED rather than failed.
  * onnxruntime  locked to a version with no CPython 3.10 wheel, so the install
                 step itself could not complete.

An import-time smoke test cannot catch this: pytest runs in an environment that
already has everything. So this walks the AST of the build packages, resolves
each top-level import to "stdlib / first-party / third-party", and asserts the
third-party ones are locked.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_LOCK = _ROOT / "requirements.lock"

# Packages CI installs explicitly alongside the lock (see release-pack.yml /
# ci.yml). They are spec/test tooling, deliberately not device dependencies.
CI_EXTRAS = {"jsonschema", "referencing", "cryptography", "pytest"}

# import name -> distribution name, where they differ.
IMPORT_TO_DIST = {
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "dateutil": "python-dateutil",
}

# Installed ON DEMAND, deliberately not in the base lock: the CoreML export job
# pip-installs coremltools+torch itself, the TFLite export job pip-installs
# tensorflow, and the semantic/ML tooling is only run by hand (semantic rescue is
# disabled). Locking them would put ~2 GB of wheels into every job that only needs
# to train a TF-IDF model.
OPTIONAL_ON_DEMAND = {
    "coremltools", "torch", "transformers", "sentence_transformers",
    "optimum", "onnx2torch", "mlflow", "tensorflow",
}

SCANNED = ("packages/buildtime", "packages/runtime")

# Subtrees not on any CI path: the multilingual semantic trainers are run by hand
# and pull the heavy ML stack.
SKIP_PARTS = ("semantic_multilingual",)


def _repo_module_names() -> set[str]:
    """Every module name importable from inside the repo.

    Sibling imports (`import text_norm`, `from scripts import ...`) resolve via
    sys.path at runtime and are NOT third-party, but an AST scan cannot tell the
    difference without this.
    """
    roots = [_ROOT, _ROOT / "scripts"]
    for pkg in SCANNED:
        pkg_root = _ROOT / pkg
        if pkg_root.is_dir():
            roots.append(pkg_root)
            roots += [d for d in pkg_root.iterdir() if d.is_dir()]

    names: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for entry in root.iterdir():
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                names.add(entry.name)
            elif entry.suffix == ".py":
                names.add(entry.stem)
    return names


def _locked() -> set[str]:
    names = set()
    for line in _LOCK.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z0-9._-]+)", line)
        if m:
            names.add(m.group(1).lower().replace("_", "-"))
    return names


def _top_level_imports(pyfile: Path) -> set[str]:
    try:
        tree = ast.parse(pyfile.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    return mods


def test_build_and_runtime_imports_are_locked():
    stdlib = set(sys.stdlib_module_names)
    locked = _locked()
    missing: dict[str, list[str]] = {}

    first_party = _repo_module_names()
    for pkg in SCANNED:
        for pyfile in (_ROOT / pkg).rglob("*.py"):
            if "__pycache__" in pyfile.parts:
                continue
            if any(part in SKIP_PARTS for part in pyfile.parts):
                continue
            for mod in _top_level_imports(pyfile):
                if (mod in stdlib or mod in first_party or mod in CI_EXTRAS
                        or mod in OPTIONAL_ON_DEMAND):
                    continue
                dist = IMPORT_TO_DIST.get(mod, mod).lower().replace("_", "-")
                if dist not in locked:
                    missing.setdefault(dist, []).append(
                        str(pyfile.relative_to(_ROOT)))

    assert not missing, (
        "third-party imports missing from requirements.lock — these work locally "
        "and fail on a clean CI runner:\n" +
        "\n".join(f"  {d}: {sorted(set(f))[:3]}" for d, f in sorted(missing.items())))


def test_pyyaml_specifically_is_locked_not_just_a_dev_extra():
    """The exact regression: the compiler needs yaml at BUILD time.

    Declaring it under [project.optional-dependencies].dev is not enough — CI
    installs the lock, and the lock is generated from requirements.txt.
    """
    assert "pyyaml" in _locked(), (
        "pyyaml is not in requirements.lock; nlu_compiler.content_bundle reads "
        "content/*.yaml and the release job will fail after training succeeds")
    assert "pyyaml" in (_ROOT / "requirements.txt").read_text(encoding="utf-8"), (
        "pyyaml must be in requirements.txt — `make lock` regenerates the lock "
        "from it, so a lock-only edit is reverted by the next regeneration")


def test_the_guard_would_notice_an_unlocked_import(tmp_path):
    """Mutation check: the scan must actually resolve names, not pass vacuously."""
    stdlib = set(sys.stdlib_module_names)
    locked = _locked()
    fake = tmp_path / "mod.py"
    fake.write_text("import some_unlocked_package\n", encoding="utf-8")
    mods = _top_level_imports(fake)
    assert "some_unlocked_package" in mods
    assert "some_unlocked_package" not in stdlib
    assert "some-unlocked-package" not in locked
