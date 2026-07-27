"""Package-boundary rules, CI-enforced (ADR-003 AI#7, ADR-005 Part 13).

Dependency-free AST scan instead of import-linter — same guarantee, no new
tooling. The load-bearing separations:

- the RUNTIME engine never imports buildtime packages (a device library
  must not depend on training/compiler code);
- the COMPILER never imports the engine (independent release trains — the
  compiler validates content against spec/, not against engine internals);
- content/ and spec/ contain no executable Python at all (facts, not code —
  the ADR-001.1 fact/behavior split, structurally enforced).
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN = {
    "packages/runtime/nlu_engine": ("nlu_compiler", "nlu_training", "nlu_export"),
    "packages/buildtime/nlu_compiler": ("nlu_engine",),
    # nlu_langpack is the CONTRACT both sides depend on, so it must depend on
    # neither. If it imported the engine, the engine could not import it back
    # without a cycle, and the boundary would stop being a boundary.
    "packages/runtime/nlu_langpack": (
        "nlu_engine", "nlu_compiler", "nlu_training", "nlu_export"),
}


def _imports(pyfile: Path) -> set[str]:
    tree = ast.parse(pyfile.read_text(encoding="utf-8"))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    return mods


def test_forbidden_cross_package_imports():
    violations = []
    for pkg, banned in FORBIDDEN.items():
        for pyfile in (REPO_ROOT / pkg).rglob("*.py"):
            if "__pycache__" in pyfile.parts:
                continue
            hit = _imports(pyfile) & set(banned)
            if hit:
                violations.append(f"{pyfile.relative_to(REPO_ROOT)} imports {sorted(hit)}")
    assert not violations, violations


def test_content_and_spec_hold_no_python():
    stray = [p.relative_to(REPO_ROOT).as_posix()
             for d in ("content", "spec", "language_packs")
             for p in (REPO_ROOT / d).rglob("*.py")]
    assert not stray, f"facts, not code — no .py belongs here: {stray}"


def test_runtime_engine_is_dependency_lean():
    """The engine may use only the lean runtime deps (no torch/transformers/
    sklearn on the inference path — architecture rule)."""
    heavy = {"torch", "transformers", "sklearn", "tensorflow", "coremltools", "skl2onnx"}
    violations = []
    for pyfile in (REPO_ROOT / "packages/runtime/nlu_engine").glob("*.py"):
        hit = _imports(pyfile) & heavy
        if hit:
            violations.append(f"{pyfile.name} imports {sorted(hit)}")
    assert not violations, violations
