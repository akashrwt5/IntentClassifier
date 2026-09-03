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


# --------------------------------------------------------------------------- #
# No deployment endpoints inside signed artifacts
# --------------------------------------------------------------------------- #

def test_exported_weights_carry_no_endpoint():
    """A trained artifact must not carry a network address.

    `genai_base_url` shipped inside `intent_classifier_weights*.json` for as long
    as those files have existed, and its value was the placeholder
    `https://genai.yourcompany.com/chat?query=` — a host that does not exist,
    signed into every pack, delivered to every device, and read by nothing:
    not VoiceAIKit, not the reference engine, not a test.

    Two reasons this is a rule and not a tidy-up. An endpoint is DEPLOYMENT
    configuration, not a property of a model, so burying one in a signed artifact
    means it can only be changed by retraining and re-signing. And a pack is
    meant to be portable across deployments, which it stops being the moment it
    names one deployment's address.

    VIK-031 is what this class of mistake cost once already: unresolved turns
    returned a URL built from pack data, with the user's transcript in its query
    string.

    Asserted at the SOURCE rather than on a built payload, because the payload
    needs trained artifacts and this needs to fail in every environment.
    """
    exporters = sorted((REPO_ROOT / "packages" / "buildtime" / "nlu_export").glob("*.py"))
    assert exporters, "no exporters found — has the package moved?"

    offenders: list[str] = []
    for path in exporters:
        source = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if node.value.startswith(("http://", "https://")):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} {node.value}")

    assert not offenders, (
        "an exported artifact would carry a network address:\n  "
        + "\n  ".join(offenders)
        + "\nAn endpoint is deployment configuration; it does not belong in a "
          "signed model artifact."
    )

