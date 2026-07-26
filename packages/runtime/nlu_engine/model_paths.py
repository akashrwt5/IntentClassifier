"""
Where the engine finds a language's model artifacts.

ONE resolver, so the engine, the evaluator and the calibrator cannot drift apart
on this again — they previously held three separate hardcoded opinions
(`engine.py`, `nlu_training/evaluate.py`, `nlu_training/calibrate_languages.py`),
all pointing at the legacy `multilingual/models/<lang>/` tree that only the
retired combined-multilingual trainer ever wrote.

TWO SOURCES, IN PRECEDENCE ORDER
--------------------------------
1. **A Language Pack.** If the caller supplies one, its manifest names the
   artifacts outright (`models.intent.<lang>.artifact`) and this module just
   reports them. That is the target state: the pack is authoritative and nothing
   is inferred.

2. **The local build tree**, `models/intent/<lang>/`. What
   `nlu_training.train --lang <lang>` writes, laid out to mirror the in-bundle
   path exactly so a pack build is a copy rather than a rename.

There is deliberately no third option and no language literal: a language is
present because its directory is, never because the engine was taught its name.

LEGACY FALLBACK — AND WHY IT IS NARROW
--------------------------------------
The flat `models/*.onnx` tree predates the per-language layout. It is accepted
for one transition, but ONLY for the language it was actually built for, read
from `models/manifest.json`. It carries no language field today, so in practice
it never fires.

That narrowness was learned the hard way: an unrestricted fallback served the
ENGLISH model for French and German requests. Every fr/de turn collapsed to
GENAI/FALLBACK and nothing reported that the wrong model had been used — the
conformance fixtures caught it only because they pin decisions, not scores. A
missing model must fail loudly; a silently substituted one is far worse.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parents[3]
MODELS_DIR = BASE_DIR / "models"

__all__ = ["ModelSet", "resolve_model_set", "available_languages"]


@dataclass(frozen=True)
class ModelSet:
    """Resolved artifact paths for one language."""

    language: str
    model: Path
    labels: Path
    source: str                       # "pack" | "build" | "legacy"
    calibration: Optional[Path] = None
    weights: Optional[Path] = None

    def exists(self) -> bool:
        return self.model.exists() and self.labels.exists()


def _from_build_tree(language: str) -> Optional[ModelSet]:
    d = MODELS_DIR / "intent" / language
    model, labels = d / "model.onnx", d / "labels.pkl"
    if not (model.exists() and labels.exists()):
        return None
    cal = d / "calibration.json"
    weights = d / "weights.json"
    return ModelSet(language, model, labels, "build",
                    calibration=cal if cal.exists() else None,
                    weights=weights if weights.exists() else None)


def _from_legacy_flat(language: str) -> Optional[ModelSet]:
    """The pre-per-language artifacts at models/*.onnx.

    CRITICALLY, this tree is NOT keyed by language — it is a single model with
    no record of what it was trained on. Serving it for an arbitrary language
    means classifying (say) French with an English model: the text scores
    garbage, every turn collapses to FALLBACK, and nothing announces that the
    wrong model was used. That silent-substitution failure is worse than a
    missing model, so the fallback is offered ONLY when the caller is asking
    for the same language the legacy artifact was built for.

    That language is read from the artifact's own manifest rather than assumed,
    so there is no language literal here.
    """
    model, labels = MODELS_DIR / "intent_model.onnx", MODELS_DIR / "intent_labels.pkl"
    if not (model.exists() and labels.exists()):
        return None
    if _legacy_language() != language:
        return None
    weights = MODELS_DIR / "intent_classifier_weights.json"
    return ModelSet(language, model, labels, "legacy",
                    weights=weights if weights.exists() else None)


def _legacy_language() -> Optional[str]:
    """Which language the flat legacy artifacts belong to, per models/manifest.json."""
    import json
    mf = MODELS_DIR / "manifest.json"
    if not mf.exists():
        return None
    try:
        return json.loads(mf.read_text(encoding="utf-8")).get("language")
    except Exception:
        return None


def _from_pack(pack, language: str) -> Optional[ModelSet]:
    """Artifacts named by a LanguagePack's manifest.

    `pack.model_paths` is keyed `<kind>.<lang>` plus `.<sibling>` entries — see
    nlu_langpack.loader. Nothing is inferred here; if the pack does not declare
    an intent model, this returns None and the caller falls through.
    """
    paths = getattr(pack, "model_paths", None) or {}
    model = paths.get(f"intent.{language}")
    if model is None:
        return None
    labels = paths.get(f"intent.{language}.labels")
    if labels is None:
        candidate = Path(model).parent / "labels.pkl"
        labels = candidate if candidate.exists() else None
    if labels is None:
        return None
    return ModelSet(language, Path(model), Path(labels), "pack",
                    calibration=paths.get(f"intent.{language}.calibration"),
                    weights=paths.get(f"intent.{language}.weights"))


def resolve_model_set(language: str, *, pack=None,
                      allow_legacy: bool = True) -> ModelSet:
    """Resolve one language's artifacts: pack, then build tree, then legacy.

    Raises FileNotFoundError naming the languages that DO have models, so the
    failure tells you how to fix it rather than just that something is missing.
    """
    for candidate in (_from_pack(pack, language) if pack is not None else None,
                      _from_build_tree(language),
                      _from_legacy_flat(language) if allow_legacy else None):
        if candidate is not None and candidate.exists():
            return candidate

    have = available_languages()
    raise FileNotFoundError(
        f"No intent model for {language!r}. Looked in "
        f"{(MODELS_DIR / 'intent' / language).relative_to(BASE_DIR)}"
        + (" and the legacy flat models/ tree" if allow_legacy else "")
        + f". Languages with a built model: {have or '(none)'}. "
        f"Train one with: python -m nlu_training.train --lang {language}"
    )


def available_languages() -> list[str]:
    """Languages that currently have a built model."""
    root = MODELS_DIR / "intent"
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir()
                  if d.is_dir() and (d / "model.onnx").exists())
