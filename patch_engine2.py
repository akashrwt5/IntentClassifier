import sys
from pathlib import Path

path = Path("packages/runtime/nlu_engine/engine.py")
content = path.read_text(encoding="utf-8")
content = content.replace(
    'return bool(language) and (LOC_DIR / f"nlu_lexicon.{language}.json").exists()',
    'return bool(language) and (BASE_DIR / "language_packs" / language / "nlu_lexicon.json").exists()'
)
path.write_text(content, encoding="utf-8")
print("Patched has_lexicon.")
