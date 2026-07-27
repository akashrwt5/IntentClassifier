import sys
from pathlib import Path

path = Path("packages/runtime/nlu_engine/engine.py")
content = path.read_text(encoding="utf-8")
content = content.replace(
    'entities_path = LOC_DIR / f"nlu_entities.{language}.json"',
    'entities_path = BASE_DIR / "language_packs" / language / "nlu_entities.json"'
)
path.write_text(content, encoding="utf-8")
print("Patched _load_entities.")
