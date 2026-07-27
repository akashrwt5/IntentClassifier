import sys
from pathlib import Path

path = Path("packages/runtime/nlu_engine/engine.py")
content = path.read_text(encoding="utf-8")
content = content.replace(
    '        labels_path = LABELS_JSON_PATH\n        if not labels_path.exists():\n            return  # model not yet trained; skip during development\n        labels = set(json.loads(labels_path.read_text(encoding="utf-8")))',
    '        if not getattr(self, "labels", None):\n            return  # model not yet trained; skip during development\n        labels = set(self.labels)'
)
path.write_text(content, encoding="utf-8")
print("Patched _assert_label_schema_parity.")
