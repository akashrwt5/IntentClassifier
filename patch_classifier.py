import sys
from pathlib import Path

path = Path("packages/runtime/nlu_engine/classifier.py")
content = path.read_text(encoding="utf-8")
content = content.replace(
    'self.labels = joblib.load(str(labels_path))',
    'self.labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.suffix == ".json" else joblib.load(str(labels_path))'
)
# Make sure json is imported
if 'import json' not in content:
    content = 'import json\n' + content
path.write_text(content, encoding="utf-8")
print("Patched.")
