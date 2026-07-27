import sys
from pathlib import Path
path = Path("apps/cli/nlu_cli.py")
content = path.read_text(encoding="utf-8")
content = content.replace(
    'except Exception as e:\n        print(f"Failed to load language pack from dist/bundle-en: {e}")\n        print("Run \'./build_language.sh en\' first.")',
    'except Exception as e:\n        import traceback\n        traceback.print_exc()\n        print(f"Failed to load language pack from dist/bundle-en: {e}")\n        print("Run \'./build_language.sh en\' first.")'
)
path.write_text(content, encoding="utf-8")
print("Patched.")
