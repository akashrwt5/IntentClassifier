import sys
from pathlib import Path
sys.path.insert(0, str(Path("packages/runtime").resolve()))
from nlu_engine import NLUEngine
from nlu_langpack import load_pack

pack = load_pack("dist/bundle-en")
engine = NLUEngine(pack=pack)
intent, conf = engine.classifier.classify("in 7 min")
print(f"Predicted intent: {intent}, confidence: {conf}")
