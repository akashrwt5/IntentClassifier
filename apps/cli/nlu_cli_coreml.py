#!/usr/bin/env python3
"""
Interactive multi-turn NLU demo — the on-device Dialogflow replacement.

Usage:
    python apps/cli/nlu_cli.py
"""

import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages" / "runtime"))
from nlu_engine import NLUEngine  # noqa: E402
from nlu_langpack import load_pack  # noqa: E402


import json
import numpy as np
import coremltools as ct


import re

class CoreMLIntentBackend:
    def __init__(self, cml_path: Path, weights_path: Path, n_labels: int):
        self.model = ct.models.MLModel(str(cml_path), compute_units=ct.ComputeUnit.CPU_ONLY)
        weights_data = json.loads(weights_path.read_text())
        self.vocab = weights_data["vocab"]
        self.idf = np.array(weights_data["idf"])
        self.n_labels = n_labels

    def _swift_tokenize(self, text: str):
        words = [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w]
        tokens = list(words)
        for i in range(len(words) - 1):
            tokens.append(words[i] + " " + words[i + 1])
        return tokens

    def tfidf_logits(self, text: str) -> np.ndarray:
        tf = np.zeros(len(self.vocab), dtype=np.float32)
        counts = {}
        for tok in self._swift_tokenize(text):
            if tok in self.vocab:
                counts[tok] = counts.get(tok, 0) + 1
        for w, c in counts.items():
            tf[self.vocab[w]] = 1.0 + np.log(c)
            
        tfidf = tf * self.idf
        norm = np.linalg.norm(tfidf)
        if norm > 0:
            tfidf /= norm
        vec = tfidf
        
        output = self.model.predict({"tfidf_vector": vec})
        logits = output["logits"]
        if hasattr(logits, "shape") and len(logits.shape) == 2:
            logits = logits[0]
        return np.asarray(logits, dtype=float)

SESSION = "cli-user"


def render(r, engine, text):
    if r.interrupted_intent:
        print(f"  ⚠️  Interrupted: {r.interrupted_intent} flow cancelled")
    if r.semantic_rescue:
        via = f"🧠 semantic {r.confidence:.2f}  |  tf-idf said: {r.tfidf_intent} ({r.tfidf_confidence:.2f})"
    else:
        via = f"⚡ tf-idf {r.confidence:.2f}"
    if r.type == "FULFILL":
        params = f"  {r.parameters}" if r.parameters else ""
        print(f"  ✅ {r.intent}  →  action={r.action}{params}")
        print(f"     [{via}]")
        if r.message:
            print(f"  💬 {r.message}")
    elif r.type == "PROMPT":
        print(f"  ❓ {r.message}")
        print(f"     [{via}]")
        if r.parameters:
            print(f"     (collected so far: {r.parameters})")
    elif r.type == "CONFIRM":
        print(f"  ❓ {r.message}  [yes/no]")
    elif r.type == "FALLBACK":
        print(f"  🤖 GenAI fallback  (confidence {r.confidence:.2f})")
        if r.message:
            print(f"  💬 {r.message}")
        # The app layer (here, the CLI) builds the GenAI URL from the text it
        # already holds — the raw utterance is never returned in the result.
        if engine.genai_url:
            url = engine.genai_url + urllib.parse.quote(text)
            print(f"  🔗 {url}")
        else:
            print("  🔗 (no GenAI endpoint configured — set NLU_GENAI_URL)")


def main():
    print("=== On-device NLU (CoreML backend) ===")
    print("    Type 'exit' to quit, 'reset' to clear the conversation.\n")
    try:
        pack = load_pack("dist/bundle-en")

        cml_path = Path("dist/bundle-en/models/intent/en/IntentClassifier.mlpackage")
        weights_path = Path("dist/bundle-en/models/intent/en/intent_classifier_weights.json")
        labels = json.loads(Path("dist/bundle-en/models/intent/en/labels.json").read_text())
        print("Loading CoreML model... (this may take a few seconds)")
        backend = CoreMLIntentBackend(cml_path, weights_path, n_labels=len(labels))
        engine = NLUEngine(pack=pack, backend=backend)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Failed to load language pack from dist/bundle-en: {e}")
        print("Run './scripts/build_local_release.sh' first.")
        return
        
    while True:
        try:
            text = input("you ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.lower() == "exit":
            break
        if text.lower() == "reset":
            engine.reset(SESSION)
            print("  (conversation reset)\n")
            continue
        render(engine.handle(SESSION, text), engine, text)
        print()


if __name__ == "__main__":
    main()
