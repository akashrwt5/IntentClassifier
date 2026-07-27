import sys
import csv
import json
import time
from pathlib import Path
import numpy as np
import coremltools as ct

# Ensure packages/runtime is in the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "runtime"))

from nlu_engine.engine import NLUEngine
from nlu_langpack import load_pack


import re

class CoreMLIntentBackend:
    def __init__(self, cml_path: Path, weights_path: Path, n_labels: int):
        self.model = ct.models.MLModel(str(cml_path), compute_units=ct.ComputeUnit.CPU_ONLY)
        weights_data = json.loads(weights_path.read_text())
        self.vocab = weights_data["vocab"]
        self.idf = np.array(weights_data["idf"])
        self.n_labels = n_labels

    def _swift_tokenize(self, text: str):
        pattern = re.compile(r"(?u)\b\w\w+\b")
        words = pattern.findall(text.lower())
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

def main():
    csv_path = Path("language_packs/en/extras/semantic_holdout_2.csv")
    if not csv_path.exists():
        print(f"Error: Could not find {csv_path}")
        return

    print("Loading language pack...")
    pack = load_pack("dist/bundle-en")

    print("Initializing ONNX Engine...")
    engine_onnx = NLUEngine(pack=pack)

    print("Initializing CoreML Engine... (this might take a few seconds)")
    cml_path = Path("dist/bundle-en/models/intent/en/IntentClassifier.mlpackage")
    weights_path = Path("dist/bundle-en/models/intent/en/intent_classifier_weights.json")
    labels = json.loads(Path("dist/bundle-en/models/intent/en/labels.json").read_text())
    backend = CoreMLIntentBackend(cml_path, weights_path, n_labels=len(labels))
    engine_coreml = NLUEngine(pack=pack, backend=backend)

    mismatches = []
    onnx_correct = 0
    coreml_correct = 0
    total = 0

    print(f"\nEvaluating {csv_path.name}...\n")
    print("-" * 100)
    print(f"{'UTTERANCE':<45} | {'ONNX (Conf)':<20} | {'CoreML (Conf)':<20}")
    print("-" * 100)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["utterance"]
            expected = row.get("expected_intent", "")
            
            # Predict using ONNX
            onnx_intent, onnx_conf = engine_onnx.classifier.classify(text)
            
            # Predict using CoreML
            cml_intent, cml_conf = engine_coreml.classifier.classify(text)
            
            total += 1
            if onnx_intent == expected:
                onnx_correct += 1
            if cml_intent == expected:
                coreml_correct += 1
                
            is_mismatch = (onnx_intent != cml_intent) or (abs(onnx_conf - cml_conf) > 0.01)
            
            if is_mismatch:
                mismatches.append(row)
                print(f"{text[:43]:<45} | {onnx_intent} ({onnx_conf:.2f}) | {cml_intent} ({cml_conf:.2f})")

    print("-" * 100)
    print("\n=== SUMMARY ===")
    print(f"Total Utterances: {total}")
    print(f"ONNX Accuracy:   {onnx_correct / total * 100:.1f}% ({onnx_correct}/{total})")
    print(f"CoreML Accuracy: {coreml_correct / total * 100:.1f}% ({coreml_correct}/{total})")
    
    if len(mismatches) == 0:
        print("\n✅ PERFECT PARITY! Both models produced identical intent predictions and confidence scores.")
    else:
        print(f"\n⚠️ FOUND {len(mismatches)} DISCREPANCIES (either intent mismatch or confidence > 1% diff).")

if __name__ == "__main__":
    main()
