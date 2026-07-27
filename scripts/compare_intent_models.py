import csv
import json
import numpy as np
from pathlib import Path
import onnxruntime as ort
import coremltools as ct
import re

BASE_DIR = Path(__file__).resolve().parents[1]
LANG = "en"
MODELS_DIR = BASE_DIR / "models" / "intent" / LANG
DATA_PATH = BASE_DIR / "language_packs" / LANG / "extras" / "semantic_holdout_2.csv"

def _swift_tokenize(text: str):
    words = [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w]
    tokens = list(words)
    for i in range(len(words) - 1):
        tokens.append(words[i] + " " + words[i + 1])
    return tokens

def _vectorize(text, vocab, idf):
    n_feat = len(idf)
    counts = {}
    for tok in _swift_tokenize(text):
        j = vocab.get(tok)
        if j is not None:
            counts[j] = counts.get(j, 0) + 1
    vec = np.zeros(n_feat, dtype=np.float32)
    for j, c in counts.items():
        vec[j] = (1.0 + np.log(c)) * idf[j]
    norm = np.sqrt((vec * vec).sum())
    if norm > 0:
        vec /= norm
    return vec

def main():
    print("Loading models...")
    onnx_sess = ort.InferenceSession(str(MODELS_DIR / "model.onnx"))
    
    with open(MODELS_DIR / "intent_classifier_weights.json", "r") as f:
        weights = json.load(f)
    vocab = weights["vocab"]
    idf = weights["idf"]
    labels = weights["labels"]
    T = weights.get("temperature", 1.0)
    
    cml_model = ct.models.MLModel(str(MODELS_DIR / "IntentClassifier.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)
    
    print(f"Loaded ONNX & CoreML (vocab {len(vocab)}, T={T})")
    
    flips = []
    total = 0
    onnx_correct = 0
    cml_correct = 0
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("utterance", "").strip()
            expected = row.get("expected_intent", "").strip()
            if not text:
                continue
                
            total += 1
            
            # ONNX Predict
            onnx_out = onnx_sess.run(None, {"input": np.array([[text]])})
            # ONNX outputs [label_array, probabilities_list_of_dicts]
            onnx_top = onnx_out[0][0]
            
            # CoreML Predict
            vec = _vectorize(text, vocab, idf)
            cml_out = cml_model.predict({"tfidf_vector": vec})
            
            # Apply Temperature Scaling to logits exactly like iOS
            logits = cml_out["logits"]
            z = logits - logits.max()
            e = np.exp(z / T)
            probs = e / e.sum()
            cml_top_idx = int(np.argmax(probs))
            cml_top = labels[cml_top_idx]
            
            if onnx_top == expected: onnx_correct += 1
            if cml_top == expected: cml_correct += 1
            
            if onnx_top != cml_top:
                flips.append({
                    "text": text,
                    "expected": expected,
                    "onnx": onnx_top,
                    "coreml": cml_top
                })
    
    print("\n" + "="*50)
    print("  REPORT")
    print("="*50)
    print(f"Total phrases  : {total}")
    print(f"ONNX Accuracy  : {onnx_correct/total*100:.2f}%")
    print(f"CoreML Accuracy: {cml_correct/total*100:.2f}%")
    print(f"Total Flips    : {len(flips)}")
    print("\n--- Flipped Phrases ---")
    for f in flips:
        print(f"Text    : '{f['text']}'")
        print(f"Expected: {f['expected']}")
        print(f"ONNX    : {f['onnx']}")
        print(f"CoreML  : {f['coreml']}")
        print("-" * 30)

if __name__ == "__main__":
    main()
