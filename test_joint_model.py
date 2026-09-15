import re
import sys
from pathlib import Path
import joblib
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "runtime"))

from scripts.bio_tagger_prototype import robust_preprocess, extract_features, tokenize


class JointONNXModel:
    """
    Robust Combined NLU Model:
    1. Preprocessing Normalization (ASR repair, contractions, slang, fillers)
    2. `models/intent_model.onnx` for Intent Classification (ONNX Runtime)
    3. `models/bio_slot_tagger.pkl` for Token-level BIO Tagging & Slot Extraction
    """

    def __init__(self):
        # 1. Load ONNX Intent Model
        onnx_path = PROJECT_ROOT / "models" / "intent_model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
        self.onnx_session = ort.InferenceSession(str(onnx_path))

        # 2. Load BIO Slot Tagger
        bio_path = PROJECT_ROOT / "models" / "bio_slot_tagger.pkl"
        if not bio_path.exists():
            raise FileNotFoundError(f"BIO model not found at {bio_path}")
        data = joblib.load(bio_path)
        self.vectorizer = data["vectorizer"]
        self.classifier = data["classifier"]

    def predict(self, text: str) -> dict:
        # Step 0: Robust Preprocessing (ASR repair, contractions, fillers)
        clean_text = robust_preprocess(text)

        # Step 1: Predict Intent using ONNX
        raw_in = np.array([[clean_text]], dtype=object)
        outputs = self.onnx_session.run(["label", "probabilities"], {"input": raw_in})
        raw_label = outputs[0][0]
        probs = outputs[1][0]
        confidence = float(np.max(probs))

        # Map internal label name to standard intent
        intent = (
            "reminders.add"
            if raw_label in ("reminders.task.create", "reminders.add")
            else raw_label
        )

        # Step 2: Predict BIO Tags using Slot Model
        tokens = tokenize(clean_text)
        features = [extract_features(tokens, i) for i in range(len(tokens))]
        X_vec = self.vectorizer.transform(features)
        tags = self.classifier.predict(X_vec)

        # Step 3: Extract Title from B-TITLE and I-TITLE tags
        title_tokens = []
        is_capturing = False
        for token, tag in zip(tokens, tags):
            if tag == "B-TITLE":
                is_capturing = True
                title_tokens.append(token)
            elif tag == "I-TITLE" and is_capturing:
                title_tokens.append(token)
            elif tag != "I-TITLE":
                is_capturing = False

        title = " ".join(title_tokens) if title_tokens else None

        return {
            "text": text,
            "clean_text": clean_text,
            "intent": intent,
            "raw_onnx_label": raw_label,
            "confidence": confidence,
            "title": title,
            "tokens": tokens,
            "bio_tags": list(tags),
        }


def print_result(res: dict):
    print("\n" + "=" * 65)
    print(f"📥 Input Phrase      : \"{res['text']}\"")
    if res["clean_text"] != res["text"]:
        print(f"🧹 Normalized Form   : \"{res['clean_text']}\"")
    print("-" * 65)
    print(f"🎯 ONNX Intent       : {res['intent']}")
    print(f"🏷️  Extracted Title   : {res['title'] or '(None)'}")
    print(f"📈 ONNX Confidence   : {res['confidence']:.2f}")
    print("-" * 65)
    print("🔬 Token-by-Token BIO Breakdown:")
    for token, tag in zip(res["tokens"], res["bio_tags"]):
        tag_str = f"[{tag}]" if tag != "O" else "O"
        print(f"   {token:<15} -> {tag_str}")
    print("=" * 65)


def main():
    print("⏳ Loading Combined ONNX Intent + BIO Slot Model...")
    model = JointONNXModel()
    print("✅ Model loaded and ready in memory!\n")

    args = sys.argv[1:]
    if args:
        res = model.predict(" ".join(args))
        print_result(res)
        return

    print("=" * 65)
    print("🚀 COMBINED MODEL TESTER (Intent + BIO Tagging)")
    print("   Console me apna koi bhi phrase likhein aur Enter dabayein.")
    print("   Band karne ke liye 'exit' likhein.")
    print("=" * 65)

    turn = 1
    while True:
        try:
            phrase = input(f"\n👉 [{turn}] Enter phrase: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not phrase:
            continue
        if phrase.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        res = model.predict(phrase)
        print_result(res)
        turn += 1


if __name__ == "__main__":
    main()
