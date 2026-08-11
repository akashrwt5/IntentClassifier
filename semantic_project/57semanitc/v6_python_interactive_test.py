#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

ROOT = Path('/Users/shuklam/IntentClassifier/semantic_project/57semanitc')
ENCODER_PATH = ROOT / 'v6_production_frozen' / 'encoder'
CLASSIFIER_PATH = ROOT / 'v6_production_frozen' / 'intent_classifier.joblib'
LABEL_MAP_PATH = ROOT / 'v6_production_frozen' / 'label_map.json'

with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
    raw_map = json.load(f)
INDEX_TO_LABEL = {int(k): str(v) for k, v in raw_map.items()}

print('=' * 80)
print('V6 INTERACTIVE PYTHON INFERENCE TEST')
print('=' * 80)
print('\nLoading V6 E5 encoder...')
encoder = SentenceTransformer(str(ENCODER_PATH))
encoder.eval()
print('Loading V6 classifier...')
classifier = joblib.load(CLASSIFIER_PATH)
print('\nModel loaded successfully.')
print('Classes:', len(classifier.classes_))
print("\nType your sentence. Type 'exit' or 'quit' to stop.")
print('-' * 80)

while True:
    try:
        text = input('\nUser: ').strip()
    except (KeyboardInterrupt, EOFError):
        print('\n\nExiting...')
        break
    if not text:
        continue
    if text.lower() in {'exit', 'quit', 'q'}:
        print('\nExiting...')
        break

    embedding = encoder.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    probabilities = classifier.predict_proba(embedding)[0]
    raw_prediction = classifier.predict(embedding)[0]
    prediction = INDEX_TO_LABEL[int(raw_prediction)]
    confidence = float(probabilities[int(raw_prediction)])
    top_indices = np.argsort(probabilities)[::-1][:3]

    print('\n' + '-' * 80)
    print('Prediction :', prediction)
    print(f'Confidence : {confidence * 100:.2f}%')
    print('\nTop 3:')
    for rank, idx in enumerate(top_indices, start=1):
        label = INDEX_TO_LABEL[int(idx)]
        score = float(probabilities[idx])
        print(f'  {rank}. {label:<40} {score * 100:.2f}%')
    print('-' * 80)
