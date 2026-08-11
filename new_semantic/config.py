"""
Shared config for new_semantic/. Single source of truth — scripts import from
here so train / eval / export can never drift apart.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- data
DATA = ROOT / "data"
TRAIN_CSV = DATA / "en" / "train.csv"
LABELS_JSON = DATA / "en" / "labels.json"

LOCKED_TEST = DATA / "eval" / "locked_test_en.csv"
STRESS_TEST = DATA / "eval" / "stress_test_en.csv"
OOD_TEST = DATA / "eval" / "ood_test_en.csv"
# Authored diagnostic probe, NOT user data: every row contains a word absent
# from the vocabulary. Measures whether the model understands words or merely
# recognises them. Never train on it; never quote it as headline accuracy.
OOV_TEST = DATA / "eval" / "oov_test_en.csv"

MODELS = ROOT / "models" / "en"
REPORTS = ROOT / "reports"

FALLBACK_INTENT = "Default Fallback Intent"

# ---------------------------------------------------------------- teacher
# Runs OFFLINE at training time only. Never shipped, so its size is free.
TEACHER = "intfloat/e5-small-v2"

# ---------------------------------------------------------------- student
# Proven architecture: this exact config produced the 0.75 MB checkpoint.
# Do not change these casually — see docs/semantic-tiny-model-plan.md §5.
MAX_LEN = 24
EMBED_DIM = 64
NHEAD = 4
FF_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.10

PAD_ID = 0
UNK_ID = 1

# ---------------------------------------------------------------- training
SEED = 42
BATCH_SIZE = 128
EPOCHS = 40
PATIENCE = 7
LR = 2e-3
WEIGHT_DECAY = 1e-4
VAL_SIZE = 0.15

# distillation
TEMPERATURE = 2.0
CE_WEIGHT = 0.70
KD_WEIGHT = 0.30

# train.csv is deliberately uncapped and therefore imbalanced (55x).
# Per-class weights are how we compensate — NOT by deleting rows.
#   w[c] = n_total / (n_classes * n[c])
USE_CLASS_WEIGHTS = True

# ---------------------------------------------------------------- ship bar
# A candidate must clear ALL of these to replace the current model.
SHIP_BAR = {
    "locked_accuracy_min_delta": -0.01,  # vs previous best, absolute
    "ood_fallback_rate_min_delta": 0.0,  # must NOT get worse
    "stress_accuracy_min_delta": -0.01,
    "onnx_parity_max_mismatches": 0,
    "int8_parity_max_argmax_flips": 0,
}
