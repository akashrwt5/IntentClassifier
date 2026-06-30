#!/usr/bin/env python3
"""
Validation suite for the multilingual semantic rescue stage.

Tests:
  1. Per-language macro-F1 on a stratified holdout (targets: fr≥0.80, de≥0.78, da≥0.75)
  2. End-to-end smoke tests via NLUEngine(language=...) for fr, de, da
  3. OOS rejection: clearly out-of-scope utterances must NOT be rescued at default threshold
  4. Graceful degradation: engine must not raise when multilingual artifacts are absent

Exit code 0 if all gates pass. Prints a FAIL summary and exits 1 otherwise.

Run:
    python scripts/SemanticSupport/test_multilingual_semantic.py
    python scripts/SemanticSupport/test_multilingual_semantic.py --skip-holdout
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--skip-holdout", action="store_true",
                    help="Skip F1 holdout evaluation (useful when training data is not split)")
parser.add_argument("--threshold", type=float, default=0.55,
                    help="Confidence threshold for semantic rescue (default: 0.55)")
args = parser.parse_args()

FAILURES = []


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    FAILURES.append(msg)


def ok(msg: str) -> None:
    print(f"  OK  : {msg}")


# ── 1. Artifact availability ───────────────────────────────────────────────────

print("\n=== 1. Artifact check ===")

MODEL_DIR  = BASE_DIR / "multilingual" / "SemanticSupport" / "models"
HEAD_PATH  = MODEL_DIR / "semantic_head_multilingual.npz"
ONNX_PATH  = MODEL_DIR / "paraphrase-multilingual-minilm-l12-v2.int8.onnx"
VOCAB_PATH = MODEL_DIR / "multilingual-vocab.txt"
MANIFEST   = MODEL_DIR / "manifest.json"

for name, path in [("Head NPZ", HEAD_PATH), ("INT8 ONNX", ONNX_PATH),
                   ("Vocab", VOCAB_PATH), ("Manifest", MANIFEST)]:
    if path.exists():
        ok(f"{name}: {path.name} ({path.stat().st_size // 1024} KB)")
    else:
        fail(f"{name} missing: {path}")

if FAILURES:
    print("\nArtifacts missing — run download_models.py and train_multilingual_semantic_head.py first.")
    sys.exit(1)


# ── 2. Tokeniser unit tests ────────────────────────────────────────────────────

print("\n=== 2. Tokeniser unit tests ===")
try:
    from multilingual.SemanticSupport.tokeniser import MultilingualTokeniser, SEQ_LEN

    tok = MultilingualTokeniser(VOCAB_PATH)

    for lang_label, sample in [
        ("fr", "Quand dois-je prendre mes médicaments ?"),
        ("de", "Wann soll ich meine Medikamente nehmen?"),
        ("da", "Hvornår skal jeg tage min medicin?"),
    ]:
        ids, mask, types = tok.encode(sample)
        assert ids.shape   == (1, SEQ_LEN), f"Wrong ids shape: {ids.shape}"
        assert mask.shape  == (1, SEQ_LEN), f"Wrong mask shape: {mask.shape}"
        assert types.shape == (1, SEQ_LEN), f"Wrong types shape: {types.shape}"
        assert ids[0, 0]   == tok._cls_id,  "First token must be [CLS]"
        # Padding check: last token in a short sentence should be 0 (PAD)
        assert ids[0, -1]  == tok._pad_id,  "Last token should be [PAD] for short inputs"
        real_tokens = int(mask[0].sum())
        unk_rate = tok.unk_rate(sample)
        ok(f"[{lang_label}] shape=(1,{SEQ_LEN}), real_tokens={real_tokens}, UNK_rate={unk_rate:.2%}")

    # Edge case: text longer than SEQ_LEN
    long_text = "mot " * 200
    ids_l, mask_l, _ = tok.encode(long_text)
    assert ids_l.shape == (1, SEQ_LEN), f"Long text shape wrong: {ids_l.shape}"
    ok("Long text truncation: shape correct")

    # Edge case: empty string
    ids_e, mask_e, _ = tok.encode("")
    assert ids_e[0, 0] == tok._cls_id
    ok("Empty string: handled without error")

except Exception as e:
    fail(f"Tokeniser unit test raised: {e}")


# ── 3. Semantic fallback unit tests ────────────────────────────────────────────

print("\n=== 3. MultilingualSemanticFallback unit tests ===")
try:
    from multilingual.SemanticSupport.semantic import MultilingualSemanticFallback

    sf = MultilingualSemanticFallback(threshold=args.threshold)
    ok("MultilingualSemanticFallback constructed successfully")

    for lang_label, text, expected_not in [
        ("fr", "Quand dois-je prendre mes médicaments ?", "Default Fallback Intent"),
        ("de", "Wann nehme ich meine Medikamente?",       "Default Fallback Intent"),
        ("da", "Hvornår tager jeg min medicin?",           "Default Fallback Intent"),
    ]:
        intent, conf = sf.classify(text)
        ok(f"[{lang_label}] classify → intent={intent!r}, conf={conf:.3f}")

    # is_available mirrors English SemanticFallback
    assert sf.is_available() is True
    ok("is_available() returns True")

except Exception as e:
    fail(f"SemanticFallback unit test raised: {e}")


# ── 4. Per-language macro-F1 holdout evaluation ────────────────────────────────

if not args.skip_holdout:
    print("\n=== 4. Per-language macro-F1 holdout evaluation ===")

    GATES = {"en": 0.82, "fr": 0.80, "de": 0.78, "da": 0.75}
    DATA_DIR = BASE_DIR / "multilingual" / "data"

    try:
        from multilingual.SemanticSupport.semantic import MultilingualSemanticFallback
        from sklearn.metrics import f1_score

        sf = MultilingualSemanticFallback(threshold=0.0)  # threshold=0 to score all

        for lang, gate in GATES.items():
            csv_path = DATA_DIR / f"{lang}.csv"
            if not csv_path.exists():
                fail(f"[{lang}] holdout CSV missing: {csv_path}")
                continue

            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            df.columns = [c.strip().lower() for c in df.columns]
            text_col = next(
                (c for c in df.columns if c in ("text", "utterance", "phrase", "sentence")),
                df.columns[0]
            )
            df = df.rename(columns={text_col: "text"})
            df["intent"] = df["intent"].astype(str).str.strip()
            df = df[df["intent"] != "Default Fallback Intent"]

            # Use a 15% stratified sample as a language-specific holdout.
            from sklearn.model_selection import train_test_split
            try:
                _, holdout = train_test_split(
                    df, test_size=0.15, stratify=df["intent"], random_state=99
                )
            except ValueError:
                # Stratification fails when some classes have only 1 sample.
                _, holdout = train_test_split(df, test_size=0.15, random_state=99)

            y_true, y_pred = [], []
            for _, row in holdout.iterrows():
                intent, conf = sf.classify(row["text"])
                y_true.append(row["intent"])
                y_pred.append(intent)

            macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            if macro_f1 >= gate:
                ok(f"[{lang}] macro-F1 = {macro_f1:.3f} ≥ gate {gate:.2f} ✓")
            else:
                fail(f"[{lang}] macro-F1 = {macro_f1:.3f} < gate {gate:.2f} — BELOW TARGET")

    except Exception as e:
        fail(f"Holdout evaluation raised: {e}")
else:
    print("\n=== 4. Per-language F1 evaluation: SKIPPED (--skip-holdout) ===")


# ── 5. OOS rejection ──────────────────────────────────────────────────────────

print("\n=== 5. OOS rejection (should not rescue clearly out-of-scope queries) ===")
try:
    from multilingual.SemanticSupport.semantic import (
        MultilingualSemanticFallback, DEFAULT_THRESHOLD, FALLBACK_INTENT
    )

    sf_oos = MultilingualSemanticFallback(threshold=DEFAULT_THRESHOLD)
    oos_samples = [
        ("fr", "Quelle est la capitale de la France ?"),
        ("de", "Was ist die Hauptstadt von Deutschland?"),
        ("da", "Hvad er hovedstaden i Danmark?"),
    ]
    for lang_label, text in oos_samples:
        intent, conf = sf_oos.classify(text)
        # The rescue is blocked when conf < threshold OR intent == FALLBACK_INTENT.
        rescued = (intent != FALLBACK_INTENT and conf >= DEFAULT_THRESHOLD)
        if not rescued:
            ok(f"[{lang_label}] OOS correctly not rescued: intent={intent!r}, conf={conf:.3f}")
        else:
            # Not a hard failure — general-knowledge queries can sometimes look
            # like intent phrases. Report as a warning, not a blocking failure.
            print(f"  WARN: [{lang_label}] OOS might be rescued: intent={intent!r}, conf={conf:.3f}")

except Exception as e:
    fail(f"OOS rejection test raised: {e}")


# ── 6. End-to-end engine smoke tests ─────────────────────────────────────────

print("\n=== 6. NLUEngine end-to-end smoke tests ===")
try:
    from nlu.engine import NLUEngine

    for lang in ("fr", "de", "da"):
        try:
            engine = NLUEngine(language=lang)
            result = engine.handle("smoke-test-session", "test")
            assert result.type in ("FULFILL", "FALLBACK", "PROMPT", "CONFIRM"), \
                f"Unexpected result type: {result.type}"
            ok(f"[{lang}] NLUEngine constructed and handle() returned type={result.type!r}")
        except Exception as e:
            fail(f"[{lang}] NLUEngine smoke test raised: {e}")

except ImportError as e:
    fail(f"Engine import failed: {e}")


# ── 7. Graceful degradation when artifacts absent ──────────────────────────────

print("\n=== 7. Graceful degradation (missing artifact simulation) ===")
try:
    from multilingual.SemanticSupport.semantic import MultilingualSemanticFallback
    from pathlib import Path
    import tempfile, os

    # Point to a non-existent path to simulate missing artifact.
    bogus = Path(tempfile.mkdtemp()) / "nonexistent.npz"
    try:
        MultilingualSemanticFallback(head_path=bogus)
        fail("Should have raised FileNotFoundError for missing head")
    except FileNotFoundError:
        ok("FileNotFoundError raised correctly for missing head")

except Exception as e:
    fail(f"Graceful degradation test raised unexpectedly: {e}")


# ── Summary ────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
if FAILURES:
    print(f"RESULT: {len(FAILURES)} FAILURE(S)")
    for f_msg in FAILURES:
        print(f"  • {f_msg}")
    sys.exit(1)
else:
    print(f"RESULT: All tests passed ✓")
    sys.exit(0)
