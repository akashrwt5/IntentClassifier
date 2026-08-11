#!/usr/bin/env python3
"""
V3 PRODUCTION AUDIT — PYTHON
============================

Read-only audit of the locked V3 ONNX model.

Checks:
  1. Model integrity / SHA256
  2. ONNX metadata and fixed shapes
  3. Vocabulary integrity
  4. Locked 595-row reproduction
  5. Fixed-shape repeated inference parity
  6. Inference latency
  7. Model file size
  8. Confidence statistics
  9. Critical command accuracy
 10. OOD rejection
 11. Runtime memory, when psutil is available

NO:
  - training
  - threshold fitting
  - model modification
  - ONNX export
  - quantization
  - use of 595 rows for training

Run:
  cd /Users/shuklam/IntentClassifier/semantic_project
  python3 v3_production_audit.py
"""

from pathlib import Path
import hashlib
import json
import re
import statistics
import time
import tracemalloc

import numpy as np
import pandas as pd
import onnxruntime as ort

from sklearn.metrics import accuracy_score, f1_score


ROOT = Path("/Users/shuklam/IntentClassifier/semantic_project")

V3_ONNX = ROOT / "tiny_semantic_student_v3_fp32" / "v3_semantic_student_fp32.onnx"
UNSEEN_CSV = ROOT / "unseen_semantic_stress_test.csv"
OOD_CSV = ROOT / "production_calibration_v2" / "production_ood_calibration.csv"

VOCAB_CANDIDATES = [
    ROOT / "tiny_semantic_student_v2_balanced" / "vocab.json",
    ROOT / "tiny_semantic_student_v3_balanced" / "vocab.json",
    ROOT / "tiny_semantic_student_v3_fp32" / "vocab.json",
]

OUTPUT_DIR = ROOT / "v3_production_audit"

MAX_LEN = 24
VOCAB_SIZE = 895
NUM_CLASSES = 11

LABELS = [
    "device.memory.change",
    "device.volume.decrease",
    "device.volume.increase",
    "device.volume.mute",
    "device.volume.unmute",
    "find.phone.locate",
    "help.reminder.show",
    "reminders.task.complete",
    "reminders.task.create",
    "streaming.session.start",
    "streaming.session.stop",
]

# Same canonical critical set used in the prior audit/benchmark.
CRITICAL_PHRASES = [
    "make it louder",
    "make it quieter",
    "mute it",
    "unmute it",
    "turn off",
    "turn the sound back on",
    "completely silent",
    "keep it on",
]


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def normalize_text(x):
    return re.sub(r"\s+", " ", str(x).strip())


def find_column(df, candidates):
    cols = {
        str(c).strip().lower(): c
        for c in df.columns
    }
    for name in candidates:
        if name.lower() in cols:
            return cols[name.lower()]
    return None


def load_vocab():
    exact = []

    for path in VOCAB_CANDIDATES:
        if not path.exists():
            continue

        obj = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(obj, dict) and "vocab" in obj:
            obj = obj["vocab"]

        if isinstance(obj, dict) and len(obj) == VOCAB_SIZE:
            exact.append((path, obj))

    if not exact:
        raise FileNotFoundError(
            f"No {VOCAB_SIZE}-token vocab.json found."
        )

    # Preserve the vocabulary previously resolved for V3.
    preferred = [
        x for x in exact
        if "tiny_semantic_student_v2_balanced" in str(x[0])
    ]

    return preferred[0] if preferred else exact[0]


def get_token_id(vocab, names, default):
    for name in names:
        if name in vocab:
            return int(vocab[name])
    return default


def tokenize(text, vocab):
    pad_id = get_token_id(vocab, ["<pad>", "[PAD]"], 0)
    unk_id = get_token_id(vocab, ["<unk>", "[UNK]"], 1)

    cls_id = None
    sep_id = None

    for name in ["<cls>", "[CLS]"]:
        if name in vocab:
            cls_id = int(vocab[name])
            break

    for name in ["<sep>", "[SEP]"]:
        if name in vocab:
            sep_id = int(vocab[name])
            break

    ids = []

    if cls_id is not None:
        ids.append(cls_id)

    for token in normalize_text(text).lower().split():
        ids.append(int(vocab.get(token, unk_id)))

    if sep_id is not None:
        ids.append(sep_id)

    ids = ids[:MAX_LEN]

    if len(ids) < MAX_LEN:
        ids.extend([pad_id] * (MAX_LEN - len(ids)))

    return np.asarray(ids, dtype=np.int64)


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def load_unseen():
    df = pd.read_csv(UNSEEN_CSV)

    text_col = find_column(
        df,
        ["text", "utterance", "query", "sentence"],
    )
    label_col = find_column(
        df,
        ["intent", "label", "target", "expected_intent"],
    )

    if text_col is None or label_col is None:
        raise RuntimeError(
            "Could not identify text/intent columns in locked 595 CSV."
        )

    out = pd.DataFrame({
        "text": df[text_col].map(normalize_text),
        "intent": df[label_col].astype(str).str.strip(),
    })

    if len(out) != 595:
        raise RuntimeError(
            f"Expected 595 rows, got {len(out)}."
        )

    if not out["intent"].isin(LABELS).all():
        bad = sorted(set(out["intent"]) - set(LABELS))
        raise RuntimeError(f"Unexpected labels: {bad}")

    return out


def load_ood():
    if not OOD_CSV.exists():
        return None

    df = pd.read_csv(OOD_CSV)

    text_col = find_column(
        df,
        ["text", "utterance", "query", "sentence"],
    )

    if text_col is None:
        return None

    return pd.DataFrame({
        "text": df[text_col].map(normalize_text)
    })


def predict_single(session, input_name, output_name, ids):
    logits = session.run(
        [output_name],
        {input_name: ids.reshape(1, MAX_LEN)},
    )[0]
    return np.asarray(logits[0], dtype=np.float32)


def predict_single_all(session, input_name, output_name, ids):
    rows = []
    for row in ids:
        rows.append(
            predict_single(
                session,
                input_name,
                output_name,
                row,
            )
        )
    return np.stack(rows)


def predict_single_repeated(session, input_name, output_name, ids):
    """
    V3 ONNX has a FIXED [1,24] input.
    It cannot accept [N,24]. Therefore "batch inference" is
    intentionally NOT attempted. This function runs the same
    valid [1,24] inference repeatedly and returns [N,11].
    """
    rows = []
    for row in ids:
        rows.append(
            predict_single(
                session,
                input_name,
                output_name,
                row,
            )
        )
    return np.stack(rows)


def critical_mask(texts):
    return np.asarray([
        any(
            phrase in normalize_text(text).lower()
            for phrase in CRITICAL_PHRASES
        )
        for text in texts
    ], dtype=bool)


def reject_rate(probs, threshold):
    return float(
        (probs.max(axis=1) < threshold).mean()
    )


def percentile(values, p):
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def main():
    print("=" * 78)
    print("V3 PRODUCTION AUDIT — PYTHON")
    print("=" * 78)

    print("\nMODE: READ-ONLY")
    print("Training             : NO")
    print("Threshold fitting    : NO")
    print("Model modification   : NO")
    print("ONNX export          : NO")
    print("INT8 quantization    : NO")
    print("595-row training use : NO")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # FILE INTEGRITY
    # --------------------------------------------------------
    if not V3_ONNX.exists():
        raise FileNotFoundError(V3_ONNX)

    if not UNSEEN_CSV.exists():
        raise FileNotFoundError(UNSEEN_CSV)

    vocab_path, vocab = load_vocab()

    model_size_bytes = V3_ONNX.stat().st_size
    model_size_mb = model_size_bytes / (1024 * 1024)

    v3_sha = sha256_file(V3_ONNX)
    vocab_sha = sha256_file(vocab_path)
    unseen_sha = sha256_file(UNSEEN_CSV)

    print("\n--- FILE INTEGRITY ---")
    print("V3 ONNX :", V3_ONNX)
    print("V3 SHA256:", v3_sha)
    print(f"Model size: {model_size_bytes:,} bytes ({model_size_mb:.4f} MB)")
    print("Vocab    :", vocab_path)
    print("Vocab SHA256:", vocab_sha)
    print("Vocab size:", len(vocab))
    print("595 CSV  :", UNSEEN_CSV)
    print("595 SHA256:", unseen_sha)

    # --------------------------------------------------------
    # ORT METADATA
    # --------------------------------------------------------
    session = ort.InferenceSession(
        str(V3_ONNX),
        providers=["CPUExecutionProvider"],
    )

    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]

    print("\n--- ONNX METADATA ---")
    print("Input name :", inp.name)
    print("Input shape:", inp.shape)
    print("Input type :", inp.type)
    print("Output name:", out.name)
    print("Output shape:", out.shape)
    print("Output type :", out.type)
    print("Providers  :", session.get_providers())

    if inp.shape != [1, MAX_LEN]:
        raise RuntimeError(
            f"Expected V3 input [1,{MAX_LEN}], got {inp.shape}"
        )

    if out.shape != [1, NUM_CLASSES]:
        raise RuntimeError(
            f"Expected V3 output [1,{NUM_CLASSES}], got {out.shape}"
        )

    # --------------------------------------------------------
    # DATASET / TOKENIZER
    # --------------------------------------------------------
    df = load_unseen()

    label_to_id = {
        label: i for i, label in enumerate(LABELS)
    }

    truth = np.asarray([
        label_to_id[x]
        for x in df["intent"]
    ], dtype=np.int64)

    ids = np.stack([
        tokenize(text, vocab)
        for text in df["text"]
    ])

    token_sha = hashlib.sha256(
        ids.tobytes()
    ).hexdigest()

    print("\n--- TOKENIZER ---")
    print("Rows:", len(df))
    print("Tensor:", ids.shape)
    print("Token tensor SHA256:", token_sha)

    # --------------------------------------------------------
    # MEMORY SNAPSHOT
    # --------------------------------------------------------
    tracemalloc.start()

    # --------------------------------------------------------
    # BASELINE SINGLE-ROW INFERENCE
    # --------------------------------------------------------
    print("\n--- 595 SINGLE-ROW INFERENCE ---")

    start = time.perf_counter()

    single_logits = predict_single_all(
        session,
        inp.name,
        out.name,
        ids,
    )

    single_elapsed = (
        time.perf_counter() - start
    )

    single_probs = softmax(single_logits)
    single_pred = single_probs.argmax(axis=1)
    single_conf = single_probs.max(axis=1)

    # --------------------------------------------------------
    # FIXED-SHAPE INFERENCE PARITY
    # --------------------------------------------------------
    print("\n--- FIXED-SHAPE INFERENCE PARITY ---")
    print(
        "V3 input is fixed [1,24]; "
        "[595,24] batch inference is NOT supported."
    )
    print(
        "Therefore the audit uses valid [1,24] calls only."
    )

    start = time.perf_counter()

    repeated_logits = predict_single_repeated(
        session,
        inp.name,
        out.name,
        ids,
    )

    repeated_elapsed = (
        time.perf_counter() - start
    )

    max_abs_logit_diff = float(
        np.max(
            np.abs(
                single_logits
                -
                repeated_logits
            )
        )
    )

    repeated_pred = softmax(
        repeated_logits
    ).argmax(axis=1)

    prediction_parity = bool(
        np.array_equal(
            single_pred,
            repeated_pred,
        )
    )

    print(
        f"First pass total    : {single_elapsed:.6f} sec"
    )
    print(
        f"Second pass total   : {repeated_elapsed:.6f} sec"
    )
    print(
        f"First pass avg      : "
        f"{single_elapsed / len(ids) * 1000:.4f} ms"
    )
    print(
        f"Second pass avg     : "
        f"{repeated_elapsed / len(ids) * 1000:.4f} ms"
    )
    print(
        f"Max abs logit diff  : "
        f"{max_abs_logit_diff:.10f}"
    )
    print(
        "Prediction parity   :",
        "PASS" if prediction_parity else "FAIL",
    )

    if not prediction_parity:
        raise RuntimeError(
            "Repeated fixed-shape predictions differ."
        )

    # --------------------------------------------------------
    # CANONICAL METRICS
    # --------------------------------------------------------
    accuracy = accuracy_score(
        truth,
        single_pred,
    )

    macro_f1 = f1_score(
        truth,
        single_pred,
        average="macro",
        zero_division=0,
    )

    prediction_sha = hashlib.sha256(
        single_pred.astype(np.int64).tobytes()
    ).hexdigest()

    logits_sha = hashlib.sha256(
        single_logits.astype(np.float32).tobytes()
    ).hexdigest()

    print("\n--- CANONICAL 595 RESULT ---")
    print(f"Accuracy : {accuracy * 100:.4f}%")
    print(f"Macro F1 : {macro_f1 * 100:.4f}%")
    print("Prediction SHA256:", prediction_sha)
    print("Logits SHA256:", logits_sha)

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------
    print("\n--- CONFIDENCE ---")
    print(
        f"Mean   : {single_conf.mean():.6f}"
    )
    print(
        f"Median : {np.median(single_conf):.6f}"
    )
    print(
        f"Min    : {single_conf.min():.6f}"
    )
    print(
        f"P05    : {percentile(single_conf, 5):.6f}"
    )
    print(
        f"P25    : {percentile(single_conf, 25):.6f}"
    )
    print(
        f"P75    : {percentile(single_conf, 75):.6f}"
    )
    print(
        f"P95    : {percentile(single_conf, 95):.6f}"
    )
    print(
        f"Max    : {single_conf.max():.6f}"
    )

    # --------------------------------------------------------
    # CRITICAL
    # --------------------------------------------------------
    crit = critical_mask(df["text"])

    print("\n--- CRITICAL ---")
    print("Rows:", int(crit.sum()))

    if crit.any():
        crit_acc = accuracy_score(
            truth[crit],
            single_pred[crit],
        )
        print(
            f"Critical accuracy: "
            f"{crit_acc * 100:.2f}%"
        )
    else:
        crit_acc = None
        print("No critical rows detected.")

    # --------------------------------------------------------
    # PER INTENT
    # --------------------------------------------------------
    print("\n--- PER INTENT ---")

    per_intent = []

    for idx, label in enumerate(LABELS):
        mask = truth == idx
        acc = (
            float(
                (single_pred[mask] == idx).mean()
            )
            if mask.any()
            else 0.0
        )

        per_intent.append({
            "intent": label,
            "support": int(mask.sum()),
            "accuracy": acc,
        })

        print(
            f"{label:35s} "
            f"{acc * 100:7.2f}% "
            f"support={int(mask.sum())}"
        )

    # --------------------------------------------------------
    # OOD
    # --------------------------------------------------------
    ood = load_ood()
    ood_results = {}

    print("\n--- OOD ---")

    if ood is None or len(ood) == 0:
        print("OOD file unavailable; skipped.")
    else:
        ood_ids = np.stack([
            tokenize(text, vocab)
            for text in ood["text"]
        ])

        print("OOD rows:", len(ood))

        print(
            "V3 OOD input is fixed [1,24]; "
            "running one row at a time..."
        )

        ood_logits = predict_single_repeated(
            session,
            inp.name,
            out.name,
            ood_ids,
        )

        ood_probs = softmax(
            ood_logits
        )

        for threshold in [
            0.50,
            0.60,
            0.70,
            0.80,
            0.87,
            0.90,
            0.95,
            0.97,
        ]:
            rate = reject_rate(
                ood_probs,
                threshold,
            )

            ood_results[
                f"{threshold:.2f}"
            ] = rate

            print(
                f"threshold {threshold:.2f} | "
                f"reject {rate * 100:6.2f}%"
            )

    # --------------------------------------------------------
    # RUNTIME MEMORY
    # --------------------------------------------------------
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("\n--- PYTHON MEMORY SNAPSHOT ---")
    print(
        f"Current traced : "
        f"{current / (1024*1024):.3f} MB"
    )
    print(
        f"Peak traced    : "
        f"{peak / (1024*1024):.3f} MB"
    )

    psutil_info = None

    try:
        import psutil

        process = psutil.Process()

        before = process.memory_info().rss

        # A small extra inference confirms the runtime remains usable.
        _ = session.run(
            [out.name],
            {inp.name: ids[:1]},
        )

        after = process.memory_info().rss

        psutil_info = {
            "rss_before_bytes": int(before),
            "rss_after_bytes": int(after),
        }

        print(
            f"RSS before : "
            f"{before / (1024*1024):.3f} MB"
        )
        print(
            f"RSS after  : "
            f"{after / (1024*1024):.3f} MB"
        )

    except ImportError:
        print(
            "psutil not installed; process RSS skipped."
        )

    # --------------------------------------------------------
    # PRODUCTION CHECKS
    # --------------------------------------------------------
    checks = {
        "model_exists": True,
        "vocab_size_895": len(vocab) == VOCAB_SIZE,
        "input_shape_1x24": inp.shape == [1, MAX_LEN],
        "output_shape_1x11": inp.shape == [1, MAX_LEN]
        and out.shape == [1, NUM_CLASSES],
        "595_rows": len(df) == 595,
        "fixed_shape_prediction_parity":
            prediction_parity,
        "canonical_accuracy":
            abs(accuracy - 0.9630252100840336) < 1e-10,
        "canonical_macro_f1":
            abs(macro_f1 - 0.963117) < 0.001,
    }

    # Note:
    # The exact canonical accuracy above is derived from 573/595.
    # Macro F1 is checked with tolerance because sklearn versions
    # can display/report floating-point rounding differently.

    print("\n" + "=" * 78)
    print("PRODUCTION AUDIT CHECKS")
    print("=" * 78)

    for name, value in checks.items():
        print(
            f"{name:38s}: "
            f"{'PASS' if value else 'FAIL'}"
        )

    production_eligible = all(checks.values())

    print("\nSTATUS:")
    if production_eligible:
        print(
            "V3 PYTHON PRODUCTION AUDIT: PASS"
        )
        print(
            "Proceed to Android/iOS integration testing."
        )
    else:
        print(
            "V3 PYTHON PRODUCTION AUDIT: REVIEW REQUIRED"
        )
        print(
            "Do not freeze deployment until failed checks are resolved."
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------
    details = df.copy()

    details["prediction"] = [
        LABELS[x]
        for x in single_pred
    ]

    details["confidence"] = single_conf
    details["correct"] = (
        single_pred == truth
    )
    details["critical"] = crit

    details_path = (
        OUTPUT_DIR
        / "v3_595_production_audit_details.csv"
    )

    details.to_csv(
        details_path,
        index=False,
    )

    summary = {
        "model": str(V3_ONNX),
        "model_sha256": v3_sha,
        "model_size_bytes": model_size_bytes,
        "model_size_mb": model_size_mb,

        "vocab": str(vocab_path),
        "vocab_sha256": vocab_sha,
        "vocab_size": len(vocab),

        "unseen_csv": str(UNSEEN_CSV),
        "unseen_sha256": unseen_sha,
        "unseen_rows": len(df),

        "onnx": {
            "input_name": inp.name,
            "input_shape": inp.shape,
            "input_type": inp.type,
            "output_name": out.name,
            "output_shape": out.shape,
            "output_type": out.type,
            "providers": session.get_providers(),
        },

        "token_tensor_sha256": token_sha,
        "prediction_sha256": prediction_sha,
        "logits_sha256": logits_sha,

        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),

        "single_inference_total_sec":
            single_elapsed,
        "single_inference_avg_ms":
            single_elapsed / len(ids) * 1000,

        "second_fixed_shape_pass_total_sec":
            repeated_elapsed,
        "second_fixed_shape_pass_avg_ms":
            repeated_elapsed / len(ids) * 1000,

        "max_abs_logit_difference_single_vs_batch":
            max_abs_logit_diff,

        "prediction_parity":
            prediction_parity,

        "critical_rows":
            int(crit.sum()),
        "critical_accuracy":
            None
            if crit_acc is None
            else float(crit_acc),

        "confidence": {
            "mean": float(single_conf.mean()),
            "median": float(np.median(single_conf)),
            "min": float(single_conf.min()),
            "p05": percentile(single_conf, 5),
            "p25": percentile(single_conf, 25),
            "p75": percentile(single_conf, 75),
            "p95": percentile(single_conf, 95),
            "max": float(single_conf.max()),
        },

        "ood_rejection": ood_results,

        "python_tracemalloc": {
            "current_bytes": int(current),
            "peak_bytes": int(peak),
        },

        "process_memory": psutil_info,

        "checks": checks,
        "production_audit_pass":
            production_eligible,

        "training_occurred": False,
        "threshold_fitting_occurred": False,
        "model_modified": False,
        "onnx_export_occurred": False,
        "int8_quantization_occurred": False,
        "unseen_used_for_training": False,
    }

    summary_path = (
        OUTPUT_DIR
        / "v3_production_audit_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSaved:")
    print(details_path)
    print(summary_path)


if __name__ == "__main__":
    main()
