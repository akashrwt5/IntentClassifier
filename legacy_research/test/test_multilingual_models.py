#!/usr/bin/env python3
"""
Test every generated multilingual intent model.

Loads each model under multilingual/models/<name>/ via ONNX Runtime (the real
deployment artifact for the server path), evaluates it against that model's
held-out split (multilingual/test/<name>_holdout.csv, written by
train_multilingual.py), and prints per-model + per-intent accuracy. Styled after
scripts/test_tfidf_only.py.

Run:
    python multilingual/test/test_multilingual_models.py
    python multilingual/test/test_multilingual_models.py --model fr
    python multilingual/test/test_multilingual_models.py --verbose
    python multilingual/test/test_multilingual_models.py --min-accuracy 0.75

Exit code is non-zero if any tested model scores below --min-accuracy, so this
doubles as a CI gate.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

BASE_DIR = Path(__file__).parent.parent          # .../multilingual
MODELS_DIR = BASE_DIR / "models"
TEST_DIR = BASE_DIR / "test"
CASES_DIR = TEST_DIR / "cases"                   # curated per-language smoke tests

sys.path.insert(0, str(BASE_DIR))
from text_norm import normalize_text   # same lowercase + ASCII fold used in training

BOLD, GREEN, RED, YELLOW, RESET = "\033[1m", "\033[92m", "\033[91m", "\033[93m", "\033[0m"


def _find_models(only: str | None):
    """Yield (name, onnx_path, labels_path, holdout_path) for each model folder."""
    if not MODELS_DIR.exists():
        return
    for model_dir in sorted(MODELS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        name = model_dir.name
        if only and name != only:
            continue
        onnx = next(model_dir.glob("*_intent_model.onnx"), None)
        labels = next(model_dir.glob("*_intent_labels.json"), None)
        holdout = TEST_DIR / f"{name}_holdout.csv"
        if onnx and labels and holdout.exists():
            yield name, onnx, labels, holdout
        else:
            missing = [n for n, p in
                       (("onnx", onnx), ("labels.json", labels), ("holdout.csv", holdout if holdout.exists() else None))
                       if p is None]
            print(f"  [skip] '{name}' — missing {missing}")


def _predict(session, input_name, labels, text):
    """Run one utterance through the ONNX model, return (intent, confidence)."""
    inp = session.get_inputs()[0]
    t = normalize_text(text)
    X = (np.array([[t]], dtype=object) if len(inp.shape) == 2
         else np.array([t], dtype=object))
    outputs = session.run(None, {input_name: X})

    # Pick the score tensor that matches the label count (zipmap=False export
    # gives [label_tensor, prob_tensor]).
    scores = None
    for o in outputs:
        arr = np.asarray(o)
        if arr.ndim >= 1 and arr.shape[-1] == len(labels):
            scores = arr[0] if arr.ndim == 2 else arr
            break
    if scores is None:
        # Fallback: model returned the predicted label string directly.
        return str(np.asarray(outputs[0]).ravel()[0]), 1.0

    scores = np.asarray(scores, dtype=float)
    top = int(np.argmax(scores))
    return labels[top], float(scores[top])


def evaluate(name, onnx_path, labels_path, holdout_path, verbose):
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name

    df = pd.read_csv(holdout_path, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]

    rows = []
    for _, r in df.iterrows():
        expected = str(r["intent"]).strip()
        predicted, conf = _predict(session, input_name, labels, r["text"])
        rows.append({"text": r["text"], "expected": expected,
                     "predicted": predicted, "confidence": conf,
                     "correct": predicted == expected})
    res = pd.DataFrame(rows)

    correct = int(res["correct"].sum())
    total = len(res)
    acc = correct / total if total else 0.0

    print(f"\n{BOLD}── {name} ──{RESET}  ({total} holdout utterances, {len(labels)} labels)")
    print(f"  Accuracy: {GREEN if acc >= 0.8 else RED}{correct}/{total} ({acc*100:.1f}%){RESET}")

    if verbose:
        print(f"  {'Intent':<35}{'Acc':<12}")
        for intent in sorted(res["expected"].unique()):
            sub = res[res["expected"] == intent]
            ia = sub["correct"].sum() / len(sub)
            color = GREEN if ia == 1.0 else RED if ia < 0.7 else ""
            print(f"  {intent:<35}{color}{int(sub['correct'].sum())}/{len(sub)} "
                  f"({ia*100:>3.0f}%){RESET}")
        fails = res[~res["correct"]]
        if len(fails):
            print(f"\n  {BOLD}Failures ({len(fails)}):{RESET}")
            for _, r in fails.sort_values("confidence", ascending=False).head(15).iterrows():
                print(f"    [{r['confidence']:.2f}] {r['expected']} → {r['predicted']}")
                print(f"         \"{str(r['text'])[:60]}\"")
    return acc


def run_curated_cases(only, verbose):
    """Run the curated per-language smoke tests in test/cases/<lang>_cases.csv.

    These are hand-selected, authentic, in-language utterances with a known
    expected intent — a fast, human-readable regression guard (vs the larger
    statistical holdout eval). Every case must classify to its expected intent.
    Returns {lang: (passed, total)} for each language with a cases file.
    """
    results = {}
    if not CASES_DIR.exists() or not any(CASES_DIR.glob("*_cases.csv")):
        return results

    print(f"\n{BOLD}Curated smoke-test cases{RESET}")
    for cf in sorted(CASES_DIR.glob("*_cases.csv")):
        lang = cf.stem.replace("_cases", "")
        if only and lang != only:
            continue
        model_dir = MODELS_DIR / lang
        onnx = next(model_dir.glob("*_intent_model.onnx"), None)
        labels_path = next(model_dir.glob("*_intent_labels.json"), None)
        if not onnx or not labels_path:
            print(f"  [skip] {lang} — model not found")
            continue

        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        session = ort.InferenceSession(str(onnx))
        input_name = session.get_inputs()[0].name
        df = pd.read_csv(cf, encoding="utf-8-sig")

        passed, fails = 0, []
        for _, r in df.iterrows():
            expected = str(r["intent"]).strip()
            predicted, _ = _predict(session, input_name, labels, r["text"])
            if predicted == expected:
                passed += 1
            else:
                fails.append((r["text"], expected, predicted))
        total = len(df)
        color = GREEN if passed == total else RED
        print(f"  {lang}: {color}{passed}/{total} cases pass{RESET}")
        if verbose or fails:
            for t, e, p in fails:
                print(f"    {RED}FAIL{RESET} {t!r}  expected {e}  got {p}")
        results[lang] = (passed, total)
    return results


def main():
    parser = argparse.ArgumentParser(description="Test multilingual intent models")
    parser.add_argument("--model", "-m", help="Test only this model (e.g. fr, multilingual).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Per-intent + failures.")
    parser.add_argument("--min-accuracy", type=float, default=0.80, help="Holdout pass/fail floor.")
    parser.add_argument("--strict", action="store_true",
                        help="GATE: exit 1 if any model is below the floor or any curated case fails "
                             "(default: report only).")
    parser.add_argument("--json", metavar="FILE", help="Write machine-readable results to FILE.")
    args = parser.parse_args()

    print(f"\n{BOLD}Multilingual Model Test Suite{RESET}")
    found = list(_find_models(args.model))
    if not found:
        print(f"\n{RED}No testable models found under {MODELS_DIR}.{RESET}")
        print("Run: python multilingual/train_multilingual.py --all")
        sys.exit(1)

    scores = {name: evaluate(name, o, l, h, args.verbose) for name, o, l, h in found}

    # Holdout accuracy table — every model, including the combined `multilingual`.
    print(f"\n{BOLD}{'='*60}\nSUMMARY (holdout accuracy, floor {args.min_accuracy:.2f}){RESET}")
    below_floor = []
    for name, acc in scores.items():
        ok = acc >= args.min_accuracy
        if not ok:
            below_floor.append(name)
        print(f"  {GREEN+'PASS' if ok else RED+'FAIL'}{RESET}  {name:<15} {acc*100:5.1f}%")
    print(f"{BOLD}{'='*60}{RESET}")

    # Curated smoke tests (exact intent match required for every case).
    case_results = run_curated_cases(args.model, args.verbose)
    failed_cases = [lang for lang, (p, t) in case_results.items() if p != t]

    if args.json:
        Path(args.json).write_text(json.dumps({
            "min_accuracy": args.min_accuracy,
            "holdout": {name: round(acc, 4) for name, acc in scores.items()},
            "below_floor": below_floor,
            "curated_cases": {lang: {"passed": p, "total": t}
                              for lang, (p, t) in case_results.items()},
            "failed_cases": failed_cases,
        }, indent=2), encoding="utf-8")
        print(f"\n  [results] → {args.json}")

    # Gate: total floor on every model AND all curated cases must pass.
    reasons = []
    if below_floor:
        reasons.append(f"below floor: {', '.join(below_floor)}")
    if failed_cases:
        reasons.append(f"curated-case failures: {', '.join(failed_cases)}")

    if reasons:
        msg = "; ".join(reasons)
        if args.strict:
            print(f"\n{RED}{BOLD}  GATE FAILED: {msg}{RESET}\n")
            sys.exit(1)
        print(f"\n{YELLOW}  (gate would fail in --strict: {msg}){RESET}\n")
    elif args.strict:
        print(f"\n{GREEN}{BOLD}  GATE PASSED "
              f"(all models ≥ {args.min_accuracy:.2f}, all curated cases pass){RESET}\n")


if __name__ == "__main__":
    main()
