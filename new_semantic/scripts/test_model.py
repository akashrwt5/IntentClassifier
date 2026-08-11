#!/usr/bin/env python3
"""
test_model.py — installed student model ko test karo (jo production mein chalega).

PRODUCTION MODEL: models/semantic_student/en/student.onnx
Yahi file runtime par load hoti hai — same as what the device runs.

Size info aur model info automatically show hoga.

Usage:
    python scripts/test_model.py
    python scripts/test_model.py "make it louder"
    python scripts/test_model.py "mute it" "volume up" "switch to restaurant"
    python scripts/test_model.py --file phrases.txt
    python scripts/test_model.py --topk 5
    python scripts/test_model.py --with-stage2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
REPO        = Path(__file__).resolve().parents[2]
STUDENT_DIR = REPO / "models" / "semantic_student" / "en"
META_PATH   = STUDENT_DIR / "meta.json"
ONNX_PATH   = STUDENT_DIR / "student.onnx"
VOCAB_PATH  = STUDENT_DIR / "vocab.json"
LABELS_PATH = STUDENT_DIR / "labels.json"

# for --with-stage2
STAGE2_ONNX    = REPO / "models" / "intent_model.onnx"
STAGE2_WEIGHTS = REPO / "models" / "intent_classifier_weights.json"
STAGE2_LABELS  = REPO / "models" / "intent_labels.json"

PAD_ID = 0
UNK_ID = 1

import re
import unicodedata
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


# ── tokenizer (byte-identical to runtime semantic.py) ─────────────────────────
def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKD", str(text)).replace("\u2019", "'")

def _word_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(_normalize(text).lower())

def _wordpiece(word: str, vocab: dict) -> list[str]:
    out, start = [], 0
    while start < len(word):
        end = len(word)
        cur = None
        while start < end:
            piece = word[start:end] if start == 0 else "##" + word[start:end]
            if piece in vocab:
                cur = piece
                break
            end -= 1
        if cur is None:
            out.append("[UNK]")
            start += 1
        else:
            out.append(cur)
            start = end
    return out

def encode_subword(text: str, vocab: dict, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    pieces = [p for w in _word_tokens(text) for p in _wordpiece(w, vocab)]
    ids = [vocab.get(p, UNK_ID) for p in pieces][:max_len]
    ids += [PAD_ID] * (max_len - len(ids))
    arr = np.array([ids], dtype=np.int64)
    return arr, arr != PAD_ID

def encode_word(text: str, vocab: dict, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    toks = _word_tokens(text)
    ids = [vocab.get(t, UNK_ID) for t in toks][:max_len]
    ids += [PAD_ID] * (max_len - len(ids))
    arr = np.array([ids], dtype=np.int64)
    return arr, arr != PAD_ID


# ── stage 2 helper ────────────────────────────────────────────────────────────
def _stage2_predict(text: str):
    import onnxruntime as ort
    w = json.loads(STAGE2_WEIGHTS.read_text(encoding="utf-8"))
    T = float(w.get("temperature", 1.0))
    gate = float(w.get("conf_threshold", 0.70))
    labels = json.loads(STAGE2_LABELS.read_text(encoding="utf-8"))
    sess = ort.InferenceSession(str(STAGE2_ONNX), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    _, scores = sess.run(None, {name: np.array([text], dtype=object).reshape(-1, 1)})
    z = np.array(scores, dtype=np.float64)[0] / T
    z -= z.max()
    e = np.exp(z)
    p = e / e.sum()
    top = int(np.argmax(p))
    return labels[top], float(p[top]), gate


# ── colours ───────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def _bar(p: float, width: int = 28) -> str:
    filled = int(p * width)
    return "█" * filled + "░" * (width - filled)

def _conf_color(conf: float, threshold: float) -> str:
    if conf >= threshold:   return GREEN
    if conf >= threshold * 0.7: return YELLOW
    return RED


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Test the installed student model (production artifact)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python scripts/test_model.py
  python scripts/test_model.py "make it louder"
  python scripts/test_model.py "mute it" "volume up" "turn on restaurant mode"
  python scripts/test_model.py --with-stage2 "i need it quieter"
  python scripts/test_model.py --file phrases.txt --topk 5
"""
    )
    ap.add_argument("text",  nargs="*",  help="utterance(s) to test (omit for interactive mode)")
    ap.add_argument("--topk",       type=int,   default=3,    help="show top-K intents (default: 3)")
    ap.add_argument("--with-stage2",action="store_true",      help="also show Stage 2 (TF-IDF) prediction for comparison")
    ap.add_argument("--file",       type=Path,  default=None, help="text file: one utterance per line")
    ap.add_argument("--no-color",   action="store_true",      help="disable ANSI colors")
    args = ap.parse_args()

    # ── validate install ───────────────────────────────────────────────────────
    for f in (ONNX_PATH, VOCAB_PATH, LABELS_PATH):
        if not f.exists():
            raise SystemExit(
                f"ERROR: {f.name} not found in {STUDENT_DIR}\n"
                "Run:  python scripts/install_student.py --tag subw_vol5_s1 --threshold 0.40"
            )

    if args.no_color:
        for g in ["GREEN","RED","YELLOW","CYAN","BOLD","DIM","RESET"]:
            globals()[g] = ""

    # ── load artifacts ─────────────────────────────────────────────────────────
    import onnxruntime as ort

    meta   = json.loads(META_PATH.read_text(encoding="utf-8"))
    raw    = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    vocab  = raw["vocab"] if isinstance(raw, dict) and "vocab" in raw else raw
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    tok_mode   = meta.get("tokenizer", "word")
    max_len    = int(meta.get("max_len", 32))
    threshold  = float(meta.get("threshold", 0.40))
    temperature = float(meta.get("temperature", 1.0))

    sess   = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    in_ids  = sess.get_inputs()[0].name
    in_mask = sess.get_inputs()[1].name

    onnx_mb = ONNX_PATH.stat().st_size / 1_000_000
    vocab_kb = VOCAB_PATH.stat().st_size / 1_000

    # ── header ─────────────────────────────────────────────────────────────────
    print()
    print(f"{BOLD}{'━'*60}{RESET}")
    print(f"{BOLD}{CYAN}  🎧 Intent Classifier — Installed Model Info{RESET}")
    print(f"{BOLD}{'━'*60}{RESET}")
    print(f"  Model file : {ONNX_PATH.name}  ({CYAN}{onnx_mb:.2f} MB{RESET})")
    print(f"  Tag        : {meta.get('tag', 'unknown')}")
    print(f"  Tokenizer  : {CYAN}{tok_mode}{RESET}   vocab {len(vocab)} tokens  max_len {max_len}")
    print(f"  Teacher    : {meta.get('teacher', '—')}")
    print(f"  Temperature: {temperature}  (calibrated — ECE 0.019)")
    print(f"  Gate       : {threshold}  (ACCEPT if confidence ≥ {threshold})")
    print(f"  Intents    : {len(labels)}")
    print(f"  Vocab file : {vocab_kb:.1f} KB")
    if meta.get("synthetic_rows"):
        print(f"  {DIM}Trained with {meta['synthetic_rows']} synthetic rows  (synthetic_text={meta.get('synthetic_text')}){RESET}")
    print(f"{BOLD}{'━'*60}{RESET}")
    print()

    # ── predict fn ────────────────────────────────────────────────────────────
    def predict(text: str) -> tuple[np.ndarray, list[str]]:
        if tok_mode == "subword":
            ids, mask = encode_subword(text, vocab, max_len)
        else:
            ids, mask = encode_word(text, vocab, max_len)
        logits = sess.run(None, {in_ids: ids, in_mask: mask})[0][0]
        z = logits / temperature
        z = z - z.max()
        e = np.exp(z)
        p = e / e.sum()
        # token display
        if tok_mode == "subword":
            toks = [p for w in _word_tokens(text) for p in _wordpiece(w, vocab)]
            tok_display = [(t, t == "[UNK]") for t in toks]
        else:
            toks = _word_tokens(text)
            tok_display = [(w, w not in vocab) for w in toks]
        return p, tok_display

    # ── show fn ───────────────────────────────────────────────────────────────
    def show(text: str):
        text = text.strip()
        if not text:
            return
        p, tok_display = predict(text)

        # tokens
        rendered = " ".join(
            f"{RED}[{t}]{RESET}" if unk else t
            for t, unk in tok_display
        )
        unk_words = [t for t, u in tok_display if u]
        print(f"  {DIM}tokens  :{RESET}  {rendered}")
        if unk_words:
            print(f"  {RED}⚠  {len(unk_words)}/{len(tok_display)} pieces unknown → {unk_words}{RESET}")

        # top-k intents
        order = np.argsort(-p)[: args.topk]
        print()
        for rank, i in enumerate(order):
            color = _conf_color(float(p[i]), threshold) if rank == 0 else DIM
            marker = "▶" if rank == 0 else " "
            bar = _bar(float(p[i]))
            print(f"  {color}{marker} #{rank+1}  {labels[i]:<35} {p[i]:.4f}  {bar}{RESET}")
        print()

        # verdict
        top_label = labels[order[0]]
        top_conf  = float(p[order[0]])
        fallback_intent = "Default Fallback Intent"
        if top_label == fallback_intent:
            verdict = f"{RED}✗  REJECT  (predicted fallback){RESET}"
        elif top_conf < threshold:
            verdict = f"{YELLOW}✗  REJECT  (conf {top_conf:.4f} < gate {threshold}){RESET}"
        else:
            verdict = f"{GREEN}✔  ACCEPT  →  {top_label}  ({top_conf:.4f}){RESET}"
        print(f"  {BOLD}verdict : {RESET}{verdict}")

        # optional stage 2
        if args.with_stage2 and STAGE2_ONNX.exists():
            s2_label, s2_conf, s2_gate = _stage2_predict(text)
            s2_fires = s2_conf >= s2_gate
            s2_color = GREEN if s2_fires else DIM
            s2_mark  = "FIRES" if s2_fires else "hands over to Stage 3"
            print(f"  {s2_color}Stage 2 : {s2_label:<35} {s2_conf:.4f}  {s2_mark}{RESET}")

        print(f"  {'─'*56}")
        print()

    # ── run ───────────────────────────────────────────────────────────────────
    if args.file:
        lines = [l for l in args.file.read_text(encoding="utf-8").splitlines() if l.strip()]
        for line in lines:
            print(f"{BOLD}> {line}{RESET}")
            show(line)
        return 0

    if args.text:
        for t in args.text:
            print(f"{BOLD}> {t}{RESET}")
            show(t)
        return 0

    # interactive
    print(f"  {DIM}Interactive mode — type utterance, Enter to test, blank/Ctrl-D to quit{RESET}\n")
    while True:
        try:
            t = input(f"{BOLD}> {RESET}")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not t.strip():
            return 0
        show(t)


if __name__ == "__main__":
    raise SystemExit(main())
