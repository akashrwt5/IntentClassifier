"""Try the shipped model by hand.

Deliberately loads ONLY what the phone loads — the INT8 ONNX file and
runtime_config.json. It does not import the training pipeline, does not touch
the pickled sklearn model, and recomputes the gate from the config rather than
from any Python object. If this disagrees with the app, the app is wrong; if it
agrees with the training-time numbers, the exported artefact is sound.

    python scripts/predict.py                       # interactive
    python scripts/predict.py "turn it up a bit"    # one-shot
    python scripts/predict.py --file questions.txt  # batch, one per line
    python scripts/predict.py --model models/final_student_256/onnx

Output is the full runtime contract, including the refusals — a REJECT with its
reason is the interesting case, not a failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = "models/final_student_256/onnx"


class Runtime:
    def __init__(self, onnx_dir: Path, quant: str = "int8"):
        cfg_path = onnx_dir / "runtime_config.json"
        if not cfg_path.exists():
            raise SystemExit(f"no runtime_config.json in {onnx_dir}")
        self.cfg = json.loads(cfg_path.read_text())
        model = onnx_dir / f"intent_{quant}.onnx"
        if not model.exists():
            model = onnx_dir / "intent_fp32.onnx"
        self.sess = ort.InferenceSession(str(model),
                                         providers=["CPUExecutionProvider"])
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.model_path = model

        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(str(onnx_dir / "tokenizer"),
                                                 local_files_only=True)
        self.labels = self.cfg["labels"]
        self.gate = self.cfg["gate"]
        self.max_len = self.cfg.get("tokenizer", {}).get("max_len", 64)
        self.prefix = self.cfg.get("tokenizer", {}).get("prefix", "")
        ood = self.cfg.get("ood") or {}
        self.centroids = (np.array(ood["whitened_centroids"], dtype=np.float64)
                          if ood.get("whitened_centroids") else None)
        self.ood_threshold = self.cfg.get("ood_threshold")

        import re
        pat = self.gate.get("corrective_pattern")
        self.corrective = re.compile(pat) if pat else None

        # Signal 6. Absent from runtime_config until fit_asr_threshold.py has
        # been run on real recordings from the target device, and inert when
        # absent — an app that passes no confidence behaves exactly as before.
        asr = self.cfg.get("asr") or {}
        self.asr_enabled = bool(asr.get("enabled"))
        self.asr_min = asr.get("min_confidence")

        # Health check on the artefact itself. Two failure modes have shipped
        # into this directory during development and both look, from the
        # outside, like a cautious model rather than a broken one. Say so up
        # front rather than letting someone spend an afternoon typing
        # sentences into a model that was never going to accept any of them.
        self.warnings = []
        by_risk = self.gate.get("conf_by_risk") or {}
        for tier, thr in by_risk.items():
            if thr >= 0.995:
                self.warnings.append(
                    f"{tier}-risk threshold is {thr} — this is the value the "
                    f"threshold fitter returns when it GIVES UP, not a fitted "
                    f"number. Nothing in that tier will ever be accepted. "
                    f"Look for 'WARNING: no confidence threshold reaches' in "
                    f"the train_classifier.py output that produced this model.")
        if self.ood_threshold is not None and self.ood_threshold < 25:
            self.warnings.append(
                f"OOD threshold is {self.ood_threshold:.1f}, which is low. If "
                f"ordinary short phrases come back as 'unlike anything in "
                f"training', the OOD score has lost its meaning rather than "
                f"the input being unusual — check the AUROC line in the "
                f"training output (below ~0.85 means this signal is noise).")

    def __call__(self, text: str, asr_confidence: float | None = None) -> dict:
        enc = self.tok([self.prefix + text], padding="max_length",
                       truncation=True, max_length=self.max_len,
                       return_tensors="np")
        out = self.sess.run(None, {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64)})
        probs = out[self.out_names.index("probs")][0].astype(np.float64)

        ood_score = None
        if self.centroids is not None and "whitened_embedding" in self.out_names:
            z = out[self.out_names.index("whitened_embedding")][0].astype(np.float64)
            d2 = ((z ** 2).sum() - 2 * self.centroids @ z
                  + (self.centroids ** 2).sum(1))
            ood_score = float(np.sqrt(max(d2.min(), 0.0)))

        order = np.argsort(-probs)
        i1, i2 = int(order[0]), int(order[1])
        c1, c2 = float(probs[i1]), float(probs[i2])
        intent = self.labels[i1]
        margin = c1 - c2
        tier = (self.gate.get("risk_of") or {}).get(intent, "normal")
        thr = (self.gate.get("conf_by_risk") or {}).get(
            tier, self.gate["conf_threshold"])

        # Raw text, not normalized — the comma is the structural signal.
        is_corr = bool(self.gate.get("reject_corrective") and self.corrective
                       and self.corrective.search(text.lower()))

        # ASR confidence is checked FIRST. Everything below it reasons about
        # what the words mean; this asks whether the words were heard, and
        # whether the person was talking to the device at all. There is no
        # point weighing a 57-way softmax over a sentence the recognizer
        # scraped out of someone else's conversation.
        if self.asr_enabled and self.asr_min is not None \
                and asr_confidence is not None and asr_confidence < self.asr_min:
            accepted, reason = False, (f"recognizer confidence "
                                       f"{asr_confidence:.3f} < {self.asr_min:.3f}")
        elif intent == self.gate["reject_label"]:
            accepted, reason = False, "classified as unsupported"
        elif is_corr:
            accepted, reason = False, "corrective phrasing — ask again"
        elif ood_score is not None and self.ood_threshold and \
                ood_score > self.ood_threshold:
            accepted, reason = False, "unlike anything in training"
        elif c1 < thr:
            accepted, reason = False, f"below {tier}-risk threshold {thr:.3f}"
        elif margin < self.gate["margin_threshold"]:
            accepted, reason = False, "top-1/top-2 margin too small"
        else:
            accepted, reason = True, "above calibrated threshold"

        return dict(text=text, intent=intent, confidence=c1, risk=tier,
                    threshold=thr, top2=self.labels[i2], top2_score=c2,
                    margin=margin, ood_score=ood_score,
                    asr_confidence=asr_confidence,
                    accepted=accepted, reason=reason)


def show(r: dict) -> None:
    mark = "ACCEPT" if r["accepted"] else "REJECT"
    ood = f"  ood={r['ood_score']:.2f}" if r["ood_score"] is not None else ""
    if r.get("asr_confidence") is not None:
        ood += f"  asr={r['asr_confidence']:.2f}"
    print(f"\n  {mark}  {r['intent']}"
          + (f"  [{r['risk']}-risk]" if r["risk"] != "normal" else ""))
    print(f"    confidence {r['confidence']:.4f} (needs {r['threshold']:.3f})"
          f"   margin {r['margin']:.4f}{ood}")
    print(f"    runner-up  {r['top2']} {r['top2_score']:.4f}")
    print(f"    {r['reason']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*")
    ap.add_argument("--model", default=DEFAULT)
    ap.add_argument("--quant", default="int8", choices=["int8", "fp32"])
    ap.add_argument("--file", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--asr-confidence", type=float, default=None,
                    help="simulate a recognizer score for this utterance, to "
                         "see signal 6 fire. Only does anything once "
                         "fit_asr_threshold.py has written an asr block.")
    args = ap.parse_args()

    rt = Runtime(ROOT / args.model, args.quant)
    print(f"{rt.model_path.name}  "
          f"{rt.model_path.stat().st_size/1e6:.2f} MB  "
          f"{len(rt.labels)} intents")
    print(f"gate: normal>={rt.gate['conf_by_risk']['normal']}  "
          f"high>={rt.gate['conf_by_risk']['high']}  "
          f"margin>={rt.gate['margin_threshold']}  "
          f"ood<={rt.ood_threshold:.2f}" if rt.ood_threshold else "")

    if rt.asr_enabled:
        print(f"      asr confidence >= {rt.asr_min:.3f}")
    else:
        print("      asr confidence: not fitted — signal 6 inert "
              "(see scripts/fit_asr_threshold.py)")

    for w in rt.warnings:
        print(f"\n  !! {w}")
    if rt.warnings:
        print("\n  Testing this artefact will tell you about the breakage, not "
              "about the design. Rebuild before drawing conclusions.")

    if args.file:
        for line in Path(args.file).read_text().splitlines():
            if line.strip():
                show(rt(line.strip(), args.asr_confidence))
        return
    if args.text:
        r = rt(" ".join(args.text), args.asr_confidence)
        print(json.dumps(r, indent=2) if args.json else "", end="")
        if not args.json:
            show(r)
        return

    print("\nType a command. Blank line or Ctrl-D to quit.\n")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            return
        # "text @0.42" sets a recognizer confidence for that one line.
        conf = args.asr_confidence
        if "@" in line:
            head, _, tail = line.rpartition("@")
            try:
                conf, line = float(tail), head.strip()
            except ValueError:
                pass
        show(rt(line, conf))


if __name__ == "__main__":
    main()
