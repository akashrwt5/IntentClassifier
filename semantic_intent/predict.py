"""
Runtime predictor. One ONNX file + one vocab file, no scikit-learn, no torch.

    from semantic_intent import SemanticIntentClassifier
    clf = SemanticIntentClassifier()
    clf.predict("it's too quiet here can you make it louder")
    # Prediction(intent='device.volume.increase', confidence=0.97,
    #            ood_score=0.71, accepted=True)

CLI:
    python -m semantic_intent.predict "turn it down a bit"
    python -m semantic_intent.predict --file utterances.txt --json

Two gates decide acceptance, and they answer different questions:
  * `confidence`  — "given this is in scope, how sure am I which intent?"
  * `ood_score`   — "is this in scope at all?" (max cosine to a training
                    prototype; a linear head has no way to say 'none of these')
An utterance is accepted only if both clear their thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Sequence

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
DEFAULT_MODEL = MODEL_DIR / "semantic_intent.onnx"
DEFAULT_VOCAB = MODEL_DIR / "minilm-vocab.txt"

FALLBACK_INTENT = "Default Fallback Intent"


@dataclass
class Prediction:
    intent: str
    confidence: float
    ood_score: float
    accepted: bool
    text: str = ""

    @property
    def routed_intent(self) -> str:
        """Intent to act on — the fallback label when the gates reject."""
        return self.intent if self.accepted else FALLBACK_INTENT

    def __repr__(self) -> str:  # compact, readable in logs
        return (
            f"Prediction(intent={self.intent!r}, confidence={self.confidence:.2f}, "
            f"ood_score={self.ood_score:.2f}, accepted={self.accepted})"
        )


class SemanticIntentClassifier:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        vocab_path: str | Path = DEFAULT_VOCAB,
        max_len: int = 64,
        threads: int = 2,
        conf_threshold: float | None = None,
        ood_threshold: float | None = None,
    ):
        import onnxruntime as ort
        from tokenizers import BertWordPieceTokenizer

        model_path, vocab_path = Path(model_path), Path(vocab_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found. Train it first:\n"
                f"  python -m semantic_intent.train --data <dataset.xlsx>"
            )
        if not vocab_path.exists():
            raise FileNotFoundError(vocab_path)

        self.tokenizer = BertWordPieceTokenizer(str(vocab_path), lowercase=True)
        self.tokenizer.enable_truncation(max_len)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), opts, providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self.session.get_inputs()}

        meta = self.session.get_modelmeta().custom_metadata_map
        self.labels = np.array(meta["intent_labels"].split(","))
        self.conf_threshold = (
            conf_threshold if conf_threshold is not None else float(meta.get("conf_threshold", 0.5))
        )
        self.ood_threshold = (
            ood_threshold if ood_threshold is not None else float(meta.get("ood_threshold", 0.5))
        )

        self._warm_up()

    # ------------------------------------------------------------------
    def _warm_up(self) -> None:
        """First ORT run pays graph/allocator setup; keep it off the first turn."""
        try:
            self._run("warm up")
        except Exception:
            pass

    def _run(self, text: str):
        enc = self.tokenizer.encode(text if text.strip() else "[UNK]")
        ids = np.array([enc.ids], np.int64)
        mask = np.array([enc.attention_mask], np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feeds["token_type_ids"] = np.zeros_like(ids)
        probs, ood, _ = self.session.run(["probabilities", "ood_score", "embedding"], feeds)
        return probs[0], float(ood[0])

    # ------------------------------------------------------------------
    def predict(self, text: str) -> Prediction:
        probs, ood = self._run(text)
        idx = int(probs.argmax())
        conf = float(probs[idx])
        accepted = conf >= self.conf_threshold and ood >= self.ood_threshold
        return Prediction(str(self.labels[idx]), conf, ood, accepted, text)

    def predict_batch(self, texts: Sequence[str]) -> List[Prediction]:
        # The graph is fixed at batch 1 for on-device parity; loop deliberately.
        return [self.predict(t) for t in texts]

    def top_k(self, text: str, k: int = 3):
        probs, _ = self._run(text)
        order = np.argsort(probs)[::-1][:k]
        return [(str(self.labels[i]), float(probs[i])) for i in order]


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Predict intents.")
    ap.add_argument("text", nargs="*", help="utterance(s) to classify")
    ap.add_argument("--file", type=Path, help="file with one utterance per line")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    texts = list(args.text)
    if args.file:
        texts += [l.strip() for l in args.file.read_text().splitlines() if l.strip()]
    if not texts:
        print("no input; pass utterances or --file", file=sys.stderr)
        raise SystemExit(2)

    clf = SemanticIntentClassifier(args.model, args.vocab)
    results = clf.predict_batch(texts)

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
        return
    for r in results:
        gate = "" if r.accepted else "   [rejected -> fallback]"
        print(
            f"{r.text[:52]:54s} {r.routed_intent:26s} "
            f"conf={r.confidence:.2f} ood={r.ood_score:.2f}{gate}"
        )
        if args.top_k:
            for name, p in clf.top_k(r.text, args.top_k)[1:]:
                print(f"{'':54s}   {name:26s} {p:.2f}")


if __name__ == "__main__":
    main()
