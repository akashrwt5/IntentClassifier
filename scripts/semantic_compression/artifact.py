#!/usr/bin/env python3
"""The contract an exported encoder must satisfy before it is allowed to answer.

WHY THIS FILE EXISTS
--------------------
An exported encoder is not one file. It is a model plus three facts that must
travel with it, and if any of the three is wrong the model does not fail -- it
returns confident numbers computed from the wrong rows. Every consumer in this
directory used to decide all three by hand, and each of them got at least one
wrong:

  1. WHICH TOKENIZER.  ``train_experimental_head.py`` and
     ``evaluate_compression.py`` resolved it to ``<model>/../../pytorch``. That
     is correct only for the track1/track3 layout, where the model sits one
     level deeper inside ``onnx_quantized/``. Every other export overshoots onto
     a directory that does not exist, and the fallback then downloaded the full
     30,522-token tokenizer for a model whose embedding matrix has 10,000 rows.
     Measured cost when it fires: 7.17% of token occurrences corrupted, 46.6%
     of utterances touched, holdout accuracy 0.860 -> 0.761, concentrated on
     exactly the words that separate intents (mute, streaming, decrease,
     reminder, settings, translate).

     The lookup was never necessary. Every export in this repository already
     ships its tokenizer beside the model -- the flat ONNX directories, the
     ``track*/onnx_quantized/`` directories, and ``models/`` for the 22.9 MB
     baseline. The rule here is simply "beside the model", with no fallback.

  2. WHICH POOLING.  Pooling is a property of how the encoder was trained, not
     a runtime preference: bge-small is a CLS model, the MiniLM-derived
     students are mean models. ``train_experimental_head.py`` hardcoded CLS for
     everything but the baseline, ``test_distilled_holdout.py`` hardcoded CLS,
     ``evaluate_compression.py`` hardcoded mean. Reading the wrong token scores
     a working encoder at 0.005 -- below the 1/57 a coin flip would give --
     because every sentence collapses onto nearly the same vector.

  3. WHETHER THE IDS FIT.  ``input_ids[input_ids >= 10000] = 100`` turned a
     fatal artifact mismatch into a silent 7% corruption, and is the reason
     defect (1) survived unnoticed: nothing ever crashed. It was written for
     ``track1_pruned_l3``, whose tokenizer genuinely was never pruned, and then
     copied into scripts handling artifacts where it was dead code.

None of those decisions belongs in six copies. They are made here once, read
from what the artifact itself declares, and where an artifact declares nothing
this module refuses to guess and raises instead.

SELF-CONTAINED ON PURPOSE
-------------------------
This module imports nothing from the wider repository so that
``scripts/semantic_compression/`` can be lifted into a separate project
unchanged. onnxruntime and transformers are the only runtime dependencies, and
the contract checks below work without either.

USE AS A COMMAND
----------------
    python3 artifact.py output_models/*/model_quantized.onnx

prints the contract status of each artifact and exits non-zero if any fails.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VALID_POOLING = ("cls", "mean")
POOLING_FILE = "pooling.json"


class ArtifactContractError(RuntimeError):
    """An artifact does not describe itself, or describes itself inconsistently.

    Raised rather than worked around. An encoder whose tokenizer and embedding
    matrix disagree does not produce a slightly worse number; it produces a
    confident number computed from the wrong rows, and no downstream metric
    reveals it.
    """


class OrtEncoder:
    """Minimal ONNX Runtime adapter: token tensors in, token embeddings out.

    Behaviourally identical to ``packages.runtime.nlu_engine.inference
    .OrtEmbedderBackend`` -- ``outputs[0][0]``, i.e. the first output with the
    batch dimension squeezed, shape (seq_len, hidden). Reimplemented here only
    so this directory carries no dependency on the rest of the repository.
    """

    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        self.model_path = Path(model_path)
        self._session = ort.InferenceSession(str(model_path))

    def embed_tokens(self, input_ids, attention_mask, token_type_ids) -> np.ndarray:
        outputs = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        return outputs[0][0]


@dataclass(frozen=True)
class EncoderArtifact:
    """One exported encoder, with all three contract decisions already resolved."""

    model_path: Path
    tokenizer: object
    pooling: str
    vocab_size: int
    hidden_size: int
    layers: int
    source: str
    backend: object

    @property
    def name(self) -> str:
        return _label(self.model_path.parent)

    @property
    def size_mb(self) -> float:
        return self.model_path.stat().st_size / (1024 * 1024)

    def summary(self) -> str:
        """The one line every consumer prints before it reports a number.

        A run that says only "loading the 9 MB model" leaves the reader to
        infer which encoder produced the result -- and the two 9.09 MB exports
        in this directory differ only by their pooling. Naming the artifact and
        the sentence-transformers run it came from removes the guess.
        """
        line = (
            f"{self.name} | {self.size_mb:.2f} MB | {self.layers} layers | "
            f"hidden {self.hidden_size} | vocab {self.vocab_size} | pooling {self.pooling}"
        )
        return f"{line}\n  distilled from: {self.source}" if self.source else line


def _label(model_dir: Path) -> str:
    """A directory name that identifies the artifact in an error message.

    Several exports live in directories called ``onnx_quantized``, so the leaf
    name alone does not say which artifact failed. Where the leaf is that
    generic, the parent is included.
    """
    model_dir = Path(model_dir)
    if model_dir.name in ("onnx_quantized", "pytorch", "temp_pt"):
        return f"{model_dir.parent.name}/{model_dir.name}"
    return model_dir.name


def _read_json(path: Path) -> dict:
    """Read a JSON file, or say which part of the contract is missing.

    A file the contract requires and cannot find is a contract violation, not an
    I/O accident, and it has to arrive as one: a bare FileNotFoundError escapes
    every ``except ArtifactContractError`` in this directory, so a half-present
    artifact crashes a caller that was written to skip it. Found by running these
    tests somewhere the exports are only partly on disk -- the case they will
    meet in CI, and the one they had never been run in.
    """
    path = Path(path)
    if not path.is_file():
        raise ArtifactContractError(
            f"{_label(path.parent)}: {path.name} is missing. An artifact is a model "
            f"plus the files that describe it; without {path.name} nothing here can "
            f"say what this encoder is."
        )
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise ArtifactContractError(
            f"{_label(path.parent)}: {path.name} is not valid JSON ({exc})."
        ) from exc


def resolve_tokenizer_dir(model_path: Path) -> Path:
    """The directory holding this model's tokenizer: the model's own directory.

    No search, no fallback, no hub download. A model shipped without its
    tokenizer is an incomplete artifact, and substituting another one is how
    the 30,522-vs-10,000 mismatch entered the pipeline. An on-device pipeline
    must never reach the network to find a tokenizer.
    """
    model_path = Path(model_path)
    tokenizer_dir = model_path.parent
    if not (tokenizer_dir / "tokenizer.json").exists():
        raise ArtifactContractError(
            f"{model_path} has no tokenizer.json beside it (looked in {tokenizer_dir}). "
            "Export the tokenizer with the model. Substituting another tokenizer is "
            "what produced the 0.860 -> 0.761 regression this module exists to prevent."
        )
    return tokenizer_dir


def head_path(model_path: Path) -> Path:
    """Where this model's classifier head lives: beside the model.

    Same rule as the tokenizer and the pooling declaration -- everything an
    artifact needs to answer a question sits in the artifact's own directory.

    The head used to be written to ``<model>/../classifier_head.pkl``, one
    level up. Two exports share that parent, so ``final_distilled_onnx`` (mean
    pooled) and ``stage2_contrastive_bge_small_onnx`` (CLS pooled) wrote to the
    same file and silently overwrote each other. Whichever ran last in the
    MODELS dict won, and the evaluation script then scored one encoder with the
    other encoder's head -- producing a plausible number from a mismatched
    pair, which is the failure this module exists to make impossible.
    """
    return Path(model_path).parent / "classifier_head.pkl"


def read_declared_pooling(model_dir: Path) -> str:
    """Pooling as the artifact declares it. Deliberately has no default.

    A missing declaration means the export did not record how the encoder was
    trained, and guessing is precisely the failure this module prevents: CLS on
    a mean-trained encoder is not a small error, it is a dead model that still
    returns numbers.

    ``pooling.json`` mirrors the ``pooling_mode`` key sentence-transformers
    already writes in ``1_Pooling/config.json``, so the export step copies an
    existing convention instead of inventing a second one.
    """
    model_dir = Path(model_dir)
    pooling_file = model_dir / POOLING_FILE
    if not pooling_file.exists():
        raise ArtifactContractError(
            f"{model_dir} does not declare its pooling (expected {pooling_file}). "
            "Copy pooling_mode from the source sentence-transformers "
            "1_Pooling/config.json, or run backfill_artifact_metadata.py. "
            "Refusing to guess: reading the wrong token scores a working encoder "
            "at 0.005."
        )
    mode = _read_json(pooling_file).get("pooling_mode")
    if mode not in VALID_POOLING:
        raise ArtifactContractError(
            f"{pooling_file} declares pooling_mode={mode!r}; expected one of {VALID_POOLING}."
        )
    return mode


def check_vocab_contract(model_dir: Path) -> tuple[int, int]:
    """Assert the tokenizer and the embedding matrix agree on vocabulary size.

    This is the single check that would have caught the ``track1_pruned_l3``
    defect the day it was created. ``build_pruned_l3.py`` rewrote ``vocab.txt``
    to the pruned 10,000 tokens and then reloaded the tokenizer from that
    directory -- where the fast tokenizer read ``tokenizer.json``, still the
    full 30,522. The pruning silently did not happen, and the clamp downstream
    hid it. Every number ever produced from that artifact had ~7% of its input
    replaced by [UNK].

    Returns (vocab_size, hidden_size) as the model config declares them.
    """
    model_dir = Path(model_dir)
    config = _read_json(model_dir / "config.json")
    model_vocab = int(config["vocab_size"])
    hidden = int(config["hidden_size"])

    tok_vocab = len(_read_json(model_dir / "tokenizer.json")["model"]["vocab"])
    if tok_vocab != model_vocab:
        raise ArtifactContractError(
            f"{_label(model_dir)}: tokenizer.json has {tok_vocab} entries but the model's "
            f"embedding matrix has {model_vocab} rows. This artifact cannot produce "
            "correct embeddings. Re-export it; do not clamp the ids."
        )

    vocab_txt = model_dir / "vocab.txt"
    if vocab_txt.exists():
        lines = vocab_txt.read_text(encoding="utf-8").split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
        if len(lines) != model_vocab:
            raise ArtifactContractError(
                f"{_label(model_dir)}: vocab.txt has {len(lines)} lines but the model has "
                f"{model_vocab} embedding rows. Python reads tokenizer.json and would not "
                "notice, but a native Swift or Kotlin wordpiece reads vocab.txt and will "
                "index out of range on device. Run backfill_artifact_metadata.py."
            )
    return model_vocab, hidden


def load_encoder(model_path: Path, backend_factory=OrtEncoder) -> EncoderArtifact:
    """Load an exported encoder with every contract check applied first.

    There is no pooling override and no default. Every encoder this directory
    loads is one it exported, and it declares its pooling in ``pooling.json``.

    ``backend_factory`` is injectable so the contract checks stay testable
    without an ONNX Runtime session.
    """
    from transformers import AutoTokenizer

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model_dir = resolve_tokenizer_dir(model_path)
    vocab_size, hidden = check_vocab_contract(model_dir)
    pooling = read_declared_pooling(model_dir)

    config = _read_json(model_dir / "config.json")
    source = _read_json(model_dir / POOLING_FILE).get("_source", "")

    return EncoderArtifact(
        model_path=model_path,
        tokenizer=AutoTokenizer.from_pretrained(str(model_dir)),
        pooling=pooling,
        vocab_size=vocab_size,
        hidden_size=hidden,
        layers=int(config.get("num_hidden_layers", 0)),
        source=source.split("/")[0] if source else "",
        backend=backend_factory(model_path),
    )


def encode(artifact: EncoderArtifact, texts, max_length: int = 64) -> np.ndarray:
    """Embed texts through the artifact, pooled as the artifact declares.

    Returns L2-normalised sentence vectors, shape (len(texts), hidden_size).

    Token ids are asserted into range rather than clamped. A clamp converts an
    incompatible tokenizer into a model that still answers, with roughly 7% of
    its input silently replaced by [UNK].
    """
    if isinstance(texts, str):
        texts = [texts]

    out = np.zeros((len(texts), artifact.hidden_size), dtype=np.float32)
    for i, text in enumerate(texts):
        encoded = artifact.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )
        input_ids = encoded["input_ids"]
        max_id = int(input_ids.max())
        if max_id >= artifact.vocab_size:
            raise ArtifactContractError(
                f"{artifact.name}: token id {max_id} is outside the model's "
                f"{artifact.vocab_size}-row embedding matrix, on input {text!r}. "
                "The tokenizer and the model do not belong together."
            )

        attention_mask = encoded["attention_mask"]
        token_type_ids = encoded.get("token_type_ids", np.zeros_like(input_ids))
        token_embeddings = artifact.backend.embed_tokens(input_ids, attention_mask, token_type_ids)

        if artifact.pooling == "cls":
            vec = token_embeddings[0]
        else:
            mask = attention_mask[0][:, np.newaxis].astype(np.float32)
            vec = (token_embeddings * mask).sum(axis=0) / np.clip(mask.sum(), 1e-9, None)

        norm = float(np.linalg.norm(vec))
        out[i] = vec / norm if norm > 0 else vec
    return out


def describe(model_path: Path) -> str:
    """One line stating what an artifact declares. Used by reports and by CI."""
    model_path = Path(model_path)
    model_dir = model_path.parent
    vocab, hidden = check_vocab_contract(model_dir)
    pooling = read_declared_pooling(model_dir)
    layers = _read_json(model_dir / "config.json").get("num_hidden_layers", "?")
    size_mb = model_path.stat().st_size / (1024 * 1024)
    return (
        f"{_label(model_dir)}: {size_mb:.2f} MB | {layers} layers | vocab {vocab} "
        f"| hidden {hidden} | pooling {pooling}"
    )


def main(argv: list[str]) -> int:
    """Validate every artifact named on the command line."""
    if not argv:
        print(__doc__.strip().splitlines()[-1])
        print("\nusage: python3 artifact.py <model.onnx> [<model.onnx> ...]")
        return 2

    failures = 0
    for raw in argv:
        path = Path(raw)
        try:
            print(f"PASS  {describe(path)}")
        except (ArtifactContractError, FileNotFoundError, KeyError) as exc:
            failures += 1
            print(f"FAIL  {_label(path.parent)}")
            for line in str(exc).split(". "):
                if line.strip():
                    print(f"        {line.strip().rstrip('.')}.")
    print(f"\n{len(argv) - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
