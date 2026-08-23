#!/usr/bin/env python3
"""Tests for the encoder artifact contract.

Each test below corresponds to a defect that actually occurred in this
directory and was found only by measurement, never by a failing run. The point
of the contract is that these now fail loudly, so the tests assert that they
fail -- not that they are handled.

Self-contained: builds its own fixtures on a temp directory and needs neither
the real 9 MB artifacts nor onnxruntime. Two optional checks run against the
real exports when they are present.

    python3 test_artifact_contract.py        # standalone
    pytest test_artifact_contract.py         # or under pytest
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from artifact import (  # noqa: E402
    ArtifactContractError,
    EncoderArtifact,
    check_vocab_contract,
    encode,
    head_path,
    read_declared_pooling,
    resolve_tokenizer_dir,
)

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def make_artifact_dir(
    root: Path, vocab_size: int, tok_vocab: int, vocab_txt_lines=None, pooling: str | None = "cls"
) -> Path:
    """A directory shaped like a real export, with the three facts controllable."""
    d = root / "fake_export"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(
        json.dumps({"vocab_size": vocab_size, "hidden_size": 384}), encoding="utf-8"
    )
    (d / "tokenizer.json").write_text(
        json.dumps({"model": {"vocab": {f"t{i}": i for i in range(tok_vocab)}}}),
        encoding="utf-8",
    )
    if vocab_txt_lines is not None:
        (d / "vocab.txt").write_text(
            "\n".join(f"t{i}" for i in range(vocab_txt_lines)) + "\n", encoding="utf-8"
        )
    if pooling is not None:
        (d / "pooling.json").write_text(json.dumps({"pooling_mode": pooling}), encoding="utf-8")
    (d / "model_quantized.onnx").write_bytes(b"not a real graph")
    return d


class FakeTokenizer:
    """Emits the ids it is told to, so id-range handling can be tested directly."""

    def __init__(self, ids):
        self._ids = list(ids)

    def __call__(self, text, **kw):
        n = kw.get("max_length", 64)
        ids = (self._ids + [0] * n)[:n]
        mask = [1] * len(self._ids) + [0] * (n - len(self._ids))
        return {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.array([mask[:n]], dtype=np.int64),
        }


class FakeBackend:
    """Returns a fixed (seq, dim) block, matching OrtEncoder's squeezed shape."""

    def __init__(self, *_a, **_k):
        pass

    def embed_tokens(self, input_ids, attention_mask, token_type_ids):
        seq = input_ids.shape[1]
        out = np.zeros((seq, 384), dtype=np.float32)
        out[0, 0] = 1.0  # a distinctive CLS row
        out[1:, 1] = 1.0  # distinct from every other row
        return out


# summary() reports the artifact's size, so the fixture points at a file that
# really exists rather than at a path that only looks like one.
_FIXTURE_DIR = tempfile.mkdtemp(prefix="artifact_contract_")
_FIXTURE_MODEL = Path(_FIXTURE_DIR) / "fake_export" / "model_quantized.onnx"
_FIXTURE_MODEL.parent.mkdir(parents=True, exist_ok=True)
_FIXTURE_MODEL.write_bytes(b"not a real graph")


def fake_encoder(tok, pooling="cls", vocab_size=10000) -> EncoderArtifact:
    return EncoderArtifact(
        model_path=_FIXTURE_MODEL,
        tokenizer=tok,
        pooling=pooling,
        vocab_size=vocab_size,
        hidden_size=384,
        layers=3,
        source="fake_source_l3",
        backend=FakeBackend(),
    )


# --------------------------------------------------------------------------
# B1 -- a substituted tokenizer must not be silently accepted
# --------------------------------------------------------------------------
def test_missing_tokenizer_is_rejected_not_substituted():
    """The hub fallback is gone: no tokenizer beside the model is a hard error."""
    with tempfile.TemporaryDirectory() as tmp:
        d = make_artifact_dir(Path(tmp), 10000, 10000, 10000)
        (d / "tokenizer.json").unlink()
        try:
            resolve_tokenizer_dir(d / "model_quantized.onnx")
        except ArtifactContractError as exc:
            assert "tokenizer.json" in str(exc)
            return
        raise AssertionError("a model with no tokenizer beside it was accepted")


# --------------------------------------------------------------------------
# B10 -- build_pruned_l3.py rewrote vocab.txt but not tokenizer.json,
#        leaving a 30,522-entry tokenizer against a 10,000-row model
# --------------------------------------------------------------------------
def test_tokenizer_larger_than_embedding_matrix_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        d = make_artifact_dir(Path(tmp), vocab_size=10000, tok_vocab=30522, vocab_txt_lines=30522)
        try:
            check_vocab_contract(d)
        except ArtifactContractError as exc:
            assert "30522" in str(exc) and "10000" in str(exc)
            return
        raise AssertionError("a tokenizer larger than the embedding matrix was accepted")


# --------------------------------------------------------------------------
# B3 -- Python reads tokenizer.json and never notices a stale vocab.txt;
#       a native Swift/Kotlin wordpiece reads vocab.txt and indexes out of range
# --------------------------------------------------------------------------
def test_stale_vocab_txt_is_rejected_even_when_tokenizer_json_is_correct():
    with tempfile.TemporaryDirectory() as tmp:
        d = make_artifact_dir(Path(tmp), vocab_size=10000, tok_vocab=10000, vocab_txt_lines=30522)
        try:
            check_vocab_contract(d)
        except ArtifactContractError as exc:
            assert "vocab.txt" in str(exc)
            return
        raise AssertionError("a stale vocab.txt was accepted")


def test_consistent_artifact_passes():
    with tempfile.TemporaryDirectory() as tmp:
        d = make_artifact_dir(Path(tmp), vocab_size=10000, tok_vocab=10000, vocab_txt_lines=10000)
        vocab, hidden = check_vocab_contract(d)
        assert (vocab, hidden) == (10000, 384)
        assert read_declared_pooling(d) == "cls"


# --------------------------------------------------------------------------
# B2 -- pooling was hardcoded in three different scripts, three different ways
# --------------------------------------------------------------------------
def test_undeclared_pooling_is_rejected_rather_than_defaulted():
    """No default. A guess here scores a working encoder at 0.005."""
    with tempfile.TemporaryDirectory() as tmp:
        d = make_artifact_dir(Path(tmp), 10000, 10000, 10000, pooling=None)
        try:
            read_declared_pooling(d)
        except ArtifactContractError as exc:
            assert "pooling" in str(exc).lower()
            return
        raise AssertionError("an artifact with no declared pooling was given a default")


def test_invalid_pooling_value_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        d = make_artifact_dir(Path(tmp), 10000, 10000, 10000, pooling="max")
        try:
            read_declared_pooling(d)
        except ArtifactContractError as exc:
            assert "max" in str(exc)
            return
        raise AssertionError("an unsupported pooling_mode was accepted")


def test_cls_and_mean_actually_differ():
    """Guards against a pooling branch that silently does the same thing twice."""
    tok = FakeTokenizer([101, 5, 6, 102])
    cls_vec = encode(fake_encoder(tok, pooling="cls"), ["x"])[0]
    mean_vec = encode(fake_encoder(tok, pooling="mean"), ["x"])[0]
    assert not np.allclose(cls_vec, mean_vec), "cls and mean pooling returned the same vector"
    assert abs(np.linalg.norm(cls_vec) - 1.0) < 1e-5, "output is not L2-normalised"
    assert abs(np.linalg.norm(mean_vec) - 1.0) < 1e-5, "output is not L2-normalised"


def test_mean_pooling_ignores_padding():
    """Padding must not drag the mean. Different pad lengths, same sentence, same vector."""
    short = encode(fake_encoder(FakeTokenizer([101, 5, 102]), "mean"), ["x"], max_length=8)[0]
    long_ = encode(fake_encoder(FakeTokenizer([101, 5, 102]), "mean"), ["x"], max_length=64)[0]
    assert np.allclose(short, long_, atol=1e-6), "padding changed the pooled vector"


# --------------------------------------------------------------------------
# B4 -- the clamp turned a fatal mismatch into a silent 7% corruption
# --------------------------------------------------------------------------
def test_out_of_range_token_id_raises_instead_of_clamping():
    tok = FakeTokenizer([101, 20101, 102])  # 20101 is a real observed id
    try:
        encode(fake_encoder(tok, vocab_size=10000), ["please mute my hearing aids"])
    except ArtifactContractError as exc:
        assert "20101" in str(exc) and "10000" in str(exc)
        return
    raise AssertionError("an out-of-range token id was clamped instead of raising")


def test_summary_names_the_artifact_and_its_pooling():
    """A run must say WHICH encoder produced a number.

    Two exports in this directory are both 9.09 MB and differ only by pooling,
    so "loading the 9 MB model" left the reader to infer which one ran.
    """
    line = fake_encoder(FakeTokenizer([101, 5, 102]), pooling="cls").summary()
    assert line.split("|")[0].strip() == "fake_export", "summary must name the directory"
    assert "pooling cls" in line
    assert "3 layers" in line
    assert "fake_source_l3" in line, "summary should name the run it was distilled from"


def test_in_range_token_ids_are_accepted():
    tok = FakeTokenizer([101, 9999, 102])
    vec = encode(fake_encoder(tok, vocab_size=10000), ["x"])
    assert vec.shape == (1, 384)


# --------------------------------------------------------------------------
# B12 -- the head was written one level above the model, so two exports
#        sharing that parent silently overwrote each other's head
# --------------------------------------------------------------------------
def test_head_lives_beside_its_model():
    model = Path("output_models/some_export/model_quantized.onnx")
    assert head_path(model) == Path("output_models/some_export/classifier_head.pkl")


def test_two_exports_do_not_share_a_head_path():
    """The defect exactly: both of these used to resolve to the same file.

    final_distilled_onnx is mean-pooled and stage2_contrastive_bge_small_onnx
    is CLS-pooled. Whichever ran last won, and the evaluation then scored one
    encoder with the other one's head.
    """
    a = head_path(Path("output_models/final_distilled_onnx/model_quantized.onnx"))
    b = head_path(Path("output_models/stage2_contrastive_bge_small_onnx/model_quantized.onnx"))
    assert a != b, "two exports resolve to the same classifier head path"


def test_every_real_export_has_a_distinct_head_path():
    out = HERE / "output_models"
    if not out.exists():
        print("      (skipped: no exports on disk)")
        return
    models = sorted(out.glob("*/model_quantized.onnx")) + sorted(
        out.glob("*/onnx_quantized/model_quantized.onnx")
    )
    paths = [head_path(m) for m in models]
    assert len(paths) == len(set(paths)), f"head paths collide: {paths}"


# --------------------------------------------------------------------------
# optional: the real exports, when they are present
# --------------------------------------------------------------------------
def test_real_exports_satisfy_the_contract():
    """Skips when the artifacts are absent -- they are gitignored and regenerated."""
    out = HERE / "output_models"
    checked = 0
    for name in (
        "final_distilled_onnx",
        "stage2_contrastive_onnx",
        "stage2_contrastive_bge_small_onnx",
    ):
        d = out / name
        if not (d / "model_quantized.onnx").exists():
            continue
        check_vocab_contract(d)
        assert read_declared_pooling(d) in ("cls", "mean")
        checked += 1
    if checked == 0:
        print("      (skipped: no exports on disk)")


def test_a_half_present_artifact_reports_a_contract_error_not_an_io_error():
    """A missing config.json is a contract violation and must arrive as one.

    Every caller here is written as `except ArtifactContractError: skip it`. A
    bare FileNotFoundError walks straight through that and takes the caller down
    with it. This surfaced the first time these tests ran somewhere the exports
    were only partly on disk -- which is what CI looks like, and where they had
    never been run.
    """
    with tempfile.TemporaryDirectory() as tmp:
        d = make_artifact_dir(Path(tmp), 10000, 10000, 10000)
        (d / "config.json").unlink()
        try:
            check_vocab_contract(d)
        except ArtifactContractError as exc:
            assert "config.json" in str(exc)
            return
        except FileNotFoundError:
            raise AssertionError(
                "a missing config.json raised FileNotFoundError; callers catch "
                "ArtifactContractError and this one escapes them"
            ) from None
        raise AssertionError("a directory with no config.json was accepted")


def test_track1_is_known_broken_and_still_reported_as_such():
    """track1_pruned_l3 must keep failing until it is re-exported or retired.

    Pinned deliberately: if this ever starts passing, someone fixed or replaced
    the artifact, and the defect register in the plan needs updating.
    """
    d = HERE / "output_models" / "track1_pruned_l3" / "onnx_quantized"
    if not (d / "config.json").exists():
        print("      (skipped: track1_pruned_l3 not on disk)")
        return
    try:
        check_vocab_contract(d)
    except ArtifactContractError:
        return
    raise AssertionError(
        "track1_pruned_l3 now satisfies the contract -- update the defect register"
    )


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failed.append(name)
            print(f"  FAIL  {name}\n          {exc}")
        except Exception as exc:  # noqa: BLE001 - a test that errors is a failure
            failed.append(name)
            print(f"  ERROR {name}\n          {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
