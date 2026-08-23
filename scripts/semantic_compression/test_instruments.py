#!/usr/bin/env python3
"""Tests for the instrument layer: normaliser parity, exactness, power, guard.

Each test here exists because something in this project was once believed
without being checked:

  * the vendored normaliser is a COPY of the repository's, and a copy drifts;
  * prefix filtering is an optimisation, and an optimisation that drops a pair
    silently inflates dev_hard;
  * the plan's power table was hand-computed, and hand-computed tables go stale
    the moment the code that should agree with them is written;
  * a guard nobody has seen fail is not known to work.

    python3 test_instruments.py        # standalone
    pytest test_instruments.py         # or under pytest
"""

from __future__ import annotations

import csv
import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_instruments  # noqa: E402
import split_dev_sets  # noqa: E402
from instruments import (  # noqa: E402
    minimum_detectable_effect,
    near_duplicate_flags,
    normalize_text,
    token_set,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


# --------------------------------------------------------------------------
# the vendored normaliser must not drift from the repository's
# --------------------------------------------------------------------------
def test_vendored_normaliser_matches_the_repository():
    """Skips when lifted out of this repo -- which is the point of vendoring."""
    src = REPO / "packages" / "buildtime"
    if not (src / "nlu_training" / "leakage.py").exists():
        print("      (skipped: repository normaliser not present)")
        return
    sys.path.insert(0, str(src))
    from nlu_training.leakage import normalize_text as reference  # noqa: PLC0415

    cases = [
        "Don't  turn UP the volume, please?",
        "what's  my battery   level",
        "volume,please",
        "TURN\tit\ndown",
        "café où",  # accents must survive
        "ıstanbul",  # dotless i, casefold edge
        "ﬁnd my hearing aids",  # NFKC ligature
        "mute my aids",  # non-breaking space
        "don’t / dont / don't",  # three apostrophe forms
        "",
        "   ",
        "100% volume!!!",
    ]
    for case in cases:
        assert normalize_text(case) == reference(case), (
            f"vendored normaliser drifted on {case!r}: "
            f"{normalize_text(case)!r} != {reference(case)!r}"
        )


# --------------------------------------------------------------------------
# prefix filtering must be exact, not approximate
# --------------------------------------------------------------------------
def test_prefix_filter_finds_every_pair_brute_force_finds():
    rng = random.Random(20260822)
    words = [f"w{i}" for i in range(40)]
    train = [" ".join(rng.sample(words, rng.randint(2, 9))) for _ in range(400)]
    probe = [" ".join(rng.sample(words, rng.randint(2, 9))) for _ in range(200)]
    # plus near-copies of real training rows, which is what the filter must catch
    probe += [t + " please" for t in train[:40]]
    probe += [" ".join(t.split()[::-1]) for t in train[40:80]]

    train_sets = [token_set(t) for t in train]
    for threshold in (0.7, 0.8, 0.9, 1.0):
        fast = near_duplicate_flags(probe, train, threshold)
        for i, text in enumerate(probe):
            s = token_set(text)
            brute = any(len(s & ts) / len(s | ts) >= threshold for ts in train_sets if (s | ts))
            assert brute == (
                fast[i] is not None
            ), f"threshold {threshold}: prefix filter and brute force disagree on {text!r}"


def test_near_duplicate_returns_the_matched_training_text():
    train = ["turn up the volume please", "what is my battery level"]
    got = near_duplicate_flags(["please turn up the volume"], train, 0.8)
    assert got[0] == "turn up the volume please", got


def test_word_order_does_not_defeat_the_check():
    """Token-set Jaccard is order-blind on purpose: a reordering is a paraphrase."""
    assert near_duplicate_flags(["volume the up turn"], ["turn up the volume"], 0.8)[0]


def test_punctuation_and_case_do_not_defeat_the_check():
    assert near_duplicate_flags(["Turn Up The Volume!!"], ["turn up the volume"], 0.8)[0]


def test_a_genuinely_different_sentence_is_not_flagged():
    assert (
        near_duplicate_flags(
            ["remind me to call my daughter tomorrow"], ["turn up the volume"], 0.8
        )[0]
        is None
    )


# --------------------------------------------------------------------------
# the plan's power table must be reproducible from the code
# --------------------------------------------------------------------------
PLAN_POWER_TABLE = {
    (46, 0.10): 0.131,
    (46, 0.15): 0.160,
    (46, 0.25): 0.207,
    (300, 0.10): 0.051,
    (300, 0.15): 0.063,
    (300, 0.25): 0.081,
    (500, 0.10): 0.040,
    (500, 0.15): 0.049,
    (500, 0.25): 0.063,
    (813, 0.10): 0.031,
    (813, 0.15): 0.038,
    (813, 0.25): 0.049,
    (1000, 0.10): 0.028,
    (1000, 0.15): 0.034,
    (1000, 0.25): 0.044,
    (1470, 0.10): 0.023,
    (1470, 0.15): 0.028,
    (1470, 0.25): 0.037,
    (3000, 0.10): 0.016,
    (3000, 0.15): 0.020,
    (3000, 0.25): 0.026,
}


def test_power_table_in_the_plan_matches_the_code():
    """§10 of semantic_compression_plan.md. If this fails, one of the two is wrong."""
    for (n, disc), expected in PLAN_POWER_TABLE.items():
        got = round(minimum_detectable_effect(n, disc), 3)
        assert got == expected, f"n={n} discordance={disc}: plan says {expected}, code says {got}"


def test_more_rows_detect_smaller_effects():
    prev = float("inf")
    for n in (46, 300, 813, 1470, 3000):
        mde = minimum_detectable_effect(n, 0.15)
        assert mde < prev, "MDE must shrink as rows grow"
        prev = mde


# --------------------------------------------------------------------------
# the guard must actually fail -- an unfired guard is not a guard
# --------------------------------------------------------------------------
def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["text", "intent"])
        w.writeheader()
        w.writerows(rows)


def _fixture(tmp: Path, extra_train=()):
    """A miniature pack: train, holdout, and the derived split."""
    pack = tmp / "language_packs" / "en"
    train = [
        {"text": f"turn up the volume number {i}", "intent": "Cmd.VolumeIncrease"} for i in range(6)
    ]
    train += [
        {"text": f"remind me to call person {i}", "intent": "Cmd.SetReminder"} for i in range(6)
    ]
    holdout = [
        {"text": "turn up the volume number 3 please", "intent": "Cmd.VolumeIncrease"},
        {"text": "what is the weather in oslo tomorrow", "intent": "Default Fallback Intent"},
        {"text": "book me a table for four at seven", "intent": "Default Fallback Intent"},
    ]
    _write_csv(pack / "train.csv", list(train) + list(extra_train))
    _write_csv(pack / "holdout_honest.csv", holdout)
    (pack / "extras").mkdir(parents=True, exist_ok=True)
    return pack


def _point_modules_at(pack: Path):
    split_dev_sets.PACK = pack
    split_dev_sets.TRAIN_CSV = pack / "train.csv"
    split_dev_sets.HOLDOUT_CSV = pack / "holdout_honest.csv"
    check_instruments.PACK = pack
    check_instruments.TRAIN_CSV = pack / "train.csv"
    check_instruments.HOLDOUT_CSV = pack / "holdout_honest.csv"
    check_instruments.HOLDOUT_MANIFEST = pack / "extras" / "holdout_honest.manifest.json"
    check_instruments.DEV_HARD = pack / "dev_hard.csv"
    check_instruments.DEV_NEAR = pack / "dev_near.csv"


def _restore(saved):
    for module, attrs in saved.items():
        for k, v in attrs.items():
            setattr(module, k, v)


def _snapshot():
    return {
        split_dev_sets: {
            k: getattr(split_dev_sets, k) for k in ("PACK", "TRAIN_CSV", "HOLDOUT_CSV")
        },
        check_instruments: {
            k: getattr(check_instruments, k)
            for k in (
                "PACK",
                "TRAIN_CSV",
                "HOLDOUT_CSV",
                "HOLDOUT_MANIFEST",
                "DEV_HARD",
                "DEV_NEAR",
            )
        },
    }


def test_contamination_guard_fires_when_training_data_swallows_a_dev_hard_row():
    """The Super Dataset scenario, in miniature."""
    saved = _snapshot()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            pack = _fixture(Path(tmp))
            _point_modules_at(pack)
            split_dev_sets.build(0.8, pack)

            hard = list(csv.DictReader((pack / "dev_hard.csv").open(encoding="utf-8")))
            assert hard, "fixture produced an empty dev_hard"

            clean = check_instruments.Failure()
            check_instruments.check_dev_hard_contamination(clean)
            assert not clean, f"guard fired on clean data: {clean}"

            # a generated row that paraphrases a dev_hard row enters TRAINING
            victim = hard[0]["text"]
            rows = list(csv.DictReader((pack / "train.csv").open(encoding="utf-8")))
            rows.append({"text": victim + " today", "intent": hard[0]["intent"]})
            _write_csv(pack / "train.csv", rows)

            dirty = check_instruments.Failure()
            check_instruments.check_dev_hard_contamination(dirty)
            assert dirty, "guard did NOT fire after a dev_hard row was paraphrased into train"
            assert "contaminated" in dirty[0]
            assert victim in dirty[0], "the guard must name the affected dev_hard row"
    finally:
        _restore(saved)


def test_partition_guard_fires_when_a_row_goes_missing():
    saved = _snapshot()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            pack = _fixture(Path(tmp))
            _point_modules_at(pack)
            split_dev_sets.build(0.8, pack)

            hard = list(csv.DictReader((pack / "dev_hard.csv").open(encoding="utf-8")))
            _write_csv(pack / "dev_hard.csv", hard[:-1])

            problems = check_instruments.Failure()
            check_instruments.check_partition(problems)
            assert problems, "a dropped row did not fail the partition check"
    finally:
        _restore(saved)


def test_manifest_freshness_guard_fires_on_an_edit_that_keeps_the_row_count():
    """B9's actual shape: content changed, row count identical."""
    saved = _snapshot()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            pack = _fixture(Path(tmp))
            _point_modules_at(pack)
            manifest = pack / "extras" / "holdout_honest.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "rows": {"train": 12, "holdout": 3},
                        "sha256": {"train.csv": "0" * 64, "holdout_honest.csv": "0" * 64},
                    }
                ),
                encoding="utf-8",
            )
            problems = check_instruments.Failure()
            check_instruments.check_manifest_freshness(problems)
            assert len(problems) == 2, problems
    finally:
        _restore(saved)


# --------------------------------------------------------------------------
# what the derived files look like on disk
# --------------------------------------------------------------------------
def test_derived_csvs_use_lf_endings_like_their_source():
    """csv.writer defaults to CRLF; the pack is LF.

    A CRLF file is invisible in a diff and in an editor, so this surfaced only
    when the repository's mixed-line-ending hook rewrote both derived sets
    mid-commit -- which changed their sha256 after they had been pinned, leaving
    dev_split.manifest.json describing files that no longer existed.
    """
    saved = _snapshot()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            pack = _fixture(Path(tmp))
            _point_modules_at(pack)
            split_dev_sets.build(0.8, pack)
            for name in ("dev_near.csv", "dev_hard.csv"):
                raw = (pack / name).read_bytes()
                assert b"\r" not in raw, f"{name} was written with CR in it"
                assert raw.endswith(b"\n") and not raw.endswith(
                    b"\n\n"
                ), f"{name} must end with exactly one newline"
    finally:
        _restore(saved)


def test_generated_markdown_ends_with_exactly_one_newline():
    """The end-of-file-fixer hook rewrites anything else, after generation."""
    path = HERE / "INSTRUMENTS.md"
    if not path.exists():
        print("      (skipped: INSTRUMENTS.md not generated yet)")
        return
    raw = path.read_bytes()
    assert b"\r" not in raw, "INSTRUMENTS.md contains CR"
    assert raw.endswith(b"\n") and not raw.endswith(
        b"\n\n"
    ), "INSTRUMENTS.md must end with exactly one newline"


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
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"  ERROR {name}\n          {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
