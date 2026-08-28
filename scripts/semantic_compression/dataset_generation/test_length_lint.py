#!/usr/bin/env python3
"""Fixtures for the length gates. No API, no files, no network.

The point of these is the same as ``test_request_signal.py``'s: the gate was
wrong before and nobody noticed, because nothing ever asserted what it should
decide. Every expectation below is hand-checked arithmetic, not a recording of
what the code currently returns -- a snapshot test would have passed against the
fixed-fraction gate too.

    python3 -m pytest test_length_lint.py -q
"""

from __future__ import annotations

from length_lint import (
    ALPHA,
    DELIVERY_ALLOWANCE,
    MIN_ROW_EXCESS,
    assess,
    p_at_least,
    p_at_most,
    percentile,
)


def rec(target: float, cap: int = 4, deployed_rows: int = 100) -> dict:
    return {
        "short_max_words": cap,
        "target_short_share": target,
        "deployed_rows": deployed_rows,
    }


# --- the arithmetic ------------------------------------------------------


def test_binomial_tails_are_exact():
    # n=3, p=0.5 -> P(X<=1) = (1+3)/8 = 0.5 ; P(X>=2) = (3+1)/8 = 0.5
    assert abs(p_at_most(1, 3, 0.5) - 0.5) < 1e-12
    assert abs(p_at_least(2, 3, 0.5) - 0.5) < 1e-12
    # the two tails must partition the whole distribution at a shared boundary
    assert abs(p_at_most(1, 3, 0.5) + p_at_least(2, 3, 0.5) - 1.0) < 1e-12


def test_binomial_degenerate_inputs_do_not_fail_an_intent():
    # A rate of 0 or 1, or no rows, means the test cannot resolve anything. It
    # must return 1.0 so the caller passes, never 0.0 so the caller fails.
    for call in (p_at_most, p_at_least):
        assert call(0, 0, 0.5) == 1.0
        assert call(0, 10, 0.0) == 1.0
        assert call(0, 10, 1.0) == 1.0


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 0.0) == 1
    assert percentile([1, 2, 3, 4], 1.0) == 4
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([], 0.9) != percentile([], 0.9)  # nan


# --- the short gate ------------------------------------------------------


def test_short_gate_passes_an_intent_that_lands_on_the_allowance():
    # target 0.40, null 0.40*0.75 = 0.30, n=25 -> expected 7.5. Deliver 8.
    r = assess("X", [1] * 8 + [20] * 17, [5] * 100, rec(0.40))
    assert r["short_hits"] == 8
    assert abs(r["short_expected"] - 7.5) < 1e-9
    assert not r["short_bad"]


def test_short_gate_fails_a_real_shortfall():
    # target 0.72, null 0.54, n=25 -> expected 13.5. Deliver 6. This is
    # Help_FindMyHearingAids on the recorded pilot: p = 0.002.
    r = assess("X", [1] * 6 + [20] * 19, [5] * 100, rec(0.72, cap=7))
    assert r["short_bad"]
    assert r["short_p"] < ALPHA
    assert (r["short_expected"] - r["short_hits"]) >= MIN_ROW_EXCESS


def test_short_gate_will_not_fail_on_a_sub_three_row_discrepancy():
    # A significant p-value is not enough on its own: a two-row miss is a batch
    # size artefact. Two rows short of expectation must pass whatever p says.
    r = assess("X", [1] * 6 + [20] * 19, [5] * 100, rec(0.42, cap=4))
    assert (r["short_expected"] - r["short_hits"]) < MIN_ROW_EXCESS
    assert not r["short_bad"]


def test_short_gate_ignores_a_target_of_zero():
    r = assess("X", [20] * 25, [20] * 100, rec(0.0))
    assert not r["short_bad"]


def test_delivery_allowance_is_the_null_not_the_target():
    # The regression this file exists to prevent: testing against the raw ask.
    # At target 0.62 an intent delivering 0.48 is within the measured got/ask
    # curve and must pass; against the raw ask it would not.
    r = assess("X", [1] * 12 + [20] * 13, [5] * 100, rec(0.62, cap=4))
    assert r["short_share"] < r["target"]
    assert not r["short_bad"]
    assert abs(r["short_null"] - 0.62 * DELIVERY_ALLOWANCE) < 1e-12


# --- the long gate -------------------------------------------------------


def test_long_gate_fails_a_fat_tail_the_mean_rule_missed():
    # Deployed: 90 rows of 4 words, 10 of 6 -> p90 cut = 6.0, deployed rate
    # above the cut = 0. Guard against the degenerate-rate case by giving the
    # deployed set a real tail: 5 rows of 12.
    deployed = [4] * 85 + [6] * 10 + [12] * 5
    generated = [3] * 17 + [30] * 8  # 8 monsters behind a healthy short share
    r = assess("X", generated, deployed, rec(0.60, cap=4))
    assert r["long_hits"] == 8
    assert r["long_bad"]
    assert r["long_p"] < ALPHA
    # and the short share is fine, which is the whole point of a separate gate
    assert not r["short_bad"]


def test_long_gate_passes_a_tail_that_matches_deployed():
    deployed = [4] * 85 + [6] * 10 + [12] * 5
    generated = [3] * 23 + [30] * 2  # roughly the deployed tail rate
    r = assess("X", generated, deployed, rec(0.60, cap=4))
    assert not r["long_bad"]


def test_long_gate_is_silent_without_deployed_rows():
    # Nothing to calibrate against. Failing against an assumed rate is an
    # assumption wearing a measurement's clothes.
    r = assess("X", [30] * 25, [], rec(0.40))
    assert not r["calibrated"]
    assert not r["long_bad"]


# --- provenance ----------------------------------------------------------


def test_inferred_targets_are_flagged_not_trusted():
    r = assess("X", [1] * 25, [5] * 100, rec(0.40, deployed_rows=0))
    assert r["inferred"]
