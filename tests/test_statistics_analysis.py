"""Tests for pure-numeric inferential statistics helpers.

Every expected value is hand-computed in the test, not produced by the function
under test. Each behavioural claim is paired with a negative control whose
assertion flips on a degenerate or broken input.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from ntqr_allotment.statistics_analysis import (
    MeanCIResult,
    OLSResult,
    SeparationResult,
    bootstrap_ci,
    bootstrap_slope_ci,
    ci_overlap_verdict,
    cohens_d,
    exact_sign_test,
    mean_ci_summary,
    ols_slope,
    strategy_separation,
)


# --------------------------------------------------------------------------- #
# bootstrap_ci
# --------------------------------------------------------------------------- #
def test_bootstrap_ci_of_constant_is_zero_width():
    # Every resample mean of a constant array equals that constant exactly.
    lo, hi = bootstrap_ci([3.5, 3.5, 3.5, 3.5], n_boot=200, seed=0)
    assert lo == 3.5
    assert hi == 3.5


def test_bootstrap_ci_negative_control_nonconstant_has_positive_width():
    # NEGATIVE CONTROL: a non-constant sample must NOT give a zero-width CI,
    # so the "zero width" assertion above flips here.
    lo, hi = bootstrap_ci([0.0, 10.0, 20.0, 30.0], n_boot=500, seed=1)
    assert hi > lo


def test_bootstrap_ci_brackets_true_mean():
    # Known sample with mean exactly 5.0; the percentile CI should contain it.
    sample = [2.0, 4.0, 5.0, 6.0, 8.0]
    true_mean = 5.0
    lo, hi = bootstrap_ci(sample, n_boot=2000, seed=7)
    assert lo <= true_mean <= hi


def test_bootstrap_ci_is_reproducible():
    sample = [1.0, 2.0, 9.0, 4.0, 7.0]
    first = bootstrap_ci(sample, n_boot=1000, seed=42)
    second = bootstrap_ci(sample, n_boot=1000, seed=42)
    assert first == second
    # NEGATIVE CONTROL: a different seed must change the interval.
    other = bootstrap_ci(sample, n_boot=1000, seed=43)
    assert other != first


def test_bootstrap_ci_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        bootstrap_ci([], seed=0)


def test_bootstrap_ci_rejects_bad_n_boot():
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_ci([1.0, 2.0], n_boot=0, seed=0)


def test_bootstrap_ci_rejects_bad_alpha():
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_ci([1.0, 2.0], alpha=1.5, seed=0)


def test_ci_overlap_verdict_separated_pair():
    assert ci_overlap_verdict((0.0, 1.0), (2.0, 3.0)) == "separated"


def test_ci_overlap_verdict_overlapping_pair():
    assert ci_overlap_verdict((0.0, 2.0), (1.0, 3.0)) == "overlapping"


def test_ci_overlap_verdict_touching_boundary_is_overlapping():
    assert ci_overlap_verdict((0.0, 1.0), (1.0, 2.0)) == "overlapping"


def test_ci_overlap_verdict_is_order_independent():
    separated = ((0.0, 1.0), (4.0, 5.0))
    assert ci_overlap_verdict(*separated) == "separated"
    assert ci_overlap_verdict(*separated[::-1]) == "separated"

    overlapping = ((0.0, 3.0), (2.0, 4.0))
    assert ci_overlap_verdict(*overlapping) == "overlapping"
    assert ci_overlap_verdict(*overlapping[::-1]) == "overlapping"


def test_ci_overlap_verdict_rejects_wrong_length_interval():
    with pytest.raises(ValueError, match="2-tuple"):
        ci_overlap_verdict((0.0, 1.0, 2.0), (3.0, 4.0))


def test_ci_overlap_verdict_rejects_non_tuple_interval():
    with pytest.raises(ValueError, match="2-tuple"):
        ci_overlap_verdict(cast(tuple[float, float], [0.0, 1.0]), (3.0, 4.0))


def test_ci_overlap_verdict_rejects_inverted_interval():
    with pytest.raises(ValueError, match="lo <= hi"):
        ci_overlap_verdict((2.0, 1.0), (3.0, 4.0))


def test_strategy_separation_of_clearly_separated_groups():
    group_a = [0.10, 0.12, 0.11, 0.09, 0.13]
    group_b = [0.88, 0.91, 0.89, 0.92, 0.90]

    result = strategy_separation(group_a, group_b, n_boot=500, alpha=0.05, seed=17)

    assert isinstance(result, SeparationResult)
    assert result.mean_a == pytest.approx(0.11)
    assert result.mean_b == pytest.approx(0.90)
    assert result.verdict == "separated"
    assert result.signed_difference == pytest.approx(-0.79)


def test_strategy_separation_of_overlapping_groups():
    group_a = [0.49, 0.51, 0.50, 0.52, 0.48, 0.50]
    group_b = [0.50, 0.52, 0.49, 0.51, 0.48, 0.50]

    result = strategy_separation(group_a, group_b, n_boot=500, alpha=0.05, seed=4)

    assert result.verdict == "overlapping"


def test_strategy_separation_is_deterministic_for_fixed_seed():
    group_a = [0.12, 0.15, 0.11, 0.14, 0.13]
    group_b = [0.44, 0.46, 0.43, 0.47, 0.45]

    first = strategy_separation(group_a, group_b, n_boot=400, alpha=0.1, seed=21)
    second = strategy_separation(group_a, group_b, n_boot=400, alpha=0.1, seed=21)

    assert first == second


def test_strategy_separation_negative_control_gap_vanishes_flips_verdict():
    separated_a = [0.10, 0.11, 0.12, 0.13, 0.14]
    separated_b = [0.86, 0.87, 0.88, 0.89, 0.90]
    separated = strategy_separation(
        separated_a, separated_b, n_boot=400, alpha=0.05, seed=8
    )
    assert separated.verdict == "separated"

    combined = sorted(separated_a + separated_b)
    interleaved_a = combined[::2]
    interleaved_b = combined[1::2]
    overlapping = strategy_separation(
        interleaved_a, interleaved_b, n_boot=400, alpha=0.05, seed=8
    )
    assert overlapping.verdict == "overlapping"


def test_strategy_separation_reuses_bootstrap_ci_seed_convention():
    group_a = [0.20, 0.21, 0.22, 0.23, 0.24]
    group_b = [0.60, 0.61, 0.62, 0.63, 0.64]

    result = strategy_separation(group_a, group_b, n_boot=600, alpha=0.1, seed=31)

    assert result.ci_a == bootstrap_ci(group_a, n_boot=600, alpha=0.1, seed=31)
    assert result.ci_b == bootstrap_ci(group_b, n_boot=600, alpha=0.1, seed=32)


def test_strategy_separation_propagates_empty_group_error():
    with pytest.raises(ValueError, match="non-empty"):
        strategy_separation([], [0.1, 0.2], seed=5)


def test_mean_ci_summary_constant_sample_is_descriptive_point_interval() -> None:
    result = mean_ci_summary([2.0, 2.0, 2.0], n_boot=200, seed=0)

    assert isinstance(result, MeanCIResult)
    assert result.n == 3
    assert result.mean == 2.0
    assert result.std == 0.0
    assert result.ci_low == 2.0
    assert result.ci_high == 2.0


def test_mean_ci_summary_matches_bootstrap_ci_for_nonconstant_sample() -> None:
    values = [0.0, 1.0, 3.0, 7.0]

    result = mean_ci_summary(values, n_boot=400, seed=9)

    assert result.n == 4
    assert result.mean == pytest.approx(2.75)
    assert result.std > 0.0
    assert (result.ci_low, result.ci_high) == bootstrap_ci(
        values, n_boot=400, seed=9
    )


def test_mean_ci_summary_rejects_empty_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        mean_ci_summary([], seed=0)
    with pytest.raises(ValueError, match="finite"):
        mean_ci_summary([1.0, float("nan")], seed=0)


# --------------------------------------------------------------------------- #
# cohens_d
# --------------------------------------------------------------------------- #
def test_cohens_d_identical_groups_is_zero():
    # NEGATIVE CONTROL: identical groups => zero numerator => exactly 0.0.
    assert cohens_d([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_cohens_d_known_separated_pair():
    # a = [1,2,3]: mean 2, var(ddof=1)=1.0 ; b=[4,5,6]: mean 5, var=1.0
    # pooled_var = (2*1 + 2*1)/(3+3-2) = 1.0 ; d = (2 - 5)/sqrt(1) = -3.0
    result = cohens_d([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert abs(result - (-3.0)) < 1e-9
    # Swapping arguments flips the sign (sanity / direction check).
    assert abs(cohens_d([4.0, 5.0, 6.0], [1.0, 2.0, 3.0]) - 3.0) < 1e-9


def test_cohens_d_zero_pooled_variance_returns_zero():
    # Both groups constant but at different levels: pooled variance is 0.
    # The chosen contract returns 0.0 (no standardized spread to measure).
    assert cohens_d([5.0, 5.0, 5.0], [9.0, 9.0, 9.0]) == 0.0


def test_cohens_d_rejects_small_groups():
    with pytest.raises(ValueError, match="at least two"):
        cohens_d([1.0], [4.0, 5.0])
    with pytest.raises(ValueError, match="at least two"):
        cohens_d([1.0, 2.0], [4.0])


# --------------------------------------------------------------------------- #
# ols_slope
# --------------------------------------------------------------------------- #
def test_ols_slope_exact_line():
    # Points exactly on y = 2x + 1.
    xs = [0.0, 1.0, 2.0, 3.0, 4.0]
    ys = [1.0, 3.0, 5.0, 7.0, 9.0]
    result = ols_slope(xs, ys)
    assert isinstance(result, OLSResult)
    assert abs(result.slope - 2.0) < 1e-9
    assert abs(result.intercept - 1.0) < 1e-9
    assert abs(result.r_squared - 1.0) < 1e-9


def test_ols_slope_flat_line_is_zero_slope():
    # NEGATIVE CONTROL: a flat dependent variable => slope 0, NOT 2.
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [7.0, 7.0, 7.0, 7.0]
    result = ols_slope(xs, ys)
    assert abs(result.slope - 0.0) < 1e-9
    assert abs(result.intercept - 7.0) < 1e-9
    # ys is constant: the flat fit explains it exactly.
    assert result.r_squared == 1.0


def test_ols_slope_partial_fit_r_squared_between_zero_and_one():
    # y has scatter around the line, so 0 < R^2 < 1 (not the exact-fit case).
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [0.0, 1.0, 2.0, 10.0]
    result = ols_slope(xs, ys)
    assert 0.0 < result.r_squared < 1.0


def test_ols_slope_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        ols_slope([0.0, 1.0], [1.0])


def test_ols_slope_rejects_too_few_points():
    with pytest.raises(ValueError, match="at least two"):
        ols_slope([1.0], [2.0])


def test_ols_slope_rejects_zero_variance_xs():
    # NEGATIVE CONTROL: constant xs cannot define a slope.
    with pytest.raises(ValueError, match="positive variance"):
        ols_slope([3.0, 3.0, 3.0], [1.0, 2.0, 3.0])


# --------------------------------------------------------------------------- #
# bootstrap_slope_ci
# --------------------------------------------------------------------------- #
def test_bootstrap_slope_ci_brackets_ols_slope():
    rng = np.random.default_rng(123)
    xs = np.linspace(0.0, 10.0, 40)
    ys = 1.5 * xs + 2.0 + rng.normal(0.0, 0.5, size=xs.size)
    point_slope = ols_slope(xs, ys).slope
    lo, hi = bootstrap_slope_ci(xs, ys, n_boot=1000, seed=5)
    assert lo <= point_slope <= hi


def test_bootstrap_slope_ci_is_reproducible():
    xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [0.1, 2.2, 3.9, 6.1, 8.0, 9.8]
    first = bootstrap_slope_ci(xs, ys, n_boot=800, seed=11)
    second = bootstrap_slope_ci(xs, ys, n_boot=800, seed=11)
    assert first == second
    # NEGATIVE CONTROL: a different seed changes the interval.
    other = bootstrap_slope_ci(xs, ys, n_boot=800, seed=12)
    assert other != first


def test_bootstrap_slope_ci_skips_degenerate_resamples():
    # With only two distinct x values, some resamples draw the same x twice
    # (zero-variance => skipped). The non-degenerate resamples still yield a CI.
    xs = [0.0, 1.0]
    ys = [0.0, 4.0]
    lo, hi = bootstrap_slope_ci(xs, ys, n_boot=200, seed=3)
    # Every non-degenerate resample uses both points, slope = (4-0)/(1-0) = 4.
    assert abs(lo - 4.0) < 1e-9
    assert abs(hi - 4.0) < 1e-9


def test_bootstrap_slope_ci_all_degenerate_raises():
    # NEGATIVE CONTROL: n_boot=1 with a 2-point sample can still draw a valid
    # resample, so instead force degeneracy by validating the guard directly:
    # a constant-x base fit is rejected before bootstrapping.
    with pytest.raises(ValueError, match="positive variance"):
        bootstrap_slope_ci([2.0, 2.0], [1.0, 5.0], n_boot=10, seed=0)


def test_bootstrap_slope_ci_rejects_bad_n_boot():
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_slope_ci([0.0, 1.0], [0.0, 1.0], n_boot=0, seed=0)


def test_bootstrap_slope_ci_rejects_bad_alpha():
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_slope_ci([0.0, 1.0], [0.0, 1.0], alpha=0.0, seed=0)


def test_bootstrap_slope_ci_all_resamples_degenerate_branch():
    # Force the "all degenerate" branch: a sample where every drawn pair of
    # indices for n_boot resamples collapses x-variance is hard with replacement,
    # so we construct a 2-point set and request a single resample that is the
    # degenerate draw [0,0]. seed chosen so the first (only) row is [0,0].
    xs = [0.0, 1.0]
    ys = [0.0, 1.0]
    # Search a seed whose single resample is degenerate to exercise the raise.
    degenerate_seed = None
    for s in range(50):
        idx = np.random.default_rng(s).integers(0, 2, size=(1, 2))
        if idx[0, 0] == idx[0, 1]:
            degenerate_seed = s
            break
    assert degenerate_seed is not None
    with pytest.raises(ValueError, match="degenerate"):
        bootstrap_slope_ci(xs, ys, n_boot=1, seed=degenerate_seed)


def test_holm_bonferroni_no_rejections_when_all_large() -> None:
    from ntqr_allotment.statistics_analysis import holm_bonferroni

    res = holm_bonferroni([0.4, 0.6, 0.9, 0.5])
    assert res.n_rejected == 0
    assert all(p >= 0.05 for p in res.adjusted)


def test_holm_bonferroni_rejects_only_strong_after_correction() -> None:
    from ntqr_allotment.statistics_analysis import holm_bonferroni

    # One tiny p among 4: raw 0.001 -> adjusted 0.001*4 = 0.004 < 0.05 (rejected);
    # the 0.04 raw becomes 0.04*3 = 0.12 (NOT rejected after correction).
    res = holm_bonferroni([0.001, 0.04, 0.30, 0.60])
    assert res.rejected[0] is True
    assert res.rejected[1] is False
    assert res.n_rejected == 1
    assert res.adjusted[0] == pytest.approx(0.004, abs=1e-9)


def test_holm_bonferroni_is_monotone_nondecreasing_in_sorted_order() -> None:
    from ntqr_allotment.statistics_analysis import holm_bonferroni

    res = holm_bonferroni([0.01, 0.02, 0.03])
    ordered = sorted(res.adjusted)
    assert list(res.adjusted) == ordered  # already ascending input stays ascending
    assert all(a <= b for a, b in zip(ordered, ordered[1:]))


def test_holm_bonferroni_rejects_bad_input() -> None:
    from ntqr_allotment.statistics_analysis import holm_bonferroni

    with pytest.raises(ValueError):
        holm_bonferroni([])
    with pytest.raises(ValueError):
        holm_bonferroni([0.1, 1.5])  # p out of [0,1]
    with pytest.raises(ValueError):
        holm_bonferroni([0.1], alpha=0.0)


def test_exact_sign_test_all_four_negative_is_small_n_descriptive() -> None:
    result = exact_sign_test([-0.07, -0.049, -0.045, -0.042])

    assert result.negative == 4
    assert result.positive == 0
    assert result.zero == 0
    assert result.n_nonzero == 4
    assert result.p_value == pytest.approx(0.125)


def test_exact_sign_test_ignores_zero_deltas() -> None:
    result = exact_sign_test([-1.0, 0.0, 1.0, 0.0])

    assert result.negative == 1
    assert result.positive == 1
    assert result.zero == 2
    assert result.n_nonzero == 2
    assert result.p_value == pytest.approx(1.0)


def test_exact_sign_test_all_zero_is_uninformative() -> None:
    result = exact_sign_test([0.0, 0.0, 0.0])

    assert result.n_nonzero == 0
    assert result.p_value == 1.0
