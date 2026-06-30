"""Tests for the statistical power-analysis module.

Discipline: no primitive is checked against itself. The normal helpers are pinned to
*published* constants; analytic power is pinned to a textbook G*Power value and to its
own Type-I corner (power at d=0 equals alpha); the analytic formula is cross-checked
against a Monte-Carlo simulation that runs an entirely different algorithm (the actual
permutation test). Impostor mutants -- a power function ignoring n or d, a permutation
test ignoring the labels -- are explicitly killed.
"""

from __future__ import annotations


import pytest

from ntqr_allotment.power_analysis import (
    NullDiagnosis,
    analytic_power,
    cohens_d_safe,
    diagnose_null,
    min_detectable_effect,
    norm_cdf,
    norm_ppf,
    permutation_test,
    power_curve_over_effect,
    power_curve_over_n,
    sample_size_for_power,
    simulate_power,
)

# --------------------------------------------------------------------------------------
# Normal helpers vs PUBLISHED constants (independent reference, not self-referential).
# --------------------------------------------------------------------------------------


def test_norm_cdf_matches_published_values() -> None:
    assert float(norm_cdf(0.0)) == pytest.approx(0.5, abs=1e-12)
    assert float(norm_cdf(1.0)) == pytest.approx(0.8413447461, abs=1e-7)
    assert float(norm_cdf(1.959963985)) == pytest.approx(0.975, abs=1e-7)
    assert float(norm_cdf(-1.959963985)) == pytest.approx(0.025, abs=1e-7)
    assert float(norm_cdf(2.575829304)) == pytest.approx(0.995, abs=1e-7)


def test_norm_cdf_vectorized() -> None:
    out = norm_cdf([-1.0, 0.0, 1.0])
    assert out.shape == (3,)
    assert out[1] == pytest.approx(0.5, abs=1e-12)
    # Monotone increasing.
    assert out[0] < out[1] < out[2]


def test_norm_ppf_matches_published_critical_values() -> None:
    assert norm_ppf(0.975) == pytest.approx(1.959963985, abs=1e-4)
    assert norm_ppf(0.95) == pytest.approx(1.644853627, abs=1e-4)
    assert norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
    assert norm_ppf(0.8) == pytest.approx(0.841621234, abs=1e-4)
    assert norm_ppf(0.025) == pytest.approx(-1.959963985, abs=1e-4)


def test_norm_ppf_is_inverse_of_cdf_in_tails() -> None:
    # Round-trip consistency across central and tail regions (Acklam break-points).
    for p in (0.001, 0.02, 0.3, 0.7, 0.98, 0.999):
        assert float(norm_cdf(norm_ppf(p))) == pytest.approx(p, abs=1e-6)


def test_norm_ppf_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        norm_ppf(0.0)
    with pytest.raises(ValueError):
        norm_ppf(1.0)
    with pytest.raises(ValueError):
        norm_ppf(-0.1)


# --------------------------------------------------------------------------------------
# Analytic power: negative-control corner, monotonicity, textbook value, impostor kill.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
def test_analytic_power_at_zero_effect_equals_alpha(alpha: float) -> None:
    # The Type-I corner: with no true effect, a level-alpha test rejects at rate alpha.
    assert analytic_power(0.0, 50, alpha=alpha) == pytest.approx(alpha, abs=1e-9)


def test_analytic_power_monotone_in_n_and_effect() -> None:
    powers_n = [analytic_power(0.5, n) for n in (10, 30, 60, 120, 500)]
    assert powers_n == sorted(powers_n)
    assert powers_n[-1] > 0.99  # n -> large drives power to 1
    powers_d = [analytic_power(d, 40) for d in (0.0, 0.2, 0.5, 0.8, 1.5)]
    assert powers_d == sorted(powers_d)


def test_analytic_power_matches_textbook_gpower_value() -> None:
    # Two-sample t, d=0.5, n=64/group, alpha=0.05 two-sided -> power ~ 0.80 (G*Power).
    assert analytic_power(0.5, 64, alpha=0.05, two_sided=True) == pytest.approx(0.80, abs=0.02)


def test_analytic_power_one_sided_exceeds_two_sided() -> None:
    one = analytic_power(0.5, 40, two_sided=False)
    two = analytic_power(0.5, 40, two_sided=True)
    assert one > two


def test_analytic_power_ignoring_effect_would_fail() -> None:
    # Impostor guard: a constant/effect-ignoring implementation cannot pass this.
    assert analytic_power(1.2, 40) - analytic_power(0.1, 40) > 0.3


def test_analytic_power_unbalanced_groups() -> None:
    balanced = analytic_power(0.5, 40, 40)
    unbalanced = analytic_power(0.5, 10, 200)
    # The harmonic mean of sizes drives power; a 10-vs-200 split is weaker than 40-40.
    assert unbalanced < balanced


def test_analytic_power_rejects_bad_args() -> None:
    with pytest.raises(ValueError):
        analytic_power(0.5, 0)
    with pytest.raises(ValueError):
        analytic_power(0.5, 40, alpha=0.0)


# --------------------------------------------------------------------------------------
# Permutation test: validity floor, separation, calibration, determinism, impostor kill.
# --------------------------------------------------------------------------------------


def test_permutation_test_separated_groups_small_p() -> None:
    a = [0.0, 0.1, -0.1, 0.05, -0.05]
    b = [5.0, 5.1, 4.9, 5.05, 4.95]
    p = permutation_test(a, b, n_perm=2000, seed=0)
    assert p < 0.01


def test_permutation_test_identical_groups_not_extreme() -> None:
    a = [1.0, 2.0, 3.0, 4.0]
    b = [1.0, 2.0, 3.0, 4.0]
    p = permutation_test(a, b, n_perm=500, seed=1)
    assert p > 0.5


def test_permutation_test_add_one_validity_floor() -> None:
    a = [0.0, 0.0, 0.0]
    b = [9.0, 9.0, 9.0]
    p = permutation_test(a, b, n_perm=100, seed=2)
    assert p >= 1.0 / (1.0 + 100)  # never zero -> exact-valid


def test_permutation_test_is_deterministic() -> None:
    a = [0.2, 0.4, 0.1, 0.9]
    b = [1.2, 0.8, 1.1, 0.7]
    assert permutation_test(a, b, n_perm=300, seed=7) == permutation_test(
        a, b, n_perm=300, seed=7
    )


def test_permutation_test_calibrated_type_one_rate() -> None:
    # Draw many same-distribution pairs; the share of p<0.05 must sit near 0.05.
    import numpy as np

    rng = np.random.default_rng(123)
    hits = 0
    trials = 200
    for _ in range(trials):
        a = rng.normal(0.0, 1.0, size=8)
        b = rng.normal(0.0, 1.0, size=8)
        if permutation_test(a, b, n_perm=200, seed=int(rng.integers(0, 1_000_000))) < 0.05:
            hits += 1
    rate = hits / trials
    assert rate < 0.15  # a valid test does not over-reject under the null


def test_permutation_test_one_sided_direction() -> None:
    a = [0.0, 0.1, -0.1]
    b = [3.0, 3.1, 2.9]
    # A minus B is strongly negative -> "less" is significant, "greater" is not.
    assert permutation_test(a, b, n_perm=1000, seed=3, alternative="less") < 0.2
    assert permutation_test(a, b, n_perm=1000, seed=3, alternative="greater") > 0.8


def test_permutation_test_rejects_bad_args() -> None:
    with pytest.raises(ValueError):
        permutation_test([], [1.0], n_perm=10, seed=0)
    with pytest.raises(ValueError):
        permutation_test([1.0], [2.0], n_perm=0, seed=0)


# --------------------------------------------------------------------------------------
# simulate_power: Type-I negative control, monotonicity, and the cross-vendor-style
# INDEPENDENT cross-check that analytic == simulated (two different algorithms agree).
# --------------------------------------------------------------------------------------


def test_simulate_power_type_one_rate_near_alpha() -> None:
    # No true effect -> empirical power must collapse to ~alpha (the Type-I rate).
    rate = simulate_power(0.0, 12, alpha=0.05, n_sim=400, n_perm=200, seed=11)
    assert rate < 0.12


def test_simulate_power_rises_with_effect() -> None:
    low = simulate_power(0.0, 20, alpha=0.05, n_sim=300, n_perm=200, seed=5)
    high = simulate_power(1.2, 20, alpha=0.05, n_sim=300, n_perm=200, seed=5)
    assert high - low > 0.4


def test_simulated_power_agrees_with_analytic() -> None:
    # Independent reference: closed-form normal power vs permutation Monte-Carlo.
    # At a moderate effect / size the normal approximation is good and the two routes
    # must land within Monte-Carlo tolerance of each other.
    effect, n = 0.9, 30
    ana = analytic_power(effect, n, alpha=0.05)
    sim = simulate_power(effect, n, alpha=0.05, n_sim=500, n_perm=300, seed=21)
    assert sim == pytest.approx(ana, abs=0.08)


def test_simulate_power_one_sided_path() -> None:
    # Exercises the one-sided ("greater") permutation branch under mixed outcomes;
    # a one-sided test has at least as much power as the two-sided test for a true
    # positive effect.
    one = simulate_power(0.6, 25, alpha=0.05, n_sim=200, n_perm=150, seed=31, two_sided=False)
    two = simulate_power(0.6, 25, alpha=0.05, n_sim=200, n_perm=150, seed=31, two_sided=True)
    assert 0.0 <= two <= 1.0 and 0.0 <= one <= 1.0
    # A correctly-directed one-sided test has real (non-zero) power for a true effect.
    assert one > 0.3


def test_simulate_power_one_sided_negative_effect() -> None:
    # A negative effect must select the opposite tail and still recover real power
    # (regression guard for the one-sided direction bug).
    p = simulate_power(-0.7, 25, alpha=0.05, n_sim=200, n_perm=150, seed=32, two_sided=False)
    assert p > 0.3


def test_simulate_power_rejects_bad_args() -> None:
    with pytest.raises(ValueError):
        simulate_power(0.5, 0, seed=0)
    with pytest.raises(ValueError):
        simulate_power(0.5, 10, n_sim=0, seed=0)
    with pytest.raises(ValueError):
        simulate_power(0.5, 10, sd=0.0, seed=0)


# --------------------------------------------------------------------------------------
# Sample-size solver and MDE: textbook value, minimality, monotonicity, round-trip.
# --------------------------------------------------------------------------------------


def test_sample_size_matches_textbook_and_is_minimal() -> None:
    n = sample_size_for_power(0.5, power=0.8, alpha=0.05)
    assert 60 <= n <= 68  # G*Power: 64/group for d=0.5
    # Minimality: one fewer subject drops below the target power.
    assert analytic_power(0.5, n, alpha=0.05) >= 0.8
    assert analytic_power(0.5, n - 1, alpha=0.05) < 0.8


def test_sample_size_monotone_decreasing_in_effect() -> None:
    sizes = [sample_size_for_power(d, power=0.8) for d in (0.2, 0.5, 0.8, 1.2)]
    assert sizes == sorted(sizes, reverse=True)


def test_sample_size_rejects_zero_effect_and_bad_args() -> None:
    with pytest.raises(ValueError):
        sample_size_for_power(0.0)
    with pytest.raises(ValueError):
        sample_size_for_power(0.5, power=1.0)
    with pytest.raises(ValueError):
        sample_size_for_power(0.0001, power=0.8, n_max=10)


def test_min_detectable_effect_round_trips_with_power() -> None:
    n = 50
    mde = min_detectable_effect(n, power=0.8, alpha=0.05)
    # An effect exactly at the MDE is detected at ~the target power.
    assert analytic_power(mde, n, alpha=0.05) == pytest.approx(0.8, abs=0.02)


def test_min_detectable_effect_monotone_decreasing_in_n() -> None:
    mdes = [min_detectable_effect(n) for n in (5, 20, 80, 320)]
    assert mdes == sorted(mdes, reverse=True)
    # Strictly decreasing (not merely non-increasing): more samples resolve a
    # smaller effect. A real ordering claim, not f(x) == f(x).
    assert all(a > b for a, b in zip(mdes, mdes[1:]))


def test_min_detectable_effect_rejects_bad_args() -> None:
    with pytest.raises(ValueError):
        min_detectable_effect(0)
    with pytest.raises(ValueError):
        min_detectable_effect(10, power=0.0)


# --------------------------------------------------------------------------------------
# Power curves.
# --------------------------------------------------------------------------------------


def test_power_curve_over_n_is_monotone() -> None:
    curve = power_curve_over_n(0.5, [10, 50, 200])
    assert [n for n, _ in curve] == [10, 50, 200]
    powers = [p for _, p in curve]
    assert powers == sorted(powers)


def test_power_curve_over_effect_is_monotone() -> None:
    curve = power_curve_over_effect([0.0, 0.4, 0.9], 40)
    powers = [p for _, p in curve]
    assert powers == sorted(powers)
    assert powers[0] == pytest.approx(0.05, abs=1e-9)


# --------------------------------------------------------------------------------------
# diagnose_null: honest MDE framing (NO retrospective observed power) -- ISC-79 guard.
# --------------------------------------------------------------------------------------


def test_diagnose_null_flags_underpowered_small_design() -> None:
    # Tiny design (n=3 seeds) with a small observed effect: cannot detect its own point.
    diag = diagnose_null(0.3, 3, target_power=0.8, alpha=0.05)
    assert isinstance(diag, NullDiagnosis)
    assert diag.underpowered is True
    assert diag.mde > 0.3
    # The honest budget answer: many more seeds would be needed.
    assert diag.seeds_for_target is not None and diag.seeds_for_target > 3


def test_diagnose_null_well_powered_large_design() -> None:
    diag = diagnose_null(1.0, 100, target_power=0.8, alpha=0.05)
    assert diag.underpowered is False
    assert diag.mde < 1.0


def test_diagnose_null_zero_effect_has_no_target_size() -> None:
    diag = diagnose_null(0.0, 10)
    assert diag.seeds_for_target is None
    assert diag.underpowered is True  # |0| < mde always


def test_diagnose_null_does_not_expose_retrospective_observed_power() -> None:
    # ISC-79: the honesty contract forbids reporting "power to detect the observed
    # effect". The result type must not carry such a field under any spelling.
    diag = diagnose_null(0.4, 8)
    fields = set(diag._fields)
    for forbidden in ("observed_power", "post_hoc_power", "retrospective_power", "achieved_power"):
        assert forbidden not in fields


# --------------------------------------------------------------------------------------
# cohens_d_safe: degenerate-safe standardized mean difference. Covers the empty-group
# guard, the denom<=0 (single-element) corner, the zero-pooled-variance corner, and a
# hand-computed value against an INDEPENDENT reference (not the function under test).
# --------------------------------------------------------------------------------------


def test_cohens_d_safe_rejects_empty_group() -> None:
    # Lines 165-166: an empty group has no defined mean -> ValueError, not 0.0.
    with pytest.raises(ValueError, match="non-empty"):
        cohens_d_safe([], [1.0, 2.0])
    with pytest.raises(ValueError, match="non-empty"):
        cohens_d_safe([1.0, 2.0], [])


def test_cohens_d_safe_single_element_groups_return_zero() -> None:
    # Lines 172-173: with one point per group, denom = 1 + 1 - 2 = 0 (no pooled
    # variance is defined) -> the degenerate-safe contract returns 0.0 rather than
    # dividing by zero.
    assert cohens_d_safe([4.0], [9.0]) == 0.0
    # NEGATIVE CONTROL: a non-degenerate separated pair must NOT be flattened to 0.0.
    assert cohens_d_safe([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) != 0.0


def test_cohens_d_safe_zero_pooled_variance_returns_zero() -> None:
    # Lines 175-176: both groups constant (at different levels) -> within-group
    # variances are 0 -> pooled_var <= 0 -> 0.0 (no standardized spread to express).
    assert cohens_d_safe([5.0, 5.0, 5.0], [9.0, 9.0, 9.0]) == 0.0


def test_cohens_d_safe_known_separated_pair() -> None:
    # Line 177 (the real computation), checked against a hand-derived reference:
    # a = [1,2,3]: mean 2, sample var (ddof=1) = 1.0 ; b = [4,5,6]: mean 5, var 1.0.
    # pooled_var = ((3-1)*1.0 + (3-1)*1.0) / (3+3-2) = 1.0 ; s_pooled = 1.0.
    # d = (mean_a - mean_b) / s_pooled = (2 - 5) / 1.0 = -3.0.
    result = cohens_d_safe([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert result == pytest.approx(-3.0, abs=1e-9)
    # Swapping the arguments flips the sign (direction check / independent corner).
    assert cohens_d_safe([4.0, 5.0, 6.0], [1.0, 2.0, 3.0]) == pytest.approx(3.0, abs=1e-9)


def test_diagnose_null_returns_none_budget_for_unpowerable_tiny_effect() -> None:
    """A near-zero (nonzero) effect needs a budget beyond n_max -> None, not a raise.

    Surfaced by the 96-seed run: ``sample_size_for_power`` raises when even n_max
    per group cannot reach target power. ``diagnose_null`` must absorb that into the
    same "no practical budget" sentinel (None) it already uses for a zero effect,
    rather than crashing the power table. NEGATIVE CONTROL: a healthy moderate
    effect still returns a finite integer budget.
    """
    from ntqr_allotment.power_analysis import diagnose_null

    tiny = diagnose_null(observed_effect=1e-6, n_per_group=96)
    assert tiny.seeds_for_target is None
    assert tiny.underpowered is True  # 1e-6 is below any reasonable MDE

    healthy = diagnose_null(observed_effect=0.8, n_per_group=96)
    assert isinstance(healthy.seeds_for_target, int)
    assert healthy.seeds_for_target > 0
