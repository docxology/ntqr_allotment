"""Tests for the N-judge ensemble module (no mocks; real NTQR numerics only).

Negative controls (assertions that FLIP on a degenerate / broken input):

* ``test_moments_negative_control_identical_vs_independent`` -- an all-identical
  (maximally co-voting) panel MUST produce a strictly larger mean absolute pair
  frequency moment than an independent panel. A constant-output bug (always
  returning the same moment) would make these equal and FAIL the test.
* ``test_alarm_power_negative_control_safety`` -- the same panel/items must yield
  DIFFERENT alarm power under a tight safety spec ``(1, 1)`` (fires) versus a
  loose spec ``(3, 3)`` (does not). A constant-power bug would make them equal
  and FAIL.
"""

from __future__ import annotations

from itertools import product
import math

import pytest

from ntqr_allotment.dependence import sample_votes_correlated
from ntqr_allotment.ensemble import (
    ALARM_DEFAULT_MAX_Q,
    alarm_power_curve,
    observed_vote_counts,
    panel_agreement_moments,
)
from ntqr_allotment.experts import generate_population, sample_items, sample_votes
from ntqr_allotment.ntqr_eval import alarm_misaligned


def _panel_votes(n_judges: int, n_items: int, *, seed: int) -> list[tuple[str, ...]]:
    experts = generate_population(n_judges + 2, seed=seed)[:n_judges]
    items = sample_items(n_items, prevalence_a=0.5, seed=seed + 1)
    return [sample_votes(e, items, seed=seed + 10 + j) for j, e in enumerate(experts)]


def _panel_from_pattern_counts(
    counts: dict[tuple[str, ...], int],
) -> list[list[str]]:
    """Expand an item-pattern counts table into aligned per-judge votes."""
    return [
        [pattern[judge] for pattern, count in counts.items() for _ in range(count)]
        for judge in range(3)
    ]


def _trio_b_reference(
    votes: list[list[str]],
) -> tuple[list[float], dict[tuple[int, int], float], float, float]:
    """Compute the centered trio moment directly from raw aligned votes."""
    n_items = len(votes[0])
    marginals = [sum(vote == "b" for vote in row) / n_items for row in votes]
    co_frequencies = {
        (i, j): sum(
            votes[i][k] == "b" and votes[j][k] == "b" for k in range(n_items)
        )
        / n_items
        for i, j in ((0, 1), (0, 2), (1, 2))
    }
    pair_moments = {
        (i, j): frequency - marginals[i] * marginals[j]
        for (i, j), frequency in co_frequencies.items()
    }
    f123 = sum(
        all(votes[judge][k] == "b" for judge in range(3)) for k in range(n_items)
    ) / n_items
    centered = (
        f123
        - math.prod(marginals)
        - marginals[0] * pair_moments[(1, 2)]
        - marginals[1] * pair_moments[(0, 2)]
        - marginals[2] * pair_moments[(0, 1)]
    )
    return marginals, pair_moments, f123, centered


# --------------------------------------------------------------------------- #
# ObservedVoteCounts construction
# --------------------------------------------------------------------------- #
def test_observed_vote_counts_n4_counts_sum_to_q() -> None:
    votes = _panel_votes(4, 20, seed=11)
    ovc = observed_vote_counts(votes)
    assert ovc.ensemble_size == 4
    assert sum(ovc.vote_counts.values()) == 20
    # every pattern is a length-4 tuple over the binary label space
    for pattern in ovc.vote_counts:
        assert len(pattern) == 4
        assert set(pattern) <= {"a", "b"}


def test_observed_vote_counts_n5_counts_sum_to_q() -> None:
    votes = _panel_votes(5, 17, seed=22)
    ovc = observed_vote_counts(votes)
    assert ovc.ensemble_size == 5
    assert sum(ovc.vote_counts.values()) == 17


def test_observed_vote_counts_validates_min_judges() -> None:
    with pytest.raises(ValueError):
        observed_vote_counts([["a", "b", "a"]])  # only 1 judge


def test_observed_vote_counts_validates_no_items() -> None:
    with pytest.raises(ValueError):
        observed_vote_counts([[], []])


def test_observed_vote_counts_validates_ragged_rows() -> None:
    with pytest.raises(ValueError):
        observed_vote_counts([["a", "b"], ["a"]])


# --------------------------------------------------------------------------- #
# Moments are real, finite floats
# --------------------------------------------------------------------------- #
def test_panel_agreement_moments_finite_floats_n4() -> None:
    votes = _panel_votes(4, 20, seed=33)
    moments = panel_agreement_moments(votes)
    expected_keys = {
        "mean_abs_pair_moment_a",
        "mean_abs_pair_moment_b",
        "max_abs_pair_moment",
        "trio_frequency_moment_b",
    }
    assert expected_keys <= set(moments)
    for value in moments.values():
        assert isinstance(value, float)
        assert math.isfinite(value)
    assert moments["max_abs_pair_moment"] >= moments["mean_abs_pair_moment_a"] - 1e-12


def test_panel_agreement_moments_two_judges_trio_is_nan() -> None:
    # With only 2 judges there is no trio moment -> NaN (honest absence).
    votes = _panel_votes(2, 15, seed=44)
    moments = panel_agreement_moments(votes)
    assert math.isnan(moments["trio_frequency_moment_b"])
    assert math.isfinite(moments["mean_abs_pair_moment_a"])


def test_trio_moment_independent_uniform_table_is_zero() -> None:
    votes = _panel_from_pattern_counts(
        {pattern: 1 for pattern in product(("a", "b"), repeat=3)}
    )
    assert panel_agreement_moments(votes)["trio_frequency_moment_b"] == pytest.approx(
        0.0, abs=1e-15
    )


def test_trio_moment_uses_f123_with_fixed_marginals_and_pair_moments() -> None:
    # Adding a third-order interaction +/-1 to even/odd parity cells changes only
    # f123: every single-judge and pair marginal is unchanged. The nonuniform base
    # also keeps the pair moments nonzero, exercising every centered-formula term.
    patterns = tuple(product(("a", "b"), repeat=3))
    base_counts = (6, 3, 4, 3, 5, 3, 3, 5)
    low_counts: dict[tuple[str, ...], int] = {}
    high_counts: dict[tuple[str, ...], int] = {}
    for pattern, count in zip(patterns, base_counts):
        interaction = 1 if pattern.count("b") % 2 == 0 else -1
        low_counts[pattern] = count + interaction
        high_counts[pattern] = count - interaction

    low_votes = _panel_from_pattern_counts(low_counts)
    high_votes = _panel_from_pattern_counts(high_counts)
    low_marginals, low_pairs, low_f123, low_expected = _trio_b_reference(low_votes)
    high_marginals, high_pairs, high_f123, high_expected = _trio_b_reference(high_votes)

    assert low_marginals == pytest.approx(high_marginals)
    assert low_pairs == pytest.approx(high_pairs)
    assert any(moment != 0.0 for moment in low_pairs.values())
    assert high_f123 - low_f123 == pytest.approx(2 / 32)

    low_moment = panel_agreement_moments(low_votes)["trio_frequency_moment_b"]
    high_moment = panel_agreement_moments(high_votes)["trio_frequency_moment_b"]
    assert low_moment == pytest.approx(low_expected, abs=1e-15)
    assert high_moment == pytest.approx(high_expected, abs=1e-15)
    assert low_moment != high_moment


# --------------------------------------------------------------------------- #
# NEGATIVE CONTROL #1 — moments discriminate identical vs independent panels
# --------------------------------------------------------------------------- #
def test_moments_negative_control_identical_vs_independent() -> None:
    """An all-identical panel must out-agree an independent panel.

    Discriminator is REAL: identical votes maximize pair co-voting (moment
    -> f(1 - f)), while independent judges drive the centred moment toward 0.
    A constant-output bug would make these equal and fail the assertion.
    """
    shared = ["a", "b", "a", "a", "b", "a", "b", "b", "a", "b", "a", "a"]
    identical = [list(shared) for _ in range(4)]

    experts = generate_population(8, seed=55)[:4]
    items = sample_items(len(shared), prevalence_a=0.5, seed=56)
    independent = [sample_votes(e, items, seed=60 + j) for j, e in enumerate(experts)]

    m_identical = panel_agreement_moments(identical)
    m_independent = panel_agreement_moments(independent)

    # the flip: equality here would mean the moment is not actually computed
    assert m_identical["mean_abs_pair_moment_a"] != m_independent["mean_abs_pair_moment_a"]
    assert m_identical["mean_abs_pair_moment_a"] > m_independent["mean_abs_pair_moment_a"]


def test_moments_track_error_correlation_rho() -> None:
    """Higher induced error-correlation (rho) raises the agreement moments.

    Same experts, items, and seed; only rho differs. rho≈1 makes correctness
    co-move across judges -> larger pair moments than the rho=0 independent
    draw. Equality would indicate a constant-output bug.
    """
    experts = generate_population(6, seed=66)[:4]
    items = sample_items(14, prevalence_a=0.5, seed=67)
    correlated = sample_votes_correlated(experts, items, rho=0.99, seed=68)
    independent = sample_votes_correlated(experts, items, rho=0.0, seed=68)

    m_corr = panel_agreement_moments(correlated)
    m_indep = panel_agreement_moments(independent)
    assert m_corr["mean_abs_pair_moment_a"] > m_indep["mean_abs_pair_moment_a"]


# --------------------------------------------------------------------------- #
# Alarm power curve — range, determinism
# --------------------------------------------------------------------------- #
def test_alarm_power_curve_in_range_and_shape() -> None:
    experts = generate_population(10, seed=77, mean_expertise=0.95, expertise_heterogeneity=0.02)
    items = sample_items(12, prevalence_a=0.5, seed=78)
    curve = alarm_power_curve(experts, items, [2, 3, 4, 5], seeds=[1, 2, 3, 4], safety=(1, 1))
    assert [size for size, _ in curve] == [2, 3, 4, 5]
    for _, power in curve:
        assert 0.0 <= power <= 1.0
    # tight safety on a high-competence panel: the alarm actually fires
    assert any(power > 0.0 for _, power in curve)


def test_alarm_power_curve_is_reproducible() -> None:
    experts = generate_population(10, seed=88)
    items = sample_items(15, prevalence_a=0.5, seed=89)
    curve_a = alarm_power_curve(experts, items, [3, 4, 5], seeds=[1, 2, 3], safety=(1, 1))
    curve_b = alarm_power_curve(experts, items, [3, 4, 5], seeds=[1, 2, 3], safety=(1, 1))
    assert curve_a == curve_b


def test_alarm_power_curve_matches_original_per_size_reference() -> None:
    """Prefix reuse must exactly match the former size/seed/judge loop."""
    experts = generate_population(6, seed=90)
    items = sample_items(8, prevalence_a=0.5, seed=91)
    sizes = [4, 2, 4, 3, 2]
    seeds = [3, 1, 3]
    safety = (2, 2)

    expected: list[tuple[int, float]] = []
    for n in sizes:
        panel = experts[:n]
        fires = 0
        for seed in seeds:
            votes_by_judge = [
                sample_votes(e, items, seed=seed + 1009 * (j + 1))
                for j, e in enumerate(panel)
            ]
            if alarm_misaligned(
                votes_by_judge, safety_factors=safety, max_q=len(items)
            ):
                fires += 1
        expected.append((n, fires / len(seeds)))

    actual = alarm_power_curve(experts, items, sizes, seeds=seeds, safety=safety)
    assert actual == expected
    assert [n for n, _ in actual] == sizes


# --------------------------------------------------------------------------- #
# NEGATIVE CONTROL #2 — alarm power flips with the safety specification
# --------------------------------------------------------------------------- #
def test_alarm_power_negative_control_safety() -> None:
    """Same panel/items: tight safety fires, loose safety does not.

    The flip is real -- a tight ``(1, 1)`` spec admits no consistent key for a
    high-competence panel (power 1.0) while a loose ``(3, 3)`` spec does (power
    0.0). A constant-power bug would make these equal and fail the assertion.
    """
    experts = generate_population(10, seed=99, mean_expertise=0.95, expertise_heterogeneity=0.02)
    items = sample_items(12, prevalence_a=0.5, seed=100)
    sizes = [3, 4, 5]
    seeds = [1, 2, 3, 4, 5]

    tight = alarm_power_curve(experts, items, sizes, seeds=seeds, safety=(1, 1))
    loose = alarm_power_curve(experts, items, sizes, seeds=seeds, safety=(3, 3))

    assert tight != loose  # the flip
    assert all(p == 1.0 for _, p in tight)
    assert all(p == 0.0 for _, p in loose)


# --------------------------------------------------------------------------- #
# Small-Q guard and other validation
# --------------------------------------------------------------------------- #
def test_alarm_power_curve_small_q_guard_raises() -> None:
    experts = generate_population(8, seed=111)
    items = sample_items(ALARM_DEFAULT_MAX_Q + 5, prevalence_a=0.5, seed=112)
    with pytest.raises(ValueError, match="exceeds max_q"):
        alarm_power_curve(experts, items, [3], seeds=[1])


def test_alarm_power_curve_custom_max_q_allows_larger() -> None:
    experts = generate_population(8, seed=113)
    q = ALARM_DEFAULT_MAX_Q + 3
    items = sample_items(q, prevalence_a=0.5, seed=114)
    curve = alarm_power_curve(experts, items, [3], seeds=[1], safety=(1, 1), max_q=q)
    assert len(curve) == 1
    assert 0.0 <= curve[0][1] <= 1.0


def test_alarm_power_curve_empty_sizes_raises() -> None:
    experts = generate_population(8, seed=115)
    items = sample_items(10, prevalence_a=0.5, seed=116)
    with pytest.raises(ValueError, match="sizes must be non-empty"):
        alarm_power_curve(experts, items, [], seeds=[1])


def test_alarm_power_curve_empty_seeds_raises() -> None:
    experts = generate_population(8, seed=117)
    items = sample_items(10, prevalence_a=0.5, seed=118)
    with pytest.raises(ValueError, match="seeds must be non-empty"):
        alarm_power_curve(experts, items, [3], seeds=[])


def test_alarm_power_curve_empty_items_raises() -> None:
    experts = generate_population(8, seed=119)
    with pytest.raises(ValueError, match="items must be non-empty"):
        alarm_power_curve(experts, [], [3], seeds=[1])


def test_alarm_power_curve_size_too_small_raises() -> None:
    experts = generate_population(8, seed=120)
    items = sample_items(10, prevalence_a=0.5, seed=121)
    with pytest.raises(ValueError, match="at least 2"):
        alarm_power_curve(experts, items, [1], seeds=[1], safety=(1, 1))


def test_alarm_power_curve_size_exceeds_pool_raises() -> None:
    experts = generate_population(4, seed=122)
    items = sample_items(10, prevalence_a=0.5, seed=123)
    with pytest.raises(ValueError, match="exceeds expert pool"):
        alarm_power_curve(experts, items, [6], seeds=[1], safety=(1, 1))


def test_alarm_power_curve_bad_safety_length_raises() -> None:
    experts = generate_population(8, seed=124)
    items = sample_items(10, prevalence_a=0.5, seed=125)
    with pytest.raises(ValueError, match="label count"):
        alarm_power_curve(experts, items, [3], seeds=[1], safety=(2, 2, 2))
