"""Tests for the error-dependence spine (no mocks; real NTQR correlation)."""

from __future__ import annotations

import pytest

from ntqr_allotment.dependence import (
    CorrelationReport,
    measure_error_correlations,
    sample_votes_correlated,
)
from ntqr_allotment.experts import generate_population, sample_items


@pytest.fixture
def trio_and_items():
    pop = generate_population(12, seed=1, mean_expertise=0.75, expertise_heterogeneity=0.05)
    items = sample_items(600, prevalence_a=0.5, seed=2)
    return pop[:3], items


def test_sample_votes_correlated_deterministic(trio_and_items):
    trio, items = trio_and_items
    a = sample_votes_correlated(trio, items, rho=0.5, seed=7)
    b = sample_votes_correlated(trio, items, rho=0.5, seed=7)
    assert a == b
    assert all(set(v) <= {"a", "b"} for v in a)
    assert all(len(v) == len(items) for v in a)


def test_sample_votes_correlated_rho_validation(trio_and_items):
    trio, items = trio_and_items
    with pytest.raises(ValueError):
        sample_votes_correlated(trio, items, rho=1.5, seed=1)


def test_measure_error_correlations_trio_only(trio_and_items):
    _, items = trio_and_items
    with pytest.raises(ValueError):
        measure_error_correlations([("a",), ("b",)], items[:1])


def test_measure_returns_report(trio_and_items):
    trio, items = trio_and_items
    votes = sample_votes_correlated(trio, items, rho=0.0, seed=3)
    rep = measure_error_correlations(votes, items)
    assert isinstance(rep, CorrelationReport)
    assert len(rep.pair_correlations) == 6  # 3 pairs x 2 labels
    assert set(rep.three_way) == {"a", "b"}
    assert rep.mean_abs_pair >= 0.0


def test_correlation_rises_with_rho(trio_and_items):
    """The validated mechanism: measured error-correlation increases with rho."""
    trio, items = trio_and_items
    low = measure_error_correlations(
        sample_votes_correlated(trio, items, rho=0.0, seed=7), items
    ).mean_abs_pair
    high = measure_error_correlations(
        sample_votes_correlated(trio, items, rho=0.9, seed=7), items
    ).mean_abs_pair
    assert high > low  # fails if the correlated generator is reverted to independent
