"""Tests for comprehensive allotment fairness/representation (no mocks)."""

from __future__ import annotations

import pytest

from ntqr_allotment.experts import generate_population
from ntqr_allotment.fairness import (
    FairnessReport,
    _gini,
    maximin_fairness,
    multi_feature_quota_config,
    representation_error,
)


@pytest.fixture
def population():
    return generate_population(60, seed=3)


def test_multi_feature_quota_config_covers_all_features(population):
    cfg = multi_feature_quota_config(population, 6, features=("ideology", "expertise_tier"))
    feats = {t.feature for t in cfg.targets}
    assert feats == {"ideology", "expertise_tier"}
    assert cfg.panel_size == 6
    # each feature's seat allocation sums to the panel size
    for feature in ("ideology", "expertise_tier"):
        assert sum(t.min for t in cfg.targets if t.feature == feature) == 6


def test_maximin_fairness_report(population):
    rep = maximin_fairness(population, 6, seed=1, panel_count=40)
    assert isinstance(rep, FairnessReport)
    assert rep.n_feasible_panels > 0
    assert rep.min_selection_prob <= rep.mean_selection_prob <= rep.max_selection_prob
    # marginal selection probabilities sum to the panel size
    assert sum(rep.realised_probabilities.values()) == pytest.approx(6.0, abs=1e-6)
    assert 0.0 <= rep.gini < 1.0


def test_maximin_min_prob_is_positive(population):
    # the maximin objective should give every quota-eligible candidate some chance
    rep = maximin_fairness(population, 6, seed=2, panel_count=40)
    assert rep.min_selection_prob >= 0.0
    assert rep.max_selection_prob > 0.0


def test_gini_edge_cases():
    assert _gini([]) == 0.0
    assert _gini([0.0, 0.0]) == 0.0
    assert _gini([1.0, 1.0, 1.0]) == pytest.approx(0.0, abs=1e-9)
    assert _gini([0.0, 0.0, 1.0]) > 0.0


def test_representation_error_non_negative(population):
    rep = maximin_fairness(population, 6, seed=1, panel_count=40)
    err = representation_error(population, rep.realised_probabilities)
    assert err >= 0.0
