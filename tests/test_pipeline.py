"""Tests for the end-to-end trial pipeline (no mocks)."""

from __future__ import annotations

import pytest

from ntqr_allotment.experts import generate_population, sample_items
from ntqr_allotment.pipeline import TrialConfig, TrialResult, run_trial, votes_for
from ntqr_allotment.sortition import STRATEGIES


def test_votes_for_returns_aligned_votes():
    pop = generate_population(10, seed=1)
    items = sample_items(50, prevalence_a=0.5, seed=1)
    votes = votes_for(pop[:4], items, seed=3)
    assert len(votes) == 4
    assert all(len(v) == 50 for v in votes)


@pytest.mark.parametrize("strategy", list(STRATEGIES))
def test_run_trial_all_strategies(strategy):
    r = run_trial(TrialConfig(strategy=strategy, panel_size=6, n_experts=60, n_items=300, seed=3))
    assert isinstance(r, TrialResult)
    assert r.eie_error >= 0.0
    assert r.mv_error >= 0.0
    assert 0.0 <= r.oracle.prevalence_a <= 1.0
    assert r.alarm_misaligned is None  # alarm off by default


def test_run_trial_requires_trio_panel():
    with pytest.raises(ValueError):
        run_trial(TrialConfig(strategy="random_selection", panel_size=2))


def test_run_trial_is_deterministic():
    cfg = TrialConfig(strategy="random_selection", n_items=200, seed=5)
    assert run_trial(cfg).eie_error == run_trial(cfg).eie_error


def test_run_trial_with_optin_alarm_small_q():
    r = run_trial(
        TrialConfig(strategy="representative_sortition", panel_size=6, n_items=200,
                    compute_alarm=True, alarm_q=18, seed=3)
    )
    assert isinstance(r.alarm_misaligned, bool)


def test_representative_carries_audit_hash():
    r = run_trial(TrialConfig(strategy="representative_sortition", panel_size=6, seed=3))
    assert r.panel.audit_hash is not None
