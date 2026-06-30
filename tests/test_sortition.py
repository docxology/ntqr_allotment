"""Tests for panel-formation strategies + the allotment adapter (no mocks)."""

from __future__ import annotations

import pytest

from ntqr_allotment.experts import generate_population
from ntqr_allotment.sortition import (
    STRATEGIES,
    _largest_remainder,
    build_pool,
    expertise_threshold,
    ideological_selection,
    random_selection,
    representative_sortition,
)


@pytest.fixture
def population():
    return generate_population(60, seed=3)


def test_largest_remainder_sums_to_total():
    seats = _largest_remainder({"left": 20, "center": 20, "right": 20}, 7)
    assert sum(seats.values()) == 7
    assert all(v >= 0 for v in seats.values())


def test_largest_remainder_empty_raises():
    with pytest.raises(ValueError):
        _largest_remainder({"left": 0, "right": 0}, 5)


def test_build_pool_has_features_and_candidates(population):
    pool = build_pool(population)
    names = {f.name for f in pool.features}
    assert names == {"ideology", "expertise_tier"}
    assert len(pool.candidates) == 60
    assert all("ideology" in c.features for c in pool.candidates)


def test_representative_sortition_is_audited_and_representative(population):
    draw = representative_sortition(population, 6, seed=3)
    assert draw.strategy == "representative_sortition"
    assert len(draw.expert_ids) == 6
    assert draw.audit_hash is not None and len(draw.audit_hash) == 64
    # 60 experts evenly across 3 ideologies -> 2 seats each.
    assert draw.composition["ideology"] == {"left": 2, "center": 2, "right": 2}
    assert draw.feasibility_warnings == ()


def test_representative_sortition_reproducible(population):
    a = representative_sortition(population, 6, seed=3)
    b = representative_sortition(population, 6, seed=3)
    assert a.expert_ids == b.expert_ids
    assert a.audit_hash == b.audit_hash


def test_random_selection_size_and_unique(population):
    draw = random_selection(population, 8, seed=5)
    assert len(draw.expert_ids) == 8
    assert len(set(draw.expert_ids)) == 8
    assert draw.audit_hash is None


def test_ideological_selection_is_single_bloc(population):
    draw = ideological_selection(population, 6, seed=3)
    # all from one ideology when the bloc is large enough
    assert len(draw.composition["ideology"]) == 1


def test_ideological_selection_spills_over_when_bloc_too_small(population):
    # 20 per ideology; asking for 25 forces spillover into a second bloc.
    draw = ideological_selection(population, 25, seed=0, ideology="left")
    assert len(draw.expert_ids) == 25
    assert len(draw.composition["ideology"]) >= 2


def test_expertise_threshold_picks_top_k(population):
    draw = expertise_threshold(population, 5, seed=1)
    chosen = {e: True for e in draw.expert_ids}
    by_id = {e.id: e for e in population}
    chosen_min = min(by_id[i].expertise for i in draw.expert_ids)
    others_max = max(by_id[e.id].expertise for e in population if e.id not in chosen)
    assert chosen_min >= others_max


def test_strategies_registry_complete():
    assert set(STRATEGIES) == {
        "representative_sortition",
        "random_selection",
        "ideological_selection",
        "expertise_threshold",
    }
