"""Tests for the synthetic expert/judgment generator (no mocks; real numerics)."""

from __future__ import annotations

import pytest

from ntqr_allotment.experts import (
    LABELS,
    Expert,
    Item,
    expertise_tier_of,
    generate_population,
    make_expert,
    population_feature_marginals,
    sample_items,
    sample_votes,
)


def test_labels_are_binary_ab():
    assert LABELS == ("a", "b")


def test_generate_population_is_deterministic():
    a = generate_population(50, seed=7)
    b = generate_population(50, seed=7)
    assert [e.id for e in a] == [e.id for e in b]
    assert [e.expertise for e in a] == [e.expertise for e in b]
    assert [e.bias for e in a] == [e.bias for e in b]


def test_generate_population_different_seeds_differ():
    a = generate_population(50, seed=1)
    b = generate_population(50, seed=2)
    assert [e.expertise for e in a] != [e.expertise for e in b]


def test_generate_population_rejects_nonpositive():
    with pytest.raises(ValueError):
        generate_population(0, seed=1)


def test_population_size_and_ids_unique():
    pop = generate_population(40, seed=3)
    assert len(pop) == 40
    assert len({e.id for e in pop}) == 40


@pytest.mark.parametrize(
    "expertise,tier",
    [(0.5, "low"), (0.64, "low"), (0.65, "med"), (0.79, "med"), (0.8, "high"), (0.99, "high")],
)
def test_expertise_tier_boundaries(expertise, tier):
    assert expertise_tier_of(expertise) == tier


def test_make_expert_positive_bias_skews_toward_a():
    e = make_expert("x", expertise=0.75, bias=0.8, ideology="right")
    assert e.accuracy_a > e.accuracy_b  # positive bias -> better at 'a'
    assert 0.0 < e.accuracy_a < 1.0 and 0.0 < e.accuracy_b < 1.0


def test_make_expert_clips_extremes():
    e = make_expert("x", expertise=0.99, bias=1.0, ideology="right", bias_spread=0.5)
    assert e.accuracy_a <= 0.99
    assert e.accuracy_b >= 0.01


def test_mean_accuracy_property():
    e = make_expert("x", expertise=0.7, bias=0.0, ideology="center")
    assert e.mean_accuracy == pytest.approx(0.5 * (e.accuracy_a + e.accuracy_b))


def test_sample_items_prevalence_and_determinism():
    items = sample_items(2000, prevalence_a=0.3, seed=11)
    frac_a = sum(1 for it in items if it.true_label == "a") / len(items)
    assert 0.25 < frac_a < 0.35
    again = sample_items(2000, prevalence_a=0.3, seed=11)
    assert [it.true_label for it in items] == [it.true_label for it in again]


def test_sample_items_validation():
    with pytest.raises(ValueError):
        sample_items(0, prevalence_a=0.5, seed=1)
    with pytest.raises(ValueError):
        sample_items(10, prevalence_a=1.5, seed=1)


def test_sample_votes_accuracy_matches_label_conditional():
    # A near-perfect 'a' judge, weak 'b' judge: empirical rates track parameters.
    e = make_expert("x", expertise=0.9, bias=0.0, ideology="center")
    items = sample_items(3000, prevalence_a=0.5, seed=2)
    votes = sample_votes(e, items, seed=5)
    a_items = [(i, it) for i, it in enumerate(items) if it.true_label == "a"]
    a_correct = sum(1 for i, it in a_items if votes[i] == "a") / len(a_items)
    assert a_correct == pytest.approx(e.accuracy_a, abs=0.05)


def test_sample_votes_only_emits_valid_labels():
    e = make_expert("x", expertise=0.7, bias=0.5, ideology="left")
    items = sample_items(100, prevalence_a=0.5, seed=1)
    votes = sample_votes(e, items, seed=1)
    assert len(votes) == 100
    assert set(votes) <= {"a", "b"}


def test_population_feature_marginals_sum_to_population():
    pop = generate_population(60, seed=4)
    marg = population_feature_marginals(pop)
    assert sum(marg["ideology"].values()) == 60
    assert sum(marg["expertise_tier"].values()) == 60


def test_item_and_expert_are_frozen():
    it = Item(index=0, true_label="a")
    e = make_expert("x", 0.7, 0.0, "center")
    with pytest.raises(Exception):
        it.index = 5  # type: ignore[misc]
    with pytest.raises(Exception):
        e.id = "y"  # type: ignore[misc]
    assert isinstance(e, Expert)
