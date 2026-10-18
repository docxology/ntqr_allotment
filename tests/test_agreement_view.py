"""The prepared view preserves the original upstream primitive calculations."""

from itertools import combinations
from ntqr_allotment.agreement import AgreementMoments
from ntqr_allotment.ensemble import (
    observed_vote_counts,
    panel_agreement_moments,
    _pair_frequency_moment,
)
from ntqr_allotment.experts import generate_population, sample_items, sample_votes


def test_prepared_moments_equal_unprepared_ntqr_primitives():
    population = generate_population(6, seed=20)
    items = sample_items(30, prevalence_a=0.6, seed=21)
    votes = [
        sample_votes(expert, items, seed=22 + i) for i, expert in enumerate(population)
    ]
    observed = observed_vote_counts(votes)
    prepared = AgreementMoments(observed)
    moments = panel_agreement_moments(votes)
    for label in "ab":
        original = []
        for pair in combinations(range(len(votes)), 2):
            value = _pair_frequency_moment(observed, pair, label)
            assert prepared.centered_pair(pair, label) == value
            original.append(abs(value))
        assert moments[f"mean_abs_pair_moment_{label}"] == sum(original) / len(original)
    assert len(prepared._marginals) == 12
    assert len(prepared._pairs) == 30
