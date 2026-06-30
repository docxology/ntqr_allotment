"""Synthetic expert population and their label-conditional judgments.

This is the *generator* layer of the project: it produces noisy binary experts
whose competence is fully parameterized (precision/expertise, bias,
heterogeneity) and a corpus of items with KNOWN ground truth. Because the truth
is known here, downstream code can compare no-answer-key `ntqr` evaluations
against the supervised oracle on equal footing (the honest comparator).

Label space is the binary ``("a", "b")`` to match ``ntqr.r2``.

All randomness is seeded (``numpy.random.default_rng``) -> deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

LABELS: tuple[str, str] = ("a", "b")
_DEFAULT_BIAS_SPREAD = 0.15


@dataclass(frozen=True)
class Item:
    """A single item to be judged, with its (synthetic) ground-truth label."""

    index: int
    true_label: str


@dataclass(frozen=True)
class Expert:
    """A noisy binary judge with label-conditional accuracies.

    ``accuracy_a`` = P(votes 'a' | true label is 'a'); ``accuracy_b`` likewise.
    ``expertise`` is the mean precision; ``bias`` in [-1, 1] is the asymmetry
    that skews errors toward one label (the thing NTQR's per-label accuracy
    estimates are sensitive to). ``ideology`` and ``expertise_tier`` are
    categorical features used by the sortition layer for stratification.
    """

    id: str
    accuracy_a: float
    accuracy_b: float
    expertise: float
    bias: float
    ideology: str
    expertise_tier: str

    @property
    def mean_accuracy(self) -> float:
        return 0.5 * (self.accuracy_a + self.accuracy_b)


def expertise_tier_of(expertise: float) -> str:
    """Bucket a continuous expertise score into a categorical tier."""
    if expertise < 0.65:
        return "low"
    if expertise < 0.80:
        return "med"
    return "high"


def make_expert(
    expert_id: str,
    expertise: float,
    bias: float,
    ideology: str,
    *,
    bias_spread: float = _DEFAULT_BIAS_SPREAD,
) -> Expert:
    """Build one :class:`Expert`, mapping expertise+bias to label accuracies."""
    acc_a = float(np.clip(expertise + bias * bias_spread, 0.01, 0.99))
    acc_b = float(np.clip(expertise - bias * bias_spread, 0.01, 0.99))
    return Expert(
        id=expert_id,
        accuracy_a=acc_a,
        accuracy_b=acc_b,
        expertise=float(expertise),
        bias=float(bias),
        ideology=ideology,
        expertise_tier=expertise_tier_of(expertise),
    )


def generate_population(
    n_experts: int,
    *,
    seed: int,
    mean_expertise: float = 0.72,
    expertise_heterogeneity: float = 0.08,
    bias_mean: float = 0.0,
    bias_std: float = 0.3,
    ideologies: Sequence[str] = ("left", "center", "right"),
    bias_spread: float = _DEFAULT_BIAS_SPREAD,
) -> list[Expert]:
    """Generate a heterogeneous expert population.

    ``expertise_heterogeneity`` is the std of the per-expert expertise (the
    spread that NTQR's error-independence assumption is stressed by);
    ``bias_mean``/``bias_std`` control the population's systematic skew; the
    ``ideology`` of each expert is correlated with its bias sign so that
    *ideological* selection produces a genuinely more correlated panel than a
    *representative* draw (the contrast the project measures).
    """
    if n_experts <= 0:
        raise ValueError("n_experts must be positive")
    rng = np.random.default_rng(seed)
    experts: list[Expert] = []
    n_id = len(ideologies)
    for i in range(n_experts):
        expertise = float(np.clip(rng.normal(mean_expertise, expertise_heterogeneity), 0.5, 0.99))
        ideology = str(ideologies[i % n_id])
        # ideology nudges bias sign: left -> negative, right -> positive.
        ideo_shift = (ideologies.index(ideology) - (n_id - 1) / 2.0) / max(n_id, 1)
        bias = float(np.clip(rng.normal(bias_mean + ideo_shift, bias_std), -1.0, 1.0))
        experts.append(
            make_expert(f"e{i:04d}", expertise, bias, ideology, bias_spread=bias_spread)
        )
    return experts


def sample_items(n_items: int, *, prevalence_a: float, seed: int) -> list[Item]:
    """Create a corpus of items with known labels; ``prevalence_a`` = P(true='a')."""
    if n_items <= 0:
        raise ValueError("n_items must be positive")
    if not 0.0 <= prevalence_a <= 1.0:
        raise ValueError("prevalence_a must be in [0, 1]")
    rng = np.random.default_rng(seed)
    draws = rng.random(n_items)
    return [Item(index=i, true_label="a" if draws[i] < prevalence_a else "b") for i in range(n_items)]


def sample_votes(expert: Expert, items: Sequence[Item], *, seed: int) -> tuple[str, ...]:
    """Sample one expert's binary votes over ``items`` (aligned by position)."""
    rng = np.random.default_rng(seed)
    draws = rng.random(len(items))
    votes: list[str] = []
    for k, item in enumerate(items):
        acc = expert.accuracy_a if item.true_label == "a" else expert.accuracy_b
        correct = draws[k] < acc
        if item.true_label == "a":
            votes.append("a" if correct else "b")
        else:
            votes.append("b" if correct else "a")
    return tuple(votes)


def population_feature_marginals(experts: Sequence[Expert]) -> dict[str, dict[str, int]]:
    """Count experts by feature value, for building representative quotas."""
    marginals: dict[str, dict[str, int]] = {"ideology": {}, "expertise_tier": {}}
    for e in experts:
        marginals["ideology"][e.ideology] = marginals["ideology"].get(e.ideology, 0) + 1
        marginals["expertise_tier"][e.expertise_tier] = (
            marginals["expertise_tier"].get(e.expertise_tier, 0) + 1
        )
    return marginals


__all__ = [
    "LABELS",
    "Item",
    "Expert",
    "expertise_tier_of",
    "make_expert",
    "generate_population",
    "sample_items",
    "sample_votes",
    "population_feature_marginals",
]
