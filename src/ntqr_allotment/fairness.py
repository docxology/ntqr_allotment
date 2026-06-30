"""Comprehensive use of the allotment maximin engine: fairness & representation.

Beyond drawing one panel, the maximin lottery defines a *probability
distribution over feasible panels* and hence a marginal selection probability
for every candidate. This module exposes that distribution and the fairness
metrics it was designed to optimize (Flanigan et al. 2021: maximize the minimum
selection probability), plus multi-feature stratification and a representation
error against the population marginals.

Uses the real engine: ``generate_feasible_panels``, ``solve_maximin_weights``,
and ``run_draw``/``realised_probabilities``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from allotment.domain import QuotaConfig, QuotaTarget
from allotment.selection_core.audit import run_draw
from allotment.selection_core.maximin import solve_maximin_weights
from allotment.selection_core.panels import generate_feasible_panels

from .experts import Expert
from .sortition import _largest_remainder, build_pool


def multi_feature_quota_config(
    experts: Sequence[Expert],
    panel_size: int,
    *,
    features: Sequence[str] = ("ideology", "expertise_tier"),
) -> QuotaConfig:
    """Build a QuotaConfig stratifying on MULTIPLE features simultaneously.

    Each feature's value counts are apportioned to ``panel_size`` by largest
    remainder, so the panel mirrors the population on every named feature at
    once (a richer use of allotment than single-feature quotas).
    """
    targets: list[QuotaTarget] = []
    for feature in features:
        counts: dict[str, int] = {}
        for e in experts:
            val = getattr(e, feature)
            counts[val] = counts.get(val, 0) + 1
        seats = _largest_remainder(counts, panel_size)
        for val in sorted(seats):
            targets.append(QuotaTarget(feature=feature, value=val, min=seats[val], max=seats[val]))
    return QuotaConfig(panel_size=panel_size, targets=targets)


@dataclass(frozen=True)
class FairnessReport:
    """Maximin fairness of a draw over the feasible-panel distribution."""

    n_feasible_panels: int
    min_selection_prob: float  # the maximin objective (higher = fairer)
    mean_selection_prob: float
    max_selection_prob: float
    gini: float  # inequality of selection probabilities (0 = perfectly equal)
    realised_probabilities: dict[str, float]


def _gini(values: Sequence[float]) -> float:
    arr = np.sort(np.array(list(values), dtype=float))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * arr) / (n * arr.sum())) - (n + 1.0) / n)


def maximin_fairness(
    experts: Sequence[Expert],
    panel_size: int,
    *,
    seed: int,
    panel_count: int = 50,
    features: Sequence[str] = ("ideology",),
) -> FairnessReport:
    """Compute the maximin selection-probability distribution and its fairness.

    Builds feasible panels, solves the maximin weights (a distribution over
    panels), and reports per-candidate marginal selection probabilities plus
    min/mean/max and Gini. ``min_selection_prob`` is the quantity the maximin
    lottery maximizes -- the core fairness guarantee of sortition.
    """
    pool = build_pool(experts)
    config = multi_feature_quota_config(experts, panel_size, features=features)
    panels = generate_feasible_panels(pool, config, panel_count, random.Random(seed))
    candidate_ids = [c.id for c in pool.candidates]
    weights = solve_maximin_weights(panels, candidate_ids)

    marginals: dict[str, float] = {cid: 0.0 for cid in candidate_ids}
    for panel, w in weights.items():
        for cid in panel:
            marginals[cid] += w
    probs = list(marginals.values())
    return FairnessReport(
        n_feasible_panels=len(panels),
        min_selection_prob=float(min(probs)) if probs else 0.0,
        mean_selection_prob=float(np.mean(probs)) if probs else 0.0,
        max_selection_prob=float(max(probs)) if probs else 0.0,
        gini=_gini(probs),
        realised_probabilities=marginals,
    )


def representation_error(experts: Sequence[Expert], realised_probabilities: dict[str, float]) -> float:
    """L1 gap between selection-probability mass per ideology and population share.

    Zero means the (probabilistic) panel mirrors the population's ideology
    composition exactly; larger means systematic over/under-representation.
    """
    by_id = {e.id: e for e in experts}
    total = sum(realised_probabilities.values()) or 1.0
    ideo_mass: dict[str, float] = {}
    pop_count: dict[str, int] = {}
    for e in experts:
        pop_count[e.ideology] = pop_count.get(e.ideology, 0) + 1
    for cid, p in realised_probabilities.items():
        ideo = by_id[cid].ideology
        ideo_mass[ideo] = ideo_mass.get(ideo, 0.0) + p / total
    n = len(experts)
    return float(sum(abs(ideo_mass.get(k, 0.0) - pop_count[k] / n) for k in pop_count))


__all__ = [
    "multi_feature_quota_config",
    "FairnessReport",
    "maximin_fairness",
    "representation_error",
    "run_draw",  # re-exported for convenience
]
