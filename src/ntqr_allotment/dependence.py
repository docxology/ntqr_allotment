"""Error-dependence: the scientific spine of the study.

NTQR's exact trio solver assumes the judges' errors are conditionally
*independent*. Sortition's purpose is to engineer panel diversity. This module
makes that link measurable in simulation:

1. ``sample_votes_correlated`` — a generative model with a tunable error-
   correlation ``rho``: each judge's correctness is driven by a mixture of a
   SHARED per-item latent and an INDEPENDENT per-(judge,item) latent. ``rho=0``
   reproduces independent errors; ``rho=1`` makes every judge succeed/fail
   together on each item (maximal positive error-correlation).

2. ``measure_error_correlations`` — uses NTQR's OWN supervised estimator
   (``pair_label_error_correlation`` / ``three_way_label_error_correlation``) to
   report the realized error-correlation the exact solver assumes is zero.

Together they let us *measure* (not merely motivate) the mechanism: this module
reports the realized error-correlation the exact solver assumes is zero. It does
NOT itself compute oracle-referenced recovery error; the recovery-error-vs-
correlation relationship is studied separately by ``independence_sweep.py`` (and
its shipped ``independence_sweep.csv``), against the analytical sign prediction in
``theory.py``. All seeded -> deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np

from .experts import Expert, Item
from .ntqr_eval import trio_label_vote_counts


def sample_votes_correlated(
    experts: Sequence[Expert],
    items: Sequence[Item],
    *,
    rho: float,
    seed: int,
) -> list[tuple[str, ...]]:
    """Sample each judge's votes with a shared latent inducing error-correlation.

    For item ``k`` a shared uniform ``s_k`` is drawn once; each judge ``j`` draws
    an independent ``u_{j,k}`` and is correct iff
    ``rho * s_k + (1 - rho) * u_{j,k} < accuracy_j(true_label)``. The shared term
    makes correctness co-move across judges -> positive error-correlation that
    grows with ``rho``. ``rho=0`` is the independent baseline.

    KNOWN LIMITATION: a convex combination of two uniforms is not uniform (it is
    triangular, concentrated near 0.5), so ``P(mixed < acc) != acc`` for
    ``0 < rho < 1`` -- this model does NOT preserve marginal accuracy, which
    inflates then deflates realized accuracy with a peak near ``rho=0.5``. It is
    retained as the *correlation diagnostic* (the realized correlation does rise
    with ``rho``), but recovery-vs-correlation conclusions are drawn from the
    marginal-accuracy-preserving Gaussian copula in
    :func:`ntqr_allotment.bloc_confound.sample_votes_bloc_correlated` instead.
    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    rng = np.random.default_rng(seed)
    n = len(items)
    shared = rng.random(n)
    votes: list[tuple[str, ...]] = []
    for j, e in enumerate(experts):
        indep = np.random.default_rng(seed + 101 * (j + 1)).random(n)
        mixed = rho * shared + (1.0 - rho) * indep
        row: list[str] = []
        for k, item in enumerate(items):
            acc = e.accuracy_a if item.true_label == "a" else e.accuracy_b
            correct = mixed[k] < acc
            if item.true_label == "a":
                row.append("a" if correct else "b")
            else:
                row.append("b" if correct else "a")
        votes.append(tuple(row))
    return votes


@dataclass(frozen=True)
class CorrelationReport:
    """Realized error-correlation among a trio of judges (NTQR-measured)."""

    pair_correlations: dict[str, float]  # "(i,j)|label" -> correlation
    three_way: dict[str, float]  # label -> correlation
    mean_abs_pair: float


def measure_error_correlations(
    votes_by_judge: Sequence[Sequence[str]], items: Sequence[Item]
) -> CorrelationReport:
    """Measure realized error-correlation via NTQR's supervised estimator.

    Uses ``SupervisedEvaluation.pair_label_error_correlation`` and
    ``three_way_label_error_correlation`` (the very quantities the exact
    error-independent solver assumes are zero). Trio-only (NTQR R=2 trio API).
    """
    if len(votes_by_judge) != 3:
        raise ValueError("error-correlation measurement is trio-only (3 judges)")
    from ntqr.r2.datasketches import TrioLabelVoteCounts
    from ntqr.r2.evaluators import SupervisedEvaluation

    sup = SupervisedEvaluation(TrioLabelVoteCounts(trio_label_vote_counts(votes_by_judge, items)))
    pair: dict[str, float] = {}
    vals: list[float] = []
    for i, j in combinations(range(3), 2):
        for label in ("a", "b"):
            c = float(sup.pair_label_error_correlation((i, j), label))
            pair[f"({i},{j})|{label}"] = c
            vals.append(abs(c))
    three: dict[str, float] = {}
    for label in ("a", "b"):
        three[label] = float(sup.three_way_label_error_correlation((0, 1, 2), label))
    return CorrelationReport(
        pair_correlations=pair,
        three_way=three,
        mean_abs_pair=float(np.mean(vals)) if vals else 0.0,
    )


__all__ = [
    "sample_votes_correlated",
    "CorrelationReport",
    "measure_error_correlations",
]
