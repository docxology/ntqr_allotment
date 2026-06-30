"""Per-trio conditioning diagnostics: *why* does enlarging the panel change error?

The ensemble-of-trios (:func:`ntqr_allotment.pipeline.run_trial_ensemble`) averages
the oracle-referenced EIE error over up to ``n_trios`` **usable** trios scanned in
combination order. This module re-walks the SAME trios and records, per trio, three
candidate explanators for the panel-size effect so the manuscript can cite a
measured mechanism instead of asserting one:

* ``mean_abs_corr`` -- mean ``|pairwise error-correlation|`` of the trio (the very
  quantity the error-independent solver assumes is zero). Tests the "larger panels
  pull in more error-correlated / worse-conditioned trios" hypothesis.
* ``mean_judge_accuracy`` -- mean label-averaged accuracy of the trio's three
  judges. Tests a selection effect (larger panels seat lower-accuracy judges).
* ``trio_rank`` -- 0-based position in the usable-trio scan. Because a deterministic
  strategy's first three members are the same at every panel size, ``trio_rank == 0``
  is (often) the identical trio across sizes; if higher-rank trios carry higher
  error, the size effect is an averaging-in-of-later-trios effect, not a property of
  the judges themselves.

Plus the per-trio ``eie_error`` itself. Nothing here is asserted; the mechanism the
manuscript reports is whatever these measured associations actually support.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

from .dependence import measure_error_correlations
from .experts import generate_population, sample_items
from .ntqr_eval import (
    closest_solution,
    error_independent_solutions,
    majority_voting_solutions,
    supervised_oracle,
)
from .pipeline import _DEFAULT_MAX_TRIO_SCAN, TrialConfig, votes_for
from .sortition import STRATEGIES


@dataclass(frozen=True)
class TrioDiagnostic:
    """One usable trio's conditioning record, over the same trio the ensemble averages."""

    strategy: str
    panel_size: int
    seed: int
    mean_expertise: float
    bias_std: float
    trio_rank: int
    eie_error: float
    mean_abs_corr: float
    mean_judge_accuracy: float


def panel_trio_diagnostics(
    config: TrialConfig, *, n_trios: int, max_scan: int = _DEFAULT_MAX_TRIO_SCAN
) -> list[TrioDiagnostic]:
    """Per-usable-trio diagnostics for one panel.

    Mirrors :func:`run_trial_ensemble`'s population/panel/vote construction exactly
    (same seed offsets), then walks combinations in the same deterministic order,
    skipping degenerate trios exactly as the ensemble does, and records one
    :class:`TrioDiagnostic` per usable trio (up to ``n_trios``). The returned
    ``eie_error`` values are exactly the per-trio errors the ensemble would average.
    """
    if config.panel_size < 3:
        raise ValueError("panel_size must be >= 3 (the exact NTQR solver needs a trio)")
    if n_trios < 1:
        raise ValueError("n_trios must be >= 1")

    population = generate_population(
        config.n_experts,
        seed=config.seed,
        mean_expertise=config.mean_expertise,
        expertise_heterogeneity=config.expertise_heterogeneity,
        bias_std=config.bias_std,
    )
    items = sample_items(config.n_items, prevalence_a=config.prevalence_a, seed=config.seed + 7)
    panel = STRATEGIES[config.strategy](population, config.panel_size, seed=config.seed + 13)
    by_id = {e.id: e for e in population}
    panel_experts = [by_id[eid] for eid in panel.expert_ids]
    panel_votes = votes_for(panel_experts, items, seed=config.seed + 29)

    records: list[TrioDiagnostic] = []
    for scanned, trio_indices in enumerate(combinations(range(len(panel_experts)), 3)):
        if len(records) >= n_trios or scanned >= max_scan:
            break
        trio_votes = [panel_votes[k] for k in trio_indices]
        eie_sols = error_independent_solutions(trio_votes)
        mv_sols = majority_voting_solutions(trio_votes)
        if not eie_sols or not mv_sols:
            continue  # degenerate trio: the ensemble skips it, so do we
        oracle = supervised_oracle(trio_votes, items)
        eie = closest_solution(eie_sols, oracle)
        records.append(
            TrioDiagnostic(
                strategy=config.strategy,
                panel_size=config.panel_size,
                seed=config.seed,
                mean_expertise=config.mean_expertise,
                bias_std=config.bias_std,
                trio_rank=len(records),
                eie_error=eie.error_vs(oracle),
                mean_abs_corr=measure_error_correlations(trio_votes, items).mean_abs_pair,
                mean_judge_accuracy=sum(panel_experts[k].mean_accuracy for k in trio_indices) / 3.0,
            )
        )
    return records


def mean_by_size(records: Sequence[TrioDiagnostic], field: str) -> dict[int, float]:
    """Mean of one numeric field grouped by panel size (across all strategies/regimes)."""
    buckets: dict[int, list[float]] = {}
    for rec in records:
        buckets.setdefault(rec.panel_size, []).append(float(getattr(rec, field)))
    return {size: sum(vals) / len(vals) for size, vals in sorted(buckets.items()) if vals}


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation; 0.0 if either series is constant (undefined slope)."""
    n = len(xs)
    if n != len(ys) or n < 2:
        raise ValueError("pearson needs two equal-length series of length >= 2")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0.0 or syy == 0.0:
        return 0.0
    return sxy / (sxx**0.5 * syy**0.5)


__all__ = [
    "TrioDiagnostic",
    "panel_trio_diagnostics",
    "mean_by_size",
    "pearson",
]
