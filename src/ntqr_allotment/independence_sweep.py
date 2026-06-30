"""Error-correlation tolerance sweep: the centerpiece measurement.

NTQR's exact trio solver assumes the judges' errors are conditionally
*independent* (:mod:`ntqr_allotment.dependence`). This module turns the slogan
"diversity helps" into a measured **tolerance curve**: it injects a controlled
error-correlation ``rho`` into a trio's votes, measures the *realized*
correlation NTQR itself reports, computes the unsupervised evaluation, and scores
it against the supervised oracle. Sweeping ``rho`` (and panel-formation
strategy and panel size, aggregated over seeds) shows how recovery error
degrades as injected correlation rises.

Honest framing: this module *measures* the degradation; it does not assume it.
A cell where the trio admits no real error-independent solution is recorded
with the :data:`DEGENERATE` sentinel and excluded from scoring, exactly as
:mod:`ntqr_allotment.sweeps` handles its degenerate trios. The realized
correlation is the NTQR-measured ``mean_abs_pair`` from
:func:`ntqr_allotment.dependence.measure_error_correlations`, not the injected
``rho``.

All randomness flows through ``numpy.random.default_rng(seed)`` -> deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from math import sqrt
from typing import Any

import numpy as np

from .dependence import measure_error_correlations, sample_votes_correlated
from .experts import generate_population, sample_items
from .ntqr_eval import closest_solution, error_independent_solutions, supervised_oracle
from .sortition import STRATEGIES
from .theory import fit_error_correlation_slope

#: Sentinel recorded for a degenerate trial (no real solution / non-finite).
#: Mirrors :data:`ntqr_allotment.sweeps.DEGENERATE_ERROR`; negative because a
#: real recovery error is a non-negative distance.
DEGENERATE = -1.0

_TRIO_SIZE = 3


@dataclass(frozen=True)
class IndependenceGrid:
    """Frozen specification of the error-correlation tolerance sweep.

    Each axis is a tuple of values; the trial grid is their Cartesian product
    crossed with ``seeds``. ``rhos`` are the *injected* correlation levels in
    ``[0, 1]``; ``panel_sizes`` must be at least :data:`_TRIO_SIZE` because the
    NTQR exact solver scores the first three panel experts (the C1 trio).
    """

    rhos: tuple[float, ...]
    strategies: tuple[str, ...]
    panel_sizes: tuple[int, ...]
    seeds: tuple[int, ...]
    n_experts: int = 24
    n_items: int = 120

    def __post_init__(self) -> None:
        axes: dict[str, tuple[Any, ...]] = {
            "rhos": self.rhos,
            "strategies": self.strategies,
            "panel_sizes": self.panel_sizes,
            "seeds": self.seeds,
        }
        for axis_name, axis_values in axes.items():
            if not axis_values:
                raise ValueError(f"{axis_name} must be non-empty")
        for rho in self.rhos:
            if not 0.0 <= rho <= 1.0:
                raise ValueError("every rho must be in [0, 1]")
        for strategy in self.strategies:
            if strategy not in STRATEGIES:
                raise ValueError(f"invalid strategies value: {strategy}")
        for panel_size in self.panel_sizes:
            if panel_size < _TRIO_SIZE:
                raise ValueError(f"panel_size must be >= {_TRIO_SIZE}")
        if self.n_experts < _TRIO_SIZE:
            raise ValueError(f"n_experts must be >= {_TRIO_SIZE}")
        if self.n_items <= 0:
            raise ValueError("n_items must be positive")


@dataclass(frozen=True)
class IndependenceRow:
    """One per-(rho, strategy, panel_size, seed) trial outcome.

    ``eie_error`` is the recovered evaluation's distance to the supervised
    oracle (:data:`DEGENERATE` if the trio admitted no real solution).
    ``realized_corr`` is the NTQR-measured ``mean_abs_pair`` (always finite).
    """

    rho: float
    strategy: str
    panel_size: int
    n_experts: int
    n_items: int
    seed: int
    eie_error: float
    realized_corr: float
    degenerate: bool

    def config_key(self) -> tuple[Any, ...]:
        return (self.rho, self.strategy, self.panel_size, self.n_experts, self.n_items)


@dataclass(frozen=True)
class IndependenceAggregate:
    """Multi-seed aggregate for one (rho, strategy, panel_size) cell."""

    rho: float
    strategy: str
    panel_size: int
    n_experts: int
    n_items: int
    n: int
    eie_mean: float
    eie_std: float
    eie_ci95: float
    corr_mean: float


def _is_degenerate_error(value: float) -> bool:
    return not math.isfinite(value) or value < 0.0


def _trio_experts(strategy: str, pop: list[Any], panel_size: int, seed: int) -> list[Any]:
    """Form a panel and return the first three panel experts (the C1 trio)."""
    panel = STRATEGIES[strategy](pop, panel_size, seed=seed + 13)
    by_id = {e.id: e for e in pop}
    return [by_id[eid] for eid in panel.expert_ids[:_TRIO_SIZE]]


def _run_trial(
    *, rho: float, strategy: str, panel_size: int, seed: int, n_experts: int, n_items: int
) -> IndependenceRow:
    """Execute one trial: inject ``rho``, measure realized corr, score recovery.

    The realized correlation is always computed (so it is recorded even for a
    degenerate trio); recovery error is :data:`DEGENERATE` when the exact solver
    yields no real solution.
    """
    pop = generate_population(n_experts, seed=seed)
    items = sample_items(n_items, prevalence_a=0.5, seed=seed + 7)
    trio = _trio_experts(strategy, pop, panel_size, seed)
    votes = sample_votes_correlated(trio, items, rho=rho, seed=seed + 29)
    realized_corr = measure_error_correlations(votes, items).mean_abs_pair

    sols = error_independent_solutions(votes)
    if not sols:
        return IndependenceRow(
            rho=rho,
            strategy=strategy,
            panel_size=panel_size,
            n_experts=n_experts,
            n_items=n_items,
            seed=seed,
            eie_error=DEGENERATE,
            realized_corr=realized_corr,
            degenerate=True,
        )
    oracle = supervised_oracle(votes, items)
    eie = closest_solution(sols, oracle)
    return IndependenceRow(
        rho=rho,
        strategy=strategy,
        panel_size=panel_size,
        n_experts=n_experts,
        n_items=n_items,
        seed=seed,
        eie_error=eie.error_vs(oracle),
        realized_corr=realized_corr,
        degenerate=False,
    )


def run_independence_sweep(grid: IndependenceGrid) -> list[IndependenceRow]:
    """Run every (rho, strategy, panel_size, seed) trial in ``grid``.

    Returns one :class:`IndependenceRow` per trial in deterministic Cartesian
    order; degenerate trials are kept (with the sentinel error) so callers can
    audit how often a regime collapses.
    """
    rows: list[IndependenceRow] = []
    for rho, strategy, panel_size in product(grid.rhos, grid.strategies, grid.panel_sizes):
        for seed in grid.seeds:
            rows.append(
                _run_trial(
                    rho=rho,
                    strategy=strategy,
                    panel_size=panel_size,
                    seed=seed,
                    n_experts=grid.n_experts,
                    n_items=grid.n_items,
                )
            )
    return rows


def _sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def _ci95(std: float, n: int) -> float:
    if n <= 1:
        return 0.0
    return float(std / sqrt(n) * 1.96)


def aggregate_independence(rows: list[IndependenceRow]) -> list[IndependenceAggregate]:
    """Aggregate trials by (rho, strategy, panel_size) over seeds.

    ``eie_mean``/``eie_std``/``eie_ci95`` are computed over the non-degenerate
    trials only (``eie_mean`` is :data:`DEGENERATE` when every trial in the cell
    degenerated). ``corr_mean`` averages the realized correlation over ALL
    trials in the cell, since the realized correlation is always finite.
    """
    grouped: dict[tuple[Any, ...], list[IndependenceRow]] = {}
    for row in rows:
        grouped.setdefault(row.config_key(), []).append(row)

    aggregates: list[IndependenceAggregate] = []
    for config_key in sorted(grouped):
        group = grouped[config_key]
        good = [r.eie_error for r in group if not _is_degenerate_error(r.eie_error)]
        if good:
            eie_mean = float(np.mean(np.asarray(good, dtype=float)))
            eie_std = _sample_std(good)
            eie_ci95 = _ci95(eie_std, len(good))
        else:
            eie_mean, eie_std, eie_ci95 = DEGENERATE, 0.0, 0.0
        corr_mean = float(np.mean(np.asarray([r.realized_corr for r in group], dtype=float)))
        rho, strategy, panel_size, n_experts, n_items = config_key
        aggregates.append(
            IndependenceAggregate(
                rho=rho,
                strategy=strategy,
                panel_size=panel_size,
                n_experts=n_experts,
                n_items=n_items,
                n=len(good),
                eie_mean=eie_mean,
                eie_std=eie_std,
                eie_ci95=eie_ci95,
                corr_mean=corr_mean,
            )
        )
    return aggregates


def tolerance_slope(aggregates: list[IndependenceAggregate]) -> float:
    """OLS slope of recovery error vs realized correlation across cells.

    Reuses :func:`ntqr_allotment.theory.fit_error_correlation_slope` on the
    non-degenerate aggregate cells: x = realized ``corr_mean``, y = ``eie_mean``.
    A positive slope is the measured "tolerance" signal (error rises with
    correlation). Raises ``ValueError`` if fewer than two non-degenerate cells
    are available or the correlations have no variance.
    """
    usable = [a for a in aggregates if a.n > 0 and not _is_degenerate_error(a.eie_mean)]
    # De-duplicate identical (corr, error) points: a panel_size axis that does not
    # change the evaluated trio would otherwise feed size-invariant duplicate
    # cells into the regression, double-counting points and faking the slope's
    # effective N (see scripts/run_independence_sweep.py for why the default grid
    # fixes the trio size).
    unique = sorted({(a.corr_mean, a.eie_mean) for a in usable})
    if len(unique) < 2:
        raise ValueError("need at least two distinct non-degenerate aggregate cells")
    return fit_error_correlation_slope(
        [corr for corr, _ in unique],
        [eie for _, eie in unique],
    )


__all__ = [
    "DEGENERATE",
    "IndependenceGrid",
    "IndependenceRow",
    "IndependenceAggregate",
    "run_independence_sweep",
    "aggregate_independence",
    "tolerance_slope",
]
