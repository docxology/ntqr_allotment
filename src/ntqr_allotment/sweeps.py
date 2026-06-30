from __future__ import annotations

import math
import signal
from dataclasses import dataclass
from itertools import product
from math import sqrt
from multiprocessing import Pool
from types import FrameType
from typing import Any

import numpy as np

from .pipeline import TrialConfig, run_trial_ensemble
from .sortition import STRATEGIES

DEGENERATE_ERROR = -1.0


class TrialTimeoutError(TimeoutError):
    pass


@dataclass(frozen=True)
class SweepGrid:
    strategies: tuple[str, ...]
    panel_sizes: tuple[int, ...]
    mean_expertises: tuple[float, ...]
    expertise_heterogeneities: tuple[float, ...]
    bias_stds: tuple[float, ...]
    n_items_values: tuple[int, ...]
    prevalence_as: tuple[float, ...]
    seeds: tuple[int, ...]
    n_experts: int = 60
    n_trios: int = 5

    def __post_init__(self) -> None:
        axes: dict[str, tuple[Any, ...]] = {
            "strategies": self.strategies,
            "panel_sizes": self.panel_sizes,
            "mean_expertises": self.mean_expertises,
            "expertise_heterogeneities": self.expertise_heterogeneities,
            "bias_stds": self.bias_stds,
            "n_items_values": self.n_items_values,
            "prevalence_as": self.prevalence_as,
            "seeds": self.seeds,
        }
        for axis_name, axis_values in axes.items():
            if not axis_values:
                raise ValueError(f"{axis_name} must be non-empty")
        for strategy in self.strategies:
            if strategy not in STRATEGIES:
                raise ValueError(f"invalid strategies value: {strategy}")
        if self.n_experts <= 0:
            raise ValueError("n_experts must be positive")
        if self.n_trios < 1:
            raise ValueError("n_trios must be >= 1")

    def cells(self) -> list[TrialConfig]:
        cells: list[TrialConfig] = []
        for values in product(
            self.strategies,
            self.panel_sizes,
            self.mean_expertises,
            self.expertise_heterogeneities,
            self.bias_stds,
            self.n_items_values,
            self.prevalence_as,
        ):
            strategy, panel_size, mean_expertise, expertise_heterogeneity, bias_std, n_items, prevalence_a = values
            cells.append(
                TrialConfig(
                    strategy=strategy,
                    panel_size=panel_size,
                    n_experts=self.n_experts,
                    n_items=n_items,
                    prevalence_a=prevalence_a,
                    mean_expertise=mean_expertise,
                    expertise_heterogeneity=expertise_heterogeneity,
                    bias_std=bias_std,
                    seed=0,
                    compute_alarm=False,
                )
            )
        return cells


@dataclass(frozen=True)
class SweepRow:
    strategy: str
    panel_size: int
    mean_expertise: float
    expertise_heterogeneity: float
    bias_std: float
    n_items: int
    prevalence_a: float
    n_experts: int
    seed: int
    n_trios: int
    eie_error: float
    mv_error: float

    def config_key(self) -> tuple[Any, ...]:
        return (
            self.strategy,
            self.panel_size,
            self.mean_expertise,
            self.expertise_heterogeneity,
            self.bias_std,
            self.n_items,
            self.prevalence_a,
            self.n_experts,
        )

    def regime_key(self) -> tuple[Any, ...]:
        return self.config_key()[1:]


@dataclass(frozen=True)
class Aggregate:
    config_key: tuple[Any, ...]
    strategy: str
    n: int
    eie_mean: float
    eie_std: float
    eie_ci95: float
    mv_mean: float
    mv_std: float
    mv_ci95: float

    def regime_key(self) -> tuple[Any, ...]:
        return self.config_key[1:]


@dataclass(frozen=True)
class RepVsIdeoEffect:
    regime_key: tuple[Any, ...]
    rep_mean: float
    ideo_mean: float
    effect: float
    ci95: float


def _is_degenerate_error(value: float) -> bool:
    return not math.isfinite(value) or value < 0.0


def _degenerate_row(config: TrialConfig) -> SweepRow:
    return SweepRow(
        strategy=config.strategy,
        panel_size=config.panel_size,
        mean_expertise=config.mean_expertise,
        expertise_heterogeneity=config.expertise_heterogeneity,
        bias_std=config.bias_std,
        n_items=config.n_items,
        prevalence_a=config.prevalence_a,
        n_experts=config.n_experts,
        seed=config.seed,
        n_trios=0,
        eie_error=DEGENERATE_ERROR,
        mv_error=DEGENERATE_ERROR,
    )


def _trial_configs(grid: SweepGrid) -> list[TrialConfig]:
    configs: list[TrialConfig] = []
    for cell in grid.cells():
        for seed in grid.seeds:
            configs.append(
                TrialConfig(
                    strategy=cell.strategy,
                    panel_size=cell.panel_size,
                    n_experts=cell.n_experts,
                    n_items=cell.n_items,
                    prevalence_a=cell.prevalence_a,
                    mean_expertise=cell.mean_expertise,
                    expertise_heterogeneity=cell.expertise_heterogeneity,
                    bias_std=cell.bias_std,
                    seed=seed,
                    compute_alarm=False,
                    alarm_q=cell.alarm_q,
                )
            )
    return configs


def _raise_timeout(_signum: int, _frame: FrameType | None) -> None:
    raise TrialTimeoutError("trial evaluation exceeded timeout")


class _trial_time_limit:
    def __init__(self, seconds: float | None):
        self.seconds = seconds
        self._old_handler: Any = None
        self._old_timer: tuple[float, float] | None = None

    def __enter__(self) -> None:
        if self.seconds is None or self.seconds <= 0:
            return
        self._old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
        self._old_timer = signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, *_exc: object) -> None:
        if self.seconds is None or self.seconds <= 0:
            return
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        if self._old_handler is not None:
            signal.signal(signal.SIGALRM, self._old_handler)
        if self._old_timer is not None and self._old_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *self._old_timer)


def _run_sweep_config(config: TrialConfig, n_trios: int, trial_timeout_s: float | None) -> SweepRow:
    try:
        with _trial_time_limit(trial_timeout_s):
            result = run_trial_ensemble(config, n_trios=n_trios)
    except (ValueError, TrialTimeoutError):
        return _degenerate_row(config)
    if (
        result.n_trios == 0
        or _is_degenerate_error(result.eie_error)
        or _is_degenerate_error(result.mv_error)
    ):
        return _degenerate_row(config)
    return SweepRow(
        strategy=config.strategy,
        panel_size=config.panel_size,
        mean_expertise=config.mean_expertise,
        expertise_heterogeneity=config.expertise_heterogeneity,
        bias_std=config.bias_std,
        n_items=config.n_items,
        prevalence_a=config.prevalence_a,
        n_experts=config.n_experts,
        seed=config.seed,
        n_trios=result.n_trios,
        eie_error=result.eie_error,
        mv_error=result.mv_error,
    )


def _run_sweep_task(task: tuple[TrialConfig, int, float | None]) -> SweepRow:
    config, n_trios, trial_timeout_s = task
    return _run_sweep_config(config, n_trios, trial_timeout_s)


def run_sweep(grid: SweepGrid, *, trial_timeout_s: float | None = None) -> list[SweepRow]:
    rows: list[SweepRow] = []
    for config in _trial_configs(grid):
        rows.append(_run_sweep_config(config, grid.n_trios, trial_timeout_s))
    return rows


def run_sweep_parallel(
    grid: SweepGrid,
    *,
    workers: int = 1,
    trial_timeout_s: float | None = None,
) -> list[SweepRow]:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if workers == 1:
        return run_sweep(grid, trial_timeout_s=trial_timeout_s)
    tasks = [
        (config, grid.n_trios, trial_timeout_s)
        for config in _trial_configs(grid)
    ]
    with Pool(processes=workers) as pool:
        return list(
            pool.imap(
                _run_sweep_task,
                tasks,
                chunksize=max(1, min(8, len(tasks) // (workers * 4) or 1)),
            )
        )


def _sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def _ci95(std: float, n: int) -> float:
    if n <= 1:
        return 0.0
    return float(std / sqrt(n) * 1.96)


def _non_degenerate(values: list[float]) -> list[float]:
    return [v for v in values if not _is_degenerate_error(v)]


def _mean_std_ci(values: list[float]) -> tuple[float, float, float]:
    non_degenerate = _non_degenerate(values)
    if not non_degenerate:
        return DEGENERATE_ERROR, 0.0, 0.0
    mean = float(np.mean(np.asarray(non_degenerate, dtype=float)))
    std = _sample_std(non_degenerate)
    return mean, std, _ci95(std, len(non_degenerate))


def aggregate(rows: list[SweepRow]) -> list[Aggregate]:
    grouped: dict[tuple[Any, ...], list[SweepRow]] = {}
    for row in rows:
        grouped.setdefault(row.config_key(), []).append(row)

    aggregates: list[Aggregate] = []
    for config_key in sorted(grouped):
        group_rows = grouped[config_key]
        eie_values = [row.eie_error for row in group_rows]
        mv_values = [row.mv_error for row in group_rows]
        eie_mean, eie_std, eie_ci95 = _mean_std_ci(eie_values)
        mv_mean, mv_std, mv_ci95 = _mean_std_ci(mv_values)
        n = sum(1 for value in eie_values if not _is_degenerate_error(value))
        aggregates.append(
            Aggregate(
                config_key=config_key,
                strategy=config_key[0],
                n=n,
                eie_mean=eie_mean,
                eie_std=eie_std,
                eie_ci95=eie_ci95,
                mv_mean=mv_mean,
                mv_std=mv_std,
                mv_ci95=mv_ci95,
            )
        )
    return aggregates


def _weighted_mean_and_ci(values: list[float], weights: list[int]) -> tuple[float, float]:
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("weights must sum to a positive integer")
    mean = sum(value * weight for value, weight in zip(values, weights, strict=True)) / total_weight
    if total_weight <= 1:
        return mean, 0.0
    variance = sum(
        weight * (value - mean) ** 2 for value, weight in zip(values, weights, strict=True)
    ) / (total_weight - 1)
    std = sqrt(variance)
    return mean, float(std / sqrt(total_weight) * 1.96)


def strategy_ranking(aggregated: list[Aggregate]) -> list[tuple[str, float, float]]:
    by_strategy: dict[str, list[Aggregate]] = {}
    for item in aggregated:
        if item.n > 0:
            by_strategy.setdefault(item.strategy, []).append(item)

    ranking: list[tuple[str, float, float]] = []
    for strategy in sorted(by_strategy):
        items = by_strategy[strategy]
        mean, ci95 = _weighted_mean_and_ci(
            [item.eie_mean for item in items],
            [item.n for item in items],
        )
        ranking.append((strategy, mean, ci95))
    return sorted(ranking, key=lambda item: (item[1], item[0]))


def representative_vs_ideological(aggregated: list[Aggregate]) -> list[RepVsIdeoEffect]:
    by_regime: dict[tuple[Any, ...], dict[str, Aggregate]] = {}
    for item in aggregated:
        by_regime.setdefault(item.regime_key(), {})[item.strategy] = item

    effects: list[RepVsIdeoEffect] = []
    for regime_key in sorted(by_regime):
        strategies = by_regime[regime_key]
        representative = strategies.get("representative_sortition")
        ideological = strategies.get("ideological_selection")
        if representative is None or ideological is None:
            continue
        if representative.n == 0 or ideological.n == 0:
            continue
        effects.append(
            RepVsIdeoEffect(
                regime_key=regime_key,
                rep_mean=representative.eie_mean,
                ideo_mean=ideological.eie_mean,
                effect=ideological.eie_mean - representative.eie_mean,
                ci95=float(sqrt(representative.eie_ci95 ** 2 + ideological.eie_ci95 ** 2)),
            )
        )
    return effects


__all__ = [
    "SweepGrid",
    "SweepRow",
    "Aggregate",
    "RepVsIdeoEffect",
    "TrialTimeoutError",
    "run_sweep",
    "run_sweep_parallel",
    "aggregate",
    "strategy_ranking",
    "representative_vs_ideological",
]
