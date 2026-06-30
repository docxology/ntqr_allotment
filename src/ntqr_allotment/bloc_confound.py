"""Bloc-coupled error confound: the channel that makes panel *composition* matter.

The headline collapse of the main sweep -- representative sortition, random
selection, and single-bloc ideological selection recovering at statistically
indistinguishable EIE error -- is a *structural* property of the baseline
generator, not a statistical accident. In ``experts.py`` an expert's ideology
only shifts its *marginal* per-label accuracy asymmetry, and ``sample_votes``
draws every judge's errors from an independent stream. So NTQR's
error-independence assumption holds *equally* for every panel, and the only lever
that moves recovery is mean expertise. Composition is cosmetic.

This module implements the channel sortition theory actually predicts and that the
baseline omits: **experts who share a group attribute share a latent error confound
and so err together**. Concretely, ``sample_votes_bloc_correlated`` drives each
judge's correctness through a Gaussian copula whose shared component is keyed on a
chosen grouping attribute ``axis`` (default the judge's ideological bloc):

    z_j = sqrt(rho) * g_{group(j)} + sqrt(1 - rho) * eps_j ,   correct iff z_j < Phi^-1(acc_j)

* ``g_g`` is one standard normal per (group, item), shared by every judge in group
  ``g``; ``eps_j`` is an independent standard normal per (judge, item). The group
  stream is keyed by a STABLE hash of the group value (``_group_stream_offset``),
  so the same group reuses the same confound across panels and across worker
  processes -- never on its rank among the groups that happen to be present.
* A high shared ``z`` flips a judge's vote symmetrically (toward "b" on true-"a"
  AND toward "a" on true-"b"): the shared channel is a **common error/competence
  shock, not directional bias** -- directional bias (``acc_a != acc_b``) is set
  independently in ``experts.py`` and is preserved exactly here.
* ``rho`` is the LATENT within-group (copula) correlation of ``z``. NTQR's own
  supervised evaluator reports a label-conditional error-correlation *statistic*
  whose magnitude is much smaller than ``rho`` -- e.g. ``rho=0.9`` registers as a
  measured trio statistic near 0.13 for a single-group panel, not 0.9. We report
  that measured statistic throughout as the "realized correlation"; it is the
  quantity the exact solver assumes is zero, not the latent ``rho``.
* Because ``z_j`` is marginally ``N(0, 1)``, ``P(z_j < Phi^-1(acc)) = acc``
  *exactly* per label: the construction is **marginal-accuracy preserving**, so any
  change in recovery is attributable to error *correlation*, not to a confounded
  accuracy shift. ``rho = 0`` reproduces independent errors (the baseline collapse).

Consequences for the four strategies, when the confound rides on the SAME axis the
panel is balanced on (``axis="ideology"``), all emergent (not imposed):

* ``ideological_selection`` -> a single-group trio -> maximal shared confound ->
  NTQR's independence assumption is violated -> recovery degrades as ``rho`` rises.
* ``representative_sortition`` -> a group-balanced trio -> the shared confounds are
  group-specific and decorrelate across members -> recovery is preserved.
* ``random_selection`` -> mixed composition -> intermediate.
* ``expertise_threshold`` -> high accuracy and (since expertise is independent of
  ideology) typically mixed groups -> best.

This robustness is CONDITIONAL, not magical: it holds *iff* the panel is balanced
on the axis the confound rides on. The ``axis`` parameter exists precisely so the
study can run a negative control -- keying the confound on an axis sortition does
NOT balance (e.g. ``axis="expertise_tier"``), where representative sortition is
expected to lose its protection. Whether correlated errors *cost* EIE recovery, and
how much, is left to the exact NTQR solver to answer per regime; this module only
installs the mechanism and measures the realized correlation with NTQR's own
supervised estimator. All randomness is seeded -> deterministic.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from itertools import combinations
from multiprocessing import Pool
from typing import Any, Sequence

import numpy as np

from .dependence import measure_error_correlations
from .experts import Expert, Item, generate_population, sample_items
from .ntqr_eval import (
    closest_solution,
    error_independent_solutions,
    majority_voting_solutions,
    supervised_oracle,
)
from .sortition import STRATEGIES

# Acklam's rational approximation to the inverse standard-normal CDF. Pure
# Python, deterministic, and accurate to ~1.15e-9 over the open interval -- the
# project ships no scipy, and a Gaussian copula needs Phi^-1 to preserve marginals.
_A = (
    -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
    1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
)
_B = (
    -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
    6.680131188771972e01, -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
    -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
)
_D = (
    7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
    3.754408661907416e00,
)
_P_LOW = 0.02425
_P_HIGH = 1.0 - _P_LOW


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (quantile function) for ``p`` in (0, 1)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in the open interval (0, 1)")
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    if p <= _P_HIGH:
        q = p - 0.5
        r = q * q
        return (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / (
            ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
        (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
    )


#: Confound grouping axes the sampler understands (attributes on :class:`Expert`).
CONFOUND_AXES: tuple[str, ...] = ("ideology", "expertise_tier")


def _group_stream_offset(value: str) -> int:
    """Stable, process-independent integer offset for a group's shared stream.

    Python's built-in ``hash`` is salted per process, so it cannot key a shared
    RNG stream that must agree across ``multiprocessing`` workers. A fixed BLAKE2b
    digest of the group value gives the same offset everywhere and depends on the
    group IDENTITY, never on its rank among the groups present in a given panel.
    """
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def sample_votes_bloc_correlated(
    panel_experts: Sequence[Expert],
    items: Sequence[Item],
    *,
    bloc_correlation: float,
    seed: int,
    axis: str = "ideology",
) -> list[tuple[str, ...]]:
    """Sample panel votes where judges sharing ``axis`` share a latent confound.

    Returns one vote tuple per expert (aligned to ``panel_experts`` order), using
    the Gaussian copula described in the module docstring. ``bloc_correlation`` is
    the latent within-group copula correlation ``rho`` in [0, 1]; judges in
    different ``axis`` groups stay independent. Marginal per-label accuracy is
    preserved exactly. ``axis`` selects the grouping attribute (default
    ``"ideology"``; ``"expertise_tier"`` is the orthogonal negative control).
    """
    if not 0.0 <= bloc_correlation <= 1.0:
        raise ValueError("bloc_correlation must be in [0, 1]")
    if axis not in CONFOUND_AXES:
        raise ValueError(f"axis must be one of {CONFOUND_AXES}; got {axis!r}")
    n = len(items)
    rho = float(bloc_correlation)
    a, b = math.sqrt(rho), math.sqrt(1.0 - rho)

    # One shared standard-normal stream per group, seeded by a STABLE hash of the
    # group value (not its rank), so the same group reuses the same confound across
    # panels and across worker processes.
    groups = {str(getattr(e, axis)) for e in panel_experts}
    shared: dict[str, np.ndarray] = {
        group: np.random.default_rng(seed + _group_stream_offset(group)).standard_normal(n)
        for group in groups
    }

    votes: list[tuple[str, ...]] = []
    for j, e in enumerate(panel_experts):
        eps = np.random.default_rng(seed + 101 * (j + 1)).standard_normal(n)
        z = a * shared[str(getattr(e, axis))] + b * eps
        thr_a = _norm_ppf(float(e.accuracy_a))
        thr_b = _norm_ppf(float(e.accuracy_b))
        row: list[str] = []
        for k, item in enumerate(items):
            if item.true_label == "a":
                correct = z[k] < thr_a
                row.append("a" if correct else "b")
            else:
                correct = z[k] < thr_b
                row.append("b" if correct else "a")
        votes.append(tuple(row))
    return votes


@dataclass(frozen=True)
class BlocTrialResult:
    """One bloc-confound trial: recovery error plus the realized correlation."""

    strategy: str
    bloc_correlation: float
    bias_std: float
    mean_expertise: float
    panel_size: int
    n_items: int
    seed: int
    n_trios: int
    eie_error: float
    mv_error: float
    mean_abs_pair_corr: float
    axis: str = "ideology"


def _scan_trios(
    panel_votes: Sequence[Sequence[str]],
    items: Sequence[Item],
    *,
    n_trios: int,
    max_scan: int,
) -> tuple[list[float], list[float], float]:
    """Scan trios of a panel, scoring up to ``n_trios`` usable ones against oracle.

    Shared by the strategy and concentration trials so both score panels
    identically: deterministic-order scan, skip trios with no physical EIE root,
    measure the realized correlation on the first usable trio. Returns the
    per-trio EIE and majority-vote error lists and that first-trio correlation.
    """
    eie_errors: list[float] = []
    mv_errors: list[float] = []
    first_trio_corr = float("nan")
    for scanned, trio_indices in enumerate(combinations(range(len(panel_votes)), 3)):
        if len(eie_errors) >= n_trios or scanned >= max_scan:
            break
        trio_votes = [panel_votes[i] for i in trio_indices]
        eie_sols = error_independent_solutions(trio_votes)
        mv_sols = majority_voting_solutions(trio_votes)
        if not eie_sols or not mv_sols:
            continue
        oracle = supervised_oracle(trio_votes, items)
        eie_errors.append(closest_solution(eie_sols, oracle).error_vs(oracle))
        mv_errors.append(closest_solution(mv_sols, oracle).error_vs(oracle))
        if math.isnan(first_trio_corr):
            first_trio_corr = measure_error_correlations(trio_votes, items).mean_abs_pair
    return eie_errors, mv_errors, first_trio_corr


def run_bloc_trial(
    *,
    strategy: str,
    bloc_correlation: float,
    panel_size: int = 6,
    n_experts: int = 96,
    n_items: int = 300,
    prevalence_a: float = 0.5,
    mean_expertise: float = 0.72,
    expertise_heterogeneity: float = 0.08,
    bias_std: float = 0.3,
    seed: int = 0,
    n_trios: int = 6,
    max_scan: int = 32,
    axis: str = "ideology",
) -> BlocTrialResult:
    """Population -> panel -> bloc-correlated votes -> NTQR recovery, one cell.

    Mirrors ``pipeline.run_trial_ensemble`` (same population, item, and panel
    seeding offsets) but swaps the independent vote sampler for the bloc-coupled
    one and additionally measures the realized trio error-correlation.
    """
    if panel_size < 3:
        raise ValueError("panel_size must be >= 3 (the exact NTQR solver needs a trio)")
    if strategy not in STRATEGIES:
        raise ValueError(f"invalid strategy: {strategy}")

    population = generate_population(
        n_experts,
        seed=seed,
        mean_expertise=mean_expertise,
        expertise_heterogeneity=expertise_heterogeneity,
        bias_std=bias_std,
    )
    items = sample_items(n_items, prevalence_a=prevalence_a, seed=seed + 7)
    panel = STRATEGIES[strategy](population, panel_size, seed=seed + 13)
    by_id = {e.id: e for e in population}
    panel_experts = [by_id[eid] for eid in panel.expert_ids]
    panel_votes = sample_votes_bloc_correlated(
        panel_experts, items, bloc_correlation=bloc_correlation, seed=seed + 29, axis=axis
    )

    eie_errors, mv_errors, first_trio_corr = _scan_trios(
        panel_votes, items, n_trios=n_trios, max_scan=max_scan
    )

    if not eie_errors:
        return BlocTrialResult(
            strategy=strategy,
            bloc_correlation=float(bloc_correlation),
            bias_std=float(bias_std),
            mean_expertise=float(mean_expertise),
            panel_size=int(panel_size),
            n_items=int(n_items),
            seed=int(seed),
            n_trios=0,
            eie_error=float("nan"),
            mv_error=float("nan"),
            mean_abs_pair_corr=first_trio_corr,
            axis=axis,
        )
    return BlocTrialResult(
        strategy=strategy,
        bloc_correlation=float(bloc_correlation),
        bias_std=float(bias_std),
        mean_expertise=float(mean_expertise),
        panel_size=int(panel_size),
        n_items=int(n_items),
        seed=int(seed),
        n_trios=len(eie_errors),
        eie_error=sum(eie_errors) / len(eie_errors),
        mv_error=sum(mv_errors) / len(mv_errors),
        mean_abs_pair_corr=first_trio_corr,
        axis=axis,
    )


@dataclass(frozen=True)
class BlocPhaseGrid:
    """The bloc-confound phase sweep: strategy x bloc_correlation x regime."""

    strategies: tuple[str, ...]
    bloc_correlations: tuple[float, ...]
    bias_stds: tuple[float, ...]
    mean_expertises: tuple[float, ...]
    panel_sizes: tuple[int, ...]
    seeds: tuple[int, ...]
    n_experts: int = 96
    n_items: int = 300
    prevalence_a: float = 0.5
    expertise_heterogeneity: float = 0.08
    n_trios: int = 6
    axis: str = "ideology"

    def __post_init__(self) -> None:
        axes = {
            "strategies": self.strategies,
            "bloc_correlations": self.bloc_correlations,
            "bias_stds": self.bias_stds,
            "mean_expertises": self.mean_expertises,
            "panel_sizes": self.panel_sizes,
            "seeds": self.seeds,
        }
        for name, values in axes.items():
            if not values:
                raise ValueError(f"{name} must be non-empty")
        for strategy in self.strategies:
            if strategy not in STRATEGIES:
                raise ValueError(f"invalid strategy: {strategy}")
        if self.axis not in CONFOUND_AXES:
            raise ValueError(f"axis must be one of {CONFOUND_AXES}; got {self.axis!r}")

    def tasks(self) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        for strategy in self.strategies:
            for rho in self.bloc_correlations:
                for bias_std in self.bias_stds:
                    for mean_expertise in self.mean_expertises:
                        for panel_size in self.panel_sizes:
                            for seed in self.seeds:
                                tasks.append(
                                    {
                                        "strategy": strategy,
                                        "bloc_correlation": rho,
                                        "bias_std": bias_std,
                                        "mean_expertise": mean_expertise,
                                        "panel_size": panel_size,
                                        "seed": seed,
                                        "n_experts": self.n_experts,
                                        "n_items": self.n_items,
                                        "prevalence_a": self.prevalence_a,
                                        "expertise_heterogeneity": self.expertise_heterogeneity,
                                        "n_trios": self.n_trios,
                                        "axis": self.axis,
                                    }
                                )
        return tasks


def _run_task(task: dict[str, Any]) -> BlocTrialResult:
    return run_bloc_trial(**task)


def run_bloc_phase(grid: BlocPhaseGrid, *, workers: int = 1) -> list[BlocTrialResult]:
    """Run every cell of the phase grid; ``workers`` > 1 parallelizes via a Pool."""
    tasks = grid.tasks()
    if workers <= 1:
        return [_run_task(t) for t in tasks]
    with Pool(processes=workers) as pool:
        return list(pool.imap(_run_task, tasks, chunksize=max(1, len(tasks) // (workers * 8) or 1)))


def concentration_panel(
    population: Sequence[Expert],
    panel_size: int,
    *,
    concentration: float,
    seed: int,
) -> tuple[str, ...]:
    """Form a panel with a tunable degree of single-bloc concentration.

    ``concentration`` in [0, 1] dials representativeness continuously: at
    ``concentration=1`` every seat is drawn from one ideological bloc (the
    single-bloc extreme that ``ideological_selection`` approximates); at
    ``concentration=0`` the seats are filled round-robin across blocs (the balanced
    extreme that ``representative_sortition`` approximates). Intermediate values
    seat ``round(concentration * panel_size)`` members from a single target bloc
    and balance the remainder. Returns the selected expert ids. Deterministic.
    """
    if not 0.0 <= concentration <= 1.0:
        raise ValueError("concentration must be in [0, 1]")
    if panel_size < 3:
        raise ValueError("panel_size must be >= 3 (the exact NTQR solver needs a trio)")
    rng = np.random.default_rng(seed)
    blocs = sorted({e.ideology for e in population})
    by_bloc: dict[str, list[str]] = {b: [e.id for e in population if e.ideology == b] for b in blocs}
    for b in blocs:
        rng.shuffle(by_bloc[b])
    target = blocs[seed % len(blocs)]

    n_conc = max(0, min(panel_size, int(round(concentration * panel_size))))
    chosen: list[str] = list(by_bloc[target][:n_conc])
    taken = set(chosen)
    # Balance the remaining seats round-robin across blocs (target included),
    # so concentration=0 is a fully balanced draw and the dial is monotone.
    pools = {b: [i for i in by_bloc[b] if i not in taken] for b in blocs}
    idx = 0
    while len(chosen) < panel_size and any(pools[b] for b in blocs):
        b = blocs[idx % len(blocs)]
        if pools[b]:
            chosen.append(pools[b].pop(0))
        idx += 1
    return tuple(chosen[:panel_size])


@dataclass(frozen=True)
class ConcentrationTrialResult:
    """One concentration-dial trial at fixed coupling."""

    concentration: float
    bloc_correlation: float
    bias_std: float
    mean_expertise: float
    panel_size: int
    n_items: int
    seed: int
    n_trios: int
    eie_error: float
    mv_error: float
    mean_abs_pair_corr: float


def run_concentration_trial(
    *,
    concentration: float,
    bloc_correlation: float = 0.9,
    panel_size: int = 6,
    n_experts: int = 96,
    n_items: int = 300,
    prevalence_a: float = 0.5,
    mean_expertise: float = 0.72,
    expertise_heterogeneity: float = 0.08,
    bias_std: float = 0.3,
    seed: int = 0,
    n_trios: int = 6,
    max_scan: int = 32,
) -> ConcentrationTrialResult:
    """Population -> concentration-dialled panel -> bloc-correlated votes -> NTQR.

    Mirrors :func:`run_bloc_trial` but forms the panel with
    :func:`concentration_panel` instead of a named strategy, so recovery error can
    be traced against the continuous representativeness dial at fixed coupling.
    """
    population = generate_population(
        n_experts,
        seed=seed,
        mean_expertise=mean_expertise,
        expertise_heterogeneity=expertise_heterogeneity,
        bias_std=bias_std,
    )
    items = sample_items(n_items, prevalence_a=prevalence_a, seed=seed + 7)
    panel_ids = concentration_panel(population, panel_size, concentration=concentration, seed=seed + 13)
    by_id = {e.id: e for e in population}
    panel_experts = [by_id[eid] for eid in panel_ids]
    panel_votes = sample_votes_bloc_correlated(
        panel_experts, items, bloc_correlation=bloc_correlation, seed=seed + 29, axis="ideology"
    )
    eie_errors, mv_errors, first_corr = _scan_trios(panel_votes, items, n_trios=n_trios, max_scan=max_scan)
    n = len(eie_errors)
    return ConcentrationTrialResult(
        concentration=float(concentration),
        bloc_correlation=float(bloc_correlation),
        bias_std=float(bias_std),
        mean_expertise=float(mean_expertise),
        panel_size=int(panel_size),
        n_items=int(n_items),
        seed=int(seed),
        n_trios=n,
        eie_error=(sum(eie_errors) / n) if n else float("nan"),
        mv_error=(sum(mv_errors) / n) if n else float("nan"),
        mean_abs_pair_corr=first_corr,
    )


def run_concentration_sweep(
    *,
    concentrations: Sequence[float],
    bloc_correlation: float = 0.9,
    bias_stds: Sequence[float] = (0.1, 0.3, 0.5),
    mean_expertises: Sequence[float] = (0.65, 0.75),
    panel_size: int = 6,
    seeds: Sequence[int] = tuple(range(24)),
    n_experts: int = 96,
    n_items: int = 300,
    n_trios: int = 6,
    workers: int = 1,
) -> list[ConcentrationTrialResult]:
    """Sweep the representativeness dial at fixed coupling over regimes/seeds."""
    tasks = [
        {
            "concentration": c,
            "bloc_correlation": bloc_correlation,
            "bias_std": b,
            "mean_expertise": m,
            "panel_size": panel_size,
            "seed": s,
            "n_experts": n_experts,
            "n_items": n_items,
            "n_trios": n_trios,
        }
        for c in concentrations
        for b in bias_stds
        for m in mean_expertises
        for s in seeds
    ]
    if workers <= 1:
        return [run_concentration_trial(**t) for t in tasks]
    with Pool(processes=workers) as pool:
        return list(
            pool.imap(_run_concentration_task, tasks, chunksize=max(1, len(tasks) // (workers * 8) or 1))
        )


def _run_concentration_task(task: dict[str, Any]) -> ConcentrationTrialResult:
    return run_concentration_trial(**task)


__all__ = [
    "CONFOUND_AXES",
    "sample_votes_bloc_correlated",
    "BlocTrialResult",
    "run_bloc_trial",
    "BlocPhaseGrid",
    "run_bloc_phase",
    "concentration_panel",
    "ConcentrationTrialResult",
    "run_concentration_trial",
    "run_concentration_sweep",
]
