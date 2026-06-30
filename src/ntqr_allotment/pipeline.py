"""End-to-end trial: population -> panel formation -> NTQR evaluation -> metrics.

A *trial* is the atomic unit the parameter sweep repeats. It answers, for one
configuration: given this population and this way of forming the panel, what is
the oracle-referenced error of the no-answer-key `ntqr` evaluation, and is the
panel jointly consistent (the alarm)?

The exact NTQR solver needs a trio, so the evaluated trio is the first three
experts the strategy selects -- meaning the *strategy's own ordering* (stratified
vs single-bloc vs competence-first) is exactly what is being tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

from .experts import Expert, Item, generate_population, sample_items, sample_votes
from .ntqr_eval import (
    Evaluation,
    alarm_misaligned,
    closest_solution,
    error_independent_solutions,
    majority_voting_solutions,
    supervised_oracle,
)
from .sortition import STRATEGIES, PanelDraw

#: Cap on how many candidate trios the ensemble scans per panel. Generous enough
#: that a well-conditioned panel finds its ``n_trios`` usable trios well within it,
#: but bounds the O(C(n,3)) sympy cost on heavily ill-conditioned panels whose trios
#: mostly yield only non-physical roots (which the [0,1] filter now rejects).
_DEFAULT_MAX_TRIO_SCAN = 32


@dataclass(frozen=True)
class TrialConfig:
    """One point in the parameter space."""

    strategy: str
    panel_size: int = 3
    n_experts: int = 60
    n_items: int = 200
    prevalence_a: float = 0.5
    mean_expertise: float = 0.72
    expertise_heterogeneity: float = 0.08
    bias_std: float = 0.3
    seed: int = 0
    compute_alarm: bool = False
    alarm_q: int = 25  # corpus prefix used for the (opt-in, O(Q^3)) alarm


@dataclass(frozen=True)
class TrialResult:
    """Metrics from one trial."""

    config: TrialConfig
    panel: PanelDraw
    oracle: Evaluation
    eie_estimate: Evaluation
    mv_estimate: Evaluation
    eie_error: float
    mv_error: float
    alarm_misaligned: bool | None


@dataclass(frozen=True)
class EnsembleTrialResult:
    config: TrialConfig
    panel: PanelDraw
    n_trios: int
    eie_error: float
    mv_error: float


def votes_for(experts: Sequence[Expert], items: Sequence[Item], *, seed: int) -> list[tuple[str, ...]]:
    """Sample each expert's votes over the corpus with per-expert seed offsets."""
    return [sample_votes(e, items, seed=seed + 1000 * (i + 1)) for i, e in enumerate(experts)]


def run_trial(config: TrialConfig) -> TrialResult:
    """Run one full population -> panel -> NTQR trial and return its metrics."""
    if config.panel_size < 3:
        raise ValueError("panel_size must be >= 3 (the exact NTQR solver needs a trio)")

    population = generate_population(
        config.n_experts,
        seed=config.seed,
        mean_expertise=config.mean_expertise,
        expertise_heterogeneity=config.expertise_heterogeneity,
        bias_std=config.bias_std,
    )
    items = sample_items(config.n_items, prevalence_a=config.prevalence_a, seed=config.seed + 7)

    strategy_fn = STRATEGIES[config.strategy]
    panel = strategy_fn(population, config.panel_size, seed=config.seed + 13)

    by_id = {e.id: e for e in population}
    panel_experts = [by_id[eid] for eid in panel.expert_ids]
    panel_votes = votes_for(panel_experts, items, seed=config.seed + 29)

    trio_votes = panel_votes[:3]

    oracle = supervised_oracle(trio_votes, items)
    eie = closest_solution(error_independent_solutions(trio_votes), oracle)
    mv = closest_solution(majority_voting_solutions(trio_votes), oracle)

    alarm: bool | None = None
    if config.compute_alarm:
        capped = [v[: config.alarm_q] for v in panel_votes]
        alarm = alarm_misaligned(capped, max_q=config.alarm_q)

    return TrialResult(
        config=config,
        panel=panel,
        oracle=oracle,
        eie_estimate=eie,
        mv_estimate=mv,
        eie_error=eie.error_vs(oracle),
        mv_error=mv.error_vs(oracle),
        alarm_misaligned=alarm,
    )


def run_trial_ensemble(
    config: TrialConfig, *, n_trios: int = 5, max_scan: int = _DEFAULT_MAX_TRIO_SCAN
) -> EnsembleTrialResult:
    if config.panel_size < 3:
        raise ValueError("panel_size must be >= 3 (the exact NTQR solver needs a trio)")
    if n_trios < 1:
        raise ValueError("n_trios must be >= 1")
    if max_scan < n_trios:
        raise ValueError("max_scan must be >= n_trios")

    population = generate_population(
        config.n_experts,
        seed=config.seed,
        mean_expertise=config.mean_expertise,
        expertise_heterogeneity=config.expertise_heterogeneity,
        bias_std=config.bias_std,
    )
    items = sample_items(config.n_items, prevalence_a=config.prevalence_a, seed=config.seed + 7)

    strategy_fn = STRATEGIES[config.strategy]
    panel = strategy_fn(population, config.panel_size, seed=config.seed + 13)

    by_id = {e.id: e for e in population}
    panel_experts = [by_id[eid] for eid in panel.expert_ids]
    panel_votes = votes_for(panel_experts, items, seed=config.seed + 29)

    # Scan combinations in deterministic order, collecting up to ``n_trios``
    # USABLE trios. A degenerate trio (no real, *physical* error-independent
    # solution for its vote pattern) is skipped and the scan continues, so a single
    # bad expert does not starve the ensemble while it still averages over the
    # requested number of well-posed trios. The scan is bounded to the first
    # ``max_scan`` candidate trios: a heavily ill-conditioned panel (many trios with
    # only non-physical roots) would otherwise force an O(C(n,3)) sympy sweep, and
    # bounding it both keeps the cost predictable and down-weights pathological
    # panels that cannot form ``n_trios`` well-posed trios from their first
    # ``max_scan`` candidates.
    eie_errors: list[float] = []
    mv_errors: list[float] = []
    for scanned, trio_indices in enumerate(combinations(range(len(panel_experts)), 3)):
        if len(eie_errors) >= n_trios or scanned >= max_scan:
            break
        trio_votes = [panel_votes[index] for index in trio_indices]
        eie_sols = error_independent_solutions(trio_votes)
        mv_sols = majority_voting_solutions(trio_votes)
        if not eie_sols or not mv_sols:
            continue  # degenerate trio: skip honestly, do not score it
        oracle = supervised_oracle(trio_votes, items)
        eie = closest_solution(eie_sols, oracle)
        mv = closest_solution(mv_sols, oracle)
        eie_errors.append(eie.error_vs(oracle))
        mv_errors.append(mv.error_vs(oracle))

    if not eie_errors:
        # Every trio in the whole panel is degenerate: the unsupervised
        # estimate is undefined for this cell. Record an honest NaN (n_trios=0)
        # so a sweep can surface "no recovery possible here" rather than crash.
        return EnsembleTrialResult(
            config=config,
            panel=panel,
            n_trios=0,
            eie_error=float("nan"),
            mv_error=float("nan"),
        )

    return EnsembleTrialResult(
        config=config,
        panel=panel,
        n_trios=len(eie_errors),
        eie_error=sum(eie_errors) / len(eie_errors),
        mv_error=sum(mv_errors) / len(mv_errors),
    )


__all__ = [
    "TrialConfig",
    "TrialResult",
    "EnsembleTrialResult",
    "votes_for",
    "run_trial",
    "run_trial_ensemble",
]
