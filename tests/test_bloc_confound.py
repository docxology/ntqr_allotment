"""Tests for the bloc-confound channel (no mocks; real numerics).

These lock the three properties the scientific claim rests on:
1. the Gaussian copula preserves marginal accuracy (separation is about
   correlation, not a confounded accuracy shift);
2. same-ideology trios correlate while cross-ideology trios do not;
3. at zero coupling the strategies collapse, and the
   ideological-minus-representative gap grows as coupling rises (the fan-out).
"""

from __future__ import annotations

import math

import pytest

from ntqr_allotment.bloc_confound import (
    BlocPhaseGrid,
    _norm_ppf,
    concentration_panel,
    run_bloc_trial,
    run_concentration_trial,
    sample_votes_bloc_correlated,
)
from ntqr_allotment.dependence import measure_error_correlations
from ntqr_allotment.experts import generate_population, sample_items


def test_norm_ppf_matches_known_quantiles() -> None:
    cases = {0.5: 0.0, 0.975: 1.959963985, 0.95: 1.644853627, 0.025: -1.959963985, 0.1: -1.281551566}
    for p, expected in cases.items():
        assert abs(_norm_ppf(p) - expected) < 1e-6


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_norm_ppf_rejects_out_of_range(bad: float) -> None:
    with pytest.raises(ValueError):
        _norm_ppf(bad)


def test_bloc_correlation_out_of_range_raises() -> None:
    pop = generate_population(12, seed=0)
    items = sample_items(20, prevalence_a=0.5, seed=1)
    with pytest.raises(ValueError):
        sample_votes_bloc_correlated(pop[:3], items, bloc_correlation=1.5, seed=0)


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
def test_marginal_accuracy_preserved_per_label(rho: float) -> None:
    """Empirical PER-LABEL accuracy tracks intended acc_a/acc_b for any rho.

    Checking the mean alone would pass a thr_a<->thr_b swap at prevalence 0.5;
    the copula preserves each label-conditional accuracy separately, so the test
    must too.
    """
    pop = generate_population(48, seed=0, mean_expertise=0.74, bias_std=0.6)
    items = sample_items(8000, prevalence_a=0.5, seed=7)
    trio = pop[:3]
    votes = sample_votes_bloc_correlated(trio, items, bloc_correlation=rho, seed=29)
    for expert, vote in zip(trio, votes):
        a_items = [(v, it) for v, it in zip(vote, items) if it.true_label == "a"]
        b_items = [(v, it) for v, it in zip(vote, items) if it.true_label == "b"]
        emp_a = sum(v == "a" for v, _ in a_items) / len(a_items)
        emp_b = sum(v == "b" for v, _ in b_items) / len(b_items)
        assert abs(emp_a - expert.accuracy_a) < 0.025
        assert abs(emp_b - expert.accuracy_b) < 0.025


def test_confound_keyed_on_group_identity_not_rank() -> None:
    """A group's shared confound depends on its IDENTITY, not its rank among the
    groups present in a panel (the stable-hash keying). At rho=1 a judge's votes
    are driven entirely by its group stream, so the same expert votes identically
    regardless of which other groups share its panel."""
    pop = generate_population(96, seed=5)
    items = sample_items(400, prevalence_a=0.5, seed=7)
    right = next(e for e in pop if e.ideology == "right")
    center = next(e for e in pop if e.ideology == "center")
    # Panel A: {center, right} -> 'right' is the higher-ranked group.
    votes_a = sample_votes_bloc_correlated([center, right], items, bloc_correlation=1.0, seed=29)
    # Panel B: {right} only -> 'right' is now the only/lowest-ranked group.
    votes_b = sample_votes_bloc_correlated([right], items, bloc_correlation=1.0, seed=29)
    assert votes_a[1] == votes_b[0]  # rank-keying would make these differ


def test_same_bloc_correlates_more_than_cross_bloc() -> None:
    pop = generate_population(96, seed=3, mean_expertise=0.72, bias_std=0.3)
    items = sample_items(4000, prevalence_a=0.5, seed=7)
    first = pop[0].ideology
    same = [e for e in pop if e.ideology == first][:3]
    cross, seen = [], set()
    for e in pop:
        if e.ideology not in seen:
            cross.append(e)
            seen.add(e.ideology)
        if len(cross) == 3:
            break
    same_corr = measure_error_correlations(
        sample_votes_bloc_correlated(same, items, bloc_correlation=0.8, seed=29), items
    ).mean_abs_pair
    cross_corr = measure_error_correlations(
        sample_votes_bloc_correlated(cross, items, bloc_correlation=0.8, seed=29), items
    ).mean_abs_pair
    assert same_corr > 5 * cross_corr
    assert cross_corr < 0.02


def test_zero_coupling_is_near_independent() -> None:
    pop = generate_population(96, seed=1, mean_expertise=0.72, bias_std=0.3)
    items = sample_items(4000, prevalence_a=0.5, seed=7)
    same = [e for e in pop if e.ideology == pop[0].ideology][:3]
    corr = measure_error_correlations(
        sample_votes_bloc_correlated(same, items, bloc_correlation=0.0, seed=29), items
    ).mean_abs_pair
    assert corr < 0.02


def test_sampler_is_deterministic() -> None:
    pop = generate_population(24, seed=2)
    items = sample_items(100, prevalence_a=0.5, seed=3)
    a = sample_votes_bloc_correlated(pop[:4], items, bloc_correlation=0.6, seed=11)
    b = sample_votes_bloc_correlated(pop[:4], items, bloc_correlation=0.6, seed=11)
    assert a == b


def test_run_bloc_trial_returns_finite_recovery() -> None:
    r = run_bloc_trial(
        strategy="representative_sortition",
        bloc_correlation=0.5,
        panel_size=6,
        n_items=150,
        seed=0,
        n_trios=3,
    )
    assert r.n_trios > 0
    assert math.isfinite(r.eie_error) and 0.0 <= r.eie_error <= 2.0
    assert r.strategy == "representative_sortition"


def test_run_bloc_trial_rejects_small_panel() -> None:
    with pytest.raises(ValueError):
        run_bloc_trial(strategy="random_selection", bloc_correlation=0.0, panel_size=2)


def test_fanout_separation_grows_with_coupling() -> None:
    """The headline: ideological degrades relative to representative as coupling
    rises, while at rho=0 the two are close (the reproduced collapse).

    Asserted via the robust correlation MECHANISM and a PAIRED rise-differential
    (each strategy compared to itself across coupling, which cancels its baseline
    noise), not via absolute single-config error point estimates. The exact EIE
    fan-out magnitude is the manuscript's job, measured at scale with CIs by
    ``scripts/run_bloc_phase.py``; here we only need the direction + mechanism,
    kept robust to the borderline-degenerate-trio jitter the NTQR solver shows
    under load (see the BLAS-nondeterminism note in project memory)."""
    seeds = range(16)

    def agg(strategy: str, rho: float) -> tuple[float, float]:
        results = [
            run_bloc_trial(
                strategy=strategy,
                bloc_correlation=rho,
                panel_size=6,
                mean_expertise=0.70,
                bias_std=0.4,
                n_items=300,
                seed=s,
                n_trios=4,
            )
            for s in seeds
        ]
        eie = sum(r.eie_error for r in results) / len(results)
        corr = [r.mean_abs_pair_corr for r in results if r.mean_abs_pair_corr == r.mean_abs_pair_corr]
        return eie, (sum(corr) / len(corr))

    ideo_lo_eie, ideo_lo_corr = agg("ideological_selection", 0.0)
    rep_lo_eie, rep_lo_corr = agg("representative_sortition", 0.0)
    ideo_hi_eie, ideo_hi_corr = agg("ideological_selection", 0.9)
    rep_hi_eie, rep_hi_corr = agg("representative_sortition", 0.9)

    # Paired fan-out: ideological's error rises with coupling far more than the
    # bloc-balanced panel's does (each strategy vs itself -> baseline noise cancels).
    ideo_rise = ideo_hi_eie - ideo_lo_eie
    rep_rise = rep_hi_eie - rep_lo_eie
    assert ideo_rise > rep_rise + 0.03
    # The mechanism (rock-solid ~5x gap): at zero coupling both panels are
    # near-independent; at high coupling the single-bloc panel carries far more
    # measured error-correlation than the bloc-balanced one.
    assert ideo_hi_corr > 0.05
    assert ideo_hi_corr > 2.0 * rep_hi_corr
    assert ideo_lo_corr < 0.04 and rep_lo_corr < 0.04
    assert ideo_hi_corr > 2.5 * rep_hi_corr


def test_axis_validation() -> None:
    pop = generate_population(12, seed=0)
    items = sample_items(20, prevalence_a=0.5, seed=1)
    with pytest.raises(ValueError):
        sample_votes_bloc_correlated(pop[:3], items, bloc_correlation=0.5, seed=0, axis="nonsense")


def test_negative_control_representative_protection_is_axis_conditional() -> None:
    """The CRITICAL guard against the axis tautology. Representative sortition is
    not innately robust to correlated error: it decorrelates a shared confound only
    when it balances the axis the confound rides on. Keyed on ideology (the axis it
    balances), a representative panel stays nearly decorrelated at high coupling;
    keyed on expertise tier (an axis it does NOT balance), the same panel can no
    longer break up the shared shock, so its measured error-correlation rises
    substantially. The protection is conditional, not magical."""
    seeds = range(8)

    def rep_corr(axis: str) -> float:
        vals = [
            run_bloc_trial(
                strategy="representative_sortition",
                bloc_correlation=0.9,
                axis=axis,
                panel_size=6,
                mean_expertise=0.70,
                bias_std=0.4,
                n_items=200,
                seed=s,
                n_trios=4,
            ).mean_abs_pair_corr
            for s in seeds
        ]
        finite = [v for v in vals if v == v]
        return sum(finite) / len(finite)

    corr_matched = rep_corr("ideology")
    corr_orthogonal = rep_corr("expertise_tier")
    # Representative loses its decorrelation when the confound rides on an axis it
    # does not balance: its measured correlation climbs by a clear margin.
    assert corr_orthogonal > 1.4 * corr_matched


def test_concentration_panel_extremes_and_validation() -> None:
    pop = generate_population(96, seed=0)
    by_id = {e.id: e for e in pop}
    full = concentration_panel(pop, 6, concentration=1.0, seed=13)
    balanced = concentration_panel(pop, 6, concentration=0.0, seed=13)
    assert len(full) == 6 and len(balanced) == 6
    # Fully concentrated -> one ideological bloc; balanced -> multiple blocs.
    assert len({by_id[i].ideology for i in full}) == 1
    assert len({by_id[i].ideology for i in balanced}) >= 2
    with pytest.raises(ValueError):
        concentration_panel(pop, 6, concentration=1.5, seed=0)
    with pytest.raises(ValueError):
        concentration_panel(pop, 2, concentration=0.5, seed=0)


def test_concentration_dial_error_and_correlation_rise_with_concentration() -> None:
    """The graded law: at fixed coupling, the more single-bloc-concentrated the
    panel, the higher its measured error-correlation and its recovery error."""
    seeds = range(12)

    def agg(concentration: float) -> tuple[float, float]:
        rs = [
            run_concentration_trial(
                concentration=concentration,
                bloc_correlation=0.9,
                panel_size=6,
                mean_expertise=0.70,
                bias_std=0.4,
                n_items=200,
                seed=s,
                n_trios=4,
            )
            for s in seeds
        ]
        eie = sum(r.eie_error for r in rs) / len(rs)
        corr = [r.mean_abs_pair_corr for r in rs if r.mean_abs_pair_corr == r.mean_abs_pair_corr]
        return eie, sum(corr) / len(corr)

    eie_bal, corr_bal = agg(0.0)
    eie_conc, corr_conc = agg(1.0)
    assert eie_conc > eie_bal + 0.02
    assert corr_conc > 2.0 * corr_bal


def test_phase_grid_validation_and_task_count() -> None:
    grid = BlocPhaseGrid(
        strategies=("representative_sortition", "ideological_selection"),
        bloc_correlations=(0.0, 0.9),
        bias_stds=(0.3,),
        mean_expertises=(0.72,),
        panel_sizes=(6,),
        seeds=(0, 1, 2),
    )
    assert len(grid.tasks()) == 2 * 2 * 1 * 1 * 1 * 3
    with pytest.raises(ValueError):
        BlocPhaseGrid(
            strategies=(),
            bloc_correlations=(0.0,),
            bias_stds=(0.3,),
            mean_expertises=(0.72,),
            panel_sizes=(6,),
            seeds=(0,),
        )
    with pytest.raises(ValueError):
        BlocPhaseGrid(
            strategies=("not_a_strategy",),
            bloc_correlations=(0.0,),
            bias_stds=(0.3,),
            mean_expertises=(0.72,),
            panel_sizes=(6,),
            seeds=(0,),
        )
