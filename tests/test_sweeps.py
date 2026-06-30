from __future__ import annotations

from math import prod

import pytest

from ntqr_allotment.pipeline import TrialConfig, run_trial, run_trial_ensemble
from ntqr_allotment.sweeps import (
    DEGENERATE_ERROR,
    Aggregate,
    SweepGrid,
    SweepRow,
    aggregate,
    representative_vs_ideological,
    run_sweep,
    run_sweep_parallel,
    strategy_ranking,
)
from ntqr_allotment.sweeps import _weighted_mean_and_ci


def _small_grid() -> SweepGrid:
    return SweepGrid(
        strategies=("representative_sortition", "random_selection", "ideological_selection"),
        panel_sizes=(3, 6),
        mean_expertises=(0.70, 0.74),
        expertise_heterogeneities=(0.08,),
        bias_stds=(0.25,),
        n_items_values=(80,),
        prevalence_as=(0.5,),
        seeds=(0, 1),
        n_experts=36,
        n_trios=3,
    )


def test_grid_size():
    grid = _small_grid()
    expected_cells = prod(
        [
            len(grid.strategies),
            len(grid.panel_sizes),
            len(grid.mean_expertises),
            len(grid.expertise_heterogeneities),
            len(grid.bias_stds),
            len(grid.n_items_values),
            len(grid.prevalence_as),
        ]
    )
    assert len(grid.cells()) == expected_cells
    assert len(run_sweep(grid)) == expected_cells * len(grid.seeds)


def test_determinism():
    grid = _small_grid()
    assert run_sweep(grid) == run_sweep(grid)


def test_parallel_sweep_matches_serial_small_grid():
    grid = SweepGrid(
        strategies=("random_selection", "ideological_selection"),
        panel_sizes=(3,),
        mean_expertises=(0.70,),
        expertise_heterogeneities=(0.08,),
        bias_stds=(0.25,),
        n_items_values=(50,),
        prevalence_as=(0.5,),
        seeds=(0, 1),
        n_experts=24,
        n_trios=1,
    )
    assert run_sweep_parallel(grid, workers=2) == run_sweep(grid)


def test_parallel_sweep_rejects_invalid_worker_count():
    with pytest.raises(ValueError, match="workers must be >= 1"):
        run_sweep_parallel(_small_grid(), workers=0)


def test_aggregate_matches_manual():
    grid = SweepGrid(
        strategies=("random_selection",),
        panel_sizes=(6,),
        mean_expertises=(0.72,),
        expertise_heterogeneities=(0.08,),
        bias_stds=(0.30,),
        n_items_values=(90,),
        prevalence_as=(0.5,),
        seeds=(0, 1),
        n_experts=32,
        n_trios=2,
    )
    rows = run_sweep(grid)
    grouped = [row.eie_error for row in rows if row.config_key() == rows[0].config_key()]
    manual_mean = sum(grouped) / len(grouped)
    manual_std = ((sum((value - manual_mean) ** 2 for value in grouped) / (len(grouped) - 1)) ** 0.5)
    manual_ci95 = manual_std / (len(grouped) ** 0.5) * 1.96

    item = aggregate(rows)[0]
    assert item.eie_mean == pytest.approx(manual_mean, abs=1e-9)
    assert item.eie_std == pytest.approx(manual_std, abs=1e-9)
    assert item.eie_ci95 == pytest.approx(manual_ci95, abs=1e-9)


def test_strategy_ranking_shape():
    aggregated = aggregate(run_sweep(_small_grid()))
    ranking = strategy_ranking(aggregated)
    assert {strategy for strategy, _, _ in ranking} == {
        "representative_sortition",
        "random_selection",
        "ideological_selection",
    }
    assert all(ranking[index][1] <= ranking[index + 1][1] for index in range(len(ranking) - 1))


def test_ensemble_degeneracy_panel3():
    config = TrialConfig(
        strategy="random_selection",
        panel_size=3,
        n_experts=30,
        n_items=100,
        seed=2,
    )
    ensemble = run_trial_ensemble(config, n_trios=5)
    single = run_trial(config)
    assert ensemble.eie_error == single.eie_error
    assert ensemble.mv_error == single.mv_error


def test_rep_vs_ideo_is_data_driven():
    base_key = ("representative_sortition", 6, 0.72, 0.08, 0.30, 90, 0.5, 32)
    rep = Aggregate(
        config_key=base_key,
        strategy="representative_sortition",
        n=2,
        eie_mean=0.12,
        eie_std=0.01,
        eie_ci95=0.02,
        mv_mean=0.10,
        mv_std=0.01,
        mv_ci95=0.02,
    )
    ideo_low = Aggregate(
        config_key=("ideological_selection",) + base_key[1:],
        strategy="ideological_selection",
        n=2,
        eie_mean=0.10,
        eie_std=0.01,
        eie_ci95=0.03,
        mv_mean=0.11,
        mv_std=0.01,
        mv_ci95=0.03,
    )
    ideo_high = Aggregate(
        config_key=("ideological_selection",) + base_key[1:],
        strategy="ideological_selection",
        n=2,
        eie_mean=0.40,
        eie_std=0.01,
        eie_ci95=0.03,
        mv_mean=0.11,
        mv_std=0.01,
        mv_ci95=0.03,
    )

    low_effect = representative_vs_ideological([rep, ideo_low])[0]
    high_effect = representative_vs_ideological([rep, ideo_high])[0]
    assert low_effect.effect != high_effect.effect
    assert low_effect.effect == pytest.approx(-0.02, abs=1e-12)
    assert high_effect.effect == pytest.approx(0.28, abs=1e-12)


# ---------------------------------------------------------------------------
# SweepGrid.__post_init__ validation (lines 43, 46, 48, 50)
# Each test pairs a failing construction with a corrected one that builds fine,
# so the assertion flips (no error) if the guard were removed: a negative
# control proving the guard fires ONLY on the bad value.
# ---------------------------------------------------------------------------

def _grid_kwargs() -> dict:
    """Valid kwargs for a minimal SweepGrid; mutate one field per test."""
    return dict(
        strategies=("random_selection",),
        panel_sizes=(3,),
        mean_expertises=(0.72,),
        expertise_heterogeneities=(0.08,),
        bias_stds=(0.30,),
        n_items_values=(90,),
        prevalence_as=(0.5,),
        seeds=(0,),
        n_experts=32,
        n_trios=2,
    )


def test_empty_axis_raises():
    kwargs = _grid_kwargs()
    kwargs["seeds"] = ()
    with pytest.raises(ValueError, match="seeds must be non-empty"):
        SweepGrid(**kwargs)
    # negative control: with a non-empty axis the guard does not fire
    kwargs["seeds"] = (0,)
    assert SweepGrid(**kwargs).seeds == (0,)


def test_invalid_strategy_raises():
    kwargs = _grid_kwargs()
    kwargs["strategies"] = ("not_a_real_strategy",)
    with pytest.raises(ValueError, match="invalid strategies value: not_a_real_strategy"):
        SweepGrid(**kwargs)
    # negative control: a registered strategy builds fine
    kwargs["strategies"] = ("random_selection",)
    assert SweepGrid(**kwargs).strategies == ("random_selection",)


def test_n_experts_nonpositive_raises():
    kwargs = _grid_kwargs()
    kwargs["n_experts"] = 0
    with pytest.raises(ValueError, match="n_experts must be positive"):
        SweepGrid(**kwargs)
    # negative control: a positive count builds fine
    kwargs["n_experts"] = 1
    assert SweepGrid(**kwargs).n_experts == 1


def test_n_trios_below_one_raises():
    kwargs = _grid_kwargs()
    kwargs["n_trios"] = 0
    with pytest.raises(ValueError, match="n_trios must be >= 1"):
        SweepGrid(**kwargs)
    # negative control: n_trios == 1 builds fine
    kwargs["n_trios"] = 1
    assert SweepGrid(**kwargs).n_trios == 1


# ---------------------------------------------------------------------------
# SweepRow.regime_key (line 109)
# ---------------------------------------------------------------------------

def _sweep_row(strategy: str = "random_selection", *, panel_size: int = 6,
               eie: float = 0.12, mv: float = 0.10, n_trios: int = 2) -> SweepRow:
    return SweepRow(
        strategy=strategy,
        panel_size=panel_size,
        mean_expertise=0.72,
        expertise_heterogeneity=0.08,
        bias_std=0.30,
        n_items=90,
        prevalence_a=0.5,
        n_experts=32,
        seed=0,
        n_trios=n_trios,
        eie_error=eie,
        mv_error=mv,
    )


def test_sweep_row_regime_key_drops_strategy():
    row = _sweep_row()
    # regime_key is config_key minus the leading strategy element
    assert row.regime_key() == row.config_key()[1:]
    assert row.strategy not in row.regime_key()
    # negative control: two rows differing ONLY in strategy share a regime_key
    other = _sweep_row(strategy="representative_sortition")
    assert other.regime_key() == row.regime_key()
    assert other.config_key() != row.config_key()


# ---------------------------------------------------------------------------
# run_sweep degenerate path when run_trial_ensemble raises ValueError
# (lines 177-179). A panel_size < 3 cell builds a valid TrialConfig (no
# construction-time guard) but run_trial_ensemble raises -> degenerate row.
# ---------------------------------------------------------------------------

def test_run_sweep_degenerate_on_ensemble_value_error():
    bad_grid = SweepGrid(
        strategies=("random_selection",),
        panel_sizes=(2,),  # < 3 -> run_trial_ensemble raises ValueError
        mean_expertises=(0.72,),
        expertise_heterogeneities=(0.08,),
        bias_stds=(0.30,),
        n_items_values=(90,),
        prevalence_as=(0.5,),
        seeds=(0,),
        n_experts=32,
        n_trios=2,
    )
    rows = run_sweep(bad_grid)
    assert len(rows) == 1
    assert rows[0].eie_error == DEGENERATE_ERROR
    assert rows[0].mv_error == DEGENERATE_ERROR
    assert rows[0].n_trios == 0
    # negative control: a panel_size >= 3 cell yields a real, non-degenerate row
    good_grid = SweepGrid(
        strategies=("random_selection",),
        panel_sizes=(6,),
        mean_expertises=(0.72,),
        expertise_heterogeneities=(0.08,),
        bias_stds=(0.30,),
        n_items_values=(90,),
        prevalence_as=(0.5,),
        seeds=(0,),
        n_experts=32,
        n_trios=2,
    )
    good_rows = run_sweep(good_grid)
    assert good_rows[0].eie_error != DEGENERATE_ERROR


# ---------------------------------------------------------------------------
# aggregate -> _mean_std_ci all-degenerate -> DEGENERATE (line 225)
# strategy_ranking n==0 filtering (line 277->276)
# ---------------------------------------------------------------------------

def test_aggregate_all_degenerate_rows_returns_degenerate_sentinel():
    deg = SweepRow(
        strategy="random_selection",
        panel_size=2,
        mean_expertise=0.72,
        expertise_heterogeneity=0.08,
        bias_std=0.30,
        n_items=90,
        prevalence_a=0.5,
        n_experts=32,
        seed=0,
        n_trios=0,
        eie_error=DEGENERATE_ERROR,
        mv_error=DEGENERATE_ERROR,
    )
    aggs = aggregate([deg])
    assert len(aggs) == 1
    assert aggs[0].n == 0
    assert aggs[0].eie_mean == DEGENERATE_ERROR
    assert aggs[0].eie_std == 0.0
    assert aggs[0].eie_ci95 == 0.0
    # negative control: a real row aggregates to a finite, non-sentinel mean
    real = _sweep_row(eie=0.20, mv=0.18)
    real_agg = aggregate([real])[0]
    assert real_agg.n == 1
    assert real_agg.eie_mean == pytest.approx(0.20, abs=1e-12)


def test_strategy_ranking_skips_zero_n_aggregates():
    deg_agg = Aggregate(
        config_key=("random_selection", 2, 0.72, 0.08, 0.30, 90, 0.5, 32),
        strategy="random_selection",
        n=0,
        eie_mean=DEGENERATE_ERROR,
        eie_std=0.0,
        eie_ci95=0.0,
        mv_mean=DEGENERATE_ERROR,
        mv_std=0.0,
        mv_ci95=0.0,
    )
    # only a zero-n aggregate -> filtered out -> empty ranking
    assert strategy_ranking([deg_agg]) == []
    # negative control: an n>0 aggregate of the same strategy IS ranked
    good_agg = Aggregate(
        config_key=("random_selection", 6, 0.72, 0.08, 0.30, 90, 0.5, 32),
        strategy="random_selection",
        n=2,
        eie_mean=0.12,
        eie_std=0.01,
        eie_ci95=0.02,
        mv_mean=0.10,
        mv_std=0.01,
        mv_ci95=0.02,
    )
    ranking = strategy_ranking([deg_agg, good_agg])
    assert ranking == [("random_selection", pytest.approx(0.12, abs=1e-12), pytest.approx(0.0, abs=1e-12))]


# ---------------------------------------------------------------------------
# _weighted_mean_and_ci zero-weight raise (line 263) and single-weight
# branch (line 266). Reached publicly via strategy_ranking, but the zero-total
# branch is only reachable by calling the helper directly with empty weights.
# ---------------------------------------------------------------------------

def test_weighted_mean_zero_total_weight_raises():
    with pytest.raises(ValueError, match="weights must sum to a positive integer"):
        _weighted_mean_and_ci([0.5], [0])
    # negative control: a positive weight does not raise
    mean, ci95 = _weighted_mean_and_ci([0.5], [1])
    assert mean == pytest.approx(0.5, abs=1e-12)


def test_weighted_mean_single_weight_zero_ci():
    mean, ci95 = _weighted_mean_and_ci([0.30], [1])
    assert mean == pytest.approx(0.30, abs=1e-12)
    assert ci95 == 0.0  # total_weight <= 1 -> no spread reported
    # negative control: total_weight > 1 produces a non-degenerate CI
    mean2, ci2 = _weighted_mean_and_ci([0.10, 0.30], [1, 2])
    assert ci2 > 0.0


# ---------------------------------------------------------------------------
# representative_vs_ideological: missing strategy (line 302) and n==0 skip
# (line 304).
# ---------------------------------------------------------------------------

def _agg(strategy: str, *, n: int, eie_mean: float) -> Aggregate:
    return Aggregate(
        config_key=(strategy, 6, 0.72, 0.08, 0.30, 90, 0.5, 32),
        strategy=strategy,
        n=n,
        eie_mean=eie_mean,
        eie_std=0.01,
        eie_ci95=0.02,
        mv_mean=0.10,
        mv_std=0.01,
        mv_ci95=0.02,
    )


def test_rep_vs_ideo_skips_missing_strategy():
    rep_only = _agg("representative_sortition", n=2, eie_mean=0.12)
    # ideological_selection absent for this regime -> skipped (line 302)
    assert representative_vs_ideological([rep_only]) == []
    # negative control: add the ideological aggregate -> one effect emitted
    ideo = _agg("ideological_selection", n=2, eie_mean=0.20)
    effects = representative_vs_ideological([rep_only, ideo])
    assert len(effects) == 1
    assert effects[0].effect == pytest.approx(0.08, abs=1e-12)


def test_rep_vs_ideo_skips_zero_n_strategy():
    rep = _agg("representative_sortition", n=2, eie_mean=0.12)
    ideo_zero = _agg("ideological_selection", n=0, eie_mean=DEGENERATE_ERROR)
    # ideological has n==0 -> regime skipped (line 304)
    assert representative_vs_ideological([rep, ideo_zero]) == []
    # negative control: both strategies with n>0 -> effect emitted
    ideo_real = _agg("ideological_selection", n=2, eie_mean=0.20)
    assert len(representative_vs_ideological([rep, ideo_real])) == 1
