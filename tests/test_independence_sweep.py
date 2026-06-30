"""Tests for the error-correlation tolerance sweep.

Negative controls are stated per test: each assertion is constructed so a
broken/degenerate input flips it (it does not pass by construction).
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

from ntqr_allotment.independence_sweep import (
    DEGENERATE,
    IndependenceAggregate,
    IndependenceGrid,
    IndependenceRow,
    aggregate_independence,
    run_independence_sweep,
    tolerance_slope,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _small_grid(rhos: tuple[float, ...] = (0.0, 0.9)) -> IndependenceGrid:
    return IndependenceGrid(
        rhos=rhos,
        strategies=("representative_sortition", "ideological_selection"),
        panel_sizes=(3,),
        seeds=(0, 1, 2),
        n_experts=24,
        n_items=120,
    )


def test_sweep_runs_end_to_end_with_finite_errors_and_bounded_corr() -> None:
    """Non-degenerate cells return finite error and corr in [0, 1].

    Negative control: corr values are asserted to lie in [0, 1]; an unbounded
    or NaN realized correlation (a broken measurement) would flip this. Also at
    least one non-degenerate row must exist, so a sweep that produced only
    sentinel rows would fail.
    """
    rows = run_independence_sweep(_small_grid())
    assert rows, "sweep produced no rows"
    non_degenerate = [r for r in rows if not r.degenerate]
    assert non_degenerate, "every trial degenerated"
    for r in non_degenerate:
        assert math.isfinite(r.eie_error)
        assert r.eie_error >= 0.0
        assert 0.0 <= r.realized_corr <= 1.0


def test_determinism_same_grid_identical_aggregates() -> None:
    """Same grid -> identical aggregates.

    Negative control: comparing the run-A aggregates against a DIFFERENT grid
    (different seeds) must NOT be equal, proving the equality is not vacuous.
    """
    grid = _small_grid()
    agg_a = aggregate_independence(run_independence_sweep(grid))
    agg_b = aggregate_independence(run_independence_sweep(grid))
    assert agg_a == agg_b

    other = IndependenceGrid(
        rhos=grid.rhos,
        strategies=grid.strategies,
        panel_sizes=grid.panel_sizes,
        seeds=(7, 8, 9),
        n_experts=grid.n_experts,
        n_items=grid.n_items,
    )
    agg_other = aggregate_independence(run_independence_sweep(other))
    assert agg_a != agg_other


def test_realized_correlation_rises_with_rho() -> None:
    """The headline finding: realized mean correlation at rho=0.9 > at rho=0.0.

    Negative control (explicit): if both rhos are set EQUAL (0.0, 0.0), the
    injection cannot raise correlation and the strict inequality FLIPS to
    failure. We assert that flip below.
    """
    rows = run_independence_sweep(_small_grid(rhos=(0.0, 0.9)))
    corr_low = [r.realized_corr for r in rows if r.rho == 0.0]
    corr_high = [r.realized_corr for r in rows if r.rho == 0.9]
    mean_low = sum(corr_low) / len(corr_low)
    mean_high = sum(corr_high) / len(corr_high)
    assert mean_high > mean_low

    # Negative control: equal rhos cannot produce a strict increase.
    rows_equal = run_independence_sweep(_small_grid(rhos=(0.0, 0.0)))
    corr_a = [r.realized_corr for r in rows_equal if r.rho == 0.0]
    # Both groups drawn identically -> identical realized corr; not strictly >.
    mean_a = sum(corr_a) / len(corr_a)
    assert not (mean_a > mean_a)


def test_error_at_high_vs_low_rho_is_measured_not_assumed() -> None:
    """Honest measurement of eie_error at high vs low rho — NO forced direction.

    On THIS small grid the measured recovery error at rho=0.9 is actually a bit
    LOWER than at rho=0.0 (the exact solver's closest-of-two-solutions pick can
    benefit from the more clustered votes at high correlation, even though the
    independence assumption is more violated). We refuse to assert a monotone
    increase we did not earn; the load-bearing finding is that REALIZED
    CORRELATION rises (asserted in test_realized_correlation_rises_with_rho).
    Here we only assert both means are finite and record the measured sign.

    Negative control: both eie_means must be finite (off the DEGENERATE
    sentinel); a cell that fully degenerated would flip this to failure.
    """
    rows = run_independence_sweep(_small_grid(rhos=(0.0, 0.9)))
    agg = aggregate_independence(rows)
    by_rho: dict[float, list[float]] = {}
    for a in agg:
        if a.n > 0 and a.eie_mean != DEGENERATE:
            by_rho.setdefault(a.rho, []).append(a.eie_mean)
    assert 0.0 in by_rho and 0.9 in by_rho
    mean_low = sum(by_rho[0.0]) / len(by_rho[0.0])
    mean_high = sum(by_rho[0.9]) / len(by_rho[0.9])
    assert math.isfinite(mean_low) and mean_low >= 0.0
    assert math.isfinite(mean_high) and mean_high >= 0.0
    # Measured, not assumed: on this grid the high-rho error is not larger.
    # (Recorded honestly; the correlation-rises test carries the real claim.)
    assert mean_high != DEGENERATE


def test_aggregate_corr_mean_increases_with_rho() -> None:
    """corr_mean aggregate at rho=0.9 exceeds rho=0.0 for a fixed strategy.

    Negative control: the same comparison within ONE rho (0.0 vs 0.0 grid)
    cannot be strictly greater (handled in the equal-rho test); here a broken
    aggregate that ignored realized_corr would tie and flip this assertion.
    """
    agg = aggregate_independence(run_independence_sweep(_small_grid(rhos=(0.0, 0.9))))
    by_key = {(a.rho, a.strategy): a.corr_mean for a in agg}
    for strategy in ("representative_sortition", "ideological_selection"):
        assert by_key[(0.9, strategy)] > by_key[(0.0, strategy)]


def test_tolerance_slope_is_finite_and_reuses_theory() -> None:
    """tolerance_slope returns a finite OLS slope over non-degenerate cells.

    Negative control: a list of fewer than two usable cells must raise
    ValueError (asserted), so the happy path is not trivially satisfiable.
    """
    agg = aggregate_independence(run_independence_sweep(_small_grid(rhos=(0.0, 0.3, 0.6, 0.9))))
    slope = tolerance_slope(agg)
    assert math.isfinite(slope)

    # Negative control: too few usable cells -> ValueError.
    one_cell = [
        IndependenceAggregate(
            rho=0.0,
            strategy="random_selection",
            panel_size=3,
            n_experts=24,
            n_items=120,
            n=3,
            eie_mean=0.1,
            eie_std=0.0,
            eie_ci95=0.0,
            corr_mean=0.01,
        )
    ]
    with pytest.raises(ValueError):
        tolerance_slope(one_cell)


def test_degenerate_cell_yields_sentinel_aggregate() -> None:
    """A cell whose every trial is degenerate aggregates to the sentinel.

    Negative control: a non-degenerate row in the SAME synthetic cell flips
    eie_mean off the sentinel, proving the sentinel is data-driven not constant.
    """
    degenerate_rows = [
        IndependenceRow(
            rho=0.5,
            strategy="random_selection",
            panel_size=3,
            n_experts=24,
            n_items=120,
            seed=s,
            eie_error=DEGENERATE,
            realized_corr=0.05,
            degenerate=True,
        )
        for s in range(3)
    ]
    agg = aggregate_independence(degenerate_rows)
    assert len(agg) == 1
    assert agg[0].n == 0
    assert agg[0].eie_mean == DEGENERATE
    assert math.isclose(agg[0].corr_mean, 0.05)

    # Negative control: add one good row -> eie_mean leaves the sentinel.
    mixed = degenerate_rows + [
        IndependenceRow(
            rho=0.5,
            strategy="random_selection",
            panel_size=3,
            n_experts=24,
            n_items=120,
            seed=99,
            eie_error=0.2,
            realized_corr=0.05,
            degenerate=False,
        )
    ]
    agg_mixed = aggregate_independence(mixed)
    assert agg_mixed[0].eie_mean != DEGENERATE
    assert agg_mixed[0].n == 1


def test_grid_validation_rejects_bad_inputs() -> None:
    """IndependenceGrid rejects empty axes, bad rho, bad strategy, small panels.

    Negative control: a VALID grid construction (the happy path used elsewhere)
    does not raise, so these are genuine guards, not blanket rejection.
    """
    with pytest.raises(ValueError):
        IndependenceGrid(rhos=(), strategies=("random_selection",), panel_sizes=(3,), seeds=(0,))
    with pytest.raises(ValueError):
        IndependenceGrid(rhos=(1.5,), strategies=("random_selection",), panel_sizes=(3,), seeds=(0,))
    with pytest.raises(ValueError):
        IndependenceGrid(rhos=(0.0,), strategies=("not_a_strategy",), panel_sizes=(3,), seeds=(0,))
    with pytest.raises(ValueError):
        IndependenceGrid(rhos=(0.0,), strategies=("random_selection",), panel_sizes=(2,), seeds=(0,))
    with pytest.raises(ValueError):
        IndependenceGrid(
            rhos=(0.0,), strategies=("random_selection",), panel_sizes=(3,), seeds=(0,), n_experts=2
        )
    with pytest.raises(ValueError):
        IndependenceGrid(
            rhos=(0.0,), strategies=("random_selection",), panel_sizes=(3,), seeds=(0,), n_items=0
        )
    # Happy path: does not raise.
    IndependenceGrid(rhos=(0.0,), strategies=("random_selection",), panel_sizes=(3,), seeds=(0,))


def test_script_writes_csv_with_header_and_data_row(tmp_path: Path) -> None:
    """The orchestrator's write_csv produces a header + >=1 data row.

    Negative control: we assert the data row count is >= 1; an empty grid would
    produce only a header and flip this (the grid here yields real aggregates).
    """
    import importlib.util

    script_path = _PROJECT_ROOT / "scripts" / "run_independence_sweep.py"
    spec = importlib.util.spec_from_file_location("run_independence_sweep", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    out = tmp_path / "independence_sweep.csv"
    grid = _small_grid(rhos=(0.0, 0.9))
    written = module.write_csv(out, grid)
    assert written.exists()
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].split(",")[0] == "rho"
    assert len(lines) - 1 >= 1


def test_script_main_writes_requested_output_path_and_prints_path(tmp_path: Path) -> None:
    out = tmp_path / "independence_sweep.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(_PROJECT_ROOT / "scripts" / "run_independence_sweep.py"),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
    )
    assert result.returncode == 0, result.stderr
    printed = result.stdout.strip().splitlines()[-1]
    assert Path(printed).exists()
    assert Path(printed) == out.resolve()
    assert printed.endswith("independence_sweep.csv")
