"""Tests for the per-trio conditioning diagnostic (:mod:`trio_conditioning`).

Discipline mirrors the sibling suites: no mocks, every expected value hand-derived
or cross-checked against an independent shipped function, each behavioural claim
paired with a negative control.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ntqr_allotment.pipeline import TrialConfig, run_trial_ensemble
from ntqr_allotment.trio_conditioning import (
    TrioDiagnostic,
    mean_by_size,
    panel_trio_diagnostics,
    pearson,
    summarize_trio_diagnostics,
)
from scripts.run_trio_conditioning import _write_summary


def _cfg(strategy: str, size: int, seed: int = 3) -> TrialConfig:
    return TrialConfig(
        strategy=strategy,
        panel_size=size,
        n_experts=40,
        n_items=120,
        prevalence_a=0.5,
        mean_expertise=0.72,
        expertise_heterogeneity=0.08,
        bias_std=0.2,
        seed=seed,
    )


def test_per_trio_errors_average_exactly_to_the_ensemble_error() -> None:
    """The diagnostic must walk the SAME usable trios the ensemble averages.

    This is the load-bearing consistency check: the mean of the per-trio
    ``eie_error`` values equals :func:`run_trial_ensemble`'s ``eie_error`` to
    machine precision, and the count matches ``n_trios``. If the diagnostic walked
    different trios (different order, different degeneracy handling) this fails.
    """
    cfg = _cfg("random_selection", 6)
    recs = panel_trio_diagnostics(cfg, n_trios=8)
    ens = run_trial_ensemble(cfg, n_trios=8)

    assert len(recs) == ens.n_trios
    assert recs  # this config is not all-degenerate
    mean_err = sum(r.eie_error for r in recs) / len(recs)
    assert mean_err == pytest.approx(ens.eie_error, abs=1e-9)
    # trio_rank is the dense 0-based scan position.
    assert [r.trio_rank for r in recs] == list(range(len(recs)))
    # Every record carries real, non-negative recovery error and an |corr| in [0,1].
    for r in recs:
        assert r.eie_error >= 0.0
        assert 0.0 <= r.mean_abs_corr <= 1.0
        assert 0.0 <= r.mean_judge_accuracy <= 1.0


def test_size_three_panel_yields_a_single_rank_zero_trio() -> None:
    cfg = _cfg("expertise_threshold", 3)
    recs = panel_trio_diagnostics(cfg, n_trios=8)
    assert len(recs) == 1
    assert recs[0].trio_rank == 0
    assert recs[0].panel_size == 3


def test_n_trios_cap_is_respected() -> None:
    # NEGATIVE CONTROL: asking for 2 usable trios from a size-9 panel returns 2,
    # not the full C(9,3)=84 scan.
    cfg = _cfg("random_selection", 9)
    recs = panel_trio_diagnostics(cfg, n_trios=2)
    assert len(recs) <= 2


def test_panel_size_below_three_and_bad_n_trios_raise() -> None:
    with pytest.raises(ValueError, match="panel_size"):
        panel_trio_diagnostics(_cfg("random_selection", 2), n_trios=4)
    with pytest.raises(ValueError, match="n_trios"):
        panel_trio_diagnostics(_cfg("random_selection", 6), n_trios=0)


def test_mean_by_size_groups_and_averages() -> None:
    recs = [
        TrioDiagnostic("s", 3, 0, 0.7, 0.2, 0, 0.40, 0.01, 0.7),
        TrioDiagnostic("s", 3, 1, 0.7, 0.2, 0, 0.60, 0.03, 0.7),
        TrioDiagnostic("s", 6, 0, 0.7, 0.2, 0, 0.50, 0.05, 0.7),
    ]
    assert mean_by_size(recs, "eie_error") == {3: pytest.approx(0.50), 6: pytest.approx(0.50)}
    assert mean_by_size(recs, "mean_abs_corr") == {3: pytest.approx(0.02), 6: pytest.approx(0.05)}


def test_pearson_known_values_and_constant_series() -> None:
    assert pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    assert pearson([1.0, 2.0, 3.0], [6.0, 4.0, 2.0]) == pytest.approx(-1.0)
    # NEGATIVE CONTROL: a constant series has undefined correlation -> 0.0, not a crash.
    assert pearson([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]) == 0.0
    with pytest.raises(ValueError, match="length >= 2"):
        pearson([1.0], [2.0])
    with pytest.raises(ValueError, match="length >= 2"):
        pearson([], [])


def test_empty_trio_summary_serializes_and_renders_unavailable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary = summarize_trio_diagnostics([], n_seeds=12)
    assert summary["n_records"] == 0
    assert summary["sizes"] == []
    assert summary["mean_error_by_size"] == {}
    assert summary["mean_abs_corr_by_size"] == {}
    assert summary["mean_judge_accuracy_by_size"] == {}
    assert summary["mean_error_first_trio"] is None
    assert summary["mean_error_later_trios"] is None
    assert summary["associations_pearson"] == {
        "error_vs_corr": None,
        "error_vs_accuracy": None,
        "error_vs_trio_rank": None,
    }
    out = tmp_path / "empty.json"
    _write_summary(out, [], n_seeds=12)
    assert json.loads(out.read_text(encoding="utf-8")) == summary
    assert "unavailable" in capsys.readouterr().out


@pytest.mark.parametrize("rank", [0, 1])
def test_single_trio_keeps_missing_rank_partition_and_associations_null(
    rank: int, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = TrioDiagnostic("s", 3, 0, 0.7, 0.2, rank, 0.4, 0.01, 0.7)
    summary = summarize_trio_diagnostics([record], n_seeds=1)
    assert summary["n_records"] == 1
    assert summary["mean_error_by_size"] == {3: 0.4}
    assert summary["mean_error_first_trio"] == (0.4 if rank == 0 else None)
    assert summary["mean_error_later_trios"] == (0.4 if rank > 0 else None)
    assert all(value is None for value in summary["associations_pearson"].values())
    out = tmp_path / f"rank-{rank}.json"
    _write_summary(out, [record], n_seeds=1)
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["mean_error_first_trio"] == summary["mean_error_first_trio"]
    assert saved["mean_error_later_trios"] == summary["mean_error_later_trios"]
    assert saved["associations_pearson"] == summary["associations_pearson"]
    assert "unavailable" in capsys.readouterr().out


def test_summary_preserves_measured_associations_and_size_and_rank_means() -> None:
    records = [
        TrioDiagnostic("s", 3, 0, 0.7, 0.2, 0, 0.2, 0.01, 0.8),
        TrioDiagnostic("s", 6, 1, 0.7, 0.2, 1, 0.4, 0.03, 0.6),
    ]
    summary = summarize_trio_diagnostics(records, n_seeds=2)
    assert summary["sizes"] == [3, 6]
    assert summary["mean_error_by_size"] == {3: 0.2, 6: 0.4}
    assert summary["mean_abs_corr_by_size"] == {3: 0.01, 6: 0.03}
    assert summary["mean_judge_accuracy_by_size"] == {3: 0.8, 6: 0.6}
    assert summary["mean_error_first_trio"] == 0.2
    assert summary["mean_error_later_trios"] == 0.4
    assert summary["associations_pearson"] == {
        "error_vs_corr": pytest.approx(1.0),
        "error_vs_accuracy": pytest.approx(-1.0),
        "error_vs_trio_rank": pytest.approx(1.0),
    }
