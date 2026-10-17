"""Reject invalid study designs before they can masquerade as empirical failures."""

from dataclasses import replace
import math
import signal
import os
import subprocess
import sys

import pytest

from ntqr_allotment.experts import generate_population
from ntqr_allotment.ntqr_eval import NoPhysicalSolutionError, closest_solution
from ntqr_allotment.pipeline import TrialConfig
from ntqr_allotment.sortition import STRATEGIES, build_pool
from ntqr_allotment.sweeps import SweepGrid, aggregate, run_sweep, SweepRow


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("size", [0, -1, 5, True, 1.5])
def test_selection_never_silently_returns_partial_or_invalid_panels(strategy, size):
    with pytest.raises(ValueError, match="panel_size"):
        STRATEGIES[strategy](generate_population(4, seed=0), size, seed=1)


def test_duplicate_expert_identity_is_rejected():
    population = generate_population(4, seed=0)
    population[-1] = replace(population[-1], id=population[0].id)
    with pytest.raises(ValueError, match="unique"):
        build_pool(population)


@pytest.mark.parametrize(
    "updates",
    [
        {"prevalence_a": math.nan},
        {"bias_std": math.inf},
        {"n_items": 0},
        {"panel_size": 61},
        {"strategy": "missing"},
        {"seed": -1},
        {"mean_expertise": 1.1},
        {"expertise_heterogeneity": -0.1},
    ],
)
def test_trial_configuration_rejects_invalid_values(updates):
    with pytest.raises(ValueError):
        TrialConfig(**{"strategy": "random_selection", **updates})


def grid():
    return SweepGrid(
        **dict(
            strategies=("random_selection",),
            panel_sizes=(3,),
            mean_expertises=(0.72,),
            expertise_heterogeneities=(0.08,),
            bias_stds=(0.3,),
            n_items_values=(30,),
            prevalence_as=(0.5,),
            seeds=(0,),
            n_experts=12,
            n_trios=1,
        )
    )


def test_duplicate_seeds_cannot_inflate_replication_count():
    kwargs = dict(
        strategies=("random_selection",),
        panel_sizes=(3,),
        mean_expertises=(0.72,),
        expertise_heterogeneities=(0.08,),
        bias_stds=(0.3,),
        n_items_values=(30,),
        prevalence_as=(0.5,),
        seeds=(0, 0),
    )
    with pytest.raises(ValueError, match="duplicate"):
        SweepGrid(**kwargs)


@pytest.mark.parametrize("seconds", [0, -1, math.nan, math.inf, True])
def test_invalid_timeout_is_not_reported_as_solver_degeneracy(seconds):
    with pytest.raises(ValueError, match="trial_timeout_s"):
        run_sweep(grid(), trial_timeout_s=seconds)


def test_real_sweep_timeout_has_explicit_failure_reason():
    original_handler = signal.getsignal(signal.SIGALRM)
    rows = run_sweep(grid(), trial_timeout_s=0.000001)
    assert signal.getsignal(signal.SIGALRM) == original_handler
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    assert len(rows) == 1
    assert rows[0].failure_reason == "timeout"
    assert rows[0].n_trios == 0


def test_only_physical_solver_failure_has_domain_exception():
    with pytest.raises(NoPhysicalSolutionError):
        closest_solution([], None)


def test_aggregation_uses_the_same_eligible_rows_for_both_methods():
    cfg = dict(
        strategy="random_selection",
        panel_size=3,
        mean_expertise=0.72,
        expertise_heterogeneity=0.08,
        bias_std=0.3,
        n_items=30,
        prevalence_a=0.5,
        n_experts=12,
        seed=0,
        n_trios=1,
    )
    valid = SweepRow(**cfg, eie_error=0.1, mv_error=0.2)
    partial = SweepRow(**{**cfg, "seed": 1}, eie_error=0.9, mv_error=math.nan)
    result = aggregate([valid, partial])[0]
    assert result.n == 1
    assert result.eie_mean == pytest.approx(0.1)
    assert result.mv_mean == pytest.approx(0.2)


def test_explicit_empty_solver_override_is_rejected_in_a_real_process():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from ntqr_allotment.solver import configure_cbc; configure_cbc()",
        ],
        env={**os.environ, "NTQR_CBC_PATH": ""},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "NTQR_CBC_PATH" in result.stderr


@pytest.mark.parametrize("factors", [(1, 2), (2, 1), (1, 1)])
def test_impossible_safety_boundary_matches_real_ntqr_enumeration(factors):
    from ntqr import Labels
    from ntqr.alarms import SingleClassifierAxiomsAlarm
    from ntqr.r2.evaluations import SingleClassifierEvaluations
    from ntqr.r2.raxioms import SingleClassifierAxioms
    from ntqr_allotment.ntqr_eval import alarm_misaligned

    labels = Labels(("a", "b"))
    alarm = SingleClassifierAxiomsAlarm(
        2,
        [SingleClassifierAxioms(labels, i) for i in range(2)],
        SingleClassifierEvaluations,
    )
    alarm.set_safety_specification(factors)
    assert alarm.are_misaligned([[1, 1], [2, 0]]) is True
    assert alarm_misaligned([("a", "b"), ("a", "a")], safety_factors=factors) is True


@pytest.mark.parametrize(
    "updates",
    [
        {"n_trios": True},
        {"n_trios": 1.5},
        {"max_scan": True},
        {"max_scan": 1.5},
        {"max_scan": 0},
    ],
)
def test_direct_ensemble_rejects_invalid_trio_and_scan_counts(updates):
    from ntqr_allotment.pipeline import run_trial_ensemble

    with pytest.raises(ValueError, match="integer"):
        run_trial_ensemble(TrialConfig(strategy="random_selection"), **updates)
