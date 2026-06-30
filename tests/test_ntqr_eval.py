"""Tests for the NTQR adapter (no mocks; real ntqr algebra)."""

from __future__ import annotations

import sympy
import pytest

from ntqr_allotment.experts import Item, generate_population, sample_items
from ntqr_allotment.ntqr_eval import (
    Evaluation,
    _real_or_none,
    _to_evaluation,
    alarm_misaligned,
    closest_solution,
    error_independent_solutions,
    majority_voting_solutions,
    supervised_oracle,
    trio_label_vote_counts,
    trio_vote_counts,
)
from ntqr_allotment.pipeline import votes_for

# A sympy value that ``is_real is True`` yet overflows to inf when cast to float.
# (sympy.Float("inf"), sympy.nan, and sympy.oo all report is_real != True, so they
# do NOT exercise the real-but-non-finite guard; this construction does.)
_REAL_BUT_INFINITE = sympy.Float(sympy.Integer(10) ** 400)


@pytest.fixture
def trio_votes_and_items():
    pop = generate_population(30, seed=1)
    items = sample_items(400, prevalence_a=0.5, seed=2)
    votes = votes_for(pop[:3], items, seed=9)
    return votes, items


def test_trio_vote_counts_sum_equals_n_items(trio_votes_and_items):
    votes, items = trio_votes_and_items
    counts = trio_vote_counts(votes)
    assert sum(counts.values()) == len(items)
    assert len(counts) == 8  # 2^3 patterns


def test_trio_vote_counts_requires_three_judges():
    with pytest.raises(ValueError):
        trio_vote_counts([("a", "b"), ("a", "b")])


def test_trio_vote_counts_requires_equal_lengths():
    with pytest.raises(ValueError):
        trio_vote_counts([("a", "b"), ("a", "b"), ("a",)])


def test_trio_label_vote_counts_split_sums(trio_votes_and_items):
    votes, items = trio_votes_and_items
    counts = trio_label_vote_counts(votes, items)
    total = sum(counts["a"].values()) + sum(counts["b"].values())
    assert total == len(items)


def test_trio_label_vote_counts_requires_three_judges(trio_votes_and_items):
    _, items = trio_votes_and_items
    with pytest.raises(ValueError):
        trio_label_vote_counts([("a",), ("b",)], items)


def test_error_independent_returns_two_solutions(trio_votes_and_items):
    votes, _ = trio_votes_and_items
    sols = error_independent_solutions(votes)
    assert len(sols) == 2
    assert all(isinstance(s, Evaluation) for s in sols)


def test_majority_voting_returns_two_solutions(trio_votes_and_items):
    votes, _ = trio_votes_and_items
    sols = majority_voting_solutions(votes)
    assert len(sols) == 2


def test_supervised_oracle_recovers_true_prevalence(trio_votes_and_items):
    votes, items = trio_votes_and_items
    oracle = supervised_oracle(votes, items)
    true_prev = sum(1 for it in items if it.true_label == "a") / len(items)
    assert oracle.prevalence_a == pytest.approx(true_prev, abs=1e-6)


def test_closest_solution_picks_nearest():
    oracle = Evaluation(prevalence_a=0.5, accuracies=(0.8, 0.8, 0.8))
    near = Evaluation(prevalence_a=0.52, accuracies=(0.78, 0.81, 0.79))
    far = Evaluation(prevalence_a=0.1, accuracies=(0.2, 0.2, 0.2))
    assert closest_solution([far, near], oracle) is near


def test_evaluation_error_vs_is_zero_for_identical():
    e = Evaluation(prevalence_a=0.4, accuracies=(0.7, 0.6))
    assert e.error_vs(e) == pytest.approx(0.0)


def test_unsupervised_estimate_is_near_oracle(trio_votes_and_items):
    votes, items = trio_votes_and_items
    oracle = supervised_oracle(votes, items)
    best = closest_solution(error_independent_solutions(votes), oracle)
    # error-independent judges -> the exact solver should land close to truth
    assert best.error_vs(oracle) < 0.15


def test_alarm_runs_at_small_q(trio_votes_and_items):
    votes, _ = trio_votes_and_items
    capped = [v[:20] for v in votes]
    result = alarm_misaligned(capped, max_q=25)
    assert isinstance(result, bool)


def test_alarm_guards_against_large_q(trio_votes_and_items):
    votes, _ = trio_votes_and_items
    with pytest.raises(ValueError):
        alarm_misaligned([v[:100] for v in votes], max_q=30)


def test_closest_solution_empty_raises():
    oracle = Evaluation(prevalence_a=0.5, accuracies=(0.8, 0.8, 0.8))
    with pytest.raises(ValueError):
        closest_solution([], oracle)


# --- _real_or_none degenerate-root branches (real sympy values, no mocks) -------


def test_real_or_none_real_but_non_finite_is_none():
    """Covers the real-valued-but-NaN/inf guard: is_real is True yet float() == inf."""
    # Sanity: this value really does report as real to sympy.
    assert sympy.sympify(_REAL_BUT_INFINITE).is_real is True
    assert _real_or_none(_REAL_BUT_INFINITE) is None


def test_real_or_none_imag_outside_tolerance_is_none():
    """Covers the complex-root guard: imaginary part exceeds _REAL_TOLERANCE -> None."""
    # imag 0.3 >> 1e-9 tolerance; not a real solution.
    assert _real_or_none(complex(0.5, 0.3)) is None
    # sympy's imaginary unit (im == 1.0) takes the same path.
    assert _real_or_none(sympy.sqrt(-1)) is None


def test_real_or_none_imag_within_tolerance_but_real_part_non_finite_is_none():
    """Covers: imag within tolerance, but the real part is non-finite -> None."""
    expr = _REAL_BUT_INFINITE + sympy.Float("1e-12") * sympy.I
    # is_real is not True (it carries a tiny imaginary part) so it enters the
    # complex branch; imag 1e-12 < tolerance, real part is inf.
    assert expr.is_real is not True
    assert float(sympy.N(sympy.im(expr))) < 1e-9
    assert _real_or_none(expr) is None


def test_real_or_none_clean_real_passes_through():
    """Negative control: an ordinary finite real survives both guards."""
    assert _real_or_none(0.42) == pytest.approx(0.42)
    # A complex value with negligible imaginary part returns its real part.
    assert _real_or_none(complex(0.42, 1e-12)) == pytest.approx(0.42)


# --- _to_evaluation degenerate-solution branches -------------------------------


def test_to_evaluation_non_real_prevalence_is_none():
    """Covers: a solution whose prevalence is non-real -> the whole solution is None."""
    sol = {
        "prevalence": {"a": complex(0.5, 0.3)},
        "accuracy": [
            {"a": 0.8, "b": 0.8},
            {"a": 0.7, "b": 0.7},
            {"a": 0.6, "b": 0.6},
        ],
    }
    assert _to_evaluation(sol) is None


def test_to_evaluation_non_real_judge_accuracy_is_none():
    """Covers: real prevalence but a non-real per-judge accuracy -> None."""
    sol = {
        "prevalence": {"a": 0.5},
        "accuracy": [
            {"a": complex(0.5, 0.3), "b": 0.8},  # judge 0 accuracy_a is complex
            {"a": 0.7, "b": 0.7},
            {"a": 0.6, "b": 0.6},
        ],
    }
    assert _to_evaluation(sol) is None


def test_to_evaluation_all_real_returns_evaluation():
    """Negative control: a fully real solution yields a concrete Evaluation."""
    sol = {
        "prevalence": {"a": 0.5},
        "accuracy": [
            {"a": 0.8, "b": 0.8},
            {"a": 0.7, "b": 0.7},
            {"a": 0.6, "b": 0.6},
        ],
    }
    ev = _to_evaluation(sol)
    assert isinstance(ev, Evaluation)
    assert ev.prevalence_a == pytest.approx(0.5)
    assert ev.accuracies == pytest.approx((0.8, 0.7, 0.6))


# --- supervised_oracle degenerate-trio guard -----------------------------------


def test_supervised_oracle_degenerate_trio_raises():
    """Covers the line-163 guard end-to-end via a genuinely degenerate labeled trio.

    When every item shares one label and all three judges cast the identical vote,
    the real NTQR supervised solver returns a degenerate (non-finite) solution; the
    oracle must fail loudly rather than fabricate a value.
    """
    items = [Item(index=i, true_label="a") for i in range(4)]
    votes = [["a", "a", "a", "a"]] * 3
    with pytest.raises(ValueError, match="degenerate"):
        supervised_oracle(votes, items)


# --------------------------------------------------------------------------- #
# Negative controls for the under-verified solver layer (RedTeam oracle audit)
# --------------------------------------------------------------------------- #
def test_error_vs_isolates_prevalence_and_accuracy_terms() -> None:
    """error_vs must sum BOTH a prevalence term and a mean per-judge term.

    Dropping either term (e.g. ``return acc_err``) passes the >=0/zero-distance
    checks but fails here. The prevalence-only and accuracy-only fixtures isolate
    each term; the combined fixture pins the exact composite.
    """
    # Same accuracies -> only the prevalence term survives.
    assert Evaluation(0.5, (0.8,)).error_vs(Evaluation(0.3, (0.8,))) == pytest.approx(0.2)
    # Same prevalence -> only the mean accuracy term survives.
    assert Evaluation(0.5, (0.9,)).error_vs(Evaluation(0.5, (0.6,))) == pytest.approx(0.3)
    # Combined: |0.5-0.3| + (|0.9-0.8| + |0.7-0.5|)/2 = 0.2 + 0.15 = 0.35.
    assert Evaluation(0.5, (0.9, 0.7)).error_vs(
        Evaluation(0.3, (0.8, 0.5))
    ) == pytest.approx(0.35)


def test_to_evaluation_balances_class_conditional_accuracies() -> None:
    """Per-judge accuracy is the BALANCED 0.5*(acc_a+acc_b), not acc_a alone.

    The only other _to_evaluation test feeds acc_a == acc_b, so a mutation that
    keeps just acc_a (dropping class-b accuracy) passes it; this fixture with
    acc_a != acc_b catches that and pins the published per-judge accuracy.
    """
    ev = _to_evaluation({"prevalence": {"a": 0.5}, "accuracy": [{"a": 0.9, "b": 0.5}]})
    assert ev is not None
    assert ev.accuracies[0] == pytest.approx(0.7)  # 0.5*(0.9+0.5), NOT 0.9


def test_to_evaluation_rejects_non_physical_solutions() -> None:
    """A real but out-of-[0,1] root is not a valid evaluation -> None (degenerate).

    Prevalence and accuracies are probabilities; the solver can return real roots
    like accuracy 5.5 or prevalence 1.4, which are impossible and previously scored
    as huge oracle distances (the heavy-tailed error outliers). NEGATIVE CONTROL:
    an all-physical solution still converts, and a value a hair over 1 (float noise)
    is tolerated.
    """
    # accuracy 5.525 (the real defect observed in the sweep) -> rejected.
    assert _to_evaluation({"prevalence": {"a": 0.4}, "accuracy": [{"a": 5.525, "b": 0.7}]}) is None
    # prevalence outside [0,1] -> rejected.
    assert _to_evaluation({"prevalence": {"a": 1.4}, "accuracy": [{"a": 0.6, "b": 0.6}]}) is None
    # a negative accuracy -> rejected.
    assert _to_evaluation({"prevalence": {"a": 0.5}, "accuracy": [{"a": -0.2, "b": 0.6}]}) is None
    # all-physical -> still a valid Evaluation.
    assert _to_evaluation({"prevalence": {"a": 0.5}, "accuracy": [{"a": 0.8, "b": 0.6}]}) is not None
    # float-noise hair over 1.0 -> tolerated (not rejected).
    assert _to_evaluation({"prevalence": {"a": 1.0 + 1e-9}, "accuracy": [{"a": 1.0, "b": 0.6}]}) is not None


def test_majority_voting_solutions_differ_from_error_independent() -> None:
    """MV and EIE are DISTINCT estimators; the manuscript's headline metric is
    ``eie_mean - mv_mean``. A mutation collapsing MV onto EIE (the crippled
    comparator) would make this metric ~0 everywhere; this trio (both estimators
    return real, non-empty solutions with different prevalences) fails on it.
    """
    # A trio where both estimators yield PHYSICAL (prevalence/accuracy in [0,1])
    # solutions with different prevalences (eie {0.444, 0.556} vs mv {0.417, 0.583}).
    votes = [
        ("b", "a", "a", "b", "b", "a", "b", "b", "b", "a", "b", "b"),
        ("a", "a", "a", "a", "a", "a", "a", "b", "b", "a", "b", "b"),
        ("b", "b", "b", "a", "a", "b", "a", "a", "a", "a", "a", "a"),
    ]
    eie = error_independent_solutions(votes)
    mv = majority_voting_solutions(votes)
    assert eie and mv  # both estimators yield real, physical solutions on this trio

    def _key(sols: list[Evaluation]) -> list[tuple[float, tuple[float, ...]]]:
        return sorted(
            (round(s.prevalence_a, 6), tuple(round(a, 6) for a in s.accuracies))
            for s in sols
        )

    assert _key(eie) != _key(mv)  # estimators are not the same object/algebra
    eie_prev = sorted(round(s.prevalence_a, 4) for s in eie)
    mv_prev = sorted(round(s.prevalence_a, 4) for s in mv)
    assert eie_prev == [0.4444, 0.5556]
    assert mv_prev == [0.4167, 0.5833]
    assert eie_prev != mv_prev
