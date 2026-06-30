"""Analytical complements to the Monte Carlo NTQR/sortition study.

This module adds a small theory layer on top of the existing simulation code:

1. The single-classifier NTQR axiom is exposed as explicit SymPy expressions
   that vanish on a consistent answer key.
2. The exact trio solver's two-solution ambiguity is stated symbolically and
   checked on a deterministic worked example drawn from the existing generator
   and dependence modules.
3. The expected effect of positive error-correlation on NTQR recovery error is
   encoded as a monotone prediction and compared against an empirical slope.
4. The alarm's answer-key enumeration count is written in closed form to make
   the observed cubic growth explicit.

The functions are pure symbolic/numeric helpers. They import existing project
modules where the simulation logic already lives and do not duplicate it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import sympy
from ntqr import Labels
from ntqr.r2.raxioms import SingleClassifierAxioms

from .dependence import sample_votes_correlated
from .experts import generate_population, sample_items
from .ntqr_eval import error_independent_solutions

_TWO_SOLUTION_TOLERANCE = 1e-9
_DEFAULT_PREDICTION_BASE = 0.0
_DEFAULT_PREDICTION_SLOPE = 1.0

MONOTONE_PREDICTION_RATIONALE = (
    "Positive error-correlation reduces the effective independence that the "
    "exact trio solver exploits, so recovery error should not decrease as rho "
    "increases. A linear non-decreasing heuristic is sufficient because this "
    "module needs the prediction's sign, not a fitted mechanistic law."
)


@dataclass(frozen=True)
class AxiomStatement:
    """Rendered forms of the single-classifier NTQR consistency equations."""

    expressions: dict[str, sympy.Expr]
    latex: dict[str, str]
    as_str: dict[str, str]


@dataclass(frozen=True)
class MonotonePrediction:
    """A monotone analytical prediction for error as a function of correlation."""

    predict: Callable[[float], float]
    rationale: str
    slope: float
    base: float


def symbolic_single_classifier_axiom(
    labels: Sequence[str] = ("a", "b"), *, classifier: str = "c0"
) -> AxiomStatement:
    """Return the per-label NTQR axiom expressions that equal zero when consistent.

    The single-classifier axiom encodes the algebraic consistency conditions for
    one classifier's response counts and pairwise response-transfer counts. Each
    returned SymPy expression is the left-hand side of an equality-to-zero
    statement: a consistent answer key makes every expression vanish.
    """
    if not labels:
        raise ValueError("labels must be a non-empty sequence")

    sca = SingleClassifierAxioms(Labels(tuple(labels)), classifier)
    expressions = {label: sympy.sympify(expr) for label, expr in sca.algebraic_expressions.items()}
    return AxiomStatement(
        expressions=expressions,
        latex={label: sympy.latex(expr) for label, expr in expressions.items()},
        as_str={label: str(expr) for label, expr in expressions.items()},
    )


def two_solution_prevalence_relationship() -> sympy.Eq:
    """Return the symbolic ambiguity p and 1-p for the trio solver's prevalence.

    For the exact error-independent trio solver, the two real logical solutions
    are prevalence mirrors of one another. Written in terms of the two possible
    ``prevalence_a`` values ``p1`` and ``p2``, the analytical statement is
    ``p1 + p2 = 1``.
    """
    p1, p2 = sympy.symbols("p1 p2", real=True)
    return sympy.Eq(p1 + p2, 1)


def _worked_example_votes() -> list[tuple[str, ...]]:
    population = generate_population(60, seed=7)
    items = sample_items(200, prevalence_a=0.5, seed=7)
    return sample_votes_correlated(population[:3], items, rho=0.0, seed=7)


def verify_two_solution_sum(votes_by_judge: Sequence[Sequence[str]]) -> tuple[float, float]:
    """Return the two real ``prevalence_a`` values and verify they sum to one.

    This function bridges the symbolic ambiguity to a concrete trio by calling
    the existing :func:`ntqr_allotment.ntqr_eval.error_independent_solutions`.
    It is intentionally narrow: if the trio does not yield exactly two real
    solutions, a ``ValueError`` is raised because degenerate trios are outside
    this analytical check.
    """
    solutions = error_independent_solutions(votes_by_judge)
    if len(solutions) != 2:
        raise ValueError("expected exactly two real error-independent solutions")

    prevalences = tuple(sorted(float(solution.prevalence_a) for solution in solutions))
    residual = sympy.N(sympy.Float(prevalences[0]) + sympy.Float(prevalences[1]) - 1.0)
    assert abs(float(residual)) <= _TWO_SOLUTION_TOLERANCE
    return prevalences


def predicted_error_vs_correlation() -> MonotonePrediction:
    """Return a monotone heuristic claim: recovery error does not decrease in rho.

    The prediction is the linear function ``g(rho) = base + slope * rho`` with
    ``base >= 0`` and ``slope >= 0``. It is not a fitted law for the study; it
    encodes the analytical sign claim that increasing positive error-correlation
    weakens the independence assumptions the exact solver depends on.
    """

    def predict(rho: float) -> float:
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho must be in [0, 1]")
        return _DEFAULT_PREDICTION_BASE + _DEFAULT_PREDICTION_SLOPE * float(rho)

    return MonotonePrediction(
        predict=predict,
        rationale=MONOTONE_PREDICTION_RATIONALE,
        slope=_DEFAULT_PREDICTION_SLOPE,
        base=_DEFAULT_PREDICTION_BASE,
    )


def fit_error_correlation_slope(rhos: Sequence[float], errors: Sequence[float]) -> float:
    """Estimate the OLS slope of recovery error as a function of correlation rho.

    A positive fitted slope means empirical recovery error increases with
    correlation, matching the analytical sign prediction. The returned value is
    the ordinary least squares coefficient for ``errors ~ alpha + slope * rhos``.
    """
    if not rhos:
        raise ValueError("rhos must be a non-empty sequence")
    if not errors:
        raise ValueError("errors must be a non-empty sequence")
    if len(rhos) != len(errors):
        raise ValueError("rhos and errors must have the same length")
    if len(rhos) < 2:
        raise ValueError("at least two (rho, error) points are required")

    xs = [float(rho) for rho in rhos]
    ys = [float(error) for error in errors]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if math.isclose(denominator, 0.0, abs_tol=0.0):
        raise ValueError("rhos must have positive variance")

    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    return numerator / denominator


def prediction_matches_simulation(rhos: Sequence[float], errors: Sequence[float]) -> bool:
    """Return whether the empirical slope agrees with the non-decreasing theory.

    Because :func:`predicted_error_vs_correlation` encodes a non-negative slope,
    agreement means the empirically fitted slope is also non-negative.
    """
    empirical_slope = fit_error_correlation_slope(rhos, errors)
    predicted = predicted_error_vs_correlation()
    return empirical_slope >= 0.0 and predicted.slope >= 0.0


def herfindahl_index(group_counts: Mapping[str, int]) -> float:
    """Herfindahl-Hirschman concentration index ``H = sum_b p_b^2`` of a panel.

    ``p_b`` is the fraction of panel seats in group ``b``. ``H`` equals the
    probability that two members drawn *with replacement* fall in the same group,
    i.e. the panel's exposure to a group-keyed shared error confound. Over ``B``
    groups it is minimised at ``1/B`` by a perfectly balanced panel and maximised
    at ``1`` by a single-group panel. This is the closed-form panel statistic the
    bloc-confound phase transition is governed by: realized trio error-correlation
    is monotone in ``H`` over the confound's grouping axis.
    """
    total = sum(group_counts.values())
    if total <= 0:
        raise ValueError("group_counts must sum to a positive integer")
    return float(sum((count / total) ** 2 for count in group_counts.values()))


def same_group_pair_probability(group_counts: Mapping[str, int]) -> float:
    """Probability that two *distinct* members share a group (sampling w/o replacement).

    ``P = sum_b c_b (c_b - 1) / (N (N - 1))`` for group counts ``c_b`` summing to
    ``N``. This is the finite-panel correction to :func:`herfindahl_index`; the
    expected number of same-group pairs among the three pairs of a random trio is
    ``3 P``, the trio's shared-confound exposure.
    """
    total = sum(group_counts.values())
    if total < 2:
        raise ValueError("need at least two members to form a pair")
    same = sum(count * (count - 1) for count in group_counts.values())
    return float(same / (total * (total - 1)))


def expected_same_group_trio_pairs(group_counts: Mapping[str, int]) -> float:
    """Expected count of same-group pairs among the 3 pairs of a uniform random trio."""
    return 3.0 * same_group_pair_probability(group_counts)


def concentration_herfindahl(concentration: float, n_groups: int) -> float:
    """Closed-form Herfindahl index of a concentration-dialled panel (large-panel limit).

    A panel with a fraction ``c`` of seats massed in one of ``n_groups`` groups and
    the remaining ``1 - c`` spread evenly across all groups has, in the large-panel
    limit, group fractions ``c + (1-c)/n`` for the target group and ``(1-c)/n`` for
    each other. Its Herfindahl index is monotone increasing in ``c`` from ``1/n``
    (balanced, ``c=0``) to ``1`` (single-group, ``c=1``) -- the formal reason the
    representativeness dial monotonically raises shared-confound exposure.
    """
    if not 0.0 <= concentration <= 1.0:
        raise ValueError("concentration must be in [0, 1]")
    if n_groups < 1:
        raise ValueError("n_groups must be >= 1")
    base = (1.0 - concentration) / n_groups
    target = concentration + base
    return float(target**2 + (n_groups - 1) * base**2)


def number_apriori_evaluations(q: int) -> int:
    """Return the exact count of answer-key evaluations in the binary alarm scan.

    For a binary corpus of ``q`` items, the a-priori enumeration over the
    answer-key simplex has tetrahedral count ``(q + 1)(q + 2)(q + 3) / 6``.
    This is the closed-form statement behind the measured ``O(Q^3)`` growth
    noted in :mod:`ntqr_allotment.ntqr_eval`.
    """
    if q < 0:
        raise ValueError("q must be non-negative")
    return (q + 1) * (q + 2) * (q + 3) // 6


def apriori_growth_is_cubic(q1: int, q2: int) -> float:
    """Return the evaluation-count ratio whose large-q limit reveals cubic growth.

    When ``q2 = 2 * q1`` and both are large, the ratio approaches ``2^3 = 8``,
    the signature of ``O(Q^3)`` scaling.
    """
    if q1 < 1:
        raise ValueError("q1 must be at least 1")
    if q2 < 0:
        raise ValueError("q2 must be non-negative")
    return number_apriori_evaluations(q2) / number_apriori_evaluations(q1)


__all__ = [
    "AxiomStatement",
    "MonotonePrediction",
    "MONOTONE_PREDICTION_RATIONALE",
    "symbolic_single_classifier_axiom",
    "two_solution_prevalence_relationship",
    "_worked_example_votes",
    "verify_two_solution_sum",
    "predicted_error_vs_correlation",
    "fit_error_correlation_slope",
    "prediction_matches_simulation",
    "herfindahl_index",
    "same_group_pair_probability",
    "expected_same_group_trio_pairs",
    "concentration_herfindahl",
    "number_apriori_evaluations",
    "apriori_growth_is_cubic",
]
