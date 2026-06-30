"""NTQR adapter: turn panel judgments into no-answer-key evaluations.

Two NTQR capabilities are wrapped here, matching the project's two questions:

1. **Accuracy recovery (exact, trio-only).** ``ntqr.r2.evaluators`` solves the
   error-independent system for exactly THREE binary judges, returning two
   logically consistent (prevalence, per-judge accuracy) solutions. Because the
   synthetic ground truth is known, we also compute the *supervised* oracle and
   measure how close the unsupervised estimate gets.

2. **Statistical power / alarm (scales to N judges).** ``ntqr.alarms`` flags
   when a set of judges cannot all be simultaneously consistent with any answer
   key at a given safety specification. As the panel grows or the corpus grows,
   the consistent region tightens -- the basis for "how many experts do we need".

These are thin wrappers: all algebra lives in ``ntqr``. NaN/degenerate solver
output is surfaced, never silently swallowed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import sympy
from ntqr import Labels
from ntqr.alarms import SingleClassifierAxiomsAlarm
from ntqr.r2.datasketches import TrioLabelVoteCounts, TrioVoteCounts
from ntqr.r2.evaluations import SingleClassifierEvaluations
from ntqr.r2.evaluators import (
    ErrorIndependentEvaluation,
    MajorityVotingEvaluation,
    SupervisedEvaluation,
)
from ntqr.r2.raxioms import SingleClassifierAxioms

from .experts import Item

_PATTERNS = [(a, b, c) for a in "ab" for b in "ab" for c in "ab"]
_REAL_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Evaluation:
    """A single (prevalence, per-judge mean-accuracy) point in evaluation space."""

    prevalence_a: float
    accuracies: tuple[float, ...]  # per-judge mean accuracy (0.5*(acc_a+acc_b))

    def error_vs(self, oracle: "Evaluation") -> float:
        """L1-style distance to an oracle: prevalence + mean per-judge accuracy."""
        prev_err = abs(self.prevalence_a - oracle.prevalence_a)
        n = min(len(self.accuracies), len(oracle.accuracies))
        acc_err = sum(abs(self.accuracies[i] - oracle.accuracies[i]) for i in range(n)) / max(n, 1)
        return prev_err + acc_err


def trio_vote_counts(votes_by_judge: Sequence[Sequence[str]]) -> dict[tuple[str, ...], int]:
    """Count joint vote patterns for exactly 3 binary judges (unlabeled)."""
    if len(votes_by_judge) != 3:
        raise ValueError("the exact NTQR solver requires exactly 3 judges")
    n_items = len(votes_by_judge[0])
    if any(len(v) != n_items for v in votes_by_judge):
        raise ValueError("all judges must vote on the same number of items")
    counts = {p: 0 for p in _PATTERNS}
    for k in range(n_items):
        pattern = (votes_by_judge[0][k], votes_by_judge[1][k], votes_by_judge[2][k])
        counts[pattern] += 1
    return counts


def trio_label_vote_counts(
    votes_by_judge: Sequence[Sequence[str]], items: Sequence[Item]
) -> dict[str, dict[tuple[str, ...], int]]:
    """Count joint vote patterns split by TRUE label (labeled -> oracle path)."""
    if len(votes_by_judge) != 3:
        raise ValueError("the exact NTQR solver requires exactly 3 judges")
    counts: dict[str, dict[tuple[str, ...], int]] = {
        "a": {p: 0 for p in _PATTERNS},
        "b": {p: 0 for p in _PATTERNS},
    }
    for k, item in enumerate(items):
        pattern = (votes_by_judge[0][k], votes_by_judge[1][k], votes_by_judge[2][k])
        counts[item.true_label][pattern] += 1
    return counts


def _real_or_none(value: object) -> float | None:
    """Cast a sympy/Python solver value to a finite real float, else ``None``."""
    expr = sympy.sympify(value)
    if expr.is_real is True:
        as_float = float(expr)
        if math.isnan(as_float) or math.isinf(as_float):
            return None
        return as_float

    imag = float(sympy.N(sympy.im(expr)))
    if math.isnan(imag) or math.isinf(imag) or abs(imag) > _REAL_TOLERANCE:
        return None

    real = float(sympy.N(sympy.re(expr)))
    if math.isnan(real) or math.isinf(real):
        return None
    return real


#: Probabilities (prevalence, per-judge accuracies) must lie in [0, 1]; allow a
#: small float-noise slack before declaring a root non-physical.
_PROB_TOLERANCE = 1e-6


def _in_unit_interval(value: float) -> bool:
    """True if ``value`` is a valid probability within float-noise tolerance."""
    return -_PROB_TOLERANCE <= value <= 1.0 + _PROB_TOLERANCE


def _to_evaluation(sol: dict) -> Evaluation | None:
    """Convert one solver solution to an :class:`Evaluation`, or ``None`` if degenerate.

    Returns ``None`` when prevalence or any per-judge accuracy is non-real,
    non-finite, **or non-physical** (outside [0, 1]). Prevalence and accuracies are
    probabilities, so a real algebraic root that puts them outside [0, 1] is not a
    logically consistent evaluation -- it is as degenerate as a complex root, and
    scoring it as a huge oracle distance produced the heavy-tailed error outliers
    that dominated the means. The caller must skip a ``None``.
    """
    prev = _real_or_none(sol["prevalence"]["a"])
    if prev is None or not _in_unit_interval(prev):
        return None
    accs: list[float] = []
    for judge in sol["accuracy"]:
        acc_a = _real_or_none(judge["a"])
        acc_b = _real_or_none(judge["b"])
        if acc_a is None or acc_b is None:
            return None
        if not _in_unit_interval(acc_a) or not _in_unit_interval(acc_b):
            return None
        accs.append(0.5 * (acc_a + acc_b))
    return Evaluation(prevalence_a=prev, accuracies=tuple(accs))


def error_independent_solutions(votes_by_judge: Sequence[Sequence[str]]) -> list[Evaluation]:
    """The consistent unsupervised solutions from the exact algebraic solver.

    Complex roots (no real unsupervised solution) are dropped, so the returned
    list may hold two, one, or zero solutions. An empty list means the trio
    admits no real error-independent solution.
    """
    tvc = TrioVoteCounts(trio_vote_counts(votes_by_judge))
    eie = ErrorIndependentEvaluation(tvc)
    return [ev for sol in eie.evaluation_float if (ev := _to_evaluation(sol)) is not None]


def majority_voting_solutions(votes_by_judge: Sequence[Sequence[str]]) -> list[Evaluation]:
    """The majority-voting solutions (crowd-right / crowd-wrong).

    Complex roots (no real unsupervised solution) are dropped from the returned
    list.
    """
    tvc = TrioVoteCounts(trio_vote_counts(votes_by_judge))
    mv = MajorityVotingEvaluation(tvc)
    return [ev for sol in mv.evaluation_float if (ev := _to_evaluation(sol)) is not None]


def supervised_oracle(
    votes_by_judge: Sequence[Sequence[str]], items: Sequence[Item]
) -> Evaluation:
    """Ground-truth evaluation -- the oracle the unsupervised estimate is scored against.

    The supervised solution is read directly from labeled counts and is always
    real and finite; a degenerate oracle would be a contract violation, so this
    fails loudly rather than returning ``None``.
    """
    tlvc = TrioLabelVoteCounts(trio_label_vote_counts(votes_by_judge, items))
    sup = SupervisedEvaluation(tlvc)
    oracle = _to_evaluation(sup.evaluation_float)
    if oracle is None:
        raise ValueError(
            "supervised oracle returned a degenerate (complex/non-finite) "
            "solution; this should be impossible from labeled counts"
        )
    return oracle


def closest_solution(solutions: Sequence[Evaluation], oracle: Evaluation) -> Evaluation:
    """Pick the consistent solution nearest the oracle (resolves the 2-fold ambiguity).

    Raises ``ValueError`` on an empty list: a degenerate trio yields no real
    solution, and silently inventing one would corrupt the measurement. Callers
    that may encounter degenerate trios must check ``solutions`` first and skip.
    """
    if not solutions:
        raise ValueError("no real solution to select from")
    return min(solutions, key=lambda s: s.error_vs(oracle))


#: The local NTQR alarm path enumerates the answer-key simplex with cost ~O(Q^3)
#: (measured: Q=20 ~1.4s, Q=50 ~18s, Q=100 ~200s). It is therefore an
#: opt-in primitive for the SMALL-corpus power regime, not the sweep hot path.
ALARM_DEFAULT_MAX_Q = 30


def alarm_misaligned(
    votes_by_judge: Sequence[Sequence[str]],
    *,
    safety_factors: Sequence[int] = (2, 2),
    max_q: int = ALARM_DEFAULT_MAX_Q,
) -> bool:
    """Return True iff no answer key lets ALL judges meet the safety spec.

    Scales (in judge count) to any number of judges (one
    ``SingleClassifierAxioms`` each). ``True`` means the panel's votes are
    jointly inconsistent with every key at this safety level -- a stronger
    signal as the panel grows. The corpus is capped at ``max_q`` items because
    the underlying NTQR enumeration is ~O(Q^3); pass a larger ``max_q``
    deliberately if you accept the cost. Raises if Q exceeds ``max_q``.
    """
    q = len(votes_by_judge[0])
    if q > max_q:
        raise ValueError(
            f"alarm corpus Q={q} exceeds max_q={max_q}; this NTQR alarm path is "
            f"~O(Q^3). Cap the item count or raise max_q deliberately."
        )
    labels = Labels(("a", "b"))
    axioms = [SingleClassifierAxioms(labels, i) for i in range(len(votes_by_judge))]
    alarm = SingleClassifierAxiomsAlarm(q, axioms, SingleClassifierEvaluations)
    alarm.set_safety_specification(list(safety_factors))
    responses = [[sum(1 for v in votes if v == "a"), sum(1 for v in votes if v == "b")] for votes in votes_by_judge]
    return bool(alarm.are_misaligned(responses))


__all__ = [
    "Evaluation",
    "trio_vote_counts",
    "trio_label_vote_counts",
    "error_independent_solutions",
    "majority_voting_solutions",
    "supervised_oracle",
    "closest_solution",
    "alarm_misaligned",
]
