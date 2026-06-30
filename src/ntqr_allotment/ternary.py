"""R=3 (ternary) axiom-CONSISTENCY track.

R=3 error-independent EXACT recovery is UNSOLVED upstream and is anti-vision.
This module does NOT recover an answer key. It is a CONSISTENCY / feasibility
instrument: it uses ``ntqr.r3``'s single-classifier axioms to check whether a
ternary response-count table is internally consistent with having been produced
by SOME answer key. This is the R=3 analog of the alarm's consistency
machinery; it never performs ternary evaluation or recovery.

Concretely: given a ternary labeled confusion table ``confusion[true][pred]`` of
integer response counts for one classifier, the single-classifier axioms of
:class:`ntqr.r3.raxioms.SingleClassifierAxioms` give one algebraic residual per
label that vanishes exactly when the table's cell counts and their row totals
(the per-true-label question counts ``Q_t``) and column totals (the marginal
response counts ``R_{p}``) are mutually consistent. We expose:

* a seeded synthetic generator of a ternary confusion table (a multinomial judge
  with a tunable per-label accuracy);
* the full ``eval_dict`` mapping every free axiom symbol to an integer;
* a boolean consistency check (wraps ``satisfies_axioms``);
* the numeric per-label residuals (via ``evaluate_axioms`` + substitution).

Ternary label space is ``("a", "b", "c")`` to match ``ntqr.r3``. All randomness
is seeded (``numpy.random.default_rng``) -> deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import sympy
from ntqr import Labels
from ntqr.r3.raxioms import SingleClassifierAxioms

TERNARY_LABELS: tuple[str, str, str] = ("a", "b", "c")


def _build_axioms(classifier: str) -> SingleClassifierAxioms:
    """Construct the ternary single-classifier axioms for ``classifier``."""
    return SingleClassifierAxioms(Labels(TERNARY_LABELS), classifier)


def _free_symbols_by_name(sca: SingleClassifierAxioms) -> dict[str, sympy.Symbol]:
    """Index every free symbol of the axiom expressions by its string name."""
    by_name: dict[str, sympy.Symbol] = {}
    for expr in sca.algebraic_expressions.values():
        for sym in expr.free_symbols:
            by_name[str(sym)] = sym
    return by_name


@dataclass(frozen=True)
class TernaryConfusion:
    """A deterministic ternary labeled confusion table for one classifier.

    ``counts[true_label][pred_label]`` is the integer count of items whose true
    label is ``true_label`` and whose predicted label is ``pred_label``. The
    table is built by a seeded multinomial judge whose diagonal mass grows with
    ``accuracy`` (the per-label probability of a correct prediction); the
    remaining ``1 - accuracy`` mass is split evenly across the two wrong labels.
    """

    counts: dict[str, dict[str, int]]
    accuracy: float
    seed: int
    per_label_items: int

    def row_total(self, true_label: str) -> int:
        """Total items whose true label is ``true_label`` (a ``Q_t`` value)."""
        return sum(self.counts[true_label].values())

    def column_total(self, pred_label: str) -> int:
        """Total items predicted ``pred_label`` (a marginal ``R_{p}`` value)."""
        return sum(self.counts[t][pred_label] for t in TERNARY_LABELS)

    def diagonal_mass(self) -> int:
        """Total correctly classified items (sum of the diagonal cells)."""
        return sum(self.counts[t][t] for t in TERNARY_LABELS)


def make_ternary_confusion(
    *,
    accuracy: float,
    seed: int,
    per_label_items: int = 100,
) -> TernaryConfusion:
    """Build a seeded synthetic ternary confusion table.

    For each true label, ``per_label_items`` items are drawn from a multinomial
    whose probability vector places ``accuracy`` on the correct label and the
    remaining mass split evenly over the two incorrect labels. The draw is
    seeded with ``numpy.random.default_rng(seed)`` so the same seed yields a
    byte-identical table.
    """
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError("accuracy must be in [0, 1]")
    if per_label_items <= 0:
        raise ValueError("per_label_items must be positive")

    rng = np.random.default_rng(seed)
    wrong = (1.0 - accuracy) / 2.0
    counts: dict[str, dict[str, int]] = {}
    for true_label in TERNARY_LABELS:
        probs = np.array(
            [accuracy if p == true_label else wrong for p in TERNARY_LABELS],
            dtype=float,
        )
        # Guard against tiny floating-point drift so probs sum to exactly 1.
        probs = probs / probs.sum()
        drawn = rng.multinomial(per_label_items, probs)
        counts[true_label] = {p: int(drawn[i]) for i, p in enumerate(TERNARY_LABELS)}
    return TernaryConfusion(
        counts=counts,
        accuracy=float(accuracy),
        seed=int(seed),
        per_label_items=int(per_label_items),
    )


def _as_counts(confusion: TernaryConfusion | Mapping[str, Mapping[str, int]]) -> Mapping[str, Mapping[str, int]]:
    """Return the raw ``counts`` mapping from either accepted input type."""
    if isinstance(confusion, TernaryConfusion):
        return confusion.counts
    return confusion


def axiom_eval_dict(
    confusion: TernaryConfusion | Mapping[str, Mapping[str, int]],
    *,
    classifier: str = "c0",
) -> dict[sympy.Symbol, int]:
    """Map every free axiom symbol to its integer value for ``confusion``.

    Builds the substitution dictionary required by ``satisfies_axioms`` /
    ``evaluate_axioms``: each off-diagonal cell symbol ``R_{m_{c0},t}`` is set to
    ``confusion[t][m]``, each per-true-label total ``Q_t`` to the row sum, and
    each marginal response total ``R_{p_{c0}}`` to the column sum. Diagonal
    cells are determined by the marginals and are not free symbols, so they are
    not added (they would be redundant).
    """
    counts = _as_counts(confusion)
    sca = _build_axioms(classifier)
    rbl = sca.responses_by_label
    by_name = _free_symbols_by_name(sca)

    eval_dict: dict[sympy.Symbol, int] = {}
    for true_label in TERNARY_LABELS:
        for pred_label in TERNARY_LABELS:
            sym = rbl[true_label][pred_label]
            if sym in by_name.values():
                eval_dict[sym] = int(counts[true_label][pred_label])
        # Row sum -> Q_t question total.
        eval_dict[by_name[f"Q_{true_label}"]] = int(
            sum(counts[true_label][p] for p in TERNARY_LABELS)
        )
    for pred_label in TERNARY_LABELS:
        # Column sum -> R_{pred_label} marginal response total.
        eval_dict[by_name[f"R_{{{pred_label}_{{{classifier}}}}}"]] = int(
            sum(counts[t][pred_label] for t in TERNARY_LABELS)
        )
    return eval_dict


def corrupt_eval_dict(
    eval_dict: Mapping[sympy.Symbol, int],
    *,
    true_label: str,
    pred_label: str,
    delta: int,
    classifier: str = "c0",
) -> dict[sympy.Symbol, int]:
    """Return a copy of ``eval_dict`` with one off-diagonal cell shifted by ``delta``.

    Critically, the cell symbol ``R_{pred_label_{c0},true_label}`` is changed but
    its row total ``Q_{true_label}`` and column total ``R_{pred_label}`` are left
    untouched. This breaks the marginal bookkeeping that the single-classifier
    axioms enforce, which is exactly the inconsistency the feasibility instrument
    is designed to detect. Diagonal cells are not free symbols and cannot be
    corrupted this way.
    """
    if true_label == pred_label:
        raise ValueError("only off-diagonal cells are free symbols and can be corrupted")
    sca = _build_axioms(classifier)
    sym = sca.responses_by_label[true_label][pred_label]
    corrupted = dict(eval_dict)
    if sym not in corrupted:
        raise KeyError(f"{sym} is not present in eval_dict")
    corrupted[sym] = int(corrupted[sym]) + int(delta)
    return corrupted


def eval_dict_is_consistent(
    eval_dict: Mapping[sympy.Symbol, int], *, classifier: str = "c0"
) -> bool:
    """Return whether a (possibly hand-corrupted) ``eval_dict`` satisfies the axioms."""
    sca = _build_axioms(classifier)
    return bool(sca.satisfies_axioms(dict(eval_dict)))


def eval_dict_residuals(
    eval_dict: Mapping[sympy.Symbol, int], *, classifier: str = "c0"
) -> dict[str, float]:
    """Return the numeric per-label residuals for a (possibly corrupted) eval_dict."""
    sca = _build_axioms(classifier)
    residual_exprs = sca.evaluate_axioms(dict(eval_dict))
    residuals: dict[str, float] = {}
    for label, expr in residual_exprs.items():
        residuals[str(label)] = float(sympy.sympify(expr).subs(dict(eval_dict)))
    return residuals


def is_axiom_consistent(
    confusion: TernaryConfusion | Mapping[str, Mapping[str, int]],
    *,
    classifier: str = "c0",
) -> bool:
    """Return whether ``confusion`` satisfies the ternary single-classifier axioms.

    This is the consistency / feasibility check: ``True`` means the table is
    internally consistent with having been produced by some answer key. It does
    NOT recover any answer key. Wraps ``SingleClassifierAxioms.satisfies_axioms``.

    Note: a table whose ``Q_t`` row totals and ``R_p`` column totals are derived
    from its own cells (the only way to build one from a plain ``counts``
    mapping) is consistent by construction. The honesty of the instrument lives
    in :func:`corrupt_eval_dict`, which shifts a cell while holding its marginals
    fixed — see the negative-control tests.
    """
    eval_dict = axiom_eval_dict(confusion, classifier=classifier)
    return eval_dict_is_consistent(eval_dict, classifier=classifier)


def axiom_residuals(
    confusion: TernaryConfusion | Mapping[str, Mapping[str, int]],
    *,
    classifier: str = "c0",
) -> dict[str, float]:
    """Return the numeric per-label axiom residuals for ``confusion``.

    Each residual is the left-hand side of a single-classifier consistency
    equation evaluated at the table's counts and marginals; a consistent table
    gives ``0.0`` for every label. Computed via ``evaluate_axioms`` followed by
    substitution of the :func:`axiom_eval_dict` values.
    """
    eval_dict = axiom_eval_dict(confusion, classifier=classifier)
    return eval_dict_residuals(eval_dict, classifier=classifier)


__all__ = [
    "TERNARY_LABELS",
    "TernaryConfusion",
    "make_ternary_confusion",
    "axiom_eval_dict",
    "corrupt_eval_dict",
    "eval_dict_is_consistent",
    "eval_dict_residuals",
    "is_axiom_consistent",
    "axiom_residuals",
]
