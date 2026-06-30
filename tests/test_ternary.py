"""Tests for the R=3 ternary axiom-consistency instrument.

No mocks: every test uses a real seeded numeric ternary confusion table and the
real ``ntqr.r3`` single-classifier axioms. Each behavioural test carries an
explicit negative control whose assertion flips on a broken/degenerate input.
"""

from __future__ import annotations

import pytest

from ntqr_allotment.ternary import (
    TERNARY_LABELS,
    TernaryConfusion,
    axiom_eval_dict,
    axiom_residuals,
    corrupt_eval_dict,
    eval_dict_is_consistent,
    eval_dict_residuals,
    is_axiom_consistent,
    make_ternary_confusion,
)


def test_honest_table_is_axiom_consistent() -> None:
    """An honestly-built table is consistent: residuals 0, is_consistent True.

    NEGATIVE CONTROL: corrupting one off-diagonal CELL count while holding its
    row total Q_t and column total R_p fixed (via corrupt_eval_dict) breaks the
    marginal bookkeeping, so eval_dict_is_consistent flips True -> False and at
    least one residual becomes non-zero.
    """
    conf = make_ternary_confusion(accuracy=0.8, seed=7)

    assert is_axiom_consistent(conf) is True
    residuals = axiom_residuals(conf)
    assert set(residuals) == set(TERNARY_LABELS)
    for value in residuals.values():
        assert value == pytest.approx(0.0, abs=1e-9)

    # NEGATIVE CONTROL: shift cell (true=b, pred=a) by +50, marginals untouched.
    eval_dict = axiom_eval_dict(conf)
    corrupted = corrupt_eval_dict(eval_dict, true_label="b", pred_label="a", delta=50)
    assert eval_dict_is_consistent(corrupted) is False
    bad_residuals = eval_dict_residuals(corrupted)
    assert any(abs(v) > 0.0 for v in bad_residuals.values())


def test_eval_dict_covers_all_free_symbols() -> None:
    """axiom_eval_dict supplies every free symbol the axioms reference.

    NEGATIVE CONTROL: dropping one required symbol from the eval_dict leaves a
    residual expression with an unsubstituted free symbol, so the substituted
    residual is no longer a plain number and float() raises TypeError. This
    flips the "always reducible to a number" property to a failure.
    """
    import sympy
    from ntqr import Labels
    from ntqr.r3.raxioms import SingleClassifierAxioms

    conf = make_ternary_confusion(accuracy=0.75, seed=11)
    eval_dict = axiom_eval_dict(conf)

    sca = SingleClassifierAxioms(Labels(TERNARY_LABELS), "c0")
    needed: set[sympy.Symbol] = set()
    for expr in sca.algebraic_expressions.values():
        needed |= expr.free_symbols
    assert needed.issubset(set(eval_dict))

    # NEGATIVE CONTROL: drop one required symbol; a residual stays symbolic.
    a_symbol = next(iter(needed))
    partial = {k: v for k, v in eval_dict.items() if k is not a_symbol}
    residual_exprs = sca.evaluate_axioms(partial)
    symbolic = [
        sympy.sympify(expr).subs(partial)
        for expr in residual_exprs.values()
        if sympy.sympify(expr).subs(partial).free_symbols
    ]
    assert symbolic, "dropping a required symbol must leave a symbolic residual"
    with pytest.raises(TypeError):
        float(symbolic[0])


def test_determinism_same_seed_identical_table() -> None:
    """Same seed -> byte-identical confusion table.

    NEGATIVE CONTROL: a different seed produces a different table, so the
    equality assertion flips to inequality.
    """
    a = make_ternary_confusion(accuracy=0.7, seed=123)
    b = make_ternary_confusion(accuracy=0.7, seed=123)
    assert a.counts == b.counts
    assert a == b

    # NEGATIVE CONTROL: a different seed yields a different table.
    c = make_ternary_confusion(accuracy=0.7, seed=124)
    assert c.counts != a.counts


def test_judge_honors_accuracy_parameter() -> None:
    """Diagonal mass increases with the accuracy parameter.

    NEGATIVE CONTROL: a chance-level accuracy (1/3) puts no extra mass on the
    diagonal, so the high-accuracy diagonal strictly exceeds it; were accuracy
    ignored, the diagonal masses would be statistically equal and the strict
    inequality would fail.
    """
    low = make_ternary_confusion(accuracy=0.40, seed=5, per_label_items=300)
    high = make_ternary_confusion(accuracy=0.90, seed=5, per_label_items=300)
    assert high.diagonal_mass() > low.diagonal_mass()

    # NEGATIVE CONTROL: chance-level accuracy has far less diagonal mass.
    chance = make_ternary_confusion(accuracy=1.0 / 3.0, seed=5, per_label_items=300)
    assert chance.diagonal_mass() < high.diagonal_mass()


def test_row_and_column_totals() -> None:
    """Row totals equal per_label_items; columns redistribute under noise.

    NEGATIVE CONTROL: the column totals need NOT equal per_label_items (errors
    redistribute mass across columns), so asserting every column equals the row
    count fails for at least one label on a noisy table.
    """
    conf = make_ternary_confusion(accuracy=0.6, seed=9, per_label_items=120)
    for label in TERNARY_LABELS:
        assert conf.row_total(label) == 120

    # NEGATIVE CONTROL: at least one column total differs from per_label_items.
    assert any(conf.column_total(label) != 120 for label in TERNARY_LABELS)


def test_accepts_raw_mapping_and_dataclass_equivalently() -> None:
    """A raw counts mapping and the dataclass give identical consistency results.

    NEGATIVE CONTROL: corrupting a cell in the eval_dict built from the raw
    mapping (holding marginals) makes it inconsistent, flipping the result
    relative to the honest dataclass.
    """
    conf = make_ternary_confusion(accuracy=0.85, seed=3)
    assert is_axiom_consistent(conf.counts) == is_axiom_consistent(conf)
    assert is_axiom_consistent(conf.counts) is True

    # NEGATIVE CONTROL: corrupted eval_dict from the raw mapping disagrees.
    eval_dict = axiom_eval_dict(conf.counts)
    corrupted = corrupt_eval_dict(eval_dict, true_label="c", pred_label="a", delta=33)
    assert eval_dict_is_consistent(corrupted) != is_axiom_consistent(conf)


def test_corrupt_eval_dict_guards() -> None:
    """corrupt_eval_dict rejects diagonal cells and missing symbols.

    NEGATIVE CONTROL: a valid off-diagonal corruption on a present symbol does
    NOT raise, so the guards are specific to the bad inputs.
    """
    conf = make_ternary_confusion(accuracy=0.7, seed=2)
    eval_dict = axiom_eval_dict(conf)

    with pytest.raises(ValueError):
        corrupt_eval_dict(eval_dict, true_label="a", pred_label="a", delta=1)
    with pytest.raises(KeyError):
        corrupt_eval_dict({}, true_label="a", pred_label="b", delta=1)

    # NEGATIVE CONTROL: a legitimate corruption succeeds and returns a new dict.
    out = corrupt_eval_dict(eval_dict, true_label="a", pred_label="b", delta=1)
    assert isinstance(out, dict)
    assert out is not eval_dict


def test_invalid_parameters_raise() -> None:
    """Out-of-range generator parameters raise ValueError.

    NEGATIVE CONTROL: a valid accuracy/per_label_items pair does NOT raise, so
    the error is specific to the bad inputs rather than unconditional.
    """
    with pytest.raises(ValueError):
        make_ternary_confusion(accuracy=1.5, seed=1)
    with pytest.raises(ValueError):
        make_ternary_confusion(accuracy=0.5, seed=1, per_label_items=0)

    # NEGATIVE CONTROL: valid arguments succeed (no raise).
    ok = make_ternary_confusion(accuracy=0.5, seed=1, per_label_items=10)
    assert isinstance(ok, TernaryConfusion)
