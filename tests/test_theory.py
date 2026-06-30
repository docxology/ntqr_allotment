from __future__ import annotations

import pytest
import sympy

from ntqr_allotment.theory import (
    _worked_example_votes,
    apriori_growth_is_cubic,
    concentration_herfindahl,
    expected_same_group_trio_pairs,
    fit_error_correlation_slope,
    herfindahl_index,
    number_apriori_evaluations,
    predicted_error_vs_correlation,
    prediction_matches_simulation,
    same_group_pair_probability,
    symbolic_single_classifier_axiom,
    two_solution_prevalence_relationship,
    verify_two_solution_sum,
)


def test_symbolic_single_classifier_axiom_returns_sympy_and_renderings():
    result = symbolic_single_classifier_axiom()
    assert set(result.expressions) == {"a", "b"}
    assert isinstance(result.expressions["a"], sympy.Expr)
    assert isinstance(result.expressions["b"], sympy.Expr)
    assert result.latex["a"]
    assert result.latex["b"]
    assert result.as_str["a"]
    assert result.as_str["b"]


def test_symbolic_single_classifier_axiom_rejects_empty_labels():
    with pytest.raises(ValueError):
        symbolic_single_classifier_axiom([])


def test_two_solution_relationship_and_numeric_verification():
    relation = two_solution_prevalence_relationship()
    p1, p2 = sympy.symbols("p1 p2", real=True)
    assert isinstance(relation, sympy.Equality)
    assert sympy.simplify(relation.lhs - relation.rhs - (p1 + p2 - 1)) == 0

    prevalence_pair = verify_two_solution_sum(_worked_example_votes())
    assert prevalence_pair[0] + prevalence_pair[1] == pytest.approx(1.0, abs=1e-6)


def test_verify_two_solution_sum_rejects_non_trio_input():
    with pytest.raises(ValueError):
        verify_two_solution_sum([("a", "b"), ("a", "b")])


def test_predicted_error_vs_correlation_is_monotone():
    prediction = predicted_error_vs_correlation()
    rho_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    predicted_values = [prediction.predict(rho) for rho in rho_grid]

    assert prediction.rationale
    assert prediction.slope >= 0.0
    assert predicted_values == sorted(predicted_values)


def test_predicted_error_vs_correlation_validates_rho():
    prediction = predicted_error_vs_correlation()
    with pytest.raises(ValueError):
        prediction.predict(1.1)


def test_fit_error_correlation_slope_and_prediction_agreement():
    rhos = [0.0, 0.25, 0.5, 0.75, 1.0]
    errors = [0.1, 0.2, 0.3, 0.4, 0.5]
    slope = fit_error_correlation_slope(rhos, errors)
    assert slope > 0.0
    assert prediction_matches_simulation(rhos, errors) is True


@pytest.mark.parametrize(
    ("rhos", "errors"),
    [
        ([], [0.1]),
        ([0.0], []),
        ([0.0, 0.5], [0.1]),
        ([0.5], [0.1]),
        ([0.5, 0.5], [0.1, 0.2]),
    ],
)
def test_fit_error_correlation_slope_validation(rhos, errors):
    with pytest.raises(ValueError):
        fit_error_correlation_slope(rhos, errors)


def test_number_apriori_evaluations_matches_tetrahedral_formula():
    assert number_apriori_evaluations(0) == 1
    assert number_apriori_evaluations(1) == 4
    for q in (2, 5, 10):
        assert number_apriori_evaluations(q) == (q + 1) * (q + 2) * (q + 3) // 6


def test_apriori_growth_is_cubic_for_large_doubling():
    ratio = apriori_growth_is_cubic(50, 100)
    assert 7.0 < ratio < 9.0


def test_apriori_growth_validation():
    with pytest.raises(ValueError):
        number_apriori_evaluations(-1)
    with pytest.raises(ValueError):
        apriori_growth_is_cubic(0, 100)
    with pytest.raises(ValueError):
        apriori_growth_is_cubic(10, -1)


def test_herfindahl_index_balanced_and_concentrated() -> None:
    # Perfectly balanced over B groups -> 1/B; single group -> 1.
    assert herfindahl_index({"a": 2, "b": 2, "c": 2}) == pytest.approx(1 / 3)
    assert herfindahl_index({"a": 6}) == pytest.approx(1.0)
    # Concentration raises it.
    assert herfindahl_index({"a": 4, "b": 1, "c": 1}) > herfindahl_index({"a": 2, "b": 2, "c": 2})
    with pytest.raises(ValueError):
        herfindahl_index({"a": 0})


def test_same_group_pair_probability_and_trio_pairs() -> None:
    # Single 6-member group: every pair is same-group -> probability 1, 3 trio pairs.
    assert same_group_pair_probability({"a": 6}) == pytest.approx(1.0)
    assert expected_same_group_trio_pairs({"a": 6}) == pytest.approx(3.0)
    # Balanced panel has fewer same-group pairs than a concentrated one.
    assert same_group_pair_probability({"a": 2, "b": 2, "c": 2}) < same_group_pair_probability(
        {"a": 4, "b": 1, "c": 1}
    )
    with pytest.raises(ValueError):
        same_group_pair_probability({"a": 1})


def test_concentration_herfindahl_is_monotone_in_concentration() -> None:
    values = [concentration_herfindahl(c, 3) for c in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert values[0] == pytest.approx(1 / 3)
    assert values[-1] == pytest.approx(1.0)
    assert all(b >= a for a, b in zip(values, values[1:]))
    with pytest.raises(ValueError):
        concentration_herfindahl(1.5, 3)


def test_herfindahl_predicts_strategy_correlation_ordering() -> None:
    """The formal bridge: at matched competence the three composition strategies'
    Herfindahl indices order exactly as their measured error-correlation does
    (representative < random < single-bloc). Computed from real panel
    compositions, no hand-set numbers."""
    from ntqr_allotment.experts import generate_population
    from ntqr_allotment.sortition import STRATEGIES

    def mean_hhi(strategy: str) -> float:
        hhis = []
        for s in range(12):
            pop = generate_population(96, seed=s, mean_expertise=0.70, bias_std=0.4)
            panel = STRATEGIES[strategy](pop, 6, seed=s + 13)
            hhis.append(herfindahl_index(panel.composition["ideology"]))
        return sum(hhis) / len(hhis)

    h_rep = mean_hhi("representative_sortition")
    h_rand = mean_hhi("random_selection")
    h_ideo = mean_hhi("ideological_selection")
    assert h_rep < h_rand < h_ideo
    assert h_rep == pytest.approx(1 / 3, abs=1e-9)  # balanced quota -> exactly 1/B
    assert h_ideo == pytest.approx(1.0)  # single bloc
