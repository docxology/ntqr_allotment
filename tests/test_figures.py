from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.text import Text
import pytest

import ntqr_allotment.figures as figure_module
from ntqr_allotment.fairness import FairnessReport
from ntqr_allotment.figure_parts._common import (
    HEATMAP_CELL_FONT_SIZE,
    SOURCE_NOTE_FONT_SIZE,
    SOURCE_NOTE_WRAP_WIDTH,
    TITLE_FONT_SIZE,
    _add_source_note,
)
from ntqr_allotment.figures import (
    DEFAULT_ALARM_MEASURED_POINTS,
    plot_alarm_cost_curve,
    plot_bloc_phase_diagram,
    plot_concentration_dial,
    plot_alarm_power,
    plot_cross_family_contrast,
    plot_cross_family_multiseed,
    plot_error_vs_correlation,
    plot_fairness_maximin,
    plot_method_pipeline_schematic,
    plot_postdoc_age_bias_heatmap,
    plot_postdoc_empirical_alignment,
    plot_postdoc_strategy_ranking,
    plot_pre_post_ntqr_heatmap,
    plot_power_curve,
    plot_power_design_diagnosis,
    plot_power_vs_n,
    plot_rep_vs_ideo_effect,
    plot_rep_vs_ideo_heatmap,
    plot_strategy_correlation,
    plot_strategy_ranking,
    plot_theory_alignment_heatmap,
    plot_track_ranking_inversion,
)
from ntqr_allotment.independence_sweep import IndependenceAggregate


def _assert_nontrivial_png(path: Path) -> None:
    assert path.exists()
    assert path.suffix == ".png"
    assert path.stat().st_size > 1000


def _visible_text_sizes(fig) -> list[float]:
    return [
        text.get_fontsize()
        for text in fig.findobj(Text)
        if text.get_visible() and text.get_text().strip()
    ]


def _bloc_phase_cells() -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    profile = {
        "expertise_threshold": (0.04, 0.09),
        "representative_sortition": (0.15, 0.17),
        "random_selection": (0.16, 0.20),
        "ideological_selection": (0.16, 0.27),
    }
    corr = {
        "expertise_threshold": (0.01, 0.02),
        "representative_sortition": (0.01, 0.02),
        "random_selection": (0.01, 0.06),
        "ideological_selection": (0.01, 0.13),
    }
    for strategy, (lo, hi) in profile.items():
        for rho, val in ((0.0, lo), (0.9, hi)):
            cells.append(
                {
                    "strategy": strategy,
                    "bloc_correlation": rho,
                    "n": 100,
                    "eie_mean": val,
                    "eie_ci95": 0.01,
                    "corr_mean": corr[strategy][0 if rho == 0.0 else 1],
                }
            )
    return cells


def test_plot_bloc_phase_diagram_writes_png(tmp_path: Path) -> None:
    output_path = plot_bloc_phase_diagram(_bloc_phase_cells(), tmp_path / "bloc_phase.png")
    assert output_path == tmp_path / "bloc_phase.png"
    _assert_nontrivial_png(output_path)


def test_plot_bloc_phase_diagram_rejects_empty() -> None:
    with pytest.raises(ValueError):
        plot_bloc_phase_diagram([], Path("unused.png"))


def _concentration_cells() -> list[dict[str, object]]:
    profile = [(0.0, 0.15, 0.02), (0.25, 0.17, 0.04), (0.5, 0.20, 0.08), (0.75, 0.23, 0.11), (1.0, 0.25, 0.13)]
    return [
        {"concentration": c, "n": 100, "eie_mean": e, "eie_ci95": 0.01, "corr_mean": r}
        for c, e, r in profile
    ]


def test_plot_concentration_dial_writes_png(tmp_path: Path) -> None:
    out = plot_concentration_dial(_concentration_cells(), tmp_path / "dial.png")
    assert out == tmp_path / "dial.png"
    _assert_nontrivial_png(out)


def test_plot_concentration_dial_rejects_empty() -> None:
    with pytest.raises(ValueError):
        plot_concentration_dial([], Path("unused.png"))


def test_plot_strategy_ranking_writes_png(tmp_path: Path) -> None:
    ranking = [
        ["representative_sortition", 0.11, 0.02],
        ["expertise_threshold", 0.08, 0.01],
        ["random_selection", 0.14, 0.03],
    ]

    output_path = plot_strategy_ranking(ranking, tmp_path / "strategy_ranking.png")

    assert output_path == tmp_path / "strategy_ranking.png"
    _assert_nontrivial_png(output_path)


def test_plot_strategy_ranking_directly_labels_mean_and_ci(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_text: list[str] = []

    def capture_and_save(fig, output_path):
        observed_text.extend(text.get_text() for text in fig.axes[0].texts)
        return original_save(fig, output_path)

    original_save = figure_module._save_figure
    monkeypatch.setattr(figure_module, "_save_figure", capture_and_save)

    plot_strategy_ranking(
        [["random_selection", 0.42, 0.27], ["expertise_threshold", 0.06, 0.04]],
        tmp_path / "strategy_ranking.png",
    )

    assert "0.060 +/- 0.040" in observed_text
    assert "0.420 +/- 0.270" in observed_text


def test_manuscript_plot_text_sizes_are_readable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_sizes: list[float] = []
    observed_titles: list[str] = []

    def capture_and_save(fig, output_path):
        observed_sizes.extend(_visible_text_sizes(fig))
        observed_titles.extend(ax.get_title() for ax in fig.axes if ax.get_title())
        return original_save(fig, output_path)

    original_save = figure_module._save_figure
    monkeypatch.setattr(figure_module, "_save_figure", capture_and_save)

    plot_strategy_ranking(
        [["random_selection", 0.42, 0.27], ["expertise_threshold", 0.06, 0.04]],
        tmp_path / "strategy_ranking.png",
    )
    plot_pre_post_ntqr_heatmap(
        [
            {
                "strategy": strategy,
                "panel_size": panel,
                "mean_expertise": expertise,
                "bias_std": bias,
                "eie_minus_mv": -0.02 + panel / 300 + bias / 20 - expertise / 50,
            }
            for strategy in ("representative_sortition", "ideological_selection")
            for panel in (3, 6)
            for expertise in (0.68, 0.74)
            for bias in (0.1, 0.3)
        ],
        tmp_path / "pre_post.png",
    )

    assert observed_sizes
    assert min(observed_sizes) >= SOURCE_NOTE_FONT_SIZE
    assert any(size >= TITLE_FONT_SIZE for size in observed_sizes)
    assert any(size >= HEATMAP_CELL_FONT_SIZE for size in observed_sizes)
    assert "Formation strategy sets the no-answer-key error ceiling" in observed_titles


def test_source_note_wraps_long_reader_contract() -> None:
    fig, _ = plt.subplots()
    note = (
        "Source: output/data/sweep_aggregated.csv; caveat: profile-bounded "
        "synthetic reader contract repeated to force wrapping across the static "
        "figure footer without shrinking the provenance text."
    )

    _add_source_note(fig, note)

    rendered = fig.texts[0].get_text()
    assert "\n" in rendered
    assert max(len(line) for line in rendered.splitlines()) <= SOURCE_NOTE_WRAP_WIDTH
    plt.close(fig)


def test_plot_strategy_ranking_rejects_empty_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="strategy ranking"):
        plot_strategy_ranking([], tmp_path / "strategy_ranking.png")


def test_plot_power_curve_writes_png_and_filters_degenerate_rows(tmp_path: Path) -> None:
    rows = [
        {"panel_size": 3, "mean_expertise": 0.70, "eie_error": 0.18},
        {"panel_size": 3, "mean_expertise": 0.70, "eie_error": 0.22},
        {"panel_size": 6, "mean_expertise": 0.70, "eie_error": 0.15},
        {"panel_size": 6, "mean_expertise": 0.70, "eie_error": 0.17},
        {"panel_size": 3, "mean_expertise": 0.82, "eie_error": 0.11},
        {"panel_size": 6, "mean_expertise": 0.82, "eie_error": 0.09},
        {"panel_size": 6, "mean_expertise": 0.82, "eie_error": -1.0},
    ]

    output_path = plot_power_curve(rows, tmp_path / "power_curve.png")

    assert output_path == tmp_path / "power_curve.png"
    _assert_nontrivial_png(output_path)


def test_plot_power_curve_uses_strategy_series_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_labels: list[str] = []

    def capture_and_save(fig, output_path):
        observed_labels.extend(text.get_text() for text in fig.axes[0].get_legend().texts)
        return original_save(fig, output_path)

    original_save = figure_module._save_figure
    monkeypatch.setattr(figure_module, "_save_figure", capture_and_save)
    rows = [
        {"strategy": "representative_sortition", "panel_size": 3, "mean_expertise": 0.70, "eie_error": 0.18},
        {"strategy": "representative_sortition", "panel_size": 6, "mean_expertise": 0.70, "eie_error": 0.15},
        {"strategy": "ideological_selection", "panel_size": 3, "mean_expertise": 0.70, "eie_error": 0.25},
        {"strategy": "ideological_selection", "panel_size": 6, "mean_expertise": 0.70, "eie_error": 0.22},
    ]

    plot_power_curve(rows, tmp_path / "strategy_power_curve.png")

    assert observed_labels == ["ideological selection", "representative sortition"]


def test_plot_power_curve_labels_size_direction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_text: list[str] = []

    def capture_and_save(fig, output_path):
        observed_text.extend(text.get_text() for text in fig.axes[0].texts)
        return original_save(fig, output_path)

    original_save = figure_module._save_figure
    monkeypatch.setattr(figure_module, "_save_figure", capture_and_save)
    rows = [
        {"strategy": "a_strategy", "panel_size": 3, "eie_error": 0.30},
        {"strategy": "a_strategy", "panel_size": 6, "eie_error": 0.20},
        {"strategy": "b_strategy", "panel_size": 3, "eie_error": 0.10},
        {"strategy": "b_strategy", "panel_size": 6, "eie_error": 0.15},
    ]

    plot_power_curve(rows, tmp_path / "power_curve.png")

    assert "a strategy falls" in observed_text
    assert "b strategy rises" in observed_text


def test_plot_power_curve_rejects_all_degenerate_rows(tmp_path: Path) -> None:
    rows = [
        {"panel_size": 3, "mean_expertise": 0.70, "eie_error": -1.0},
        {"panel_size": 6, "mean_expertise": 0.82, "eie_error": float("nan")},
    ]

    with pytest.raises(ValueError, match="filtering degenerate"):
        plot_power_curve(rows, tmp_path / "power_curve.png")


def test_plot_rep_vs_ideo_effect_writes_png(tmp_path: Path) -> None:
    effects = [
        {
            "regime_key": [3, 0.70, 0.08, 0.30, 120, 0.5, 48],
            "effect": -0.03,
            "ci95": 0.02,
        },
        {
            "regime_key": [6, 0.82, 0.08, 0.30, 120, 0.5, 48],
            "effect": 0.05,
            "ci95": 0.03,
        },
    ]

    output_path = plot_rep_vs_ideo_effect(effects, tmp_path / "rep_vs_ideo_effect.png")

    assert output_path == tmp_path / "rep_vs_ideo_effect.png"
    _assert_nontrivial_png(output_path)


def test_plot_rep_vs_ideo_effect_rejects_empty_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="rep_vs_ideo effect"):
        plot_rep_vs_ideo_effect([], tmp_path / "rep_vs_ideo_effect.png")


def _rep_heatmap_cells() -> list[dict[str, object]]:
    return [
        {
            "panel_size": panel,
            "mean_expertise": expertise,
            "bias_std": bias,
            "effect": 0.1 + panel / 100 + bias / 10 - expertise / 20,
            "ci95": 0.03,
            "excludes_zero": bias > 0.2,
            "prediction_status": "aligned_resolved" if bias > 0.2 else "aligned_uncertain",
        }
        for panel in (3, 6)
        for expertise in (0.68, 0.74)
        for bias in (0.1, 0.3)
    ]


def test_plot_rep_vs_ideo_heatmap_labels_regime_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_titles: list[str] = []
    observed_labels: list[str] = []
    observed_text: list[str] = []

    def capture_and_save(fig, output_path):
        for ax in fig.axes:
            observed_titles.append(ax.get_title())
            observed_labels.append(ax.get_xlabel())
            observed_labels.append(ax.get_ylabel())
            observed_text.extend(text.get_text() for text in ax.texts)
        return original_save(fig, output_path)

    original_save = figure_module._save_figure
    monkeypatch.setattr(figure_module, "_save_figure", capture_and_save)

    plot_rep_vs_ideo_heatmap(_rep_heatmap_cells(), tmp_path / "rep_heatmap.png")

    assert "Panel size 3" in observed_titles
    assert "Panel size 6" in observed_titles
    assert "Bias spread" in observed_labels
    assert "Mean expertise" in observed_labels
    assert any(text.endswith("*") for text in observed_text)


def test_plot_pre_post_ntqr_heatmap_accepts_strategy_facets(tmp_path: Path) -> None:
    cells = [
        {
            "strategy": strategy,
            "panel_size": panel,
            "mean_expertise": expertise,
            "bias_std": bias,
            "eie_minus_mv": -0.02 + panel / 300 + bias / 20 - expertise / 50,
        }
        for strategy in ("representative_sortition", "ideological_selection")
        for panel in (3, 6)
        for expertise in (0.68, 0.74)
        for bias in (0.1, 0.3)
    ]

    output_path = plot_pre_post_ntqr_heatmap(cells, tmp_path / "pre_post.png")

    _assert_nontrivial_png(output_path)


def test_plot_theory_alignment_heatmap_counts_prediction_matches(tmp_path: Path) -> None:
    payload = {
        "rep_vs_ideo_cells": _rep_heatmap_cells(),
        "monotone_checks": [
            {
                "axis": "bias_std",
                "status": "tested",
                "n_aligned": 2,
                "n_comparisons": 4,
            }
        ],
    }

    output_path = plot_theory_alignment_heatmap(payload, tmp_path / "alignment.png")

    _assert_nontrivial_png(output_path)


def test_plot_alarm_cost_curve_writes_png(tmp_path: Path) -> None:
    measured_points = DEFAULT_ALARM_MEASURED_POINTS + ((150, 650.0),)

    output_path = plot_alarm_cost_curve(
        tmp_path / "alarm_cost_curve.png",
        measured_points=measured_points,
    )

    assert output_path == tmp_path / "alarm_cost_curve.png"
    _assert_nontrivial_png(output_path)


def test_plot_alarm_cost_curve_uses_log_log_axes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_scales: list[tuple[str, str]] = []

    def capture_and_save(fig, output_path):
        ax = fig.axes[0]
        observed_scales.append((ax.get_xscale(), ax.get_yscale()))
        return original_save(fig, output_path)

    original_save = figure_module._save_figure
    monkeypatch.setattr(figure_module, "_save_figure", capture_and_save)

    plot_alarm_cost_curve(tmp_path / "alarm_cost_curve.png")

    assert observed_scales == [("log", "log")]


def test_plot_alarm_cost_curve_rejects_empty_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="alarm cost curve"):
        plot_alarm_cost_curve(tmp_path / "alarm_cost_curve.png", measured_points=())


def test_plot_power_curve_rejects_empty_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one sweep row"):
        plot_power_curve([], tmp_path / "power_curve.png")


def test_plot_strategy_ranking_rejects_wrong_row_width(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="strategy, mean, ci95"):
        plot_strategy_ranking([["random_selection", 0.1]], tmp_path / "r.png")


def test_plot_strategy_ranking_rejects_empty_strategy_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="strategy must be a non-empty string"):
        plot_strategy_ranking([["", 0.1, 0.02]], tmp_path / "r.png")


def test_plot_strategy_ranking_rejects_non_numeric_mean(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mean must be numeric"):
        plot_strategy_ranking([["random_selection", "x", 0.02]], tmp_path / "r.png")


def test_plot_strategy_ranking_rejects_non_finite_mean(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mean must be finite"):
        plot_strategy_ranking(
            [["random_selection", float("inf"), 0.02]], tmp_path / "r.png"
        )


def test_plot_strategy_ranking_rejects_negative_ci95(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ci95 must be non-negative"):
        plot_strategy_ranking([["random_selection", 0.1, -0.02]], tmp_path / "r.png")


def test_plot_rep_vs_ideo_effect_rejects_non_sequence_regime_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="regime_key must be a sequence"):
        plot_rep_vs_ideo_effect(
            [{"regime_key": 3, "effect": 0.1, "ci95": 0.0}], tmp_path / "e.png"
        )


def test_plot_rep_vs_ideo_effect_rejects_short_regime_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must include panel_size and mean_expertise"):
        plot_rep_vs_ideo_effect(
            [{"regime_key": [3], "effect": 0.1, "ci95": 0.0}], tmp_path / "e.png"
        )


def test_plot_alarm_cost_curve_rejects_malformed_point(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"expected \(Q, seconds\)"):
        plot_alarm_cost_curve(tmp_path / "a.png", measured_points=((20,),))


def test_plot_alarm_cost_curve_rejects_non_positive_q(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Q must be a positive integer"):
        plot_alarm_cost_curve(tmp_path / "a.png", measured_points=((0, 1.4),))


def test_plot_alarm_cost_curve_rejects_non_integer_q(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Q must be a positive integer"):
        plot_alarm_cost_curve(tmp_path / "a.png", measured_points=((2.5, 1.4),))


def test_plot_alarm_cost_curve_rejects_non_positive_seconds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="seconds must be positive"):
        plot_alarm_cost_curve(tmp_path / "a.png", measured_points=((20, 0.0),))


def _aggregate(
    *,
    rho: float,
    strategy: str,
    corr_mean: float,
    eie_mean: float,
    eie_ci95: float = 0.01,
) -> IndependenceAggregate:
    return IndependenceAggregate(
        rho=rho,
        strategy=strategy,
        panel_size=3,
        n_experts=24,
        n_items=120,
        n=5,
        eie_mean=eie_mean,
        eie_std=0.02,
        eie_ci95=eie_ci95,
        corr_mean=corr_mean,
    )


def test_plot_error_vs_correlation_accepts_mapping_rows(tmp_path: Path) -> None:
    aggregates = [
        {"rho": 0.0, "corr_mean": 0.02, "eie_mean": 0.05, "eie_ci95": 0.01, "strategy": "a"},
        {"rho": 0.0, "corr_mean": 0.10, "eie_mean": 0.12, "eie_ci95": 0.02, "strategy": "a"},
        {"rho": 0.5, "corr_mean": 0.30, "eie_mean": 0.25, "eie_ci95": 0.03, "strategy": "b"},
    ]

    output_path = plot_error_vs_correlation(aggregates, tmp_path / "error_vs_corr.png")

    assert output_path == tmp_path / "error_vs_corr.png"
    _assert_nontrivial_png(output_path)


def test_plot_error_vs_correlation_accepts_dataclass_instances(tmp_path: Path) -> None:
    aggregates = [
        _aggregate(rho=0.0, strategy="random_selection", corr_mean=0.03, eie_mean=0.06),
        _aggregate(rho=0.5, strategy="random_selection", corr_mean=0.28, eie_mean=0.21),
    ]

    output_path = plot_error_vs_correlation(aggregates, tmp_path / "error_vs_corr_dc.png")

    _assert_nontrivial_png(output_path)


def test_plot_error_vs_correlation_rejects_empty_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="independence aggregate"):
        plot_error_vs_correlation([], tmp_path / "error_vs_corr.png")


def test_plot_error_vs_correlation_rejects_all_degenerate_cells(tmp_path: Path) -> None:
    aggregates = [
        _aggregate(rho=0.0, strategy="a", corr_mean=0.02, eie_mean=-1.0),
        _aggregate(rho=0.5, strategy="b", corr_mean=0.30, eie_mean=float("nan")),
    ]

    with pytest.raises(ValueError, match="filtering degenerate"):
        plot_error_vs_correlation(aggregates, tmp_path / "error_vs_corr.png")


def test_plot_error_vs_correlation_different_inputs_differ(tmp_path: Path) -> None:
    """Negative control: a stub ignoring its input would write identical bytes."""
    dataset_a = [
        _aggregate(rho=0.0, strategy="a", corr_mean=0.02, eie_mean=0.05),
        _aggregate(rho=0.0, strategy="a", corr_mean=0.10, eie_mean=0.09),
    ]
    dataset_b = [
        _aggregate(rho=0.0, strategy="a", corr_mean=0.02, eie_mean=0.40),
        _aggregate(rho=0.0, strategy="a", corr_mean=0.10, eie_mean=0.70),
    ]

    path_a = plot_error_vs_correlation(dataset_a, tmp_path / "a.png")
    path_b = plot_error_vs_correlation(dataset_b, tmp_path / "b.png")
    path_a_again = plot_error_vs_correlation(dataset_a, tmp_path / "a_again.png")

    bytes_a = path_a.read_bytes()
    bytes_b = path_b.read_bytes()
    assert bytes_a != bytes_b
    # Determinism: identical input -> identical bytes.
    assert bytes_a == path_a_again.read_bytes()


def test_plot_alarm_power_writes_png(tmp_path: Path) -> None:
    curve = [(3, 0.0), (6, 0.4), (9, 0.8)]

    output_path = plot_alarm_power(curve, tmp_path / "alarm_power.png")

    assert output_path == tmp_path / "alarm_power.png"
    _assert_nontrivial_png(output_path)


def test_plot_alarm_power_rejects_empty_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="alarm power curve"):
        plot_alarm_power([], tmp_path / "alarm_power.png")


def test_plot_alarm_power_rejects_malformed_row(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"expected \(panel_size, power\)"):
        plot_alarm_power([(3,)], tmp_path / "alarm_power.png")


def test_plot_alarm_power_rejects_power_out_of_range(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"power must be in \[0, 1\]"):
        plot_alarm_power([(3, 1.5)], tmp_path / "alarm_power.png")


def test_plot_fairness_maximin_accepts_report(tmp_path: Path) -> None:
    report = FairnessReport(
        n_feasible_panels=10,
        min_selection_prob=0.1,
        mean_selection_prob=0.2,
        max_selection_prob=0.4,
        gini=0.15,
        realised_probabilities={"e0": 0.1, "e1": 0.2, "e2": 0.4, "e3": 0.1},
    )

    output_path = plot_fairness_maximin(report, tmp_path / "fairness.png")

    assert output_path == tmp_path / "fairness.png"
    _assert_nontrivial_png(output_path)


def test_plot_fairness_maximin_accepts_probability_mapping(tmp_path: Path) -> None:
    probs = {"e0": 0.25, "e1": 0.25, "e2": 0.5}

    output_path = plot_fairness_maximin(probs, tmp_path / "fairness_map.png")

    _assert_nontrivial_png(output_path)


def test_plot_fairness_maximin_rejects_empty_mapping(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        plot_fairness_maximin({}, tmp_path / "fairness.png")


def test_plot_fairness_maximin_rejects_non_mapping(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="probability mapping or FairnessReport"):
        plot_fairness_maximin([0.1, 0.2], tmp_path / "fairness.png")


def test_plot_fairness_maximin_rejects_negative_probability(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="probability must be non-negative"):
        plot_fairness_maximin({"e0": -0.1}, tmp_path / "fairness.png")


def test_plot_strategy_correlation_accepts_mapping_rows(tmp_path: Path) -> None:
    aggregates = [
        {"rho": 0.0, "corr_mean": 0.02, "eie_mean": 0.05, "eie_ci95": 0.01, "strategy": "random_selection"},
        {"rho": 0.5, "corr_mean": 0.20, "eie_mean": 0.18, "eie_ci95": 0.02, "strategy": "random_selection"},
        {"rho": 0.5, "corr_mean": 0.35, "eie_mean": 0.30, "eie_ci95": 0.03, "strategy": "expertise_threshold"},
    ]

    output_path = plot_strategy_correlation(aggregates, tmp_path / "strategy_corr.png")

    assert output_path == tmp_path / "strategy_corr.png"
    _assert_nontrivial_png(output_path)


def test_plot_strategy_correlation_accepts_dataclass_instances(tmp_path: Path) -> None:
    aggregates = [
        _aggregate(rho=0.0, strategy="random_selection", corr_mean=0.02, eie_mean=0.05),
        _aggregate(rho=0.5, strategy="expertise_threshold", corr_mean=0.35, eie_mean=0.30),
    ]

    output_path = plot_strategy_correlation(aggregates, tmp_path / "strategy_corr_dc.png")

    _assert_nontrivial_png(output_path)


def test_plot_strategy_correlation_rejects_empty_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="independence aggregate"):
        plot_strategy_correlation([], tmp_path / "strategy_corr.png")


def test_plot_strategy_correlation_rejects_empty_strategy_name(tmp_path: Path) -> None:
    aggregates = [
        {"rho": 0.0, "corr_mean": 0.02, "eie_mean": 0.05, "eie_ci95": 0.01, "strategy": ""},
    ]

    with pytest.raises(ValueError, match="strategy must be a non-empty string"):
        plot_strategy_correlation(aggregates, tmp_path / "strategy_corr.png")


# --- Session 6: cross-family + power design-diagnosis figures -----------------

def _cross_contrast(label: str = "live empirical, n-limited") -> dict[str, object]:
    return {
        "mean_abs_same_family": 0.0582,
        "mean_abs_cross_family": 0.0,
        "delta_cross_minus_same": -0.0582,
        "label": label,
    }


def test_plot_cross_family_contrast_writes_png(tmp_path: Path) -> None:
    out = plot_cross_family_contrast(_cross_contrast(), tmp_path / "cf.png")
    assert out == tmp_path / "cf.png"
    _assert_nontrivial_png(out)


def test_plot_cross_family_contrast_labels_scope_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_text: list[str] = []
    contrast = _cross_contrast()
    contrast.update(
        {
            "n_items": 150,
            "n_same_pairs": 2,
            "n_cross_pairs": 4,
            "nonzero_pairs": 1,
            "total_pairs": 6,
        }
    )

    def capture_and_save(fig, output_path):
        observed_text.extend(text.get_text() for text in fig.axes[0].texts)
        return original_save(fig, output_path)

    original_save = figure_module._save_figure
    monkeypatch.setattr(figure_module, "_save_figure", capture_and_save)

    plot_cross_family_contrast(contrast, tmp_path / "cf.png")

    joined = "\n".join(observed_text)
    assert "items=150" in joined
    assert "pairs: same=2, cross=4" in joined
    assert "nonzero=1/6" in joined
    assert "single-run test: none" in joined


def test_plot_cross_family_contrast_is_byte_deterministic(tmp_path: Path) -> None:
    a = plot_cross_family_contrast(_cross_contrast(), tmp_path / "a.png")
    b = plot_cross_family_contrast(_cross_contrast(), tmp_path / "b.png")
    assert a.read_bytes() == b.read_bytes()


def test_plot_cross_family_contrast_rejects_negative_corr(tmp_path: Path) -> None:
    bad = _cross_contrast()
    bad["mean_abs_same_family"] = -0.1
    with pytest.raises(ValueError):
        plot_cross_family_contrast(bad, tmp_path / "cf.png")


def test_plot_cross_family_contrast_rejects_empty_label(tmp_path: Path) -> None:
    bad = _cross_contrast(label="")
    with pytest.raises(ValueError):
        plot_cross_family_contrast(bad, tmp_path / "cf.png")


def _power_rows() -> list[dict[str, object]]:
    return [
        {"contrast": "a_vs_b", "observed_d": -1.3, "mde_80": 2.29},  # underpowered
        {"contrast": "c_vs_d", "observed_d": 3.98, "mde_80": 2.29},  # well-powered
    ]


def test_plot_power_design_diagnosis_writes_png(tmp_path: Path) -> None:
    out = plot_power_design_diagnosis(_power_rows(), tmp_path / "diag.png")
    assert out == tmp_path / "diag.png"
    _assert_nontrivial_png(out)


def test_plot_power_design_diagnosis_uses_readable_labels_and_stats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_yticks: list[str] = []
    observed_text: list[str] = []
    observed_titles: list[str] = []

    def capture_and_save(fig, output_path):
        ax = fig.axes[0]
        observed_yticks.extend(text.get_text() for text in ax.get_yticklabels())
        observed_text.extend(text.get_text() for text in ax.texts)
        observed_titles.append(ax.get_title())
        return original_save(fig, output_path)

    original_save = figure_module._save_figure
    monkeypatch.setattr(figure_module, "_save_figure", capture_and_save)
    rows = [
        {
            "contrast": "pair_random_selection__vs__representative_sortition_p6",
            "group_a": "random_selection",
            "group_b": "representative_sortition",
            "panel_size": 6,
            "observed_d": -0.4,
            "mde_80": 0.7,
            "perm_p": 0.125,
            "seeds_for_80": 48,
            "verdict": "underpowered-null",
        }
    ]

    plot_power_design_diagnosis(rows, tmp_path / "diag.png")

    assert observed_yticks == ["random selection vs representative sortition (p=6)"]
    assert any("p=0.125" in text for text in observed_text)
    assert any("n80=48" in text for text in observed_text)
    assert any("underpowered-null" in text for text in observed_text)
    assert "seed budget" not in observed_titles[0].lower()
    assert "design limits" in observed_titles[0]


def test_plot_modules_do_not_reintroduce_stale_generic_titles() -> None:
    source = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in [
            "src/ntqr_allotment/figure_parts/power.py",
            "src/ntqr_allotment/figure_parts/independence.py",
            "src/ntqr_allotment/figure_parts/fairness.py",
            "src/ntqr_allotment/figure_parts/ranking.py",
        ]
    )

    for stale_title in [
        "Power diagnosis: observed effect, MDE, p-value, and seed budget",
        "Statistical power vs sample size",
        "Alarm Power Curve vs Panel Size",
        "Error-Correlation Tolerance Curve",
        "Strategy Trio Error-Correlation",
        "Maximin Selection Probabilities",
        "Representative vs Ideological Selection Effect",
    ]:
        assert stale_title not in source


def test_plot_power_design_diagnosis_is_byte_deterministic(tmp_path: Path) -> None:
    a = plot_power_design_diagnosis(_power_rows(), tmp_path / "a.png")
    b = plot_power_design_diagnosis(_power_rows(), tmp_path / "b.png")
    assert a.read_bytes() == b.read_bytes()


def test_plot_power_design_diagnosis_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        plot_power_design_diagnosis([], tmp_path / "diag.png")


def test_plot_power_design_diagnosis_rejects_negative_mde(tmp_path: Path) -> None:
    bad = _power_rows()
    bad[0]["mde_80"] = -1.0
    with pytest.raises(ValueError):
        plot_power_design_diagnosis(bad, tmp_path / "diag.png")


def test_plot_cross_family_contrast_accepts_object_attr_rows(tmp_path: Path) -> None:
    """Covers the attribute-access (non-Mapping) branch via a real FamilyContrast."""
    from ntqr_allotment.cross_family import FamilyContrast

    contrast = FamilyContrast(
        mean_abs_same_family=0.0582,
        mean_abs_cross_family=0.0,
        delta_cross_minus_same=-0.0582,
        n_same_pairs=2,
        n_cross_pairs=4,
        label="live empirical, n-limited",
    )
    out = plot_cross_family_contrast(contrast, tmp_path / "cf_obj.png")
    _assert_nontrivial_png(out)


def test_plot_cross_family_multiseed_writes_png(tmp_path: Path) -> None:
    out = plot_cross_family_multiseed([-0.07, -0.049, -0.045, -0.042], tmp_path / "ms.png")
    _assert_nontrivial_png(out)


def test_plot_cross_family_multiseed_reports_sign_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_titles: list[str] = []
    observed_labels: list[str] = []

    def capture_and_save(fig, output_path):
        ax = fig.axes[0]
        observed_titles.append(ax.get_title())
        observed_labels.extend(text.get_text() for text in ax.get_legend().texts)
        return original_save(fig, output_path)

    original_save = figure_module._save_figure
    monkeypatch.setattr(figure_module, "_save_figure", capture_and_save)

    plot_cross_family_multiseed(
        [-0.07, -0.049, -0.045, -0.042],
        tmp_path / "ms.png",
    )

    assert "negative 4/4" in observed_titles[0]
    assert "sign p=0.125" in observed_titles[0]
    assert "mean 95% CI" in observed_labels


def test_plot_cross_family_multiseed_deterministic(tmp_path: Path) -> None:
    d = [-0.07, -0.049, -0.045, -0.042]
    a = plot_cross_family_multiseed(d, tmp_path / "a.png")
    b = plot_cross_family_multiseed(d, tmp_path / "b.png")
    assert a.read_bytes() == b.read_bytes()


def test_plot_cross_family_multiseed_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        plot_cross_family_multiseed([], tmp_path / "ms.png")


def test_plot_power_vs_n_writes_png(tmp_path: Path) -> None:
    curves = [
        ("d=0.2", [10, 40, 160], [0.1, 0.3, 0.7]),
        ("d=0.5", [10, 40, 160], [0.2, 0.6, 0.98]),
    ]
    out = plot_power_vs_n(curves, tmp_path / "pn.png")
    _assert_nontrivial_png(out)


def test_plot_power_vs_n_rejects_empty_and_malformed(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        plot_power_vs_n([], tmp_path / "pn.png")
    with pytest.raises(ValueError):
        plot_power_vs_n([("bad", [1, 2])], tmp_path / "pn.png")  # not a triple


def _postdoc_payload() -> dict[str, object]:
    aggregates = []
    for track in ("analytical", "live"):
        for strategy, base in [
            ("expertise_threshold", 0.08),
            ("representative_sortition", 0.12),
            ("random_selection", 0.14),
            ("ideological_selection", 0.19),
        ]:
            for panel_size in (3, 6):
                aggregates.append(
                    {
                        "track": track,
                        "strategy": strategy,
                        "strategy_label": strategy.replace("_", " "),
                        "panel_size": panel_size,
                        "n": 2,
                        "eie_mean": base + (0.02 if track == "live" else 0.0),
                        "eie_ci_low": base - 0.01,
                        "eie_ci_high": base + 0.01,
                        "mv_mean": base + 0.04,
                        "age_disparity_mean": 0.04 if strategy == "ideological_selection" else 0.01,
                        "age_disparity_ci_low": 0.0,
                        "age_disparity_ci_high": 0.06,
                    }
                )
    return {"aggregates": aggregates}


def test_plot_postdoc_strategy_ranking_distinguishes_tracks(tmp_path: Path) -> None:
    out = plot_postdoc_strategy_ranking(
        _postdoc_payload(),
        tmp_path / "postdoc_strategy_ranking.png",
    )

    _assert_nontrivial_png(out)


def test_plot_postdoc_age_bias_heatmap_labels_disparity_units(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_labels: list[str] = []
    observed_title: list[str] = []

    def capture_and_save(fig, output_path):
        ax = fig.axes[0]
        observed_labels.append(ax.get_xlabel())
        observed_labels.append(ax.get_ylabel())
        observed_title.append(ax.get_title())
        return original_save(fig, output_path)

    original_save = figure_module._save_figure
    monkeypatch.setattr(figure_module, "_save_figure", capture_and_save)

    plot_postdoc_age_bias_heatmap(_postdoc_payload(), tmp_path / "bias.png")

    assert "Panel size" in observed_labels
    assert "Sampling strategy" in observed_labels
    assert "age-disparity" in observed_title[0]


def test_plot_postdoc_empirical_alignment_marks_unresolved_cells(tmp_path: Path) -> None:
    payload = {
        "cells": [
            {
                "strategy": "representative_sortition",
                "strategy_label": "representative sortition",
                "panel_size": 3,
                "sign_agrees": True,
                "unresolved": False,
            },
            {
                "strategy": "ideological_selection",
                "strategy_label": "same-bias selection",
                "panel_size": 3,
                "sign_agrees": False,
                "unresolved": True,
            },
        ],
        "caveat": "single-model live companion; descriptive only",
    }

    out = plot_postdoc_empirical_alignment(payload, tmp_path / "alignment.png")

    _assert_nontrivial_png(out)


# --- Method schematic + cross-track ranking inversion -------------------------

def _matched_grain_rows() -> tuple[list, list]:
    """Synthetic size-3 vs live three-seat rows whose extremes invert."""
    synthetic = [
        ["expertise_threshold", 0.037],
        ["random_selection", 0.439],
        ["representative_sortition", 0.577],
        ["ideological_selection", 0.503],
    ]
    live = [
        ["representative_sortition", 0.230],
        ["ideological_selection", 0.243],
        ["random_selection", 0.254],
        ["expertise_threshold", 0.382],
    ]
    return synthetic, live


def test_plot_method_pipeline_schematic_writes_png(tmp_path: Path) -> None:
    out = plot_method_pipeline_schematic(
        {"n_experts": 96, "n_items": 300, "n_trios": 8},
        tmp_path / "schematic.png",
    )
    assert out == tmp_path / "schematic.png"
    _assert_nontrivial_png(out)


def test_plot_method_pipeline_schematic_is_readable_and_uses_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_sizes: list[float] = []
    observed_text: list[str] = []
    observed_titles: list[str] = []

    def capture_and_save(fig, output_path):
        observed_sizes.extend(_visible_text_sizes(fig))
        observed_text.extend(
            text.get_text() for text in fig.findobj(Text) if text.get_visible()
        )
        observed_titles.extend(ax.get_title() for ax in fig.axes if ax.get_title())
        return original_save(fig, output_path)

    original_save = figure_module._save_figure
    monkeypatch.setattr(figure_module, "_save_figure", capture_and_save)

    plot_method_pipeline_schematic(
        {"n_experts": 96, "n_items": 300, "n_trios": 8},
        tmp_path / "schematic.png",
    )

    joined = "\n".join(observed_text)
    assert "96 experts" in joined
    assert "300 items" in joined
    assert "8 trios" in joined
    assert min(observed_sizes) >= SOURCE_NOTE_FONT_SIZE
    assert any(size >= TITLE_FONT_SIZE for size in observed_sizes)
    assert "The instrument measures upstream panel formation, not the estimator" in observed_titles


def test_plot_method_pipeline_schematic_is_byte_deterministic(tmp_path: Path) -> None:
    counts = {"n_experts": 96, "n_items": 300, "n_trios": 8}
    a = plot_method_pipeline_schematic(counts, tmp_path / "a.png")
    b = plot_method_pipeline_schematic(counts, tmp_path / "b.png")
    assert a.read_bytes() == b.read_bytes()


def test_plot_method_pipeline_schematic_rejects_bad_counts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="n_experts"):
        plot_method_pipeline_schematic({"n_items": 300, "n_trios": 8}, tmp_path / "s.png")
    with pytest.raises(ValueError, match="n_trios"):
        plot_method_pipeline_schematic(
            {"n_experts": 96, "n_items": 300, "n_trios": 0}, tmp_path / "s.png"
        )
    with pytest.raises(ValueError, match="must be a mapping"):
        plot_method_pipeline_schematic([96, 300, 8], tmp_path / "s.png")


def test_plot_track_ranking_inversion_writes_png(tmp_path: Path) -> None:
    synthetic, live = _matched_grain_rows()
    out = plot_track_ranking_inversion(synthetic, live, tmp_path / "inversion.png")
    assert out == tmp_path / "inversion.png"
    _assert_nontrivial_png(out)


def test_plot_track_ranking_inversion_is_readable_and_labels_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_sizes: list[float] = []
    observed_text: list[str] = []
    observed_titles: list[str] = []

    def capture_and_save(fig, output_path):
        observed_sizes.extend(_visible_text_sizes(fig))
        observed_text.extend(text.get_text() for text in fig.axes[0].texts)
        observed_titles.append(fig.axes[0].get_title())
        return original_save(fig, output_path)

    original_save = figure_module._save_figure
    monkeypatch.setattr(figure_module, "_save_figure", capture_and_save)

    synthetic, live = _matched_grain_rows()
    plot_track_ranking_inversion(synthetic, live, tmp_path / "inversion.png")

    assert "0.037" in observed_text  # synthetic-best expertise threshold
    assert "0.382" in observed_text  # live-worst expertise threshold
    assert min(observed_sizes) >= SOURCE_NOTE_FONT_SIZE
    assert "Strategy ranking inverts between the synthetic and live tracks" in observed_titles


def test_plot_track_ranking_inversion_is_byte_deterministic(tmp_path: Path) -> None:
    synthetic, live = _matched_grain_rows()
    a = plot_track_ranking_inversion(synthetic, live, tmp_path / "a.png")
    b = plot_track_ranking_inversion(synthetic, live, tmp_path / "b.png")
    assert a.read_bytes() == b.read_bytes()


def test_plot_track_ranking_inversion_different_inputs_differ(tmp_path: Path) -> None:
    """Negative control: a stub ignoring its input would write identical bytes."""
    synthetic, live = _matched_grain_rows()
    # A non-inverting live ranking that mirrors the synthetic order.
    aligned_live = [
        ["expertise_threshold", 0.230],
        ["random_selection", 0.243],
        ["ideological_selection", 0.254],
        ["representative_sortition", 0.382],
    ]
    path_inverted = plot_track_ranking_inversion(synthetic, live, tmp_path / "inv.png")
    path_aligned = plot_track_ranking_inversion(synthetic, aligned_live, tmp_path / "aln.png")
    assert path_inverted.read_bytes() != path_aligned.read_bytes()


def test_plot_track_ranking_inversion_rejects_bad_input(tmp_path: Path) -> None:
    synthetic, live = _matched_grain_rows()
    with pytest.raises(ValueError, match="synthetic track has no rows"):
        plot_track_ranking_inversion([], live, tmp_path / "x.png")
    with pytest.raises(ValueError, match="expected .strategy, eie_error."):
        plot_track_ranking_inversion([["expertise_threshold"]], live, tmp_path / "x.png")
    with pytest.raises(ValueError, match="duplicate strategy"):
        dup = [["a", 0.1], ["a", 0.2]]
        plot_track_ranking_inversion(dup, dup, tmp_path / "x.png")
    with pytest.raises(ValueError, match="must rank the same strategies"):
        plot_track_ranking_inversion(
            synthetic, live[:3] + [["other_strategy", 0.4]], tmp_path / "x.png"
        )
    with pytest.raises(ValueError, match="eie_error must be non-negative"):
        plot_track_ranking_inversion(
            [["a", -0.1], ["b", 0.2]], [["a", 0.1], ["b", 0.2]], tmp_path / "x.png"
        )


# --------------------------------------------------------------------------- #
# Data-binding negative controls (RedTeam viz + oracle audit): a figure that
# plots swapped/wrong data must FAIL, not just produce a >1000-byte PNG.
# --------------------------------------------------------------------------- #
def test_plot_strategy_ranking_bar_widths_equal_means(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each bar's width must equal that strategy's mean EIE error.

    Size-only assertions pass even if the bars plot the wrong values; this binds
    the geometry to the input so a swapped/scaled series is caught.
    """
    captured: dict[str, list[float]] = {}

    def capture_and_save(fig, output_path):
        ax = fig.axes[0]
        captured["widths"] = sorted(round(p.get_width(), 6) for p in ax.patches)
        return original_save(fig, output_path)

    original_save = figure_module._save_figure
    monkeypatch.setattr(figure_module, "_save_figure", capture_and_save)

    ranking = [
        ["representative_sortition", 0.11, 0.02],
        ["expertise_threshold", 0.08, 0.01],
        ["random_selection", 0.14, 0.03],
    ]
    plot_strategy_ranking(ranking, tmp_path / "r.png")
    assert captured["widths"] == [0.08, 0.11, 0.14]  # exactly the input means


def test_plot_track_ranking_inversion_binds_value_to_column_and_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The inversion must place each value at the correct (column, rank).

    Synthetic-best expertise (0.037) belongs in the LEFT column at the TOP rank;
    live-worst expertise (0.382) in the RIGHT column at the BOTTOM rank. A column
    swap or a rank-direction flip (n-rank -> rank) moves these and fails here,
    closing the green-by-construction gap the value-text-only test left open.
    """
    positions: dict[str, tuple[float, float]] = {}

    def capture_and_save(fig, output_path):
        ax = fig.axes[0]
        for t in ax.texts:
            if t.get_text() in ("0.037", "0.382"):
                positions[t.get_text()] = (round(t.xy[0], 3), round(t.xy[1], 3))
        return original_save(fig, output_path)

    original_save = figure_module._save_figure
    monkeypatch.setattr(figure_module, "_save_figure", capture_and_save)

    synthetic, live = _matched_grain_rows()
    plot_track_ranking_inversion(synthetic, live, tmp_path / "inv.png")

    assert "0.037" in positions and "0.382" in positions
    syn_x, syn_y = positions["0.037"]
    live_x, live_y = positions["0.382"]
    assert syn_x == 0.0       # synthetic (left) column
    assert live_x == 1.0      # live (right) column
    assert syn_y > live_y     # best rank (top) is above worst rank (bottom)
