from __future__ import annotations

from ntqr_allotment.figure_parts._common import (
    DEFAULT_ALARM_MEASURED_POINTS,
    _save_figure as _save_figure,
)
from ntqr_allotment.figure_parts.empirical import (
    plot_cross_family_contrast,
    plot_cross_family_multiseed,
    plot_postdoc_age_bias_heatmap,
    plot_postdoc_empirical_alignment,
    plot_postdoc_strategy_ranking,
)
from ntqr_allotment.figure_parts.fairness import plot_fairness_maximin
from ntqr_allotment.figure_parts.heatmaps import (
    plot_pre_post_ntqr_heatmap,
    plot_rep_vs_ideo_heatmap,
    plot_theory_alignment_heatmap,
)
from ntqr_allotment.figure_parts.independence import (
    plot_error_vs_correlation,
    plot_strategy_correlation,
)
from ntqr_allotment.figure_parts.power import (
    plot_alarm_cost_curve,
    plot_alarm_power,
    plot_power_curve,
    plot_power_design_diagnosis,
    plot_power_vs_n,
)
from ntqr_allotment.figure_parts.ranking import (
    plot_rep_vs_ideo_effect,
    plot_strategy_ranking,
    plot_track_ranking_inversion,
)
from ntqr_allotment.figure_parts.mechanism import plot_trio_conditioning
from ntqr_allotment.figure_parts.phase import plot_bloc_phase_diagram, plot_concentration_dial
from ntqr_allotment.figure_parts.schematic import plot_method_pipeline_schematic

__all__ = [
    "DEFAULT_ALARM_MEASURED_POINTS",
    "plot_bloc_phase_diagram",
    "plot_concentration_dial",
    "plot_trio_conditioning",
    "plot_alarm_cost_curve",
    "plot_alarm_power",
    "plot_cross_family_contrast",
    "plot_cross_family_multiseed",
    "plot_error_vs_correlation",
    "plot_fairness_maximin",
    "plot_method_pipeline_schematic",
    "plot_postdoc_age_bias_heatmap",
    "plot_postdoc_empirical_alignment",
    "plot_postdoc_strategy_ranking",
    "plot_pre_post_ntqr_heatmap",
    "plot_power_curve",
    "plot_power_design_diagnosis",
    "plot_power_vs_n",
    "plot_rep_vs_ideo_effect",
    "plot_rep_vs_ideo_heatmap",
    "plot_strategy_correlation",
    "plot_strategy_ranking",
    "plot_theory_alignment_heatmap",
    "plot_track_ranking_inversion",
]
