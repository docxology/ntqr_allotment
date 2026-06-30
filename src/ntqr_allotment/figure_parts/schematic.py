from __future__ import annotations

from pathlib import Path
from typing import Mapping

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ntqr_allotment.figure_parts._common import (
    ANNOTATION_FONT_SIZE,
    AXIS_LABEL_FONT_SIZE,
    _add_source_note,
    _coerce_positive_int,
    _set_claim_title,
    plt,
    save_figure,
)

# Stage palette: alternating slate tones with the scoring stage highlighted.
_STAGE_EDGE = ("#34495E", "#5D6D7E", "#34495E", "#5D6D7E", "#1F618D")
_ARROW_COLOR = "#566573"
_UNSUPERVISED_COLOR = "#7D3C98"
_SCORING_COLOR = "#1F618D"


def _validated_stage_counts(stage_counts: Mapping[str, object]) -> tuple[int, int, int]:
    """Validate the three integer counts the schematic annotates.

    The counts are read by the orchestrator from ``manuscript_variables.json``
    (tokens ``N_EXPERTS``/``N_ITEMS``/``N_TRIOS``) so the schematic never
    hardcodes a population/corpus size that could drift from the manuscript.
    """

    if not isinstance(stage_counts, Mapping):
        raise ValueError("pipeline schematic: stage_counts must be a mapping")
    context = "pipeline schematic stage_counts"
    n_experts = _coerce_positive_int(stage_counts.get("n_experts"), "n_experts", context)
    n_items = _coerce_positive_int(stage_counts.get("n_items"), "n_items", context)
    n_trios = _coerce_positive_int(stage_counts.get("n_trios"), "n_trios", context)
    return n_experts, n_items, n_trios


def plot_method_pipeline_schematic(
    stage_counts: Mapping[str, object],
    output_path: Path,
) -> Path:
    """Render the deterministic-instrument pipeline schematic (front matter).

    A one-glance model of the synthetic instrument: population -> panel
    formation -> blind NTQR error-independent evaluation over ensembles of
    trios -> oracle scoring. It is explanatory front matter, *not* an empirical
    result; every number is supplied by the caller from real tokens.
    """

    n_experts, n_items, n_trios = _validated_stage_counts(stage_counts)

    fig, ax = plt.subplots(figsize=(13.0, 5.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    stages = (
        (
            "Synthetic\nexpert population",
            f"{n_experts} experts\nexpertise · bias ·\nheterogeneity",
        ),
        (
            "Panel formation\n(4 strategies)",
            "representative · random ·\nexpertise-threshold ·\nsingle-bloc",
        ),
        (
            f"Panel judges\n{n_items} items",
            "binary votes;\nlabels hidden from\nthe estimator",
        ),
        (
            "NTQR EIE\n(blind, no key)",
            f"ensemble of up to\n{n_trios} trios; ≤2 logically\nconsistent solutions",
        ),
        (
            "Oracle scoring",
            "L1 recovery error\n(prevalence + per-judge\naccuracy)",
        ),
    )

    n = len(stages)
    box_w, box_h = 16.0, 40.0
    gap = (100 - n * box_w) / (n + 1)
    center_y = 52.0
    centers: list[float] = []
    for i, (title, body) in enumerate(stages):
        color = _STAGE_EDGE[i]
        x0 = gap + i * (box_w + gap)
        cx = x0 + box_w / 2
        centers.append(cx)
        ax.add_patch(
            FancyBboxPatch(
                (x0, center_y - box_h / 2),
                box_w,
                box_h,
                boxstyle="round,pad=0.6,rounding_size=2.2",
                linewidth=1.6,
                edgecolor=color,
                facecolor=color,
                alpha=0.12,
            )
        )
        ax.text(
            cx,
            center_y + box_h / 2 - 4.5,
            title,
            ha="center",
            va="top",
            fontsize=AXIS_LABEL_FONT_SIZE,
            fontweight="bold",
            color=color,
        )
        ax.text(
            cx,
            center_y - box_h / 2 + 4.5,
            body,
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONT_SIZE,
            color="#2c3e50",
        )

    for i in range(n - 1):
        start = centers[i] + box_w / 2
        end = centers[i + 1] - box_w / 2
        ax.add_patch(
            FancyArrowPatch(
                (start, center_y),
                (end, center_y),
                arrowstyle="-|>",
                mutation_scale=20,
                linewidth=1.8,
                color=_ARROW_COLOR,
            )
        )

    ax.annotate(
        "no answer key (unsupervised)",
        xy=((centers[2] + centers[3]) / 2, center_y + box_h / 2 + 3),
        ha="center",
        va="bottom",
        fontsize=ANNOTATION_FONT_SIZE,
        color=_UNSUPERVISED_COLOR,
        fontweight="bold",
    )
    ax.annotate(
        "known labels used\nonly for scoring",
        xy=(centers[4], center_y - box_h / 2 - 3),
        ha="center",
        va="top",
        fontsize=ANNOTATION_FONT_SIZE,
        color=_SCORING_COLOR,
        fontweight="bold",
    )

    _set_claim_title(
        ax,
        "The instrument measures upstream panel formation, not the estimator",
    )
    _add_source_note(
        fig,
        "Schematic of the deterministic synthetic instrument "
        "(src/ntqr_allotment/pipeline.py); explanatory front matter, not an empirical result.",
        y=0.015,
    )
    return save_figure(fig, output_path)
