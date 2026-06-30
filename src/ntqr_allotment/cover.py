from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from ntqr_allotment.config import ExperimentProfile, load_experiment_profile

COVER_FILENAME = "ntqr_cover.png"
MANIFEST_FILENAME = "cover_manifest.json"
LARGE_CONCEPT = (
    "external design-tool concept render (large/page layout); "
    "not part of the regeneration pipeline"
)
MOBILE_CONCEPT = (
    "external design-tool concept render (mobile/portrait layout); "
    "not part of the regeneration pipeline"
)

IVORY = "#fbf7ef"
INK = "#172026"
MUTED = "#61707a"
LINE = "#c9d2d8"
TEAL = "#0b6b74"
TEAL_LIGHT = "#c7e3e1"
GOLD = "#b77a25"
GOLD_LIGHT = "#f0d6a7"
RED = "#a63c32"
PANEL = "#ffffff"


@dataclass(frozen=True)
class CoverPaths:
    cover: Path
    manifest: Path


@dataclass(frozen=True)
class CoverCase:
    panel_size: int
    bias_std: float

    @property
    def label(self) -> str:
        return f"n={self.panel_size} | bias {self.bias_std:.2f}"


def render_cover(
    repo_root: Path,
    *,
    output_path: Path | None = None,
    manifest_path: Path | None = None,
) -> CoverPaths:
    root = Path(repo_root)
    cover_path = output_path or root / "output" / "figures" / COVER_FILENAME
    manifest = manifest_path or root / "output" / "data" / MANIFEST_FILENAME
    config_path = root / "manuscript" / "config.yaml"
    config_hash = _file_hash(config_path)
    profile = load_experiment_profile(config_path)
    cases = _cover_cases(profile)

    _draw_cover(cover_path, profile=profile, cases=cases)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            _manifest_payload(
                root,
                cover_path,
                manifest,
                config_path,
                config_hash,
                profile,
                cases,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return CoverPaths(cover=cover_path, manifest=manifest)


def _draw_cover(
    output_path: Path,
    *,
    profile: ExperimentProfile,
    cases: tuple[CoverCase, ...],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(17.6, 9.0), dpi=220)
    fig.patch.set_facecolor(IVORY)
    ax.set_facecolor(IVORY)
    ax.set_xlim(0, 178)
    ax.set_ylim(0, 100)
    ax.axis("off")

    _frame(ax)
    _lottery(ax)
    _panel_compositions(ax)
    _confound_coupling(ax)
    _matrix(ax)
    _phase_fanout(ax)
    _case_grid(ax, profile, cases)
    _flow_arrows(ax)
    _cover_caveat(ax)

    fig.savefig(output_path, facecolor=IVORY, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _frame(ax: plt.Axes) -> None:
    ax.add_patch(Rectangle((3, 3), 172, 94, fill=False, edgecolor=LINE, linewidth=1.3))
    ax.add_patch(
        Rectangle((5.5, 5.5), 167, 89, fill=False, edgecolor="#e4ded2", linewidth=0.7)
    )
    # Title band, left-anchored so the phase-transition hero owns the right.
    ax.text(
        11,
        89.5,
        "SORTITION",
        ha="left",
        va="center",
        color=INK,
        fontsize=33,
        fontweight="bold",
    )
    ax.text(
        11,
        82.6,
        "upstream of NTQR",
        ha="left",
        va="center",
        color=TEAL,
        fontsize=20,
        fontweight="bold",
    )
    ax.text(
        11,
        77.4,
        "how the panel is drawn sets the ceiling on ground-truth-free recovery",
        ha="left",
        color=MUTED,
        fontsize=11.5,
    )
    # Three-act spine labels across the top.
    for x, label in ((30, "1 · draw the panel"), (84, "2 · errors couple"), (140, "3 · phase transition")):
        ax.text(x, 70.5, label, ha="center", color=MUTED, fontsize=10.0, fontweight="bold")


#: Three ideological blocs get three distinct token colours; this single mapping
#: drives both the lottery urn and the panel-composition chips so the visual story
#: (representative = mixed colours, single-bloc = one colour) is internally honest.
_BLOC_FILL = ("#7fb8d6", GOLD_LIGHT, "#d79b9b")
_BLOC_EDGE = (TEAL, GOLD, RED)


def _lottery(ax: plt.Axes) -> None:
    ax.add_patch(FancyBboxPatch((9, 40), 22, 26, boxstyle="round,pad=0.8,rounding_size=6", facecolor=PANEL, edgecolor=GOLD, linewidth=1.8))
    ax.add_patch(Rectangle((12, 63), 16, 4.4, facecolor=GOLD_LIGHT, edgecolor=GOLD, linewidth=1.1))
    # A mixed urn of three-bloc tokens waiting to be drawn.
    token_positions = [(14, 58), (20, 55), (26, 58), (16, 49), (23, 47), (28, 52), (19, 43)]
    for index, (x, y) in enumerate(token_positions):
        fill = _BLOC_FILL[index % 3]
        edge = _BLOC_EDGE[index % 3]
        ax.add_patch(Circle((x, y), 2.7, facecolor=fill, edgecolor=edge, linewidth=0.8))
    ax.text(20, 36.4, "audited maximin draw", ha="center", color=MUTED, fontsize=9.6)
    ax.text(20, 33.0, "(3 ideological blocs)", ha="center", color=MUTED, fontsize=8.2)


def _panel_compositions(ax: plt.Axes) -> None:
    """Two contrasting panels the draw can yield: balanced vs single-bloc."""
    specs = (
        (38, 56, "representative", "balanced blocs", (0, 1, 2, 0, 1, 2), TEAL),
        (38, 33, "single-bloc", "one ideology", (2, 2, 2, 2, 2, 2), RED),
    )
    for x0, y0, title, sub, blocs, accent in specs:
        ax.add_patch(
            FancyBboxPatch(
                (x0, y0 - 7.5), 24, 13.5,
                boxstyle="round,pad=0.4,rounding_size=3",
                facecolor="#fffdf8", edgecolor=accent, linewidth=1.3,
                linestyle="-" if accent == TEAL else "--",
            )
        )
        for j, b in enumerate(blocs):
            cx = x0 + 4.0 + (j % 3) * 5.4
            cy = y0 + 1.7 - (j // 3) * 5.0
            ax.add_patch(Circle((cx, cy), 1.9, facecolor=_BLOC_FILL[b], edgecolor=_BLOC_EDGE[b], linewidth=0.7))
        ax.text(x0 + 12, y0 + 4.4, title, ha="center", color=INK, fontsize=9.4, fontweight="bold")
        ax.text(x0 + 12, y0 - 9.6, sub, ha="center", color=MUTED, fontsize=8.0)


def _confound_coupling(ax: plt.Axes) -> None:
    """Act 2: same-bloc judges share a latent confound -> correlated errors."""
    # Decorrelated trio (mixed blocs) vs correlated trio (one bloc) feeding NTQR.
    trios = (
        (74, 58, (0, 1, 2), TEAL, False, "errors decorrelate"),
        (74, 36, (2, 2, 2), RED, True, "errors co-move"),
    )
    for cx, cy, blocs, accent, correlated, caption in trios:
        nodes = [(cx - 6, cy - 3), (cx, cy + 4), (cx + 6, cy - 3)]
        ax.plot(
            [nodes[0][0], nodes[1][0], nodes[2][0], nodes[0][0]],
            [nodes[0][1], nodes[1][1], nodes[2][1], nodes[0][1]],
            color=accent if correlated else LINE,
            linewidth=2.0 if correlated else 1.0,
            alpha=0.9 if correlated else 0.7,
        )
        if correlated:
            # A shared latent star wired to all three same-bloc nodes.
            sx, sy = cx, cy - 8.5
            ax.add_patch(Circle((sx, sy), 1.5, facecolor=accent, edgecolor=accent, linewidth=0.8))
            for nx, ny in nodes:
                ax.plot([sx, nx], [sy, ny], color=accent, linewidth=0.7, linestyle=":", alpha=0.8)
            ax.text(sx + 7.5, sy - 0.2, "shared\nconfound", ha="left", va="center", color=accent, fontsize=7.2)
        for (nx, ny), b in zip(nodes, blocs):
            ax.add_patch(Circle((nx, ny), 2.4, facecolor=_BLOC_FILL[b], edgecolor=_BLOC_EDGE[b], linewidth=1.0))
        ax.text(cx, cy + 7.6, caption, ha="center", color=accent, fontsize=8.4, fontweight="bold")


def _matrix(ax: plt.Axes) -> None:
    left, bottom, size = 96, 40, 3.6
    matrix_colors = [[TEAL, TEAL_LIGHT, GOLD_LIGHT], [TEAL_LIGHT, TEAL, "#e9edf0"], [GOLD_LIGHT, "#e9edf0", TEAL]]
    for row in range(3):
        for col in range(3):
            ax.add_patch(Rectangle((left + col * size, bottom + (2 - row) * size), size, size, facecolor=matrix_colors[row][col], edgecolor=INK, linewidth=0.6))
    ax.text(left + size * 1.5, bottom + 13, "agreement", ha="center", color=MUTED, fontsize=8.6)
    ax.text(left + size * 1.5, bottom - 3.6, "NTQR EIE", ha="center", color=INK, fontsize=12, fontweight="bold")
    ax.text(left + size * 1.5, bottom - 7.4, "no answer key", ha="center", color=MUTED, fontsize=7.8)


def _phase_fanout(ax: plt.Axes) -> None:
    """Act 3 (hero): stylized recovery-error curves fanning out with coupling.

    Schematic, not measured values: the cover's contract forbids baking empirical
    numbers into the raster. The curves only encode the qualitative finding -
    representative stays flat, single-bloc climbs, random sits between.
    """
    import numpy as np

    ox, oy, w, h = 118, 18, 52, 44  # plot box
    # Axes.
    ax.add_patch(Rectangle((ox, oy), w, h, facecolor="#fffdf8", edgecolor=LINE, linewidth=1.0))
    ax.add_patch(FancyArrowPatch((ox, oy), (ox + w + 2, oy), arrowstyle="-|>", mutation_scale=12, color=INK, linewidth=1.1))
    ax.add_patch(FancyArrowPatch((ox, oy), (ox, oy + h + 2), arrowstyle="-|>", mutation_scale=12, color=INK, linewidth=1.1))
    ax.text(ox + w / 2, oy - 4.6, "within-bloc error coupling  ρ", ha="center", color=MUTED, fontsize=9.2)
    ax.text(ox - 3.0, oy + h / 2, "recovery error", ha="center", va="center", rotation=90, color=MUTED, fontsize=9.2)

    t = np.linspace(0.0, 1.0, 60)
    # (start, end, curvature, color, label) - deterministic illustrative curves.
    curves = (
        (0.06, 0.22, 1.0, "#4C78A8", "expertise"),
        (0.55, 0.50, 0.4, TEAL, "representative"),
        (0.60, 0.78, 1.6, "#72B7B2", "random"),
        (0.58, 0.97, 2.4, RED, "single-bloc"),
    )
    for y_lo, y_hi, curv, color, label in curves:
        y = y_lo + (y_hi - y_lo) * (t ** curv)
        px = ox + 3 + t * (w - 6)
        py = oy + 4 + y * (h - 8)
        ax.plot(px, py, color=color, linewidth=2.6, solid_capstyle="round")
        ax.text(px[-1] + 1.2, py[-1], label, ha="left", va="center", color=color, fontsize=8.2, fontweight="bold")
    # Collapse bracket at the left edge (the reproduced baseline).
    ax.annotate(
        "collapse\nat ρ=0", xy=(ox + 3.4, oy + 4 + 0.57 * (h - 8)), xytext=(ox + 8.5, oy + h - 5),
        ha="center", color=MUTED, fontsize=7.6,
        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7),
    )
    ax.text(ox + w / 2, oy + h + 5.2, "representativeness is protective", ha="center", color=INK, fontsize=10.5, fontweight="bold")


def _flow_arrows(ax: plt.Axes) -> None:
    for start, end, color in [
        ((31, 49), (37, 52), GOLD),       # urn -> representative panel
        ((31, 47), (37, 35), GOLD),       # urn -> single-bloc panel
        ((62, 54), (67, 56), TEAL),       # panels -> trios
        ((62, 34), (67, 36), RED),
        ((86, 47), (95, 47), TEAL),       # trios -> NTQR matrix
        ((112, 47), (118, 42), INK),      # NTQR -> phase diagram
    ]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color=color))


def _case_grid(
    ax: plt.Axes,
    profile: ExperimentProfile,
    cases: tuple[CoverCase, ...],
) -> None:
    # Subordinate bottom strip: the panel-size x bias case grid (manifest contract).
    ax.add_patch(
        FancyBboxPatch(
            (9, 6),
            101,
            18,
            boxstyle="round,pad=0.5,rounding_size=3",
            facecolor="#fffdf8",
            edgecolor=LINE,
            linewidth=0.9,
        )
    )
    ax.text(
        13,
        20.4,
        "sortition cases considered",
        ha="left",
        color=INK,
        fontsize=10.6,
        fontweight="bold",
    )
    ax.text(
        13,
        16.8,
        f"profile {profile.name}: size x bias corners",
        ha="left",
        color=MUTED,
        fontsize=8.0,
    )

    grid_cases = _corner_cases(cases)
    positions = [(46.0, 13.6), (62.0, 13.6), (78.0, 13.6), (94.0, 13.6)]
    for case, (x, y) in zip(grid_cases, positions, strict=False):
        is_high_bias = case.bias_std >= max(item.bias_std for item in cases)
        edge = RED if is_high_bias else TEAL
        fill = "#fbebe8" if is_high_bias else "#eef8f7"
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                14.5,
                7.6,
                boxstyle="round,pad=0.22,rounding_size=2",
                facecolor=fill,
                edgecolor=edge,
                linewidth=0.8,
            )
        )
        _mini_panel(ax, x + 4.4, y + 4.6, case.panel_size, edge)
        _bias_glyph(ax, x + 8.0, y + 5.4, case.bias_std, edge)
        ax.text(x + 7.2, y + 1.4, case.label, ha="center", color=INK, fontsize=6.4)


def _mini_panel(
    ax: plt.Axes,
    cx: float,
    cy: float,
    panel_size: int,
    edge: str,
) -> None:
    offsets = {
        3: [(-2.1, -1.2), (0, 1.7), (2.1, -1.2)],
        6: [(-2.4, -1.4), (-1.4, 1.5), (0, -0.4), (1.4, 1.5), (2.4, -1.4), (0, 2.9)],
        9: [
            (-2.5, -1.8),
            (0, -1.8),
            (2.5, -1.8),
            (-2.5, 0.3),
            (0, 0.3),
            (2.5, 0.3),
            (-2.5, 2.4),
            (0, 2.4),
            (2.5, 2.4),
        ],
        12: [
            (-3.0, -1.8),
            (-1.0, -1.8),
            (1.0, -1.8),
            (3.0, -1.8),
            (-3.0, 0.1),
            (-1.0, 0.1),
            (1.0, 0.1),
            (3.0, 0.1),
            (-3.0, 2.0),
            (-1.0, 2.0),
            (1.0, 2.0),
            (3.0, 2.0),
        ],
    }
    points = offsets.get(panel_size)
    if points is None:
        count = min(panel_size, 12)
        points = [((i % 4) * 1.25 - 2.1, (i // 4) * 1.45 - 1.0) for i in range(count)]
    for dx, dy in points:
        ax.add_patch(
            Circle(
                (cx + dx, cy + dy),
                0.62,
                facecolor=TEAL_LIGHT,
                edgecolor=edge,
                linewidth=0.55,
            )
        )


def _bias_glyph(
    ax: plt.Axes,
    x: float,
    y: float,
    bias_std: float,
    edge: str,
) -> None:
    skew = min(max(bias_std, 0.0), 0.5)
    left_width = 4.2 - skew * 3.0
    right_width = 4.2 + skew * 3.0
    ax.add_patch(Rectangle((x, y), left_width, 1.1, facecolor=TEAL_LIGHT, edgecolor="none"))
    ax.add_patch(
        Rectangle(
            (x + left_width + 0.25, y),
            right_width,
            1.1,
            facecolor=GOLD_LIGHT,
            edgecolor="none",
        )
    )
    ax.plot([x, x + 8.8], [y - 0.55, y - 0.55], color=edge, linewidth=0.55)


def _cover_caveat(ax: plt.Axes) -> None:
    ax.text(
        140,
        8.2,
        "schematic cover - not empirical evidence; curves are illustrative",
        ha="center",
        color=MUTED,
        fontsize=8.6,
    )


def _cover_cases(profile: ExperimentProfile) -> tuple[CoverCase, ...]:
    return tuple(
        CoverCase(panel_size=panel_size, bias_std=bias_std)
        for panel_size in profile.grid.panel_sizes
        for bias_std in profile.grid.bias_stds
    )


def _corner_cases(cases: tuple[CoverCase, ...]) -> tuple[CoverCase, ...]:
    panel_sizes = sorted({case.panel_size for case in cases})
    bias_stds = sorted({case.bias_std for case in cases})
    if not panel_sizes or not bias_stds:
        return cases[:4]
    corners = (
        (panel_sizes[0], bias_stds[0]),
        (panel_sizes[0], bias_stds[-1]),
        (panel_sizes[-1], bias_stds[0]),
        (panel_sizes[-1], bias_stds[-1]),
    )
    by_key = {(case.panel_size, case.bias_std): case for case in cases}
    return tuple(by_key[key] for key in corners if key in by_key)


def _case_manifest(cases: tuple[CoverCase, ...]) -> list[dict[str, float | int | str]]:
    return [
        {
            "panel_size": case.panel_size,
            "bias_std": case.bias_std,
            "label": case.label,
        }
        for case in cases
    ]


def _manifest_payload(
    root: Path,
    cover_path: Path,
    manifest_path: Path,
    config_path: Path,
    config_hash: str,
    profile: ExperimentProfile,
    cases: tuple[CoverCase, ...],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "cover_image": _relative(root, cover_path),
        "manifest": _relative(root, manifest_path),
        "source_config": _relative(root, config_path),
        "source_config_hash": config_hash,
        "profile_name": profile.name,
        "displayed_case_axes": {
            "panel_sizes": list(profile.grid.panel_sizes),
            "bias_stds": list(profile.grid.bias_stds),
        },
        "displayed_cases": _case_manifest(cases),
        "concept_references": {
            "large_page": LARGE_CONCEPT,
            "mobile_portrait": MOBILE_CONCEPT,
        },
        "semantic_role": "schematic front-matter cover for the sortition-to-NTQR evaluation pipeline",
        "caveat": "Deterministic schematic only; not an empirical result and not counted as a manuscript data figure.",
        "visual_contract": {
            "reading_order": (
                "audited lottery -> representative vs single-bloc panel composition -> "
                "shared-confound error coupling -> NTQR agreement matrix -> phase-transition "
                "fan-out (recovery error vs within-bloc coupling), with a subordinate "
                "panel-size x bias case grid"
            ),
            "editable_layers": "title, author, affiliation, ORCID, DOI, and source notes remain renderer-owned text",
            "forbidden_claims": "no exact performance values, DOI identifiers, or source claims are baked into the raster",
        },
    }


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


__all__ = ["CoverPaths", "render_cover"]
