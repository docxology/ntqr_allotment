"""Panel-formation strategies (the layer *upstream* of NTQR).

The central comparison of the project: given the same expert population and the
same panel size, how do different ways of *forming the panel* change the
oracle-referenced error of no-answer-key NTQR evaluations?

Strategies implemented:

* ``representative_sortition`` -- auditable maximin lottery via the upstream
  ``allotment`` engine, with quotas that make the panel mirror the population's
  ideology composition (Flanigan et al. 2021 maximin).
* ``random_selection``        -- uniform draw without replacement (baseline).
* ``ideological_selection``   -- pick from a single ideology bucket: the
  deliberately *correlated / non-representative* comparator.
* ``expertise_threshold``     -- pick the top-k most expert (competence-first).

Every strategy is deterministic given its seed. The ``allotment``-backed draw
additionally carries the engine's SHA-256 audit hash for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from allotment.domain import Candidate, FeatureSpec, Pool, QuotaConfig, QuotaTarget
from allotment.quotas import precheck_feasibility
from allotment.selection_core.audit import build_audit_record, run_draw

from .experts import Expert

_DEFAULT_PANEL_COUNT = 20


@dataclass(frozen=True)
class PanelDraw:
    """Result of forming a panel: the selected expert ids plus provenance."""

    strategy: str
    expert_ids: tuple[str, ...]
    seed: int
    composition: dict[str, dict[str, int]] = field(default_factory=dict)
    audit_hash: str | None = None
    feasibility_warnings: tuple[str, ...] = ()


def _composition(experts_by_id: dict[str, Expert], ids: Sequence[str]) -> dict[str, dict[str, int]]:
    comp: dict[str, dict[str, int]] = {"ideology": {}, "expertise_tier": {}}
    for eid in ids:
        e = experts_by_id[eid]
        comp["ideology"][e.ideology] = comp["ideology"].get(e.ideology, 0) + 1
        comp["expertise_tier"][e.expertise_tier] = comp["expertise_tier"].get(e.expertise_tier, 0) + 1
    return comp


def _largest_remainder(counts: dict[str, int], total: int) -> dict[str, int]:
    """Allocate ``total`` seats across keys proportional to ``counts``.

    Largest-remainder (Hamilton) apportionment -> integer seats summing to
    ``total`` that mirror the population proportions as closely as possible.
    """
    pop = sum(counts.values())
    if pop == 0:
        raise ValueError("empty population")
    quotas = {k: total * v / pop for k, v in counts.items()}
    floors = {k: int(np.floor(q)) for k, q in quotas.items()}
    remaining = total - sum(floors.values())
    order = sorted(counts, key=lambda k: (quotas[k] - floors[k], counts[k]), reverse=True)
    for k in order[:remaining]:
        floors[k] += 1
    return floors


def build_pool(experts: Sequence[Expert]) -> Pool:
    """Turn an expert population into an ``allotment`` candidate pool."""
    ideologies = sorted({e.ideology for e in experts})
    tiers = sorted({e.expertise_tier for e in experts})
    candidates = [
        Candidate(
            id=e.id,
            features={"ideology": e.ideology, "expertise_tier": e.expertise_tier},
            contact_ref=f"{e.id}@synthetic.local",
        )
        for e in experts
    ]
    return Pool(
        features=[
            FeatureSpec(name="ideology", values=ideologies),
            FeatureSpec(name="expertise_tier", values=tiers),
        ],
        candidates=candidates,
    )


def representative_sortition(
    experts: Sequence[Expert],
    panel_size: int,
    *,
    seed: int,
    panel_count: int = _DEFAULT_PANEL_COUNT,
) -> PanelDraw:
    """Auditable maximin lottery with ideology quotas mirroring the population."""
    experts_by_id = {e.id: e for e in experts}
    pool = build_pool(experts)
    ideo_counts: dict[str, int] = {}
    for e in experts:
        ideo_counts[e.ideology] = ideo_counts.get(e.ideology, 0) + 1
    seats = _largest_remainder(ideo_counts, panel_size)
    targets = [
        QuotaTarget(feature="ideology", value=val, min=seats[val], max=seats[val])
        for val in sorted(seats)
    ]
    config = QuotaConfig(panel_size=panel_size, targets=targets)
    warnings = tuple(precheck_feasibility(pool, config))
    result = run_draw(pool, config, panel_count=panel_count, seed=seed)
    audit = build_audit_record(pool, config, result)
    ids = tuple(result.selection.candidate_ids)
    return PanelDraw(
        strategy="representative_sortition",
        expert_ids=ids,
        seed=seed,
        composition=_composition(experts_by_id, ids),
        audit_hash=audit.input_hash,
        feasibility_warnings=warnings,
    )


def random_selection(experts: Sequence[Expert], panel_size: int, *, seed: int) -> PanelDraw:
    """Uniform draw without replacement (the simplest honest baseline)."""
    experts_by_id = {e.id: e for e in experts}
    rng = np.random.default_rng(seed)
    ids = tuple(str(x) for x in rng.choice([e.id for e in experts], size=panel_size, replace=False))
    return PanelDraw(
        strategy="random_selection",
        expert_ids=ids,
        seed=seed,
        composition=_composition(experts_by_id, ids),
    )


def ideological_selection(
    experts: Sequence[Expert],
    panel_size: int,
    *,
    seed: int,
    ideology: str | None = None,
) -> PanelDraw:
    """Pick from a single ideology bucket -> a deliberately correlated panel."""
    experts_by_id = {e.id: e for e in experts}
    rng = np.random.default_rng(seed)
    if ideology is None:
        ideologies = sorted({e.ideology for e in experts})
        ideology = ideologies[seed % len(ideologies)]
    pref = [e.id for e in experts if e.ideology == ideology]
    rest = [e.id for e in experts if e.ideology != ideology]
    rng.shuffle(pref)
    rng.shuffle(rest)
    ordered = pref + rest  # fill from the chosen bloc first, then spill over
    ids = tuple(ordered[:panel_size])
    return PanelDraw(
        strategy="ideological_selection",
        expert_ids=ids,
        seed=seed,
        composition=_composition(experts_by_id, ids),
    )


def expertise_threshold(experts: Sequence[Expert], panel_size: int, *, seed: int) -> PanelDraw:
    """Competence-first: select the top-k experts by expertise."""
    experts_by_id = {e.id: e for e in experts}
    ranked = sorted(experts, key=lambda e: (e.expertise, e.id), reverse=True)
    ids = tuple(e.id for e in ranked[:panel_size])
    return PanelDraw(
        strategy="expertise_threshold",
        expert_ids=ids,
        seed=seed,
        composition=_composition(experts_by_id, ids),
    )


STRATEGIES = {
    "representative_sortition": representative_sortition,
    "random_selection": random_selection,
    "ideological_selection": ideological_selection,
    "expertise_threshold": expertise_threshold,
}


__all__ = [
    "PanelDraw",
    "build_pool",
    "representative_sortition",
    "random_selection",
    "ideological_selection",
    "expertise_threshold",
    "STRATEGIES",
]
