from __future__ import annotations

from typing import NamedTuple

from ntqr_allotment.power_parts.analytic import (
    min_detectable_effect,
    sample_size_for_power,
)


class NullDiagnosis(NamedTuple):
    observed_effect: float
    n_per_group: int
    mde: float
    underpowered: bool
    seeds_for_target: int | None
    target_power: float
    alpha: float


def diagnose_null(
    observed_effect: float,
    n_per_group: int,
    *,
    target_power: float = 0.8,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> NullDiagnosis:
    mde = min_detectable_effect(
        n_per_group, power=target_power, alpha=alpha, two_sided=two_sided
    )
    if observed_effect == 0.0:
        seeds: int | None = None
    else:
        try:
            seeds = sample_size_for_power(
                observed_effect, power=target_power, alpha=alpha, two_sided=two_sided
            )
        except ValueError:
            # Effect so small that even n_max per group cannot reach target power:
            # report no practical budget (None), exactly as for a zero effect.
            # Surfaced by the 96-seed run, where a near-zero contrast needs an
            # observation budget beyond the n_max ceiling.
            seeds = None
    return NullDiagnosis(
        observed_effect=float(observed_effect),
        n_per_group=int(n_per_group),
        mde=float(mde),
        underpowered=bool(abs(observed_effect) < mde),
        seeds_for_target=seeds,
        target_power=float(target_power),
        alpha=float(alpha),
    )
