from __future__ import annotations

import math
from typing import Sequence

from ntqr_allotment.power_parts.normal import _check_alpha, norm_cdf, norm_ppf


def analytic_power(
    effect: float,
    n_a: int,
    n_b: int | None = None,
    *,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    _check_alpha(alpha)
    nb = n_a if n_b is None else n_b
    if n_a < 1 or nb < 1:
        raise ValueError("group sizes must be >= 1")

    ncp = abs(effect) / math.sqrt(1.0 / n_a + 1.0 / nb)
    if two_sided:
        z_crit = norm_ppf(1.0 - alpha / 2.0)
        power = float(norm_cdf(ncp - z_crit) + norm_cdf(-ncp - z_crit))
    else:
        z_crit = norm_ppf(1.0 - alpha)
        power = float(norm_cdf(ncp - z_crit))
    return min(1.0, max(0.0, power))


def sample_size_for_power(
    effect: float,
    *,
    power: float = 0.8,
    alpha: float = 0.05,
    two_sided: bool = True,
    n_max: int = 1_000_000,
) -> int:
    _check_alpha(alpha)
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in the open interval (0, 1)")
    if effect == 0.0:
        raise ValueError("effect must be non-zero")

    z_a = norm_ppf(1.0 - alpha / 2.0) if two_sided else norm_ppf(1.0 - alpha)
    z_p = norm_ppf(power)
    seed_n = max(2, int(math.ceil(2.0 * (z_a + z_p) ** 2 / effect**2)))
    n = max(1, seed_n - 3)
    while n <= n_max:
        if analytic_power(effect, n, alpha=alpha, two_sided=two_sided) >= power:
            return n
        n += 1
    raise ValueError(f"target power {power} not reached by n_max={n_max}")


def min_detectable_effect(
    n: int,
    *,
    power: float = 0.8,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    _check_alpha(alpha)
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in the open interval (0, 1)")
    if n < 1:
        raise ValueError("n must be >= 1")

    z_a = norm_ppf(1.0 - alpha / 2.0) if two_sided else norm_ppf(1.0 - alpha)
    z_p = norm_ppf(power)
    return float((z_a + z_p) * math.sqrt(2.0 / n))


def power_curve_over_n(
    effect: float,
    ns: Sequence[int],
    *,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> list[tuple[int, float]]:
    return [
        (int(n), analytic_power(effect, int(n), alpha=alpha, two_sided=two_sided))
        for n in ns
    ]


def power_curve_over_effect(
    effects: Sequence[float],
    n: int,
    *,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> list[tuple[float, float]]:
    return [
        (float(e), analytic_power(float(e), n, alpha=alpha, two_sided=two_sided))
        for e in effects
    ]
