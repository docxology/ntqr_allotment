from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from ntqr_allotment.power_parts.normal import _check_alpha


def _mean_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(a.mean() - b.mean())


def permutation_test(
    group_a: Sequence[float],
    group_b: Sequence[float],
    *,
    n_perm: int = 10000,
    seed: int,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> float:
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError("both groups must be non-empty")
    if n_perm < 1:
        raise ValueError("n_perm must be >= 1")

    observed = _mean_diff(a, b)
    pooled = np.concatenate([a, b])
    na = a.size
    rng = np.random.default_rng(seed)

    extreme = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        stat = _mean_diff(perm[:na], perm[na:])
        if alternative == "two-sided":
            if abs(stat) >= abs(observed) - 1e-12:
                extreme += 1
        elif alternative == "greater":
            if stat >= observed - 1e-12:
                extreme += 1
        else:
            if stat <= observed + 1e-12:
                extreme += 1
    return (1.0 + extreme) / (1.0 + n_perm)


def simulate_power(
    effect: float,
    n_a: int,
    n_b: int | None = None,
    *,
    alpha: float = 0.05,
    n_sim: int = 400,
    n_perm: int = 200,
    seed: int,
    sd: float = 1.0,
    two_sided: bool = True,
) -> float:
    _check_alpha(alpha)
    nb = n_a if n_b is None else n_b
    if n_a < 1 or nb < 1:
        raise ValueError("group sizes must be >= 1")
    if n_sim < 1:
        raise ValueError("n_sim must be >= 1")
    if sd <= 0.0:
        raise ValueError("sd must be positive")

    rng = np.random.default_rng(seed)
    child_seeds = rng.integers(0, 2**32 - 1, size=n_sim)
    alt: Literal["two-sided", "greater", "less"]
    if two_sided:
        alt = "two-sided"
    elif effect >= 0.0:
        alt = "less"
    else:
        alt = "greater"

    rejections = 0
    for i in range(n_sim):
        sub = np.random.default_rng(int(child_seeds[i]))
        a = sub.normal(0.0, sd, size=n_a)
        b = sub.normal(effect * sd, sd, size=nb)
        p = permutation_test(a, b, n_perm=n_perm, seed=int(child_seeds[i]), alternative=alt)
        if p < alpha:
            rejections += 1
    return rejections / n_sim
