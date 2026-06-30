"""Pure-numeric inferential statistics helpers for honest uncertainty.

This module puts measurable uncertainty on the study's sweeps without depending
on any upstream NTQR or sortition code. Everything here is plain NumPy:

1. ``bootstrap_ci`` — percentile bootstrap confidence interval of the mean,
   seeded via :func:`numpy.random.default_rng` so the same seed yields the same
   interval.
2. ``cohens_d`` — pooled-standard-deviation standardized mean difference between
   two groups, an effect-size companion to a significance test.
3. ``ols_slope`` — ordinary least squares line fit returning slope, intercept,
   and the coefficient of determination.
4. ``bootstrap_slope_ci`` — percentile bootstrap confidence interval for the OLS
   slope, resampling (x, y) pairs.

The OLS slope is implemented independently of
:func:`ntqr_allotment.theory.fit_error_correlation_slope`: it solves for the
mean-centred covariance/variance ratio with NumPy arrays and also reports the
intercept and R-squared, which that function does not.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Sequence

import numpy as np


class OLSResult(NamedTuple):
    """Result of an ordinary least squares straight-line fit."""

    slope: float
    intercept: float
    r_squared: float


class SeparationResult(NamedTuple):
    """Result of comparing two strategy means and their bootstrap intervals."""

    mean_a: float
    mean_b: float
    ci_a: tuple[float, float]
    ci_b: tuple[float, float]
    verdict: str
    signed_difference: float


class HolmResult(NamedTuple):
    """Holm step-down multiple-comparison correction over a family of p-values."""

    adjusted: tuple[float, ...]
    rejected: tuple[bool, ...]
    n_rejected: int


class SignTestResult(NamedTuple):
    """Exact two-sided sign-test result for directional stability."""

    negative: int
    positive: int
    zero: int
    n_nonzero: int
    p_value: float


class MeanCIResult(NamedTuple):
    """Descriptive mean/spread/CI summary for a numeric sample."""

    n: int
    mean: float
    std: float
    ci_low: float
    ci_high: float


def mean_ci_summary(
    values: Sequence[float],
    *,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int,
) -> MeanCIResult:
    """Return descriptive mean, sample SD, and percentile-bootstrap mean CI.

    This is intentionally descriptive: it summarizes the observed sample and
    does not turn a small n into a formal inference claim. A one-point sample
    gets a point CI, because there is no resampling uncertainty to estimate.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("values must be non-empty")
    if np.any(~np.isfinite(arr)):
        raise ValueError("values must be finite")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in the open interval (0, 1)")

    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    if arr.size == 1:
        ci_low = ci_high = mean
    else:
        ci_low, ci_high = bootstrap_ci(arr, n_boot=n_boot, alpha=alpha, seed=seed)
    return MeanCIResult(
        n=int(arr.size),
        mean=mean,
        std=std,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def exact_sign_test(values: Sequence[float]) -> SignTestResult:
    """Return an exact two-sided binomial sign test against equally likely signs.

    Zero values are reported and excluded from the binomial denominator because
    they carry no direction. The two-sided p-value is the probability, under
    ``p=0.5``, of seeing a sign count at least as imbalanced as the observed one.
    For the current four-run cross-family result, four negative and zero positive
    nonzero deltas give ``p=0.125``.
    """
    vals = [float(value) for value in values]
    negative = sum(1 for value in vals if value < 0.0)
    positive = sum(1 for value in vals if value > 0.0)
    zero = len(vals) - negative - positive
    n_nonzero = negative + positive
    if n_nonzero == 0:
        return SignTestResult(
            negative=negative,
            positive=positive,
            zero=zero,
            n_nonzero=0,
            p_value=1.0,
        )

    smaller_tail = min(negative, positive)
    tail_probability = sum(
        math.comb(n_nonzero, k) for k in range(smaller_tail + 1)
    ) / (2**n_nonzero)
    return SignTestResult(
        negative=negative,
        positive=positive,
        zero=zero,
        n_nonzero=n_nonzero,
        p_value=min(1.0, 2.0 * tail_probability),
    )


def holm_bonferroni(
    pvalues: Sequence[float], *, alpha: float = 0.05
) -> HolmResult:
    """Holm-Bonferroni step-down correction for a family of hypothesis tests.

    Controls the family-wise error rate without the conservatism of plain
    Bonferroni: p-values are sorted ascending, the k-th smallest is multiplied by
    ``(m - k)``, a running max enforces monotonicity, and each adjusted value is
    capped at 1.0. Results are returned in the ORIGINAL input order. This is the
    correction the pairwise strategy contrasts need — without it, reporting the
    smallest of many p-values overstates significance.

    Args:
        pvalues: The family of raw p-values (non-empty), each in ``[0, 1]``.
        alpha: Family-wise significance level.

    Returns:
        A :class:`HolmResult` with input-order ``adjusted`` p-values, the
        ``rejected`` flags (``adjusted < alpha``), and the ``n_rejected`` count.

    Raises:
        ValueError: If ``pvalues`` is empty, any p is outside ``[0, 1]``, or
            ``alpha`` is not in the open interval ``(0, 1)``.
    """
    p = [float(v) for v in pvalues]
    m = len(p)
    if m == 0:
        raise ValueError("pvalues must be non-empty")
    if any(not 0.0 <= v <= 1.0 for v in p):
        raise ValueError("every p-value must lie in [0, 1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in the open interval (0, 1)")

    order = sorted(range(m), key=lambda i: p[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    rejected = tuple(adjusted[i] < alpha for i in range(m))
    return HolmResult(
        adjusted=tuple(adjusted),
        rejected=rejected,
        n_rejected=int(sum(rejected)),
    )


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int,
) -> tuple[float, float]:
    """Return a percentile bootstrap confidence interval for the mean.

    Resamples ``values`` with replacement ``n_boot`` times, takes the mean of
    each resample, and returns the ``alpha/2`` and ``1 - alpha/2`` percentiles of
    those bootstrap means. The draw is seeded with
    :func:`numpy.random.default_rng`, so a fixed ``seed`` gives an identical
    interval. A constant input yields a zero-width interval because every
    resample mean equals that constant.

    Args:
        values: Observed sample (must be non-empty).
        n_boot: Number of bootstrap resamples.
        alpha: Two-sided miscoverage; the interval spans ``1 - alpha``.
        seed: Seed for the random generator.

    Returns:
        The ``(low, high)`` percentile interval of the bootstrap means.

    Raises:
        ValueError: If ``values`` is empty, ``n_boot < 1``, or ``alpha`` is not
            strictly inside ``(0, 1)``.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("values must be non-empty")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in the open interval (0, 1)")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[indices].mean(axis=1)
    low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return (low, high)


def _validate_ci_interval(
    interval: tuple[float, float], *, name: str
) -> tuple[float, float]:
    if not isinstance(interval, tuple) or len(interval) != 2:
        raise ValueError(f"{name} must be a 2-tuple (lo, hi)")
    lo = float(interval[0])
    hi = float(interval[1])
    if lo > hi:
        raise ValueError(f"{name} must satisfy lo <= hi")
    return (lo, hi)


def ci_overlap_verdict(
    ci_a: tuple[float, float], ci_b: tuple[float, float]
) -> str:
    """Classify whether two confidence intervals are separated.

    The contract is conservative: intervals that touch at exactly one endpoint
    still count as overlapping, so only strict non-overlap is "separated".

    Args:
        ci_a: First ``(lo, hi)`` confidence interval.
        ci_b: Second ``(lo, hi)`` confidence interval.

    Returns:
        ``"separated"`` if the intervals do not overlap, else ``"overlapping"``.

    Raises:
        ValueError: If either interval is not a 2-tuple or violates ``lo <= hi``.
    """
    lo_a, hi_a = _validate_ci_interval(ci_a, name="ci_a")
    lo_b, hi_b = _validate_ci_interval(ci_b, name="ci_b")
    if hi_a < lo_b or hi_b < lo_a:
        return "separated"
    return "overlapping"


def strategy_separation(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int,
) -> SeparationResult:
    """Compare two strategy means using separately bootstrapped mean intervals.

    Reuses :func:`bootstrap_ci` for each group's mean interval. The two
    bootstrap draws are seeded independently as ``seed`` and ``seed + 1`` so the
    intervals are reproducible without sharing identical resample streams.

    Args:
        values_a: Per-seed values for strategy A (must be non-empty).
        values_b: Per-seed values for strategy B (must be non-empty).
        n_boot: Number of bootstrap resamples per group.
        alpha: Two-sided miscoverage for both intervals.
        seed: Base seed for the random generators.

    Returns:
        A :class:`SeparationResult` containing means, CIs, verdict, and signed
        difference ``mean_a - mean_b``.

    Raises:
        ValueError: If either group's bootstrap CI is invalid, including empty
            inputs, ``n_boot < 1``, or ``alpha`` outside ``(0, 1)``.
    """
    ci_a = bootstrap_ci(values_a, n_boot=n_boot, alpha=alpha, seed=seed)
    ci_b = bootstrap_ci(values_b, n_boot=n_boot, alpha=alpha, seed=seed + 1)
    mean_a = float(np.asarray(values_a, dtype=float).mean())
    mean_b = float(np.asarray(values_b, dtype=float).mean())
    return SeparationResult(
        mean_a=mean_a,
        mean_b=mean_b,
        ci_a=ci_a,
        ci_b=ci_b,
        verdict=ci_overlap_verdict(ci_a, ci_b),
        signed_difference=mean_a - mean_b,
    )


def cohens_d(group_a: Sequence[float], group_b: Sequence[float]) -> float:
    """Return Cohen's d, the pooled-SD standardized mean difference.

    Computes ``(mean(a) - mean(b)) / s_pooled`` where ``s_pooled`` is the pooled
    standard deviation using the sample (ddof=1) variances. Identical groups give
    exactly ``0.0``. If the pooled variance is zero (both groups constant) the
    function returns ``0.0`` rather than raising, since a zero numerator over a
    zero spread is reported as no standardized effect.

    Args:
        group_a: First group (needs at least two points).
        group_b: Second group (needs at least two points).

    Returns:
        The standardized mean difference.

    Raises:
        ValueError: If either group has fewer than two points.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if a.size < 2 or b.size < 2:
        raise ValueError("each group must have at least two points")

    var_a = float(a.var(ddof=1))
    var_b = float(b.var(ddof=1))
    na = a.size
    nb = b.size
    pooled_var = ((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2)
    if pooled_var == 0.0:
        return 0.0
    return (float(a.mean()) - float(b.mean())) / float(np.sqrt(pooled_var))


def ols_slope(xs: Sequence[float], ys: Sequence[float]) -> OLSResult:
    """Fit ``y = slope * x + intercept`` by ordinary least squares.

    Solves for the slope as the mean-centred covariance divided by the variance
    of ``xs``, derives the intercept from the means, and reports R-squared as
    ``1 - SS_res / SS_tot``. Points exactly on a line return that line with
    ``r_squared == 1.0``.

    Args:
        xs: Independent variable (at least two points, positive variance).
        ys: Dependent variable (same length as ``xs``).

    Returns:
        An :class:`OLSResult` with ``slope``, ``intercept``, ``r_squared``.

    Raises:
        ValueError: If the inputs differ in length, have fewer than two points,
            or ``xs`` has zero variance.
    """
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size != y.size:
        raise ValueError("xs and ys must have the same length")
    if x.size < 2:
        raise ValueError("at least two points are required")

    mean_x = float(x.mean())
    mean_y = float(y.mean())
    dx = x - mean_x
    dy = y - mean_y
    ss_x = float(np.sum(dx * dx))
    if ss_x == 0.0:
        raise ValueError("xs must have positive variance")

    slope = float(np.sum(dx * dy) / ss_x)
    intercept = mean_y - slope * mean_x
    ss_tot = float(np.sum(dy * dy))
    residuals = y - (slope * x + intercept)
    ss_res = float(np.sum(residuals * residuals))
    if ss_tot == 0.0:
        # ys is constant: a flat fit explains it exactly.
        r_squared = 1.0
    else:
        r_squared = 1.0 - ss_res / ss_tot
    return OLSResult(slope=slope, intercept=intercept, r_squared=r_squared)


def bootstrap_slope_ci(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int,
) -> tuple[float, float]:
    """Return a percentile bootstrap confidence interval for the OLS slope.

    Resamples ``(x, y)`` pairs with replacement, refits :func:`ols_slope` on each
    resample (skipping degenerate resamples whose ``xs`` collapse to zero
    variance), and returns the ``alpha/2`` and ``1 - alpha/2`` percentiles of the
    bootstrap slopes. Seeded for reproducibility.

    Args:
        xs: Independent variable (at least two points, positive variance).
        ys: Dependent variable (same length as ``xs``).
        n_boot: Number of bootstrap resamples.
        alpha: Two-sided miscoverage.
        seed: Seed for the random generator.

    Returns:
        The ``(low, high)`` percentile interval of the bootstrap slopes.

    Raises:
        ValueError: If the base fit is invalid, ``n_boot < 1``, ``alpha`` is not
            in ``(0, 1)``, or every resample is degenerate.
    """
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    # Validate the base configuration (length, count, variance) via ols_slope.
    ols_slope(x, y)
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in the open interval (0, 1)")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, x.size, size=(n_boot, x.size))
    slopes: list[float] = []
    for row in indices:
        xb = x[row]
        if float(np.sum((xb - xb.mean()) ** 2)) == 0.0:
            continue
        slopes.append(ols_slope(xb, y[row]).slope)
    if not slopes:
        raise ValueError("all bootstrap resamples were degenerate")

    boot = np.asarray(slopes, dtype=float)
    low = float(np.percentile(boot, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
    return (low, high)


__all__ = [
    "OLSResult",
    "SeparationResult",
    "HolmResult",
    "SignTestResult",
    "MeanCIResult",
    "bootstrap_ci",
    "ci_overlap_verdict",
    "cohens_d",
    "exact_sign_test",
    "holm_bonferroni",
    "mean_ci_summary",
    "ols_slope",
    "bootstrap_slope_ci",
    "strategy_separation",
]
