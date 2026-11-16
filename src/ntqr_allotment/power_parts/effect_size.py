from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def cohens_d_safe(group_a: Sequence[float], group_b: Sequence[float]) -> float:
    """Return a finite standardized difference, or reject an undefined effect.

    Equal observed means have no directional effect and return zero. Different
    means with no estimable pooled spread do not: flattening that case to zero
    would reverse the direction of the evidence.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError("both groups must be non-empty")
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
        raise ValueError("both groups must contain only finite values")

    # Cohen's d is invariant to a common positive rescaling. Normalize first so
    # finite inputs cannot overflow squared deviations or underflow real spread.
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    if scale == 0.0:
        return 0.0
    a = a / scale
    b = b / scale
    mean_a = float(a.mean())
    mean_b = float(b.mean())
    if mean_a == mean_b:
        return 0.0

    na, nb = a.size, b.size
    denom = na + nb - 2
    if denom <= 0:
        raise ValueError("Cohen's d is undefined without pooled variance")
    var_a = float(a.var(ddof=1)) if na > 1 else 0.0
    var_b = float(b.var(ddof=1)) if nb > 1 else 0.0
    pooled_var = ((na - 1) * var_a + (nb - 1) * var_b) / denom
    if pooled_var <= 0.0:
        raise ValueError("Cohen's d is undefined when different means have zero spread")
    return (mean_a - mean_b) / float(math.sqrt(pooled_var))
