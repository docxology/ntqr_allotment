from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def cohens_d_safe(group_a: Sequence[float], group_b: Sequence[float]) -> float:
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError("both groups must be non-empty")

    na, nb = a.size, b.size
    var_a = float(a.var(ddof=1)) if na > 1 else 0.0
    var_b = float(b.var(ddof=1)) if nb > 1 else 0.0
    denom = na + nb - 2
    if denom <= 0:
        return 0.0
    pooled_var = ((na - 1) * var_a + (nb - 1) * var_b) / denom
    if pooled_var <= 0.0:
        return 0.0
    return (float(a.mean()) - float(b.mean())) / float(math.sqrt(pooled_var))
