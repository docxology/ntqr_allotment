"""Make hash-order-dependent code reproducible across processes.

``representative_sortition`` draws its panel through the external ``allotment``
maximin lottery, whose tie-breaking among equiprobable candidates iterates a set
and is therefore sensitive to Python's per-process hash randomization
(``PYTHONHASHSEED``). The ``seed`` argument controls the lottery RNG but not that
iteration order, so two processes can select *different* panels for the same seed.

The project's reproducibility contract requires byte-identical artifacts, so every
entry point that forms a representative panel calls
:func:`ensure_deterministic_hashing` first. It pins ``PYTHONHASHSEED=0`` by
re-executing the interpreter once if needed; ``multiprocessing`` workers spawned
afterwards inherit the pinned environment, so a whole parallel sweep is
deterministic.
"""
from __future__ import annotations

import os
import sys


def ensure_deterministic_hashing() -> None:
    """Pin ``PYTHONHASHSEED=0``, re-executing this process once if it is not set.

    No-op when already pinned (the common case after the single re-exec), so it is
    safe to call unconditionally at the top of a ``main()``.
    """
    if os.environ.get("PYTHONHASHSEED") == "0":
        return
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, *sys.argv])


__all__ = ["ensure_deterministic_hashing"]
