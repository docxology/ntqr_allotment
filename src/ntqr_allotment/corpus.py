"""A labeled binary-item corpus for the empirical (LLM-persona) track.

Domain: arithmetic truth-judgement. Each item is an equation ``x + y = z`` that
is either correct (label ``"a"`` = TRUE) or off by a small delta (label ``"b"``
= FALSE). This domain is deliberately chosen so that an LLM judge's accuracy
*varies* with the difficulty of the item and with the competence the persona is
instructed to adopt — giving the noisy-but-better-than-chance judges NTQR needs.

Ground truth is known, so persona judgments feed the SAME oracle-scored pipeline
as the synthetic track. Deterministic given the seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CorpusItem:
    """One labeled item. ``true_label`` is 'a' (TRUE) or 'b' (FALSE)."""

    index: int
    text: str
    true_label: str

    @property
    def statement(self) -> str:
        """Human-readable statement presented to a judge."""
        return self.text


def make_arithmetic_corpus(
    n_items: int,
    *,
    seed: int,
    prevalence_true: float = 0.5,
    max_operand: int = 99,
    max_error: int = 3,
) -> list[CorpusItem]:
    """Build ``n_items`` arithmetic equations, ``prevalence_true`` of them correct.

    A FALSE item is the correct sum perturbed by a non-zero delta in
    ``[-max_error, max_error]`` so the falsehood is plausible (off-by-a-little),
    not absurd. Label 'a' == TRUE, 'b' == FALSE (matches ``ntqr.r2`` label space).
    """
    if n_items <= 0:
        raise ValueError("n_items must be positive")
    if not 0.0 <= prevalence_true <= 1.0:
        raise ValueError("prevalence_true must be in [0, 1]")
    rng = np.random.default_rng(seed)
    items: list[CorpusItem] = []
    for i in range(n_items):
        x = int(rng.integers(0, max_operand + 1))
        y = int(rng.integers(0, max_operand + 1))
        true_sum = x + y
        is_true = rng.random() < prevalence_true
        if is_true:
            shown = true_sum
            label = "a"
        else:
            delta = 0
            while delta == 0:
                delta = int(rng.integers(-max_error, max_error + 1))
            shown = true_sum + delta
            label = "b"
        items.append(CorpusItem(index=i, text=f"{x} + {y} = {shown}", true_label=label))
    return items


__all__ = ["CorpusItem", "make_arithmetic_corpus"]
