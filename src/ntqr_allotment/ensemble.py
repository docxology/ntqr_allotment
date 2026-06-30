"""N-judge (panel size N > 3) binary ensemble analysis.

This module scales the project's two ground-truth-free signals beyond the exact
trio solver, in order to study *how many experts a panel needs*:

1. **Panel-agreement frequency moments.** NTQR's ``ObservedVoteCounts`` defines
   per-label classifier frequencies, pairwise co-voting frequencies, and their
   centred *frequency moments* (co-voting minus the product of marginals -- a
   covariance-like agreement-structure summary) plus a trio third moment. We
   build the observed pattern table over all N judges and report these moments
   as real numbers. They are agreement-structure descriptors, NOT accuracy
   estimates: a higher pair moment means judges co-vote more than independence
   predicts (the very correlation the exact solver assumes away).

   .. note::
      ``ObservedVoteCounts.pair_frequency_moment`` / ``trio_frequency_moment``
      are unusable in the installed NTQR build: their internal
      ``classifier_label_frequency`` calls the module-level
      ``classifier_label_votes`` / ``classifiers_labels_votes`` helpers WITHOUT
      the required ``vote_patterns`` argument and raise ``TypeError``. We
      therefore compute the moments from the *working* primitives
      (``to_frequencies_exact`` plus those two module helpers, supplied with
      the observed vote patterns) -- the identical algebra the broken wrappers
      document, routed around the library defect. We never claim to call the
      broken wrappers.

2. **Alarm power curve.** ``ntqr.alarms.SingleClassifierAxiomsAlarm`` (one
   ``SingleClassifierAxioms`` per judge) fires when no answer key lets ALL
   judges meet a safety specification. Sweeping panel size N and counting the
   fraction of seeds that fire gives a *power curve* in N: the ground-truth-free
   consistency signal as a function of how many experts you poll. The alarm is
   ~O(Q^3), so the corpus is capped at a small ``max_q`` (mirroring
   :func:`ntqr_allotment.ntqr_eval.alarm_misaligned`).

The label space is binary ``("a", "b")`` to match ``ntqr.r2``. All randomness
is seeded via ``numpy.random.default_rng`` -> deterministic.
"""

from __future__ import annotations

import math
from typing import Sequence

from ntqr import Labels
from ntqr.alarms import SingleClassifierAxiomsAlarm
from ntqr.r2.datasketches import (
    ObservedVoteCounts,
    classifier_label_votes,
    classifiers_labels_votes,
)
from ntqr.r2.evaluations import SingleClassifierEvaluations
from ntqr.r2.raxioms import SingleClassifierAxioms

from .experts import Expert, Item, sample_votes

#: Same cost characteristic as ``ntqr_eval.ALARM_DEFAULT_MAX_Q``: the alarm
#: enumerates the answer-key simplex with cost ~O(Q^3), so the power curve runs
#: only on a SMALL corpus by default.
ALARM_DEFAULT_MAX_Q = 25


def observed_vote_counts(votes_by_judge: Sequence[Sequence[str]]) -> ObservedVoteCounts:
    """Build an :class:`ObservedVoteCounts` over an N-judge panel.

    ``votes_by_judge`` is ``N`` rows of per-item binary votes, aligned by
    position; the result is the joint pattern->count table over the panel
    (counts sum to the number of items). Requires ``N >= 2`` and equal-length
    rows; raises ``ValueError`` otherwise.
    """
    n_judges = len(votes_by_judge)
    if n_judges < 2:
        raise ValueError("an ensemble needs at least 2 judges")
    n_items = len(votes_by_judge[0])
    if n_items == 0:
        raise ValueError("judges must vote on at least one item")
    if any(len(v) != n_items for v in votes_by_judge):
        raise ValueError("all judges must vote on the same number of items")
    counts: dict[tuple[str, ...], int] = {}
    for k in range(n_items):
        pattern = tuple(votes_by_judge[j][k] for j in range(n_judges))
        counts[pattern] = counts.get(pattern, 0) + 1
    return ObservedVoteCounts(counts)


def _classifier_label_frequency(ovc: ObservedVoteCounts, classifier: int, label: str) -> float:
    """Fraction of items ``classifier`` voted ``label`` (NTQR primitive, fixed call)."""
    freqs = ovc.to_frequencies_exact()
    patterns = list(ovc.vote_counts.keys())
    matched = classifier_label_votes(classifier, label, patterns)
    return float(sum((freqs[v] for v in matched), start=0))


def _pair_label_frequency(ovc: ObservedVoteCounts, pair: tuple[int, int], label: str) -> float:
    """Fraction of items both judges in ``pair`` voted ``label`` (NTQR primitive, fixed call)."""
    freqs = ovc.to_frequencies_exact()
    patterns = list(ovc.vote_counts.keys())
    matched = classifiers_labels_votes(pair, (label, label), patterns)
    return float(sum((freqs[v] for v in matched), start=0))


def _pair_frequency_moment(ovc: ObservedVoteCounts, pair: tuple[int, int], label: str) -> float:
    """Centred pair frequency moment: co-vote frequency minus product of marginals.

    This is exactly the quantity ``ObservedVoteCounts.pair_frequency_moment``
    documents (``f_{li,lj} - f_{li} * f_{lj}``), computed via the working
    primitives because the wrapper raises ``TypeError`` in this NTQR build.
    """
    co = _pair_label_frequency(ovc, pair, label)
    marginals = [_classifier_label_frequency(ovc, c, label) for c in pair]
    return co - math.prod(marginals)


def panel_agreement_moments(votes_by_judge: Sequence[Sequence[str]]) -> dict[str, float]:
    """Report real-valued panel agreement-structure moments for an N-judge panel.

    Returns a dict of finite floats:

    * ``mean_abs_pair_moment_a`` / ``mean_abs_pair_moment_b`` -- mean absolute
      centred pair frequency moment over all judge pairs, per label. Larger =>
      judges co-vote more than independence predicts (positive error-correlation
      pressure on the trio solver's independence assumption).
    * ``max_abs_pair_moment`` -- the strongest pairwise agreement over both labels.
    * ``trio_frequency_moment_b`` -- the third (trio) frequency moment over the
      first three judges for label ``b`` (NTQR's ``trio_frequency_moment``
      definition; needs ``N >= 3``, ``nan`` if fewer judges).

    These are agreement descriptors, not accuracy estimates.
    """
    ovc = observed_vote_counts(votes_by_judge)
    n = ovc.ensemble_size
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    out: dict[str, float] = {}
    abs_all: list[float] = []
    for label in ("a", "b"):
        abs_label = [abs(_pair_frequency_moment(ovc, p, label)) for p in pairs]
        out[f"mean_abs_pair_moment_{label}"] = (
            float(sum(abs_label) / len(abs_label)) if abs_label else 0.0
        )
        abs_all.extend(abs_label)
    out["max_abs_pair_moment"] = float(max(abs_all)) if abs_all else 0.0

    if n >= 3:
        clf_b = [_classifier_label_frequency(ovc, c, "b") for c in range(3)]
        pm = {
            (1, 2): _pair_frequency_moment(ovc, (1, 2), "b"),
            (0, 2): _pair_frequency_moment(ovc, (0, 2), "b"),
            (0, 1): _pair_frequency_moment(ovc, (0, 1), "b"),
        }
        trio = (
            math.prod(clf_b)
            + clf_b[0] * pm[(1, 2)]
            + clf_b[1] * pm[(0, 2)]
            + clf_b[2] * pm[(0, 1)]
        )
        out["trio_frequency_moment_b"] = float(trio)
    else:
        out["trio_frequency_moment_b"] = math.nan
    return out


def _responses_for(votes_by_judge: Sequence[Sequence[str]]) -> list[list[int]]:
    """Per-judge ``[count_a, count_b]`` response vectors (each sums to Q)."""
    return [
        [sum(1 for v in votes if v == "a"), sum(1 for v in votes if v == "b")]
        for votes in votes_by_judge
    ]


def _alarm_fires(
    votes_by_judge: Sequence[Sequence[str]],
    *,
    safety: Sequence[int],
    q: int,
) -> bool:
    """Run the N-judge axioms alarm; True iff the panel is jointly misaligned."""
    labels = Labels(("a", "b"))
    axioms = [SingleClassifierAxioms(labels, i) for i in range(len(votes_by_judge))]
    alarm = SingleClassifierAxiomsAlarm(q, axioms, SingleClassifierEvaluations)
    alarm.set_safety_specification(list(safety))
    return bool(alarm.are_misaligned(_responses_for(votes_by_judge)))


def alarm_power_curve(
    experts: Sequence[Expert],
    items: Sequence[Item],
    sizes: Sequence[int],
    *,
    seeds: Sequence[int],
    safety: Sequence[int] = (2, 2),
    max_q: int = ALARM_DEFAULT_MAX_Q,
) -> list[tuple[int, float]]:
    """Alarm firing power as a function of panel size N.

    For each ``N`` in ``sizes`` the first ``N`` experts form a panel; for each
    seed in ``seeds`` the panel votes on ``items`` (one re-seeded
    :func:`sample_votes` per judge) and the N-judge axioms alarm is run. The
    reported *power* is the fraction of seeds for which the alarm fires
    (``are_misaligned`` True at the safety level) -- the ground-truth-free
    consistency signal scaled in N.

    The corpus is capped at ``max_q`` because the alarm is ~O(Q^3); a larger
    corpus raises ``ValueError`` (mirroring
    :func:`ntqr_allotment.ntqr_eval.alarm_misaligned`). ``sizes`` larger than
    the available expert pool, empty ``sizes``/``seeds``, sizes ``< 2``, and the
    ``safety`` length not matching the label count (2) also raise ``ValueError``.

    Returns a list of ``(panel_size, power)`` with ``power`` in ``[0, 1]``,
    ordered as ``sizes`` is given.
    """
    q = len(items)
    if q == 0:
        raise ValueError("items must be non-empty")
    if q > max_q:
        raise ValueError(
            f"alarm corpus Q={q} exceeds max_q={max_q}; the NTQR alarm is "
            f"~O(Q^3). Cap the item count or raise max_q deliberately."
        )
    if not sizes:
        raise ValueError("sizes must be non-empty")
    if not seeds:
        raise ValueError("seeds must be non-empty")
    if len(safety) != 2:
        raise ValueError("safety specification length must equal the label count (2)")

    curve: list[tuple[int, float]] = []
    for n in sizes:
        if n < 2:
            raise ValueError("panel size must be at least 2")
        if n > len(experts):
            raise ValueError(f"panel size {n} exceeds expert pool of {len(experts)}")
        panel = experts[:n]
        fires = 0
        for seed in seeds:
            votes_by_judge = [
                sample_votes(e, items, seed=seed + 1009 * (j + 1))
                for j, e in enumerate(panel)
            ]
            if _alarm_fires(votes_by_judge, safety=safety, q=q):
                fires += 1
        curve.append((n, fires / len(seeds)))
    return curve


__all__ = [
    "ALARM_DEFAULT_MAX_Q",
    "observed_vote_counts",
    "panel_agreement_moments",
    "alarm_power_curve",
]
