"""Cross-FAMILY decorrelation: does model-family diversity buy error-independence?

NTQR's exact trio solver assumes the judges' errors are conditionally
*independent*. The project's sortition layer engineers panel diversity to push a
panel toward that assumption. This module tests the specific, falsifiable
hypothesis the Advisor twice named as the only thing that would *validate* (not
merely motivate) the remedy:

    Judges drawn from DIFFERENT model families (e.g. ``qwen2`` vs ``gemma3``)
    make more error-INDEPENDENT mistakes than judges from the SAME family, so a
    cross-family panel sits closer to NTQR's error-independence assumption.

Two paths, kept strictly separate:

* A DETERMINISTIC OFFLINE CORE (:func:`make_family_correlated_votes`) where a
  per-family shared-error latent is injected with a controllable strength. Same-
  family judges co-move (positive realized error-correlation) by construction;
  different families draw independent shared streams (near-zero cross-family
  correlation). This is the TESTED path: it proves the *measurement* recovers a
  known injected structure. It is NOT empirical evidence about real models.

* A LIVE path (:func:`collect_live_family_votes`) that runs real
  :class:`~ntqr_allotment.personas.OllamaJudge` models. It is GUARDED by
  :func:`ollama_panel_available` and is never invoked at import or build time
  inside this module; the orchestrator script drives it separately.

The realized pairwise error-correlations are measured with NTQR's OWN supervised
estimator, reused through :func:`ntqr_allotment.dependence.measure_error_correlations`
(which calls ``SupervisedEvaluation.pair_label_error_correlation``). That helper
is trio-only, but the pair-(0,1) correlation it reports is invariant to the
identity of the third judge, so an arbitrary pair is measured by forming a trio
``[votes_i, votes_j, votes_i]`` and reading the ``(0,1)`` entry.

ANTI-OVERCLAIM (ISC-88): :class:`FamilyContrast` reports a SIGNED magnitude
(cross minus same), explicitly labeled ``"live empirical, n-limited"``. A
negative delta means the measurement recovered the expected ordering on the
sample at hand; it must NEVER be reported as "sortition validated" in absolute
terms.

All offline randomness is seeded (:func:`numpy.random.default_rng`) ->
deterministic.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import combinations
from typing import NamedTuple, Sequence

import numpy as np

from .corpus import CorpusItem
from .dependence import measure_error_correlations
from .personas import OllamaJudge, PersonaSpec
from .statistics_analysis import mean_ci_summary


def model_family(model: str) -> str:
    """Return the coarse model FAMILY for an Ollama model tag.

    The family is the portion of the tag before the first ``":"`` with any
    trailing minor-version run removed, so size/quant suffixes and point
    releases collapse to a single family:

    * ``"qwen2.5:3b"`` -> ``"qwen2"``
    * ``"gemma3:4b"``  -> ``"gemma3"``
    * ``"llama3.1:8b"``-> ``"llama3"``

    The rule keeps the leading integer that names the generation (the ``2`` in
    ``qwen2``, the ``3`` in ``gemma3``) and strips only a *dotted* minor part
    (``.5``, ``.1``). A bare name with no dotted minor (``"gemma3"``) is
    returned unchanged.

    Args:
        model: An Ollama model tag such as ``"qwen2.5:3b"``.

    Returns:
        The lowercase family string.

    Raises:
        ValueError: If ``model`` is empty or whitespace-only.
    """
    base = model.split(":", 1)[0].strip().lower()
    if not base:
        raise ValueError("model tag must be non-empty")
    # Drop a trailing dotted minor-version run (".5", ".1.2") but keep the
    # leading generation integer that is part of the family name.
    return base.split(".", 1)[0]


@dataclass(frozen=True)
class TaggedJudgeVotes:
    """One judge's votes over a fixed item order, tagged with its model family."""

    judge_id: str
    family: str
    votes: tuple[str, ...]


@dataclass(frozen=True)
class PairwiseCorrelationMatrix:
    """Realized pairwise |error-correlation| over a panel of judges.

    ``pair_abs_corr`` maps ``"(i,j)"`` (with ``i < j``, indices into
    ``judge_ids``) to the mean over labels ``a``/``b`` of the absolute realized
    error-correlation for that pair, as measured by NTQR's supervised estimator.
    """

    judge_ids: tuple[str, ...]
    families: tuple[str, ...]
    pair_abs_corr: dict[str, float]

    def value(self, i: int, j: int) -> float:
        """Return the symmetric pair |corr| for judges ``i`` and ``j``.

        Args:
            i: First judge index.
            j: Second judge index (must differ from ``i``).

        Returns:
            The mean absolute realized error-correlation for the pair.

        Raises:
            ValueError: If ``i == j`` or either index is out of range.
        """
        n = len(self.judge_ids)
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(f"judge index out of range for panel of size {n}")
        if i == j:
            raise ValueError("a judge has no error-correlation with itself")
        lo, hi = (i, j) if i < j else (j, i)
        return self.pair_abs_corr[f"({lo},{hi})"]


def _pair_mean_abs_corr(
    votes_i: Sequence[str], votes_j: Sequence[str], items: Sequence[object]
) -> float:
    """Mean over labels of |realized error-correlation| for one judge pair.

    Reuses :func:`ntqr_allotment.dependence.measure_error_correlations` (NTQR's
    supervised estimator). That helper requires exactly three vote sequences;
    its pair-``(0,1)`` correlation is invariant to the third judge, so a filler
    copy of ``votes_i`` is supplied at position 2.
    """
    report = measure_error_correlations([votes_i, votes_j, votes_i], items)
    return float(
        np.mean(
            [abs(report.pair_correlations[f"(0,1)|{label}"]) for label in ("a", "b")]
        )
    )


def build_pairwise_error_correlation_matrix(
    judges_votes: Sequence[TaggedJudgeVotes],
    items: Sequence[object],
) -> PairwiseCorrelationMatrix:
    """Measure the realized |error-correlation| for every pair in the panel.

    Args:
        judges_votes: At least two tagged judges, all voting over ``items`` in
            the same order. ``items`` are the project's ``Item``/``CorpusItem``
            objects (only ``.true_label`` is read, by NTQR's estimator).
        items: The labeled items the votes correspond to.

    Returns:
        A :class:`PairwiseCorrelationMatrix` over the panel.

    Raises:
        ValueError: If fewer than two judges are given, or any judge's vote
            count does not match ``len(items)``.
    """
    if len(judges_votes) < 2:
        raise ValueError("need at least 2 judges to form a pair")
    n_items = len(items)
    for jv in judges_votes:
        if len(jv.votes) != n_items:
            raise ValueError(
                f"judge {jv.judge_id!r} has {len(jv.votes)} votes; expected {n_items}"
            )
    pair_abs_corr: dict[str, float] = {}
    for i, j in combinations(range(len(judges_votes)), 2):
        pair_abs_corr[f"({i},{j})"] = _pair_mean_abs_corr(
            judges_votes[i].votes, judges_votes[j].votes, items
        )
    return PairwiseCorrelationMatrix(
        judge_ids=tuple(jv.judge_id for jv in judges_votes),
        families=tuple(jv.family for jv in judges_votes),
        pair_abs_corr=pair_abs_corr,
    )


class FamilyContrast(NamedTuple):
    """Signed same-vs-cross family error-correlation contrast (n-limited).

    ``delta_cross_minus_same`` is a SIGNED magnitude: a negative value means
    cross-family pairs were *less* error-correlated than same-family pairs on
    this sample. ``label`` is always ``"live empirical, n-limited"`` to keep the
    provenance honest. This is NEVER to be reported as "sortition validated" in
    absolute terms — it is a measured, sample-size-limited contrast.
    """

    mean_abs_same_family: float
    mean_abs_cross_family: float
    delta_cross_minus_same: float
    n_same_pairs: int
    n_cross_pairs: int
    label: str


class PairCorrelationRecord(NamedTuple):
    pair_id: str
    i: int
    j: int
    family_i: str
    family_j: str
    relation: str
    abs_corr: float


class PairGroupSummary(NamedTuple):
    relation: str
    n_pairs: int
    mean_abs_corr: float
    std_abs_corr: float
    ci_low: float
    ci_high: float
    nonzero_pairs: int


def pair_correlation_records(
    matrix: PairwiseCorrelationMatrix,
) -> tuple[PairCorrelationRecord, ...]:
    records: list[PairCorrelationRecord] = []
    for i, j in combinations(range(len(matrix.families)), 2):
        family_i = matrix.families[i]
        family_j = matrix.families[j]
        relation = "same" if family_i == family_j else "cross"
        pair_id = f"({i},{j})"
        records.append(
            PairCorrelationRecord(
                pair_id=pair_id,
                i=i,
                j=j,
                family_i=family_i,
                family_j=family_j,
                relation=relation,
                abs_corr=matrix.value(i, j),
            )
        )
    return tuple(records)


def summarize_pair_groups(
    matrix: PairwiseCorrelationMatrix,
    *,
    seed: int = 0,
    n_boot: int = 10000,
) -> tuple[PairGroupSummary, ...]:
    grouped: dict[str, list[float]] = {}
    for record in pair_correlation_records(matrix):
        grouped.setdefault(record.relation, []).append(record.abs_corr)

    summaries: list[PairGroupSummary] = []
    for offset, relation in enumerate(sorted(grouped)):
        values = grouped[relation]
        summary = mean_ci_summary(values, n_boot=n_boot, seed=seed + offset)
        summaries.append(
            PairGroupSummary(
                relation=relation,
                n_pairs=summary.n,
                mean_abs_corr=summary.mean,
                std_abs_corr=summary.std,
                ci_low=summary.ci_low,
                ci_high=summary.ci_high,
                nonzero_pairs=sum(1 for value in values if abs(value) > 0.0),
            )
        )
    return tuple(summaries)


def contrast_same_vs_cross_family(matrix: PairwiseCorrelationMatrix) -> FamilyContrast:
    """Partition panel pairs by family and report the signed |corr| contrast.

    Same-family pairs share a family tag; cross-family pairs do not. The
    returned contrast reports the mean absolute realized error-correlation for
    each group and the SIGNED delta ``cross - same``. An empty group yields a
    ``nan`` mean (and is propagated into the delta) rather than raising, so a
    single-family or single-pair panel still returns. The result is labeled
    ``"live empirical, n-limited"`` and is a signed magnitude — never an
    absolute "sortition validated" claim.

    Args:
        matrix: A populated :class:`PairwiseCorrelationMatrix`.

    Returns:
        A :class:`FamilyContrast`.
    """
    same: list[float] = []
    cross: list[float] = []
    families = matrix.families
    for i, j in combinations(range(len(families)), 2):
        corr = matrix.value(i, j)
        if families[i] == families[j]:
            same.append(corr)
        else:
            cross.append(corr)
    mean_same = float(np.mean(same)) if same else float("nan")
    mean_cross = float(np.mean(cross)) if cross else float("nan")
    return FamilyContrast(
        mean_abs_same_family=mean_same,
        mean_abs_cross_family=mean_cross,
        delta_cross_minus_same=mean_cross - mean_same,
        n_same_pairs=len(same),
        n_cross_pairs=len(cross),
        label="live empirical, n-limited",
    )


class MultiSeedContrast(NamedTuple):
    """Aggregate of per-run :class:`FamilyContrast` deltas across seeds/corpora.

    Upgrades a single n-limited run toward a measured estimate: reports how stable
    the *sign* of the cross-minus-same delta is across independent runs, with its
    mean and spread. ``sign_stability`` is the fraction of runs whose delta is
    negative (the decorrelation direction); 1.0 means every run agreed, 0.5 means
    the sign is a coin-flip. Still labeled n-limited — more runs, not a different
    claim class.
    """

    n_runs: int
    mean_delta: float
    std_delta: float
    sign_stability: float
    min_delta: float
    max_delta: float
    deltas: tuple[float, ...]
    label: str


def aggregate_contrasts(contrasts: Sequence[FamilyContrast]) -> MultiSeedContrast:
    """Aggregate per-run family contrasts into a sign-stability summary.

    Args:
        contrasts: One :class:`FamilyContrast` per independent run (non-empty);
            runs with a non-finite delta (a degenerate single-family panel) are
            dropped before aggregation.

    Returns:
        A :class:`MultiSeedContrast` over the finite-delta runs.

    Raises:
        ValueError: If ``contrasts`` is empty or every run had a non-finite delta.
    """
    if not contrasts:
        raise ValueError("contrasts must be non-empty")
    deltas = [
        c.delta_cross_minus_same
        for c in contrasts
        if np.isfinite(c.delta_cross_minus_same)
    ]
    if not deltas:
        raise ValueError("no finite-delta runs to aggregate")
    arr = np.asarray(deltas, dtype=float)
    n = int(arr.size)
    negative = int(np.sum(arr < 0.0))
    return MultiSeedContrast(
        n_runs=n,
        mean_delta=float(arr.mean()),
        std_delta=float(arr.std(ddof=1)) if n > 1 else 0.0,
        sign_stability=negative / n,
        min_delta=float(arr.min()),
        max_delta=float(arr.max()),
        deltas=tuple(float(d) for d in deltas),
        label="live empirical, n-limited (multi-seed)",
    )


def make_family_correlated_votes(
    *,
    families: Sequence[str],
    corpus: Sequence[CorpusItem],
    target_accuracy: float = 0.75,
    shared_strength: float = 0.7,
    seed: int = 0,
) -> list[TaggedJudgeVotes]:
    """Deterministic offline panel with a per-FAMILY shared error component.

    Generative model (mirrors the shared-latent idiom in
    :func:`ntqr_allotment.dependence.sample_votes_correlated`, but the shared
    latent is keyed by *family* rather than shared by all judges):

    * For each distinct family draw one shared per-item latent stream
      ``s_fam[family][k] ~ U(0, 1)``.
    * For each judge ``j`` draw an independent per-item latent
      ``u_j[k] ~ U(0, 1)``.
    * Judge ``j`` (of family ``f``) is correct on item ``k`` iff
      ``shared_strength * s_fam[f][k] + (1 - shared_strength) * u_j[k]
      < target_accuracy``.

    Two judges of the SAME family read the same shared stream, so their
    correctness co-moves (positive realized error-correlation); judges of
    DIFFERENT families read independent shared streams, so their errors are
    near-independent. ``shared_strength = 0`` collapses to fully independent
    judges (the negative-control regime).

    Args:
        families: One family tag per judge (non-empty).
        corpus: Labeled corpus items (non-empty); votes align by position.
        target_accuracy: Per-judge target accuracy in ``(0, 1)``.
        shared_strength: Weight on the shared family latent, in ``[0, 1]``.
        seed: RNG seed; identical seeds give identical panels.

    Returns:
        One :class:`TaggedJudgeVotes` per family, in input order.

    Raises:
        ValueError: On empty ``families``/``corpus`` or out-of-range
            ``target_accuracy``/``shared_strength``.
    """
    if not families:
        raise ValueError("families must be non-empty")
    if not corpus:
        raise ValueError("corpus must be non-empty")
    if not 0.0 < target_accuracy < 1.0:
        raise ValueError("target_accuracy must be in (0, 1)")
    if not 0.0 <= shared_strength <= 1.0:
        raise ValueError("shared_strength must be in [0, 1]")

    n = len(corpus)
    distinct = sorted(set(families))
    # One independent shared stream per distinct family. The stream seed is
    # derived deterministically from the family's stable position so distinct
    # families never share a stream.
    shared: dict[str, np.ndarray] = {
        fam: np.random.default_rng(seed + 1009 * (idx + 1)).random(n)
        for idx, fam in enumerate(distinct)
    }

    tagged: list[TaggedJudgeVotes] = []
    for j, fam in enumerate(families):
        indep = np.random.default_rng(seed + 7919 * (j + 1)).random(n)
        mixed = shared_strength * shared[fam] + (1.0 - shared_strength) * indep
        row: list[str] = []
        for k, item in enumerate(corpus):
            correct = mixed[k] < target_accuracy
            if correct:
                row.append(item.true_label)
            else:
                row.append("b" if item.true_label == "a" else "a")
        tagged.append(
            TaggedJudgeVotes(judge_id=f"{fam}-{j}", family=fam, votes=tuple(row))
        )
    return tagged


def ollama_panel_available(
    models: Sequence[str], base_url: str = "http://localhost:11434"
) -> bool:
    """Cheap guard: is a live Ollama server reachable for the panel?

    Probes only the FIRST model's server via
    :meth:`~ntqr_allotment.personas.OllamaJudge.available` (which already
    swallows connection errors). No votes are cast.

    Args:
        models: The model tags the live panel would use.
        base_url: Ollama base URL.

    Returns:
        ``True`` iff the server answers; ``False`` if ``models`` is empty or the
        server is unreachable.
    """
    if not models:
        return False
    return OllamaJudge(model=models[0], base_url=base_url).available()


def collect_live_family_votes(
    *,
    models: Sequence[str],
    persona: PersonaSpec,
    corpus: Sequence[CorpusItem],
    base_url: str = "http://localhost:11434",
    temperature: float = 0.6,
    timeout: float = 30.0,
    num_predict: int = 4,
    progress_every: int | None = None,
) -> list[TaggedJudgeVotes]:
    """Run real Ollama models over the corpus, one tagged judge per model.

    This is the LIVE path. It is invoked only by the orchestrator script after
    :func:`ollama_panel_available` returns ``True``; it is never called at
    import or build time. The per-item network loop carries ``pragma: no
    cover`` because it cannot execute without a running server, but the input
    validation around it is covered.

    Args:
        models: Model tags; the family is derived via :func:`model_family`.
        persona: The shared persona spec presented to every judge.
        corpus: Labeled corpus items.
        base_url: Ollama base URL.
        temperature: Sampling temperature for every judge.

    Returns:
        One :class:`TaggedJudgeVotes` per model, in input order.

    Raises:
        ValueError: If ``models`` is empty.
    """
    if not models:
        raise ValueError("models must be non-empty")
    if timeout <= 0.0:
        raise ValueError("timeout must be positive")
    if num_predict <= 0:
        raise ValueError("num_predict must be positive")
    if progress_every is not None and progress_every <= 0:
        raise ValueError("progress_every must be positive")
    return _collect_live_votes_impl(  # pragma: no cover - requires a live server
        models, persona, corpus, base_url, temperature, timeout, num_predict, progress_every
    )


def _collect_live_votes_impl(  # pragma: no cover - requires a live Ollama server
    models: Sequence[str],
    persona: PersonaSpec,
    corpus: Sequence[CorpusItem],
    base_url: str,
    temperature: float,
    timeout: float,
    num_predict: int,
    progress_every: int | None,
) -> list[TaggedJudgeVotes]:
    """Network loop for :func:`collect_live_family_votes` (live-only)."""
    tagged: list[TaggedJudgeVotes] = []
    for idx, model in enumerate(models):
        family = model_family(model)
        judge = OllamaJudge(
            model=model,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
            seed=idx,
            num_predict=num_predict,
        )
        votes_list: list[str] = []
        for item_index, item in enumerate(corpus, start=1):
            votes_list.append(judge.judge(persona, item.text))
            if progress_every is not None and item_index % progress_every == 0:
                print(
                    f"{model} judge {idx}: {item_index}/{len(corpus)} items",
                    file=sys.stderr,
                    flush=True,
                )
        votes = tuple(votes_list)
        # Index keeps the id unique when the same model appears twice (the
        # script runs two seeded judges per model for same-family pairs).
        tagged.append(
            TaggedJudgeVotes(
                judge_id=f"{family}-{model}-{idx}", family=family, votes=votes
            )
        )
    return tagged


__all__ = [
    "model_family",
    "TaggedJudgeVotes",
    "PairwiseCorrelationMatrix",
    "PairCorrelationRecord",
    "PairGroupSummary",
    "build_pairwise_error_correlation_matrix",
    "pair_correlation_records",
    "summarize_pair_groups",
    "FamilyContrast",
    "contrast_same_vs_cross_family",
    "MultiSeedContrast",
    "aggregate_contrasts",
    "make_family_correlated_votes",
    "ollama_panel_available",
    "collect_live_family_votes",
]
