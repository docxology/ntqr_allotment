"""Empirical track: LLM-persona judges (the centerpiece).

Expert competence and bias are encoded into a *persona prompt* and realized by a
local Ollama model, queried over plain HTTP via ``requests``. Persona votes feed
the SAME sortition -> NTQR -> oracle pipeline as the synthetic track, so the two
tracks are directly comparable (and kept clearly separate — no conflation).

Judges implement a tiny protocol, so the empirical pipeline can run against:
  * ``OllamaJudge``        -- a live local model (the real centerpiece run), or
  * ``DeterministicJudge`` -- a real, seeded, in-process judge at a target
    accuracy (NOT a mock: genuine logic), used for offline runs and for tests
    that must stay green without a server.

Live-model tests are gated behind ``@pytest.mark.requires_ollama`` and skip
cleanly when no server is reachable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import requests

from .corpus import CorpusItem
from .experts import Expert
from .ntqr_eval import (
    Evaluation,
    closest_solution,
    error_independent_solutions,
    majority_voting_solutions,
    supervised_oracle,
)

_EXPERTISE_LEVELS = ("novice", "competent", "expert")
_DEFAULT_OLLAMA_HOST = "http://localhost:11434"
_DEFAULT_NUM_PREDICT = 4


@dataclass(frozen=True)
class PersonaSpec:
    """A judge persona: qualitative competence/leaning + a quantitative target.

    ``expertise_level`` and ``leaning`` are the qualitative knobs surfaced in the
    prompt; ``target_accuracy`` is the quantitative competence used by the
    deterministic judge and for reporting. ``ideology`` carries the sortition
    feature so personas can be drawn by the same strategies as synthetic experts.
    """

    id: str
    expertise_level: str
    leaning: str  # "toward_true" | "toward_false" | "neutral"
    ideology: str
    target_accuracy: float

    def build_prompt(self, item_text: str) -> tuple[str, str]:
        """Return (system, user) prompts encoding competence + leaning.

        The persona is told to answer with a single token 'a' (TRUE) or 'b'
        (FALSE). Competence and leaning shape the instruction, not the parser.
        """
        care = {
            "novice": "You are hasty and often make arithmetic slips; do not double-check.",
            "competent": "You are reasonably careful with arithmetic.",
            "expert": "You are a meticulous arithmetician; verify the sum exactly.",
        }[self.expertise_level]
        lean = {
            "toward_true": "When unsure, you tend to assume statements are correct.",
            "toward_false": "When unsure, you tend to assume statements are wrong.",
            "neutral": "When unsure, you guess without preference.",
        }[self.leaning]
        system = (
            f"You judge whether an arithmetic equation is correct. {care} {lean} "
            "Answer with EXACTLY one letter: 'a' if the equation is correct, "
            "'b' if it is incorrect. Output only that single letter."
        )
        user = f"Equation: {item_text}\nIs it correct? Answer 'a' or 'b'."
        return system, user


@dataclass(frozen=True)
class ModelProvenance:
    """Model identity plus decode settings for a reproducible persona judge."""

    model: str
    family: str | None
    digest: str | None
    temperature: float
    seed: int
    num_predict: int

    def to_dict(self) -> dict[str, str | float | int | None]:
        """Return a JSON-serializable representation of the provenance."""
        return {
            "model": self.model,
            "family": self.family,
            "digest": self.digest,
            "temperature": self.temperature,
            "seed": self.seed,
            "num_predict": self.num_predict,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str | float | int | None]) -> ModelProvenance:
        """Reconstruct a provenance record from ``to_dict()`` output."""
        if not isinstance(data, dict):
            raise TypeError("model provenance payload must be a dict")
        family = data.get("family")
        digest = data.get("digest")
        if family is not None and not isinstance(family, str):
            raise TypeError("family must be a string or None")
        if digest is not None and not isinstance(digest, str):
            raise TypeError("digest must be a string or None")
        return cls(
            model=str(data["model"]),
            family=family,
            digest=digest,
            temperature=float(data["temperature"]),
            seed=int(data["seed"]),
            num_predict=int(data["num_predict"]),
        )


def personas_from_population(experts: Sequence[Expert]) -> list[PersonaSpec]:
    """Map synthetic experts onto LLM personas (qualitative + quantitative)."""
    personas: list[PersonaSpec] = []
    for e in experts:
        level = _EXPERTISE_LEVELS[min(2, max(0, int((e.expertise - 0.5) / 0.17)))]
        if e.bias > 0.15:
            leaning = "toward_true"
        elif e.bias < -0.15:
            leaning = "toward_false"
        else:
            leaning = "neutral"
        personas.append(
            PersonaSpec(
                id=e.id,
                expertise_level=level,
                leaning=leaning,
                ideology=e.ideology,
                target_accuracy=e.mean_accuracy,
            )
        )
    return personas


def parse_label(text: str) -> str | None:
    """Robustly extract 'a'/'b' from a model's free-text answer; None if unclear.

    Word cues win over a bare letter scan ("incorrect" is checked before
    "correct" since it contains it), so "false"/"incorrect" do not get
    mis-read as 'a' by the letter in the word.
    """
    low = text.strip().lower()
    if any(w in low for w in ("incorrect", "false", "wrong")):
        return "b"
    if any(w in low for w in ("correct", "true")):
        return "a"
    for ch in low:
        if ch == "a":
            return "a"
        if ch == "b":
            return "b"
    return None


def _matching_model_names(model: str) -> tuple[str, ...]:
    names = [model]
    if ":" not in model:
        names.append(f"{model}:latest")
    return tuple(names)


def fetch_model_provenance(
    model: str,
    host: str = _DEFAULT_OLLAMA_HOST,
    *,
    timeout: float = 3.0,
) -> ModelProvenance | None:
    """Fetch digest/family for ``model`` from Ollama ``GET /api/tags``.

    Returns ``None`` when the HTTP request raises ``requests.RequestException``,
    when Ollama replies with a non-200 status, or when no model entry matches
    ``model`` exactly (also accepting ``f"{model}:latest"`` when ``model`` has
    no explicit tag). Raises ``ValueError`` if the top-level ``/api/tags``
    payload does not match the expected ``{"models": [...]}`` shape.

    The returned object uses the deterministic decode defaults; call
    ``provenance_for_judge()`` to bind the live decode parameters from a
    specific ``OllamaJudge`` instance.
    """
    try:
        response = requests.get(f"{host}/api/tags", timeout=timeout)
    except requests.RequestException:
        return None
    if response.status_code != 200:
        return None

    payload = response.json()
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise ValueError("unexpected Ollama /api/tags response shape")

    candidate_names = _matching_model_names(model)
    for candidate in candidate_names:
        for entry in models:
            if not isinstance(entry, dict):
                continue
            if entry.get("name") != candidate:
                continue
            details = entry.get("details")
            family = details.get("family") if isinstance(details, dict) else None
            digest = entry.get("digest")
            return ModelProvenance(
                model=model,
                family=family if isinstance(family, str) else None,
                digest=digest if isinstance(digest, str) else None,
                temperature=0.0,
                seed=0,
                num_predict=_DEFAULT_NUM_PREDICT,
            )
    return None


class Judge(Protocol):
    """Anything that can turn (persona, item text) into a binary label."""

    def judge(self, persona: PersonaSpec, item_text: str) -> str: ...


@dataclass
class OllamaJudge:
    """A live local-model judge backed by an Ollama server.

    ``temperature``, ``seed``, and ``num_predict`` are the decode parameters
    recorded into persona-run provenance. Defaults remain deterministic
    (temp 0, seed 0, num_predict 4).
    """

    model: str = "qwen2.5:3b"
    base_url: str = _DEFAULT_OLLAMA_HOST
    timeout: float = 30.0
    temperature: float = 0.0
    seed: int = 0
    num_predict: int = _DEFAULT_NUM_PREDICT
    #: Transient-failure retries. A single slow generation (Ollama model reload,
    #: memory pressure) must not kill a multi-thousand-call run; retries are NOT a
    #: decode parameter, so the vote-cache key is unchanged and a resumed run still
    #: reuses prior votes. A persistent failure still raises after the retries.
    max_retries: int = 4

    def available(self) -> bool:
        """True iff the Ollama server answers AND this judge's model is pulled.

        Model-aware on purpose: a server that is up but lacks ``self.model`` 404s on
        ``/api/generate``, so treating "server answers" as "available" makes guarded
        live tests *fail* instead of *skip* when the model is absent (e.g. evicted
        under disk pressure). Checking model presence in ``/api/tags`` honors the
        documented "skips if down/unavailable" contract.
        """
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3.0)
            if r.status_code != 200:
                return False
            names = {
                m.get("name")
                for m in r.json().get("models", [])
                if isinstance(m, dict)
            }
            return self.model in names
        except requests.RequestException:
            return False

    def judge(self, persona: PersonaSpec, item_text: str) -> str:
        system, user = persona.build_prompt(item_text)
        payload = {
            "model": self.model,
            "prompt": user,
            "system": system,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_predict": self.num_predict,
            },
        }
        last_exc: requests.RequestException | None = None
        for _attempt in range(max(1, self.max_retries)):
            try:
                r = requests.post(
                    f"{self.base_url}/api/generate", json=payload, timeout=self.timeout
                )
                r.raise_for_status()
                label = parse_label(r.json().get("response", ""))
                return label if label is not None else "b"  # unparseable -> conservative 'b'
            except (requests.Timeout, requests.ConnectionError) as exc:
                # Transient: server stalled or briefly unreachable. Retry rather than
                # abort the whole run; the vote already in the cache is never re-asked.
                last_exc = exc
                continue
        # Retries exhausted: fail loudly rather than silently fabricate a vote.
        raise last_exc if last_exc is not None else RuntimeError("ollama judge failed")


def provenance_for_judge(judge: OllamaJudge, host: str | None = None) -> ModelProvenance:
    """Build a provenance record for one ``OllamaJudge``.

    Decode parameters always come from the judge instance. If Ollama metadata is
    unavailable, ``family`` and ``digest`` are set to ``None`` while the decode
    parameters remain populated.
    """
    resolved_host = judge.base_url if host is None else host
    fetched = fetch_model_provenance(judge.model, host=resolved_host)
    return ModelProvenance(
        model=judge.model,
        family=None if fetched is None else fetched.family,
        digest=None if fetched is None else fetched.digest,
        temperature=judge.temperature,
        seed=judge.seed,
        num_predict=judge.num_predict,
    )


@dataclass
class DeterministicJudge:
    """A real, seeded in-process judge that answers at each persona's target accuracy.

    Not a mock: it computes a genuine answer from the item's true label and a
    deterministic hash of (persona, item), correct with probability
    ``target_accuracy``. Enables offline empirical runs and server-free tests.
    """

    corpus: Sequence[CorpusItem]
    seed: int = 0

    def __post_init__(self) -> None:
        self._truth = {it.text: it.true_label for it in self.corpus}

    def judge(self, persona: PersonaSpec, item_text: str) -> str:
        truth = self._truth[item_text]
        h = hashlib.sha256(f"{self.seed}|{persona.id}|{item_text}".encode()).digest()
        draw = int.from_bytes(h[:8], "big") / 2**64
        correct = draw < persona.target_accuracy
        if correct:
            return truth
        return "b" if truth == "a" else "a"


def judge_corpus(
    judge: Judge,
    persona: PersonaSpec,
    items: Sequence[CorpusItem],
    *,
    cache_path: str | Path | None = None,
) -> tuple[str, ...]:
    """Collect one persona's votes over the corpus, optionally caching to JSON."""
    cache: dict[str, str] = {}
    path = Path(cache_path) if cache_path else None
    if path is not None and path.exists():
        cache = json.loads(path.read_text())
    votes: list[str] = []
    dirty = False
    for it in items:
        key = f"{persona.id}|{it.index}"
        if key in cache:
            votes.append(cache[key])
            continue
        v = judge.judge(persona, it.text)
        cache[key] = v
        votes.append(v)
        dirty = True
    if path is not None and dirty:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=2, sort_keys=True))
    return tuple(votes)


@dataclass(frozen=True)
class PersonaTrioResult:
    """Empirical NTQR evaluation of a trio of persona judges."""

    persona_ids: tuple[str, ...]
    oracle: Evaluation
    eie_estimate: Evaluation
    mv_estimate: Evaluation
    eie_error: float
    mv_error: float


def run_persona_trio(
    judge: Judge,
    personas: Sequence[PersonaSpec],
    items: Sequence[CorpusItem],
    *,
    cache_path: str | Path | None = None,
) -> PersonaTrioResult:
    """Run the empirical centerpiece: 3 persona judges -> NTQR -> oracle scoring."""
    if len(personas) < 3:
        raise ValueError("need at least 3 personas for the trio solver")
    trio = list(personas[:3])
    votes = [judge_corpus(judge, p, items, cache_path=cache_path) for p in trio]
    oracle = supervised_oracle(votes, items)
    eie_sols = error_independent_solutions(votes)
    mv_sols = majority_voting_solutions(votes)
    if not eie_sols or not mv_sols:
        raise ValueError("persona trio admits no real NTQR solution (degenerate votes)")
    eie = closest_solution(eie_sols, oracle)
    mv = closest_solution(mv_sols, oracle)
    return PersonaTrioResult(
        persona_ids=tuple(p.id for p in trio),
        oracle=oracle,
        eie_estimate=eie,
        mv_estimate=mv,
        eie_error=eie.error_vs(oracle),
        mv_error=mv.error_vs(oracle),
    )


__all__ = [
    "PersonaSpec",
    "ModelProvenance",
    "personas_from_population",
    "parse_label",
    "fetch_model_provenance",
    "Judge",
    "OllamaJudge",
    "provenance_for_judge",
    "DeterministicJudge",
    "judge_corpus",
    "PersonaTrioResult",
    "run_persona_trio",
]
