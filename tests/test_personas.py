"""Tests for the corpus + LLM-persona layer (no mocks).

Server-free tests use the real ``DeterministicJudge`` (genuine seeded logic, not
a mock). Live-model tests are marked ``requires_ollama`` and skip when no server
is reachable.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Iterator

import pytest

from ntqr_allotment.corpus import make_arithmetic_corpus
from ntqr_allotment.experts import generate_population
from ntqr_allotment.personas import (
    DeterministicJudge,
    ModelProvenance,
    OllamaJudge,
    PersonaSpec,
    PersonaTrioResult,
    fetch_model_provenance,
    judge_corpus,
    parse_label,
    personas_from_population,
    provenance_for_judge,
    run_persona_trio,
)


# ---- real local HTTP server (no mocks: an actual server on a real socket) ----

def _serve_tags(body: bytes, status: int = 200) -> Iterator[str]:
    """Yield the base URL of a real, threaded HTTP server for ``/api/tags``.

    A genuine ``http.server`` instance on a real loopback socket — not a mock.
    It replies to ``GET /api/tags`` with ``body``/``status`` and 404s anything
    else, so ``fetch_model_provenance`` exercises a true HTTP round trip.
    """

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
            if self.path != "/api/tags":
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:  # silence test-server noise
            return

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5.0)
        server.server_close()


# ---- corpus -----------------------------------------------------------------

def test_corpus_is_deterministic_and_labeled():
    a = make_arithmetic_corpus(50, seed=1)
    b = make_arithmetic_corpus(50, seed=1)
    assert [it.text for it in a] == [it.text for it in b]
    assert all(it.true_label in ("a", "b") for it in a)


def test_corpus_labels_match_arithmetic():
    for it in make_arithmetic_corpus(200, seed=2):
        x, rest = it.text.split(" + ")
        y, z = rest.split(" = ")
        is_correct = int(x) + int(y) == int(z)
        assert (it.true_label == "a") == is_correct
        assert it.statement == it.text


def test_corpus_prevalence_roughly_holds():
    items = make_arithmetic_corpus(1000, seed=3, prevalence_true=0.3)
    frac_true = sum(1 for it in items if it.true_label == "a") / len(items)
    assert 0.25 < frac_true < 0.35


def test_corpus_validation():
    with pytest.raises(ValueError):
        make_arithmetic_corpus(0, seed=1)
    with pytest.raises(ValueError):
        make_arithmetic_corpus(10, seed=1, prevalence_true=2.0)


# ---- persona specs ----------------------------------------------------------

def test_build_prompt_encodes_competence_and_leaning():
    p = PersonaSpec("p1", "expert", "toward_false", "left", 0.9)
    system, user = p.build_prompt("2 + 2 = 5")
    assert "meticulous" in system  # expert competence cue
    assert "wrong" in system  # toward_false leaning cue
    assert "'a'" in system and "'b'" in system
    assert "2 + 2 = 5" in user


@pytest.mark.parametrize("level", ["novice", "competent", "expert"])
def test_build_prompt_all_levels(level):
    p = PersonaSpec("p", level, "neutral", "center", 0.7)
    system, _ = p.build_prompt("1 + 1 = 2")
    assert isinstance(system, str) and system


def test_personas_from_population_maps_params():
    pop = generate_population(30, seed=4)
    personas = personas_from_population(pop)
    assert len(personas) == 30
    assert all(p.expertise_level in ("novice", "competent", "expert") for p in personas)
    assert all(p.leaning in ("toward_true", "toward_false", "neutral") for p in personas)
    # ideology carried through; target accuracy matches the expert's mean accuracy
    by_id = {e.id: e for e in pop}
    for p in personas:
        assert p.ideology == by_id[p.id].ideology
        assert p.target_accuracy == pytest.approx(by_id[p.id].mean_accuracy)


# ---- parsing ----------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("a", "a"), ("b", "b"), ("A", "a"), (" B ", "b"),
        ("The answer is a", "a"), ("true", "a"), ("Correct", "a"),
        ("false", "b"), ("incorrect", "b"), ("wrong", "b"),
        ("42", None), ("", None),
    ],
)
def test_parse_label(text, expected):
    assert parse_label(text) == expected


# ---- deterministic judge (real, server-free) --------------------------------

def test_deterministic_judge_hits_target_accuracy():
    corpus = make_arithmetic_corpus(800, seed=5)
    persona = PersonaSpec("p", "competent", "neutral", "center", 0.75)
    judge = DeterministicJudge(corpus, seed=1)
    votes = [judge.judge(persona, it.text) for it in corpus]
    acc = sum(1 for it, v in zip(corpus, votes) if v == it.true_label) / len(corpus)
    assert acc == pytest.approx(0.75, abs=0.05)
    assert set(votes) <= {"a", "b"}


def test_deterministic_judge_is_deterministic():
    corpus = make_arithmetic_corpus(100, seed=5)
    p = PersonaSpec("p", "expert", "neutral", "center", 0.9)
    j1 = DeterministicJudge(corpus, seed=2)
    j2 = DeterministicJudge(corpus, seed=2)
    assert [j1.judge(p, it.text) for it in corpus] == [j2.judge(p, it.text) for it in corpus]


def test_judge_corpus_caches(tmp_path):
    corpus = make_arithmetic_corpus(40, seed=6)
    persona = PersonaSpec("p", "competent", "neutral", "center", 0.8)
    judge = DeterministicJudge(corpus, seed=3)
    cache = tmp_path / "votes.json"
    first = judge_corpus(judge, persona, corpus, cache_path=cache)
    assert cache.exists()
    second = judge_corpus(judge, persona, corpus, cache_path=cache)  # read from cache
    assert first == second
    assert len(first) == 40


def test_run_persona_trio_offline():
    corpus = make_arithmetic_corpus(300, seed=7)
    pop = generate_population(10, seed=7, mean_expertise=0.8, expertise_heterogeneity=0.05)
    personas = personas_from_population(pop)
    judge = DeterministicJudge(corpus, seed=7)
    result = run_persona_trio(judge, personas, corpus)
    assert isinstance(result, PersonaTrioResult)
    assert result.eie_error >= 0.0
    assert 0.0 <= result.oracle.prevalence_a <= 1.0
    assert len(result.persona_ids) == 3


def test_run_persona_trio_requires_three():
    corpus = make_arithmetic_corpus(50, seed=8)
    personas = personas_from_population(generate_population(2, seed=8))
    with pytest.raises(ValueError):
        run_persona_trio(DeterministicJudge(corpus, seed=8), personas, corpus)


def test_ollama_available_false_on_dead_port():
    judge = OllamaJudge(base_url="http://localhost:6553")  # nothing listening
    assert judge.available() is False


# ---- live Ollama (skips without a server) -----------------------------------

@pytest.mark.requires_ollama
def test_ollama_judge_returns_valid_label():
    judge = OllamaJudge()
    if not judge.available():
        pytest.skip("no live Ollama server")
    persona = PersonaSpec("p", "expert", "neutral", "center", 0.9)
    label = judge.judge(persona, "2 + 2 = 4")
    assert label in ("a", "b")


@pytest.mark.requires_ollama
def test_ollama_persona_trio_runs_live():
    judge = OllamaJudge()
    if not judge.available():
        pytest.skip("no live Ollama server")
    corpus = make_arithmetic_corpus(40, seed=9, max_operand=99, max_error=2)
    personas = personas_from_population(generate_population(6, seed=9, mean_expertise=0.85))
    # Near-perfect, highly-correlated live judges can yield a DEGENERATE NTQR
    # system (no real error-independent solution) -- a genuine property, not a
    # bug. Accept either a finite evaluation or the documented degenerate raise.
    try:
        result = run_persona_trio(judge, personas, corpus)
    except ValueError as exc:
        assert "degenerate" in str(exc)
        return
    assert result.eie_error >= 0.0
    assert len(result.persona_ids) == 3


# ---- model provenance (ISC-55): offline round-trip + guards ------------------

def test_model_provenance_round_trips_through_dict():
    """to_dict()/from_dict() reconstruct an equal object (no live call)."""
    prov = ModelProvenance(
        model="qwen2.5:3b",
        family="qwen2",
        digest="sha256:deadbeef",
        temperature=0.6,
        seed=7,
        num_predict=4,
    )
    as_dict = prov.to_dict()
    # JSON-serializable, so it can be embedded in the artifact on disk.
    json.dumps(as_dict)
    # Negative-control: a dropped field in from_dict would fail this equality.
    assert ModelProvenance.from_dict(as_dict) == prov
    assert as_dict["digest"] == "sha256:deadbeef"
    assert as_dict["num_predict"] == 4


def test_provenance_embeds_as_per_judge_block_in_artifact():
    """A trio's provenance embeds as a JSON-serializable per-judge block."""
    judges = [
        ModelProvenance("qwen2.5:3b", "qwen2", "sha256:aaa", 0.6, 1, 4),
        ModelProvenance("gemma3:4b", "gemma3", "sha256:bbb", 0.6, 2, 4),
        ModelProvenance("qwen2.5:3b", "qwen2", "sha256:aaa", 1.0, 3, 4),
    ]
    artifact = {
        "judge_models": ["qwen2.5:3b@0.6", "gemma3:4b@0.6", "qwen2.5:3b@1.0"],
        "provenance": [p.to_dict() for p in judges],
    }
    reloaded = json.loads(json.dumps(artifact))  # round-trip through real JSON
    assert len(reloaded["provenance"]) == 3
    restored = [ModelProvenance.from_dict(p) for p in reloaded["provenance"]]
    assert restored == judges
    # Distinct seeds/temperatures are preserved per judge (reproducibility).
    assert [p.seed for p in restored] == [1, 2, 3]
    assert {p.family for p in restored} == {"qwen2", "gemma3"}


def test_from_dict_rejects_non_dict_payload():
    with pytest.raises(TypeError):
        ModelProvenance.from_dict([1, 2, 3])  # type: ignore[arg-type]


def test_from_dict_rejects_wrongly_typed_family():
    bad = {
        "model": "m",
        "family": 5,  # not a string
        "digest": None,
        "temperature": 0.0,
        "seed": 0,
        "num_predict": 4,
    }
    with pytest.raises(TypeError):
        ModelProvenance.from_dict(bad)


def test_from_dict_rejects_wrongly_typed_digest():
    bad = {
        "model": "m",
        "family": None,
        "digest": 123,  # not a string
        "temperature": 0.0,
        "seed": 0,
        "num_predict": 4,
    }
    with pytest.raises(TypeError):
        ModelProvenance.from_dict(bad)


def test_fetch_model_provenance_returns_none_on_dead_port():
    """Guard: an unreachable Ollama yields None, not an exception."""
    assert fetch_model_provenance("qwen2.5:3b", host="http://localhost:6553") is None


def test_fetch_model_provenance_reads_digest_and_family_from_real_server():
    """Success path against a real local HTTP server (no mocks)."""
    body = json.dumps(
        {
            "models": [
                {"name": "other:1b", "digest": "sha256:zzz", "details": {"family": "other"}},
                {"name": "qwen2.5:3b", "digest": "sha256:abc123", "details": {"family": "qwen2"}},
            ]
        }
    ).encode()
    server = _serve_tags(body)
    host = next(server)
    try:
        prov = fetch_model_provenance("qwen2.5:3b", host=host)
    finally:
        server.close()
    assert prov is not None
    # Negative-control: matching the wrong entry would change these values.
    assert prov.digest == "sha256:abc123"
    assert prov.family == "qwen2"
    assert prov.model == "qwen2.5:3b"


def test_fetch_model_provenance_accepts_latest_tag_for_bare_name():
    """A bare model name matches the ``:latest`` entry when no tag is given."""
    body = json.dumps(
        {"models": [{"name": "llama3:latest", "digest": "sha256:l", "details": {"family": "llama"}}]}
    ).encode()
    server = _serve_tags(body)
    host = next(server)
    try:
        prov = fetch_model_provenance("llama3", host=host)
    finally:
        server.close()
    assert prov is not None
    assert prov.digest == "sha256:l"
    assert prov.family == "llama"


def test_fetch_model_provenance_returns_none_when_model_absent():
    body = json.dumps({"models": [{"name": "other:1b", "digest": "sha256:z"}]}).encode()
    server = _serve_tags(body)
    host = next(server)
    try:
        assert fetch_model_provenance("qwen2.5:3b", host=host) is None
    finally:
        server.close()


def test_fetch_model_provenance_none_on_non_200_status():
    server = _serve_tags(b"nope", status=503)
    host = next(server)
    try:
        assert fetch_model_provenance("qwen2.5:3b", host=host) is None
    finally:
        server.close()


def test_fetch_model_provenance_raises_on_bad_shape():
    body = json.dumps({"unexpected": "payload"}).encode()
    server = _serve_tags(body)
    host = next(server)
    try:
        with pytest.raises(ValueError):
            fetch_model_provenance("qwen2.5:3b", host=host)
    finally:
        server.close()


def test_fetch_model_provenance_skips_non_dict_model_entries():
    """A malformed (non-dict) entry in the models list is skipped, not trusted."""
    body = json.dumps(
        {
            "models": [
                "not-a-dict",
                {"name": "qwen2.5:3b", "digest": "sha256:ok", "details": {"family": "qwen2"}},
            ]
        }
    ).encode()
    server = _serve_tags(body)
    host = next(server)
    try:
        prov = fetch_model_provenance("qwen2.5:3b", host=host)
    finally:
        server.close()
    assert prov is not None
    assert prov.digest == "sha256:ok"


def test_fetch_model_provenance_handles_missing_details_block():
    """An entry without a ``details`` block yields family=None, digest kept."""
    body = json.dumps({"models": [{"name": "qwen2.5:3b", "digest": "sha256:d"}]}).encode()
    server = _serve_tags(body)
    host = next(server)
    try:
        prov = fetch_model_provenance("qwen2.5:3b", host=host)
    finally:
        server.close()
    assert prov is not None
    assert prov.family is None
    assert prov.digest == "sha256:d"


def test_ollama_judge_default_num_predict_is_four():
    """Single source of truth: the default decode budget is 4 tokens."""
    assert OllamaJudge().num_predict == 4


def test_provenance_for_judge_uses_judge_decode_params_offline():
    """Decode params come from the judge; digest/family are None when offline."""
    judge = OllamaJudge(
        model="qwen2.5:3b",
        base_url="http://localhost:6553",  # dead port -> no live metadata
        temperature=0.6,
        seed=11,
        num_predict=4,
    )
    prov = provenance_for_judge(judge)
    # Decode params are populated straight from the judge (reproducibility).
    assert prov.model == "qwen2.5:3b"
    assert prov.temperature == 0.6
    assert prov.seed == 11
    assert prov.num_predict == 4
    # Offline -> no live digest/family, but the block is still well-formed.
    assert prov.digest is None
    assert prov.family is None
    json.dumps(prov.to_dict())


def test_provenance_for_judge_binds_live_metadata_from_real_server():
    """provenance_for_judge merges live digest/family with judge decode params."""
    body = json.dumps(
        {"models": [{"name": "qwen2.5:3b", "digest": "sha256:live", "details": {"family": "qwen2"}}]}
    ).encode()
    server = _serve_tags(body)
    host = next(server)
    judge = OllamaJudge(model="qwen2.5:3b", temperature=0.9, seed=5, num_predict=4)
    try:
        prov = provenance_for_judge(judge, host=host)
    finally:
        server.close()
    assert prov.digest == "sha256:live"
    assert prov.family == "qwen2"
    assert prov.temperature == 0.9
    assert prov.seed == 5


# ---- live Ollama provenance (skips without a server) -------------------------

@pytest.mark.requires_ollama
def test_fetch_model_provenance_live_returns_non_empty_digest():
    if not OllamaJudge().available():
        pytest.skip("no live Ollama server")
    prov = fetch_model_provenance("qwen2.5:3b", host="http://localhost:11434")
    assert prov is not None
    assert isinstance(prov.digest, str) and prov.digest
    assert isinstance(prov.family, str) and prov.family


# ---- model-aware availability (real local server, no mocks) ------------------

def test_available_true_when_model_present_on_real_server():
    body = json.dumps({"models": [{"name": "qwen2.5:3b"}, {"name": "gemma3:4b"}]}).encode()
    server = _serve_tags(body)
    host = next(server)
    try:
        assert OllamaJudge(model="qwen2.5:3b", base_url=host).available() is True
    finally:
        server.close()


def test_available_false_when_model_absent_on_real_server():
    """Server up but the requested model is not pulled -> available() is False
    (so guarded live tests SKIP rather than 404-FAIL)."""
    body = json.dumps({"models": [{"name": "other:1b"}]}).encode()
    server = _serve_tags(body)
    host = next(server)
    try:
        assert OllamaJudge(model="qwen2.5:3b", base_url=host).available() is False
    finally:
        server.close()


def test_available_false_on_non_200_status():
    server = _serve_tags(b"{}", status=500)
    host = next(server)
    try:
        assert OllamaJudge(model="qwen2.5:3b", base_url=host).available() is False
    finally:
        server.close()


# ---- OllamaJudge transient-timeout retry (real server, no mocks) -------------
import time as _time  # noqa: E402
from http.server import ThreadingHTTPServer  # noqa: E402

import requests as _requests  # noqa: E402


def _serve_generate_with_delays(delays: list[float]):
    """Threaded real server for POST /api/generate; the i-th call sleeps
    ``delays[i]`` seconds then returns ``{"response": "a"}``. A delay exceeding the
    client read timeout forces a real ``requests.ReadTimeout`` on that attempt.
    """
    state = {"n": 0}

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            i = state["n"]
            state["n"] += 1
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            if i < len(delays) and delays[i] > 0:
                _time.sleep(delays[i])
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"response": "a"}).encode())
            except (BrokenPipeError, ConnectionError):
                pass  # client already timed out and closed; that is the point

        def log_message(self, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, thread, f"http://{host}:{port}", state


def test_ollama_judge_retries_transient_timeout_then_succeeds() -> None:
    # First call stalls past the 0.25s read timeout (real ReadTimeout); the retry
    # returns immediately. With retries, judge() must recover and return the label.
    server, thread, url, state = _serve_generate_with_delays([0.6, 0.0])
    try:
        judge = OllamaJudge(model="gemma3:4b", base_url=url, timeout=0.25, max_retries=4)
        persona = PersonaSpec("p1", "expert", "neutral", "left", 0.9)
        label = judge.judge(persona, "2 + 2 = 4")
        assert label == "a"
        assert state["n"] >= 2  # proves at least one retry happened
    finally:
        server.shutdown()
        thread.join(timeout=5.0)
        server.server_close()


def test_ollama_judge_raises_after_exhausting_retries() -> None:
    # Every attempt stalls past the timeout: judge() must FAIL LOUDLY after the
    # retries, never silently fabricate a vote.
    server, thread, url, state = _serve_generate_with_delays([0.6, 0.6])
    try:
        judge = OllamaJudge(model="gemma3:4b", base_url=url, timeout=0.2, max_retries=2)
        persona = PersonaSpec("p1", "expert", "neutral", "left", 0.9)
        with pytest.raises(_requests.RequestException):
            judge.judge(persona, "2 + 2 = 4")
        assert state["n"] == 2  # exactly max_retries attempts were made
    finally:
        server.shutdown()
        thread.join(timeout=5.0)
        server.server_close()
