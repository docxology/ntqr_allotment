"""Use real cached synthetic judges to verify provenance and atomic checkpoints."""

from dataclasses import replace
import json
import math

import pytest

from ntqr_allotment.artifacts import write_json
from ntqr_allotment.corpus import make_arithmetic_corpus
from ntqr_allotment.experts import generate_population
from ntqr_allotment.personas import (
    DeterministicJudge,
    OllamaJudge,
    PersonaSpec,
    judge_corpus,
    personas_from_population,
    run_persona_trio,
)


def test_failed_json_encoding_preserves_existing_file(tmp_path):
    path = tmp_path / "cache.json"
    write_json(path, {"previous": "α"})
    path.chmod(0o640)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        write_json(path, {"bad": math.nan})
    assert path.read_bytes() == before
    write_json(path, {"updated": True})
    assert path.stat().st_mode & 0o777 == 0o640
    assert list(tmp_path.iterdir()) == [path]


def test_vote_cache_is_bound_to_judge_seed_and_persona_content(tmp_path):
    items = make_arithmetic_corpus(40, seed=8)
    persona = PersonaSpec("p", "competent", "neutral", "center", 0.7)
    path = tmp_path / "cache.json"
    first = judge_corpus(
        DeterministicJudge(items, seed=1), persona, items, cache_path=path
    )
    second_judge = DeterministicJudge(items, seed=2)
    second = judge_corpus(second_judge, persona, items, cache_path=path)
    assert second == judge_corpus(second_judge, persona, items)
    assert first != second
    assert len(json.loads(path.read_text())) == 80
    changed = replace(persona, target_accuracy=0.95)
    assert judge_corpus(second_judge, changed, items, cache_path=path) == judge_corpus(
        second_judge, changed, items
    )
    assert len(json.loads(path.read_text())) == 120


def test_legacy_index_only_keys_are_preserved_without_being_trusted(tmp_path):
    items = make_arithmetic_corpus(4, seed=8)
    persona = PersonaSpec("p", "expert", "neutral", "center", 0.99)
    path = tmp_path / "cache.json"
    old = {f"p|{item.index}": "b" for item in items}
    write_json(path, old)
    judge = DeterministicJudge(items, seed=1)
    assert judge_corpus(judge, persona, items, cache_path=path) == judge_corpus(
        judge, persona, items
    )
    stored = json.loads(path.read_text())
    assert all(stored[key] == value for key, value in old.items())
    assert sum(key.startswith("v2:") for key in stored) == 4


def test_cached_ollama_judge_requires_immutable_digest(tmp_path):
    items = make_arithmetic_corpus(1, seed=8)
    persona = PersonaSpec("p", "expert", "neutral", "center", 0.99)
    path = tmp_path / "live-cache.json"
    judge = OllamaJudge(base_url="http://localhost:6553")
    with pytest.raises(ValueError, match="immutable model digest"):
        judge_corpus(judge, persona, items, cache_path=path)
    assert not path.exists()


def test_run_persona_trio_forwards_digest_cache_namespace(tmp_path):
    items = make_arithmetic_corpus(300, seed=7)
    population = generate_population(
        10, seed=7, mean_expertise=0.8, expertise_heterogeneity=0.05
    )
    personas = personas_from_population(population)
    judge = DeterministicJudge(items, seed=7)
    path = tmp_path / "trio-cache.json"

    run_persona_trio(
        judge, personas, items, cache_path=path, cache_namespace="sha256:first"
    )
    assert len(json.loads(path.read_text())) == 3 * len(items)
    run_persona_trio(
        judge, personas, items, cache_path=path, cache_namespace="sha256:second"
    )
    assert len(json.loads(path.read_text())) == 6 * len(items)
