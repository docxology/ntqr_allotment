"""Exercise persona CLI configuration and real, explicitly synthetic offline votes."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ntqr_allotment.corpus import make_arithmetic_corpus
from ntqr_allotment.experts import generate_population
from ntqr_allotment.personas import DeterministicJudge, judge_corpus, personas_from_population
from scripts.run_personas import _live_judges

_ROOT = Path(__file__).resolve().parents[1]


def _run_personas_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "run_personas.py"), *args],
        cwd=_ROOT,
        env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONPATH": str(_ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize("model", ["qwen2.5:3b", "qwen3:8b"])
def test_requested_model_reaches_both_replacement_slots_without_changing_protocol(model: str) -> None:
    judges = _live_judges(model, seed=17)
    assert [judge.model for judge in judges] == [model, "gemma3:4b", model]
    assert [judge.temperature for judge in judges] == [0.6, 0.6, 1.0]
    assert [judge.seed for judge in judges] == [18, 19, 20]


def test_offline_cli_reports_the_distinct_seeds_actually_used_for_votes(tmp_path: Path) -> None:
    destination = tmp_path / "personas"
    result = _run_personas_cli(
        "--offline",
        "--output-dir", str(destination),
        "--n-experts", "9",
        "--n-items", "24",
        "--seed", "17",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads((destination / "persona_results.json").read_text(encoding="utf-8"))
    assert payload["live_ollama"] is False
    assert payload["judge_seeds"] == [18, 19, 20]
    assert payload["judge_models"] == [
        "deterministic-offline@seed=18",
        "deterministic-offline@seed=19",
        "deterministic-offline@seed=20",
    ]
    corpus = make_arithmetic_corpus(24, seed=17, max_operand=9999, max_error=5)
    personas = personas_from_population(generate_population(9, seed=17, mean_expertise=0.78))
    by_id = {persona.id: persona for persona in personas}
    for index, (persona_id, seed) in enumerate(zip(payload["persona_ids"], payload["judge_seeds"])):
        actual_caches = list(destination.glob(f"persona_votes_j{index}_*.json"))
        assert len(actual_caches) == 1
        reference_cache = tmp_path / "reference" / f"judge-{index}.json"
        judge_corpus(
            DeterministicJudge(corpus, seed=seed),
            by_id[persona_id],
            corpus,
            cache_path=reference_cache,
        )
        assert json.loads(actual_caches[0].read_text(encoding="utf-8")) == json.loads(
            reference_cache.read_text(encoding="utf-8")
        )


def test_offline_cli_refuses_recorded_live_destination() -> None:
    result = _run_personas_cli("--offline", "--output-dir", str(_ROOT / "output" / "data"))
    assert result.returncode != 0
    assert "separate from recorded live results" in result.stderr
