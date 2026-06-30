from __future__ import annotations

from scripts.run_cross_family import _offline_judge_provenance
from scripts.run_cross_family_multiseed import _offline_judge_provenance as _offline_ms


def _assert_provenance_contract(records: list[dict[str, object]]) -> None:
    assert records
    for record in records:
        assert isinstance(record["judge_id"], str)
        assert isinstance(record["model"], str)
        assert isinstance(record["family"], str)
        assert "digest" in record
        assert isinstance(record["temperature"], float)
        assert isinstance(record["seed"], int)
        assert isinstance(record["num_predict"], int)


def test_cross_family_offline_artifact_provenance_contract() -> None:
    records = _offline_judge_provenance(["qwen2.5:3b", "gemma3:4b"])

    _assert_provenance_contract(records)
    assert len(records) == 4
    assert {record["family"] for record in records} == {"qwen2", "gemma3"}


def test_cross_family_multiseed_uses_same_provenance_contract() -> None:
    records = _offline_ms(["qwen2.5:3b", "gemma3:4b"])

    _assert_provenance_contract(records)
    assert len(records) == 4


def test_offline_provenance_honors_replicates_per_model() -> None:
    records = _offline_judge_provenance(
        ["qwen2.5:3b", "gemma3:4b"], replicates_per_model=3
    )

    _assert_provenance_contract(records)
    assert len(records) == 6
    assert sum(1 for record in records if record["family"] == "qwen2") == 3
    assert sum(1 for record in records if record["family"] == "gemma3") == 3
