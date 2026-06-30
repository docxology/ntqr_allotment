from __future__ import annotations

import json
from pathlib import Path

from ntqr_allotment.cover import render_cover


def test_render_cover_is_deterministic_and_writes_manifest(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    first_cover = tmp_path / "first.png"
    first_manifest = tmp_path / "first.json"
    second_cover = tmp_path / "second.png"
    second_manifest = tmp_path / "second.json"

    render_cover(repo_root, output_path=first_cover, manifest_path=first_manifest)
    render_cover(repo_root, output_path=second_cover, manifest_path=second_manifest)

    assert first_cover.read_bytes() == second_cover.read_bytes()
    assert first_cover.stat().st_size > 50_000

    payload = json.loads(first_manifest.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["source_config"] == "manuscript/config.yaml"
    assert len(payload["source_config_hash"]) == 12
    assert payload["semantic_role"].startswith("schematic front-matter cover")
    assert "not an empirical result" in payload["caveat"]
    assert "not counted as a manuscript data figure" in payload["caveat"]
    assert "/Users/" not in payload["concept_references"]["large_page"]
    assert "/Users/" not in payload["concept_references"]["mobile_portrait"]
    assert "not part of the regeneration pipeline" in payload["concept_references"]["large_page"]
    assert "not part of the regeneration pipeline" in payload["concept_references"]["mobile_portrait"]
    assert payload["profile_name"] == "manuscript_contrast"
    assert payload["displayed_case_axes"] == {
        "panel_sizes": [3, 6, 9, 12],
        "bias_stds": [0.1, 0.2, 0.35, 0.5],
    }
    assert {"n=3 | bias 0.10", "n=12 | bias 0.50"} <= {
        case["label"] for case in payload["displayed_cases"]
    }
    assert "panel-size x bias case grid" in payload["visual_contract"]["reading_order"]
