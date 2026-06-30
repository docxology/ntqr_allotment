from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from ntqr_allotment.config import (
    load_experiment_profile,
    load_live_postdoc_panel_config,
)


def test_sweep_artifacts_match_active_config_profile() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    profile = load_experiment_profile(repo_root / "manuscript" / "config.yaml")
    sweep_json = repo_root / "output" / "data" / "sweep_results.json"
    sweep_csv = repo_root / "output" / "data" / "sweep_aggregated.csv"

    payload = json.loads(sweep_json.read_text(encoding="utf-8"))
    metadata = payload["metadata"]

    assert payload["schema_version"] == 2
    assert metadata["profile_name"] == profile.name
    assert metadata["config_hash"] == profile.config_hash
    assert metadata["seed_list"] == list(profile.grid.seeds)
    assert metadata["row_count"] == len(payload["rows"])

    with sweep_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert {row["profile_name"] for row in rows} == {profile.name}
    assert {row["config_hash"] for row in rows} == {profile.config_hash}


def test_cross_family_artifacts_carry_current_run_metadata() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    single = json.loads(
        (repo_root / "output/data/cross_family_results.json").read_text(
            encoding="utf-8"
        )
    )
    multiseed = json.loads(
        (repo_root / "output/data/cross_family_multiseed.json").read_text(
            encoding="utf-8"
        )
    )

    assert single["schema_version"] == 2
    assert multiseed["schema_version"] == 2
    for payload in (single, multiseed):
        assert len(payload["config_hash"]) == 12
        assert len(payload["source_config_hash"]) == 12
        assert payload["live_ollama"] is True
        assert payload.get("require_live") is True
        assert payload["replicates_per_model"] >= 2
        assert payload["n_items"] > 0
        assert payload["num_predict"] > 0
        assert payload["judge_provenance"]

    assert single["run_count"] == 1
    assert single["total_pairs"] == len(single["pair_abs_corr"])
    assert single["nonzero_pairs"] <= single["total_pairs"]
    assert single["pair_records"]
    assert single["pair_group_summaries"]
    assert multiseed["run_count"] == multiseed["n_runs"]
    assert multiseed["total_pairs_per_run"] >= (
        multiseed["same_pairs_per_run"] + multiseed["cross_pairs_per_run"]
    )
    assert len(multiseed["nonzero_pairs_per_run"]) == multiseed["n_runs"]


def test_postdoc_panel_artifact_carries_live_gemma_metadata() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (repo_root / "output/data/postdoc_panel_results.json").read_text(
            encoding="utf-8"
        )
    )
    alignment = json.loads(
        (repo_root / "output/data/postdoc_panel_alignment.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["schema_version"] == 1
    assert payload["model"] == "gemma3:4b"
    assert payload["live_ollama"] is True
    assert payload["tracks"]["live"]["live_ollama"] is True
    assert payload["tracks"]["live"]["model_provenance"]["digest"]
    assert payload["tracks"]["live"]["decode_params"]["num_predict"] == 1
    assert payload["tracks"]["live"]["vote_cache_provenance"]["entries"] > 0
    assert len(payload["config_hash"]) == 12
    assert payload["config"]["model"] == "gemma3:4b"
    assert payload["config"]["require_live"] is True
    # Bind the artifact's sampling shape to the live config so it tracks the
    # configured sample size (e.g. the 12-seed run) instead of a stale literal.
    cfg = load_live_postdoc_panel_config(repo_root / "manuscript" / "config.yaml")
    assert payload["config"]["seed_list"] == list(cfg.seeds)
    assert payload["config"]["n_reviewers"] == cfg.n_reviewers
    assert payload["config"]["n_applications"] == cfg.n_applications
    assert payload["config"]["panel_sizes"] == list(cfg.panel_sizes)
    assert payload["tracks"]["live"]["seed_list"] == list(cfg.seeds)
    assert payload["tracks"]["live"]["n_reviewers"] == cfg.n_reviewers
    assert payload["tracks"]["live"]["n_applications"] == cfg.n_applications
    assert "qwen" not in json.dumps(payload).lower()
    assert alignment["schema_version"] == 1
    assert alignment["model"] == "gemma3:4b"
    assert alignment["cells"]


def test_front_matter_metadata_is_template_readable() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (repo_root / "manuscript/config.yaml").read_text(encoding="utf-8")
    )

    assert "authors" in config
    assert "authors" not in config["paper"]
    author = config["authors"][0]
    assert author["name"] == "Daniel Ari Friedman"
    assert author["affiliation"] == "Active Inference Institute"
    assert author["orcid"] == "0000-0001-6232-9096"
    assert author["corresponding"] is True
    assert config["paper"]["cover"]["image"] == "figures/ntqr_cover.png"
    assert config["paper"]["title"] == "Sortition Upstream of NTQR"
    assert (
        config["paper"]["subtitle"]
        == "How Panel Formation and Size Shape Ground-Truth-Free Evaluation"
    )

    publication = config["publication"]
    assert "doi_status" not in publication
    assert publication.get("doi", "").startswith("10.5281/zenodo.")
    assert publication.get("version_doi", "").startswith("10.5281/zenodo.")
    assert publication.get("version_record", "").startswith("https://zenodo.org/records/")


def test_cover_manifest_contract_and_data_figure_boundary() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / "output/data/cover_manifest.json"
    cover_path = repo_root / "output/figures/ntqr_cover.png"

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert cover_path.is_file()
    assert cover_path.stat().st_size > 50_000
    assert payload["schema_version"] == 1
    assert payload["cover_image"] == "output/figures/ntqr_cover.png"
    assert payload["manifest"] == "output/data/cover_manifest.json"
    assert payload["source_config"] == "manuscript/config.yaml"
    assert len(payload["source_config_hash"]) == 12
    assert "large_page" in payload["concept_references"]
    assert "mobile_portrait" in payload["concept_references"]
    assert "not an empirical result" in payload["caveat"]
    assert "not counted as a manuscript data figure" in payload["caveat"]


def test_analytical_prediction_artifact_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (repo_root / "output/data/analytical_predictions.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["schema_version"] == 1
    assert payload["metadata"]["profile_name"] == "manuscript_contrast"
    assert payload["rep_vs_ideo_cells"]
    assert payload["pre_post_cells"]
    assert payload["monotone_checks"]
    first = payload["rep_vs_ideo_cells"][0]
    assert {
        "panel_size",
        "mean_expertise",
        "bias_std",
        "effect",
        "ci95",
        "prediction_status",
    } <= set(first)
    assert {cell["panel_size"] for cell in payload["rep_vs_ideo_cells"]} == {3, 6, 9, 12}
    assert {cell["bias_std"] for cell in payload["rep_vs_ideo_cells"]} == {
        0.1,
        0.2,
        0.35,
        0.5,
    }
