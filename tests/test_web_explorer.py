from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ntqr_allotment.web_explorer import (
    build_explorer_model,
    render_explorer_html,
    write_explorer,
)


def _write_fixture(root: Path) -> None:
    data = root / "output" / "data"
    figures = root / "output" / "figures"
    manuscript = root / "docs" / "manuscript"
    data.mkdir(parents=True)
    figures.mkdir(parents=True)
    manuscript.mkdir(parents=True)
    (figures / "strategy_ranking.png").write_bytes(b"png")
    (figures / "power_curve.png").write_bytes(b"png")
    (figures / "cross_family_contrast.png").write_bytes(b"png")
    (figures / "postdoc_strategy_ranking.png").write_bytes(b"png")
    (figures / "postdoc_age_bias_heatmap.png").write_bytes(b"png")
    (figures / "postdoc_empirical_alignment.png").write_bytes(b"png")
    (figures / "fairness_maximin.png").write_bytes(b"png")
    (figures / "ntqr_cover.png").write_bytes(b"cover")
    (manuscript / "config.yaml").write_text(
        """
paper:
  title: Fixture NTQR title
  subtitle: Fixture subtitle
  date: "2026-06-25"
authors:
  - name: Daniel Ari Friedman
    affiliation: Active Inference Institute
    email: danielarifriedman@gmail.com
    orcid: 0000-0001-6232-9096
    corresponding: true
publication:
  doi_status: forthcoming
""",
        encoding="utf-8",
    )
    (data / "cover_manifest.json").write_text(
        json.dumps(
            {
                "cover_image": "output/figures/ntqr_cover.png",
                "manifest": "output/data/cover_manifest.json",
                "semantic_role": "schematic front-matter cover",
                "caveat": "Deterministic schematic only; not an empirical result.",
            }
        ),
        encoding="utf-8",
    )
    (data / "sweep_results.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "profile_name": "smoke",
                    "config_hash": "abc123",
                },
                "rows": [
                    {
                        "strategy": "random_selection",
                        "eie_error": 0.1,
                        "nested": {"source": "fixture"},
                    },
                    ["not", "a", "mapping"],
                ],
            }
        ),
        encoding="utf-8",
    )
    (data / "sweep_aggregated.csv").write_text(
        "strategy,panel_size,eie_mean\nrandom_selection,3,0.1\n",
        encoding="utf-8",
    )
    (data / "power_analysis.csv").write_text(
        "contrast,observed_d,mde_80\nx_vs_y,0.4,0.7\n",
        encoding="utf-8",
    )
    (data / "analytical_predictions.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rep_vs_ideo_cells": [
                    {
                        "panel_size": 3,
                        "mean_expertise": 0.72,
                        "bias_std": 0.3,
                        "effect": 0.1,
                        "ci95": 0.02,
                        "prediction_status": "aligned_resolved",
                    }
                ],
                "pre_post_cells": [
                    {
                        "strategy": "representative_sortition",
                        "panel_size": 3,
                        "mean_expertise": 0.72,
                        "bias_std": 0.3,
                        "eie_minus_mv": -0.02,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (data / "postdoc_panel_results.json").write_text(
        json.dumps(
            {
                "config_hash": "pd123",
                "model": "gemma3:4b",
                "live_ollama": True,
                "config": {"seed_list": [0, 1]},
                "aggregates": [
                    {
                        "track": "live",
                        "strategy": "representative_sortition",
                        "panel_size": 3,
                        "eie_mean": 0.12,
                        "age_disparity_mean": 0.01,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (data / "postdoc_panel_alignment.json").write_text(
        json.dumps(
            {
                "agreement_rate_resolved": 0.5,
                "cells": [
                    {
                        "strategy": "representative_sortition",
                        "panel_size": 3,
                        "sign_agrees": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_build_explorer_model_exposes_source_tables_and_figures(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    model = build_explorer_model(tmp_path)

    assert model.metadata["sweep_profile"] == "smoke"
    assert model.metadata["postdoc_config_hash"] == "pd123"
    assert model.metadata["postdoc_model"] == "gemma3:4b"
    assert model.metadata["postdoc_live"] is True
    assert model.metadata["postdoc_runs"] == 2
    assert model.metadata["author_orcid"] == "0000-0001-6232-9096"
    assert model.metadata["paper_subtitle"] == "Fixture subtitle"
    assert model.metadata["analytical_prediction_schema"] == 1
    assert model.metadata["publication_doi"] == "forthcoming"
    assert model.metadata["cover_image"] == "output/figures/ntqr_cover.png"
    assert model.metadata["cover_manifest"] == "output/data/cover_manifest.json"
    assert model.metadata["figure_count"] == 6
    assert {dataset.dataset_id for dataset in model.datasets} == {
        "sweep_rows",
        "sweep_aggregates",
        "power_diagnostics",
        "analytical_predictions",
        "pre_post_cells",
        "postdoc_panel_aggregates",
        "postdoc_alignment",
    }
    assert {figure.filename for figure in model.figures} == {
        "fairness_maximin.png",
        "postdoc_age_bias_heatmap.png",
        "postdoc_empirical_alignment.png",
        "postdoc_strategy_ranking.png",
        "power_curve.png",
        "strategy_ranking.png",
    }
    strategy_figure = next(
        figure for figure in model.figures if figure.filename == "strategy_ranking.png"
    )
    postdoc_figure = next(
        figure for figure in model.figures if figure.filename == "postdoc_strategy_ranking.png"
    )
    assert strategy_figure.category == "Strategy ranking"
    assert strategy_figure.tier == "generated local artifact"
    assert postdoc_figure.category == "Live companion"
    assert postdoc_figure.tier == "single-model live companion"
    sweep_rows = next(dataset for dataset in model.datasets if dataset.dataset_id == "sweep_rows")
    assert sweep_rows.rows[0]["nested"] == '{"source": "fixture"}'


def test_render_explorer_html_has_filter_contract_and_no_raw_tokens(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)

    html = render_explorer_html(build_explorer_model(tmp_path))

    assert "NTQR Local Artifact Explorer" in html
    assert "dataset-select" in html
    assert "table-filter" in html
    assert "figure-category-select" in html
    assert "figure-search" in html
    assert "Full-size PNG" in html
    assert 'data-category="Strategy ranking"' in html
    assert "single-model live companion" in html
    assert "Manuscript front matter" in html
    assert "Fixture NTQR title" in html
    assert "Fixture subtitle" in html
    assert "ORCID: 0000-0001-6232-9096" in html
    assert "output/data/cover_manifest.json" in html
    assert "output/data/sweep_results.json" in html
    assert "Local explorer only" in html
    assert "&quot;datasets&quot;" not in html
    assert "{{" not in html

    state = html.split('<script id="explorer-state" type="application/json">', 1)[1]
    state = state.split("</script>", 1)[0]
    assert json.loads(state)["metadata"]["paper_title"] == "Fixture NTQR title"


def test_write_explorer_creates_local_web_file(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    out = write_explorer(tmp_path)

    assert out == tmp_path / "output" / "web" / "ntqr_explorer.html"
    assert out.exists()
    assert "strategy_ranking.png" in out.read_text(encoding="utf-8")


def test_build_explorer_model_rejects_non_object_json(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    (tmp_path / "output" / "data" / "sweep_results.json").write_text(
        "[]",
        encoding="utf-8",
    )

    try:
        build_explorer_model(tmp_path)
    except ValueError as exc:
        assert "JSON object" in str(exc)
    else:
        raise AssertionError("expected non-object JSON to fail")


def test_render_explorer_html_treats_stored_markup_as_literal_text(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)
    data = tmp_path / "output" / "data"
    json_key = "<img src=x onerror=alert('json-key')>"
    csv_header = "<svg onload=alert('csv-header')>"
    dataset_title = "<img src=x onerror=alert('title')>"
    dataset_source = "</script><script>alert('source')</script>"

    sweep_path = data / "sweep_results.json"
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    sweep["rows"] = [{json_key: "literal JSON header"}]
    sweep_path.write_text(json.dumps(sweep), encoding="utf-8")
    (data / "sweep_aggregated.csv").write_text(
        f"{csv_header},value\nliteral CSV header,1\n",
        encoding="utf-8",
    )

    model = build_explorer_model(tmp_path)
    datasets = list(model.datasets)
    datasets[0] = replace(
        datasets[0],
        title=dataset_title,
        source=dataset_source,
    )
    rendered = render_explorer_html(replace(model, datasets=tuple(datasets)))

    assert ".innerHTML" not in rendered
    assert ".outerHTML" not in rendered
    assert "title.textContent = dataset.title;" in rendered
    assert "sourceCode.textContent = dataset.source;" in rendered
    assert "header.textContent = column;" in rendered
    before_state = rendered.split('<script id="explorer-state"', 1)[0]
    assert dataset_title not in before_state
    assert dataset_source not in rendered

    state = rendered.split('<script id="explorer-state" type="application/json">', 1)[1]
    state = state.split("</script>", 1)[0]
    state_datasets = json.loads(state)["datasets"]
    sweep_dataset = next(item for item in state_datasets if item["dataset_id"] == "sweep_rows")
    csv_dataset = next(
        item for item in state_datasets if item["dataset_id"] == "sweep_aggregates"
    )
    assert sweep_dataset["title"] == dataset_title
    assert sweep_dataset["source"] == dataset_source
    assert sweep_dataset["rows"] == [{json_key: "literal JSON header"}]
    assert csv_dataset["rows"] == [{csv_header: "literal CSV header", "value": "1"}]


def test_build_explorer_model_marks_small_json_and_csv_not_truncated(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)

    model = build_explorer_model(tmp_path)

    json_dataset = next(
        dataset for dataset in model.datasets if dataset.dataset_id == "postdoc_alignment"
    )
    csv_dataset = next(
        dataset for dataset in model.datasets if dataset.dataset_id == "power_diagnostics"
    )
    for dataset in (json_dataset, csv_dataset):
        assert len(dataset.rows) == 1
        assert dataset.source_row_count == 1
        assert dataset.truncated is False
        assert dataset.row_limit == 500


def test_build_explorer_model_retains_json_and_csv_truncation_metadata(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)
    data = tmp_path / "output" / "data"
    sweep_path = data / "sweep_results.json"
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    sweep["rows"] = [{"index": index} for index in range(501)]
    sweep_path.write_text(json.dumps(sweep), encoding="utf-8")
    (data / "sweep_aggregated.csv").write_text(
        "index\n" + "".join(f"{index}\n" for index in range(501)),
        encoding="utf-8",
    )

    model = build_explorer_model(tmp_path)

    for dataset_id in ("sweep_rows", "sweep_aggregates"):
        dataset = next(item for item in model.datasets if item.dataset_id == dataset_id)
        assert len(dataset.rows) == 500
        assert dataset.source_row_count == 501
        assert dataset.truncated is True
        assert dataset.row_limit == 500

    rendered = render_explorer_html(model)
    assert "showing first ${dataset.rows.length} of ${dataset.source_row_count} rows" in rendered
    state = rendered.split('<script id="explorer-state" type="application/json">', 1)[1]
    state = state.split("</script>", 1)[0]
    state_datasets = json.loads(state)["datasets"]
    for dataset_id in ("sweep_rows", "sweep_aggregates"):
        dataset = next(item for item in state_datasets if item["dataset_id"] == dataset_id)
        assert dataset["source_row_count"] == 501
        assert dataset["truncated"] is True
        assert dataset["row_limit"] == 500


@pytest.mark.parametrize(
    "bad_number",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
    ],
)
def test_write_explorer_rejects_nonfinite_numbers_without_writing_html(
    tmp_path: Path,
    bad_number: float,
) -> None:
    _write_fixture(tmp_path)
    sweep_path = tmp_path / "output" / "data" / "sweep_results.json"
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    sweep["rows"][0]["eie_error"] = bad_number
    sweep_path.write_text(json.dumps(sweep), encoding="utf-8")
    output_path = tmp_path / "output" / "web" / "ntqr_explorer.html"

    with pytest.raises(ValueError, match="embedded JSON.*NaN and Infinity"):
        write_explorer(tmp_path, output_path)

    assert not output_path.exists()


def test_write_explorer_rejects_nested_nonfinite_numbers_without_writing_html(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)
    sweep_path = tmp_path / "output" / "data" / "sweep_results.json"
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    sweep["rows"][0]["nested"] = {"bad": float("nan")}
    sweep_path.write_text(json.dumps(sweep), encoding="utf-8")
    output_path = tmp_path / "output" / "web" / "ntqr_explorer.html"

    with pytest.raises(ValueError, match="row field 'nested'.*NaN and Infinity"):
        write_explorer(tmp_path, output_path)

    assert not output_path.exists()
