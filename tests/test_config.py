from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ntqr_allotment.config import (
    DEFAULT_CONFIG,
    load_experiment_profile,
    load_live_cross_family_config,
    load_live_postdoc_panel_config,
)


def _write_config(path: Path, text: str = DEFAULT_CONFIG) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _minimal_config(
    *,
    experiment_updates: dict[str, object] | None = None,
    live_updates: dict[str, object] | None = None,
    include_profiles: bool = True,
    include_live: bool = True,
) -> str:
    experiment: dict[str, object] = {
        "strategies": ["random_selection"],
        "panel_sizes": [3],
        "mean_expertises": [0.72],
        "expertise_heterogeneities": [0.08],
        "bias_stds": [0.3],
        "n_items_values": [90],
        "prevalence_as": [0.5],
        "seeds": [0],
        "n_experts": 32,
        "n_trios": 1,
    }
    if experiment_updates:
        experiment.update(experiment_updates)
    live: dict[str, object] = {
        "models": ["qwen2.5:3b", "gemma3:4b"],
        "replicates_per_model": 2,
        "seeds": [0, 1],
        "n_items": 12,
        "target_accuracy": 0.75,
        "shared_strength": 0.85,
        "base_url": "http://localhost:11434",
        "temperature": 0.4,
        "num_predict": 1,
        "timeout": 7.5,
        "progress_every": 3,
    }
    if live_updates:
        live.update(live_updates)
    data: dict[str, object] = {"default_profile": "smoke"}
    if include_profiles:
        data["experiment_profiles"] = {"smoke": {"experiment": experiment}}
    else:
        data["experiment"] = experiment
    if include_live:
        data["live_cross_family"] = live
    return yaml.safe_dump(data, sort_keys=False)


def test_load_experiment_profile_selects_default_and_hashes(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "config.yaml")

    profile = load_experiment_profile(config)

    assert profile.name == "manuscript_contrast"
    assert profile.grid.n_experts == 96
    assert profile.grid.n_trios == 8
    assert profile.grid.panel_sizes == (3, 6, 9, 12)
    assert profile.grid.mean_expertises == (0.62, 0.68, 0.74, 0.8)
    assert profile.grid.bias_stds == (0.1, 0.2, 0.35, 0.5)
    assert profile.grid.n_items_values == (300,)
    assert profile.grid.seeds == tuple(range(24))
    assert len(profile.config_hash) == 12
    assert profile.metadata()["profile_name"] == "manuscript_contrast"


def test_load_experiment_profile_can_select_smoke(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "config.yaml")

    profile = load_experiment_profile(config, "smoke")

    assert profile.name == "smoke"
    assert profile.grid.seeds == (0, 1, 2)
    assert profile.metadata()["n_seeds"] == 3


def test_broad_and_panel_ladder_profiles_are_available(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "config.yaml")

    contrast = load_experiment_profile(config, "manuscript_contrast")
    panel_ladder = load_experiment_profile(config, "panel_ladder")
    broad = load_experiment_profile(config, "research_broad")

    assert contrast.grid.panel_sizes == (3, 6, 9, 12)
    assert contrast.grid.seeds == tuple(range(24))
    assert panel_ladder.grid.panel_sizes == (3, 4, 5, 6, 9, 12)
    assert broad.grid.prevalence_as == (0.35, 0.5, 0.65)
    assert broad.grid.mean_expertises == (0.64, 0.68, 0.72, 0.78)
    assert broad.grid.bias_stds == (0.1, 0.2, 0.3, 0.4)
    assert broad.metadata()["panel_sizes"] == "3,4,5,6,9"


def test_load_experiment_profile_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_experiment_profile(tmp_path / "missing.yaml")


def test_load_experiment_profile_accepts_legacy_top_level_experiment(
    tmp_path: Path,
) -> None:
    config = _write_config(
        tmp_path / "config.yaml",
        _minimal_config(include_profiles=False, include_live=False),
    )

    profile = load_experiment_profile(config)

    assert profile.name == "smoke"
    assert profile.description == "Legacy top-level experiment block."


def test_load_experiment_profile_rejects_missing_experiment_mapping(
    tmp_path: Path,
) -> None:
    config = _write_config(
        tmp_path / "config.yaml",
        yaml.safe_dump(
            {
                "default_profile": "smoke",
                "experiment_profiles": {"smoke": {"description": "bad"}},
            }
        ),
    )

    with pytest.raises(ValueError, match="missing experiment mapping"):
        load_experiment_profile(config)


def test_load_live_cross_family_config_rejects_missing_and_bad_mapping(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        load_live_cross_family_config(tmp_path / "missing.yaml")
    bad = _write_config(tmp_path / "bad.yaml", "- just\n- a list\n")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_live_cross_family_config(bad)
    missing = _write_config(tmp_path / "missing_live.yaml", "paper: {}\n")
    with pytest.raises(ValueError, match="missing live_cross_family"):
        load_live_cross_family_config(missing)


@pytest.mark.parametrize(
    ("bad_yaml", "message"),
    [
        (
            "\n".join(["default_profile: bad", "experiment_profiles: {}", ""]),
            "unknown experiment profile",
        ),
        (
            "\n".join(
                [
                    "default_profile: smoke",
                    "experiment_profiles:",
                    "  smoke:",
                    "    experiment:",
                    "      strategies: [random_selection]",
                    "      panel_sizes: [2]",
                    "      mean_expertises: [0.72]",
                    "      expertise_heterogeneities: [0.08]",
                    "      bias_stds: [0.3]",
                    "      n_items_values: [90]",
                    "      prevalence_as: [0.5]",
                    "      seeds: [0]",
                    "      n_experts: 32",
                    "      n_trios: 2",
                    "",
                ]
            ),
            "panel_sizes must be >= 3",
        ),
        (
            "\n".join(
                [
                    "default_profile: smoke",
                    "experiment_profiles:",
                    "  smoke:",
                    "    experiment:",
                    "      strategies: [random_selection]",
                    "      panel_sizes: [3]",
                    "      mean_expertises: [1.2]",
                    "      expertise_heterogeneities: [0.08]",
                    "      bias_stds: [0.3]",
                    "      n_items_values: [90]",
                    "      prevalence_as: [0.5]",
                    "      seeds: [0]",
                    "      n_experts: 32",
                    "      n_trios: 2",
                    "",
                ]
            ),
            "mean_expertises must be in",
        ),
        (
            "\n".join(
                [
                    "default_profile: smoke",
                    "experiment_profiles:",
                    "  smoke:",
                    "    experiment:",
                    "      strategies: [random_selection]",
                    "      panel_sizes: [3]",
                    "      mean_expertises: [0.72]",
                    "      expertise_heterogeneities: [0.08]",
                    "      bias_stds: [0.3]",
                    "      n_items_values: [90]",
                    "      prevalence_as: [0.5]",
                    "      seeds: [0, 0]",
                    "      n_experts: 32",
                    "      n_trios: 2",
                    "",
                ]
            ),
            "seeds must be unique",
        ),
        (
            "\n".join(
                [
                    "default_profile: smoke",
                    "experiment_profiles:",
                    "  smoke:",
                    "    experiment:",
                    "      strategies: [random_selection]",
                    "      panel_sizes: [3]",
                    "      mean_expertises: [0.72]",
                    "      expertise_heterogeneities: [0.08]",
                    "      bias_stds: [0.3]",
                    "      n_items_values: [90]",
                    "      prevalence_as: [0.5]",
                    "      seeds: [0]",
                    "      n_experts: 32",
                    "      n_trios: 2",
                    "",
                ]
            ),
            "n_trios exceeds",
        ),
    ],
)
def test_load_experiment_profile_rejects_bad_profiles(
    tmp_path: Path,
    bad_yaml: str,
    message: str,
) -> None:
    config = _write_config(tmp_path / "config.yaml", bad_yaml)

    with pytest.raises(ValueError, match=message):
        load_experiment_profile(config)


@pytest.mark.parametrize(
    ("experiment_updates", "message"),
    [
        ({"strategies": "random_selection"}, "experiment.strategies must be a list"),
        ({"strategies": ["not_a_strategy"]}, "invalid strategies"),
        ({"panel_sizes": [99]}, "panel_size exceeds"),
        ({"expertise_heterogeneities": [-0.1]}, "non-negative"),
        ({"bias_stds": [-0.1]}, "bias_stds"),
        ({"n_items_values": [0]}, "n_items_values"),
        ({"prevalence_as": [1.0]}, "prevalence_as"),
    ],
)
def test_load_experiment_profile_rejects_range_and_axis_errors(
    tmp_path: Path,
    experiment_updates: dict[str, object],
    message: str,
) -> None:
    config = _write_config(
        tmp_path / "config.yaml",
        _minimal_config(experiment_updates=experiment_updates),
    )

    with pytest.raises(ValueError, match=message):
        load_experiment_profile(config)


def test_live_cross_family_settings_are_validated(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "config.yaml",
        "\n".join(
            [
                "default_profile: smoke",
                "experiment_profiles:",
                "  smoke:",
                "    experiment:",
                "      strategies: [random_selection]",
                "      panel_sizes: [3]",
                "      mean_expertises: [0.72]",
                "      expertise_heterogeneities: [0.08]",
                "      bias_stds: [0.3]",
                "      n_items_values: [90]",
                "      prevalence_as: [0.5]",
                "      seeds: [0]",
                "      n_experts: 32",
                "      n_trios: 2",
                "live_cross_family:",
                "  models: []",
                "  seeds: [0]",
                "  n_items: 150",
                "  target_accuracy: 0.75",
                "  shared_strength: 0.85",
                "",
            ]
        ),
    )

    with pytest.raises(ValueError, match="live_cross_family.models"):
        load_experiment_profile(config)


def test_load_live_cross_family_config_reads_runtime_controls(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "config.yaml",
        "\n".join(
            [
                "live_cross_family:",
                "  models: [qwen2.5:3b, gemma3:4b]",
                "  replicates_per_model: 3",
                "  seeds: [0, 1]",
                "  n_items: 12",
                "  target_accuracy: 0.75",
                "  shared_strength: 0.85",
                "  base_url: http://localhost:11434",
                "  temperature: 0.4",
                "  num_predict: 1",
                "  timeout: 7.5",
                "  progress_every: 3",
                "",
            ]
        ),
    )

    live = load_live_cross_family_config(config)

    assert live.models == ("qwen2.5:3b", "gemma3:4b")
    assert live.replicates_per_model == 3
    assert live.expanded_models() == (
        "qwen2.5:3b",
        "qwen2.5:3b",
        "qwen2.5:3b",
        "gemma3:4b",
        "gemma3:4b",
        "gemma3:4b",
    )
    assert live.seeds == (0, 1)
    assert len(live.config_hash) == 12
    assert live.metadata()["n_requested_judges"] == 6
    assert live.n_items == 12
    assert live.temperature == 0.4
    assert live.num_predict == 1
    assert live.timeout == 7.5
    assert live.progress_every == 3
    assert live.require_live is False


def test_load_live_cross_family_config_rejects_bad_runtime_controls(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "config.yaml",
        "\n".join(
            [
                "live_cross_family:",
                "  models: [qwen2.5:3b, gemma3:4b]",
                "  seeds: [0, 1]",
                "  n_items: 12",
                "  target_accuracy: 0.75",
                "  shared_strength: 0.85",
                "  replicates_per_model: 0",
                "  num_predict: 0",
                "",
            ]
        ),
    )

    with pytest.raises(ValueError, match="live_cross_family.replicates_per_model"):
        load_live_cross_family_config(config)


@pytest.mark.parametrize(
    ("live_updates", "message"),
    [
        ({"seeds": [0, 0]}, "seeds"),
        ({"n_items": 0}, "n_items"),
        ({"target_accuracy": 1.0}, "target_accuracy"),
        ({"shared_strength": 1.5}, "shared_strength"),
        ({"base_url": ""}, "base_url"),
        ({"temperature": -0.1}, "temperature"),
        ({"timeout": 0.0}, "timeout"),
        ({"progress_every": 0}, "progress_every"),
        ({"require_live": "yes"}, "require_live"),
    ],
)
def test_load_live_cross_family_config_rejects_each_runtime_guard(
    tmp_path: Path,
    live_updates: dict[str, object],
    message: str,
) -> None:
    config = _write_config(
        tmp_path / "config.yaml",
        _minimal_config(include_profiles=False, live_updates=live_updates),
    )

    with pytest.raises(ValueError, match=message):
        load_live_cross_family_config(config)


def test_load_live_postdoc_panel_config_reads_gemma_panel_defaults(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "config.yaml")

    postdoc = load_live_postdoc_panel_config(config)

    assert postdoc.model == "gemma3:4b"
    assert postdoc.seeds == (0, 1, 2, 3, 4, 5)
    assert postdoc.n_reviewers == 48
    assert postdoc.n_applications == 72
    assert postdoc.panel_sizes == (3, 6)
    assert postdoc.strategies == (
        "representative_sortition",
        "random_selection",
        "ideological_selection",
        "expertise_threshold",
    )
    assert postdoc.temperature == 0.2
    assert postdoc.num_predict == 1
    assert postdoc.require_live is True
    assert len(postdoc.config_hash) == 12
    assert postdoc.metadata()["cache_path"] == "output/data/postdoc_votes/gemma3_4b_votes.json"


@pytest.mark.parametrize(
    ("postdoc_updates", "message"),
    [
        ({"model": "qwen2.5:3b"}, "gemma3:4b"),
        ({"seeds": [0, 0]}, "seeds"),
        ({"panel_sizes": [2]}, "panel_sizes"),
        ({"strategies": ["bad"]}, "invalid strategy"),
        ({"n_reviewers": 2}, "n_reviewers"),
        ({"n_applications": 0}, "n_applications"),
        ({"n_trios": 99}, "n_trios"),
        ({"prevalence_strong": 1.0}, "prevalence"),
        ({"age_min": 50, "age_max": 30}, "age_min"),
        ({"mean_expertise": 1.1}, "mean_expertise"),
        ({"age_bias_std": -0.1}, "age_bias_std"),
        ({"num_predict": 0}, "num_predict"),
        ({"require_live": "yes"}, "require_live"),
        ({"cache_path": ""}, "cache_path"),
    ],
)
def test_load_live_postdoc_panel_config_rejects_bad_runtime_controls(
    tmp_path: Path,
    postdoc_updates: dict[str, object],
    message: str,
) -> None:
    postdoc: dict[str, object] = {
        "model": "gemma3:4b",
        "seeds": [0, 1],
        "n_reviewers": 12,
        "n_applications": 18,
        "panel_sizes": [3, 6],
        "strategies": ["representative_sortition", "ideological_selection"],
        "n_trios": 4,
        "require_live": True,
        "cache_path": "output/data/postdoc_votes/test.json",
    }
    postdoc.update(postdoc_updates)
    config = _write_config(
        tmp_path / "config.yaml",
        yaml.safe_dump({"live_postdoc_panel": postdoc}, sort_keys=False),
    )

    with pytest.raises(ValueError, match=message):
        load_live_postdoc_panel_config(config)
