from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .sortition import STRATEGIES
from .sweeps import SweepGrid

DEFAULT_PROFILE_NAME = "smoke"

DEFAULT_CONFIG = """paper:
  title: "Sortition Upstream of NTQR"
  subtitle: "How Panel Formation and Size Shape Ground-Truth-Free Evaluation"
  version: "0.1.0"
  date: "2026-06-25"
  cover:
    image: "figures/ntqr_cover.png"
    title: "Sortition-to-NTQR schematic cover"
    description: "Deterministic schematic cover showing the pipeline from audited lottery through trios to NTQR agreement diagnostics and strategy ranking."

authors:
  - name: "Daniel Ari Friedman"
    email: "danielarifriedman@gmail.com"
    affiliation: "Active Inference Institute"
    orcid: "0000-0001-6232-9096"
    corresponding: true

publication:
  doi_status: "forthcoming"
  year: "2026"

default_profile: manuscript_contrast

experiment_profiles:
  smoke:
    description: "Fast deterministic CI/profile smoke grid matching the original manuscript spine."
    experiment:
      strategies: [representative_sortition, random_selection, ideological_selection, expertise_threshold]
      panel_sizes: [3, 6]
      mean_expertises: [0.72]
      expertise_heterogeneities: [0.08]
      bias_stds: [0.3]
      n_items_values: [150]
      prevalence_as: [0.5]
      seeds: [0, 1, 2]
      n_experts: 48
      n_trios: 4
  manuscript_main:
    description: "Reported deterministic grid: strategy x size x competence/bias/corpus-size sensitivity."
    experiment:
      strategies: [representative_sortition, random_selection, ideological_selection, expertise_threshold]
      panel_sizes: [3, 6]
      mean_expertises: [0.68, 0.72]
      expertise_heterogeneities: [0.08]
      bias_stds: [0.2, 0.3]
      n_items_values: [120, 180]
      prevalence_as: [0.5]
      seeds: [0, 1, 2, 3, 4]
      n_experts: 48
      n_trios: 4
  manuscript_contrast:
    description: "Reported contrast grid: strategy x panel-size x expertise x bias sensitivity with one corpus-size/prevalence baseline."
    experiment:
      strategies: [representative_sortition, random_selection, ideological_selection, expertise_threshold]
      panel_sizes: [3, 6, 9, 12]
      mean_expertises: [0.62, 0.68, 0.74, 0.80]
      expertise_heterogeneities: [0.08]
      bias_stds: [0.1, 0.2, 0.35, 0.5]
      n_items_values: [300]
      prevalence_as: [0.5]
      seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
      n_experts: 96
      n_trios: 8
  tolerance:
    description: "Fixed-trio error-correlation tolerance sweep settings."
    experiment:
      strategies: [representative_sortition, ideological_selection]
      panel_sizes: [3]
      mean_expertises: [0.72]
      expertise_heterogeneities: [0.08]
      bias_stds: [0.3]
      n_items_values: [120]
      prevalence_as: [0.5]
      seeds: [0, 1, 2, 3, 4, 5]
      n_experts: 24
      n_trios: 1
  power:
    description: "Design-budget profile used to size prospective seed ladders."
    experiment:
      strategies: [representative_sortition, random_selection, ideological_selection, expertise_threshold]
      panel_sizes: [3, 6]
      mean_expertises: [0.72]
      expertise_heterogeneities: [0.08]
      bias_stds: [0.3]
      n_items_values: [150]
      prevalence_as: [0.5]
      seeds: [0, 1, 2, 3, 4, 5, 6, 7]
      n_experts: 48
      n_trios: 4
  panel_ladder:
    description: "Panel-size sensitivity ladder; research profile, not CI default."
    experiment:
      strategies: [representative_sortition, random_selection, ideological_selection, expertise_threshold]
      panel_sizes: [3, 4, 5, 6, 9, 12]
      mean_expertises: [0.72]
      expertise_heterogeneities: [0.08]
      bias_stds: [0.3]
      n_items_values: [150]
      prevalence_as: [0.5]
      seeds: [0, 1, 2, 3, 4]
      n_experts: 72
      n_trios: 8
  research_broad:
    description: "Broader sensitivity profile spanning prevalence, expertise, bias, heterogeneity, corpus size, and panel size."
    experiment:
      strategies: [representative_sortition, random_selection, ideological_selection, expertise_threshold]
      panel_sizes: [3, 4, 5, 6, 9]
      mean_expertises: [0.64, 0.68, 0.72, 0.78]
      expertise_heterogeneities: [0.04, 0.08, 0.12]
      bias_stds: [0.1, 0.2, 0.3, 0.4]
      n_items_values: [90, 120, 180, 240]
      prevalence_as: [0.35, 0.5, 0.65]
      seeds: [0, 1, 2, 3, 4, 5, 6, 7]
      n_experts: 72
      n_trios: 8

live_cross_family:
  description: "Required-live Qwen x Gemma cross-family decorrelation companion settings."
  models: [qwen2.5:3b, gemma3:4b]
  replicates_per_model: 2
  seeds: [0, 1, 2, 3, 4, 5]
  n_items: 150
  target_accuracy: 0.75
  shared_strength: 0.85
  base_url: "http://localhost:11434"
  temperature: 0.6
  num_predict: 1
  timeout: 20.0
  progress_every: 25
  require_live: true

live_postdoc_panel:
  description: "Required-live Gemma postdoctoral-review panel sampling companion settings."
  model: gemma3:4b
  seeds: [0, 1, 2, 3, 4, 5]
  n_reviewers: 48
  n_applications: 72
  panel_sizes: [3, 6]
  strategies: [representative_sortition, random_selection, ideological_selection, expertise_threshold]
  n_trios: 8
  prevalence_strong: 0.5
  age_min: 28
  age_max: 52
  mean_expertise: 0.74
  expertise_heterogeneity: 0.08
  age_bias_std: 0.65
  base_url: "http://localhost:11434"
  temperature: 0.2
  num_predict: 1
  timeout: 20.0
  progress_every: 25
  require_live: true
  cache_path: "output/data/postdoc_votes/gemma3_4b_votes.json"
"""

_REQUIRED_KEYS = (
    "strategies",
    "panel_sizes",
    "mean_expertises",
    "expertise_heterogeneities",
    "bias_stds",
    "n_items_values",
    "prevalence_as",
    "seeds",
    "n_experts",
    "n_trios",
)


@dataclass(frozen=True)
class ExperimentProfile:
    name: str
    description: str
    grid: SweepGrid
    config_hash: str

    def metadata(self) -> dict[str, str | int]:
        panel_sizes = ",".join(str(size) for size in self.grid.panel_sizes)
        n_items = ",".join(str(value) for value in self.grid.n_items_values)
        prevalences = ",".join(str(value) for value in self.grid.prevalence_as)
        return {
            "profile_name": self.name,
            "profile_description": self.description,
            "config_hash": self.config_hash,
            "n_seeds": len(self.grid.seeds),
            "n_experts": self.grid.n_experts,
            "n_trios": self.grid.n_trios,
            "panel_sizes": panel_sizes,
            "n_items_values": n_items,
            "prevalence_as": prevalences,
        }


@dataclass(frozen=True)
class LiveCrossFamilyConfig:
    config_hash: str
    models: tuple[str, ...]
    replicates_per_model: int
    seeds: tuple[int, ...]
    n_items: int
    target_accuracy: float
    shared_strength: float
    base_url: str
    temperature: float
    num_predict: int
    timeout: float
    progress_every: int
    require_live: bool

    def expanded_models(self) -> tuple[str, ...]:
        return tuple(
            model
            for model in self.models
            for _ in range(self.replicates_per_model)
        )

    def metadata(self) -> dict[str, object]:
        return {
            "config_hash": self.config_hash,
            "models": list(self.models),
            "replicates_per_model": self.replicates_per_model,
            "n_requested_judges": len(self.models) * self.replicates_per_model,
            "seed_list": list(self.seeds),
            "n_items": self.n_items,
            "target_accuracy": self.target_accuracy,
            "shared_strength": self.shared_strength,
            "temperature": self.temperature,
            "num_predict": self.num_predict,
            "timeout": self.timeout,
            "require_live": self.require_live,
        }


@dataclass(frozen=True)
class LivePostdocPanelConfig:
    config_hash: str
    model: str
    seeds: tuple[int, ...]
    n_reviewers: int
    n_applications: int
    panel_sizes: tuple[int, ...]
    strategies: tuple[str, ...]
    n_trios: int
    prevalence_strong: float
    age_min: int
    age_max: int
    mean_expertise: float
    expertise_heterogeneity: float
    age_bias_std: float
    base_url: str
    temperature: float
    num_predict: int
    timeout: float
    progress_every: int
    require_live: bool
    cache_path: str

    def metadata(self) -> dict[str, object]:
        return {
            "config_hash": self.config_hash,
            "model": self.model,
            "seed_list": list(self.seeds),
            "n_reviewers": self.n_reviewers,
            "n_applications": self.n_applications,
            "panel_sizes": list(self.panel_sizes),
            "strategies": list(self.strategies),
            "n_trios": self.n_trios,
            "prevalence_strong": self.prevalence_strong,
            "age_range": [self.age_min, self.age_max],
            "mean_expertise": self.mean_expertise,
            "expertise_heterogeneity": self.expertise_heterogeneity,
            "age_bias_std": self.age_bias_std,
            "temperature": self.temperature,
            "num_predict": self.num_predict,
            "timeout": self.timeout,
            "progress_every": self.progress_every,
            "require_live": self.require_live,
            "cache_path": self.cache_path,
        }


def load_experiment_profile(
    config_path: Path,
    profile_name: str | None = None,
) -> ExperimentProfile:
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config must be a YAML mapping")

    profiles = _profile_mapping(data)
    selected_name = profile_name or str(data.get("default_profile", DEFAULT_PROFILE_NAME))
    raw_profile = profiles.get(selected_name)
    if not isinstance(raw_profile, dict):
        available = ", ".join(sorted(profiles))
        raise ValueError(f"unknown experiment profile {selected_name!r}; available: {available}")

    experiment = raw_profile.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError(f"profile {selected_name}: missing experiment mapping")
    _validate_live_cross_family(data.get("live_cross_family"))
    _validate_live_postdoc_panel(data.get("live_postdoc_panel"))
    grid = _grid_from_experiment(selected_name, experiment)
    description = str(raw_profile.get("description", selected_name)).strip() or selected_name
    return ExperimentProfile(
        name=selected_name,
        description=description,
        grid=grid,
        config_hash=_config_hash({"profile": selected_name, "experiment": experiment}),
    )


def load_live_cross_family_config(config_path: Path) -> LiveCrossFamilyConfig:
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config must be a YAML mapping")
    config = data.get("live_cross_family")
    _validate_live_cross_family(config)
    if not isinstance(config, dict):
        raise ValueError("missing live_cross_family mapping")
    return LiveCrossFamilyConfig(
        config_hash=_config_hash({"live_cross_family": config}),
        models=tuple(str(model) for model in config["models"]),
        replicates_per_model=int(config.get("replicates_per_model", 2)),
        seeds=tuple(int(seed) for seed in config["seeds"]),
        n_items=int(config["n_items"]),
        target_accuracy=float(config["target_accuracy"]),
        shared_strength=float(config["shared_strength"]),
        base_url=str(config.get("base_url", "http://localhost:11434")),
        temperature=float(config.get("temperature", 0.6)),
        num_predict=int(config.get("num_predict", 1)),
        timeout=float(config.get("timeout", 20.0)),
        progress_every=int(config.get("progress_every", 25)),
        require_live=bool(config.get("require_live", False)),
    )


def load_live_postdoc_panel_config(config_path: Path) -> LivePostdocPanelConfig:
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config must be a YAML mapping")
    config = data.get("live_postdoc_panel")
    _validate_live_postdoc_panel(config)
    if not isinstance(config, dict):
        raise ValueError("missing live_postdoc_panel mapping")
    return LivePostdocPanelConfig(
        config_hash=_config_hash({"live_postdoc_panel": config}),
        model=str(config.get("model", "gemma3:4b")),
        seeds=tuple(int(seed) for seed in config["seeds"]),
        n_reviewers=int(config["n_reviewers"]),
        n_applications=int(config["n_applications"]),
        panel_sizes=tuple(int(size) for size in config["panel_sizes"]),
        strategies=tuple(str(strategy) for strategy in config["strategies"]),
        n_trios=int(config["n_trios"]),
        prevalence_strong=float(config.get("prevalence_strong", 0.5)),
        age_min=int(config.get("age_min", 28)),
        age_max=int(config.get("age_max", 52)),
        mean_expertise=float(config.get("mean_expertise", 0.74)),
        expertise_heterogeneity=float(config.get("expertise_heterogeneity", 0.08)),
        age_bias_std=float(config.get("age_bias_std", 0.65)),
        base_url=str(config.get("base_url", "http://localhost:11434")),
        temperature=float(config.get("temperature", 0.2)),
        num_predict=int(config.get("num_predict", 1)),
        timeout=float(config.get("timeout", 20.0)),
        progress_every=int(config.get("progress_every", 25)),
        require_live=bool(config.get("require_live", False)),
        cache_path=str(config.get("cache_path", "output/data/postdoc_votes/gemma3_4b_votes.json")),
    )


def _profile_mapping(data: dict[str, Any]) -> dict[str, Any]:
    profiles = data.get("experiment_profiles")
    if isinstance(profiles, dict):
        return profiles
    experiment = data.get("experiment")
    if isinstance(experiment, dict):
        return {
            DEFAULT_PROFILE_NAME: {
                "description": "Legacy top-level experiment block.",
                "experiment": experiment,
            }
        }
    raise ValueError("missing experiment_profiles mapping")


def _grid_from_experiment(profile_name: str, experiment: dict[str, Any]) -> SweepGrid:
    for key in _REQUIRED_KEYS:
        if key not in experiment:
            raise ValueError(f"profile {profile_name}: missing experiment.{key}")

    grid = SweepGrid(
        strategies=_tuple_axis(profile_name, experiment, "strategies"),
        panel_sizes=_tuple_axis(profile_name, experiment, "panel_sizes"),
        mean_expertises=_tuple_axis(profile_name, experiment, "mean_expertises"),
        expertise_heterogeneities=_tuple_axis(
            profile_name, experiment, "expertise_heterogeneities"
        ),
        bias_stds=_tuple_axis(profile_name, experiment, "bias_stds"),
        n_items_values=_tuple_axis(profile_name, experiment, "n_items_values"),
        prevalence_as=_tuple_axis(profile_name, experiment, "prevalence_as"),
        seeds=_tuple_axis(profile_name, experiment, "seeds"),
        n_experts=int(experiment["n_experts"]),
        n_trios=int(experiment["n_trios"]),
    )
    _validate_grid_ranges(profile_name, grid)
    return grid


def _tuple_axis(profile_name: str, experiment: dict[str, Any], key: str) -> tuple[Any, ...]:
    value = experiment[key]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"profile {profile_name}: experiment.{key} must be a list")
    return tuple(value)


def _validate_grid_ranges(profile_name: str, grid: SweepGrid) -> None:
    if len(set(grid.seeds)) != len(grid.seeds):
        raise ValueError(f"profile {profile_name}: seeds must be unique")
    for strategy in grid.strategies:
        if strategy not in STRATEGIES:
            raise ValueError(f"profile {profile_name}: invalid strategy {strategy}")
    for panel_size in grid.panel_sizes:
        if int(panel_size) < 3:
            raise ValueError(f"profile {profile_name}: panel_sizes must be >= 3")
        if int(panel_size) > grid.n_experts:
            raise ValueError(f"profile {profile_name}: panel_size exceeds n_experts")
    for value in grid.mean_expertises:
        if not 0.0 < float(value) < 1.0:
            raise ValueError(f"profile {profile_name}: mean_expertises must be in (0, 1)")
    for value in grid.expertise_heterogeneities:
        if float(value) < 0.0:
            raise ValueError(
                f"profile {profile_name}: expertise_heterogeneities must be non-negative"
            )
    for value in grid.bias_stds:
        if float(value) < 0.0:
            raise ValueError(f"profile {profile_name}: bias_stds must be non-negative")
    for value in grid.n_items_values:
        if int(value) <= 0:
            raise ValueError(f"profile {profile_name}: n_items_values must be positive")
    for value in grid.prevalence_as:
        if not 0.0 < float(value) < 1.0:
            raise ValueError(f"profile {profile_name}: prevalence_as must be in (0, 1)")
    max_trios = max(math.comb(int(panel_size), 3) for panel_size in grid.panel_sizes)
    if int(grid.n_trios) > max_trios:
        raise ValueError(
            f"profile {profile_name}: n_trios exceeds every panel-size trio count"
        )


def _validate_live_cross_family(config: object) -> None:
    if config is None:
        return
    if not isinstance(config, dict):
        raise ValueError("live_cross_family must be a mapping")
    models = config.get("models")
    if not isinstance(models, list) or not models or not all(isinstance(m, str) and m for m in models):
        raise ValueError("live_cross_family.models must be a non-empty list of model tags")
    replicates = int(config.get("replicates_per_model", 2))
    if replicates < 1:
        raise ValueError("live_cross_family.replicates_per_model must be >= 1")
    seeds = config.get("seeds", [])
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("live_cross_family.seeds must be a non-empty unique list")
    if int(config.get("n_items", 0)) <= 0:
        raise ValueError("live_cross_family.n_items must be positive")
    target_accuracy = float(config.get("target_accuracy", 0.0))
    if not 0.0 < target_accuracy < 1.0:
        raise ValueError("live_cross_family.target_accuracy must be in (0, 1)")
    shared_strength = float(config.get("shared_strength", 0.0))
    if not 0.0 <= shared_strength <= 1.0:
        raise ValueError("live_cross_family.shared_strength must be in [0, 1]")
    base_url = config.get("base_url", "http://localhost:11434")
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("live_cross_family.base_url must be a non-empty string")
    if float(config.get("temperature", 0.6)) < 0.0:
        raise ValueError("live_cross_family.temperature must be non-negative")
    if int(config.get("num_predict", 1)) <= 0:
        raise ValueError("live_cross_family.num_predict must be positive")
    if float(config.get("timeout", 20.0)) <= 0.0:
        raise ValueError("live_cross_family.timeout must be positive")
    if int(config.get("progress_every", 25)) <= 0:
        raise ValueError("live_cross_family.progress_every must be positive")
    if not isinstance(config.get("require_live", False), bool):
        raise ValueError("live_cross_family.require_live must be a boolean")


def _validate_live_postdoc_panel(config: object) -> None:
    if config is None:
        return
    if not isinstance(config, dict):
        raise ValueError("live_postdoc_panel must be a mapping")
    model = config.get("model", "gemma3:4b")
    if model != "gemma3:4b":
        raise ValueError("live_postdoc_panel.model must be gemma3:4b")
    seeds = config.get("seeds", [])
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("live_postdoc_panel.seeds must be a non-empty unique list")
    panel_sizes = config.get("panel_sizes", [])
    if not isinstance(panel_sizes, list) or not panel_sizes:
        raise ValueError("live_postdoc_panel.panel_sizes must be a non-empty list")
    if any(int(size) < 3 for size in panel_sizes):
        raise ValueError("live_postdoc_panel.panel_sizes must be >= 3")
    strategies = config.get("strategies", [])
    if not isinstance(strategies, list) or not strategies:
        raise ValueError("live_postdoc_panel.strategies must be a non-empty list")
    for strategy in strategies:
        if strategy not in STRATEGIES:
            raise ValueError(f"live_postdoc_panel invalid strategy {strategy}")
    n_reviewers = int(config.get("n_reviewers", 0))
    if n_reviewers < max(int(size) for size in panel_sizes):
        raise ValueError("live_postdoc_panel.n_reviewers must cover panel sizes")
    if int(config.get("n_applications", 0)) <= 0:
        raise ValueError("live_postdoc_panel.n_applications must be positive")
    max_trios = max(math.comb(int(size), 3) for size in panel_sizes)
    if int(config.get("n_trios", 0)) <= 0 or int(config.get("n_trios", 0)) > max_trios:
        raise ValueError("live_postdoc_panel.n_trios must fit available trios")
    prevalence = float(config.get("prevalence_strong", 0.5))
    if not 0.0 < prevalence < 1.0:
        raise ValueError("live_postdoc_panel.prevalence_strong must be in (0, 1)")
    age_min = int(config.get("age_min", 28))
    age_max = int(config.get("age_max", 52))
    if age_min >= age_max:
        raise ValueError("live_postdoc_panel.age_min must be below age_max")
    mean_expertise = float(config.get("mean_expertise", 0.74))
    if not 0.0 < mean_expertise < 1.0:
        raise ValueError("live_postdoc_panel.mean_expertise must be in (0, 1)")
    if float(config.get("expertise_heterogeneity", 0.08)) < 0.0:
        raise ValueError("live_postdoc_panel.expertise_heterogeneity must be non-negative")
    if float(config.get("age_bias_std", 0.65)) < 0.0:
        raise ValueError("live_postdoc_panel.age_bias_std must be non-negative")
    base_url = config.get("base_url", "http://localhost:11434")
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("live_postdoc_panel.base_url must be a non-empty string")
    if float(config.get("temperature", 0.2)) < 0.0:
        raise ValueError("live_postdoc_panel.temperature must be non-negative")
    if int(config.get("num_predict", 1)) <= 0:
        raise ValueError("live_postdoc_panel.num_predict must be positive")
    if float(config.get("timeout", 20.0)) <= 0.0:
        raise ValueError("live_postdoc_panel.timeout must be positive")
    if int(config.get("progress_every", 25)) <= 0:
        raise ValueError("live_postdoc_panel.progress_every must be positive")
    if not isinstance(config.get("require_live", False), bool):
        raise ValueError("live_postdoc_panel.require_live must be a boolean")
    cache_path = config.get("cache_path", "")
    if not isinstance(cache_path, str) or not cache_path.strip():
        raise ValueError("live_postdoc_panel.cache_path must be a non-empty string")


def _config_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_PROFILE_NAME",
    "ExperimentProfile",
    "LiveCrossFamilyConfig",
    "LivePostdocPanelConfig",
    "load_experiment_profile",
    "load_live_cross_family_config",
    "load_live_postdoc_panel_config",
]
