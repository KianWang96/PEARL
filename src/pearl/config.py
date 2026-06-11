"""Configuration loading and command-line overrides."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, field
from pathlib import Path
from typing import Any

import yaml

from pearl.constants import DEFAULT_METHODS


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str = "pearl"
    output_dir: str = "results/pearl"
    seeds: list[int] = field(default_factory=lambda: [1, 2, 3])
    methods: list[str] = field(default_factory=lambda: DEFAULT_METHODS.copy())

    seed: int = 1
    dataset: str = "fashion_mnist"
    data_dir: str = "data"
    download: bool = True
    num_clients: int = 50
    partition: str = "dirichlet"
    dirichlet_alpha: float = 0.3
    classes_per_client: int = 2
    min_client_size: int = 10
    train_subset: int | None = None
    test_subset: int | None = None

    graph_type: str = "erdos_renyi"
    er_prob: float = 0.15

    rounds: int = 150
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 1e-3
    num_workers: int = 2
    pin_memory: bool = True
    device: str = "auto"

    latent_dim: int = 64
    lambda_rec: float = 1.0
    lambda_cls: float = 1.0

    mixing_alpha: float = 0.3
    exchange_mode: str = "encoder_decoder_local_head"
    selection_method: str = "pearl_full"

    beta_proto: float = 1.0
    beta_quality: float = 1.0
    beta_cost: float = 0.1
    beta_explore: float = 0.2

    anchor_size: int = 64
    hard_class_threshold: float = 0.5
    explore_window: int = 10
    eval_every: int = 1
    plot_format: str = "pdf"


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    """Load a flat YAML config into an ExperimentConfig."""
    if path is None:
        return ExperimentConfig()

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config must be a mapping: {config_path}")
    return config_from_mapping(data)


def config_from_mapping(data: dict[str, Any]) -> ExperimentConfig:
    field_names = {item.name for item in fields(ExperimentConfig)}
    unknown = sorted(set(data) - field_names)
    if unknown:
        raise KeyError(f"Unknown config key(s): {', '.join(unknown)}")

    defaults = asdict(ExperimentConfig())
    coerced = {}
    for key, value in data.items():
        coerced[key] = _coerce_like_default(value, defaults[key])
    return ExperimentConfig(**{**defaults, **coerced})


def apply_overrides(
    config: ExperimentConfig,
    overrides: list[str] | None,
) -> ExperimentConfig:
    """Apply CLI overrides in key=value form."""
    if not overrides:
        return config

    data = asdict(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must have form key=value: {override}")
        key, raw_value = override.split("=", 1)
        key = key.strip()
        if key not in data:
            raise KeyError(f"Unknown override key: {key}")
        parsed = _parse_override_value(raw_value)
        data[key] = _coerce_like_default(parsed, data[key])
    return ExperimentConfig(**data)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Save the fully resolved config next to experiment outputs."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False)


def _parse_override_value(raw_value: str) -> Any:
    value = raw_value.strip()
    if value.lower() in {"none", "null"}:
        return None
    return yaml.safe_load(value)


def _coerce_like_default(value: Any, default: Any) -> Any:
    if value is None:
        return None
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in {"true", "1", "yes", "y"}:
                return True
            if lowered in {"false", "0", "no", "n"}:
                return False
        raise ValueError(f"Cannot parse boolean value: {value!r}")
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, str):
        return str(value)
    if isinstance(default, list):
        if isinstance(value, str):
            value = [item.strip() for item in value.split(",") if item.strip()]
        if not isinstance(value, list):
            raise ValueError(f"Expected a list value, got: {value!r}")
        if default and isinstance(default[0], int):
            return [int(item) for item in value]
        return list(value)
    return value
