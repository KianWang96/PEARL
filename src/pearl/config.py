"""Configuration loading and command-line overrides."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, field
from pathlib import Path
from typing import Any

import yaml

from pearl.constants import DEFAULT_METHODS, SERVER_METHODS, SUPPORTED_METHODS


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
    model_width: int = 16
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
    fedprox_mu: float = 0.01
    fedrep_head_epochs: int = 1
    fedrep_rep_epochs: int = 1
    ditto_lambda: float = 0.1
    ditto_personal_epochs: int = 1
    active_probability: float = 1.0
    descriptor_refresh_period: int = 1
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
    return _validate_config(ExperimentConfig(**{**defaults, **coerced}))


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
    return _validate_config(ExperimentConfig(**data))


def _validate_config(config: ExperimentConfig) -> ExperimentConfig:
    if config.dataset not in {"fashion_mnist", "mnist", "cifar10"}:
        raise ValueError(f"Unknown dataset: {config.dataset}")
    if not 0.0 < config.active_probability <= 1.0:
        raise ValueError("active_probability must be in (0, 1].")
    if config.descriptor_refresh_period < 1:
        raise ValueError("descriptor_refresh_period must be at least 1.")
    if config.model_width < 1:
        raise ValueError("model_width must be at least 1.")
    if config.fedprox_mu < 0.0:
        raise ValueError("fedprox_mu must be nonnegative.")
    if config.fedrep_head_epochs < 1 or config.fedrep_rep_epochs < 1:
        raise ValueError("FedRep phase epochs must be at least 1.")
    if config.ditto_lambda < 0.0 or config.ditto_personal_epochs < 1:
        raise ValueError("Ditto lambda must be nonnegative and epochs at least 1.")
    unknown_methods = sorted(set(config.methods) - SUPPORTED_METHODS)
    if unknown_methods:
        raise ValueError(f"Unknown method(s): {', '.join(unknown_methods)}")
    if config.selection_method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown selection_method: {config.selection_method}")
    if config.graph_type == "server" and not set(config.methods) <= SERVER_METHODS:
        raise ValueError("graph_type=server may only contain server reference methods.")
    return config


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Save the fully resolved config next to experiment outputs."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False)
    temp_path.replace(output_path)


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
