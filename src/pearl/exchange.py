"""Model exchange and mixing helpers."""

from __future__ import annotations

import copy
from collections.abc import Callable

import torch

from pearl.config import ExperimentConfig


def mix_state_dict(
    sd_k: dict[str, torch.Tensor],
    sd_j: dict[str, torch.Tensor],
    alpha: float,
    filter_fn: Callable[[str], bool] | None = None,
) -> dict[str, torch.Tensor]:
    out = copy.deepcopy(sd_k)
    for name in out:
        if filter_fn is None or filter_fn(name):
            out[name] = (1 - alpha) * sd_k[name].float() + alpha * sd_j[name].float()
    return out


def exchange_and_mix(
    model_k: torch.nn.Module,
    model_j: torch.nn.Module,
    config: ExperimentConfig,
) -> torch.nn.Module:
    alpha = config.mixing_alpha
    mode = config.exchange_mode
    sd_k = model_k.state_dict()
    sd_j = model_j.state_dict()

    if mode == "full_model":
        new_sd = mix_state_dict(sd_k, sd_j, alpha)
    elif mode == "encoder_only":
        new_sd = mix_state_dict(sd_k, sd_j, alpha, lambda name: name.startswith("encoder"))
    elif mode == "encoder_decoder_local_head":
        new_sd = mix_state_dict(
            sd_k,
            sd_j,
            alpha,
            lambda name: name.startswith("encoder") or name.startswith("decoder"),
        )
    else:
        raise ValueError(f"Unknown exchange_mode: {mode}")

    model_k.load_state_dict(new_sd)
    return model_k


def parameter_bytes(model: torch.nn.Module, mode: str) -> int:
    total = 0
    for name, parameter in model.named_parameters():
        include = (
            mode == "full_model"
            or (mode == "encoder_only" and name.startswith("encoder"))
            or (
                mode == "encoder_decoder_local_head"
                and (name.startswith("encoder") or name.startswith("decoder"))
            )
        )
        if include:
            total += parameter.numel() * 4
    return total
