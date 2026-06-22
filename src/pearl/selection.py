"""Peer selection criteria for budgeted decentralized learning."""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch

from pearl.config import ExperimentConfig
from pearl.descriptors import anchor_quality, get_hard_classes, sample_anchor


def synthetic_comm_cost(k: int, j: int) -> float:
    """Placeholder communication cost proportional to node-id distance."""
    return 1.0 + 0.05 * abs(k - j)


def minmax_normalize(values, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    vmin = v.min()
    vmax = v.max()
    if abs(vmax - vmin) < eps:
        return np.zeros_like(v)
    return (v - vmin) / (vmax - vmin + eps)


def prototype_distance_hard(
    proto_k: dict[int, torch.Tensor],
    proto_j: dict[int, torch.Tensor],
    hard_classes: set[int],
) -> float:
    """Mean squared L2 distance on hard shared classes, falling back to all shared."""
    shared = set(proto_k) & set(proto_j)
    if not shared:
        return float("inf")

    focus = shared & hard_classes if hard_classes else set()
    use = focus if focus else shared
    dist = sum(torch.norm(proto_k[c] - proto_j[c], p=2).item() ** 2 for c in use)
    return dist / len(use)


def classifier_distance(
    model_k: torch.nn.Module,
    model_j: torch.nn.Module,
) -> float:
    """Squared distance between final classifier layers."""
    params_k = list(model_k.classifier[-1].parameters())
    params_j = list(model_j.classifier[-1].parameters())
    return float(
        sum(torch.sum((left - right) ** 2).item() for left, right in zip(params_k, params_j))
    )


def select_peer(
    k: int,
    neighbors: dict[int, list[int]],
    descriptors: dict[int, dict],
    models_snapshot: list[torch.nn.Module],
    client_indices_ref,
    train_ds_ref,
    method: str,
    selection_history: dict[tuple[int], deque[int]],
    config: ExperimentConfig,
    static_peer_cache: dict[int, int],
    device: torch.device,
) -> int | None:
    """Select a single neighbor for client k under the configured method."""
    neigh = neighbors[k]
    if not neigh:
        return None

    if method == "local_only":
        return None

    if method in {"random_peer", "dpsgd_one_peer"}:
        return random.choice(neigh)

    if method == "static_peer":
        if k not in static_peer_cache:
            static_peer_cache[k] = random.choice(neigh)
        peer = static_peer_cache[k]
        return peer if peer in neigh else None

    if method == "model_similarity":
        return min(
            neigh,
            key=lambda j: classifier_distance(models_snapshot[k], models_snapshot[j]),
        )

    hard_classes = get_hard_classes(
        descriptors[k]["class_acc"],
        config.hard_class_threshold,
    )
    anchor_xs, anchor_ys = sample_anchor(
        client_indices_ref[k],
        train_ds_ref,
        config.anchor_size,
    )

    raw_d = []
    raw_q = []
    raw_aq = []
    raw_c = []
    raw_e = []

    for j in neigh:
        distance = prototype_distance_hard(
            descriptors[k]["prototypes"],
            descriptors[j]["prototypes"],
            hard_classes,
        )
        raw_d.append(distance if np.isfinite(distance) else 1e6)
        raw_q.append(descriptors[j]["quality"])
        raw_aq.append(anchor_quality(models_snapshot[j], anchor_xs, anchor_ys, device))
        raw_c.append(synthetic_comm_cost(k, j))

        recent_count = sum(1 for jj in selection_history[(k,)] if jj == j)
        raw_e.append(1.0 / (1.0 + recent_count))

    d_norm = minmax_normalize(raw_d)
    q_norm = minmax_normalize(raw_q)
    aq_norm = minmax_normalize(raw_aq)
    c_norm = minmax_normalize(raw_c)
    e_norm = minmax_normalize(raw_e)

    scores = {}
    for idx, j in enumerate(neigh):
        if method == "prototype_only":
            scores[j] = -d_norm[idx]
        elif method == "quality_only":
            scores[j] = q_norm[idx]
        elif method == "prototype_quality":
            scores[j] = (
                -config.beta_proto * d_norm[idx]
                + config.beta_quality * q_norm[idx]
            )
        elif method == "prototype_quality_exploration":
            scores[j] = (
                -config.beta_proto * d_norm[idx]
                + config.beta_quality * q_norm[idx]
                - config.beta_cost * c_norm[idx]
                + config.beta_explore * e_norm[idx]
            )
        elif method == "anchor_quality":
            scores[j] = aq_norm[idx]
        elif method == "hard_class_alignment":
            scores[j] = (
                -config.beta_proto * d_norm[idx]
                + config.beta_quality * aq_norm[idx]
            )
        elif method == "pearl_full":
            scores[j] = (
                -config.beta_proto * d_norm[idx]
                + config.beta_quality * aq_norm[idx]
                - config.beta_cost * c_norm[idx]
                + config.beta_explore * e_norm[idx]
            )
        else:
            raise ValueError(f"Unknown selection method: {method}")

    return max(scores, key=scores.get)
