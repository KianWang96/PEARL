"""High-level experiment runner."""

from __future__ import annotations

import copy
import time
from collections import defaultdict, deque
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from pearl.config import ExperimentConfig
from pearl.constants import (
    DECENTRALIZED_REFERENCE_METHODS,
    DESCRIPTOR_METHODS,
    SERVER_METHODS,
)
from pearl.data import get_dataset, make_client_loaders, make_partitions, make_test_loader
from pearl.descriptors import build_descriptors
from pearl.exchange import (
    exchange_and_mix,
    load_filtered_state_dict,
    parameter_bytes,
    weighted_average_state_dict,
)
from pearl.graphs import make_graph
from pearl.metrics import (
    compute_negative_transfer_rate,
    compute_selection_entropy,
    jain_fairness,
)
from pearl.models import RepAEClassifier
from pearl.selection import select_peer
from pearl.training import evaluate_clients, evaluate_model, local_train, mean_metric
from pearl.utils import resolve_device, set_seed


def run_experiment(config: ExperimentConfig, verbose: bool = True) -> pd.DataFrame:
    """Run one method and one seed, returning round-level metrics."""
    set_seed(config.seed)
    device = resolve_device(config.device)
    if verbose:
        print(f"Device: {device}")

    train_ds, test_ds, num_classes, in_channels, img_size = get_dataset(
        config.dataset,
        data_dir=config.data_dir,
        train_subset=config.train_subset,
        test_subset=config.test_subset,
        download=config.download,
    )
    client_indices = make_partitions(train_ds, config, num_classes)
    client_loaders = make_client_loaders(train_ds, client_indices, config)
    test_loader = make_test_loader(
        test_ds,
        batch_size=256,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == "cuda",
    )

    is_server_method = config.selection_method in SERVER_METHODS
    if is_server_method:
        neighbors = {k: [] for k in range(config.num_clients)}
    else:
        graph = make_graph(
            config.num_clients,
            config.graph_type,
            config.er_prob,
            config.seed,
        )
        neighbors = {k: list(graph.neighbors(k)) for k in range(config.num_clients)}

    init_model = RepAEClassifier(
        in_channels,
        num_classes,
        latent_dim=config.latent_dim,
        img_size=img_size,
        width=config.model_width,
    ).to(device)
    init_sd = copy.deepcopy(init_model.state_dict())
    models = [
        RepAEClassifier(
            in_channels,
            num_classes,
            latent_dim=config.latent_dim,
            img_size=img_size,
            width=config.model_width,
        ).to(device)
        for _ in range(config.num_clients)
    ]
    for model in models:
        model.load_state_dict(copy.deepcopy(init_sd))

    global_state = copy.deepcopy(init_sd)
    ditto_global_models = (
        [copy.deepcopy(model) for model in models]
        if config.selection_method == "ditto"
        else None
    )
    selection_history = defaultdict(lambda: deque(maxlen=config.explore_window))
    static_peer_cache: dict[int, int] = {}
    cached_descriptors: dict[int, dict] | None = None
    cached_selection_models: list[torch.nn.Module] | None = None
    last_descriptor_refresh = 0
    history: list[dict] = []
    cumulative_bytes = 0

    for round_idx in range(config.rounds):
        start = time.time()
        active_clients = _active_clients_for_round(
            config.num_clients,
            config.active_probability,
            config.seed,
            round_idx,
        )
        if is_server_method:
            active_clients = list(range(config.num_clients))
            if config.selection_method == "ditto":
                if ditto_global_models is None:
                    raise RuntimeError("Ditto global client models were not initialized.")
                for model in ditto_global_models:
                    load_filtered_state_dict(model, global_state)
            else:
                broadcast_filter = (
                    _is_shared_representation
                    if config.selection_method in {"fedper", "fedrep"}
                    else None
                )
                for model in models:
                    load_filtered_state_dict(model, global_state, broadcast_filter)

        proximal_reference = None
        proximal_mu = 0.0
        if config.selection_method == "fedprox":
            parameter_names = set(dict(models[0].named_parameters()))
            proximal_reference = {
                name: value.detach().clone()
                for name, value in global_state.items()
                if name in parameter_names
            }
            proximal_mu = config.fedprox_mu

        if config.selection_method == "ditto":
            if ditto_global_models is None:
                raise RuntimeError("Ditto global client models were not initialized.")
            for k in active_clients:
                local_train(
                    ditto_global_models[k],
                    client_loaders[k],
                    config,
                    device,
                )
            weights = [float(len(client_indices[k])) for k in active_clients]
            global_state = weighted_average_state_dict(
                [ditto_global_models[k] for k in active_clients],
                weights,
            )
            parameter_names = set(dict(models[0].named_parameters()))
            ditto_reference = {
                name: value.detach().clone()
                for name, value in global_state.items()
                if name in parameter_names
            }
            local_losses = [
                local_train(
                    models[k],
                    client_loaders[k],
                    config,
                    device,
                    proximal_reference=ditto_reference,
                    proximal_mu=config.ditto_lambda,
                    local_epochs=config.ditto_personal_epochs,
                )
                for k in active_clients
            ]
        elif config.selection_method == "fedrep":
            local_losses = []
            for k in active_clients:
                head_loss = local_train(
                    models[k],
                    client_loaders[k],
                    config,
                    device,
                    trainable_filter=_is_classifier,
                    local_epochs=config.fedrep_head_epochs,
                )
                representation_loss = local_train(
                    models[k],
                    client_loaders[k],
                    config,
                    device,
                    trainable_filter=_is_shared_representation,
                    local_epochs=config.fedrep_rep_epochs,
                )
                local_losses.append(0.5 * (head_loss + representation_loss))
        else:
            local_losses = [
                local_train(
                    models[k],
                    client_loaders[k],
                    config,
                    device,
                    proximal_reference=proximal_reference,
                    proximal_mu=proximal_mu,
                )
                for k in active_clients
            ]

        descriptors = build_descriptors(models, client_loaders, num_classes, device)
        pre_val_losses = {
            k: descriptors[k]["rec_mse"] + abs(descriptors[k]["quality"])
            for k in range(config.num_clients)
        }

        selected_pairs: list[tuple[int, int]] = []
        descriptor_age = 0
        round_bytes = 0
        effective_exchange_mode = config.exchange_mode
        active_neighbors = {
            k: [j for j in neighbors[k] if j in active_clients]
            if k in active_clients
            else []
            for k in range(config.num_clients)
        }

        if is_server_method:
            aggregation_filter = (
                _is_shared_representation
                if config.selection_method in {"fedper", "fedrep"}
                else None
            )
            if config.selection_method != "ditto":
                weights = [float(len(client_indices[k])) for k in active_clients]
                averaged_state = weighted_average_state_dict(
                    [models[k] for k in active_clients],
                    weights,
                    aggregation_filter,
                )
                if aggregation_filter is None:
                    global_state = averaged_state
                else:
                    for name in global_state:
                        if aggregation_filter(name):
                            global_state[name] = averaged_state[name].detach().clone()
                for model in models:
                    load_filtered_state_dict(model, global_state, aggregation_filter)

            server_mode = (
                "encoder_decoder_local_head"
                if config.selection_method in {"fedper", "fedrep"}
                else "full_model"
            )
            effective_exchange_mode = server_mode
            payload_bytes = parameter_bytes(models[0], server_mode)
            round_bytes = 2 * len(active_clients) * payload_bytes
        elif config.selection_method == "dpsgd_full_neighbors":
            current_snapshots = [copy.deepcopy(model) for model in models]
            pending_states: dict[int, dict[str, torch.Tensor]] = {}
            for k in active_clients:
                peers = active_neighbors[k]
                if not peers:
                    continue
                participants = [current_snapshots[k], *[current_snapshots[j] for j in peers]]
                pending_states[k] = weighted_average_state_dict(
                    participants,
                    [1.0] * len(participants),
                )
                selected_pairs.extend((k, peer) for peer in peers)
            for k, state in pending_states.items():
                load_filtered_state_dict(models[k], state)
            effective_exchange_mode = "full_model"
            payload_bytes = parameter_bytes(models[0], effective_exchange_mode)
            round_bytes = len(selected_pairs) * payload_bytes
        elif config.selection_method != "local_only":
            current_snapshots = [copy.deepcopy(model) for model in models]
            if config.selection_method in DESCRIPTOR_METHODS:
                should_refresh = (
                    cached_descriptors is None
                    or round_idx % config.descriptor_refresh_period == 0
                )
                if should_refresh:
                    cached_descriptors = copy.deepcopy(descriptors)
                    cached_selection_models = [
                        copy.deepcopy(model) for model in current_snapshots
                    ]
                    last_descriptor_refresh = round_idx
                descriptor_age = round_idx - last_descriptor_refresh
                selection_descriptors = cached_descriptors
                selection_models = cached_selection_models
            else:
                selection_descriptors = descriptors
                selection_models = current_snapshots

            if selection_descriptors is None or selection_models is None:
                raise RuntimeError("Selection descriptors were not initialized.")

            for k in active_clients:
                peer = select_peer(
                    k,
                    active_neighbors,
                    selection_descriptors,
                    selection_models,
                    client_indices,
                    train_ds,
                    config.selection_method,
                    selection_history,
                    config,
                    static_peer_cache,
                    device,
                )
                if peer is not None:
                    selected_pairs.append((k, peer))
                    selection_history[(k,)].append(peer)

            exchange_config = (
                replace(config, exchange_mode="full_model")
                if config.selection_method == "dpsgd_one_peer"
                else config
            )
            effective_exchange_mode = exchange_config.exchange_mode
            for k, peer in selected_pairs:
                exchange_and_mix(models[k], current_snapshots[peer], exchange_config)

            payload_bytes = parameter_bytes(
                models[0],
                exchange_config.exchange_mode,
            )
            round_bytes = len(selected_pairs) * payload_bytes

        cumulative_bytes += round_bytes

        post_descriptors = (
            descriptors
            if config.selection_method == "ditto"
            else build_descriptors(models, client_loaders, num_classes, device)
        )
        post_val_losses = {
            k: post_descriptors[k]["rec_mse"] + abs(post_descriptors[k]["quality"])
            for k in range(config.num_clients)
        }
        transfer_clients = (
            []
            if config.selection_method == "ditto"
            else active_clients
            if is_server_method
            else sorted({k for k, _ in selected_pairs})
        )
        neg_transfer_rate = compute_negative_transfer_rate(
            {k: pre_val_losses[k] for k in transfer_clients},
            {k: post_val_losses[k] for k in transfer_clients},
        )

        should_eval = (
            round_idx % config.eval_every == 0
            or round_idx == config.rounds - 1
        )
        if should_eval:
            client_df = evaluate_clients(models, client_loaders, device)
            if config.selection_method in {"fedavg", "fedprox"}:
                global_metric = evaluate_model(models[0], test_loader, device)
                mean_global_acc = global_metric["accuracy"]
                mean_global_f1 = global_metric["macro_f1"]
            else:
                global_metrics = [
                    evaluate_model(models[k], test_loader, device)
                    for k in range(config.num_clients)
                ]
                mean_global_acc = mean_metric(global_metrics, "accuracy")
                mean_global_f1 = mean_metric(global_metrics, "macro_f1")
            mean_client_acc = float(client_df["accuracy"].mean())
            worst_client_acc = float(client_df["accuracy"].min())
            std_client_acc = float(client_df["accuracy"].std())
            client_fairness = jain_fairness(client_df["accuracy"].values)
            selection_entropy = (
                compute_selection_entropy(
                    selection_history,
                    neighbors,
                    config.explore_window,
                )
                if config.selection_method
                not in {"local_only", *SERVER_METHODS, *DECENTRALIZED_REFERENCE_METHODS}
                else 0.0
            )

            active_degrees = [len(active_neighbors[k]) for k in active_clients]
            mean_active_degree = (
                float(np.mean(active_degrees)) if active_degrees else 0.0
            )
            no_active_peer_fraction = (
                sum(degree == 0 for degree in active_degrees) / len(active_degrees)
                if active_degrees and not is_server_method
                else 0.0
            )

            row = {
                "seed": config.seed,
                "round": round_idx,
                "method": config.selection_method,
                "comparison_family": (
                    "server_reference"
                    if is_server_method
                    else "decentralized_reference"
                    if config.selection_method in DECENTRALIZED_REFERENCE_METHODS
                    else "budget_matched_decentralized"
                ),
                "dataset": config.dataset,
                "graph": "server" if is_server_method else config.graph_type,
                "partition": config.partition,
                "dirichlet_alpha": config.dirichlet_alpha,
                "exchange_mode": effective_exchange_mode,
                "num_clients": config.num_clients,
                "active_probability": config.active_probability,
                "active_fraction": len(active_clients) / config.num_clients,
                "mean_active_degree": mean_active_degree,
                "no_active_peer_fraction": no_active_peer_fraction,
                "descriptor_refresh_period": config.descriptor_refresh_period,
                "descriptor_age": descriptor_age,
                "mean_local_loss": (
                    float(np.mean(local_losses)) if local_losses else 0.0
                ),
                "mean_global_accuracy": mean_global_acc,
                "mean_global_macro_f1": mean_global_f1,
                "mean_client_accuracy": mean_client_acc,
                "worst_client_accuracy": worst_client_acc,
                "std_client_accuracy": std_client_acc,
                "jain_client_accuracy": client_fairness,
                "neg_transfer_rate": neg_transfer_rate,
                "round_exchanges": (
                    len(active_clients) if is_server_method else len(selected_pairs)
                ),
                "round_bytes": round_bytes,
                "cum_bytes": cumulative_bytes,
                "selection_entropy": selection_entropy,
                "runtime_sec": time.time() - start,
            }
            history.append(row)

            if verbose:
                progress = 100.0 * (round_idx + 1) / config.rounds
                print(
                    f"Round {round_idx + 1:03d}/{config.rounds:03d} "
                    f"({progress:5.1f}%) | {config.selection_method:<30} | "
                    f"acc={mean_global_acc:.3f} | f1={mean_global_f1:.3f} | "
                    f"worst={worst_client_acc:.3f} | "
                    f"neg_xfer={neg_transfer_rate:.3f} | "
                    f"entropy={selection_entropy:.3f} | "
                    f"active={len(active_clients):02d}/{config.num_clients:02d} | "
                    f"MB={cumulative_bytes / 1e6:.2f}",
                    flush=True,
                )

    return pd.DataFrame(history)


def _is_shared_representation(name: str) -> bool:
    return name.startswith("encoder") or name.startswith("decoder")


def _is_classifier(name: str) -> bool:
    return name.startswith("classifier")


def _active_clients_for_round(
    num_clients: int,
    probability: float,
    seed: int,
    round_idx: int,
) -> list[int]:
    if probability >= 1.0:
        return list(range(num_clients))
    rng = np.random.default_rng(np.random.SeedSequence([seed, round_idx, 9173]))
    return np.flatnonzero(rng.random(num_clients) < probability).astype(int).tolist()


def run_baseline_suite(
    base_config: ExperimentConfig,
    methods: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    all_results = []
    for method in methods or base_config.methods:
        cfg = replace(base_config, selection_method=method)
        if verbose:
            print(f"\nRunning: {method} | graph: {cfg.graph_type} | seed: {cfg.seed}")
        all_results.append(run_experiment(cfg, verbose=verbose))
    return pd.concat(all_results, ignore_index=True)
