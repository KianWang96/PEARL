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
from pearl.data import get_dataset, make_client_loaders, make_partitions, make_test_loader
from pearl.descriptors import build_descriptors
from pearl.exchange import exchange_and_mix, parameter_bytes
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
    )

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
    ).to(device)
    init_sd = copy.deepcopy(init_model.state_dict())
    models = [
        RepAEClassifier(
            in_channels,
            num_classes,
            latent_dim=config.latent_dim,
            img_size=img_size,
        ).to(device)
        for _ in range(config.num_clients)
    ]
    for model in models:
        model.load_state_dict(copy.deepcopy(init_sd))

    comm_bytes_per_exchange = parameter_bytes(models[0], config.exchange_mode)
    selection_history = defaultdict(lambda: deque(maxlen=config.explore_window))
    static_peer_cache: dict[int, int] = {}
    history: list[dict] = []
    cumulative_bytes = 0

    for round_idx in range(config.rounds):
        start = time.time()
        local_losses = [
            local_train(models[k], client_loaders[k], config, device)
            for k in range(config.num_clients)
        ]

        descriptors = build_descriptors(models, client_loaders, num_classes, device)
        pre_val_losses = {
            k: descriptors[k]["rec_mse"] + abs(descriptors[k]["quality"])
            for k in range(config.num_clients)
        }

        selected_pairs: list[tuple[int, int]] = []
        if config.selection_method != "local_only":
            snapshots = [copy.deepcopy(model) for model in models]
            for k in range(config.num_clients):
                peer = select_peer(
                    k,
                    neighbors,
                    descriptors,
                    snapshots,
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

            for k, peer in selected_pairs:
                exchange_and_mix(models[k], snapshots[peer], config)

        round_bytes = len(selected_pairs) * comm_bytes_per_exchange
        cumulative_bytes += round_bytes

        post_descriptors = build_descriptors(models, client_loaders, num_classes, device)
        post_val_losses = {
            k: post_descriptors[k]["rec_mse"] + abs(post_descriptors[k]["quality"])
            for k in range(config.num_clients)
        }
        neg_transfer_rate = compute_negative_transfer_rate(
            pre_val_losses,
            post_val_losses,
        )

        should_eval = (
            round_idx % config.eval_every == 0
            or round_idx == config.rounds - 1
        )
        if should_eval:
            client_df = evaluate_clients(models, client_loaders, device)
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
                if config.selection_method != "local_only"
                else 0.0
            )

            row = {
                "seed": config.seed,
                "round": round_idx,
                "method": config.selection_method,
                "graph": config.graph_type,
                "partition": config.partition,
                "dirichlet_alpha": config.dirichlet_alpha,
                "exchange_mode": config.exchange_mode,
                "num_clients": config.num_clients,
                "mean_local_loss": float(np.mean(local_losses)),
                "mean_global_accuracy": mean_global_acc,
                "mean_global_macro_f1": mean_global_f1,
                "mean_client_accuracy": mean_client_acc,
                "worst_client_accuracy": worst_client_acc,
                "std_client_accuracy": std_client_acc,
                "jain_client_accuracy": client_fairness,
                "neg_transfer_rate": neg_transfer_rate,
                "round_exchanges": len(selected_pairs),
                "round_bytes": round_bytes,
                "cum_bytes": cumulative_bytes,
                "selection_entropy": selection_entropy,
                "runtime_sec": time.time() - start,
            }
            history.append(row)

            if verbose:
                print(
                    f"Round {round_idx:03d} | {config.selection_method:<30} | "
                    f"acc={mean_global_acc:.3f} | f1={mean_global_f1:.3f} | "
                    f"worst={worst_client_acc:.3f} | "
                    f"neg_xfer={neg_transfer_rate:.3f} | "
                    f"entropy={selection_entropy:.3f} | "
                    f"MB={cumulative_bytes / 1e6:.2f}"
                )

    return pd.DataFrame(history)


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
