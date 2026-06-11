"""Communication graph construction."""

from __future__ import annotations

import networkx as nx


def make_graph(
    num_clients: int,
    graph_type: str = "erdos_renyi",
    er_prob: float = 0.15,
    seed: int = 42,
) -> nx.Graph:
    if graph_type == "ring":
        return nx.cycle_graph(num_clients)

    if graph_type == "erdos_renyi":
        attempt = seed
        while True:
            graph = nx.erdos_renyi_graph(num_clients, er_prob, seed=attempt)
            if nx.is_connected(graph):
                return graph
            attempt += 1

    if graph_type == "scale_free":
        m = max(1, min(3, num_clients - 1))
        return nx.barabasi_albert_graph(num_clients, m, seed=seed)

    raise ValueError(f"Unknown graph_type: {graph_type}")
