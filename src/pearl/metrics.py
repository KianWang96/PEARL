"""Experiment metrics."""

from __future__ import annotations

import numpy as np


def compute_selection_entropy(selection_history, neighbors, window: int) -> float:
    """Mean normalized per-client entropy of recent peer selections."""
    del window
    entropies = []
    for k, nbrs in neighbors.items():
        if len(nbrs) <= 1:
            continue
        counts = np.asarray(
            [sum(1 for jj in selection_history[(k,)] if jj == j) for j in nbrs],
            dtype=float,
        )
        total = counts.sum()
        if total == 0:
            continue
        probs = counts / total
        probs = probs[probs > 0]
        h = -np.sum(probs * np.log(probs))
        entropies.append(h / np.log(len(nbrs)))
    return float(np.mean(entropies)) if entropies else 0.0


def compute_negative_transfer_rate(prev_val_losses, curr_val_losses) -> float:
    """Fraction of clients whose validation loss increased after mixing."""
    if not prev_val_losses:
        return 0.0
    events = sum(
        1
        for k in curr_val_losses
        if k in prev_val_losses and curr_val_losses[k] > prev_val_losses[k]
    )
    return events / len(curr_val_losses)


def jain_fairness(values) -> float:
    """Jain's fairness index for nonnegative client metrics."""
    arr = np.asarray(values, dtype=float)
    denom = len(arr) * np.sum(arr**2)
    if len(arr) == 0 or denom <= 0:
        return 0.0
    return float((np.sum(arr) ** 2) / denom)
