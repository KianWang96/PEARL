"""Plotting utilities for PEARL result CSVs."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from pearl.constants import METHOD_COLORS, METHOD_WIDTHS


def plot_metric_vs_rounds(
    results: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_dir: str | Path,
    fmt: str = "pdf",
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))

    for method, grp in results.groupby("method"):
        if metric not in grp.columns:
            continue
        mean, lo, hi = _mean_range_by_round(grp, metric)
        color = METHOD_COLORS.get(method, "#333333")
        lw = METHOD_WIDTHS.get(method, 1.5)
        ax.plot(mean.index, mean.values, label=method, color=color, linewidth=lw)
        ax.fill_between(mean.index, lo.values, hi.values, alpha=0.10, color=color)

    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    path = Path(output_dir) / f"pearl_{metric}_vs_rounds.{fmt}"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metric_vs_comm(
    results: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_dir: str | Path,
    fmt: str = "pdf",
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))

    for method, grp in results.groupby("method"):
        if metric not in grp.columns:
            continue
        mean_grp = grp.groupby("round")[["cum_bytes", metric]].mean().reset_index()
        color = METHOD_COLORS.get(method, "#333333")
        lw = METHOD_WIDTHS.get(method, 1.5)
        ax.plot(
            mean_grp["cum_bytes"] / 1e6,
            mean_grp[metric],
            label=method,
            color=color,
            linewidth=lw,
        )

    ax.set_xlabel("Cumulative communication (MB)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    path = Path(output_dir) / f"pearl_{metric}_vs_comm.{fmt}"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_selection_entropy(
    results: pd.DataFrame,
    output_dir: str | Path,
    fmt: str = "pdf",
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    df = results[results["method"] != "local_only"]

    for method, grp in df.groupby("method"):
        mean, _, _ = _mean_range_by_round(grp, "selection_entropy")
        color = METHOD_COLORS.get(method, "#333333")
        lw = METHOD_WIDTHS.get(method, 1.5)
        ax.plot(mean.index, mean.values, label=method, color=color, linewidth=lw)

    ax.axhline(
        1.0,
        color=METHOD_COLORS.get("random_peer", "#378ADD"),
        linestyle="--",
        linewidth=0.8,
        alpha=0.5,
        label="random_peer ceiling",
    )
    ax.set_xlabel("Round")
    ax.set_ylabel("Normalized selection entropy")
    ax.set_ylim(0, 1.1)
    ax.set_title("Peer-selection entropy")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    path = Path(output_dir) / f"pearl_selection_entropy.{fmt}"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_negative_transfer(
    results: pd.DataFrame,
    output_dir: str | Path,
    fmt: str = "pdf",
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    df = results[results["method"] != "local_only"]

    for method, grp in df.groupby("method"):
        mean, _, _ = _mean_range_by_round(grp, "neg_transfer_rate")
        color = METHOD_COLORS.get(method, "#333333")
        lw = METHOD_WIDTHS.get(method, 1.5)
        ax.plot(mean.index, mean.values, label=method, color=color, linewidth=lw)

    ax.set_xlabel("Round")
    ax.set_ylabel("Negative transfer rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Negative transfer rate over rounds")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    path = Path(output_dir) / f"pearl_neg_transfer.{fmt}"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_all(
    results: pd.DataFrame,
    output_dir: str | Path,
    fmt: str = "pdf",
) -> list[Path]:
    return [
        plot_metric_vs_rounds(
            results,
            "mean_global_macro_f1",
            "Mean macro-F1",
            "Macro-F1 vs rounds",
            output_dir,
            fmt,
        ),
        plot_metric_vs_rounds(
            results,
            "mean_global_accuracy",
            "Mean global accuracy",
            "Accuracy vs rounds",
            output_dir,
            fmt,
        ),
        plot_metric_vs_rounds(
            results,
            "worst_client_accuracy",
            "Worst-client accuracy",
            "Worst-client accuracy vs rounds",
            output_dir,
            fmt,
        ),
        plot_metric_vs_comm(
            results,
            "mean_global_macro_f1",
            "Mean macro-F1",
            "Macro-F1 vs communication",
            output_dir,
            fmt,
        ),
        plot_selection_entropy(results, output_dir, fmt),
        plot_negative_transfer(results, output_dir, fmt),
    ]


def _mean_range_by_round(grp: pd.DataFrame, metric: str):
    if "seed" in grp.columns:
        pivot = grp.pivot_table(index="round", columns="seed", values=metric)
        return pivot.mean(axis=1), pivot.min(axis=1), pivot.max(axis=1)
    mean = grp.groupby("round")[metric].mean()
    return mean, mean, mean
