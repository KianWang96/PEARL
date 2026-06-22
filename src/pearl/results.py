"""Result summarization utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def final_round_rows(results: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["method"]
    if "seed" in results.columns:
        group_cols = ["seed", "method"]
    return results.sort_values("round").groupby(group_cols).tail(1)


def final_round_summary(results: pd.DataFrame) -> pd.DataFrame:
    final_rows = final_round_rows(results)
    aggregations = {
        "mean_acc": ("mean_global_accuracy", "mean"),
        "std_acc": ("mean_global_accuracy", "std"),
        "mean_f1": ("mean_global_macro_f1", "mean"),
        "std_f1": ("mean_global_macro_f1", "std"),
        "mean_worst": ("worst_client_accuracy", "mean"),
        "std_worst": ("worst_client_accuracy", "std"),
        "mean_neg_xfer": ("neg_transfer_rate", "mean"),
        "std_neg_xfer": ("neg_transfer_rate", "std"),
        "mean_entropy": ("selection_entropy", "mean"),
        "std_entropy": ("selection_entropy", "std"),
    }
    optional = {
        "mean_active_fraction": ("active_fraction", "mean"),
        "mean_active_degree": ("mean_active_degree", "mean"),
        "mean_no_active_peer": ("no_active_peer_fraction", "mean"),
        "mean_descriptor_age": ("descriptor_age", "mean"),
        "mean_cum_bytes": ("cum_bytes", "mean"),
    }
    aggregations.update(
        {
            name: aggregation
            for name, aggregation in optional.items()
            if aggregation[0] in final_rows.columns
        }
    )
    summary = (
        final_rows.groupby("method")
        .agg(**aggregations)
        .reset_index()
        .sort_values("mean_f1", ascending=False)
    )
    return summary


def write_summary(results: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    summary = final_round_summary(results)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(summary, path)
    return summary


def write_csv_atomic(data: pd.DataFrame, output_path: str | Path) -> None:
    """Write a CSV through a same-directory temporary file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    data.to_csv(temp_path, index=False)
    temp_path.replace(path)
