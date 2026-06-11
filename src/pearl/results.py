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
    summary = (
        final_rows.groupby("method")
        .agg(
            mean_acc=("mean_global_accuracy", "mean"),
            std_acc=("mean_global_accuracy", "std"),
            mean_f1=("mean_global_macro_f1", "mean"),
            std_f1=("mean_global_macro_f1", "std"),
            mean_worst=("worst_client_accuracy", "mean"),
            std_worst=("worst_client_accuracy", "std"),
            mean_neg_xfer=("neg_transfer_rate", "mean"),
            std_neg_xfer=("neg_transfer_rate", "std"),
            mean_entropy=("selection_entropy", "mean"),
            std_entropy=("selection_entropy", "std"),
        )
        .reset_index()
        .sort_values("mean_f1", ascending=False)
    )
    return summary


def write_summary(results: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    summary = final_round_summary(results)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return summary
