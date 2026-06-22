"""Run PEARL experiments from YAML configs."""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from pearl.config import apply_overrides, load_config, save_config
from pearl.results import write_csv_atomic, write_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/paper_er150.yaml",
        help="Path to a flat YAML experiment config.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output_dir from the config.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Override methods from the config.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Override seeds from the config.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override any flat config key, for example --set rounds=10.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing per-method CSV files instead of rerunning them.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip automatic figure generation.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce per-round logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    config = apply_overrides(config, args.overrides)

    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)

    seeds = args.seeds or config.seeds
    methods = args.methods or config.methods
    config = replace(config, seeds=seeds, methods=methods)

    output_dir = Path(config.output_dir)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "resolved_config.yaml")

    total_runs = len(seeds) * len(methods)
    suite_start = time.time()
    print("=" * 78)
    print(f"Experiment : {config.experiment_name}")
    print(f"Dataset    : {config.dataset}")
    print(f"Graph      : {config.graph_type}")
    print(f"Runs       : {total_runs} ({len(methods)} methods x {len(seeds)} seeds)")
    print(f"Output     : {output_dir}")
    print("=" * 78, flush=True)

    all_results = []
    run_index = 0
    for seed in seeds:
        for method in methods:
            run_index += 1
            run_start = time.time()
            run_path = runs_dir / f"{config.experiment_name}_seed{seed}_{_safe_name(method)}.csv"
            print(
                f"\n[run {run_index:02d}/{total_runs:02d}] "
                f"method={method} seed={seed}",
                flush=True,
            )
            if args.resume and run_path.exists():
                df = _load_complete_run(
                    run_path,
                    expected_rounds={
                        round_idx
                        for round_idx in range(config.rounds)
                        if round_idx % config.eval_every == 0
                        or round_idx == config.rounds - 1
                    },
                    expected_seed=seed,
                    expected_method=method,
                )
                if df is not None:
                    print(f"Reusing complete run: {run_path}")
                else:
                    print(f"Existing run is incomplete or invalid; rerunning: {run_path}")
            else:
                df = None

            if df is None:
                from pearl.experiment import run_experiment

                run_cfg = replace(config, seed=seed, selection_method=method)
                df = run_experiment(run_cfg, verbose=not args.quiet)
                write_csv_atomic(df, run_path)
                print(f"Saved run CSV: {run_path}")
            all_results.append(df)
            partial_path = output_dir / "results_partial.csv"
            write_csv_atomic(pd.concat(all_results, ignore_index=True), partial_path)
            elapsed = time.time() - run_start
            total_elapsed = time.time() - suite_start
            print(
                f"[run {run_index:02d}/{total_runs:02d}] complete | "
                f"run={_format_duration(elapsed)} | "
                f"total={_format_duration(total_elapsed)}",
                flush=True,
            )

    if not all_results:
        print("No runs were executed.")
        return

    results_all = pd.concat(all_results, ignore_index=True)
    combined_path = output_dir / "results_all.csv"
    write_csv_atomic(results_all, combined_path)
    print(f"Saved combined CSV: {combined_path}")

    summary_path = output_dir / "summary_final.csv"
    summary = write_summary(results_all, summary_path)
    print(f"Saved summary CSV: {summary_path}")
    print(summary.to_string(index=False))

    if not args.skip_plots:
        from pearl.visualization import plot_all

        figure_paths = plot_all(
            results_all,
            output_dir / "figures",
            fmt=config.plot_format,
        )
        for path in figure_paths:
            print(f"Saved figure: {path}")

    print(
        f"Experiment complete in {_format_duration(time.time() - suite_start)}.",
        flush=True,
    )


def _safe_name(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in value
    )


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _load_complete_run(
    path: Path,
    expected_rounds: set[int],
    expected_seed: int,
    expected_method: str,
) -> pd.DataFrame | None:
    try:
        data = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None

    required = {"round", "seed", "method"}
    if data.empty or not required <= set(data.columns):
        return None
    try:
        actual_rounds = set(pd.to_numeric(data["round"]).astype(int))
        actual_seeds = set(pd.to_numeric(data["seed"]).astype(int))
        actual_methods = set(data["method"].astype(str))
    except (TypeError, ValueError):
        return None
    if actual_rounds != expected_rounds:
        return None
    if actual_seeds != {expected_seed} or actual_methods != {expected_method}:
        return None
    return data


if __name__ == "__main__":
    main()
