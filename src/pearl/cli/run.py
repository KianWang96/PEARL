"""Run PEARL experiments from YAML configs."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from pearl.config import apply_overrides, load_config, save_config
from pearl.results import write_summary


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

    all_results = []
    for seed in seeds:
        for method in methods:
            run_path = runs_dir / f"{config.experiment_name}_seed{seed}_{_safe_name(method)}.csv"
            if args.resume and run_path.exists():
                print(f"Reusing existing run: {run_path}")
                df = pd.read_csv(run_path)
            else:
                from pearl.experiment import run_experiment

                print(f"\nRunning method={method} seed={seed}")
                run_cfg = replace(config, seed=seed, selection_method=method)
                df = run_experiment(run_cfg, verbose=not args.quiet)
                df.to_csv(run_path, index=False)
                print(f"Saved run CSV: {run_path}")
            all_results.append(df)

    if not all_results:
        print("No runs were executed.")
        return

    results_all = pd.concat(all_results, ignore_index=True)
    combined_path = output_dir / "results_all.csv"
    results_all.to_csv(combined_path, index=False)
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


def _safe_name(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in value
    )


if __name__ == "__main__":
    main()
