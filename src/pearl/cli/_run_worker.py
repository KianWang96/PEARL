"""Private one-run worker used by the cluster launcher."""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

from pearl.cli.run import _format_duration, _safe_name
from pearl.config import apply_overrides, load_config
from pearl.experiment import run_experiment
from pearl.results import write_csv_atomic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override any flat config key, for example --set rounds=10.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    config = apply_overrides(config, args.overrides)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)

    run_cfg = replace(
        config,
        seeds=[args.seed],
        methods=[args.method],
        seed=args.seed,
        selection_method=args.method,
    )
    run_path = _run_path(run_cfg.output_dir, run_cfg.experiment_name, args.seed, args.method)
    start = time.time()
    print(
        f"Worker start: experiment={run_cfg.experiment_name} "
        f"method={args.method} seed={args.seed} output={run_path}",
        flush=True,
    )
    df = run_experiment(run_cfg, verbose=not args.quiet)
    write_csv_atomic(df, run_path)
    print(
        f"Worker complete: {run_path} elapsed={_format_duration(time.time() - start)}",
        flush=True,
    )


def _run_path(
    output_dir: str | Path,
    experiment_name: str,
    seed: int,
    method: str,
) -> Path:
    return (
        Path(output_dir)
        / "runs"
        / f"{experiment_name}_seed{seed}_{_safe_name(method)}.csv"
    )


if __name__ == "__main__":
    main()
