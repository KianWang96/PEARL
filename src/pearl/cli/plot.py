"""Generate PEARL result figures from an existing CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pearl.visualization import plot_all


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", help="Path to results_all.csv.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Figure output directory. Defaults to figures next to input.",
    )
    parser.add_argument(
        "--format",
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Figure format.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input_csv)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else input_path.parent / "figures"
    )
    results = pd.read_csv(input_path)
    paths = plot_all(results, output_dir, fmt=args.format)
    for path in paths:
        print(f"Saved figure: {path}")


if __name__ == "__main__":
    main()
