"""Summarize final-round PEARL results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pearl.results import write_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", help="Path to results_all.csv.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output summary CSV path. Defaults to summary_final.csv next to input.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input_csv)
    output_path = (
        Path(args.output)
        if args.output is not None
        else input_path.with_name("summary_final.csv")
    )
    results = pd.read_csv(input_path)
    summary = write_summary(results, output_path)
    print(f"Saved summary CSV: {output_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
