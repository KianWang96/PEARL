"""Run one tiny synthetic round through every implemented experiment method."""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import torch
from torch.utils.data import TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pearl.experiment as experiment_module
from pearl.config import ExperimentConfig
from pearl.constants import SERVER_METHODS, SUPPORTED_METHODS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=sorted(SUPPORTED_METHODS),
        help="Optional subset of method identifiers to exercise.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    unknown = sorted(set(args.methods) - SUPPORTED_METHODS)
    if unknown:
        raise ValueError(f"Unknown smoke-test method(s): {', '.join(unknown)}")
    warnings.filterwarnings(
        "ignore",
        message="y_pred contains classes not in y_true",
        category=UserWarning,
    )

    generator = torch.Generator().manual_seed(123)
    train = TensorDataset(
        torch.rand(12, 1, 28, 28, generator=generator),
        torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]),
    )
    test = TensorDataset(
        torch.rand(6, 1, 28, 28, generator=generator),
        torch.tensor([0, 1, 2, 0, 1, 2]),
    )
    original_get_dataset = experiment_module.get_dataset
    experiment_module.get_dataset = lambda *args, **kwargs: (
        train,
        test,
        3,
        1,
        28,
    )

    failures: list[tuple[str, str]] = []
    started = time.time()
    try:
        for index, method in enumerate(args.methods, start=1):
            method_start = time.time()
            print(
                f"[smoke {index:02d}/{len(args.methods):02d}] {method}",
                flush=True,
            )
            try:
                config = ExperimentConfig(
                    selection_method=method,
                    methods=[method],
                    dataset="mnist",
                    graph_type="server" if method in SERVER_METHODS else "ring",
                    num_clients=3,
                    partition="iid",
                    rounds=1,
                    local_epochs=1,
                    batch_size=2,
                    num_workers=0,
                    pin_memory=False,
                    latent_dim=4,
                    model_width=2,
                    anchor_size=2,
                    descriptor_refresh_period=1,
                    eval_every=1,
                )
                result = experiment_module.run_experiment(config, verbose=False)
                if result.empty or set(result["method"]) != {method}:
                    raise RuntimeError("Smoke run returned an invalid result frame.")
                print(
                    f"[smoke {index:02d}/{len(args.methods):02d}] passed "
                    f"in {_format_duration(time.time() - method_start)}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - report the complete matrix
                failures.append((method, f"{type(exc).__name__}: {exc}"))
                print(
                    f"[smoke {index:02d}/{len(args.methods):02d}] FAILED: {exc}",
                    flush=True,
                )
    finally:
        experiment_module.get_dataset = original_get_dataset

    print(f"Smoke matrix elapsed: {_format_duration(time.time() - started)}")
    if failures:
        for method, error in failures:
            print(f"FAILED {method}: {error}")
        return 1
    print(f"All {len(args.methods)} methods passed the synthetic smoke matrix.")
    return 0


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


if __name__ == "__main__":
    raise SystemExit(main())
