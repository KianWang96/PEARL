"""Run validated CIFAR-10 PEARL experiment suites with durable progress logs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pearl.config import ExperimentConfig, load_config
from pearl.data import get_dataset
from pearl.utils import resolve_device


BUDGET_MATCHED_CONFIGS = [
    "configs/cifar10/pearl_er150.yaml",
    "configs/cifar10/pearl_sf150.yaml",
    "configs/cifar10/pearl_ring150.yaml",
]

DECENTRALIZED_REFERENCE_CONFIGS = [
    "configs/cifar10/decentralized_references/dpsgd_full_er150.yaml",
    "configs/cifar10/decentralized_references/dpsgd_full_sf150.yaml",
    "configs/cifar10/decentralized_references/dpsgd_full_ring150.yaml",
]

SERVER_REFERENCE_CONFIGS = [
    "configs/cifar10/server_references150.yaml",
]

CORE_CONFIGS = [
    *BUDGET_MATCHED_CONFIGS,
    *DECENTRALIZED_REFERENCE_CONFIGS,
    *SERVER_REFERENCE_CONFIGS,
]

DYNAMIC_CONFIGS = [
    "configs/cifar10/dynamic/dropout_er_a08.yaml",
    "configs/cifar10/dynamic/dropout_er_a06.yaml",
    "configs/cifar10/dynamic/dropout_er_a04.yaml",
    "configs/cifar10/dynamic/staleness_er_r05.yaml",
    "configs/cifar10/dynamic/staleness_er_r10.yaml",
    "configs/cifar10/dynamic/staleness_er_r20.yaml",
]


class Reporter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("a", encoding="utf-8", buffering=1)

    def log(self, message: str = "") -> None:
        print(message, flush=True)
        self._stream.write(f"{message}\n")

    def child_line(self, line: str, step_stream) -> None:
        print(line, end="", flush=True)
        self._stream.write(line)
        step_stream.write(line)

    def close(self) -> None:
        self._stream.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=["core", "dynamic", "all"],
        default="core",
        help="core runs graph methods then decentralised and server references.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse only complete, valid method/seed CSVs (default: enabled).",
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to later configs after recording a failed experiment.",
    )
    parser.add_argument(
        "--skip-dataset-check",
        action="store_true",
        help="Do not pre-download/open CIFAR-10 before the suite starts.",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Do not run the synthetic all-method smoke matrix during preflight.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run validation, dataset, and smoke checks without experiments.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional suite log/status directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configs and print ordered commands without side effects.",
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Schedule all selected configs through the single-node GPU cluster runner.",
    )
    parser.add_argument(
        "--gpus",
        default="auto",
        help="Cluster GPU slots: auto, cpu, or a comma-separated list such as 0,1,2,3.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Cap cluster worker processes. Defaults to one worker per GPU slot.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configs = _configs_for_suite(args.suite)
    loaded_configs = _validate_config_matrix(configs)
    if args.cluster:
        return _run_cluster_suite(args, configs, loaded_configs)

    commands = [_command_for_config(config, args) for config in configs]

    if args.dry_run:
        _print_dry_run(args, configs, loaded_configs, commands)
        return 0

    run_dir = _resolve_run_dir(args.run_dir)
    reporter = Reporter(run_dir / "suite.log")
    status_path = run_dir / "status.json"
    status = _initial_status(args, configs, loaded_configs, run_dir)
    _write_status(status_path, status)
    suite_start = time.time()
    failures: list[str] = []

    try:
        _print_header(reporter, args, configs, loaded_configs, run_dir)
        status["state"] = "preflight"
        _write_status(status_path, status)
        _run_preflight(args, reporter, run_dir)

        if args.preflight_only:
            status["state"] = "preflight_complete"
            status["completed_at"] = _now()
            _write_status(status_path, status)
            reporter.log("Preflight completed successfully; experiments were not started.")
            return 0

        status["state"] = "running"
        _write_status(status_path, status)

        for index, (config, command) in enumerate(zip(configs, commands), start=1):
            entry = status["configs"][index - 1]
            entry["state"] = "running"
            entry["started_at"] = _now()
            _write_status(status_path, status)

            reporter.log("\n" + "-" * 78)
            reporter.log(f"[suite {index:02d}/{len(configs):02d}] {config}")
            reporter.log(f"Command: {_display_command(command)}")

            step_start = time.time()
            step_log = run_dir / f"{index:02d}_{Path(config).stem}.log"
            def record_child_pid(pid: int) -> None:
                entry["pid"] = pid
                _write_status(status_path, status)

            return_code, child_pid = _run_logged(
                command,
                reporter,
                step_log,
                on_start=record_child_pid,
            )
            elapsed = _format_duration(time.time() - step_start)
            entry["pid"] = child_pid
            entry["return_code"] = return_code
            entry["completed_at"] = _now()
            entry["elapsed"] = elapsed
            entry["log"] = str(step_log)

            if return_code == 0:
                entry["state"] = "complete"
                reporter.log(
                    f"[suite {index:02d}/{len(configs):02d}] complete in {elapsed}"
                )
                _write_status(status_path, status)
                continue

            entry["state"] = "failed"
            failures.append(config)
            reporter.log(
                f"[suite {index:02d}/{len(configs):02d}] FAILED "
                f"with exit code {return_code} after {elapsed}"
            )
            _write_status(status_path, status)
            if not args.continue_on_error:
                break

        reporter.log("\n" + "=" * 78)
        reporter.log(f"Suite elapsed: {_format_duration(time.time() - suite_start)}")
        status["completed_at"] = _now()
        if failures:
            status["state"] = "failed"
            status["failures"] = failures
            reporter.log(f"Failed configs: {', '.join(failures)}")
            _write_status(status_path, status)
            return 1

        status["state"] = "complete"
        _write_status(status_path, status)
        reporter.log("All requested configs completed successfully.")
        return 0
    except KeyboardInterrupt:
        status["state"] = "interrupted"
        status["completed_at"] = _now()
        _write_status(status_path, status)
        reporter.log("Suite interrupted by the user.")
        return 130
    except Exception as exc:  # noqa: BLE001 - persist unattended failures
        status["state"] = "failed"
        status["completed_at"] = _now()
        status["error"] = f"{type(exc).__name__}: {exc}"
        _write_status(status_path, status)
        reporter.log(f"Suite failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        reporter.close()


def _run_preflight(args, reporter: Reporter, run_dir: Path) -> None:
    reporter.log("Preflight: configuration matrix is valid and output paths are unique.")
    total, used, free = shutil.disk_usage(ROOT)
    reporter.log(
        "Preflight: disk "
        f"free={free / 2**30:.1f} GiB used={used / 2**30:.1f} GiB "
        f"total={total / 2**30:.1f} GiB"
    )
    if free < 2 * 2**30:
        raise RuntimeError("Less than 2 GiB of free disk space is available.")

    device = resolve_device("auto")
    reporter.log(f"Preflight: resolved training device is {device}.")

    if not args.skip_dataset_check:
        reporter.log("Preflight: opening CIFAR-10 (downloads it once if absent).")
        train, test, channels, size = _prepare_cifar10(reporter)
        reporter.log(
            f"Preflight: CIFAR-10 ready (train={len(train)}, test={len(test)}, "
            f"shape={channels}x{size}x{size})."
        )

    if not args.skip_smoke:
        reporter.log("Preflight: running the synthetic all-method smoke matrix.")
        smoke_log = run_dir / "preflight_smoke.log"
        command = [sys.executable, "-u", str(ROOT / "scripts" / "smoke_all_methods.py")]
        return_code, _ = _run_logged(command, reporter, smoke_log)
        if return_code != 0:
            raise RuntimeError(
                f"All-method smoke matrix failed with exit code {return_code}."
            )
        reporter.log("Preflight: all-method smoke matrix passed.")


def _run_cluster_suite(
    args,
    configs: list[str],
    loaded_configs: list[ExperimentConfig],
) -> int:
    run_dir = _resolve_run_dir(args.run_dir)
    if args.dry_run:
        return _invoke_cluster_runner(args, configs, run_dir)

    reporter = Reporter(run_dir / "suite_preflight.log")
    try:
        _print_header(reporter, args, configs, loaded_configs, run_dir)
        reporter.log("Cluster mode: running suite preflight before scheduling tasks.")
        _run_preflight(args, reporter, run_dir)
        if args.preflight_only:
            reporter.log("Preflight completed successfully; cluster tasks were not started.")
            return 0
    finally:
        reporter.close()

    return _invoke_cluster_runner(args, configs, run_dir)


def _invoke_cluster_runner(args, configs: list[str], run_dir: Path) -> int:
    from pearl.cli.cluster_run import main as cluster_main

    cluster_args: list[str] = []
    for config in configs:
        cluster_args.extend(["--config", str(ROOT / config)])
    cluster_args.extend(["--gpus", args.gpus, "--run-dir", str(run_dir)])
    if args.max_workers is not None:
        cluster_args.extend(["--max-workers", str(args.max_workers)])
    cluster_args.append("--resume" if args.resume else "--no-resume")
    if args.skip_plots:
        cluster_args.append("--skip-plots")
    if args.quiet:
        cluster_args.append("--quiet")
    if args.continue_on_error:
        cluster_args.append("--continue-on-error")
    if args.dry_run:
        cluster_args.append("--dry-run")
    if args.skip_dataset_check:
        cluster_args.append("--skip-dataset-preflight")

    previous_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        return cluster_main(cluster_args)
    finally:
        os.chdir(previous_cwd)


def _prepare_cifar10(reporter: Reporter):
    archive = ROOT / "data" / "cifar-10-python.tar.gz"
    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(120)
    try:
        for attempt in range(1, 3):
            try:
                train, test, _, channels, size = get_dataset(
                    "cifar10",
                    data_dir=ROOT / "data",
                    train_subset=1,
                    test_subset=1,
                    download=True,
                )
                return train, test, channels, size
            except (OSError, RuntimeError) as exc:
                if archive.exists() and archive.stat().st_size == 0:
                    archive.unlink()
                if attempt == 2:
                    raise RuntimeError(
                        "CIFAR-10 could not be opened after two attempts: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                reporter.log(
                    "Preflight: CIFAR-10 attempt 1 failed; retrying once "
                    f"({type(exc).__name__}: {exc})."
                )
    finally:
        socket.setdefaulttimeout(previous_timeout)


def _validate_config_matrix(configs: list[str]) -> list[ExperimentConfig]:
    loaded: list[ExperimentConfig] = []
    for relative_path in configs:
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing suite config: {path}")
        loaded.append(load_config(path))

    names = [config.experiment_name for config in loaded]
    outputs = [str((ROOT / config.output_dir).resolve()) for config in loaded]
    if len(names) != len(set(names)):
        raise ValueError("Suite experiment_name values must be unique.")
    if len(outputs) != len(set(outputs)):
        raise ValueError("Suite output_dir values must be unique.")
    return loaded


def _command_for_config(config: str, args) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "run.py"),
        "--config",
        str(ROOT / config),
    ]
    if args.resume:
        command.append("--resume")
    if args.skip_plots:
        command.append("--skip-plots")
    if args.quiet:
        command.append("--quiet")
    return command


def _run_logged(
    command: list[str],
    reporter: Reporter,
    log_path: Path,
    on_start=None,
) -> tuple[int, int]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as step_stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if on_start is not None:
            on_start(process.pid)
        try:
            if process.stdout is None:
                raise RuntimeError("Subprocess stdout was not captured.")
            for line in process.stdout:
                reporter.child_line(line, step_stream)
            return process.wait(), process.pid
        except KeyboardInterrupt:
            process.terminate()
            process.wait(timeout=30)
            raise


def _initial_status(
    args,
    configs: list[str],
    loaded: list[ExperimentConfig],
    run_dir: Path,
) -> dict:
    return {
        "suite": args.suite,
        "state": "created",
        "pid": os.getpid(),
        "started_at": _now(),
        "run_dir": str(run_dir),
        "resume": args.resume,
        "configs": [
            {
                "path": path,
                "experiment_name": config.experiment_name,
                "output_dir": config.output_dir,
                "runs": len(config.methods) * len(config.seeds),
                "state": "pending",
            }
            for path, config in zip(configs, loaded)
        ],
    }


def _write_status(path: Path, status: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".json.tmp")
    with temp_path.open("w", encoding="utf-8") as stream:
        json.dump(status, stream, indent=2)
        stream.write("\n")
    temp_path.replace(path)


def _print_header(
    reporter: Reporter,
    args,
    configs: list[str],
    loaded: list[ExperimentConfig],
    run_dir: Path,
) -> None:
    total_runs = sum(len(config.methods) * len(config.seeds) for config in loaded)
    reporter.log("=" * 78)
    reporter.log(f"CIFAR-10 suite : {args.suite}")
    reporter.log(f"Configs        : {len(configs)}")
    reporter.log(f"Method/seeds   : {total_runs}")
    reporter.log(f"Resume         : {args.resume}")
    reporter.log(f"Workspace      : {ROOT}")
    reporter.log(f"Run log/status : {run_dir}")
    reporter.log("=" * 78)


def _print_dry_run(args, configs, loaded, commands) -> None:
    total_runs = sum(len(config.methods) * len(config.seeds) for config in loaded)
    print("=" * 78)
    print(f"CIFAR-10 suite : {args.suite}")
    print(f"Configs        : {len(configs)}")
    print(f"Method/seeds   : {total_runs}")
    print(f"Resume         : {args.resume}")
    print("=" * 78)
    for index, (config, command) in enumerate(zip(configs, commands), start=1):
        print(f"[{index:02d}/{len(configs):02d}] {config}")
        print(f"  {_display_command(command)}")


def _configs_for_suite(suite: str) -> list[str]:
    if suite == "core":
        return CORE_CONFIGS.copy()
    if suite == "dynamic":
        return DYNAMIC_CONFIGS.copy()
    return [*CORE_CONFIGS, *DYNAMIC_CONFIGS]


def _resolve_run_dir(requested: str | None) -> Path:
    if requested:
        path = Path(requested)
        return path if path.is_absolute() else ROOT / path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "results" / "cifar10" / "_suite_runs" / stamp


def _display_command(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
