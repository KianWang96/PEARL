"""Run PEARL method/seed tasks concurrently across a single multi-GPU node."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pearl.cli._run_worker import _run_path
from pearl.cli.run import _format_duration, _load_complete_run
from pearl.config import ExperimentConfig, apply_overrides, load_config, save_config
from pearl.data import get_dataset
from pearl.results import write_csv_atomic, write_summary


@dataclass(frozen=True)
class ConfigEntry:
    path: Path
    config: ExperimentConfig


@dataclass(frozen=True)
class ClusterTask:
    index: int
    config_path: Path
    experiment_name: str
    output_dir: Path
    seed: int
    method: str
    rounds: int
    eval_every: int
    plot_format: str

    @property
    def run_path(self) -> Path:
        return _run_path(self.output_dir, self.experiment_name, self.seed, self.method)

    @property
    def task_name(self) -> str:
        return f"{self.experiment_name}_seed{self.seed}_{self.method}"

    @property
    def expected_rounds(self) -> set[int]:
        return {
            round_idx
            for round_idx in range(self.rounds)
            if round_idx % self.eval_every == 0 or round_idx == self.rounds - 1
        }


class Reporter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("a", encoding="utf-8", buffering=1)

    def log(self, message: str = "") -> None:
        print(message, flush=True)
        self._stream.write(f"{message}\n")

    def close(self) -> None:
        self._stream.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="Experiment config path. Repeat to schedule multiple configs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output_dir. Only valid with one --config.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Override methods from every selected config.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Override seeds from every selected config.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override any flat config key, for example --set rounds=10.",
    )
    parser.add_argument(
        "--gpus",
        default="auto",
        help="GPU slots to use: auto, cpu, or a comma-separated list such as 0,1,2,3.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Cap concurrent worker processes. Defaults to one worker per GPU slot.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse only complete per-method/seed CSVs (default: enabled).",
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue scheduling tasks after a worker fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configs and print the task plan without side effects.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional cluster log/status directory.",
    )
    parser.add_argument(
        "--skip-dataset-preflight",
        action="store_true",
        help="Do not open/download datasets before launching workers.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.output_dir is not None and len(args.config) != 1:
        parser.error("--output-dir is only valid when exactly one --config is provided.")

    try:
        entries = load_config_entries(
            args.config,
            output_dir=args.output_dir,
            methods=args.methods,
            seeds=args.seeds,
            overrides=args.overrides,
        )
        tasks = expand_tasks(entries)
        gpu_slots = parse_gpu_slots(args.gpus, args.max_workers)
    except Exception as exc:  # noqa: BLE001 - argparse should surface config failures
        parser.error(str(exc))

    if args.dry_run:
        _print_dry_run(args, entries, tasks, gpu_slots)
        return 0

    return _run_cluster(args, entries, tasks, gpu_slots)


def load_config_entries(
    config_paths: list[str],
    *,
    output_dir: str | None = None,
    methods: list[str] | None = None,
    seeds: list[int] | None = None,
    overrides: list[str] | None = None,
) -> list[ConfigEntry]:
    entries: list[ConfigEntry] = []
    for raw_path in config_paths:
        path = Path(raw_path)
        config = load_config(path)
        config = apply_overrides(config, overrides)
        if output_dir is not None:
            config = replace(config, output_dir=output_dir)
        if methods is not None:
            config = replace(config, methods=list(methods))
        if seeds is not None:
            config = replace(config, seeds=list(seeds))
        entries.append(ConfigEntry(path=path, config=config))

    output_dirs = [Path(entry.config.output_dir).resolve() for entry in entries]
    duplicates = _duplicates([str(path) for path in output_dirs])
    if duplicates:
        raise ValueError(
            "Cluster configs must write to unique output_dir values: "
            + ", ".join(duplicates)
        )
    return entries


def expand_tasks(entries: list[ConfigEntry]) -> list[ClusterTask]:
    tasks: list[ClusterTask] = []
    for entry in entries:
        for seed in entry.config.seeds:
            for method in entry.config.methods:
                tasks.append(
                    ClusterTask(
                        index=len(tasks),
                        config_path=entry.path,
                        experiment_name=entry.config.experiment_name,
                        output_dir=Path(entry.config.output_dir),
                        seed=seed,
                        method=method,
                        rounds=entry.config.rounds,
                        eval_every=entry.config.eval_every,
                        plot_format=entry.config.plot_format,
                    )
                )
    return tasks


def parse_gpu_slots(
    spec: str,
    max_workers: int | None = None,
    *,
    cuda_device_count: int | None = None,
) -> list[str]:
    if max_workers is not None and max_workers < 1:
        raise ValueError("--max-workers must be at least 1.")

    normalized = spec.strip().lower()
    if normalized == "cpu":
        return ["cpu"] * (max_workers or 1)

    if normalized == "auto":
        count = _cuda_device_count() if cuda_device_count is None else cuda_device_count
        slots = [str(index) for index in range(count)] if count > 0 else ["cpu"]
    else:
        slots = [item.strip() for item in spec.split(",") if item.strip()]
        if not slots:
            raise ValueError("--gpus must be auto, cpu, or a comma-separated list.")

    if max_workers is not None:
        slots = slots[: min(max_workers, len(slots))]
    return slots


def _run_cluster(
    args: argparse.Namespace,
    entries: list[ConfigEntry],
    tasks: list[ClusterTask],
    gpu_slots: list[str],
) -> int:
    run_dir = _resolve_run_dir(args.run_dir)
    reporter = Reporter(run_dir / "cluster.log")
    status_path = run_dir / "status.json"
    status = _initial_status(args, entries, tasks, gpu_slots, run_dir)
    _write_status(status_path, status)

    try:
        _print_header(reporter, entries, tasks, gpu_slots, run_dir)
        for entry in entries:
            output_dir = Path(entry.config.output_dir)
            (output_dir / "runs").mkdir(parents=True, exist_ok=True)
            save_config(entry.config, output_dir / "resolved_config.yaml")

        status["state"] = "preflight"
        _write_status(status_path, status)
        if args.skip_dataset_preflight:
            reporter.log("Preflight: dataset check skipped by request.")
        else:
            _preflight_datasets(entries, reporter)

        status["state"] = "running"
        _write_status(status_path, status)
        result = _schedule_tasks(args, tasks, gpu_slots, run_dir, reporter, status_path, status)

        status["state"] = "failed" if result else "complete"
        status["completed_at"] = _now()
        _write_status(status_path, status)
        reporter.log("\n" + "=" * 78)
        if result:
            reporter.log(f"Cluster run finished with failures. See {run_dir}")
        else:
            reporter.log("Cluster run completed successfully.")
        return result
    except KeyboardInterrupt:
        status["state"] = "interrupted"
        status["completed_at"] = _now()
        _write_status(status_path, status)
        reporter.log("Cluster run interrupted by the user.")
        return 130
    except Exception as exc:  # noqa: BLE001 - persist unattended failures
        status["state"] = "failed"
        status["completed_at"] = _now()
        status["error"] = f"{type(exc).__name__}: {exc}"
        _write_status(status_path, status)
        reporter.log(f"Cluster run failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        reporter.close()


def _schedule_tasks(
    args: argparse.Namespace,
    tasks: list[ClusterTask],
    gpu_slots: list[str],
    run_dir: Path,
    reporter: Reporter,
    status_path: Path,
    status: dict[str, Any],
) -> int:
    pending: list[ClusterTask] = []
    completed_frames: dict[int, pd.DataFrame] = {}
    failed_task_indices: set[int] = set()
    available_slots = list(gpu_slots)
    running: dict[int, dict[str, Any]] = {}
    stop_launching = False

    for task in tasks:
        record = status["tasks"][task.index]
        if args.resume and task.run_path.exists():
            df = _load_complete_run(
                task.run_path,
                task.expected_rounds,
                task.seed,
                task.method,
            )
            if df is not None:
                completed_frames[task.index] = df
                record["state"] = "reused"
                record["completed_at"] = _now()
                reporter.log(f"Reusing complete run: {task.run_path}")
                continue
            reporter.log(f"Existing run is incomplete or invalid; rerunning: {task.run_path}")
        pending.append(task)

    _write_all_partials(tasks, completed_frames)
    _write_status(status_path, status)

    if not pending:
        reporter.log("All tasks were satisfied by resume; writing final outputs.")

    try:
        while pending or running:
            while pending and available_slots and not stop_launching:
                task = pending.pop(0)
                gpu = available_slots.pop(0)
                process_info = _start_task(args, task, gpu, run_dir, reporter)
                running[process_info["pid"]] = process_info
                record = status["tasks"][task.index]
                record.update(
                    {
                        "state": "running",
                        "gpu": gpu,
                        "pid": process_info["pid"],
                        "log": str(process_info["log_path"]),
                        "started_at": process_info["started_at"],
                    }
                )
                _write_status(status_path, status)

            completed_pids: list[int] = []
            for pid, process_info in running.items():
                process = process_info["process"]
                return_code = process.poll()
                if return_code is None:
                    continue

                completed_pids.append(pid)
                process_info["stream"].close()
                task = process_info["task"]
                gpu = process_info["gpu"]
                available_slots.append(gpu)
                elapsed = time.time() - process_info["started"]
                record = status["tasks"][task.index]
                record["return_code"] = return_code
                record["completed_at"] = _now()
                record["elapsed"] = _format_duration(elapsed)

                if return_code == 0:
                    df = _load_complete_run(
                        task.run_path,
                        task.expected_rounds,
                        task.seed,
                        task.method,
                    )
                    if df is not None:
                        completed_frames[task.index] = df
                        record["state"] = "complete"
                        reporter.log(
                            f"Complete: {task.task_name} gpu={gpu} "
                            f"elapsed={_format_duration(elapsed)}"
                        )
                        _write_partial_for_task_output(tasks, completed_frames, task)
                    else:
                        failed_task_indices.add(task.index)
                        record["state"] = "failed"
                        record["error"] = (
                            "Worker exited successfully but did not write a complete run CSV."
                        )
                        reporter.log(f"FAILED validation: {task.task_name}")
                else:
                    failed_task_indices.add(task.index)
                    record["state"] = "failed"
                    reporter.log(
                        f"FAILED: {task.task_name} gpu={gpu} "
                        f"exit={return_code} elapsed={_format_duration(elapsed)}"
                    )

                if failed_task_indices and not args.continue_on_error:
                    stop_launching = True
                _write_status(status_path, status)

            for pid in completed_pids:
                del running[pid]

            if not completed_pids and running:
                time.sleep(0.5)

            if stop_launching and not running:
                break
    except KeyboardInterrupt:
        _terminate_running(running, reporter)
        raise

    if stop_launching and pending:
        for task in pending:
            status["tasks"][task.index]["state"] = "skipped"
        _write_status(status_path, status)

    _write_final_outputs(
        args,
        tasks,
        completed_frames,
        failed_task_indices,
        reporter,
    )
    return 1 if failed_task_indices or pending else 0


def _terminate_running(running: dict[int, dict[str, Any]], reporter: Reporter) -> None:
    for process_info in running.values():
        process = process_info["process"]
        task = process_info["task"]
        reporter.log(f"Terminating running worker: {task.task_name} pid={process.pid}")
        process.terminate()
    for process_info in running.values():
        process = process_info["process"]
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
        process_info["stream"].close()


def _start_task(
    args: argparse.Namespace,
    task: ClusterTask,
    gpu: str,
    run_dir: Path,
    reporter: Reporter,
) -> dict[str, Any]:
    log_path = (
        run_dir
        / "tasks"
        / f"{task.index + 1:04d}_{_file_safe(task.experiment_name)}"
        f"_seed{task.seed}_{_file_safe(task.method)}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("a", encoding="utf-8", buffering=1)
    env = _worker_env(gpu)
    command = _worker_command(args, task)
    started = time.time()
    reporter.log(
        f"Start: {task.task_name} gpu={gpu} log={log_path} "
        f"command={_display_command(command)}"
    )
    process = subprocess.Popen(
        command,
        cwd=Path.cwd(),
        env=env,
        stdout=stream,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    return {
        "process": process,
        "stream": stream,
        "pid": process.pid,
        "task": task,
        "gpu": gpu,
        "log_path": log_path,
        "started": started,
        "started_at": _now(),
    }


def _worker_command(args: argparse.Namespace, task: ClusterTask) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "pearl.cli._run_worker",
        "--config",
        str(task.config_path),
        "--method",
        task.method,
        "--seed",
        str(task.seed),
        "--output-dir",
        str(task.output_dir),
    ]
    for override in args.overrides:
        command.extend(["--set", override])
    if args.quiet:
        command.append("--quiet")
    return command


def _worker_env(gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_DEVICE_ORDER"] = env.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if gpu == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = gpu

    src_dir = Path(__file__).resolve().parents[2]
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(src_dir) if not existing else str(src_dir) + os.pathsep + existing
    )
    return env


def _preflight_datasets(entries: list[ConfigEntry], reporter: Reporter) -> None:
    seen: set[tuple[str, str, bool]] = set()
    for entry in entries:
        config = entry.config
        key = (config.dataset, str(config.data_dir), config.download)
        if key in seen:
            continue
        seen.add(key)
        reporter.log(
            f"Preflight: opening {config.dataset} from {config.data_dir} "
            f"(download={config.download})."
        )
        train_ds, test_ds, _, channels, size = get_dataset(
            config.dataset,
            data_dir=config.data_dir,
            train_subset=1,
            test_subset=1,
            download=config.download,
        )
        reporter.log(
            f"Preflight: {config.dataset} ready "
            f"(train={len(train_ds)}, test={len(test_ds)}, shape={channels}x{size}x{size})."
        )


def _write_partial_for_task_output(
    tasks: list[ClusterTask],
    completed_frames: dict[int, pd.DataFrame],
    task: ClusterTask,
) -> None:
    matching = [
        item
        for item in tasks
        if item.output_dir.resolve() == task.output_dir.resolve()
        and item.index in completed_frames
    ]
    if not matching:
        return
    data = pd.concat(
        [completed_frames[item.index] for item in sorted(matching, key=lambda item: item.index)],
        ignore_index=True,
    )
    write_csv_atomic(data, task.output_dir / "results_partial.csv")


def _write_all_partials(
    tasks: list[ClusterTask],
    completed_frames: dict[int, pd.DataFrame],
) -> None:
    for output_dir in sorted({task.output_dir.resolve() for task in tasks}):
        output_tasks = [task for task in tasks if task.output_dir.resolve() == output_dir]
        matching = [task for task in output_tasks if task.index in completed_frames]
        if not matching:
            continue
        data = pd.concat(
            [completed_frames[task.index] for task in sorted(matching, key=lambda item: item.index)],
            ignore_index=True,
        )
        write_csv_atomic(data, output_tasks[0].output_dir / "results_partial.csv")


def _write_final_outputs(
    args: argparse.Namespace,
    tasks: list[ClusterTask],
    completed_frames: dict[int, pd.DataFrame],
    failed_task_indices: set[int],
    reporter: Reporter,
) -> None:
    tasks_by_output: dict[Path, list[ClusterTask]] = defaultdict(list)
    for task in tasks:
        tasks_by_output[task.output_dir.resolve()].append(task)

    for _, output_tasks in sorted(tasks_by_output.items(), key=lambda item: str(item[0])):
        output_dir = output_tasks[0].output_dir
        incomplete = [
            task
            for task in output_tasks
            if task.index not in completed_frames or task.index in failed_task_indices
        ]
        if incomplete:
            reporter.log(
                f"Skipping final outputs for {output_dir}; "
                f"{len(incomplete)} task(s) are incomplete."
            )
            continue

        results_all = pd.concat(
            [
                completed_frames[task.index]
                for task in sorted(output_tasks, key=lambda item: item.index)
            ],
            ignore_index=True,
        )
        combined_path = output_dir / "results_all.csv"
        write_csv_atomic(results_all, combined_path)
        reporter.log(f"Saved combined CSV: {combined_path}")

        summary_path = output_dir / "summary_final.csv"
        write_summary(results_all, summary_path)
        reporter.log(f"Saved summary CSV: {summary_path}")

        if not args.skip_plots:
            from pearl.visualization import plot_all

            for path in plot_all(
                results_all,
                output_dir / "figures",
                fmt=output_tasks[0].plot_format,
            ):
                reporter.log(f"Saved figure: {path}")


def _initial_status(
    args: argparse.Namespace,
    entries: list[ConfigEntry],
    tasks: list[ClusterTask],
    gpu_slots: list[str],
    run_dir: Path,
) -> dict[str, Any]:
    return {
        "state": "created",
        "started_at": _now(),
        "run_dir": str(run_dir),
        "gpus": gpu_slots,
        "max_workers": len(gpu_slots),
        "resume": args.resume,
        "continue_on_error": args.continue_on_error,
        "skip_dataset_preflight": args.skip_dataset_preflight,
        "configs": [
            {
                "path": str(entry.path),
                "experiment_name": entry.config.experiment_name,
                "output_dir": entry.config.output_dir,
                "runs": len(entry.config.methods) * len(entry.config.seeds),
            }
            for entry in entries
        ],
        "tasks": [
            {
                "index": task.index,
                "state": "pending",
                "config": str(task.config_path),
                "experiment_name": task.experiment_name,
                "output_dir": str(task.output_dir),
                "seed": task.seed,
                "method": task.method,
                "run_path": str(task.run_path),
            }
            for task in tasks
        ],
    }


def _print_header(
    reporter: Reporter,
    entries: list[ConfigEntry],
    tasks: list[ClusterTask],
    gpu_slots: list[str],
    run_dir: Path,
) -> None:
    reporter.log("=" * 78)
    reporter.log("PEARL cluster run")
    reporter.log(f"Configs      : {len(entries)}")
    reporter.log(f"Tasks        : {len(tasks)}")
    reporter.log(f"Workers      : {len(gpu_slots)}")
    reporter.log(f"GPU slots    : {', '.join(gpu_slots)}")
    reporter.log(f"Run log/status: {run_dir}")
    reporter.log("=" * 78)


def _print_dry_run(
    args: argparse.Namespace,
    entries: list[ConfigEntry],
    tasks: list[ClusterTask],
    gpu_slots: list[str],
) -> None:
    print("=" * 78)
    print("PEARL cluster dry run")
    print(f"Configs   : {len(entries)}")
    print(f"Tasks     : {len(tasks)}")
    print(f"GPU slots : {', '.join(gpu_slots)}")
    print("=" * 78)
    for task in tasks:
        print(f"[{task.index + 1:04d}/{len(tasks):04d}] {task.task_name}")
        print(f"  output: {task.run_path}")
        print(f"  {_display_command(_worker_command(args, task))}")


def _resolve_run_dir(requested: str | None) -> Path:
    if requested:
        path = Path(requested)
        return path if path.is_absolute() else Path.cwd() / path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path.cwd() / "results" / "_cluster_runs" / stamp


def _write_status(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".json.tmp")
    with temp_path.open("w", encoding="utf-8") as stream:
        json.dump(status, stream, indent=2)
        stream.write("\n")
    temp_path.replace(path)


def _display_command(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def _file_safe(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _cuda_device_count() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:  # noqa: BLE001 - auto should degrade cleanly on CPU hosts
        return 0


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicate_values: list[str] = []
    for value in values:
        if value in seen and value not in duplicate_values:
            duplicate_values.append(value)
        seen.add(value)
    return duplicate_values


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
