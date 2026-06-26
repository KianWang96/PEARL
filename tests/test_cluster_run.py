from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from pearl.cli import cluster_run
from pearl.results import write_csv_atomic


def test_parse_gpu_slots_auto_explicit_and_cpu():
    assert cluster_run.parse_gpu_slots("auto", cuda_device_count=4) == ["0", "1", "2", "3"]
    assert cluster_run.parse_gpu_slots("auto", max_workers=2, cuda_device_count=4) == ["0", "1"]
    assert cluster_run.parse_gpu_slots("0,2,7", max_workers=2) == ["0", "2"]
    assert cluster_run.parse_gpu_slots("cpu", max_workers=3) == ["cpu", "cpu", "cpu"]
    assert cluster_run.parse_gpu_slots("auto", cuda_device_count=0) == ["cpu"]


def test_expand_tasks_applies_method_seed_filters(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: tiny",
                f"output_dir: {tmp_path / 'out'}",
                "seeds: [1, 2]",
                "methods: [local_only, pearl_full]",
                "rounds: 3",
                "eval_every: 2",
            ]
        ),
        encoding="utf-8",
    )

    entries = cluster_run.load_config_entries(
        [str(config_path)],
        methods=["local_only"],
        seeds=[9, 10],
    )
    tasks = cluster_run.expand_tasks(entries)

    assert [(task.seed, task.method) for task in tasks] == [
        (9, "local_only"),
        (10, "local_only"),
    ]
    assert tasks[0].expected_rounds == {0, 2}


def test_multi_config_output_dirs_must_be_unique(tmp_path):
    out = tmp_path / "shared"
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    body = "\n".join(
        [
            "seeds: [1]",
            "methods: [local_only]",
            "rounds: 1",
            "eval_every: 1",
            f"output_dir: {out}",
        ]
    )
    first.write_text(f"experiment_name: first\n{body}\n", encoding="utf-8")
    second.write_text(f"experiment_name: second\n{body}\n", encoding="utf-8")

    try:
        cluster_run.load_config_entries([str(first), str(second)])
    except ValueError as exc:
        assert "unique output_dir" in str(exc)
    else:
        raise AssertionError("Expected duplicate output_dir values to fail.")


def test_scheduler_assigns_gpus_and_writes_final_outputs(tmp_path, monkeypatch):
    tasks = [
        cluster_run.ClusterTask(
            index=index,
            config_path=tmp_path / "config.yaml",
            experiment_name="sched",
            output_dir=tmp_path / "out",
            seed=index + 1,
            method="local_only",
            rounds=2,
            eval_every=1,
            plot_format="pdf",
        )
        for index in range(3)
    ]
    assignments: list[str] = []
    pids = iter(range(100, 200))

    class FakeStream:
        def close(self):
            return None

    class FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def poll(self) -> int:
            return 0

    def fake_start_task(args, task, gpu, run_dir, reporter):
        assignments.append(gpu)
        rows = [
            _result_row(task.seed, task.method, round_idx)
            for round_idx in sorted(task.expected_rounds)
        ]
        write_csv_atomic(pd.DataFrame(rows), task.run_path)
        pid = next(pids)
        return {
            "process": FakeProcess(pid),
            "stream": FakeStream(),
            "pid": pid,
            "task": task,
            "gpu": gpu,
            "log_path": run_dir / f"{pid}.log",
            "started": time.time(),
            "started_at": cluster_run._now(),
        }

    monkeypatch.setattr(cluster_run, "_start_task", fake_start_task)
    args = argparse.Namespace(
        resume=False,
        continue_on_error=False,
        skip_plots=True,
        overrides=[],
        quiet=True,
        skip_dataset_preflight=True,
    )
    run_dir = tmp_path / "cluster"
    reporter = cluster_run.Reporter(run_dir / "cluster.log")
    status = cluster_run._initial_status(
        args,
        [],
        tasks,
        ["0", "1"],
        run_dir,
    )
    try:
        rc = cluster_run._schedule_tasks(
            args,
            tasks,
            ["0", "1"],
            run_dir,
            reporter,
            run_dir / "status.json",
            status,
        )
    finally:
        reporter.close()

    assert rc == 0
    assert assignments == ["0", "1", "0"]
    assert (tmp_path / "out" / "results_partial.csv").is_file()
    assert (tmp_path / "out" / "results_all.csv").is_file()
    assert (tmp_path / "out" / "summary_final.csv").is_file()
    assert {task["state"] for task in status["tasks"]} == {"complete"}


def test_smoke_config_cluster_cli_uses_cpu_slot(monkeypatch):
    captured = {}

    def fake_run_cluster(args, entries, tasks, gpu_slots):
        captured["entries"] = entries
        captured["tasks"] = tasks
        captured["gpu_slots"] = gpu_slots
        return 0

    monkeypatch.setattr(cluster_run, "_run_cluster", fake_run_cluster)

    rc = cluster_run.main(
        [
            "--config",
            str(Path("configs") / "smoke.yaml"),
            "--gpus",
            "cpu",
            "--skip-plots",
        ]
    )

    assert rc == 0
    assert captured["gpu_slots"] == ["cpu"]
    assert captured["entries"][0].config.experiment_name == "smoke"
    assert len(captured["tasks"]) == 3


def _result_row(seed: int, method: str, round_idx: int) -> dict:
    return {
        "seed": seed,
        "round": round_idx,
        "method": method,
        "mean_global_accuracy": 0.1,
        "mean_global_macro_f1": 0.2,
        "worst_client_accuracy": 0.05,
        "neg_transfer_rate": 0.0,
        "selection_entropy": 0.0,
    }
