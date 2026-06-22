import pandas as pd

from pearl.cli.run import _load_complete_run
from pearl.results import write_csv_atomic


def test_resume_accepts_only_complete_matching_runs(tmp_path):
    path = tmp_path / "run.csv"
    complete = pd.DataFrame(
        [
            {"round": 0, "seed": 2, "method": "fedavg"},
            {"round": 4, "seed": 2, "method": "fedavg"},
        ]
    )
    write_csv_atomic(complete, path)

    loaded = _load_complete_run(path, {0, 4}, 2, "fedavg")

    assert loaded is not None
    assert _load_complete_run(path, {0, 2, 4}, 2, "fedavg") is None
    assert _load_complete_run(path, {0, 4}, 3, "fedavg") is None
    assert _load_complete_run(path, {0, 4}, 2, "fedprox") is None
