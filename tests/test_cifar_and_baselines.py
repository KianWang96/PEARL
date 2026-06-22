import torch
import pytest
from torch import nn
from torch.utils.data import TensorDataset

from pearl.config import ExperimentConfig
from pearl.constants import SERVER_METHODS
from pearl.exchange import weighted_average_state_dict
from pearl.experiment import _active_clients_for_round, run_experiment
from pearl.models import RepAEClassifier


def test_cifar_model_preserves_rgb_image_shape():
    model = RepAEClassifier(
        in_channels=3,
        num_classes=10,
        latent_dim=16,
        img_size=32,
        width=4,
    )

    reconstructed, logits, latent = model(torch.rand(2, 3, 32, 32))

    assert reconstructed.shape == (2, 3, 32, 32)
    assert logits.shape == (2, 10)
    assert latent.shape == (2, 16)


def test_weighted_average_state_dict_uses_client_weights():
    first = nn.Linear(1, 1, bias=False)
    second = nn.Linear(1, 1, bias=False)
    first.weight.data.fill_(1.0)
    second.weight.data.fill_(5.0)

    averaged = weighted_average_state_dict([first, second], [3.0, 1.0])

    assert torch.allclose(averaged["weight"], torch.tensor([[2.0]]))


def test_dropout_schedule_is_method_independent_and_deterministic():
    first = _active_clients_for_round(50, 0.6, seed=3, round_idx=7)
    second = _active_clients_for_round(50, 0.6, seed=3, round_idx=7)

    assert first == second
    assert 0 < len(first) < 50


@pytest.mark.parametrize("method", sorted(SERVER_METHODS))
def test_server_baseline_smoke_runs_use_server_family(monkeypatch, method):
    train = TensorDataset(
        torch.rand(8, 1, 28, 28),
        torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
    )
    test = TensorDataset(
        torch.rand(4, 1, 28, 28),
        torch.tensor([0, 1, 0, 1]),
    )

    monkeypatch.setattr(
        "pearl.experiment.get_dataset",
        lambda *args, **kwargs: (train, test, 2, 1, 28),
    )
    cfg = ExperimentConfig(
        selection_method=method,
        methods=[method],
        dataset="mnist",
        graph_type="server",
        num_clients=2,
        partition="iid",
        rounds=1,
        local_epochs=1,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        latent_dim=4,
        model_width=2,
        eval_every=1,
    )

    result = run_experiment(cfg, verbose=False)

    assert result.loc[0, "comparison_family"] == "server_reference"
    assert result.loc[0, "graph"] == "server"
    assert result.loc[0, "round_exchanges"] == 2
    assert result.loc[0, "round_bytes"] > 0


@pytest.mark.parametrize(
    ("method", "expected_family"),
    [
        ("model_similarity", "budget_matched_decentralized"),
        ("dpsgd_full_neighbors", "decentralized_reference"),
    ],
)
def test_expanded_decentralized_baselines_smoke_run(
    monkeypatch,
    method,
    expected_family,
):
    train = TensorDataset(
        torch.rand(12, 1, 28, 28),
        torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]),
    )
    test = TensorDataset(
        torch.rand(6, 1, 28, 28),
        torch.tensor([0, 1, 2, 0, 1, 2]),
    )
    monkeypatch.setattr(
        "pearl.experiment.get_dataset",
        lambda *args, **kwargs: (train, test, 3, 1, 28),
    )
    cfg = ExperimentConfig(
        selection_method=method,
        methods=[method],
        dataset="mnist",
        graph_type="ring",
        num_clients=3,
        partition="iid",
        rounds=1,
        local_epochs=1,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        latent_dim=4,
        model_width=2,
        eval_every=1,
    )

    result = run_experiment(cfg, verbose=False)

    assert result.loc[0, "comparison_family"] == expected_family
    assert result.loc[0, "round_exchanges"] >= 3


def test_pearl_smoke_run_tracks_stale_descriptor_age(monkeypatch):
    train = TensorDataset(
        torch.rand(8, 1, 28, 28),
        torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
    )
    test = TensorDataset(
        torch.rand(4, 1, 28, 28),
        torch.tensor([0, 1, 0, 1]),
    )
    monkeypatch.setattr(
        "pearl.experiment.get_dataset",
        lambda *args, **kwargs: (train, test, 2, 1, 28),
    )
    cfg = ExperimentConfig(
        selection_method="pearl_full",
        methods=["pearl_full"],
        dataset="mnist",
        graph_type="ring",
        num_clients=2,
        partition="iid",
        rounds=2,
        local_epochs=1,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        latent_dim=4,
        model_width=2,
        anchor_size=2,
        descriptor_refresh_period=2,
        eval_every=1,
    )

    result = run_experiment(cfg, verbose=False)

    assert result["descriptor_age"].tolist() == [0, 1]
    assert result["round_exchanges"].tolist() == [2, 2]
    assert set(result["comparison_family"]) == {"budget_matched_decentralized"}
