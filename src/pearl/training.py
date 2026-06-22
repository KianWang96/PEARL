"""Local training and evaluation routines."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from pearl.config import ExperimentConfig


def local_train(
    model: torch.nn.Module,
    loader,
    config: ExperimentConfig,
    device: torch.device,
    proximal_reference: dict[str, torch.Tensor] | None = None,
    proximal_mu: float = 0.0,
    trainable_filter: Callable[[str], bool] | None = None,
    local_epochs: int | None = None,
) -> float:
    model.train()
    original_requires_grad = {
        name: parameter.requires_grad
        for name, parameter in model.named_parameters()
    }
    if trainable_filter is not None:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = trainable_filter(name)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("No trainable parameters selected for local training.")
    optimizer = torch.optim.Adam(trainable_parameters, lr=config.lr)
    total_loss = 0.0
    n = 0

    try:
        for _ in range(local_epochs or config.local_epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                x_hat, logits, _ = model(x)
                loss = (
                    config.lambda_rec * F.mse_loss(x_hat, x)
                    + config.lambda_cls * F.cross_entropy(logits, y)
                )
                if proximal_reference is not None and proximal_mu > 0.0:
                    proximal = sum(
                        torch.sum((parameter - proximal_reference[name]) ** 2)
                        for name, parameter in model.named_parameters()
                        if parameter.requires_grad
                    )
                    loss = loss + 0.5 * proximal_mu * proximal
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)
                n += x.size(0)
    finally:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = original_requires_grad[name]

    return total_loss / max(1, n)


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    ys: list[int] = []
    preds: list[int] = []
    rec_losses: list[float] = []
    cls_losses: list[float] = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_hat, logits, _ = model(x)
        pred = logits.argmax(dim=1)
        ys.extend(y.cpu().numpy().tolist())
        preds.extend(pred.cpu().numpy().tolist())
        rec_losses.append(F.mse_loss(x_hat, x, reduction="sum").item())
        cls_losses.append(F.cross_entropy(logits, y, reduction="sum").item())

    n = len(ys)
    if n == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "balanced_accuracy": 0.0,
            "rec_mse": 0.0,
            "cls_loss": 0.0,
        }

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="y_pred contains classes not in y_true",
            category=UserWarning,
        )
        balanced_accuracy = balanced_accuracy_score(ys, preds)

    return {
        "accuracy": float(accuracy_score(ys, preds)),
        "macro_f1": float(f1_score(ys, preds, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy),
        "rec_mse": float(sum(rec_losses) / max(1, n)),
        "cls_loss": float(sum(cls_losses) / max(1, n)),
    }


@torch.no_grad()
def evaluate_clients(
    models: list[torch.nn.Module],
    client_loaders: list,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    for k, model in enumerate(models):
        metrics = evaluate_model(model, client_loaders[k], device)
        metrics["client"] = k
        rows.append(metrics)
    return pd.DataFrame(rows)


def mean_metric(metrics: list[dict[str, float]], key: str) -> float:
    return float(np.mean([item[key] for item in metrics]))
