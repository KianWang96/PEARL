"""Lightweight client descriptors for peer selection."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from pearl.training import evaluate_model


@torch.no_grad()
def compute_prototypes_and_class_acc(
    model: torch.nn.Module,
    loader,
    num_classes: int,
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], dict[int, float]]:
    model.eval()
    proto_sums: dict[int, torch.Tensor | None] = {
        c: None for c in range(num_classes)
    }
    proto_counts = {c: 0 for c in range(num_classes)}
    class_correct = {c: 0 for c in range(num_classes)}
    class_total = {c: 0 for c in range(num_classes)}

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        z = model.encode(x)
        logits = model.classifier(z)
        preds = logits.argmax(dim=1)

        for c in range(num_classes):
            mask = y == c
            if mask.any():
                zc = z[mask].sum(dim=0).detach().cpu()
                current = proto_sums[c]
                proto_sums[c] = zc if current is None else current + zc
                proto_counts[c] += int(mask.sum().item())
                class_correct[c] += int((preds[mask] == c).sum().item())
                class_total[c] += int(mask.sum().item())

    prototypes = {
        c: proto_sums[c] / proto_counts[c]
        for c in range(num_classes)
        if proto_counts[c] > 0 and proto_sums[c] is not None
    }
    class_acc = {
        c: class_correct[c] / class_total[c]
        for c in range(num_classes)
        if class_total[c] > 0
    }
    return prototypes, class_acc


@torch.no_grad()
def anchor_quality(
    model_j: torch.nn.Module,
    anchor_xs: torch.Tensor,
    anchor_ys: torch.Tensor,
    device: torch.device,
) -> float:
    model_j.eval()
    xs = anchor_xs.to(device)
    ys = anchor_ys.to(device)
    _, logits, _ = model_j(xs)
    loss = F.cross_entropy(logits, ys).item()
    return -loss


def sample_anchor(dataset_indices, train_ds, anchor_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    indices = np.asarray(dataset_indices, dtype=int)
    n = len(indices)
    if n == 0:
        raise ValueError("Cannot sample an anchor set from an empty client dataset.")
    chosen = np.random.choice(n, size=min(anchor_size, n), replace=False)
    selected = indices[chosen]

    xs_list = []
    ys_list = []
    for idx in selected:
        x, y = train_ds[int(idx)]
        xs_list.append(x)
        ys_list.append(y)

    return torch.stack(xs_list), torch.tensor(ys_list, dtype=torch.long)


def build_descriptors(
    models: list[torch.nn.Module],
    client_loaders: list,
    num_classes: int,
    device: torch.device,
) -> dict[int, dict]:
    descriptors = {}
    for k, model in enumerate(models):
        protos, cls_acc = compute_prototypes_and_class_acc(
            model,
            client_loaders[k],
            num_classes,
            device,
        )
        metrics = evaluate_model(model, client_loaders[k], device)
        descriptors[k] = {
            "prototypes": protos,
            "class_acc": cls_acc,
            "quality": -metrics["cls_loss"],
            "rec_mse": metrics["rec_mse"],
        }
    return descriptors


def get_hard_classes(class_acc: dict[int, float], threshold: float) -> set[int]:
    return {c for c, acc in class_acc.items() if acc < threshold}
