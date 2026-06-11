"""Dataset loading and client partitioning."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from pearl.config import ExperimentConfig


def get_dataset(
    name: str = "fashion_mnist",
    data_dir: str | Path = "data",
    train_subset: int | None = None,
    test_subset: int | None = None,
    download: bool = True,
):
    transform = transforms.Compose([transforms.ToTensor()])
    root = str(data_dir)

    if name == "fashion_mnist":
        train_ds = torchvision.datasets.FashionMNIST(
            root=root,
            train=True,
            download=download,
            transform=transform,
        )
        test_ds = torchvision.datasets.FashionMNIST(
            root=root,
            train=False,
            download=download,
            transform=transform,
        )
    elif name == "mnist":
        train_ds = torchvision.datasets.MNIST(
            root=root,
            train=True,
            download=download,
            transform=transform,
        )
        test_ds = torchvision.datasets.MNIST(
            root=root,
            train=False,
            download=download,
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")

    num_classes = 10
    in_channels = 1
    img_size = 28

    if train_subset is not None:
        train_ds = Subset(train_ds, list(range(min(train_subset, len(train_ds)))))
    if test_subset is not None:
        test_ds = Subset(test_ds, list(range(min(test_subset, len(test_ds)))))

    return train_ds, test_ds, num_classes, in_channels, img_size


def get_targets(dataset) -> np.ndarray:
    """Return labels for raw torchvision datasets and Subset wrappers."""
    if isinstance(dataset, Subset):
        base_targets = np.asarray(dataset.dataset.targets)
        return base_targets[np.asarray(dataset.indices)]
    return np.asarray(dataset.targets)


def iid_partition(dataset, num_clients: int) -> list[np.ndarray]:
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    return [part.astype(int) for part in np.array_split(indices, num_clients)]


def dirichlet_partition(
    dataset,
    num_clients: int,
    num_classes: int,
    alpha: float = 0.3,
    min_size: int = 10,
    max_attempts: int = 1000,
) -> list[np.ndarray]:
    targets = get_targets(dataset)
    idx_by_class = [np.where(targets == c)[0] for c in range(num_classes)]

    for _ in range(max_attempts):
        client_indices: list[list[int]] = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            idx_c = idx_by_class[c].copy()
            np.random.shuffle(idx_c)
            proportions = np.random.dirichlet(alpha * np.ones(num_clients))
            cuts = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            for k, split in enumerate(np.split(idx_c, cuts)):
                client_indices[k].extend(split.tolist())

        if min(len(items) for items in client_indices) >= min_size:
            return [np.asarray(items, dtype=int) for items in client_indices]

    raise RuntimeError(
        "Could not create a Dirichlet partition with "
        f"min_client_size={min_size} after {max_attempts} attempts. "
        "Try reducing num_clients or min_client_size, increasing train_subset, "
        "or increasing dirichlet_alpha."
    )


def shards_partition(
    dataset,
    num_clients: int,
    num_classes: int,
    classes_per_client: int = 2,
) -> list[np.ndarray]:
    targets = get_targets(dataset)
    class_indices = {
        c: np.where(targets == c)[0].astype(int).tolist()
        for c in range(num_classes)
    }
    for c in range(num_classes):
        random.shuffle(class_indices[c])

    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    for k in range(num_clients):
        chosen = np.random.choice(num_classes, classes_per_client, replace=False)
        for c in chosen:
            denom = num_clients * classes_per_client // num_classes + 1
            take = max(1, len(class_indices[c]) // denom)
            client_indices[k].extend(class_indices[c][:take])
            class_indices[c] = class_indices[c][take:]

    remaining = [idx for c in range(num_classes) for idx in class_indices[c]]
    random.shuffle(remaining)
    for i, idx in enumerate(remaining):
        client_indices[i % num_clients].append(idx)

    return [np.asarray(items, dtype=int) for items in client_indices]


def make_partitions(
    dataset,
    config: ExperimentConfig,
    num_classes: int,
) -> list[np.ndarray]:
    if config.partition == "iid":
        return iid_partition(dataset, config.num_clients)
    if config.partition == "dirichlet":
        return dirichlet_partition(
            dataset,
            config.num_clients,
            num_classes,
            alpha=config.dirichlet_alpha,
            min_size=config.min_client_size,
        )
    if config.partition == "shards":
        return shards_partition(
            dataset,
            config.num_clients,
            num_classes,
            classes_per_client=config.classes_per_client,
        )
    raise ValueError(f"Unknown partition: {config.partition}")


def make_client_loaders(
    train_ds,
    client_indices: list[np.ndarray],
    config: ExperimentConfig,
) -> list[DataLoader]:
    return [
        DataLoader(
            Subset(train_ds, idx.astype(int).tolist()),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
        for idx in client_indices
    ]


def make_test_loader(test_ds, batch_size: int = 256, num_workers: int = 2) -> DataLoader:
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
