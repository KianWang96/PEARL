"""Representation autoencoder classifier used by PEARL."""

from __future__ import annotations

import torch
import torch.nn as nn


class RepAEClassifier(nn.Module):
    """Small encoder-decoder-classifier model for 28x28 grayscale images."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        latent_dim: int = 64,
        img_size: int = 28,
    ) -> None:
        super().__init__()
        if img_size != 28:
            raise ValueError("RepAEClassifier currently expects 28x28 inputs.")

        self.latent_dim = latent_dim
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.encoder_fc = nn.Linear(32 * 7 * 7, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, 32 * 7 * 7)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, in_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        return self.encoder_fc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        h = h.view(h.size(0), 32, 7, 7)
        return self.decoder_conv(h)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        logits = self.classifier(z)
        return x_hat, logits, z
