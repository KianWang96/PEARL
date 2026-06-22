"""Representation autoencoder classifier used by PEARL."""

from __future__ import annotations

import torch
import torch.nn as nn


class RepAEClassifier(nn.Module):
    """Small encoder-decoder-classifier model for 28x28 or 32x32 images."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        latent_dim: int = 64,
        img_size: int = 28,
        width: int = 16,
    ) -> None:
        super().__init__()
        if img_size % 4 != 0:
            raise ValueError("RepAEClassifier expects an image size divisible by 4.")

        self.latent_dim = latent_dim
        self.feature_size = img_size // 4
        hidden_channels = width * 2
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, hidden_channels, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        flattened_size = hidden_channels * self.feature_size * self.feature_size
        self.encoder_fc = nn.Linear(flattened_size, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, flattened_size)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, width, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(width, in_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        classifier_width = max(64, latent_dim)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, classifier_width),
            nn.ReLU(),
            nn.Linear(classifier_width, num_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        return self.encoder_fc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        h = h.view(
            h.size(0),
            self.decoder_conv[0].in_channels,
            self.feature_size,
            self.feature_size,
        )
        return self.decoder_conv(h)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        logits = self.classifier(z)
        return x_hat, logits, z
