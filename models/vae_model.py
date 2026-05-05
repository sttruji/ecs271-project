"""Vanilla PyTorch VAE baseline for gene-expression reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass
class VAEConfig:
    input_dim: int
    latent_dim: int = 64
    hidden_dim: int = 256
    beta: float = 1.0


class VanillaVAE(nn.Module):
    """Small dense VAE with Gaussian latent variables and MSE decoder loss."""

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(config.hidden_dim, config.latent_dim)
        self.logvar = nn.Linear(config.hidden_dim, config.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        return self.mu(hidden), self.logvar(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic reconstruction using the posterior mean."""
        mu, _ = self.encode(x)
        return self.decode(mu)


def vae_loss(
    x: torch.Tensor,
    reconstruction: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstruction_loss = nn.functional.mse_loss(reconstruction, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + beta * kl_loss, reconstruction_loss, kl_loss


def save_vae_checkpoint(
    path: Path,
    model: VanillaVAE,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    metrics: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(model.config),
            "state_dict": model.state_dict(),
            "feature_mean": feature_mean.astype(np.float32),
            "feature_std": feature_std.astype(np.float32),
            "metrics": metrics,
        },
        path,
    )


def load_vae_checkpoint(path: Path, device: torch.device | str = "cpu") -> tuple[VanillaVAE, np.ndarray, np.ndarray]:
    checkpoint = torch.load(path, map_location=device)
    config = VAEConfig(**checkpoint["config"])
    model = VanillaVAE(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint["feature_mean"], checkpoint["feature_std"]


def reconstruct_numpy(
    model: VanillaVAE,
    x: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x_scaled = (x - feature_mean) / feature_std
    with torch.no_grad():
        tensor = torch.from_numpy(x_scaled).to(device)
        reconstruction_scaled = model.reconstruct(tensor).cpu().numpy()
    return (reconstruction_scaled * feature_std + feature_mean).astype(np.float32)
