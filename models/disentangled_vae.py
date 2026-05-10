"""Disentangled VAE: latent split into z_meta (supervised) + z_bio (free).

Loss = recon + β·KL(z_bio; free-bits) + γ·KL(z_meta) + λ_sup·Σ supervised heads
       + λ_leak·HSIC(z_bio, m_observed)

The decoder takes concat(z_meta, z_bio) so flipping z_meta at inference time
actually changes the reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn


@dataclass
class DisentangledConfig:
    input_dim: int
    z_bio_dim: int = 50
    meta_fields: list[str] = field(default_factory=lambda: ["modality", "ischemia", "sex", "dthhrdy"])
    meta_dims: list[int] = field(default_factory=lambda: [1, 1, 1, 1])      # one z dim per field
    meta_kinds: list[str] = field(default_factory=lambda: ["bce", "mse", "bce", "mse"])
    hidden: tuple = (1024, 512, 256)

    @property
    def z_meta_dim(self) -> int:
        return sum(self.meta_dims)

    @property
    def latent_dim(self) -> int:
        return self.z_bio_dim + self.z_meta_dim


def _mlp(dims: list[int], dropout: float = 0.1) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class DisentangledVAE(nn.Module):
    def __init__(self, config: DisentangledConfig):
        super().__init__()
        self.config = config
        h = list(config.hidden)
        self.encoder = _mlp([config.input_dim] + h)
        self.mu_meta = nn.Linear(h[-1], config.z_meta_dim)
        self.logvar_meta = nn.Linear(h[-1], config.z_meta_dim)
        self.mu_bio = nn.Linear(h[-1], config.z_bio_dim)
        self.logvar_bio = nn.Linear(h[-1], config.z_bio_dim)
        self.decoder = _mlp([config.latent_dim] + h[::-1] + [config.input_dim])

        # Per-field heads on the dedicated z_meta slice.
        self.heads = nn.ModuleList()
        offset = 0
        for d in config.meta_dims:
            self.heads.append(nn.Linear(d, 1))
            offset += d

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.mu_meta(h), self.logvar_meta(h), self.mu_bio(h), self.logvar_bio(h)

    @staticmethod
    def reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def decode(self, z_meta: torch.Tensor, z_bio: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z_meta, z_bio], dim=1))

    def forward(self, x: torch.Tensor):
        mu_m, logvar_m, mu_b, logvar_b = self.encode(x)
        z_m = self.reparam(mu_m, logvar_m)
        z_b = self.reparam(mu_b, logvar_b)
        x_hat = self.decode(z_m, z_b)
        return x_hat, mu_m, logvar_m, mu_b, logvar_b, z_m, z_b

    def head_predictions(self, z_meta: torch.Tensor) -> list[torch.Tensor]:
        """Return one (B,) prediction per metadata field from its dedicated z slice."""
        offset = 0
        preds = []
        for head, d in zip(self.heads, self.config.meta_dims):
            preds.append(head(z_meta[:, offset:offset + d]).squeeze(-1))
            offset += d
        return preds


def kl_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Per-dim KL averaged over batch — shape (latent_dim,)."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)


def kl_with_free_bits(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float) -> torch.Tensor:
    """KL(q||N) with per-dim free-bits floor (scalar loss term)."""
    per_dim = kl_per_dim(mu, logvar)
    floor = torch.full_like(per_dim, fill_value=free_bits)
    return torch.maximum(per_dim, floor).sum()


def supervised_loss(
    preds: list[torch.Tensor],
    targets: list[torch.Tensor],          # NaN-marked; we mask
    kinds: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Returns (sum loss, per-field stats). Targets contain NaN where unknown."""
    total = preds[0].new_zeros(())
    stats: dict[str, float] = {}
    for i, (p, y, kind) in enumerate(zip(preds, targets, kinds)):
        mask = torch.isfinite(y)
        if not mask.any():
            stats[f"f{i}_loss"] = 0.0
            continue
        pm, ym = p[mask], y[mask]
        if kind == "bce":
            loss = nn.functional.binary_cross_entropy_with_logits(pm, ym)
        elif kind == "mse":
            loss = nn.functional.mse_loss(pm, ym)
        else:
            raise ValueError(f"Unknown kind {kind}")
        total = total + loss
        stats[f"f{i}_loss"] = loss.item()
    return total, stats


def hsic_penalty(z_bio: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Hilbert-Schmidt Independence Criterion with linear kernel.

    z_bio: (B, z_bio_dim), m: (B, n_meta) — m must be NaN-replaced upstream.
    Linear-kernel HSIC = ||X^T H Y||_F^2 / (B-1)^2 — cheap, no kernel-bandwidth knob.
    """
    B = z_bio.size(0)
    H = torch.eye(B, device=z_bio.device) - 1.0 / B
    Kz = z_bio @ z_bio.t()
    Km = m @ m.t()
    return ((H @ Kz @ H) * Km).sum() / max((B - 1) ** 2, 1)
