"""Disentangled VAE: latent split into z_meta (supervised) + z_bio (free).

Loss = recon + β_bio·KL(z_bio; free-bits) + β_meta·KL_cap(z_meta) + λ_sup·Σ supervised heads
       + λ_leak·HSIC(z_bio, modality) + λ_cycle·cycle_bio + λ_cycle_meta·cycle_meta

KL_cap(z_meta): each per-field KL is amplified by (1 + λ_cap·head_loss_k.detach()).
  → If a reserved z_meta dim FAILS to predict its assigned metadata, its KL weight
    rises, pushing it back toward the prior (punishment).
  → If it SUCCEEDS (head_loss_k → 0), weight → 1 (standard KL, slot is earned).
  → This implements "punish reserved dimensions unless they predict their metadata."

Report finding: only ~30 bulk PCs carry stable biology signal.  z_bio_dim is
capped at 32 by default; free-bits floor prevents unused dims from consuming
capacity, so the model naturally concentrates on true signal rather than
chasing perfect reconstruction.

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
    # z_bio capped at 32: report found only ~30 PCs carry stable biology signal.
    # Free-bits floor prevents unused dims from filling up, so active count
    # naturally stays at the information content of the data.
    z_bio_dim: int = 32
    meta_fields: list[str] = field(default_factory=lambda: ["modality", "ischemia", "sex", "dthhrdy"])
    # 2 dims per field: gives MLP head enough room to model the slot non-linearly
    # while keeping z_meta small (8 total vs 4 before).
    meta_dims: list[int] = field(default_factory=lambda: [2, 2, 2, 2])
    meta_kinds: list[str] = field(default_factory=lambda: ["bce", "mse", "bce", "mse"])
    hidden: tuple = (1024, 512, 256)
    # Hidden dims for per-field MLP prediction heads.
    meta_hidden: tuple = (32,)
    # Capacity-penalty multiplier: amplifies KL of z_meta dims that fail to
    # predict their assigned metadata.  0 = standard KL, 1 = default.
    lam_cap: float = 1.0

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


def _head_mlp(in_dim: int, hidden: tuple, out_dim: int = 1) -> nn.Module:
    """Small MLP prediction head for one z_meta field.

    in_dim > 1: Linear → GELU → (hidden layers) → Linear(out_dim).
    in_dim == 1: single Linear (nonlinearity on a scalar gives no benefit).
    """
    if in_dim == 1 or not hidden:
        return nn.Linear(in_dim, out_dim)
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden[0]), nn.GELU()]
    for i in range(len(hidden) - 1):
        layers += [nn.Linear(hidden[i], hidden[i + 1]), nn.GELU()]
    layers.append(nn.Linear(hidden[-1], out_dim))
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

        # Per-field MLP heads on each dedicated z_meta slice.
        # Each head sees only the dims assigned to its metadata field.
        self.heads = nn.ModuleList()
        for d in config.meta_dims:
            self.heads.append(_head_mlp(d, config.meta_hidden))

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
) -> tuple[torch.Tensor, dict[str, float], list[torch.Tensor]]:
    """Returns (sum loss, per-field stats, per-field loss tensors).

    Targets contain NaN where unknown — those samples are masked out for that
    field so the auxiliary loss is never penalised on missing metadata.
    per-field loss tensors are zero-dim tensors used by capacity_weighted_kl_meta.
    """
    total = preds[0].new_zeros(())
    stats: dict[str, float] = {}
    per_field: list[torch.Tensor] = []
    for i, (p, y, kind) in enumerate(zip(preds, targets, kinds)):
        mask = torch.isfinite(y)
        if not mask.any():
            stats[f"f{i}_loss"] = 0.0
            per_field.append(p.new_zeros(()))
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
        per_field.append(loss)
    return total, stats, per_field


def capacity_weighted_kl_meta(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    per_field_losses: list[torch.Tensor],
    meta_dims: list[int],
    lam_cap: float = 1.0,
    lam_cap_per_field: list[float] | None = None,
) -> torch.Tensor:
    """KL for z_meta with per-field capacity-weighted punishment.

    For each z_meta field k occupying dims [offset, offset+d):
        KL_k = (1 + cap_k * head_loss_k.detach()) * sum_KL_per_dim_k

    cap_k = lam_cap_per_field[k] if provided, else lam_cap (uniform).

    When head_loss_k is high (failed prediction) → weight > 1 → larger KL →
    dims are pushed back toward the prior (punished).
    When head_loss_k → 0 (good prediction) → weight → 1 → standard KL →
    the reserved slot earns its capacity.

    lam_cap_per_field allows softer punishment for noisy metadata fields
    (e.g. DTHHRDY is a 0-4 ordinal with uneven distribution; punishing it
    at the same rate as modality/sex creates a vicious cycle where the slot
    never gets trained because KL always crushes it back to the prior).
    """
    kl_dims = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
    total = kl_dims.new_zeros(())
    offset = 0
    for k, (loss_k, d) in enumerate(zip(per_field_losses, meta_dims)):
        cap = lam_cap_per_field[k] if lam_cap_per_field is not None else lam_cap
        w = 1.0 + cap * loss_k.detach()
        total = total + w * kl_dims[offset:offset + d].sum()
        offset += d
    return total


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
