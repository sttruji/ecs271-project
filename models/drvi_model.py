"""Additive VAE — DRVI-style decoder with Laplace (L1) reconstruction.

Standard decoder: x̂ = Dec(z₁, z₂, ..., z_K)   — all dims interact inside
                                                    the MLP, nonlinear at
                                                    the DONOR level.

Additive decoder: x̂ = Σ_k f_k(z_k)              — each f_k: ℝ → ℝ^G is a
                                                    small per-dim MLP; dims
                                                    add independently, so
                                                    nonlinearities are at the
                                                    GENE level only.

Reconstruction loss: Laplace (L1) rather than MSE.
For log-CPM data, L1 is the continuous analog of the overdispersion handling
that NB provides for raw counts: it downweights influence of high-expression
outlier genes and corresponds to a Laplace prior on reconstruction residuals.

The ELBO is:
    ℒ = -E[||x - x̂||₁] - β · KL(q(z|x) ∥ p(z))

with free-bits clipping on KL to prevent posterior collapse.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class AdditiveVAEConfig:
    input_dim: int
    n_latent: int = 12
    encoder_hidden: tuple[int, ...] = (512, 256)
    decoder_hidden: int = 64
    beta: float = 1e-3
    free_bits: float = 0.1


# ── per-dim decoder ────────────────────────────────────────────────────────

class PerDimDecoder(nn.Module):
    """K independent 1→H→G MLPs, one per latent dimension.

    Each f_k maps a scalar z_k to an additive gene-expression contribution.
    Implemented as batched einsum for efficiency — no Python loop over K.

    Memory: K × H × G floats (e.g. 12 × 64 × 11374 ≈ 8.7 M params).
    """

    def __init__(self, n_dims: int, n_genes: int, hidden: int = 64):
        super().__init__()
        self.n_dims = n_dims
        self.n_genes = n_genes
        self.hidden = hidden
        # 1 → hidden: W1 std=1 so GELU is in its active range.
        # b1 spread across [-2, 2] to give each neuron a different activation threshold —
        # effectively pre-seeding a diverse 1-D basis for the function.
        self.W1 = nn.Parameter(torch.randn(n_dims, hidden))
        b1_init = torch.linspace(-2.0, 2.0, n_dims * hidden).reshape(n_dims, hidden)
        self.b1 = nn.Parameter(b1_init.clone())
        # hidden → n_genes.  Small init: initial output will be zeroed by centering anyway.
        self.W2 = nn.Parameter(torch.randn(n_dims, hidden, n_genes) * 0.01)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, K)
        h    = F.gelu(z.unsqueeze(-1) * self.W1.unsqueeze(0) + self.b1.unsqueeze(0))  # (batch,K,H)
        h0   = F.gelu(self.b1.unsqueeze(0))   # (1,K,H) — decoder output at z_k=0

        # Zero-center: f̃_k(z_k) = f_k(z_k) - f_k(0).
        # Guarantees output=0 at init (regardless of W2 scale) and preserves full
        # gradient flow for z_k≠0. No b2 needed — offset is absorbed into h0.
        delta = (h - h0).permute(1, 0, 2)   # (K, batch, H)
        out   = torch.bmm(delta, self.W2)    # (K, batch, G)
        return out.sum(0)                    # (batch, G)

    def per_dim_curve(self, dim: int, z_range: torch.Tensor) -> torch.Tensor:
        """Gene-response curve for dim k over a 1-D grid of z values.

        Returns (len(z_range), n_genes) — useful for plotting f_k(z_k).
        """
        z_full = torch.zeros(len(z_range), self.n_dims, device=z_range.device)
        z_full[:, dim] = z_range
        with torch.no_grad():
            return self.forward(z_full)


# ── full model ─────────────────────────────────────────────────────────────

class AdditiveVAE(nn.Module):

    def __init__(self, cfg: AdditiveVAEConfig):
        super().__init__()
        self.cfg = cfg
        h = list(cfg.encoder_hidden)
        layers: list[nn.Module] = []
        in_dim = cfg.input_dim
        for out_dim in h:
            layers += [nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU()]
            in_dim = out_dim
        self.encoder = nn.Sequential(*layers)
        self.mu_head = nn.Linear(h[-1], cfg.n_latent)
        self.lv_head = nn.Linear(h[-1], cfg.n_latent)
        self.decoder = PerDimDecoder(cfg.n_latent, cfg.input_dim, cfg.decoder_hidden)
        self.n_latent = cfg.n_latent
        self.n_genes = cfg.input_dim

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu_head(h), self.lv_head(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, lv = self.encode(x)
        z = mu + (0.5 * lv).exp() * torch.randn_like(mu)
        return self.decode(z), mu, lv

    def elbo(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        return self.elbo_beta(x, self.cfg.beta)

    def elbo_beta(self, x: torch.Tensor, beta: float) -> Tuple[torch.Tensor, dict]:
        x_hat, mu, lv = self.forward(x)
        recon = (x - x_hat).abs().mean()
        kl_per_dim = -0.5 * (1.0 + lv - mu.pow(2) - lv.exp())
        kl = kl_per_dim.clamp(min=self.cfg.free_bits).sum(dim=-1).mean()
        loss = recon + beta * kl
        return loss, {"recon": recon.item(), "kl": kl.item()}

    # ── LatentModel protocol ───────────────────────────────────────────────

    @property
    def name(self) -> str:
        return f"additive_vae_K{self.n_latent}"

    def encode_np(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            t = torch.from_numpy(x.astype(np.float32)).to(next(self.parameters()).device)
            mu, _ = self.encode(t)
        return mu.cpu().numpy()

    def decode_np(self, z: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            t = torch.from_numpy(z.astype(np.float32)).to(next(self.parameters()).device)
        return self.decode(t).cpu().numpy()
