"""
SparseGlobalFiLMVAE — 80/20 mixed-sparsity decoder.

Architecture
-----------
K = z_bio_dim total latent dims, split into:
  • K_sparse (default 80%): each dim k has a *separate* linear layer
    W_k ∈ R^G with L1 penalty → learns a small gene programme.
  • K_global (default 20%): all global dims concatenated → small FiLM
    MLP with NO sparsity → can attend to any combination of genes.

Output:
  x̂ = Σ_{k<K_sparse} z_k · W_k   +   FiLM-MLP(z_global, meta)   +  FiLM-meta-shift

The sparse part is linear (interpretable per-dim loadings), the global
part is nonlinear and unconstrained.  Meta conditioning is applied as
FiLM to the global MLP only (meta always has free access through the
global path).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .meta_injection_vae import (
    FiLMMetaInjectionConfig,
    _FiLMDecoder,
    _log_normal,
    tc_minibatch,
)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class SparseGlobalConfig:
    input_dim: int
    meta_dim: int
    z_bio_dim: int = 16

    # split
    sparse_frac: float = 0.80          # fraction of dims that get L1 penalty
    lambda_sparse: float = 1e-3        # L1 weight on sparse decoder weights

    encoder_hidden: tuple[int, ...] = (512, 256)
    global_decoder_hidden: tuple[int, ...] = (128, 256)   # smaller — only sees 20% of dims
    meta_embed_dim: int = 128

    beta: float = 1e-3
    free_bits: float = 0.5
    lambda_tc: float = 0.0

    @property
    def n_sparse(self) -> int:
        return round(self.z_bio_dim * self.sparse_frac)

    @property
    def n_global(self) -> int:
        return self.z_bio_dim - self.n_sparse


def _mlp(dims):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [nn.LayerNorm(dims[i + 1]), nn.GELU()]
    return nn.Sequential(*layers)


# ── Sparse linear decoder ─────────────────────────────────────────────────────

class _SparseLinearDecoder(nn.Module):
    """K_sparse separate linear readouts, one per latent dim.

    weight[k] ∈ R^G  (no bias — bias is captured by the global path + meta).
    Forward returns sum_k z_k * W_k.
    L1 penalty is computed separately and added to the ELBO.
    """

    def __init__(self, n_sparse: int, n_genes: int):
        super().__init__()
        # One linear per dim (no shared weights) → easy per-dim L1
        self.W = nn.Parameter(torch.zeros(n_sparse, n_genes))
        nn.init.normal_(self.W, 0, 0.01)

    def forward(self, z_sparse: torch.Tensor) -> torch.Tensor:
        # z_sparse: (B, K_sparse)   W: (K_sparse, G)
        return z_sparse @ self.W    # (B, G)

    def l1_penalty(self) -> torch.Tensor:
        return self.W.abs().mean()

    def loadings(self) -> np.ndarray:
        """Return (K_sparse, G) weight matrix as numpy."""
        return self.W.detach().cpu().numpy()


# ── Full model ────────────────────────────────────────────────────────────────

class SparseGlobalFiLMVAE(nn.Module):
    """80/20 sparse-global FiLM VAE.

    The first `n_sparse` latent dims learn sparse gene programmes (L1 on
    decoder weights).  The remaining `n_global` dims pass through a small
    FiLM-conditioned MLP with no sparsity — free to capture global or
    complex patterns.

    Usage
    -----
    cfg = SparseGlobalConfig(input_dim=3000, meta_dim=4, z_bio_dim=16)
    model = SparseGlobalFiLMVAE(cfg)
    loss, info = model.elbo(x, meta)
    # info contains: recon, kl, tc, sparse_l1
    """

    def __init__(self, cfg: SparseGlobalConfig):
        super().__init__()
        self.cfg = cfg
        K = cfg.z_bio_dim
        self.n_sparse = cfg.n_sparse
        self.n_global = cfg.n_global

        # ── Encoder (shared, sees full x) ─────────────────────────────────
        enc_layers: list[nn.Module] = []
        in_d = cfg.input_dim
        for out_d in cfg.encoder_hidden:
            enc_layers += [nn.Linear(in_d, out_d), nn.LayerNorm(out_d), nn.GELU()]
            in_d = out_d
        self.encoder  = nn.Sequential(*enc_layers)
        self.mu_head  = nn.Linear(cfg.encoder_hidden[-1], K)
        self.lv_head  = nn.Linear(cfg.encoder_hidden[-1], K)

        # ── Sparse linear decoder (first n_sparse dims) ────────────────────
        self.sparse_dec = _SparseLinearDecoder(self.n_sparse, cfg.input_dim)

        # ── Global FiLM decoder (last n_global dims + meta) ───────────────
        self.meta_embed = nn.Sequential(
            nn.Linear(cfg.meta_dim, cfg.meta_embed_dim), nn.GELU(),
            nn.Linear(cfg.meta_embed_dim, cfg.meta_embed_dim),
        )
        self.global_dec = _FiLMDecoder(
            self.n_global, cfg.meta_embed_dim,
            cfg.global_decoder_hidden, cfg.input_dim
        )

        # Output bias (shared, like a gene-level intercept)
        self.bias = nn.Parameter(torch.zeros(cfg.input_dim))

        self.n_latent = K
        self.n_genes  = cfg.input_dim

    # ── encode / decode ───────────────────────────────────────────────────

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.mu_head(h), self.lv_head(h)

    def decode(self, z: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        z_sparse = z[:, :self.n_sparse]
        z_global = z[:, self.n_sparse:]
        out_sparse = self.sparse_dec(z_sparse)
        out_global = self.global_dec(z_global, self.meta_embed(meta))
        return out_sparse + out_global + self.bias

    def forward(self, x, meta):
        mu, lv = self.encode(x)
        z = mu + (0.5 * lv).exp() * torch.randn_like(mu)
        return self.decode(z, meta), mu, lv, z

    # ── ELBO ──────────────────────────────────────────────────────────────

    def elbo(self, x: torch.Tensor, meta: torch.Tensor,
             beta: Optional[float] = None,
             lambda_sparse: Optional[float] = None,
             lambda_tc: Optional[float] = None) -> tuple[torch.Tensor, dict]:
        if beta          is None: beta          = self.cfg.beta
        if lambda_sparse is None: lambda_sparse = self.cfg.lambda_sparse
        if lambda_tc     is None: lambda_tc     = self.cfg.lambda_tc

        mu, lv = self.encode(x)
        z = mu + (0.5 * lv).exp() * torch.randn_like(mu)
        x_hat = self.decode(z, meta)

        recon = F.mse_loss(x_hat, x)
        kl_per_dim = -0.5 * (1.0 + lv - mu.pow(2) - lv.exp())
        kl = kl_per_dim.clamp(min=self.cfg.free_bits).sum(-1).mean()

        sparse_l1 = self.sparse_dec.l1_penalty()
        tc_val = tc_minibatch(z, mu, lv) if lambda_tc > 0 else torch.zeros(1, device=x.device)

        loss = recon + beta * kl + lambda_sparse * sparse_l1 + lambda_tc * tc_val

        return loss, {
            "recon":     recon.item(),
            "kl":        kl.item(),
            "sparse_l1": sparse_l1.item(),
            "tc":        tc_val.item() if isinstance(tc_val, torch.Tensor) else 0.0,
            "n_sparse":  self.n_sparse,
            "n_global":  self.n_global,
        }

    # ── Inspection ────────────────────────────────────────────────────────

    def sparse_loadings(self) -> np.ndarray:
        """Return (n_sparse, n_genes) decoder weight matrix."""
        return self.sparse_dec.loadings()

    def nnz_per_dim(self, threshold: float = 0.01) -> np.ndarray:
        """Number of genes with |weight| > threshold per sparse dim."""
        W = self.sparse_loadings()
        return (np.abs(W) > threshold).sum(axis=1)

    # ── Protocol ──────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return (f"sparse_global_vae_K{self.n_latent}_"
                f"sp{self.n_sparse}gl{self.n_global}")

    def encode_np(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(
                torch.from_numpy(x.astype(np.float32)).to(next(self.parameters()).device)
            )
        return mu.cpu().numpy()

    def flip(self, x: np.ndarray, meta_original: np.ndarray,
             meta_target: np.ndarray) -> np.ndarray:
        dev = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(torch.from_numpy(x.astype(np.float32)).to(dev))
            x_hat = self.decode(mu, torch.from_numpy(meta_target.astype(np.float32)).to(dev))
        return x_hat.cpu().numpy()
