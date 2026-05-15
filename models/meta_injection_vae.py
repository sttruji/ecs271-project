"""Metadata-injection VAE (two variants).

Both variants share the same design principle: the encoder only sees x and
outputs z_bio; metadata is injected in the decoder only.

  Encoder:  x                    → z_bio
  Decoder:  f(z_bio, meta)       → x̂

``MetaInjectionVAE`` (concat):
    Simple concatenation — concat(z_bio, meta) fed into a standard MLP decoder.

``FiLMMetaInjectionVAE`` (FiLM):
    Meta is embedded by a small MLP into meta_emb; then Feature-wise Linear
    Modulation (FiLM) scales/shifts each decoder hidden layer using meta_emb.
    This gives meta direct control over every layer, not just a single input slot.
    Optionally adds a Total-Correlation (β-TCVAE) penalty on z_bio to promote
    disentanglement of biological factors.

Flip at inference is trivial: pass any meta vector to decode().
No supervised heads, no z_meta prediction from the encoder.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class MetaInjectionConfig:
    input_dim: int
    meta_dim: int
    z_bio_dim: int = 12
    encoder_hidden: tuple[int, ...] = (512, 256)
    beta: float = 1e-3
    free_bits: float = 0.1


def _mlp(dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [nn.LayerNorm(dims[i + 1]), nn.GELU()]
    return nn.Sequential(*layers)


class MetaInjectionVAE(nn.Module):
    """Encoder → z_bio; decoder gets concat(z_bio, meta)."""

    def __init__(self, cfg: MetaInjectionConfig):
        super().__init__()
        self.cfg = cfg
        h = list(cfg.encoder_hidden)

        self.encoder = _mlp([cfg.input_dim] + h)
        self.mu_head  = nn.Linear(h[-1], cfg.z_bio_dim)
        self.lv_head  = nn.Linear(h[-1], cfg.z_bio_dim)

        # Decoder input = z_bio + meta appended as raw values
        dec_in = cfg.z_bio_dim + cfg.meta_dim
        self.decoder = _mlp([dec_in] + h[::-1] + [cfg.input_dim])

        self.n_latent = cfg.z_bio_dim
        self.n_genes  = cfg.input_dim

    # ── encode / decode ────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.mu_head(h), self.lv_head(h)

    def decode(self, z_bio: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z_bio, meta], dim=1))

    def forward(self, x: torch.Tensor, meta: torch.Tensor):
        mu, lv = self.encode(x)
        z = mu + (0.5 * lv).exp() * torch.randn_like(mu)
        return self.decode(z, meta), mu, lv

    # ── ELBO ───────────────────────────────────────────────────────────────

    def elbo(self, x: torch.Tensor, meta: torch.Tensor = None) -> tuple[torch.Tensor, dict]:
        x_hat, mu, lv = self.forward(x, meta)
        recon = F.mse_loss(x_hat, x)
        kl_per_dim = -0.5 * (1 + lv - mu.pow(2) - lv.exp())
        kl = kl_per_dim.clamp(min=self.cfg.free_bits).sum(dim=-1).mean()
        loss = recon + self.cfg.beta * kl
        return loss, {"recon": recon.item(), "kl": kl.item()}

    # ── LatentModel protocol (probe / pipeline) ────────────────────────────

    @property
    def name(self) -> str:
        return f"meta_injection_vae_K{self.n_latent}"

    def encode_np(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(
                torch.from_numpy(x.astype(np.float32)).to(next(self.parameters()).device)
            )
        return mu.cpu().numpy()

    def flip(self, x: np.ndarray, meta_original: np.ndarray,
             meta_target: np.ndarray) -> np.ndarray:
        """Encode x to z_bio, decode with meta_target instead of meta_original."""
        dev = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(torch.from_numpy(x.astype(np.float32)).to(dev))
            x_hat = self.decode(mu,
                                torch.from_numpy(meta_target.astype(np.float32)).to(dev))
        return x_hat.cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════
# FiLM-conditioned variant
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FiLMMetaInjectionConfig:
    input_dim: int
    meta_dim: int
    z_bio_dim: int = 12
    encoder_hidden: tuple[int, ...] = (512, 256)
    decoder_hidden: tuple[int, ...] = (256, 512)
    meta_embed_dim: int = 128
    beta: float = 1e-3
    free_bits: float = 0.1
    lambda_tc: float = 0.0  # total-correlation weight; 0 = disabled


def _log_normal(z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Element-wise log N(z; mu, exp(logvar))."""
    return -0.5 * (logvar + (z - mu).pow(2) / logvar.exp() + math.log(2 * math.pi))


def tc_minibatch(z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Minibatch estimate of total correlation (Chen et al. 2018 β-TCVAE).

    TC = E_q[log q(z) - Σ_k log q(z_k)]

    Uses the minibatch-weighted estimator: treats the empirical batch as a
    proxy for the aggregate posterior q(z) = E_{p(x)}[q(z|x)].

    z, mu, logvar: (N, K)
    Returns: scalar TC estimate.
    """
    N = z.shape[0]
    # (N, 1, K) vs (1, N, K) → (N, N, K) per-element log q(z_i | x_j)
    log_qz_x = _log_normal(z.unsqueeze(1), mu.unsqueeze(0), logvar.unsqueeze(0))

    # log q(z_i) ≈ logsumexp_j[Σ_k log q(z_ik|x_j)] - log N
    log_qz_x_joint = log_qz_x.sum(-1)               # (N, N) joint over dims
    log_qz = torch.logsumexp(log_qz_x_joint, dim=1) - math.log(N)   # (N,)

    # log Π_k q(z_ik) = Σ_k logsumexp_j log q(z_ik|x_j) - log N
    log_qzk = torch.logsumexp(log_qz_x, dim=1) - math.log(N)        # (N, K)
    log_qz_product = log_qzk.sum(-1)                                  # (N,)

    return (log_qz - log_qz_product).mean()


class _FiLMDecoder(nn.Module):
    """MLP decoder whose hidden layers are scaled/shifted by FiLM from meta_emb.

    For layer l:  h = linear(h_prev);  h = gamma_l * h + beta_l;  h = GELU(h)
    FiLM scales are initialised to 1, biases to 0 (identity start).
    """

    def __init__(self, z_dim: int, meta_embed_dim: int,
                 hidden_dims: tuple[int, ...], out_dim: int):
        super().__init__()
        dims = [z_dim] + list(hidden_dims)
        self.linears = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.film_gamma = nn.ModuleList(
            [nn.Linear(meta_embed_dim, d) for d in hidden_dims]
        )
        self.film_beta = nn.ModuleList(
            [nn.Linear(meta_embed_dim, d) for d in hidden_dims]
        )
        self.out_proj = nn.Linear(dims[-1], out_dim)

        for g in self.film_gamma:
            nn.init.zeros_(g.weight)
            nn.init.ones_(g.bias)
        for b in self.film_beta:
            nn.init.zeros_(b.weight)
            nn.init.zeros_(b.bias)

    def forward(self, z: torch.Tensor, meta_emb: torch.Tensor) -> torch.Tensor:
        h = z
        for i, lin in enumerate(self.linears):
            h = lin(h)
            h = self.film_gamma[i](meta_emb) * h + self.film_beta[i](meta_emb)
            h = F.gelu(h)
        return self.out_proj(h)


class FiLMMetaInjectionVAE(nn.Module):
    """FiLM-conditioned MetaInjection VAE.

    Encoder sees only x → z_bio.
    Meta is embedded by a 2-layer MLP, then applied via FiLM at every decoder layer.
    Optionally adds β-TCVAE total-correlation penalty to disentangle z_bio dims.
    """

    def __init__(self, cfg: FiLMMetaInjectionConfig):
        super().__init__()
        self.cfg = cfg
        h = list(cfg.encoder_hidden)

        enc_layers: list[nn.Module] = []
        in_d = cfg.input_dim
        for out_d in h:
            enc_layers += [nn.Linear(in_d, out_d), nn.LayerNorm(out_d), nn.GELU()]
            in_d = out_d
        self.encoder    = nn.Sequential(*enc_layers)
        self.mu_head    = nn.Linear(h[-1], cfg.z_bio_dim)
        self.lv_head    = nn.Linear(h[-1], cfg.z_bio_dim)

        self.meta_embed = nn.Sequential(
            nn.Linear(cfg.meta_dim, cfg.meta_embed_dim), nn.GELU(),
            nn.Linear(cfg.meta_embed_dim, cfg.meta_embed_dim),
        )
        self.decoder = _FiLMDecoder(
            cfg.z_bio_dim, cfg.meta_embed_dim, cfg.decoder_hidden, cfg.input_dim
        )

        self.n_latent = cfg.z_bio_dim
        self.n_genes  = cfg.input_dim

    # ── encode / decode ────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.mu_head(h), self.lv_head(h)

    def decode(self, z_bio: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_bio, self.meta_embed(meta))

    def forward(self, x: torch.Tensor, meta: torch.Tensor):
        mu, lv = self.encode(x)
        z = mu + (0.5 * lv).exp() * torch.randn_like(mu)
        return self.decode(z, meta), mu, lv, z

    # ── ELBO ───────────────────────────────────────────────────────────────

    def elbo(self, x: torch.Tensor, meta: torch.Tensor,
             beta: Optional[float] = None,
             lambda_tc: Optional[float] = None) -> tuple[torch.Tensor, dict]:
        if beta      is None: beta      = self.cfg.beta
        if lambda_tc is None: lambda_tc = self.cfg.lambda_tc

        mu, lv = self.encode(x)
        z = mu + (0.5 * lv).exp() * torch.randn_like(mu)
        x_hat = self.decode(z, meta)

        recon = F.mse_loss(x_hat, x)
        kl_per_dim = -0.5 * (1.0 + lv - mu.pow(2) - lv.exp())
        kl = kl_per_dim.clamp(min=self.cfg.free_bits).sum(-1).mean()

        tc_val = tc_minibatch(z, mu, lv) if lambda_tc > 0 else torch.zeros(1, device=x.device)
        loss = recon + beta * kl + lambda_tc * tc_val

        return loss, {
            "recon": recon.item(),
            "kl":    kl.item(),
            "tc":    tc_val.item() if isinstance(tc_val, torch.Tensor) else float(tc_val),
        }

    # ── LatentModel protocol ───────────────────────────────────────────────

    @property
    def name(self) -> str:
        return f"film_meta_injection_vae_K{self.n_latent}"

    def encode_np(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(
                torch.from_numpy(x.astype(np.float32)).to(next(self.parameters()).device)
            )
        return mu.cpu().numpy()

    def flip(self, x: np.ndarray, meta_original: np.ndarray,
             meta_target: np.ndarray) -> np.ndarray:
        """Encode x → z_bio, decode with meta_target."""
        dev = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(torch.from_numpy(x.astype(np.float32)).to(dev))
            x_hat = self.decode(mu, torch.from_numpy(meta_target.astype(np.float32)).to(dev))
        return x_hat.cpu().numpy()

    def gene_loadings(self, meta_ref: np.ndarray, eps: float = 1e-2) -> np.ndarray:
        """Decoder Jacobian at z=0 for each z_bio dim.

        Uses finite differences at z=0 with the reference meta vector.
        Returns (K, G) array: loading[k, g] = ∂x̂_g / ∂z_k |_{z=0, meta=meta_ref}
        """
        dev = next(self.parameters()).device
        self.eval()
        meta_t = torch.from_numpy(meta_ref.astype(np.float32)).to(dev).unsqueeze(0)
        z0 = torch.zeros(1, self.n_latent, device=dev)
        with torch.no_grad():
            x0 = self.decode(z0, meta_t).squeeze(0)
            loadings = []
            for k in range(self.n_latent):
                zk = z0.clone()
                zk[0, k] = eps
                xk = self.decode(zk, meta_t).squeeze(0)
                loadings.append(((xk - x0) / eps).cpu().numpy())
        return np.stack(loadings, axis=0)   # (K, G)
