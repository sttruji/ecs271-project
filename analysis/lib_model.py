"""Cross-modality VAE architecture.

Mirrors src/scripts/deconvolution/cross_modality_vae.py in the bulk-project so
the trained checkpoint at data/models/cross_modality_vae.pt loads cleanly.
Kept minimal: only the modules whose state-dict keys appear in the checkpoint.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _fc_block(in_dim: int, out_dim: int, dropout: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.LeakyReLU(0.2),
        nn.Dropout(dropout),
    )


class ModalityEncoder(nn.Module):
    def __init__(self, n_genes: int, hidden=(1024, 512), latent_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        dims = [n_genes] + list(hidden)
        self.net = nn.Sequential(*[
            _fc_block(dims[i], dims[i + 1], dropout) for i in range(len(dims) - 1)
        ])
        self.mu = nn.Linear(hidden[-1], latent_dim)
        self.logv = nn.Linear(hidden[-1], latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        return self.mu(h), self.logv(h).clamp(-10, 4)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, n_genes: int, hidden=(512, 1024), dropout: float = 0.1):
        super().__init__()
        dims = [latent_dim] + list(hidden)
        self.net = nn.Sequential(*[
            _fc_block(dims[i], dims[i + 1], dropout) for i in range(len(dims) - 1)
        ])
        self.out = nn.Linear(hidden[-1], n_genes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.out(self.net(z))


class DomainDiscriminator(nn.Module):
    def __init__(self, latent_dim: int, hidden=(128, 64)):
        super().__init__()
        dims = [latent_dim] + list(hidden)
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.LeakyReLU(0.2), nn.Dropout(0.1)]
        layers.append(nn.Linear(hidden[-1], 2))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class CrossModalityVAE(nn.Module):
    def __init__(self, n_genes: int, latent_dim: int = 64,
                 enc_hidden=(1024, 512), dec_hidden=(512, 1024),
                 disc_hidden=(128, 64), dropout: float = 0.1):
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.enc_bulk = ModalityEncoder(n_genes, enc_hidden, latent_dim, dropout)
        self.enc_sc = ModalityEncoder(n_genes, enc_hidden, latent_dim, dropout)
        self.decoder = Decoder(latent_dim, n_genes, dec_hidden, dropout)
        self.discriminator = DomainDiscriminator(latent_dim, disc_hidden)


def load_trained(checkpoint_path: str, device: str = "cpu"):
    """Load the trained checkpoint and rebuild the model.

    Returns (model, shared_genes, scaler_mean, scaler_std).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    n_genes = ckpt["scaler_mean"].shape[0]
    latent_dim = ckpt["args"]["latent_dim"]
    model = CrossModalityVAE(n_genes=n_genes, latent_dim=latent_dim)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model, ckpt["shared_genes"], ckpt["scaler_mean"], ckpt["scaler_std"]
