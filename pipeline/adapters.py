"""Adapter classes that wrap concrete models into the LatentModel protocol."""
from __future__ import annotations

import numpy as np
import torch

from .data import load_shared_genes


class TrainedCrossModalityVAE:
    """Wraps the trained checkpoint at data/models/cross_modality_vae.pt."""

    name = "cross_modality_vae"

    def __init__(self, checkpoint_path: str = "/Users/rls/ecs271/data/models/cross_modality_vae.pt",
                 modality: str = "bulk", device: str = "auto"):
        # Lazy-import the architecture from the analysis package's lib_model
        # (which is a self-contained mirror of the bulk-project model).
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
        from lib_model import load_trained  # noqa: E402

        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.modality = modality
        self._model, self.shared_genes, self.scaler_mean, self.scaler_std = load_trained(
            checkpoint_path, device=device
        )
        self.n_genes = int(self._model.n_genes)
        self.latent_dim = int(self._model.latent_dim)

    def encode(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(x.astype(np.float32)).to(self.device)
            enc = self._model.enc_bulk if self.modality == "bulk" else self._model.enc_sc
            mu, _ = enc(t)
        return mu.cpu().numpy()

    def decode(self, z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(z.astype(np.float32)).to(self.device)
            x = self._model.decoder(t)
        return x.cpu().numpy()


class PCAModel:
    """A LatentModel-conforming PCA wrapper.

    Calling .fit(train) before evaluating is required.
    """

    def __init__(self, latent_dim: int = 64, name: str | None = None):
        from sklearn.decomposition import PCA
        self.latent_dim = latent_dim
        self._pca = PCA(n_components=latent_dim)
        self.name = name or f"pca_{latent_dim}"
        self.n_genes: int = 0

    def fit(self, expr: np.ndarray) -> "PCAModel":
        self._pca.fit(expr)
        self.n_genes = expr.shape[1]
        return self

    def encode(self, x: np.ndarray) -> np.ndarray:
        return self._pca.transform(x).astype(np.float32)

    def decode(self, z: np.ndarray) -> np.ndarray:
        return self._pca.inverse_transform(z).astype(np.float32)


class TorchAEAdapter:
    """Wraps a torch.nn.Module that exposes .encoder and .decoder Sequentials."""

    def __init__(self, module, latent_dim: int, n_genes: int, name: str = "torch_ae",
                 device: str = "auto"):
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self._module = module.to(device).eval()
        self.latent_dim = int(latent_dim)
        self.n_genes = int(n_genes)
        self.name = name

    def encode(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(x.astype(np.float32)).to(self.device)
            z = self._module.encoder(t) if hasattr(self._module, "encoder") else self._module(t)[1]
        return z.cpu().numpy()

    def decode(self, z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(z.astype(np.float32)).to(self.device)
            x = self._module.decoder(t)
        return x.cpu().numpy()
