"""Model protocol — anything that walks like a LatentModel can be evaluated."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class LatentModel(Protocol):
    """An encoder/decoder pair operating on (n_samples, n_genes) float32.

    Implementations only need the two methods. The pipeline never touches
    internal model state directly.
    """

    name: str
    n_genes: int
    latent_dim: int

    def encode(self, x: np.ndarray) -> np.ndarray:  # (n, n_genes) -> (n, latent_dim)
        ...

    def decode(self, z: np.ndarray) -> np.ndarray:  # (n, latent_dim) -> (n, n_genes)
        ...
