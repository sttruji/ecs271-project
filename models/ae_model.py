"""Simple autoencoder baseline for gene-expression reconstruction."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class AutoEncoderConfig:
    input_dim: int
    latent_dim: int = 64
    hidden_dim: int = 256
    max_iter: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    random_state: int = 0
    validation_fraction: float = 0.1
    early_stopping: bool = True
    n_iter_no_change: int = 20


class SklearnAutoEncoder:
    """Dense MLP autoencoder implemented as X -> X regression."""

    def __init__(self, config: AutoEncoderConfig):
        self.config = config
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(
                            config.hidden_dim,
                            config.latent_dim,
                            config.hidden_dim,
                        ),
                        activation="relu",
                        solver="adam",
                        batch_size=config.batch_size,
                        learning_rate_init=config.learning_rate,
                        max_iter=config.max_iter,
                        random_state=config.random_state,
                        validation_fraction=config.validation_fraction,
                        early_stopping=config.early_stopping,
                        n_iter_no_change=config.n_iter_no_change,
                        verbose=False,
                    ),
                ),
            ]
        )

    def fit(self, x: np.ndarray) -> "SklearnAutoEncoder":
        x = self._validate_matrix(x)
        self.pipeline.fit(x, x)
        return self

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        x = self._validate_matrix(x)
        return self.pipeline.predict(x).astype(np.float32)

    def loss_curve(self) -> list[float]:
        mlp = self.pipeline.named_steps["mlp"]
        return [float(value) for value in getattr(mlp, "loss_curve_", [])]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path: Path) -> "SklearnAutoEncoder":
        with path.open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, SklearnAutoEncoder):
            raise TypeError(f"Expected SklearnAutoEncoder, got {type(model).__name__}")
        return model

    def _validate_matrix(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected a 2D matrix, got shape {x.shape}")
        if x.shape[1] != self.config.input_dim:
            raise ValueError(
                f"Expected {self.config.input_dim} features, got {x.shape[1]}"
            )
        return x
