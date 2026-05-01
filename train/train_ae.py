"""Training utilities for the simple autoencoder baseline."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from models.ae_model import AutoEncoderConfig, SklearnAutoEncoder


def train_autoencoder(args: Namespace) -> dict[str, object]:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x = np.load(data_dir / "bulk_log_cpm.npy").astype(np.float32)
    train_indices, val_indices = train_val_split(
        n_samples=x.shape[0],
        validation_fraction=args.validation_fraction,
        random_state=args.random_state,
    )
    x_train = x[train_indices]
    x_val = x[val_indices]

    config = AutoEncoderConfig(
        input_dim=x.shape[1],
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
        validation_fraction=args.validation_fraction,
        early_stopping=not args.no_early_stopping,
        n_iter_no_change=args.n_iter_no_change,
    )

    model = SklearnAutoEncoder(config)
    model.fit(x_train)

    train_reconstruction = model.reconstruct(x_train)
    val_reconstruction = model.reconstruct(x_val)
    train_mse = mean_squared_error(x_train, train_reconstruction)
    val_mse = mean_squared_error(x_val, val_reconstruction)
    train_mae = mean_absolute_error(x_train, train_reconstruction)
    val_mae = mean_absolute_error(x_val, val_reconstruction)

    checkpoint_path = output_dir / "ae_model.pkl"
    model.save(checkpoint_path)

    metrics = {
        "model": "ae",
        "checkpoint": str(checkpoint_path),
        "input_dim": int(x.shape[1]),
        "n_bulk_samples": int(x.shape[0]),
        "n_train_samples": int(len(train_indices)),
        "n_val_samples": int(len(val_indices)),
        "latent_dim": int(args.latent_dim),
        "hidden_dim": int(args.hidden_dim),
        "max_iter": int(args.max_iter),
        "train_mse": train_mse,
        "val_mse": val_mse,
        "train_mae": train_mae,
        "val_mae": val_mae,
        "loss_curve": model.loss_curve(),
    }

    with (output_dir / "ae_train_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    np.savetxt(output_dir / "ae_train_indices.txt", train_indices, fmt="%d")
    np.savetxt(output_dir / "ae_val_indices.txt", val_indices, fmt="%d")

    return metrics


def train_val_split(
    n_samples: int,
    validation_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    rng = np.random.default_rng(random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    n_val = max(1, int(round(n_samples * validation_fraction)))
    return indices[n_val:], indices[:n_val]


def mean_squared_error(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    return float(np.mean((x_true - x_pred) ** 2))


def mean_absolute_error(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(x_true - x_pred)))
