"""Training utilities for the vanilla VAE baseline."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.vae_model import (
    VAEConfig,
    VanillaVAE,
    reconstruct_numpy,
    save_vae_checkpoint,
    vae_loss,
)
from train.train_ae import mean_absolute_error, mean_squared_error, train_val_split


def train_vae(args: Namespace) -> dict[str, object]:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(args.random_state)
    device = select_device(args.device)

    x = np.load(data_dir / "bulk_log_cpm.npy").astype(np.float32)
    train_indices, val_indices = train_val_split(
        n_samples=x.shape[0],
        validation_fraction=args.validation_fraction,
        random_state=args.random_state,
    )
    x_train = x[train_indices]
    x_val = x[val_indices]

    feature_mean = x_train.mean(axis=0).astype(np.float32)
    feature_std = x_train.std(axis=0).astype(np.float32)
    feature_std = np.maximum(feature_std, 1e-6).astype(np.float32)
    x_train_scaled = ((x_train - feature_mean) / feature_std).astype(np.float32)
    x_val_scaled = ((x_val - feature_mean) / feature_std).astype(np.float32)

    config = VAEConfig(
        input_dim=x.shape[1],
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        beta=args.beta,
    )
    model = VanillaVAE(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train_scaled)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_tensor = torch.from_numpy(x_val_scaled).to(device)

    history = []
    for epoch in range(1, args.max_iter + 1):
        model.train()
        train_total = 0.0
        train_reconstruction = 0.0
        train_kl = 0.0
        n_train_batches = 0

        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstruction, mu, logvar = model(batch)
            loss, reconstruction_loss, kl_loss = vae_loss(
                batch,
                reconstruction,
                mu,
                logvar,
                beta=args.beta,
            )
            loss.backward()
            optimizer.step()

            train_total += float(loss.item())
            train_reconstruction += float(reconstruction_loss.item())
            train_kl += float(kl_loss.item())
            n_train_batches += 1

        model.eval()
        with torch.no_grad():
            val_reconstruction, val_mu, val_logvar = model(val_tensor)
            val_loss, val_reconstruction_loss, val_kl_loss = vae_loss(
                val_tensor,
                val_reconstruction,
                val_mu,
                val_logvar,
                beta=args.beta,
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_total / n_train_batches,
                "train_reconstruction_loss": train_reconstruction / n_train_batches,
                "train_kl_loss": train_kl / n_train_batches,
                "val_loss": float(val_loss.item()),
                "val_reconstruction_loss": float(val_reconstruction_loss.item()),
                "val_kl_loss": float(val_kl_loss.item()),
            }
        )

    train_reconstruction = reconstruct_numpy(model, x_train, feature_mean, feature_std, device)
    val_reconstruction = reconstruct_numpy(model, x_val, feature_mean, feature_std, device)

    metrics = {
        "model": "vae",
        "checkpoint": str(output_dir / "vae_model.pt"),
        "input_dim": int(x.shape[1]),
        "n_bulk_samples": int(x.shape[0]),
        "n_train_samples": int(len(train_indices)),
        "n_val_samples": int(len(val_indices)),
        "latent_dim": int(args.latent_dim),
        "hidden_dim": int(args.hidden_dim),
        "max_iter": int(args.max_iter),
        "beta": float(args.beta),
        "device": str(device),
        "train_mse": mean_squared_error(x_train, train_reconstruction),
        "val_mse": mean_squared_error(x_val, val_reconstruction),
        "train_mae": mean_absolute_error(x_train, train_reconstruction),
        "val_mae": mean_absolute_error(x_val, val_reconstruction),
        "history": history,
    }

    checkpoint_path = output_dir / "vae_model.pt"
    save_vae_checkpoint(checkpoint_path, model, feature_mean, feature_std, metrics)

    with (output_dir / "vae_train_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    np.savetxt(output_dir / "vae_train_indices.txt", train_indices, fmt="%d")
    np.savetxt(output_dir / "vae_val_indices.txt", val_indices, fmt="%d")

    return metrics


def select_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
