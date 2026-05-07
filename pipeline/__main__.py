"""CLI: python -m pipeline --model {vae,pca,ae} [--out-dir ...]

Trains/loads a model and runs the full evaluation suite, writing results to
out_dir. Useful as a smoke test or for ad-hoc model comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from . import EvalConfig, run_evaluation
from .adapters import PCAModel, TorchAEAdapter, TrainedCrossModalityVAE
from .data import load_gtex_blood


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run vae_health evaluation pipeline.")
    p.add_argument("--model", required=True, choices=["vae", "pca", "ae"],
                   help="vae = trained cross-modality VAE checkpoint, "
                        "pca = sklearn PCA(64), ae = train a 3-layer MLP autoencoder")
    p.add_argument("--out-dir", default="eval_output")
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--skip-enrichment", action="store_true")
    p.add_argument("--checkpoint", default=None,
                   help="VAE checkpoint path (defaults to shared data/models/cross_modality_vae.pt)")
    return p.parse_args()


class _MLPAE(nn.Module):
    def __init__(self, n_genes, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2), nn.Dropout(0.1),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2), nn.Dropout(0.1),
            nn.Linear(512, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2), nn.Dropout(0.1),
            nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2), nn.Dropout(0.1),
            nn.Linear(1024, n_genes),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def _train_ae(gtex, latent_dim: int, epochs: int, device: str):
    model = _MLPAE(gtex.n_shared_genes, latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_t = torch.from_numpy(gtex.expr_scaled)
    loader = DataLoader(TensorDataset(train_t.to(device)), batch_size=64, shuffle=True, drop_last=True)
    for ep in range(1, epochs + 1):
        model.train()
        for (xb,) in loader:
            opt.zero_grad()
            xh, _ = model(xb)
            F.mse_loss(xh, xb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        if ep % 50 == 0:
            print(f"  AE-3 ep {ep}/{epochs}")
    return model


def main() -> int:
    args = parse_args()
    print(f"Loading GTEx whole blood (this preprocesses ~803 × 11k genes) ...")
    gtex = load_gtex_blood(checkpoint_path=args.checkpoint)

    if args.model == "vae":
        ckpt = args.checkpoint or "/Users/rls/ecs271/data/models/cross_modality_vae.pt"
        model = TrainedCrossModalityVAE(checkpoint_path=ckpt)
        name = "cross_modality_vae"
    elif args.model == "pca":
        pca = PCAModel(latent_dim=args.latent_dim, name=f"pca_{args.latent_dim}")
        pca.fit(gtex.expr_scaled)
        model = pca
        name = pca.name
    else:  # ae
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        net = _train_ae(gtex, latent_dim=args.latent_dim, epochs=args.epochs, device=device)
        model = TorchAEAdapter(net, latent_dim=args.latent_dim,
                               n_genes=gtex.n_shared_genes, name=f"ae3_d{args.latent_dim}",
                               device=device)
        name = model.name

    cfg = EvalConfig(
        name=name,
        out_dir=Path(args.out_dir),
        skip_enrichment=args.skip_enrichment,
        checkpoint_path=args.checkpoint,
    )
    summary = run_evaluation(model, cfg, gtex=gtex)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
