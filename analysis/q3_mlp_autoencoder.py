#!/usr/bin/env python3
"""Q3 — 3-layer MLP autoencoder (and a sane VAE) on bulk reconstruction.

Trains two models on the same 80/20 GTEx whole-blood split as Q1, on the
same standardised log2(CPM+1) matrix in the 11,374-gene shared space:

  • AE-3:   x → Linear → ReLU → BN → Linear(latent) → Linear → ReLU → BN → Linear(x̂)
            i.e. 3 trainable Linear blocks, single 64-dim bottleneck, no KL.
  • VAE-β: same encoder/decoder shape but with reparameterised z and
            *small* KL weight (β=1e-3, no adversarial term). This is the
            apples-to-apples 'sane' VAE comparison for the trained
            cross-modality VAE which posterior-collapsed under β=1.0 + λ=0.1.

Both models trained on bulk only (no sc, no discriminator), 200 epochs,
Adam(1e-3), batch=64, MSE recon. We report:
  • Held-out R² / MSE / per-sample r / per-gene r — same metrics as Q1
  • Number of active latent dims (var > 0.01)
  • Side-by-side bar with Q1 numbers (mean / PCA-50/200/500 / cross-modality VAE)

Run:  python q3_mlp_autoencoder.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from lib_data import align_to_shared, load_gtex_blood, standardise  # noqa: E402
from lib_model import load_trained  # noqa: E402

CHECKPOINT = "/Users/rls/ecs271/data/models/cross_modality_vae.pt"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True, parents=True)
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True, parents=True)


# ── 3-layer plain AE ──────────────────────────────────────────────────────
class MLPAutoencoder(nn.Module):
    """Encoder: [G→1024→512→64].  Decoder: [64→512→1024→G]. 3 hidden layers each side."""

    def __init__(self, n_genes: int, latent_dim: int = 64):
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z


class SaneVAE(nn.Module):
    def __init__(self, n_genes: int, latent_dim: int = 64):
        super().__init__()
        self.enc_trunk = nn.Sequential(
            nn.Linear(n_genes, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2), nn.Dropout(0.1),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2), nn.Dropout(0.1),
        )
        self.mu = nn.Linear(512, latent_dim)
        self.logv = nn.Linear(512, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2), nn.Dropout(0.1),
            nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2), nn.Dropout(0.1),
            nn.Linear(1024, n_genes),
        )

    def forward(self, x: torch.Tensor):
        h = self.enc_trunk(x)
        mu = self.mu(h)
        logv = self.logv(h).clamp(-10, 4)
        if self.training:
            z = mu + (0.5 * logv).exp() * torch.randn_like(mu)
        else:
            z = mu
        return self.decoder(z), mu, logv


def metric_block(true: np.ndarray, pred: np.ndarray) -> dict:
    diff = true - pred
    var_total = float(true.var())
    var_resid = float(diff.var())
    r2 = float(1.0 - var_resid / var_total) if var_total > 0 else float("nan")
    per_sample_r = []
    for i in range(true.shape[0]):
        a = true[i] - true[i].mean()
        b = pred[i] - pred[i].mean()
        denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
        per_sample_r.append(float((a * b).sum() / denom) if denom else 0.0)
    per_sample_r = np.asarray(per_sample_r)
    return {
        "mse": float((diff ** 2).mean()),
        "r2_overall": r2,
        "per_sample_r_mean": float(per_sample_r.mean()),
        "per_sample_r_med": float(np.median(per_sample_r)),
        "per_sample_r_min": float(per_sample_r.min()),
    }


def train_ae(model, train_t, test_t, epochs=200, lr=1e-3, batch=64, device="cpu",
             beta=0.0, vae=False):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loader = DataLoader(TensorDataset(train_t.to(device)), batch_size=batch, shuffle=True, drop_last=True)
    history = {"train_mse": [], "test_mse": [], "kl": []}
    for ep in range(1, epochs + 1):
        model.train()
        ep_mse = 0.0
        ep_kl = 0.0
        n = 0
        for (xb,) in loader:
            opt.zero_grad()
            if vae:
                xh, mu, logv = model(xb)
                recon = F.mse_loss(xh, xb)
                kl = -0.5 * (1 + logv - mu.pow(2) - logv.exp()).sum(-1).mean()
                loss = recon + beta * kl
                ep_kl += float(kl)
            else:
                xh, _ = model(xb)
                recon = F.mse_loss(xh, xb)
                loss = recon
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_mse += float(recon)
            n += 1
        model.eval()
        with torch.no_grad():
            x_test = test_t.to(device)
            if vae:
                xh, _, _ = model(x_test)
            else:
                xh, _ = model(x_test)
            te_mse = float(F.mse_loss(xh, x_test))
        history["train_mse"].append(ep_mse / n)
        history["test_mse"].append(te_mse)
        if vae:
            history["kl"].append(ep_kl / n)
        if ep % 20 == 0 or ep == 1:
            tag = "VAE-β" if vae else "AE-3"
            extra = f"  KL={history['kl'][-1]:.3f}" if vae else ""
            print(f"  [{tag}] ep {ep:>3}/{epochs}  train MSE={history['train_mse'][-1]:.4f}  test MSE={te_mse:.4f}{extra}")
    return history


def main() -> int:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    torch.manual_seed(0)
    np.random.seed(0)

    print("Loading trained-VAE checkpoint metadata for shared genes / scaler ...")
    _, shared_genes, scaler_mean, scaler_std = load_trained(CHECKPOINT, device="cpu")

    print("Loading GTEx whole blood ...")
    expr_log, gene_names = load_gtex_blood()
    expr_aligned, _, _ = align_to_shared(expr_log, gene_names, shared_genes)
    expr_scaled = standardise(expr_aligned, scaler_mean, scaler_std)
    n_genes = expr_scaled.shape[1]

    rng = np.random.default_rng(0)
    perm = rng.permutation(expr_scaled.shape[0])
    n_test = int(round(0.2 * expr_scaled.shape[0]))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    train = expr_scaled[train_idx]
    test = expr_scaled[test_idx]
    train_t = torch.from_numpy(train)
    test_t = torch.from_numpy(test)
    print(f"  split: train={len(train_idx)}  test={len(test_idx)}  genes={n_genes}")

    # ── Train plain AE ────────────────────────────────────────────────────
    print("\nTraining 3-layer MLP autoencoder ...")
    ae = MLPAutoencoder(n_genes=n_genes, latent_dim=64)
    h_ae = train_ae(ae, train_t, test_t, epochs=200, device=device)

    # eval AE
    ae.eval()
    with torch.no_grad():
        xh_train, z_train_ae = ae(train_t.to(device))
        xh_test, z_test_ae = ae(test_t.to(device))
    pred_ae = xh_test.cpu().numpy()
    metrics_ae = metric_block(test, pred_ae)
    z_test_ae = z_test_ae.cpu().numpy()
    z_train_ae = z_train_ae.cpu().numpy()
    z_all_ae = np.vstack([z_train_ae, z_test_ae])
    var_per_dim_ae = z_all_ae.var(0)
    active_ae = int((var_per_dim_ae > 0.01).sum())
    print(f"  AE-3 metrics: {metrics_ae}")
    print(f"  AE-3 active dims: {active_ae}/64")

    # ── Train sane VAE (β=1e-3, no adv) ───────────────────────────────────
    print("\nTraining sane VAE (β=1e-3, no adversarial) ...")
    vae = SaneVAE(n_genes=n_genes, latent_dim=64)
    h_vae = train_ae(vae, train_t, test_t, epochs=200, device=device, beta=1e-3, vae=True)

    vae.eval()
    with torch.no_grad():
        xh_train, mu_train, _ = vae(train_t.to(device))
        xh_test, mu_test, _ = vae(test_t.to(device))
    pred_vae = xh_test.cpu().numpy()
    metrics_vae = metric_block(test, pred_vae)
    z_all_vae = np.vstack([mu_train.cpu().numpy(), mu_test.cpu().numpy()])
    var_per_dim_vae = z_all_vae.var(0)
    active_vae = int((var_per_dim_vae > 0.01).sum())
    print(f"  Sane-VAE metrics: {metrics_vae}")
    print(f"  Sane-VAE active dims: {active_vae}/64")

    # ── Reload Q1 numbers for the comparison plot ─────────────────────────
    q1 = json.loads((RESULTS / "q1_reconstruction.json").read_text())

    out = {
        "ae3": metrics_ae | {"active_dims": active_ae},
        "vae_beta_1e3": metrics_vae | {"active_dims": active_vae},
    }
    (RESULTS / "q3_mlp_autoencoder.json").write_text(json.dumps(out, indent=2))

    # ── Comparison figure ────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    methods = [("MEAN", q1["mean_baseline"]["r2_overall"], "#7d8590"),
               ("PCA-50", q1["pca_50"]["r2_overall"], "#58a6ff"),
               ("PCA-200", q1["pca_200"]["r2_overall"], "#58a6ff"),
               ("PCA-500", q1.get("pca_500", q1["pca_200"])["r2_overall"], "#58a6ff"),
               ("Cross-mod\nVAE\n(trained)", q1["vae"]["r2_overall"], "#f78166"),
               ("AE-3\n(this work)", metrics_ae["r2_overall"], "#3fb950"),
               ("VAE β=1e-3\n(this work)", metrics_vae["r2_overall"], "#3fb950")]
    labels = [m[0] for m in methods]
    r2 = [m[1] for m in methods]
    colors = [m[2] for m in methods]
    axes[0].bar(labels, r2, color=colors)
    axes[0].axhline(0, color="#7d8590", lw=0.5)
    axes[0].set_ylabel("Held-out R²")
    axes[0].set_title("Q1+Q3: Reconstruction R² across methods")
    for i, v in enumerate(r2):
        axes[0].text(i, v + 0.02 * (1 if v >= 0 else -1), f"{v:.2f}", ha="center", fontsize=9)
    axes[0].tick_params(axis="x", rotation=0, labelsize=8)

    # training curves
    epochs = np.arange(1, len(h_ae["train_mse"]) + 1)
    axes[1].plot(epochs, h_ae["train_mse"], color="#3fb950", label="AE-3 train")
    axes[1].plot(epochs, h_ae["test_mse"], color="#3fb950", ls="--", label="AE-3 test")
    axes[1].plot(epochs, h_vae["train_mse"], color="#58a6ff", label="VAE β=1e-3 train")
    axes[1].plot(epochs, h_vae["test_mse"], color="#58a6ff", ls="--", label="VAE β=1e-3 test")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE on standardised log2(CPM+1)")
    axes[1].set_title("Training curves (200 epochs, batch=64, Adam 1e-3)")
    axes[1].legend(fontsize=9)
    axes[1].set_yscale("log")

    fig.tight_layout()
    fig.savefig(OUT / "q3_mlp_autoencoder.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {RESULTS / 'q3_mlp_autoencoder.json'}")
    print(f"Wrote {OUT / 'q3_mlp_autoencoder.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
