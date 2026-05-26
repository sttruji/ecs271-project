"""Q52 — Additive VAE (DRVI-style) on GTEx whole blood.

Architecture: x̂ = Σ_k f_k(z_k)  where each f_k: ℝ → ℝ^G is a small
              per-dim MLP (hidden=64). Nonlinearities act at the GENE level,
              not the donor level. No cross-dim interaction in the decoder.

Reconstruction loss: Laplace (L1) — the continuous analog of NB overdispersion
              handling for raw counts. L1 corresponds to a Laplace prior on
              reconstruction residuals and down-weights outlier genes (high-CPM
              genes like ACTB that dominate MSE).

Compared against: PCA-50 and standard MSE-VAE (same encoder, entangled decoder).

Linear probes: SMTSISCH (Ridge R²), DTHHRDY (LR balanced-acc), AGE_mid (Ridge R²).
Expected result: if additive structure better disentangles biological factors,
probe scores should match or exceed PCA-50 with far fewer dims.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.data import load_gtex_blood, load_metadata
from pipeline.adapters import PCAModel
from pipeline.latent import metadata_linear_probe

CKPT = "/Users/rls/Desktop/programming-projects/single-cell/bulk-project/analysis/14_cross_modality_vae/cross_modality_vae.pt"
OUT = ROOT / "analysis" / "results" / "q52_additive_vae"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

N_LATENT = 12
EPOCHS = 300
BATCH = 64
LR = 1e-3
BETA = 1e-3
FREE_BITS = 0.1
BETA_WARMUP_EPOCHS = 100  # β annealed 0→BETA over first 100 epochs


# ── Standard MSE-VAE baseline (entangled decoder) ─────────────────────────

class _MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [nn.LayerNorm(dims[i + 1]), nn.GELU()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class StandardVAE(nn.Module):
    name = f"standard_vae_K{N_LATENT}"

    def __init__(self, n_genes):
        super().__init__()
        self.encoder = _MLP([n_genes, 512, 256])
        self.mu_head = nn.Linear(256, N_LATENT)
        self.lv_head = nn.Linear(256, N_LATENT)
        self.decoder = _MLP([N_LATENT, 256, 512, n_genes])
        self.n_latent = N_LATENT
        self.n_genes = n_genes

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_head(h), self.lv_head(h)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + (0.5 * lv).exp() * torch.randn_like(mu)
        return self.decoder(z), mu, lv

    def elbo(self, x):
        x_hat, mu, lv = self.forward(x)
        recon = (x - x_hat).pow(2).mean()
        kl = (-0.5 * (1 + lv - mu.pow(2) - lv.exp())).clamp(min=FREE_BITS).sum(-1).mean()
        return recon + BETA * kl, {"recon": recon.item(), "kl": kl.item()}

    def encode_np(self, x):
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(torch.from_numpy(x.astype(np.float32)).to(DEVICE))
        return mu.cpu().numpy()


# ── training loop ─────────────────────────────────────────────────────────

def _beta_schedule(ep: int, beta_target: float, warmup: int) -> float:
    """Linear β warmup from 0 to beta_target over `warmup` epochs."""
    return beta_target * min(1.0, ep / warmup)


def train_model(model, x_train: np.ndarray, tag: str,
                beta_target: float = BETA, warmup: int = 0) -> list[dict]:
    model.to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ds = TensorDataset(torch.from_numpy(x_train.astype(np.float32)))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)
    log = []
    for ep in range(1, EPOCHS + 1):
        ep_recon = ep_kl = 0.0
        beta = _beta_schedule(ep, beta_target, warmup) if warmup > 0 else beta_target
        model.train()
        for (xb,) in dl:
            xb = xb.to(DEVICE)
            # Pass beta override for models that support it; fall back to elbo()
            if hasattr(model, "elbo_beta"):
                loss, parts = model.elbo_beta(xb, beta)
            else:
                loss, parts = model.elbo(xb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_recon += parts["recon"]
            ep_kl += parts["kl"]
        sched.step()
        if ep % 50 == 0 or ep == 1:
            nb = len(dl)
            print(f"  [{tag}] ep {ep:3d}/{EPOCHS}  β={beta:.1e}  "
                  f"recon={ep_recon/nb:.4f}  kl={ep_kl/nb:.4f}")
            log.append({"epoch": ep, "beta": beta,
                        "recon": ep_recon / nb, "kl": ep_kl / nb})
    return log


# ── main ──────────────────────────────────────────────────────────────────

def main():
    print("Loading GTEx blood ...")
    gtex = load_gtex_blood(checkpoint_path=CKPT)
    meta = load_metadata(gtex.sample_ids)
    X = gtex.expr_scaled  # (803, 11374) standardised log-CPM
    n_genes = X.shape[1]

    # 80/20 split (deterministic)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X))
    n_test = int(0.2 * len(X))
    train_idx, test_idx = perm[n_test:], perm[:n_test]
    X_train, X_test = X[train_idx], X[test_idx]

    results = {}

    # ── PCA-50 baseline ───────────────────────────────────────────────────
    print("\n── PCA-50 baseline ──")
    pca = PCAModel(latent_dim=50)
    pca.fit(X_train)
    z_pca = pca.encode(X)
    results["pca50"] = {"probes": metadata_linear_probe(z_pca, meta)}
    print(f"  probes: {results['pca50']['probes']}")

    # ── Standard VAE (MSE, entangled) ─────────────────────────────────────
    print(f"\n── Standard VAE K={N_LATENT} (MSE, entangled decoder) ──")
    vae_std = StandardVAE(n_genes)
    log_std = train_model(vae_std, X_train, "std_vae")
    vae_std.eval()
    z_std = vae_std.encode_np(X)
    results["standard_vae"] = {
        "n_active": int((z_std.var(0) > 0.01).sum()),
        "probes": metadata_linear_probe(z_std, meta),
        "train_log": log_std,
    }
    print(f"  active dims: {results['standard_vae']['n_active']}")
    print(f"  probes: {results['standard_vae']['probes']}")

    # ── Additive VAE (L1, per-dim decoder) ────────────────────────────────
    print(f"\n── Additive VAE K={N_LATENT} (Laplace/L1, additive decoder) ──")
    from models.drvi_model import AdditiveVAE, AdditiveVAEConfig
    cfg = AdditiveVAEConfig(
        input_dim=n_genes,
        n_latent=N_LATENT,
        encoder_hidden=(512, 256),
        decoder_hidden=32,
        beta=BETA,
        free_bits=FREE_BITS,
    )
    vae_add = AdditiveVAE(cfg)
    log_add = train_model(vae_add, X_train, "add_vae",
                          beta_target=BETA, warmup=BETA_WARMUP_EPOCHS)
    vae_add.eval()
    z_add = vae_add.encode_np(X)
    results["additive_vae"] = {
        "n_active": int((z_add.var(0) > 0.01).sum()),
        "probes": metadata_linear_probe(z_add, meta),
        "train_log": log_add,
    }
    print(f"  active dims: {results['additive_vae']['n_active']}")
    print(f"  probes: {results['additive_vae']['probes']}")

    # ── summary table ──────────────────────────────────────────────────────
    print("\n── Comparison ──")
    header = f"{'model':<22} {'K':>3}  {'SMTSISCH R²':>12}  {'AGE_mid R²':>11}  {'DTHHRDY bal-acc':>16}  {'chance':>7}"
    print(header)
    print("-" * len(header))
    for mname, r in results.items():
        p = r["probes"]
        k = {"pca50": 50, "standard_vae": N_LATENT, "additive_vae": N_LATENT}[mname]
        isch = p.get("SMTSISCH", {}).get("cv_r2", float("nan"))
        age  = p.get("AGE_mid", {}).get("cv_r2", float("nan"))
        hard = p.get("DTHHRDY", {}).get("cv_balanced_acc", float("nan"))
        chance = p.get("DTHHRDY", {}).get("chance", 0.2)
        print(f"{mname:<22} {k:>3}  {isch:>12.3f}  {age:>11.3f}  {hard:>16.3f}  {chance:>7.2f}")

    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {OUT}/results.json")


if __name__ == "__main__":
    main()
