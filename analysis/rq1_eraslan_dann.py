"""
RQ1 — Eraslan paired bulk/sc alignment with DANN.

Uses eraslan_paired.npz which has:
  bulk_x (92, 11374), bulk_donor, bulk_tissue
  sn_x   (261, 11374), sn_donor, sn_tissue

Runs 4 experiments in sequence:
  0. Raw gene-space cosine similarity (no model)
  1. Basic VAE baseline
  2. VAE + dropout masking (50%)
  3. VAE + dropout + DANN (gradient reversal)

Evaluates top-1 NN accuracy: for each (donor, tissue) sn sample,
does its paired bulk sample rank first among all bulk samples?

No GPU needed (MPS optional, CPU fallback).
"""
from __future__ import annotations
import sys, json, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print(f"Device: {DEVICE}")

# ── Data ─────────────────────────────────────────────────────────────────────
DATA = np.load("/Users/rls/ecs271/data/sc/eraslan/eraslan_paired.npz",
               allow_pickle=True)
bulk_x   = DATA["bulk_x"].astype(np.float32)   # (92, 11374)
bulk_don = DATA["bulk_donor"]
bulk_tis = DATA["bulk_tissue"]
sn_x     = DATA["sn_x"].astype(np.float32)     # (261, 11374)
sn_don   = DATA["sn_donor"]
sn_tis   = DATA["sn_tissue"]
genes    = DATA["shared_genes"]

print(f"Bulk: {bulk_x.shape}, SN: {sn_x.shape}, Genes: {len(genes)}")

# Build (donor, tissue) pair index
bulk_key = np.array([f"{d}|{t}" for d, t in zip(bulk_don, bulk_tis)])
sn_key   = np.array([f"{d}|{t}" for d, t in zip(sn_don, sn_tis)])
matched_sn  = [i for i, k in enumerate(sn_key)   if k in bulk_key]
matched_bulk_idx = {k: i for i, k in enumerate(bulk_key)}
print(f"Matched SN samples: {len(matched_sn)} / {len(sn_key)}")

# Data is already log-normalized (values in [-22, +22]). Just use directly.
# Replace any residual NaNs/infs with 0, then select HVGs by variance.
bulk_norm = np.nan_to_num(bulk_x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
sn_norm   = np.nan_to_num(sn_x.astype(np.float32),   nan=0.0, posinf=0.0, neginf=0.0)

# ── Select top-2000 HVGs on bulk ─────────────────────────────────────────────
gene_var = bulk_norm.var(axis=0)
top_idx  = np.argsort(gene_var)[-2000:]
bulk_h   = bulk_norm[:, top_idx]
sn_h     = sn_norm[:, top_idx]
N_GENES  = bulk_h.shape[1]
print(f"HVG subset: {N_GENES} genes  (data already log-normalized)")

# ── Evaluation helper ─────────────────────────────────────────────────────────
def cosine_nn_accuracy(z_bulk, z_sn, sn_indices, label=""):
    """top-1 / top-3 / top-5 NN accuracy (cosine) for matched sn samples."""
    z_bulk = F.normalize(torch.tensor(z_bulk, dtype=torch.float32), dim=1)
    z_sn   = F.normalize(torch.tensor(z_sn,   dtype=torch.float32), dim=1)
    sim    = z_sn[sn_indices] @ z_bulk.T        # (n_matched, n_bulk)
    ranks  = sim.argsort(dim=1, descending=True) # (n_matched, n_bulk)

    top1 = top3 = top5 = 0
    mean_cos = 0.0
    for row, si in enumerate(sn_indices):
        true_bulk = matched_bulk_idx[sn_key[si]]
        rank_pos  = (ranks[row] == true_bulk).nonzero(as_tuple=True)[0].item()
        top1 += rank_pos == 0
        top3 += rank_pos < 3
        top5 += rank_pos < 5
        mean_cos += sim[row, true_bulk].item()
    n = len(sn_indices)
    result = dict(top1=top1/n, top3=top3/n, top5=top5/n,
                  mean_paired_cos=mean_cos/n, n=n)
    print(f"  {label:35s} top1={result['top1']:.3f}  top3={result['top3']:.3f}"
          f"  top5={result['top5']:.3f}  paired_cos={result['mean_paired_cos']:.3f}")
    return result

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 0: Raw gene space
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== EXP 0: Raw HVG gene space ===")
r0 = cosine_nn_accuracy(bulk_h, sn_h, matched_sn, "raw HVG")

print("\n=== EXP 0b: PCA-50 ===")
pca = PCA(50)
bulk_pca = pca.fit_transform(bulk_h).astype(np.float32)
sn_pca   = pca.transform(sn_h).astype(np.float32)
r0b = cosine_nn_accuracy(bulk_pca, sn_pca, matched_sn, "PCA-50")

# ─────────────────────────────────────────────────────────────────────────────
# VAE building blocks
# ─────────────────────────────────────────────────────────────────────────────
Z_DIM = 32

def make_encoder(n_genes, z_dim):
    return nn.Sequential(
        nn.Linear(n_genes, 512), nn.LayerNorm(512), nn.GELU(),
        nn.Linear(512, 256),     nn.LayerNorm(256), nn.GELU(),
    ), nn.Linear(256, z_dim), nn.Linear(256, z_dim)

def make_decoder(n_genes, z_dim):
    return nn.Sequential(
        nn.Linear(z_dim, 256), nn.LayerNorm(256), nn.GELU(),
        nn.Linear(256, 512),   nn.LayerNorm(512), nn.GELU(),
        nn.Linear(512, n_genes)
    )

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam = lam; return x
    @staticmethod
    def backward(ctx, g): return -ctx.lam * g, None

class Discriminator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, z, lam=1.0):
        z_rev = GradReverse.apply(z, lam)
        return self.net(z_rev).squeeze(1)

class SimpleVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_body, self.mu_h, self.lv_h = make_encoder(N_GENES, Z_DIM)
        self.decoder = make_decoder(N_GENES, Z_DIM)

    def encode(self, x):
        h = self.enc_body(x); return self.mu_h(h), self.lv_h(h)

    def decode(self, z): return self.decoder(z)

    def forward(self, x, mask_rate=0.0):
        if mask_rate > 0 and self.training:
            x = x * (torch.rand_like(x) > mask_rate).float()
        mu, lv = self.encode(x)
        z = mu + (0.5 * lv).exp() * torch.randn_like(mu)
        return self.decode(z), mu, lv

    def elbo(self, x, mask_rate=0.0, beta=1e-3, fb=0.2):
        xhat, mu, lv = self.forward(x, mask_rate)
        recon = F.mse_loss(xhat, x)
        kl = (-0.5 * (1 + lv - mu.pow(2) - lv.exp())).clamp(min=fb).sum(-1).mean()
        return recon + beta * kl

def get_z(model, x_np):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(x_np, dtype=torch.float32).to(DEVICE)
        mu, _ = model.encode(x)
    return mu.cpu().numpy()

def train_vae(mask_rate=0.0, dann=False, epochs=120, lr=3e-4,
              batch_size=64, label="vae"):
    model = SimpleVAE().to(DEVICE)
    disc  = Discriminator(Z_DIM).to(DEVICE) if dann else None

    opt_vae  = torch.optim.AdamW(model.parameters(), lr=lr)
    opt_disc = torch.optim.AdamW(disc.parameters(), lr=lr*0.5) if dann else None

    bulk_t = torch.tensor(bulk_h, dtype=torch.float32)
    sn_t   = torch.tensor(sn_h,   dtype=torch.float32)

    dl = DataLoader(TensorDataset(bulk_t), batch_size=batch_size, shuffle=True)
    # sc pool (for DANN domain batches)
    sn_tensor = sn_t.to(DEVICE)

    for ep in range(epochs):
        model.train()
        if dann: disc.train()
        lam = (2.0 / (1 + math.exp(-10 * ep / epochs)) - 1) if dann else 0.0

        for (xb,) in dl:
            xb = xb.to(DEVICE)
            opt_vae.zero_grad()

            elbo = model.elbo(xb, mask_rate=mask_rate)

            dann_loss = torch.tensor(0.0)
            if dann:
                # sample random sc batch same size as bulk batch
                idx_sc = torch.randperm(len(sn_tensor))[:len(xb)]
                xsc    = sn_tensor[idx_sc]
                with torch.no_grad(): mu_sc, _ = model.encode(xsc)
                mu_bulk, _ = model.encode(xb)

                # discriminator step
                opt_disc.zero_grad()
                logits_bulk = disc(mu_bulk.detach(), lam=lam)
                logits_sc   = disc(mu_sc.detach(),   lam=lam)
                d_loss = (F.binary_cross_entropy_with_logits(logits_bulk, torch.ones_like(logits_bulk))
                        + F.binary_cross_entropy_with_logits(logits_sc,   torch.zeros_like(logits_sc))) / 2
                d_loss.backward()
                opt_disc.step()

                # adversarial for encoder via GRL
                logits_bulk2 = disc(mu_bulk, lam=lam)
                logits_sc2   = disc(mu_sc,   lam=lam)
                dann_loss = (F.binary_cross_entropy_with_logits(logits_bulk2, torch.zeros_like(logits_bulk2))
                           + F.binary_cross_entropy_with_logits(logits_sc2,   torch.ones_like(logits_sc2))) / 2

            total = elbo + 0.3 * dann_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_vae.step()

        if (ep + 1) % 40 == 0:
            print(f"    ep {ep+1:3d}/{epochs}  elbo={elbo.item():.4f}"
                  + (f"  dann={dann_loss.item():.4f}  lam={lam:.3f}" if dann else ""))

    return model

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: Basic VAE
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== EXP 1: Basic VAE (no masking, no DANN) ===")
t0 = time.time()
m1 = train_vae(mask_rate=0.0, dann=False, epochs=150, label="vae_base")
z_bulk_1 = get_z(m1, bulk_h)
z_sn_1   = get_z(m1, sn_h)
r1 = cosine_nn_accuracy(z_bulk_1, z_sn_1, matched_sn, "VAE baseline")
print(f"  time: {time.time()-t0:.0f}s")

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: VAE + dropout masking
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== EXP 2: VAE + 50% dropout masking ===")
t0 = time.time()
m2 = train_vae(mask_rate=0.5, dann=False, epochs=150, label="vae_drop")
z_bulk_2 = get_z(m2, bulk_h)
z_sn_2   = get_z(m2, sn_h)
r2 = cosine_nn_accuracy(z_bulk_2, z_sn_2, matched_sn, "VAE + dropout mask")
print(f"  time: {time.time()-t0:.0f}s")

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3: VAE + dropout + DANN
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== EXP 3: VAE + dropout + DANN ===")
t0 = time.time()
m3 = train_vae(mask_rate=0.5, dann=True, epochs=150, label="vae_dann")
z_bulk_3 = get_z(m3, bulk_h)
z_sn_3   = get_z(m3, sn_h)
r3 = cosine_nn_accuracy(z_bulk_3, z_sn_3, matched_sn, "VAE + dropout + DANN")
print(f"  time: {time.time()-t0:.0f}s")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY — Eraslan paired bulk→sc top-1 NN accuracy")
print("="*70)
print(f"  Random baseline:               {1/len(bulk_h):.3f}  (1/{len(bulk_h)})")
for label, r in [("Raw HVG", r0), ("PCA-50", r0b),
                 ("VAE baseline", r1),
                 ("VAE + dropout", r2),
                 ("VAE + DANN", r3)]:
    print(f"  {label:35s} top1={r['top1']:.3f}  top3={r['top3']:.3f}  "
          f"paired_cos={r['mean_paired_cos']:.3f}")

results = {
    "raw_hvg": r0, "pca50": r0b,
    "vae_base": r1, "vae_dropout": r2, "vae_dann": r3
}
import os
os.makedirs("/Users/rls/ecs271/vae_health/analysis/results", exist_ok=True)
with open("/Users/rls/ecs271/vae_health/analysis/results/rq1_eraslan_dann.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved → analysis/results/rq1_eraslan_dann.json")
