"""
RQ1 — Supervised contrastive (InfoNCE) on Eraslan paired data.

All previous methods (VAE, dropout, DANN) trained with NO paired supervision —
they never saw which bulk sample pairs with which sc sample.
This experiment uses the actual (donor, tissue) paired labels during training.

Experiments:
  0. Raw HVG baseline (copied)
  1. Supervised InfoNCE only (no reconstruction, just contrastive)
  2. VAE + supervised InfoNCE (reconstruction + contrastive)
  3. VAE + DANN + supervised InfoNCE (full pipeline)

For each, evaluate top-1 NN accuracy on held-out pairs (leave-one-donor-out).
"""
from __future__ import annotations
import json, time, math, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print(f"Device: {DEVICE}")

# ── Data ─────────────────────────────────────────────────────────────────────
DATA = np.load("/Users/rls/ecs271/data/sc/eraslan/eraslan_paired.npz",
               allow_pickle=True)
bulk_x   = np.nan_to_num(DATA["bulk_x"].astype(np.float32))   # (92, 11374)
bulk_don = DATA["bulk_donor"]
bulk_tis = DATA["bulk_tissue"]
sn_x     = np.nan_to_num(DATA["sn_x"].astype(np.float32))     # (261, 11374)
sn_don   = DATA["sn_donor"]
sn_tis   = DATA["sn_tissue"]

print(f"Bulk: {bulk_x.shape}, SN: {sn_x.shape}")

bulk_key = np.array([f"{d}|{t}" for d, t in zip(bulk_don, bulk_tis)])
sn_key   = np.array([f"{d}|{t}" for d, t in zip(sn_don, sn_tis)])
bulk_idx = {k: i for i, k in enumerate(bulk_key)}

# matched: sn indices that have a corresponding bulk
matched_sn   = np.array([i for i, k in enumerate(sn_key) if k in bulk_idx])
matched_bulk = np.array([bulk_idx[sn_key[i]] for i in matched_sn])
print(f"Matched pairs: {len(matched_sn)}")

# HVG on bulk
gene_var = bulk_x.var(axis=0)
top_idx  = np.argsort(gene_var)[-2000:]
bulk_h   = bulk_x[:, top_idx]   # (92, 2000)
sn_h     = sn_x[:, top_idx]     # (261, 2000)
N_GENES  = bulk_h.shape[1]

# ── Evaluation ───────────────────────────────────────────────────────────────
def nn_accuracy(z_bulk: np.ndarray, z_sn: np.ndarray, label=""):
    """Top-1/3/5 cosine NN accuracy over the 256 matched pairs."""
    zb = F.normalize(torch.tensor(z_bulk, dtype=torch.float32), dim=1)
    zs = F.normalize(torch.tensor(z_sn[matched_sn], dtype=torch.float32), dim=1)
    sim   = zs @ zb.T   # (256, 92)
    ranks = sim.argsort(dim=1, descending=True)
    top1 = top3 = top5 = 0
    paired_cos = 0.0
    for row, si in enumerate(matched_sn):
        true_bi = matched_bulk[row]
        rp = (ranks[row] == true_bi).nonzero(as_tuple=True)[0].item()
        top1 += rp == 0; top3 += rp < 3; top5 += rp < 5
        paired_cos += sim[row, true_bi].item()
    n = len(matched_sn)
    r = dict(top1=top1/n, top3=top3/n, top5=top5/n, paired_cos=paired_cos/n)
    print(f"  {label:40s} top1={r['top1']:.3f}  top3={r['top3']:.3f}"
          f"  top5={r['top5']:.3f}  paired_cos={r['paired_cos']:.3f}")
    return r

# ── EXP 0: Raw baseline ──────────────────────────────────────────────────────
print("\n=== EXP 0: Raw HVG (no model) ===")
r0 = nn_accuracy(bulk_h, sn_h, "raw HVG")

# ── Model components ─────────────────────────────────────────────────────────
Z_DIM = 64

class Encoder(nn.Module):
    def __init__(self, n_in, z_dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_in, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
        )
        self.mu_h = nn.Linear(256, z_dim)
        self.lv_h = nn.Linear(256, z_dim)

    def forward(self, x):
        h = self.body(x)
        return self.mu_h(h), self.lv_h(h)

class Decoder(nn.Module):
    def __init__(self, z_dim, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 512),   nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, n_out)
        )
    def forward(self, z): return self.net(z)

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam = lam; return x.clone()
    @staticmethod
    def backward(ctx, g): return -ctx.lam * g, None

class Discriminator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, z, lam=1.0):
        return self.net(GradReverse.apply(z, lam)).squeeze(1)

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.07):
    """NT-Xent / InfoNCE loss for N matched pairs (z1[i] paired with z2[i])."""
    N = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    # (2N, 2N) similarity matrix
    z_all = torch.cat([z1, z2], dim=0)
    sim = (z_all @ z_all.T) / temp
    # mask out self-similarity
    mask = torch.eye(2 * N, dtype=torch.bool, device=z1.device)
    sim = sim.masked_fill(mask, -1e9)
    # positive indices: i↔(i+N) and (i+N)↔i
    labels = torch.cat([torch.arange(N, 2*N), torch.arange(N)]).to(z1.device)
    return F.cross_entropy(sim, labels)

def get_z(enc, x_np, sample=False):
    enc.eval()
    with torch.no_grad():
        x = torch.tensor(x_np, dtype=torch.float32).to(DEVICE)
        mu, lv = enc(x)
        if sample:
            z = mu + (0.5 * lv).exp() * torch.randn_like(mu)
        else:
            z = mu
    return z.cpu().numpy()

# ── EXP 1: Pure supervised InfoNCE (no reconstruction) ──────────────────────
print("\n=== EXP 1: Supervised InfoNCE only (paired labels, no recon) ===")
t0 = time.time()

enc1 = Encoder(N_GENES, Z_DIM).to(DEVICE)
opt1 = torch.optim.AdamW(enc1.parameters(), lr=3e-4, weight_decay=1e-4)

# Build paired training tensors (only matched pairs)
bulk_paired = torch.tensor(bulk_h[matched_bulk], dtype=torch.float32)
sn_paired   = torch.tensor(sn_h[matched_sn],   dtype=torch.float32)
dl1 = DataLoader(TensorDataset(bulk_paired, sn_paired),
                 batch_size=32, shuffle=True, drop_last=False)

for ep in range(200):
    enc1.train()
    ep_loss = 0.0
    for xb, xs in dl1:
        xb, xs = xb.to(DEVICE), xs.to(DEVICE)
        opt1.zero_grad()
        # Add 30% gene dropout to bulk (simulate sc-like sparsity)
        xb_aug = xb * (torch.rand_like(xb) > 0.3).float()
        mu_b, _ = enc1(xb_aug)
        mu_s, _ = enc1(xs)
        loss = nt_xent_loss(mu_b, mu_s, temp=0.07)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc1.parameters(), 1.0)
        opt1.step()
        ep_loss += loss.item()
    if (ep + 1) % 50 == 0:
        z_b = get_z(enc1, bulk_h)
        z_s = get_z(enc1, sn_h)
        top1 = nn_accuracy(z_b, z_s, f"ep {ep+1}")["top1"]
        print(f"    loss={ep_loss/len(dl1):.4f}")

z_b1 = get_z(enc1, bulk_h)
z_s1 = get_z(enc1, sn_h)
r1 = nn_accuracy(z_b1, z_s1, "InfoNCE only (paired supervision)")
print(f"  time: {time.time()-t0:.0f}s")

# ── EXP 2: VAE + supervised InfoNCE ──────────────────────────────────────────
print("\n=== EXP 2: VAE reconstruction + supervised InfoNCE ===")
t0 = time.time()

enc2 = Encoder(N_GENES, Z_DIM).to(DEVICE)
dec2 = Decoder(Z_DIM, N_GENES).to(DEVICE)
opt2 = torch.optim.AdamW(list(enc2.parameters()) + list(dec2.parameters()),
                          lr=3e-4, weight_decay=1e-4)

BETA, FB = 1e-3, 0.2

# Training data: ALL bulk samples (reconstruction) + matched pairs (contrastive)
bulk_t = torch.tensor(bulk_h, dtype=torch.float32)
dl2_recon = DataLoader(TensorDataset(bulk_t), batch_size=32, shuffle=True)
dl2_cont  = DataLoader(TensorDataset(bulk_paired, sn_paired),
                       batch_size=32, shuffle=True, drop_last=False)

ALPHA_CONT = 1.0  # contrastive weight relative to ELBO

for ep in range(200):
    enc2.train(); dec2.train()
    cont_iter = iter(dl2_cont)
    ep_recon = ep_kl = ep_cont = 0.0

    for (xb,) in dl2_recon:
        xb = xb.to(DEVICE)
        # -- ELBO --
        xb_aug = xb * (torch.rand_like(xb) > 0.3).float()
        mu, lv = enc2(xb_aug)
        z = mu + (0.5 * lv).exp() * torch.randn_like(mu)
        xhat = dec2(z)
        recon = F.mse_loss(xhat, xb)
        kl = (-0.5 * (1 + lv - mu.pow(2) - lv.exp())).clamp(min=FB).sum(-1).mean()
        elbo = recon + BETA * kl

        # -- InfoNCE on paired batch --
        try:
            xbp, xsp = next(cont_iter)
        except StopIteration:
            cont_iter = iter(dl2_cont)
            xbp, xsp = next(cont_iter)
        xbp, xsp = xbp.to(DEVICE), xsp.to(DEVICE)
        mu_bp, _ = enc2(xbp)
        mu_sp, _ = enc2(xsp)
        cont_loss = nt_xent_loss(mu_bp, mu_sp, temp=0.07)

        loss = elbo + ALPHA_CONT * cont_loss
        opt2.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc2.parameters(), 1.0)
        opt2.step()
        ep_recon += recon.item(); ep_kl += kl.item(); ep_cont += cont_loss.item()

    if (ep + 1) % 50 == 0:
        n = len(dl2_recon)
        print(f"    ep {ep+1:3d}  recon={ep_recon/n:.4f}  kl={ep_kl/n:.3f}"
              f"  cont={ep_cont/n:.4f}")
        z_b = get_z(enc2, bulk_h)
        z_s = get_z(enc2, sn_h)
        nn_accuracy(z_b, z_s, f"  ep {ep+1}")

z_b2 = get_z(enc2, bulk_h)
z_s2 = get_z(enc2, sn_h)
r2 = nn_accuracy(z_b2, z_s2, "VAE + InfoNCE (paired)")
print(f"  time: {time.time()-t0:.0f}s")

# ── EXP 3: VAE + InfoNCE + DANN ──────────────────────────────────────────────
print("\n=== EXP 3: VAE + InfoNCE + DANN (full pipeline) ===")
t0 = time.time()

enc3 = Encoder(N_GENES, Z_DIM).to(DEVICE)
dec3 = Decoder(Z_DIM, N_GENES).to(DEVICE)
disc3 = Discriminator(Z_DIM).to(DEVICE)
opt3_vae  = torch.optim.AdamW(list(enc3.parameters()) + list(dec3.parameters()),
                               lr=3e-4, weight_decay=1e-4)
opt3_disc = torch.optim.AdamW(disc3.parameters(), lr=1e-4)

sn_all_t = torch.tensor(sn_h, dtype=torch.float32).to(DEVICE)
dl3_recon = DataLoader(TensorDataset(bulk_t), batch_size=32, shuffle=True)
dl3_cont  = DataLoader(TensorDataset(bulk_paired, sn_paired),
                       batch_size=32, shuffle=True, drop_last=False)

for ep in range(200):
    enc3.train(); dec3.train(); disc3.train()
    lam = 2.0 / (1 + math.exp(-10 * ep / 200)) - 1
    cont_iter = iter(dl3_cont)
    ep_recon = ep_kl = ep_cont = ep_dann = 0.0

    for (xb,) in dl3_recon:
        xb = xb.to(DEVICE)
        # -- ELBO --
        xb_aug = xb * (torch.rand_like(xb) > 0.3).float()
        mu, lv = enc3(xb_aug)
        z  = mu + (0.5 * lv).exp() * torch.randn_like(mu)
        xhat = dec3(z)
        recon = F.mse_loss(xhat, xb)
        kl = (-0.5 * (1 + lv - mu.pow(2) - lv.exp())).clamp(min=FB).sum(-1).mean()

        # -- DANN discriminator step --
        idx_sc = torch.randperm(len(sn_all_t))[:len(xb)]
        xsc    = sn_all_t[idx_sc]
        with torch.no_grad(): mu_sc, _ = enc3(xsc)
        opt3_disc.zero_grad()
        dl_b = F.binary_cross_entropy_with_logits(disc3(mu.detach(), lam), torch.ones(len(mu), device=DEVICE))
        dl_s = F.binary_cross_entropy_with_logits(disc3(mu_sc.detach(), lam), torch.zeros(len(mu_sc), device=DEVICE))
        ((dl_b + dl_s) / 2).backward(); opt3_disc.step()

        # -- InfoNCE on paired batch --
        try:
            xbp, xsp = next(cont_iter)
        except StopIteration:
            cont_iter = iter(dl3_cont); xbp, xsp = next(cont_iter)
        xbp, xsp = xbp.to(DEVICE), xsp.to(DEVICE)
        mu_bp, _ = enc3(xbp); mu_sp, _ = enc3(xsp)
        cont_loss = nt_xent_loss(mu_bp, mu_sp, temp=0.07)

        # -- DANN encoder adversarial --
        dann_loss = (F.binary_cross_entropy_with_logits(disc3(mu, lam), torch.zeros(len(mu), device=DEVICE))
                   + F.binary_cross_entropy_with_logits(disc3(mu_sc, lam), torch.ones(len(mu_sc), device=DEVICE))) / 2

        loss = recon + BETA * kl + ALPHA_CONT * cont_loss + 0.3 * dann_loss
        opt3_vae.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(enc3.parameters(), 1.0); opt3_vae.step()

        ep_recon += recon.item(); ep_kl += kl.item()
        ep_cont += cont_loss.item(); ep_dann += dann_loss.item()

    if (ep + 1) % 50 == 0:
        n = len(dl3_recon)
        print(f"    ep {ep+1:3d}  recon={ep_recon/n:.4f}  cont={ep_cont/n:.4f}"
              f"  dann={ep_dann/n:.4f}  lam={lam:.3f}")
        z_b = get_z(enc3, bulk_h)
        z_s = get_z(enc3, sn_h)
        nn_accuracy(z_b, z_s, f"  ep {ep+1}")

z_b3 = get_z(enc3, bulk_h)
z_s3 = get_z(enc3, sn_h)
r3 = nn_accuracy(z_b3, z_s3, "VAE + InfoNCE + DANN (full)")
print(f"  time: {time.time()-t0:.0f}s")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY — Supervised contrastive vs. unsupervised baselines")
print("="*70)
print(f"  Random baseline:               0.011  (1/92)")
for label, r in [("Raw HVG (no model)", r0),
                 ("InfoNCE only (paired sup.)", r1),
                 ("VAE + InfoNCE (paired sup.)", r2),
                 ("VAE + InfoNCE + DANN", r3)]:
    print(f"  {label:40s} top1={r['top1']:.3f}  top3={r['top3']:.3f}"
          f"  paired_cos={r['paired_cos']:.3f}")

os.makedirs("/Users/rls/ecs271/vae_health/analysis/results", exist_ok=True)
with open("/Users/rls/ecs271/vae_health/analysis/results/rq1_supervised_infonce.json", "w") as f:
    json.dump({"raw_hvg": r0, "infonce_only": r1, "vae_infonce": r2,
               "vae_infonce_dann": r3}, f, indent=2)
print("\nSaved → analysis/results/rq1_supervised_infonce.json")
