"""
RQ1 — Leave-One-Donor-Out (LODO) evaluation of VAE+InfoNCE+DANN.

Tests whether the learned encoder *generalises* to unseen donors.
For each held-out donor d:
  - Train on all (bulk, sc) paired samples except donor d
  - Evaluate: encode held-out donor's bulk and sc, check top-1 NN accuracy

This is the proper generalization test. The in-sample top-1=1.000 result
tells us the training objective works; LODO tells us if representations transfer.
"""
from __future__ import annotations
import json, time, math, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print(f"Device: {DEVICE}")

# ── Data ─────────────────────────────────────────────────────────────────────
DATA = np.load("/Users/rls/ecs271/data/sc/eraslan/eraslan_paired.npz",
               allow_pickle=True)
bulk_x   = np.nan_to_num(DATA["bulk_x"].astype(np.float32))
bulk_don = DATA["bulk_donor"]
bulk_tis = DATA["bulk_tissue"]
sn_x     = np.nan_to_num(DATA["sn_x"].astype(np.float32))
sn_don   = DATA["sn_donor"]
sn_tis   = DATA["sn_tissue"]

bulk_key = np.array([f"{d}|{t}" for d, t in zip(bulk_don, bulk_tis)])
sn_key   = np.array([f"{d}|{t}" for d, t in zip(sn_don, sn_tis)])
bulk_idx = {k: i for i, k in enumerate(bulk_key)}

matched_sn   = np.array([i for i, k in enumerate(sn_key) if k in bulk_idx])
matched_bulk = np.array([bulk_idx[sn_key[i]] for i in matched_sn])

# HVG (fit on all bulk)
gene_var = bulk_x.var(axis=0)
top_idx  = np.argsort(gene_var)[-2000:]
bulk_h   = bulk_x[:, top_idx]
sn_h     = sn_x[:, top_idx]
N_GENES  = bulk_h.shape[1]
print(f"HVGs: {N_GENES}  Matched pairs: {len(matched_sn)}")

donors = np.unique(bulk_don)
print(f"Unique donors: {len(donors)}")

# ── Model ────────────────────────────────────────────────────────────────────
Z_DIM = 64

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(N_GENES, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256),     nn.LayerNorm(256), nn.GELU(),
        )
        self.mu_h = nn.Linear(256, Z_DIM)
        self.lv_h = nn.Linear(256, Z_DIM)
    def forward(self, x):
        h = self.body(x); return self.mu_h(h), self.lv_h(h)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Z_DIM, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 512),   nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, N_GENES)
        )
    def forward(self, z): return self.net(z)

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam = lam; return x.clone()
    @staticmethod
    def backward(ctx, g): return -ctx.lam * g, None

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(Z_DIM, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, z, lam=1.0):
        return self.net(GradReverse.apply(z, lam)).squeeze(1)

def nt_xent(z1, z2, temp=0.07):
    N = z1.size(0)
    z = torch.cat([F.normalize(z1, dim=1), F.normalize(z2, dim=1)])
    sim = (z @ z.T) / temp
    sim.fill_diagonal_(-1e9)
    labels = torch.cat([torch.arange(N, 2*N), torch.arange(N)]).to(z1.device)
    return F.cross_entropy(sim, labels)

def encode_all(enc, x_np):
    enc.eval()
    with torch.no_grad():
        x = torch.tensor(x_np, dtype=torch.float32).to(DEVICE)
        mu, _ = enc(x)
    return mu.cpu().numpy()

def train_model(train_bulk_idx, train_sn_idx, epochs=150):
    """Train VAE+InfoNCE+DANN on given bulk/sn index subsets."""
    enc  = Encoder().to(DEVICE)
    dec  = Decoder().to(DEVICE)
    disc = Discriminator().to(DEVICE)
    opt_vae  = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()),
                                  lr=3e-4, weight_decay=1e-4)
    opt_disc = torch.optim.AdamW(disc.parameters(), lr=1e-4)

    # Matched pairs in train set
    pair_mask   = np.isin(matched_sn, train_sn_idx) & np.isin(matched_bulk, train_bulk_idx)
    pair_sn_i   = matched_sn[pair_mask]
    pair_bulk_i = matched_bulk[pair_mask]

    bulk_t    = torch.tensor(bulk_h[train_bulk_idx], dtype=torch.float32)
    sn_all_t  = torch.tensor(sn_h[train_sn_idx],    dtype=torch.float32).to(DEVICE)
    bulk_p_t  = torch.tensor(bulk_h[pair_bulk_i],   dtype=torch.float32)
    sn_p_t    = torch.tensor(sn_h[pair_sn_i],       dtype=torch.float32)

    if len(pair_bulk_i) < 2:
        return enc  # not enough pairs to train contrastive

    dl_recon = DataLoader(TensorDataset(bulk_t), batch_size=min(32, len(train_bulk_idx)),
                          shuffle=True)
    dl_cont  = DataLoader(TensorDataset(bulk_p_t, sn_p_t),
                          batch_size=min(32, len(pair_bulk_i)), shuffle=True)

    BETA, FB = 1e-3, 0.2
    for ep in range(epochs):
        enc.train(); dec.train(); disc.train()
        lam = 2.0 / (1 + math.exp(-10 * ep / epochs)) - 1
        cont_iter = iter(dl_cont)

        for (xb,) in dl_recon:
            xb = xb.to(DEVICE)
            xb_aug = xb * (torch.rand_like(xb) > 0.3).float()
            mu, lv = enc(xb_aug)
            z = mu + (0.5 * lv).exp() * torch.randn_like(mu)
            recon = F.mse_loss(dec(z), xb)
            kl = (-0.5*(1+lv-mu.pow(2)-lv.exp())).clamp(min=FB).sum(-1).mean()

            # DANN discriminator
            idx_sc = torch.randperm(len(sn_all_t))[:len(xb)]
            xsc = sn_all_t[idx_sc]
            with torch.no_grad(): mu_sc, _ = enc(xsc)
            opt_disc.zero_grad()
            d_loss = (F.binary_cross_entropy_with_logits(disc(mu.detach(), lam),
                                                          torch.ones(len(mu), device=DEVICE))
                    + F.binary_cross_entropy_with_logits(disc(mu_sc.detach(), lam),
                                                          torch.zeros(len(mu_sc), device=DEVICE))) / 2
            d_loss.backward(); opt_disc.step()

            # InfoNCE
            try: xbp, xsp = next(cont_iter)
            except StopIteration:
                cont_iter = iter(dl_cont); xbp, xsp = next(cont_iter)
            xbp, xsp = xbp.to(DEVICE), xsp.to(DEVICE)
            mu_bp, _ = enc(xbp); mu_sp, _ = enc(xsp)
            cont = nt_xent(mu_bp, mu_sp)

            # DANN encoder
            dann = (F.binary_cross_entropy_with_logits(disc(mu, lam),
                                                        torch.zeros(len(mu), device=DEVICE))
                  + F.binary_cross_entropy_with_logits(disc(mu_sc, lam),
                                                        torch.ones(len(mu_sc), device=DEVICE))) / 2

            loss = recon + BETA * kl + 1.0 * cont + 0.3 * dann
            opt_vae.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0); opt_vae.step()

    return enc

# ── LODO loop ────────────────────────────────────────────────────────────────
print("\n=== Leave-One-Donor-Out Evaluation ===")
print(f"  {'Donor':12s} {'held-out pairs':>14s}  top1    top3")

lodo_results = []
per_tissue_results = defaultdict(list)

for held_donor in donors:
    # Split: hold out this donor's bulk + sn
    train_bulk = np.where(bulk_don != held_donor)[0]
    train_sn   = np.where(sn_don   != held_donor)[0]
    test_bulk  = np.where(bulk_don == held_donor)[0]
    test_sn    = np.where(sn_don   == held_donor)[0]

    # Pairs in the test set
    test_sn_matched  = [i for i in test_sn if sn_key[i] in bulk_idx
                        and bulk_idx[sn_key[i]] in test_bulk]
    if not test_sn_matched:
        continue

    # Train
    enc = train_model(train_bulk, train_sn, epochs=150)

    # Encode ALL bulk (train + test) and test sn
    z_bulk_all = encode_all(enc, bulk_h)       # (92, Z_DIM)
    z_sn_test  = encode_all(enc, sn_h[test_sn_matched])

    # Evaluate: NN in ALL 92 bulk samples
    zb = F.normalize(torch.tensor(z_bulk_all, dtype=torch.float32), dim=1)
    zs = F.normalize(torch.tensor(z_sn_test,  dtype=torch.float32), dim=1)
    sim   = zs @ zb.T
    ranks = sim.argsort(dim=1, descending=True)

    top1 = top3 = 0
    for row, si in enumerate(test_sn_matched):
        true_bi = bulk_idx[sn_key[si]]
        rp = (ranks[row] == true_bi).nonzero(as_tuple=True)[0].item()
        top1 += rp == 0; top3 += rp < 3
        per_tissue_results[sn_tis[si]].append(rp == 0)

    n = len(test_sn_matched)
    print(f"  {held_donor:12s} {n:>14d}  {top1/n:.3f}  {top3/n:.3f}")
    lodo_results.append(dict(donor=held_donor, n=n, top1=top1/n, top3=top3/n))

# Overall
all_top1 = np.mean([r["top1"] for r in lodo_results])
all_top3 = np.mean([r["top3"] for r in lodo_results])
print(f"\n  {'OVERALL (macro avg)':26s}  top1={all_top1:.3f}  top3={all_top3:.3f}")
print(f"  Random baseline:                   {1/92:.3f}")

print("\nPer-tissue breakdown:")
for tis, vals in sorted(per_tissue_results.items()):
    print(f"  {tis:30s}  top1={np.mean(vals):.3f}  n={len(vals)}")

# Save
os.makedirs("/Users/rls/ecs271/vae_health/analysis/results", exist_ok=True)
out = dict(lodo_per_donor=lodo_results, overall_top1=float(all_top1),
           overall_top3=float(all_top3), random_baseline=1/92,
           per_tissue={t: float(np.mean(v)) for t, v in per_tissue_results.items()})
with open("/Users/rls/ecs271/vae_health/analysis/results/rq1_lodo_infonce.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved → analysis/results/rq1_lodo_infonce.json")
