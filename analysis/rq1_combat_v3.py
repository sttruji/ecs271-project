"""
RQ1 — COMBAT 5-fold CV v3: DANN + InfoNCE, all genes, stronger augmentation.

Building on v2 (0.211 at τ=0.05):
  1. DANN + InfoNCE combined: force modality-invariance AND match donors
  2. All 4193 shared genes (not just top 2000 HVGs)
  3. Stronger augmentation: 40% masking + Gaussian noise
  4. Fixed τ=0.05 (best from v2)
  5. 300 epochs, Z_DIM=128, same large encoder
"""
from __future__ import annotations
import json, math, os, time
import numpy as np
import torch, torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Device: {DEVICE}")

d = np.load("/Users/rls/ecs271/data/sc/combat/combat_paired.npz", allow_pickle=True)
bulk_raw = d["bulk_x"].astype(np.float32)
sc_raw   = np.log1p(d["sc_x"].astype(np.float32))
donors   = d["donors"]
N        = len(donors)

def zscore_rows(x):
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)

# Use all genes (not just HVGs)
bh = zscore_rows(bulk_raw)
sh = zscore_rows(sc_raw)
G  = bh.shape[1]
print(f"Donors: {N}  Genes: {G}  (all, z-scored)")

Z_DIM = 128

class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(G, 512),  nn.LayerNorm(512),  nn.GELU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
        )
        self.proj = nn.Linear(256, Z_DIM)
    def forward(self, x):
        return self.proj(self.body(x))

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam = lam; return x.clone()
    @staticmethod
    def backward(ctx, g): return -ctx.lam * g, None

class Disc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(Z_DIM, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, z, lam=1.0):
        return self.net(GRL.apply(z, lam)).squeeze(1)

def nt_xent(z1, z2, t=0.05):
    N = z1.size(0)
    z = torch.cat([F.normalize(z1,dim=1), F.normalize(z2,dim=1)])
    s = (z @ z.T)/t; s.fill_diagonal_(-1e9)
    lab = torch.cat([torch.arange(N,2*N), torch.arange(N)]).to(z1.device)
    return F.cross_entropy(s, lab)

def augment(x, mask=0.4, noise=0.05):
    x = x * (torch.rand_like(x) > mask).float()
    x = x + noise * torch.randn_like(x)
    return x

def encode(enc, x_np):
    enc.eval()
    with torch.no_grad():
        z = enc(torch.tensor(x_np, dtype=torch.float32).to(DEVICE))
    return z.cpu().numpy()

def train(bulk_np, sc_np, epochs=300, lr=3e-4, temp=0.05):
    enc  = Enc().to(DEVICE)
    disc = Disc().to(DEVICE)
    opt_enc  = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=1e-4)
    opt_disc = torch.optim.AdamW(disc.parameters(), lr=lr*0.5, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt_enc, T_max=epochs)

    bt = torch.tensor(bulk_np, dtype=torch.float32)
    st = torch.tensor(sc_np,   dtype=torch.float32)
    dl = DataLoader(TensorDataset(bt, st), batch_size=min(32,len(bt)), shuffle=True)
    sc_all = st.to(DEVICE)  # for DANN sc batches

    for ep in range(epochs):
        enc.train(); disc.train()
        lam = 2/(1+math.exp(-10*ep/epochs)) - 1
        for xb, xs in dl:
            xb, xs = xb.to(DEVICE), xs.to(DEVICE)
            xb_aug = augment(xb)
            zb = enc(xb_aug)
            zs = enc(xs)

            # InfoNCE
            cont = nt_xent(zb, zs, t=temp)

            # DANN discriminator step
            idx_sc = torch.randperm(len(sc_all))[:len(xb)]
            zsc = enc(sc_all[idx_sc]).detach()
            opt_disc.zero_grad()
            dl_loss = (F.binary_cross_entropy_with_logits(disc(zb.detach(), lam), torch.ones(len(zb), device=DEVICE))
                     + F.binary_cross_entropy_with_logits(disc(zsc, lam),          torch.zeros(len(zsc), device=DEVICE))) / 2
            dl_loss.backward(); opt_disc.step()

            # DANN encoder adversarial
            dann = (F.binary_cross_entropy_with_logits(disc(zb, lam), torch.zeros(len(zb), device=DEVICE))
                  + F.binary_cross_entropy_with_logits(disc(enc(sc_all[idx_sc]), lam), torch.ones(len(zsc), device=DEVICE))) / 2

            loss = cont + 0.3 * dann
            opt_enc.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(enc.parameters(), 1.0); opt_enc.step()
        sch.step()
    return enc

rng   = np.random.default_rng(42)
idx   = rng.permutation(N)
folds = np.array_split(idx, 5)

print("\n=== 5-fold CV: DANN + InfoNCE, all genes, τ=0.05 ===")
fold_top1 = []; t0 = time.time()

for fi, test_idx in enumerate(folds):
    train_idx = np.concatenate([folds[j] for j in range(5) if j != fi])
    enc = train(bh[train_idx], sh[train_idx])

    zb_all = encode(enc, bh)
    zs_te  = encode(enc, sh[test_idx])
    zb_ = F.normalize(torch.tensor(zb_all, dtype=torch.float32), dim=1)
    zs_ = F.normalize(torch.tensor(zs_te,  dtype=torch.float32), dim=1)
    nn_idx = (zs_ @ zb_.T).argmax(dim=1).numpy()
    acc = sum(nn_idx[i] == test_idx[i] for i in range(len(test_idx))) / len(test_idx)
    fold_top1.append(float(acc))
    elapsed = time.time()-t0
    print(f"  Fold {fi+1}/5: {len(test_idx)} donors, top-1={acc:.3f}"
          f"  ETA {elapsed/(fi+1)*(5-fi-1)/60:.1f}min")

mean_top1 = float(np.mean(fold_top1))
print(f"\n  DANN+InfoNCE mean top-1: {mean_top1:.3f}  (random={1/N:.3f})")
print(f"  v1=0.120  v2_best=0.211  v3={mean_top1:.3f}")

os.makedirs("analysis/results", exist_ok=True)
with open("analysis/results/rq1_combat_v3.json", "w") as f:
    json.dump({"n_donors":N,"random":1/N,"v1":0.120,"v2_best":0.211,
               "v3_dann_infonce":mean_top1,"folds":fold_top1}, f, indent=2)
print("Saved → analysis/results/rq1_combat_v3.json")
