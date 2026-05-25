"""
RQ1 — COMBAT 5-fold CV: improved version.

Changes vs v1 (baseline 0.120):
  1. Per-sample z-score normalisation (removes donor-level scale differences
     between log-CPM bulk and log-RPM sc pseudobulk)
  2. Pure InfoNCE only — no reconstruction loss (simpler, stronger contrastive)
  3. More epochs (300 vs 150)
  4. Lower temperature τ=0.05 (harder negatives)
  5. Larger latent dim (128 vs 64)
  6. Larger encoder (512→512→256→128)
  7. Try 3 temperatures and report best

Proper 5-fold CV: train from scratch on 80% donors, test on 20%.
"""
from __future__ import annotations
import json, math, os, time
import numpy as np
import torch, torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Device: {DEVICE}")

# ── Data ─────────────────────────────────────────────────────────────────────
d = np.load("/Users/rls/ecs271/data/sc/combat/combat_paired.npz", allow_pickle=True)
bulk_raw = d["bulk_x"].astype(np.float32)           # (109, 4193) log-CPM
sc_raw   = np.log1p(d["sc_x"].astype(np.float32))  # (109, 4193) log-RPM
donors   = d["donors"]
N        = len(donors)
print(f"Donors: {N}")

# HVG subset by bulk variance
top_idx = np.argsort(bulk_raw.var(0))[-2000:]
bulk_h  = bulk_raw[:, top_idx]
sc_h    = sc_raw[:, top_idx]

# Per-sample z-score (normalise each donor's vector to mean=0, std=1)
# This removes scale differences between log-CPM and log-RPM
def zscore_rows(x):
    mu  = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-8
    return (x - mu) / std

bh = zscore_rows(bulk_h)
sh = zscore_rows(sc_h)
G  = bh.shape[1]
print(f"HVGs: {G}  (per-sample z-scored)")

# ── Model ─────────────────────────────────────────────────────────────────────
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

def nt_xent(z1, z2, t=0.05):
    N  = z1.size(0)
    z  = torch.cat([F.normalize(z1, dim=1), F.normalize(z2, dim=1)])
    s  = (z @ z.T) / t
    s.fill_diagonal_(-1e9)
    lab = torch.cat([torch.arange(N, 2*N), torch.arange(N)]).to(z1.device)
    return F.cross_entropy(s, lab)

def encode(enc, x_np):
    enc.eval()
    with torch.no_grad():
        z = enc(torch.tensor(x_np, dtype=torch.float32).to(DEVICE))
    return z.cpu().numpy()

def train_infonce(bulk_np, sc_np, epochs=300, lr=3e-4, temp=0.05, mask=0.3):
    enc = Enc().to(DEVICE)
    opt = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bt  = torch.tensor(bulk_np, dtype=torch.float32)
    st  = torch.tensor(sc_np,   dtype=torch.float32)
    dl  = DataLoader(TensorDataset(bt, st), batch_size=min(32, len(bt)), shuffle=True)
    for ep in range(epochs):
        enc.train()
        for xb, xs in dl:
            xb, xs = xb.to(DEVICE), xs.to(DEVICE)
            xb_aug = xb * (torch.rand_like(xb) > mask).float()
            zb = enc(xb_aug)
            zs = enc(xs)
            loss = nt_xent(zb, zs, t=temp)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            opt.step()
        sch.step()
    return enc

# ── 5-fold CV ─────────────────────────────────────────────────────────────────
rng   = np.random.default_rng(42)
idx   = rng.permutation(N)
folds = np.array_split(idx, 5)

results_all = {}

for temp in [0.07, 0.05, 0.03]:
    print(f"\n=== 5-fold CV  τ={temp}  epochs=300 ===")
    fold_top1 = []
    t0 = time.time()

    for fi, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(5) if j != fi])
        tr_b, tr_s = bh[train_idx], sh[train_idx]
        te_s       = sh[test_idx]

        enc = train_infonce(tr_b, tr_s, epochs=300, temp=temp)

        # Evaluate: test sc vs ALL 109 bulk
        zb_all = encode(enc, bh)
        zs_te  = encode(enc, te_s)
        zb_ = F.normalize(torch.tensor(zb_all, dtype=torch.float32), dim=1)
        zs_ = F.normalize(torch.tensor(zs_te,  dtype=torch.float32), dim=1)
        nn_idx = (zs_ @ zb_.T).argmax(dim=1).numpy()
        correct = sum(nn_idx[i] == test_idx[i] for i in range(len(test_idx)))
        acc = correct / len(test_idx)
        fold_top1.append(acc)
        print(f"  Fold {fi+1}/5 (τ={temp}): {len(test_idx)} donors, top-1={acc:.3f}"
              f"  ETA {(time.time()-t0)/(fi+1)*(5-fi-1)/60:.1f}min")

    mean_top1 = float(np.mean(fold_top1))
    print(f"  τ={temp}  mean top-1: {mean_top1:.3f}  (random={1/N:.3f})")
    results_all[f"tau_{temp}"] = {"fold_top1": fold_top1, "mean": mean_top1}

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("SUMMARY")
print(f"  v1 baseline (τ=0.07, epochs=150, no z-score): 0.120")
print(f"  random: {1/N:.3f}")
for k, v in results_all.items():
    print(f"  {k}: mean top-1={v['mean']:.3f}  folds={[round(x,3) for x in v['fold_top1']]}")

os.makedirs("analysis/results", exist_ok=True)
with open("analysis/results/rq1_combat_v2.json", "w") as f:
    json.dump({"n_donors": N, "random": 1/N, "v1_baseline": 0.120,
               "results": results_all}, f, indent=2)
print("Saved → analysis/results/rq1_combat_v2.json")
