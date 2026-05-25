"""
Quick ablation: apply Hungarian assignment to v2's model (no severity weighting).
Answers: did 0.523 come from severity weighting, or just from Hungarian eval?

Also tries γ=0.0, 0.5, 1.0, 2.0 to find best severity weight.
"""
from __future__ import annotations
import json, math, os, time
import numpy as np
import torch, torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import linear_sum_assignment
import pandas as pd

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

d = np.load("/Users/rls/ecs271/data/sc/combat/combat_paired.npz", allow_pickle=True)
bulk_raw = d["bulk_x"].astype(np.float32)
sc_raw   = np.log1p(d["sc_x"].astype(np.float32))
donors   = d["donors"]
N        = len(donors)

def zscore(x):
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)

top_idx = np.argsort(bulk_raw.var(0))[-2000:]
bh = zscore(bulk_raw[:, top_idx])
sh = zscore(sc_raw[:,  top_idx])
G  = bh.shape[1]

clin = pd.read_csv("/Users/rls/ecs271/data/sc/combat/CBD-KEY-CLINVAR/COMBAT_CLINVAR_for_processed.txt", sep="\t")
sev_map = clin.drop_duplicates("COMBAT_ID").set_index("COMBAT_ID")
sev_vals = sev_map["Hospitalstay"].reindex(donors).fillna(sev_map["Hospitalstay"].median()).values.astype(np.float32)
sev_vals = (sev_vals - sev_vals.min()) / (sev_vals.max() - sev_vals.min() + 1e-8)

Z_DIM = 128

class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(G, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
        )
        self.proj = nn.Linear(256, Z_DIM)
    def forward(self, x): return self.proj(self.body(x))

def encode(enc, x_np):
    enc.eval()
    with torch.no_grad():
        return enc(torch.tensor(x_np, dtype=torch.float32).to(DEVICE)).cpu().numpy()

def nt_xent(z1, z2, t=0.05):
    N = z1.size(0)
    z = torch.cat([F.normalize(z1,dim=1), F.normalize(z2,dim=1)])
    s = (z@z.T)/t; s.fill_diagonal_(-1e9)
    lab = torch.cat([torch.arange(N,2*N), torch.arange(N)]).to(z1.device)
    return F.cross_entropy(s, lab)

def dcl_loss_weighted(z1, z2, t, sev_w_mat):
    N = z1.size(0)
    z1n = F.normalize(z1, dim=1); z2n = F.normalize(z2, dim=1)
    sim_12 = (z1n @ z2n.T) / t
    sim_11 = (z1n @ z1n.T) / t
    sim_22 = (z2n @ z2n.T) / t
    mask = torch.eye(N, dtype=torch.bool, device=z1.device)
    if sev_w_mat is not None:
        w = torch.tensor(sev_w_mat, dtype=torch.float32, device=z1.device)
        sim_12 = sim_12 + torch.where(~mask, torch.log(w.clamp(min=1e-8)), torch.zeros_like(w))
    pos = sim_12.diagonal()
    neg_fwd = torch.logsumexp(torch.cat([sim_12.masked_fill(mask,-1e9), sim_11.masked_fill(mask,-1e9)], 1), 1)
    neg_bwd = torch.logsumexp(torch.cat([sim_12.T.masked_fill(mask,-1e9), sim_22.masked_fill(mask,-1e9)], 1), 1)
    return ((-pos + neg_fwd).mean() + (-pos + neg_bwd).mean()) / 2

def train(bulk_np, sc_np, sev_np, gamma=0.0, epochs=300, lr=3e-4):
    enc = Enc().to(DEVICE)
    opt = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bt = torch.tensor(bulk_np, dtype=torch.float32)
    st = torch.tensor(sc_np,   dtype=torch.float32)
    n = len(bt)
    if gamma > 0:
        s = sev_np[:, None]; sev_w = 1 + gamma * (1 - np.abs(s - s.T))
    else:
        sev_w = None
    for ep in range(epochs):
        tau = 0.05 + (0.3 - 0.05) * (1 - ep/epochs)
        enc.train()
        zb = enc(bt.to(DEVICE) * (torch.rand(bt.shape, device=DEVICE) > 0.3).float())
        zs = enc(st.to(DEVICE))
        loss = dcl_loss_weighted(zb, zs, tau, sev_w)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(enc.parameters(), 1.0); opt.step()
        sch.step()
    return enc

def evaluate_fold(enc, test_idx):
    zb_all = encode(enc, bh)
    zs_te  = encode(enc, sh[test_idx])
    zb_ = F.normalize(torch.tensor(zb_all, dtype=torch.float32), dim=1)
    zs_ = F.normalize(torch.tensor(zs_te,  dtype=torch.float32), dim=1)
    sim = (zs_ @ zb_.T).numpy()
    # Argmax (vs all 109)
    nn_idx = sim.argmax(1)
    acc_arg = sum(nn_idx[i]==test_idx[i] for i in range(len(test_idx)))/len(test_idx)
    # Hungarian (vs 22 test only)
    r,c = linear_sum_assignment(-sim[:,test_idx])
    acc_hun = sum(test_idx[c[i]]==test_idx[i] for i in range(len(test_idx)))/len(test_idx)
    return float(acc_arg), float(acc_hun)

rng = np.random.default_rng(42)
idx = rng.permutation(N)
folds = np.array_split(idx, 5)

all_results = {}
for gamma in [0.0, 0.5, 1.0, 2.0]:
    label = f"γ={gamma}" + (" (v2-style, no sev)" if gamma==0 else "")
    print(f"\n=== {label} ===")
    arg_folds, hun_folds = [], []
    t0 = time.time()
    for fi, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(5) if j!=fi])
        enc = train(bh[train_idx], sh[train_idx], sev_vals[train_idx], gamma=gamma)
        a, h = evaluate_fold(enc, test_idx)
        arg_folds.append(a); hun_folds.append(h)
        print(f"  Fold {fi+1}: argmax={a:.3f} hungarian={h:.3f}")
    print(f"  MEAN: argmax={np.mean(arg_folds):.3f}  hungarian={np.mean(hun_folds):.3f}  ({time.time()-t0:.0f}s)")
    all_results[f"gamma_{gamma}"] = {"argmax": float(np.mean(arg_folds)), "hungarian": float(np.mean(hun_folds)),
                                      "folds_arg": arg_folds, "folds_hun": hun_folds}

print("\n=== SUMMARY ===")
print(f"{'γ':>6}  {'argmax':>8}  {'hungarian':>10}")
for k, v in all_results.items():
    g = k.split("_")[1]
    print(f"{g:>6}  {v['argmax']:>8.3f}  {v['hungarian']:>10.3f}")

os.makedirs("analysis/results", exist_ok=True)
with open("analysis/results/rq1_combat_v4b.json","w") as f:
    json.dump({"n_donors":N, "random":1/N, "results":all_results}, f, indent=2)
print("Saved → rq1_combat_v4b.json")
