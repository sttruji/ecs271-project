"""
RQ1 — COMBAT 5-fold CV v5: Staged pretraining on PBMC bulk.

Stage 1: Self-supervised pretraining on GSE162632 (89 PBMC donors, unmatched).
         Within-modality InfoNCE: augmented views of same donor = positive pairs.
         Teaches the encoder PBMC gene co-expression structure without any sc data.

Stage 2: Cross-modal fine-tune on 87 COMBAT training donors (paired bulk+sc).
         DCL + severity-weighted negatives (γ=5, best from ablation).

Eval: argmax + Hungarian (γ=5 sweep showed peak at 0.587).

Rationale for PBMC bulk pretraining:
- COMBAT bulk = whole blood (WB), sc pseudobulk = PBMC
- Whole-blood encoder learns HBB/HBA1 patterns irrelevant for PBMC retrieval
- GSE162632: 89 healthy PBMC donors → encoder learns PBMC-relevant co-expression
- 99.3% gene overlap with COMBAT genes (28 padded with zeros)
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
print(f"Device: {DEVICE}")

# ── COMBAT paired data ────────────────────────────────────────────────────────
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

# ── PBMC pretraining data ─────────────────────────────────────────────────────
pb = np.load("/Users/rls/ecs271/data/bulk/pbmc_bulk/GSE162632_pbmc_NI_combat_genes.npz",
             allow_pickle=True)
pbmc_x_full = pb["X_log1p"].astype(np.float32)   # (89, 4165)
pbmc_genes  = pb["genes"]

# Align PBMC genes to same top_idx HVG subset as COMBAT
# top_idx indexes into COMBAT's 4193 genes; PBMC has 4165 (28 missing → padded with 0)
# We need to find which of our 2000 HVGs exist in the 4165 PBMC genes
combat_genes = d["genes"]               # (4193,)
pbmc_gene_set = set(pbmc_genes.tolist())
# Build a (89, 4193) array with zeros for missing genes
pbmc_full = np.zeros((len(pbmc_x_full), len(combat_genes)), dtype=np.float32)
for i, g in enumerate(combat_genes):
    idx_in_pbmc = np.where(pbmc_genes == g)[0]
    if len(idx_in_pbmc) > 0:
        pbmc_full[:, i] = pbmc_x_full[:, idx_in_pbmc[0]]
# Select HVG subset and z-score
pbmc_h = zscore(pbmc_full[:, top_idx])
n_pbmc = len(pbmc_h)
print(f"COMBAT: {N} donors, {G} HVGs")
print(f"PBMC pretraining: {n_pbmc} donors, {G} HVGs (28 zero-padded)")

# ── Severity scores for COMBAT ────────────────────────────────────────────────
clin = pd.read_csv("/Users/rls/ecs271/data/sc/combat/CBD-KEY-CLINVAR/COMBAT_CLINVAR_for_processed.txt",
                   sep="\t")
sev_map = clin.drop_duplicates("COMBAT_ID").set_index("COMBAT_ID")
sev = sev_map["Hospitalstay"].reindex(donors).fillna(sev_map["Hospitalstay"].median()).values.astype(np.float32)
sev = (sev - sev.min()) / (sev.max() - sev.min() + 1e-8)

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
    def forward(self, x): return self.proj(self.body(x))

def encode(enc, x_np):
    enc.eval()
    with torch.no_grad():
        return enc(torch.tensor(x_np, dtype=torch.float32).to(DEVICE)).cpu().numpy()

def nt_xent(z1, z2, t=0.05):
    N = z1.size(0)
    z = torch.cat([F.normalize(z1, dim=1), F.normalize(z2, dim=1)])
    s = (z @ z.T) / t; s.fill_diagonal_(-1e9)
    lab = torch.cat([torch.arange(N, 2*N), torch.arange(N)]).to(z1.device)
    return F.cross_entropy(s, lab)

def dcl_severity(z1, z2, t, sev_w_mat):
    """DCL with severity-weighted negatives."""
    Nb = z1.size(0)
    z1n = F.normalize(z1, dim=1); z2n = F.normalize(z2, dim=1)
    s12 = (z1n @ z2n.T) / t
    s11 = (z1n @ z1n.T) / t
    s22 = (z2n @ z2n.T) / t
    mask = torch.eye(Nb, dtype=torch.bool, device=z1.device)
    w = torch.tensor(sev_w_mat, dtype=torch.float32, device=z1.device)
    s12 = s12 + torch.where(~mask, torch.log(w.clamp(1e-8)), torch.zeros_like(w))
    pos = s12.diagonal()
    l1 = (-pos + torch.logsumexp(torch.cat([s12.masked_fill(mask,-1e9), s11.masked_fill(mask,-1e9)], 1), 1)).mean()
    l2 = (-pos + torch.logsumexp(torch.cat([s12.T.masked_fill(mask,-1e9), s22.masked_fill(mask,-1e9)], 1), 1)).mean()
    return (l1 + l2) / 2

# ── Stage 1: PBMC self-supervised pretraining ─────────────────────────────────
def pretrain_pbmc(epochs=200, lr=3e-4):
    """Self-supervised on PBMC bulk: augmented views of same donor = positive pair."""
    enc = Enc().to(DEVICE)
    opt = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pt = torch.tensor(pbmc_h, dtype=torch.float32)
    dl = DataLoader(TensorDataset(pt), batch_size=32, shuffle=True)
    for ep in range(epochs):
        tau = 0.05 + 0.25 * (1 - ep / epochs)
        enc.train()
        for (xb,) in dl:
            xb = xb.to(DEVICE)
            # Two augmented views of same donor
            v1 = xb * (torch.rand_like(xb) > 0.3).float()
            v2 = xb * (torch.rand_like(xb) > 0.3).float()
            z1 = enc(v1); z2 = enc(v2)
            loss = nt_xent(z1, z2, t=tau)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(enc.parameters(), 1.0); opt.step()
        sch.step()
    return enc

# ── Stage 2: Cross-modal fine-tune ───────────────────────────────────────────
def finetune(enc_init, bulk_np, sc_np, sev_np, gamma=5.0, epochs=150, lr=1e-4):
    """Fine-tune pretrained encoder on paired COMBAT data."""
    enc = Enc().to(DEVICE)
    enc.load_state_dict(enc_init.state_dict())
    opt = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bt = torch.tensor(bulk_np, dtype=torch.float32)
    st = torch.tensor(sc_np,   dtype=torch.float32)
    s = sev_np[:, None]
    sev_w = 1 + gamma * (1 - np.abs(s - s.T))
    for ep in range(epochs):
        tau = 0.05 + 0.1 * (1 - ep / epochs)
        enc.train()
        zb = enc(bt.to(DEVICE) * (torch.rand(bt.shape, device=DEVICE) > 0.3).float())
        zs = enc(st.to(DEVICE))
        loss = dcl_severity(zb, zs, tau, sev_w)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(enc.parameters(), 1.0); opt.step()
        sch.step()
    return enc

# ── 5-fold CV ─────────────────────────────────────────────────────────────────
rng   = np.random.default_rng(42)
idx   = rng.permutation(N)
folds = np.array_split(idx, 5)

print("\n=== Stage 1: PBMC pretraining (89 donors, 200 epochs) ===")
t0 = time.time()
enc_pretrained = pretrain_pbmc(epochs=200)
print(f"  Pretraining done in {time.time()-t0:.0f}s")

print("\n=== Stage 2: 5-fold CV (fine-tune + eval) ===")
arg_folds = []; hun_folds = []
t0 = time.time()

for fi, test_idx in enumerate(folds):
    train_idx = np.concatenate([folds[j] for j in range(5) if j != fi])

    enc = finetune(enc_pretrained, bh[train_idx], sh[train_idx], sev[train_idx],
                   gamma=5.0, epochs=150)

    zb_all = encode(enc, bh)
    zs_te  = encode(enc, sh[test_idx])
    zb_ = F.normalize(torch.tensor(zb_all, dtype=torch.float32), dim=1)
    zs_ = F.normalize(torch.tensor(zs_te,  dtype=torch.float32), dim=1)
    sim  = (zs_ @ zb_.T).numpy()

    nn_idx = sim.argmax(1)
    a = sum(nn_idx[i] == test_idx[i] for i in range(len(test_idx))) / len(test_idx)
    r, c = linear_sum_assignment(-sim[:, test_idx])
    h = sum(test_idx[c[i]] == test_idx[i] for i in range(len(test_idx))) / len(test_idx)

    arg_folds.append(float(a)); hun_folds.append(float(h))
    elapsed = time.time() - t0
    print(f"  Fold {fi+1}/5: {len(test_idx)} donors  argmax={a:.3f}  hungarian={h:.3f}"
          f"  ETA {elapsed/(fi+1)*(5-fi-1)/60:.1f}min")

ma = float(np.mean(arg_folds)); mh = float(np.mean(hun_folds))
print(f"\n  MEAN:  argmax={ma:.3f}  hungarian={mh:.3f}")
print(f"  v4b γ=5 (no pretrain): argmax=0.220  hungarian=0.587")
print(f"  v5 (PBMC pretrain):    argmax={ma:.3f}  hungarian={mh:.3f}")
print(f"  random: {1/N:.3f}")

os.makedirs("analysis/results", exist_ok=True)
with open("analysis/results/rq1_combat_v5.json", "w") as f:
    json.dump({"n_donors": N, "n_pbmc_pretrain": n_pbmc, "random": 1/N,
               "v4b_argmax": 0.220, "v4b_hungarian": 0.587,
               "v5_argmax": ma, "v5_hungarian": mh,
               "folds_argmax": arg_folds, "folds_hungarian": hun_folds}, f, indent=2)
print("Saved → analysis/results/rq1_combat_v5.json")
