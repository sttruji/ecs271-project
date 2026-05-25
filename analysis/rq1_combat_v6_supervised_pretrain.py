"""
RQ1 — v6: Supervised donor-ID pretraining on PBMC bulk, then cross-modal fine-tune.

v5 failure diagnosis: augmented-view SSL teaches "be invariant to masking noise" —
but donor variation looks like noise, so the encoder learns to IGNORE donor identity.

Fix: Stage 1 uses SUPERVISED InfoNCE where the positive pair is
(view1_donor_i, view2_donor_i) and negatives are all other donors.
This explicitly teaches "donors should be distinct" — the right objective.

Stage 1: Supervised InfoNCE on 89 GSE162632 PBMC donors (both augmented views
         of same donor = positive; different donors = negatives).
Stage 2: DCL + severity (γ=5) fine-tune on COMBAT 87 training donors.

Difference from v5: Stage 1 is IDENTICAL mechanically, but the key insight is
that supervised contrastive pretraining on PBMC already orients the embedding
space for donor discrimination — not for noise robustness.
Wait — that IS what v5 did. Let me re-examine...

Actually: v5 Stage 1 used within-batch random pairs as negatives (DataLoader batch=32).
At batch=32 from N=89 donors, many negatives are easy. The real fix is:
use ALL 89 donors every step (full-batch InfoNCE) so every step contrasts
all 89 donors against each other — much stronger donor discrimination signal.
"""
from __future__ import annotations
import json, math, os, time
import numpy as np
import torch, torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment
import pandas as pd

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Device: {DEVICE}")

# ── Data ─────────────────────────────────────────────────────────────────────
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

# PBMC pretraining data
pb = np.load("/Users/rls/ecs271/data/bulk/pbmc_bulk/GSE162632_pbmc_NI_combat_genes.npz",
             allow_pickle=True)
pbmc_x_full = pb["X_log1p"].astype(np.float32)
pbmc_genes  = pb["genes"]
combat_genes = d["genes"]
pbmc_full = np.zeros((len(pbmc_x_full), len(combat_genes)), dtype=np.float32)
for i, g in enumerate(combat_genes):
    idx_in_pbmc = np.where(pbmc_genes == g)[0]
    if len(idx_in_pbmc) > 0:
        pbmc_full[:, i] = pbmc_x_full[:, idx_in_pbmc[0]]
pbmc_h = zscore(pbmc_full[:, top_idx])
n_pbmc = len(pbmc_h)
print(f"COMBAT: {N} donors, {G} HVGs | PBMC pretrain: {n_pbmc} donors")

# Severity
clin = pd.read_csv("/Users/rls/ecs271/data/sc/combat/CBD-KEY-CLINVAR/COMBAT_CLINVAR_for_processed.txt", sep="\t")
sev_map = clin.drop_duplicates("COMBAT_ID").set_index("COMBAT_ID")
sev = sev_map["Hospitalstay"].reindex(donors).fillna(sev_map["Hospitalstay"].median()).values.astype(np.float32)
sev = (sev - sev.min()) / (sev.max() - sev.min() + 1e-8)

# ── Model ─────────────────────────────────────────────────────────────────────
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

def nt_xent_full(z1, z2, t):
    """Full-batch InfoNCE: z1[i] paired with z2[i], all others negative."""
    N = z1.size(0)
    z = torch.cat([F.normalize(z1, dim=1), F.normalize(z2, dim=1)])
    s = (z @ z.T) / t; s.fill_diagonal_(-1e9)
    lab = torch.cat([torch.arange(N, 2*N), torch.arange(N)]).to(z1.device)
    return F.cross_entropy(s, lab)

def dcl_severity(z1, z2, t, sev_w):
    Nb = z1.size(0)
    z1n = F.normalize(z1,dim=1); z2n = F.normalize(z2,dim=1)
    s12=(z1n@z2n.T)/t; s11=(z1n@z1n.T)/t; s22=(z2n@z2n.T)/t
    mask=torch.eye(Nb,dtype=torch.bool,device=z1.device)
    w=torch.tensor(sev_w,dtype=torch.float32,device=z1.device)
    s12=s12+torch.where(~mask,torch.log(w.clamp(1e-8)),torch.zeros_like(w))
    pos=s12.diagonal()
    l1=(-pos+torch.logsumexp(torch.cat([s12.masked_fill(mask,-1e9),s11.masked_fill(mask,-1e9)],1),1)).mean()
    l2=(-pos+torch.logsumexp(torch.cat([s12.T.masked_fill(mask,-1e9),s22.masked_fill(mask,-1e9)],1),1)).mean()
    return (l1+l2)/2

# ── Stage 1: Full-batch supervised donor-ID pretraining on PBMC ──────────────
def pretrain_supervised(epochs=300, lr=3e-4):
    """Full-batch: all 89 PBMC donors every step. Two augmented views per donor.
    Every donor is contrasted against ALL 88 others — strong donor discrimination."""
    enc = Enc().to(DEVICE)
    opt = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pt = torch.tensor(pbmc_h, dtype=torch.float32).to(DEVICE)
    for ep in range(epochs):
        tau = 0.05 + 0.25 * (1 - ep / epochs)
        enc.train()
        # Two independent augmented views of all 89 donors simultaneously
        v1 = pt * (torch.rand_like(pt) > 0.3).float()
        v2 = pt * (torch.rand_like(pt) > 0.3).float()
        loss = nt_xent_full(enc(v1), enc(v2), tau)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(enc.parameters(), 1.0); opt.step()
        sch.step()
        if (ep+1) % 100 == 0:
            # Quick eval: top-1 on PBMC donors (self-consistency check)
            enc.eval()
            with torch.no_grad():
                z = F.normalize(enc(pt), dim=1)
                sim = (z @ z.T); sim.fill_diagonal_(-1e9)
                top1 = (sim.argmax(1) == torch.arange(n_pbmc, device=DEVICE)).float().mean()
            print(f"  pretrain ep {ep+1}: τ={tau:.3f}  PBMC self-top1={top1:.3f}")
            enc.train()
    return enc

# ── Stage 2: Cross-modal fine-tune on COMBAT ──────────────────────────────────
def finetune(enc_init, bulk_np, sc_np, sev_np, gamma=5.0, epochs=150, lr=5e-5):
    enc = Enc().to(DEVICE)
    enc.load_state_dict(enc_init.state_dict())
    opt = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bt = torch.tensor(bulk_np, dtype=torch.float32).to(DEVICE)
    st = torch.tensor(sc_np,   dtype=torch.float32).to(DEVICE)
    s = sev_np[:, None]; sev_w = 1 + gamma * (1 - np.abs(s - s.T))
    for ep in range(epochs):
        tau = 0.05 + 0.1 * (1 - ep / epochs)
        enc.train()
        zb = enc(bt * (torch.rand_like(bt) > 0.3).float())
        zs = enc(st)
        loss = dcl_severity(zb, zs, tau, sev_w)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(enc.parameters(), 1.0); opt.step()
        sch.step()
    return enc

# ── 5-fold CV ─────────────────────────────────────────────────────────────────
rng   = np.random.default_rng(42)
idx   = rng.permutation(N)
folds = np.array_split(idx, 5)

print("\n=== Stage 1: Full-batch supervised PBMC pretraining (300 epochs) ===")
t0 = time.time()
enc_pre = pretrain_supervised(epochs=300)
print(f"  Done in {time.time()-t0:.0f}s")

print("\n=== Stage 2: 5-fold CV (γ=5, 150 epochs fine-tune) ===")
arg_f = []; hun_f = []
t0 = time.time()
for fi, test_idx in enumerate(folds):
    train_idx = np.concatenate([folds[j] for j in range(5) if j != fi])
    enc = finetune(enc_pre, bh[train_idx], sh[train_idx], sev[train_idx])
    zb = encode(enc, bh); zs = encode(enc, sh[test_idx])
    zb_ = F.normalize(torch.tensor(zb, dtype=torch.float32), dim=1)
    zs_ = F.normalize(torch.tensor(zs, dtype=torch.float32), dim=1)
    sim = (zs_ @ zb_.T).numpy()
    a = sum(sim.argmax(1)[i] == test_idx[i] for i in range(len(test_idx))) / len(test_idx)
    r,c = linear_sum_assignment(-sim[:, test_idx])
    h = sum(test_idx[c[i]] == test_idx[i] for i in range(len(test_idx))) / len(test_idx)
    arg_f.append(float(a)); hun_f.append(float(h))
    print(f"  Fold {fi+1}/5: argmax={a:.3f}  hungarian={h:.3f}"
          f"  ETA {(time.time()-t0)/(fi+1)*(5-fi-1)/60:.1f}min")

ma = float(np.mean(arg_f)); mh = float(np.mean(hun_f))
print(f"\n  MEAN: argmax={ma:.3f}  hungarian={mh:.3f}")
print(f"  v4b γ=5 (no pretrain): argmax=0.220  hungarian=0.587")
print(f"  v5 (bad pretrain):     argmax=0.194  hungarian=0.506")
print(f"  v6 (supervised pretrain): argmax={ma:.3f}  hungarian={mh:.3f}")

os.makedirs("analysis/results", exist_ok=True)
with open("analysis/results/rq1_combat_v6.json","w") as f:
    json.dump({"n_donors":N,"n_pbmc_pretrain":n_pbmc,"random":1/N,
               "v4b":{"argmax":0.220,"hungarian":0.587},
               "v5":{"argmax":0.194,"hungarian":0.506},
               "v6":{"argmax":ma,"hungarian":mh,"folds_arg":arg_f,"folds_hun":hun_f}},f,indent=2)
print("Saved → rq1_combat_v6.json")
