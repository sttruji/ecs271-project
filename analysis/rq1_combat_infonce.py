"""
RQ1 — COMBAT paired blood bulk+sc InfoNCE experiment.
109 donors, single tissue (blood), 4193 shared genes.

Phase 1 (fast, ~3 min): baselines + in-sample InfoNCE
Phase 2 (warm-start LODO, ~20 min): train base model once, fine-tune per fold

Run with: python rq1_combat_infonce.py [phase1|phase2|both]
Default: both
"""
from __future__ import annotations
import sys, json, math, time, os
import numpy as np
import torch, torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA

MODE = sys.argv[1] if len(sys.argv) > 1 else "both"
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Device: {DEVICE}  Mode: {MODE}")

# ── Data ─────────────────────────────────────────────────────────────────────
d = np.load("/Users/rls/ecs271/data/sc/combat/combat_paired.npz", allow_pickle=True)
bulk_x = d["bulk_x"].astype(np.float32)          # (109, 4193) log-CPM
sc_x   = np.log1p(d["sc_x"].astype(np.float32))  # (109, 4193) log-RPM
donors = d["donors"]
N, G_all = bulk_x.shape
print(f"Donors: {N}, Genes: {G_all}")

# HVG subset (2000 genes by bulk variance)
top_idx = np.argsort(bulk_x.var(0))[-2000:]
bh = bulk_x[:, top_idx]
sh = sc_x[:, top_idx]
G = bh.shape[1]

# ── Evaluation ───────────────────────────────────────────────────────────────
def top1_acc(zb: np.ndarray, zs: np.ndarray, indices=None) -> float:
    """Cosine NN top-1. If indices given, evaluate only those sc rows vs all bulk."""
    zb_ = F.normalize(torch.tensor(zb, dtype=torch.float32), dim=1)
    zs_ = F.normalize(torch.tensor(zs if indices is None else zs[indices], dtype=torch.float32), dim=1)
    sim = zs_ @ zb_.T
    ranks = sim.argsort(dim=1, descending=True)
    if indices is None:
        correct = (ranks[:, 0] == torch.arange(len(zs_))).float().mean().item()
    else:
        correct = sum((ranks[i, 0] == idx).item() for i, idx in enumerate(indices)) / len(indices)
    return correct

# ── Model ─────────────────────────────────────────────────────────────────────
Z_DIM = 64

class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(G, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
        )
        self.mu = nn.Linear(256, Z_DIM)
        self.lv = nn.Linear(256, Z_DIM)
    def forward(self, x):
        h = self.body(x); return self.mu(h), self.lv(h)

class Dec(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Z_DIM, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 512),   nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, G)
        )
    def forward(self, z): return self.net(z)

def nt_xent(z1, z2, t=0.07):
    N = z1.size(0)
    z = torch.cat([F.normalize(z1, dim=1), F.normalize(z2, dim=1)])
    s = (z @ z.T) / t; s.fill_diagonal_(-1e9)
    lab = torch.cat([torch.arange(N, 2*N), torch.arange(N)]).to(z1.device)
    return F.cross_entropy(s, lab)

def encode(enc, x_np):
    enc.eval()
    with torch.no_grad():
        mu, _ = enc(torch.tensor(x_np, dtype=torch.float32).to(DEVICE))
    return mu.cpu().numpy()

def train_vae_infonce(enc, dec, bulk_np, sc_np, epochs=150, lr=3e-4, mask=0.3):
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()),
                             lr=lr, weight_decay=1e-4)
    bt = torch.tensor(bulk_np, dtype=torch.float32)
    st = torch.tensor(sc_np,   dtype=torch.float32)
    dl = DataLoader(TensorDataset(bt, st), batch_size=min(32, len(bt)), shuffle=True)
    for ep in range(epochs):
        enc.train(); dec.train()
        for xb, xs in dl:
            xb, xs = xb.to(DEVICE), xs.to(DEVICE)
            xb_aug = xb * (torch.rand_like(xb) > mask).float()
            mu, lv = enc(xb_aug)
            z = mu + (0.5 * lv).exp() * torch.randn_like(mu)
            recon = F.mse_loss(dec(z), xb)
            kl = (-0.5*(1+lv-mu.pow(2)-lv.exp())).clamp(min=0.2).sum(-1).mean()
            mu_s, _ = enc(xs)
            cont = nt_xent(mu, mu_s)
            loss = recon + 1e-3 * kl + 1.0 * cont
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(enc.parameters(), 1.0); opt.step()
    return enc, dec

os.makedirs("analysis/results", exist_ok=True)
results = {}

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: Baselines + in-sample InfoNCE
# ═══════════════════════════════════════════════════════════════════════════
if MODE in ("phase1", "both"):
    print("\n=== PHASE 1: Baselines ===")
    r_raw  = top1_acc(bh, sh)
    pca = PCA(50); bp = pca.fit_transform(bh); sp = pca.transform(sh)
    r_pca  = top1_acc(bp, sp)
    print(f"  Random baseline:  {1/N:.3f}")
    print(f"  Raw HVG top-1:    {r_raw:.3f}")
    print(f"  PCA-50 top-1:     {r_pca:.3f}")

    print("\n=== PHASE 1: In-sample VAE + InfoNCE ===")
    t0 = time.time()
    enc_full = Enc().to(DEVICE); dec_full = Dec().to(DEVICE)
    enc_full, dec_full = train_vae_infonce(enc_full, dec_full, bh, sh, epochs=200)
    zb = encode(enc_full, bh); zs = encode(enc_full, sh)
    r_insample = top1_acc(zb, zs)
    print(f"  In-sample top-1:  {r_insample:.3f}  ({time.time()-t0:.0f}s)")

    results.update({"n_donors": N, "random": 1/N, "raw_hvg": r_raw,
                    "pca50": r_pca, "insample": r_insample})
    with open("analysis/results/rq1_combat_infonce.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved phase 1 → analysis/results/rq1_combat_infonce.json")

    # Save base model for warm-start LODO
    torch.save({"enc": enc_full.state_dict(), "dec": dec_full.state_dict()},
               "analysis/results/rq1_combat_base.pt")
    print("Saved base model → analysis/results/rq1_combat_base.pt")

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: Warm-start LODO
# Train once on all data, then for each held-out donor: fine-tune 30 epochs
# ═══════════════════════════════════════════════════════════════════════════
if MODE in ("phase2", "both"):
    print("\n=== PHASE 2: Warm-start LODO ===")
    print(f"  Strategy: load base model, fine-tune {30} epochs per held-out donor")

    # Load or train base model
    base_path = "analysis/results/rq1_combat_base.pt"
    if os.path.exists(base_path):
        ckpt = torch.load(base_path, map_location=DEVICE)
        enc_base = Enc().to(DEVICE); enc_base.load_state_dict(ckpt["enc"])
        dec_base = Dec().to(DEVICE); dec_base.load_state_dict(ckpt["dec"])
        print("  Loaded base model from checkpoint")
    else:
        print("  Training base model from scratch...")
        enc_base = Enc().to(DEVICE); dec_base = Dec().to(DEVICE)
        enc_base, dec_base = train_vae_infonce(enc_base, dec_base, bh, sh, epochs=200)

    lodo_scores = []
    t0 = time.time()
    for ho in range(N):
        # Train indices (all except held-out)
        tr_idx = [i for i in range(N) if i != ho]
        tr_b = bh[tr_idx]; tr_s = sh[tr_idx]

        # Warm-start from base model
        enc_ft = Enc().to(DEVICE); dec_ft = Dec().to(DEVICE)
        enc_ft.load_state_dict(enc_base.state_dict())
        dec_ft.load_state_dict(dec_base.state_dict())

        # Fine-tune 30 epochs on leave-one-out data
        enc_ft, dec_ft = train_vae_infonce(enc_ft, dec_ft, tr_b, tr_s,
                                            epochs=30, lr=5e-5, mask=0.3)

        # Evaluate: encode ALL bulk, encode held-out sc, check if bulk[ho] is NN
        zb_all = encode(enc_ft, bh)
        zs_ho  = encode(enc_ft, sh[[ho]])   # shape (1, Z_DIM)
        zb_ = F.normalize(torch.tensor(zb_all, dtype=torch.float32), dim=1)
        zs_ = F.normalize(torch.tensor(zs_ho,  dtype=torch.float32), dim=1)
        nn_idx = (zs_ @ zb_.T).argmax(dim=1).item()
        r = float(nn_idx == ho)
        lodo_scores.append(r)

        elapsed = time.time() - t0
        eta = elapsed / (ho + 1) * (N - ho - 1)
        if (ho + 1) % 10 == 0 or ho == 0:
            print(f"  donor {ho+1:3d}/{N}  running top-1={sum(lodo_scores)/len(lodo_scores):.3f}"
                  f"  ETA {eta/60:.1f}min")

    lodo_top1 = sum(lodo_scores) / N
    print(f"\n  LODO top-1: {lodo_top1:.3f}  (random={1/N:.3f})")

    # Load and update results
    if os.path.exists("analysis/results/rq1_combat_infonce.json"):
        with open("analysis/results/rq1_combat_infonce.json") as f:
            results = json.load(f)
    results["lodo_top1"] = lodo_top1
    results["lodo_per_donor"] = lodo_scores
    with open("analysis/results/rq1_combat_infonce.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved final → analysis/results/rq1_combat_infonce.json")

print("\nDone.")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: Proper 5-fold CV (no data leakage)
# Train from scratch on 80% donors, test on held-out 20%
# ═══════════════════════════════════════════════════════════════════════════
if MODE in ("phase3", "both"):
    import numpy as np
    print("\n=== PHASE 3: 5-fold CV (proper train/test split) ===")
    rng = np.random.default_rng(42)
    idx = rng.permutation(N)
    folds = np.array_split(idx, 5)

    fold_top1 = []
    for fi, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(5) if j != fi])
        tr_b, tr_s = bh[train_idx], sh[train_idx]
        te_b, te_s = bh[test_idx],  sh[test_idx]

        # Train fresh model from scratch on train donors only
        enc_cv = Enc().to(DEVICE); dec_cv = Dec().to(DEVICE)
        enc_cv, dec_cv = train_vae_infonce(enc_cv, dec_cv, tr_b, tr_s,
                                            epochs=150, lr=3e-4, mask=0.3)

        # Evaluate: for each test sc, find NN in ALL 109 bulk (train+test)
        zb_all = encode(enc_cv, bh)   # all 109 bulk
        zs_te  = encode(enc_cv, te_s) # test sc only
        zb_ = F.normalize(torch.tensor(zb_all, dtype=torch.float32), dim=1)
        zs_ = F.normalize(torch.tensor(zs_te,  dtype=torch.float32), dim=1)
        sim = zs_ @ zb_.T
        nn_idx = sim.argmax(dim=1).numpy()
        correct = sum(nn_idx[i] == test_idx[i] for i in range(len(test_idx)))
        fold_acc = correct / len(test_idx)
        fold_top1.append(fold_acc)
        print(f"  Fold {fi+1}/5: {len(test_idx)} test donors, top-1={fold_acc:.3f}")

    cv_top1 = float(np.mean(fold_top1))
    print(f"\n  5-fold CV top-1: {cv_top1:.3f}  (random={1/N:.3f})")

    if os.path.exists("analysis/results/rq1_combat_infonce.json"):
        with open("analysis/results/rq1_combat_infonce.json") as f:
            results = json.load(f)
    results["cv5fold_top1"] = cv_top1
    results["cv5fold_per_fold"] = fold_top1
    results["note_lodo_leaky"] = "warm-start LODO is leaky (base model saw all donors); 5-fold CV is the honest number"
    with open("analysis/results/rq1_combat_infonce.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved → analysis/results/rq1_combat_infonce.json")
