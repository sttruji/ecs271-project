"""Q62 — Train FiLM MetaInjection VAE on PBMC pseudobulk data.

Input:  lupus_subsampled_uce_adata.h5ad
          - 261 donors (162 lupus, 99 normal)
          - 30,867 genes in raw/X
          - X_uce: (315919, 1280) per-cell UCE embeddings

Workflow:
  1. Pseudobulk per donor (sum raw counts, log-CPM normalise)
  2. Select top-3000 HVGs by inter-donor variance
  3. Fit ischemia-free FiLM VAE (z_bio_dim=16, meta=sex+batch)
     (no disease label in meta — we want disease to live in z_bio)
  4. Save:
       q62_out/pbmc_vae.pt          — VAE checkpoint
       q62_out/z_bio.npy            — (261, 16) per-donor z_bio
       q62_out/uce_mean.npy         — (261, 1280) mean-pooled UCE per donor
       q62_out/donor_meta.csv       — donor_id, disease, sex, batch
       q62_out/gene_list.json       — the 3000 HVG names
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

H5AD = Path("/tmp/lupus_demo.h5ad")
OUT  = ROOT / "analysis" / "results" / "q62_pbmc_vae"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
N_HVG = 3000
Z_DIM = 16
EPOCHS = 400
LR = 3e-4
BATCH = 32


# ── Simple FiLM VAE (no heavy dependencies on Q54b plumbing) ─────────────────

class FiLMDecoder(nn.Module):
    def __init__(self, z_dim, meta_dim, output_dim, hidden=(512, 512)):
        super().__init__()
        self.meta_emb = nn.Sequential(
            nn.Linear(meta_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
        )
        layers, in_d = [], z_dim
        self.gammas, self.betas = nn.ModuleList(), nn.ModuleList()
        for h in hidden:
            layers.append(nn.Linear(in_d, h))
            self.gammas.append(nn.Linear(128, h))
            self.betas.append(nn.Linear(128, h))
            in_d = h
        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(in_d, output_dim)
        # init FiLM to identity
        for g, b in zip(self.gammas, self.betas):
            nn.init.ones_(g.weight); nn.init.zeros_(g.bias)
            nn.init.zeros_(b.weight); nn.init.zeros_(b.bias)

    def forward(self, z, meta):
        m = self.meta_emb(meta)
        h = z
        for fc, g, b in zip(self.layers, self.gammas, self.betas):
            h = fc(h)
            h = g(m) * h + b(m)
            h = torch.nn.functional.gelu(h)
        return self.out(h)


class PBMCVAE(nn.Module):
    def __init__(self, input_dim, meta_dim, z_dim=16):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
        )
        self.mu_head  = nn.Linear(256, z_dim)
        self.lv_head  = nn.Linear(256, z_dim)
        self.decoder  = FiLMDecoder(z_dim, meta_dim, input_dim)

    def encode(self, x):
        h  = self.encoder(x)
        return self.mu_head(h), self.lv_head(h)

    def reparameterise(self, mu, lv):
        if self.training:
            return mu + torch.randn_like(mu) * (0.5 * lv).exp()
        return mu

    def forward(self, x, meta):
        mu, lv = self.encode(x)
        z = self.reparameterise(mu, lv)
        x_hat = self.decoder(z, meta)
        return x_hat, mu, lv

    def elbo(self, x, meta, beta=1e-3, free_bits=0.5):
        x_hat, mu, lv = self.forward(x, meta)
        recon = ((x_hat - x) ** 2).mean()
        kl_per_dim = -0.5 * (1 + lv - mu.pow(2) - lv.exp())
        kl = kl_per_dim.clamp(min=free_bits).mean()
        return recon + beta * kl, recon, kl


# ── Data loading ──────────────────────────────────────────────────────────────

def load_pseudobulk():
    print("Loading H5AD and pseudobulking ...")
    f = h5py.File(H5AD, "r")

    # Gene symbols from feature_name
    fn = f["var"]["feature_name"]
    cats  = [g.decode() for g in fn["categories"][:]]
    codes = np.array(fn["codes"][:])
    gene_names = [cats[c] if c >= 0 else "" for c in codes]

    # Donor and disease metadata
    def _decode(g):
        cats2 = np.array(g["categories"][:])
        codes2 = np.array(g["codes"][:])
        return np.array([cats2[c].decode() if codes2[c] >= 0 else "" for c in codes2])

    donor_ids = _decode(f["obs"]["donor_id"])
    disease   = _decode(f["obs"]["disease"])
    sex_arr   = _decode(f["obs"]["sex"])

    # Raw counts from raw/X
    raw = f["raw"]["X"]
    data2    = raw["data"][:]
    indices2 = raw["indices"][:]
    indptr2  = raw["indptr"][:]
    n_cells  = len(donor_ids)
    # raw/X has 30,867 genes
    raw_fn  = f["raw"]["var"]
    raw_fn2 = raw_fn["feature_name"]
    raw_cats  = [g.decode() for g in raw_fn2["categories"][:]]
    raw_codes = np.array(raw_fn2["codes"][:])
    raw_genes = [raw_cats[c] if c >= 0 else "" for c in raw_codes]
    n_raw_genes = len(raw_genes)

    # UCE embeddings
    X_uce = f["obsm"]["X_uce"][:]
    f.close()

    X_raw = csr_matrix((data2.astype(np.float32), indices2, indptr2),
                       shape=(n_cells, n_raw_genes))
    print(f"  Raw: {X_raw.shape}, donors: {len(set(donor_ids))}")

    # Pseudobulk per donor
    unique_donors = sorted(set(donor_ids))
    d2r = {d: i for i, d in enumerate(unique_donors)}
    n_d = len(unique_donors)
    pb  = np.zeros((n_d, n_raw_genes), dtype=np.float32)
    uce_sum  = np.zeros((n_d, 1280), dtype=np.float32)
    uce_cnt  = np.zeros(n_d, dtype=np.int32)
    meta_sex = {}
    meta_dis = {}

    for i, donor in enumerate(donor_ids):
        r = d2r[donor]
        pb[r]      += np.asarray(X_raw[i].todense()).flatten()
        uce_sum[r] += X_uce[i]
        uce_cnt[r] += 1
        meta_sex[donor] = sex_arr[i]
        meta_dis[donor] = disease[i]

    # Log-CPM per donor
    totals = pb.sum(1, keepdims=True).clip(1)
    X_lc = np.log1p(pb / totals * 1e4)

    # HVG selection by inter-donor variance
    hvg_idx = np.argsort(X_lc.var(0))[::-1][:N_HVG]
    X_hvg = X_lc[:, hvg_idx]
    hvg_names = [raw_genes[i] for i in hvg_idx]

    # UCE mean
    uce_mean = uce_sum / uce_cnt[:, None].clip(1)

    # Metadata table
    meta_df = pd.DataFrame({
        "donor_id": unique_donors,
        "disease":  [meta_dis[d] for d in unique_donors],
        "sex":      [meta_sex[d] for d in unique_donors],
    })
    meta_df["label"] = (meta_df["disease"].str.lower().str.contains("lupus")).astype(int)
    meta_df["sex_bin"] = (meta_df["sex"] == "male").astype(int)

    print(f"  Pseudobulk: {X_hvg.shape}  |  lupus: {meta_df['label'].sum()}  normal: {(meta_df['label']==0).sum()}")
    return X_hvg, uce_mean, meta_df, hvg_names


# ── Training ──────────────────────────────────────────────────────────────────

def train_vae(X_hvg, meta_df):
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    # Scale input
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_hvg).astype(np.float32)

    # Meta matrix: sex_bin only (NOT disease — that should live in z_bio)
    M = meta_df[["sex_bin"]].values.astype(np.float32)

    X_t = torch.from_numpy(X_sc)
    M_t = torch.from_numpy(M)

    model = PBMCVAE(input_dim=N_HVG, meta_dim=M.shape[1], z_dim=Z_DIM)
    opt = optim.Adam(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    n = X_sc.shape[0]
    print(f"\nTraining PBMC VAE: {n} donors × {N_HVG} HVGs → z_bio ({Z_DIM}-d)")
    print(f"  epochs={EPOCHS}  batch={BATCH}  lr={LR}  beta=1e-3  free_bits=0.5")

    best_loss, best_state = float("inf"), None
    for ep in range(1, EPOCHS + 1):
        model.train()
        idx = rng.permutation(n)
        total_loss = 0
        for i in range(0, n, BATCH):
            b = idx[i:i+BATCH]
            xb = X_t[b]; mb = M_t[b]
            loss, recon, kl = model.elbo(xb, mb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(b)
        sched.step()
        avg = total_loss / n
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 50 == 0 or ep == 1:
            print(f"  ep {ep:>4}  loss={avg:.4f}  (best={best_loss:.4f})")

    model.load_state_dict(best_state)
    return model, scaler


# ── Save artefacts ────────────────────────────────────────────────────────────

def save_artefacts(model, scaler, X_hvg, uce_mean, meta_df, hvg_names):
    model.eval()
    X_sc = scaler.transform(X_hvg).astype(np.float32)
    M    = meta_df[["sex_bin"]].values.astype(np.float32)
    with torch.no_grad():
        mu, _ = model.encode(torch.from_numpy(X_sc))
        z_bio = mu.numpy()

    np.save(OUT / "z_bio.npy",    z_bio)
    np.save(OUT / "uce_mean.npy", uce_mean)
    np.save(OUT / "X_hvg_sc.npy", X_sc)   # scaled HVG expression (for reference)
    meta_df.to_csv(OUT / "donor_meta.csv", index=False)
    (OUT / "gene_list.json").write_text(json.dumps(hvg_names))

    torch.save({
        "state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std":  scaler.scale_.tolist(),
        "z_dim": Z_DIM,
        "n_hvg": N_HVG,
        "meta_cols": ["sex_bin"],
    }, OUT / "pbmc_vae.pt")

    print(f"\nSaved to {OUT}/")
    print(f"  z_bio:    {z_bio.shape}  (16-d per-donor VAE embedding)")
    print(f"  uce_mean: {uce_mean.shape}  (1280-d mean-pooled UCE)")
    print(f"  donors:   {len(meta_df)}  ({meta_df['label'].sum()} lupus, {(meta_df['label']==0).sum()} normal)")

    # Quick sanity: how much variance does z_bio capture?
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler as SS

    z_sc = SS().fit_transform(z_bio)
    y    = meta_df["label"].values
    cv_z = cross_val_score(LogisticRegression(C=1, max_iter=500, class_weight="balanced"),
                           z_sc, y, cv=5, scoring="f1_macro").mean()
    cv_u = cross_val_score(LogisticRegression(C=0.1, max_iter=500, class_weight="balanced"),
                           SS().fit_transform(uce_mean), y, cv=5, scoring="f1_macro").mean()
    print(f"\nSanity check (5-fold CV macro-F1):")
    print(f"  z_bio alone  (16d):  {cv_z:.3f}")
    print(f"  UCE mean     (1280d): {cv_u:.3f}")


def main():
    X_hvg, uce_mean, meta_df, hvg_names = load_pseudobulk()
    model, scaler = train_vae(X_hvg, meta_df)
    save_artefacts(model, scaler, X_hvg, uce_mean, meta_df, hvg_names)


if __name__ == "__main__":
    main()
