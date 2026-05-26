"""Q60 — Interpretable basis decomposition of z_bio.

Two sub-questions:

A. Vector arithmetic: do cell-type directions in z_bio satisfy linear algebra?
   e.g. z(B-heavy) - z(B-light) + z(T-light) ≈ z(T-heavy)?
   If yes, the embedding is truly linear/composable — same geometry as word2vec.

B. Interpretable basis: what is the lowest-dimensional named set of axes that
   can bidirectionally reconstruct z_bio?
   Approaches compared:
     1. Linear cell-type basis  — 24-dim sub-cell-type scores (from Q59)
     2. NMF                     — k non-negative components of z_bio
     3. ICA                     — k independent components of z_bio
     4. Sparse autoencoder      — k-sparse overcomplete code of z_bio

   For each, we measure:
     - Forward R²: how well can the basis reconstruct z_bio? (scores → z)
     - Backward R²: how well can z_bio reconstruct the basis scores? (z → scores)
     - Reconstruction is done via 5-fold CV Ridge, not just in-sample.

   Goal: find the smallest K such that forward R² ≥ 0.90.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF, FastICA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.data import load_gtex_blood, load_metadata
from models.meta_injection_vae import FiLMMetaInjectionVAE, FiLMMetaInjectionConfig
from analysis.q57_cell_composition import build_meta_matrix, score_cell_types, build_base_meta
from analysis.q59_subcell_types import score_subcell_types, SUBCELL_MARKERS

CKPT = "/Users/rls/Desktop/programming-projects/single-cell/bulk-project/analysis/14_cross_modality_vae/cross_modality_vae.pt"
Q54B = ROOT / "analysis" / "results" / "q54b_count_weighted" / "film_resid_vae_weighted.pt"
OUT  = ROOT / "analysis" / "results" / "q60_interpretable_basis"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 0


# ──────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────

def load_z_bio(X_sc, M4):
    ckpt = torch.load(Q54B, map_location="cpu")
    cfg  = ckpt["cfg"]
    mc   = FiLMMetaInjectionConfig(
        input_dim=cfg["input_dim"], meta_dim=cfg["meta_dim"],
        z_bio_dim=cfg["z_bio_dim"], decoder_hidden=tuple(cfg["decoder_hidden"]),
        meta_embed_dim=cfg["meta_embed_dim"], beta=cfg["beta"],
        free_bits=cfg["free_bits"], lambda_tc=cfg["lambda_tc"],
    )
    m = FiLMMetaInjectionVAE(mc).eval()
    m.load_state_dict(ckpt["state_dict"])
    beta_isch = np.array(ckpt["residualizer"]["beta_isch"], dtype=np.float32)
    s_mean    = float(ckpt["residualizer"]["s_mean"])
    X_resid   = X_sc - np.outer(M4[:, 0] - s_mean, beta_isch)
    with torch.no_grad():
        return m.encode(torch.from_numpy(X_resid.astype(np.float32)))[0].numpy()


def cv_r2_multioutput(X, Y, alpha=1.0, n_splits=5):
    """5-fold CV R² predicting Y from X (joint, all dims together)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    preds = np.zeros_like(Y, dtype=np.float64)
    X_sc = StandardScaler().fit_transform(X)
    Y_sc = StandardScaler().fit_transform(Y)
    for tr, te in kf.split(X_sc):
        m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]).fit(X_sc[tr], Y_sc[tr])
        preds[te] = m.predict(X_sc[te])
    ss_res = ((Y_sc - preds) ** 2).sum()
    ss_tot = ((Y_sc - Y_sc.mean(0)) ** 2).sum()
    return float(1 - ss_res / ss_tot)


# ──────────────────────────────────────────────────────────────────────────
# Part A: Vector arithmetic
# ──────────────────────────────────────────────────────────────────────────

def part_a_vector_arithmetic(z_bio, broad_scores, broad_names, sub_scores, sub_names):
    """
    Test: z(A-heavy) - z(A-light) + z(B-light) ≈ z(B-heavy)
    For pairs: (B_cell, T_cell), (Mono_classical, NK_dim), (erythroid, neutrophil)
    """
    print("\n" + "═"*60)
    print("PART A  Vector arithmetic in z_bio")
    print("═"*60)

    # Build a combined score matrix for grouping
    all_scores = np.column_stack([broad_scores, sub_scores])
    all_names  = broad_names + sub_names

    def group_means(score_name, pct_lo=25, pct_hi=75):
        idx = all_names.index(score_name)
        vals = all_scores[:, idx]
        lo_mask = vals <= np.percentile(vals, pct_lo)
        hi_mask = vals >= np.percentile(vals, pct_hi)
        return z_bio[lo_mask].mean(0), z_bio[hi_mask].mean(0)

    pairs = [
        ("B_cell",         "T_cell"),
        ("Mono_classical", "NK_dim"),
        ("erythroid",      "neutrophil"),
        ("CD4_memory",     "CD8_cytotoxic"),
    ]

    results = []
    print(f"\n{'Pair (A→B)':<34}  {'cos(pred,true)':>14}  {'L2 err':>8}  {'L2 rand':>8}")
    print("─"*68)
    for name_a, name_b in pairs:
        try:
            z_a_lo, z_a_hi = group_means(name_a)
            z_b_lo, z_b_hi = group_means(name_b)
        except ValueError:
            continue

        direction_a = z_a_hi - z_a_lo
        z_b_hi_pred = z_b_lo + direction_a
        true = z_b_hi

        cos = float(
            np.dot(z_b_hi_pred, true) /
            (np.linalg.norm(z_b_hi_pred) * np.linalg.norm(true) + 1e-10)
        )
        l2 = float(np.linalg.norm(z_b_hi_pred - true))

        # Random baseline: random direction of same magnitude
        rng = np.random.default_rng(SEED)
        rand_dir = rng.standard_normal((100, z_bio.shape[1]))
        rand_dir *= np.linalg.norm(direction_a) / (np.linalg.norm(rand_dir, axis=1, keepdims=True) + 1e-10)
        rand_l2 = float(np.linalg.norm(rand_dir + z_b_lo - true, axis=1).mean())

        print(f"  {name_a:<14} → {name_b:<14}  cos={cos:+.3f}          L2={l2:.3f}    rand={rand_l2:.3f}")
        results.append({"pair": f"{name_a}→{name_b}", "cos": cos, "l2": l2, "rand_l2": rand_l2})

    pd.DataFrame(results).to_csv(OUT / "vector_arithmetic.csv", index=False)

    # Direction consistency: does the same concept (e.g. T-cell-heavy) transfer?
    print("\n  Direction stability: does 'high B-cell direction' align with itself?")
    # Compute the direction twice from different splits
    idx_b = all_names.index("B_cell")
    vals  = all_scores[:, idx_b]
    q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
    lo_mask = vals <= q1
    hi_mask = vals >= q3
    rng = np.random.default_rng(SEED)
    lo_idx = np.where(lo_mask)[0]
    hi_idx = np.where(hi_mask)[0]
    split = len(lo_idx) // 2
    d1 = z_bio[hi_idx[:split]].mean(0) - z_bio[lo_idx[:split]].mean(0)
    d2 = z_bio[hi_idx[split:]].mean(0) - z_bio[lo_idx[split:]].mean(0)
    cos_self = float(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-10))
    print(f"    B-cell direction split-half cosine = {cos_self:.3f}")

    return results


# ──────────────────────────────────────────────────────────────────────────
# Part B1: Linear cell-type basis
# ──────────────────────────────────────────────────────────────────────────

def part_b1_linear_basis(z_bio, broad_scores, broad_names, sub_scores, sub_names):
    print("\n" + "═"*60)
    print("PART B1  Linear cell-type basis (Q59 scores)")
    print("═"*60)

    results = {}
    bases = [
        ("broad_7",  broad_scores,                                               broad_names),
        ("sub_24",   sub_scores,                                                  sub_names),
        ("all_31",   np.column_stack([broad_scores, sub_scores]),                 broad_names + sub_names),
    ]

    for label, S, names in bases:
        fwd = cv_r2_multioutput(S, z_bio)
        bwd = cv_r2_multioutput(z_bio, S)
        print(f"  {label:<10}  n={S.shape[1]:>3}  z←scores R²={fwd:.3f}  scores←z R²={bwd:.3f}")
        results[label] = {"n": S.shape[1], "fwd_r2": fwd, "bwd_r2": bwd}

    return results


# ──────────────────────────────────────────────────────────────────────────
# Part B2: NMF — non-negative matrix factorization
# ──────────────────────────────────────────────────────────────────────────

def part_b2_nmf(z_bio):
    print("\n" + "═"*60)
    print("PART B2  NMF components of z_bio")
    print("═"*60)

    # Shift z_bio to be non-negative (NMF requirement)
    z_shift = z_bio - z_bio.min(0)

    ks = [4, 8, 12, 16, 24, 32]
    results = []
    for k in ks:
        nmf = NMF(n_components=k, random_state=SEED, max_iter=500)
        H = nmf.fit_transform(z_shift)         # samples × k
        W = nmf.components_                     # k × 16 (latent dims)
        z_hat = H @ W                           # reconstruction in shifted space
        z_hat_orig = z_hat + z_bio.min(0)       # shift back

        # In-sample reconstruction R²
        ss_res = ((z_bio - z_hat_orig) ** 2).sum()
        ss_tot = ((z_bio - z_bio.mean(0)) ** 2).sum()
        recon_r2 = float(1 - ss_res / ss_tot)

        # 5-fold CV: can we predict z from NMF coords?
        fwd = cv_r2_multioutput(H, z_bio)
        bwd = cv_r2_multioutput(z_bio, H)
        print(f"  k={k:>3}  recon R²={recon_r2:.3f}  fwd R²={fwd:.3f}  bwd R²={bwd:.3f}")
        results.append({"k": k, "recon_r2": recon_r2, "fwd_r2": fwd, "bwd_r2": bwd})

    return results


# ──────────────────────────────────────────────────────────────────────────
# Part B3: ICA — independent components
# ──────────────────────────────────────────────────────────────────────────

def part_b3_ica(z_bio):
    print("\n" + "═"*60)
    print("PART B3  ICA components of z_bio")
    print("═"*60)

    ks = [4, 8, 12, 16]
    results = []
    for k in ks:
        try:
            ica = FastICA(n_components=k, random_state=SEED, max_iter=1000, tol=1e-3)
            H = ica.fit_transform(z_bio)
            fwd = cv_r2_multioutput(H, z_bio)
            bwd = cv_r2_multioutput(z_bio, H)
            print(f"  k={k:>3}  fwd R²={fwd:.3f}  bwd R²={bwd:.3f}")
            results.append({"k": k, "fwd_r2": fwd, "bwd_r2": bwd})
        except Exception as e:
            print(f"  k={k:>3}  ICA failed: {e}")
    return results


# ──────────────────────────────────────────────────────────────────────────
# Part B4: Sparse autoencoder
# ──────────────────────────────────────────────────────────────────────────

class SparseAutoencoder(nn.Module):
    def __init__(self, z_dim: int, n_features: int, k_sparse: int):
        super().__init__()
        self.k = k_sparse
        self.encoder = nn.Linear(z_dim, n_features, bias=True)
        self.decoder = nn.Linear(n_features, z_dim, bias=True)
        nn.init.orthogonal_(self.encoder.weight)
        self.decoder.weight = nn.Parameter(self.encoder.weight.T.clone())

    def forward(self, z):
        pre = self.encoder(z)                        # (N, n_features)
        pre_relu = torch.relu(pre)
        # top-k sparsity
        topk_vals, topk_idx = pre_relu.topk(self.k, dim=1)
        sparse = torch.zeros_like(pre_relu)
        sparse.scatter_(1, topk_idx, topk_vals)
        z_hat = self.decoder(sparse)
        return z_hat, sparse

    def get_features(self, z):
        with torch.no_grad():
            _, sparse = self.forward(z)
        return sparse.numpy()


def train_sae(z_bio, n_features, k_sparse, n_epochs=500, lr=3e-3, l1=1e-3):
    z_t = torch.from_numpy(z_bio.astype(np.float32))
    model = SparseAutoencoder(z_bio.shape[1], n_features, k_sparse)
    opt = optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    for ep in range(n_epochs):
        model.train()
        opt.zero_grad()
        z_hat, sparse = model(z_t)
        recon = ((z_hat - z_t) ** 2).mean()
        l1_loss = sparse.abs().mean()
        loss = recon + l1 * l1_loss
        loss.backward()
        opt.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
    model.eval()
    return model


def part_b4_sae(z_bio):
    print("\n" + "═"*60)
    print("PART B4  Sparse autoencoder on z_bio")
    print("═"*60)

    configs = [
        (16,  8),   # same dim, k=8 active
        (32,  8),
        (32, 12),
        (64, 12),
        (64, 16),
        (128, 16),
    ]
    results = []
    for n_feat, k in configs:
        model = train_sae(z_bio, n_features=n_feat, k_sparse=k)
        H = model.get_features(torch.from_numpy(z_bio.astype(np.float32)))
        # recon
        z_t = torch.from_numpy(z_bio.astype(np.float32))
        with torch.no_grad():
            z_hat, _ = model(z_t)
        z_hat_np = z_hat.numpy()
        ss_res = ((z_bio - z_hat_np) ** 2).sum()
        ss_tot = ((z_bio - z_bio.mean(0)) ** 2).sum()
        recon_r2 = float(1 - ss_res / ss_tot)

        fwd = cv_r2_multioutput(H, z_bio)
        bwd = cv_r2_multioutput(z_bio, H)
        # sparsity
        sparsity = float((H == 0).mean())
        print(f"  features={n_feat:>4}  k={k:>3}  "
              f"recon R²={recon_r2:.3f}  fwd R²={fwd:.3f}  bwd R²={bwd:.3f}  sparsity={sparsity:.2f}")
        results.append({"n_features": n_feat, "k_sparse": k,
                        "recon_r2": recon_r2, "fwd_r2": fwd, "bwd_r2": bwd, "sparsity": sparsity})
    return results


# ──────────────────────────────────────────────────────────────────────────
# Part B5: Pathway score basis (MSigDB Hallmark gene sets)
# Using gene sets computed from available gene names only
# ──────────────────────────────────────────────────────────────────────────

HALLMARK_SUBSETS: dict[str, list[str]] = {
    "OXIDATIVE_PHOSPHORYLATION": [
        "NDUFS1", "NDUFV1", "SDHA", "SDHB", "UQCRC1", "COX5A", "ATP5F1A",
        "CYCS", "COX4I1", "ATP5MC1", "UQCRQ", "NDUFAB1", "NDUFA4",
    ],
    "INFLAMMATORY_RESPONSE": [
        "IL1B", "IL6", "TNF", "CXCL8", "CCL2", "IL1A", "PTGS2", "NFKB1",
        "ICAM1", "VCAM1", "SELE", "IL18", "CCL3", "CCL4",
    ],
    "INTERFERON_ALPHA": [
        "ISG15", "ISG20", "MX1", "OAS1", "OAS2", "IFIT1", "IFIT2", "IFIT3",
        "IFITM1", "IFI6", "IFI27", "RSAD2", "HERC5", "TRIM22",
    ],
    "INTERFERON_GAMMA": [
        "STAT1", "IRF1", "CXCL10", "CXCL9", "PSMB9", "TAP1", "B2M", "HLA-DRA",
        "CD74", "CIITA", "GBP1", "GBP2", "IDO1", "TNFSF10",
    ],
    "APOPTOSIS": [
        "CASP3", "CASP7", "CASP8", "CASP9", "BCL2", "BCL2L1", "MCL1",
        "BAX", "PUMA", "BID", "APAF1", "CYCS", "TP53",
    ],
    "CELL_CYCLE": [
        "CDK1", "CDK2", "CCNA2", "CCNB1", "CCNB2", "MKI67", "PCNA",
        "TOP2A", "BIRC5", "AURKB", "BUB1", "PLK1", "RRM2",
    ],
    "HYPOXIA": [
        "VEGFA", "HIF1A", "CA9", "LDHA", "PGAM1", "PGK1", "ALDOA",
        "ENO1", "SLC2A1", "ADM", "BNIP3", "NDRG1", "EGLN3",
    ],
    "COMPLEMENT": [
        "C1QA", "C1QB", "C1QC", "C1R", "C1S", "C3", "C4A", "C4B",
        "CFB", "CFD", "CFH", "CFI", "SERPING1", "MBL2",
    ],
    "MTORC1_SIGNALING": [
        "RPS6KB1", "EIF4E", "EIF4EBP1", "MYC", "ULK1", "PRKAA1",
        "RPTOR", "RPS6", "HSPA5", "DDIT4", "SLC7A5", "LAMTOR1",
    ],
    "PROTEIN_SECRETION": [
        "KDELR1", "SEC23A", "SEC24A", "SAR1A", "COPB1", "COPA",
        "SEC61A1", "SPCS1", "SRPR", "CANX", "CALR", "HSPA5",
    ],
    "TNFA_SIGNALING": [
        "NFKBIA", "TNFAIP3", "TNFAIP6", "CXCL2", "CXCL3", "CSF1",
        "PLAUR", "BIRC2", "BIRC3", "ZC3H12A", "SOCS3", "JUNB",
    ],
    "PI3K_AKT_MTOR": [
        "AKT1", "AKT2", "PIK3CA", "PIK3CB", "PTEN", "TSC1", "TSC2",
        "MTOR", "RHEB", "PDK1", "FOXO1", "FOXO3", "IRS1",
    ],
}


def score_pathways(X_lc, gene_names):
    upper_to_idx = {g.upper(): i for i, g in enumerate(gene_names)}
    names = list(HALLMARK_SUBSETS.keys())
    scores = np.zeros((X_lc.shape[0], len(names)), dtype=np.float32)
    for j, t in enumerate(names):
        idx = [upper_to_idx[m.upper()] for m in HALLMARK_SUBSETS[t]
               if m.upper() in upper_to_idx]
        if idx:
            scores[:, j] = X_lc[:, idx].mean(axis=1)
    z = (scores - scores.mean(0)) / (scores.std(0) + 1e-8)
    return z.astype(np.float32), names


def part_b5_pathway_basis(z_bio, X_lc, gene_names):
    print("\n" + "═"*60)
    print("PART B5  Hallmark pathway score basis")
    print("═"*60)
    scores, names = score_pathways(X_lc, gene_names)
    print(f"  {len(names)} pathway scores computed")

    # Combined with cell types
    from analysis.q59_subcell_types import score_subcell_types
    sub_sc, sub_names, _ = score_subcell_types(X_lc, gene_names)
    combined = np.column_stack([scores, sub_sc])
    combined_names = names + sub_names

    fwd_path = cv_r2_multioutput(scores, z_bio)
    bwd_path = cv_r2_multioutput(z_bio, scores)
    fwd_comb = cv_r2_multioutput(combined, z_bio)
    bwd_comb = cv_r2_multioutput(z_bio, combined)

    print(f"  pathway_12:     n=12   z←scores R²={fwd_path:.3f}  scores←z R²={bwd_path:.3f}")
    print(f"  path+subcell:   n={combined.shape[1]}   z←scores R²={fwd_comb:.3f}  scores←z R²={bwd_comb:.3f}")

    # Per-pathway prediction of z_bio
    rows = []
    for i, pname in enumerate(names):
        r2 = float(cross_val_score(Ridge(1.0),
                                   StandardScaler().fit_transform(z_bio),
                                   scores[:, i],
                                   cv=KFold(5, shuffle=True, random_state=SEED),
                                   scoring="r2").mean())
        rows.append({"pathway": pname, "z_bio_r2": r2})
    pway_df = pd.DataFrame(rows).sort_values("z_bio_r2", ascending=False)
    print(f"\n  Per-pathway R² (z_bio → pathway score):")
    for _, row in pway_df.iterrows():
        print(f"    {row['pathway']:<35}  R²={row['z_bio_r2']:.3f}")

    return {
        "pathway_12":    {"fwd_r2": fwd_path, "bwd_r2": bwd_path},
        "path+subcell":  {"fwd_r2": fwd_comb, "bwd_r2": bwd_comb},
        "per_pathway":   pway_df.to_dict(orient="records"),
    }


# ──────────────────────────────────────────────────────────────────────────
# Summary plot
# ──────────────────────────────────────────────────────────────────────────

def make_summary_plot(b1, b2, b3, b4, b5):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: forward R² vs number of interpretable features
    ax = axes[0]
    # linear basis points
    for k, v in b1.items():
        ax.scatter(v["n"], v["fwd_r2"], marker="o", s=80, color="steelblue",
                   label="Linear cell-type" if k == list(b1.keys())[0] else "")
    # NMF
    nmf_k = [r["k"] for r in b2]
    nmf_fwd = [r["fwd_r2"] for r in b2]
    ax.plot(nmf_k, nmf_fwd, marker="s", color="orange", label="NMF")
    # ICA
    if b3:
        ica_k = [r["k"] for r in b3]
        ica_fwd = [r["fwd_r2"] for r in b3]
        ax.plot(ica_k, ica_fwd, marker="^", color="green", label="ICA")
    # SAE
    sae_k = [r["n_features"] for r in b4]
    sae_fwd = [r["fwd_r2"] for r in b4]
    ax.scatter(sae_k, sae_fwd, marker="D", color="crimson", label="SAE")
    ax.axhline(0.90, color="gray", linestyle="--", label="90% target")
    ax.set_xlabel("Number of interpretable features")
    ax.set_ylabel("Forward R² (features → z_bio, 5-fold CV)")
    ax.set_title("Interpretability vs reconstruction quality")
    ax.legend()

    # Right: bidirectional R² comparison across best configs
    ax = axes[1]
    labels, fwds, bwds = [], [], []
    for k, v in b1.items():
        labels.append(f"Linear\n({k})")
        fwds.append(v["fwd_r2"])
        bwds.append(v["bwd_r2"])
    best_nmf = max(b2, key=lambda r: r["fwd_r2"])
    labels.append(f"NMF\n(k={best_nmf['k']})")
    fwds.append(best_nmf["fwd_r2"])
    bwds.append(best_nmf["bwd_r2"])
    if b3:
        best_ica = max(b3, key=lambda r: r["fwd_r2"])
        labels.append(f"ICA\n(k={best_ica['k']})")
        fwds.append(best_ica["fwd_r2"])
        bwds.append(best_ica["bwd_r2"])
    best_sae = max(b4, key=lambda r: r["fwd_r2"])
    labels.append(f"SAE\n({best_sae['n_features']}f,k={best_sae['k_sparse']})")
    fwds.append(best_sae["fwd_r2"])
    bwds.append(best_sae["bwd_r2"])

    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax.bar(x - w/2, fwds, w, label="Forward (features→z)", color="steelblue", alpha=0.8)
    bars2 = ax.bar(x + w/2, bwds, w, label="Backward (z→features)", color="orange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("5-fold CV R²")
    ax.set_title("Bidirectional R² across methods")
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(OUT / "interpretable_basis_summary.png", dpi=150)
    plt.close()
    print(f"\nPlot saved: {OUT}/interpretable_basis_summary.png")


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    print("Loading data ...")
    gtex = load_gtex_blood(checkpoint_path=CKPT)
    meta = load_metadata(gtex.sample_ids)
    M4   = build_meta_matrix(meta)
    X_sc = gtex.expr_scaled
    X_lc = gtex.expr_aligned
    gene_names = list(gtex.shared_genes)

    print("Encoding z_bio ...")
    z_bio = load_z_bio(X_sc, M4)
    print(f"  z_bio shape: {z_bio.shape}")

    print("Computing cell-type scores ...")
    from analysis.q57_cell_composition import score_cell_types
    broad_scores, broad_names, _ = score_cell_types(X_lc, gene_names)
    sub_scores, sub_names, _     = score_subcell_types(X_lc, gene_names)

    a = part_a_vector_arithmetic(z_bio, broad_scores, broad_names, sub_scores, sub_names)
    b1 = part_b1_linear_basis(z_bio, broad_scores, broad_names, sub_scores, sub_names)
    b2 = part_b2_nmf(z_bio)
    b3 = part_b3_ica(z_bio)
    b4 = part_b4_sae(z_bio)
    b5 = part_b5_pathway_basis(z_bio, X_lc, gene_names)

    make_summary_plot(b1, b2, b3, b4, b5)

    def _clean(o):
        if isinstance(o, dict):   return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):   return [_clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)): return float(o)
        return o

    out = {
        "vector_arithmetic": _clean(a),
        "linear_basis": _clean(b1),
        "nmf": _clean(b2),
        "ica": _clean(b3),
        "sae": _clean(b4),
        "pathways": _clean(b5),
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nAll results → {OUT}/")


if __name__ == "__main__":
    main()
