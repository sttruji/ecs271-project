"""Q58 — Benchmark z_bio as a deconvolution feature space.

Question: can the 16-dim VAE embedding match or beat traditional
deconvolution feature spaces (PCA-50, marker scoring, full expression)
at recovering cell type composition?

Setup:
  - Ground truth = 7 cell-type marker scores from Q57 (best simple proxy
    we have for cell composition in GTEx since we lack FACS-sorted truth)
  - Feature spaces compared:
      A. z_bio          (K=16 from Q54b VAE)
      B. PCA-50         (50d, fit on training fold)
      C. PCA-16         (matched dimensionality to z_bio)
      D. top-200 HVG    (200 most variable genes, raw)
      E. all genes      (11374d, regularized Ridge handles dim)
      F. marker scores  (7d, the very features we're predicting — upper bound)
  - Probe: cross-validated Ridge for each cell type independently
  - Metric: 5-fold CV R² per cell type, mean R² across cell types

Why this is meaningful: any feature space can predict cell composition if
big enough.  The interesting question is which compresses it best.
If 16 z_bio dims match 50-dim PCA or 200-gene HVG, the VAE is a winning
deconvolution feature (and would generalize to other downstream tasks).

Honest caveat: we lack FACS-sorted ground truth.  The marker scores ARE
themselves derived from expression, so all expression-based methods have an
unfair shared advantage over methods that ignore expression.  But all
methods we compare ARE expression-based, so it's a fair internal benchmark.
For a true 'beats CIBERSORT' claim we'd need pseudobulks from labelled
scRNA-seq — see analysis/docs/use_cases.md for that plan.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.data import load_gtex_blood, load_metadata, load_shared_genes
from models.meta_injection_vae import FiLMMetaInjectionVAE, FiLMMetaInjectionConfig
from analysis.q57_cell_composition import (
    score_cell_types, CELL_TYPE_MARKERS, build_meta_matrix
)

CKPT   = "/Users/rls/Desktop/programming-projects/single-cell/bulk-project/analysis/14_cross_modality_vae/cross_modality_vae.pt"
Q54B   = ROOT / "analysis" / "results" / "q54b_count_weighted" / "film_resid_vae_weighted.pt"
OUT    = ROOT / "analysis" / "results" / "q58_deconvolution_benchmark"
OUT.mkdir(parents=True, exist_ok=True)
SEED   = 0


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


def benchmark_features(features: dict[str, np.ndarray],
                       targets: np.ndarray, target_names: list[str]) -> pd.DataFrame:
    """5-fold CV R² for each (feature_set, cell_type) pair."""
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for fname, X in features.items():
        X_sc = StandardScaler().fit_transform(X)
        for j, tname in enumerate(target_names):
            y = targets[:, j]
            sc = cross_val_score(Ridge(alpha=1.0), X_sc, y,
                                 cv=kf, scoring="r2")
            rows.append({
                "features":  fname,
                "n_features": X.shape[1],
                "cell_type": tname,
                "r2_mean":   float(sc.mean()),
                "r2_std":    float(sc.std()),
            })
    return pd.DataFrame(rows)


def main():
    print("Loading data ...")
    gtex = load_gtex_blood(checkpoint_path=CKPT)
    meta = load_metadata(gtex.sample_ids)
    M4   = build_meta_matrix(meta)
    X_sc = gtex.expr_scaled
    X_lc = gtex.expr_aligned
    gene_names = list(gtex.shared_genes)

    # Ground truth: cell type marker scores
    print("Computing cell-type marker scores (ground truth proxy) ...")
    cell_scores, cell_names, _ = score_cell_types(X_lc, gene_names)
    print(f"  {len(cell_names)} cell types, scores shape {cell_scores.shape}")

    # Feature spaces
    print("\nBuilding feature spaces ...")
    z_bio = load_z_bio(X_sc, M4)
    print(f"  A. z_bio       (K={z_bio.shape[1]})")

    pca50 = PCA(n_components=50,  random_state=SEED).fit_transform(X_sc)
    pca16 = PCA(n_components=16,  random_state=SEED).fit_transform(X_sc)
    print(f"  B. PCA-50      (K=50)")
    print(f"  C. PCA-16      (K=16, matched to z_bio)")

    # top-200 HVG
    gene_var = X_sc.var(0)
    hvg_idx = np.argsort(gene_var)[::-1][:200]
    hvg = X_sc[:, hvg_idx]
    print(f"  D. top-200 HVG (K=200)")

    # All genes (we'll use RidgeCV with strong regularization)
    print(f"  E. all genes   (K={X_sc.shape[1]})")

    # Marker scores themselves — upper bound (the target derived from these)
    print(f"  F. marker scores (K={cell_scores.shape[1]}, upper bound)")

    features = {
        "A. z_bio (K=16, our VAE)":    z_bio,
        "B. PCA-50":                   pca50,
        "C. PCA-16 (matched)":         pca16,
        "D. top-200 HVG":              hvg,
        "E. all genes (K=11374)":      X_sc,
        "F. marker scores (upper bound)": cell_scores,
    }

    # Benchmark
    print("\n5-fold CV R² for each (feature space, cell type) ...")
    results = benchmark_features(features, cell_scores, cell_names)

    # Reshape for nice display: cell type × feature space
    pivot_r2 = results.pivot(index="cell_type", columns="features",
                              values="r2_mean").round(3)

    # Add summary rows
    summary = pd.DataFrame({
        "mean":   pivot_r2.mean(),
        "median": pivot_r2.median(),
        "min":    pivot_r2.min(),
    }).T

    print("\n" + "═" * 78)
    print("PER CELL TYPE R²")
    print("═" * 78)
    print(pivot_r2.to_string())

    print("\n" + "═" * 78)
    print("SUMMARY (across 7 cell types)")
    print("═" * 78)
    print(summary.to_string())

    # Save
    results.to_csv(OUT / "results_per_feature_celltype.csv", index=False)
    pivot_r2.to_csv(OUT / "results_pivot.csv")
    summary.to_csv(OUT / "results_summary.csv")

    # JSON for downstream use
    json_out = {
        "per_cell_type_r2":  results.to_dict(orient="records"),
        "pivot_r2":          pivot_r2.to_dict(),
        "summary":           summary.to_dict(),
        "ranking_by_mean_r2": list(summary.loc["mean"].sort_values(ascending=False).index),
    }
    (OUT / "results.json").write_text(json.dumps(json_out, indent=2))

    print(f"\nSaved → {OUT}/")

    # Verdict
    z_mean = float(summary.loc["mean", "A. z_bio (K=16, our VAE)"])
    pca50_mean = float(summary.loc["mean", "B. PCA-50"])
    pca16_mean = float(summary.loc["mean", "C. PCA-16 (matched)"])
    hvg_mean = float(summary.loc["mean", "D. top-200 HVG"])
    all_mean = float(summary.loc["mean", "E. all genes (K=11374)"])

    print("\n" + "═" * 78)
    print("VERDICT")
    print("═" * 78)
    print(f"z_bio (K=16):    mean R² = {z_mean:.3f}")
    print(f"PCA-50:          mean R² = {pca50_mean:.3f}")
    print(f"PCA-16 matched:  mean R² = {pca16_mean:.3f}")
    print(f"top-200 HVG:     mean R² = {hvg_mean:.3f}")
    print(f"all genes:       mean R² = {all_mean:.3f}")
    print()

    if z_mean > pca16_mean:
        print(f"✓ z_bio BEATS PCA-16 by {z_mean - pca16_mean:.3f} at same dimensionality")
    else:
        print(f"✗ z_bio loses to PCA-16 by {pca16_mean - z_mean:.3f}")
    if z_mean > pca50_mean:
        print(f"✓ z_bio BEATS PCA-50 (3x lower dim!) by {z_mean - pca50_mean:.3f}")
    else:
        print(f"  z_bio is {pca50_mean - z_mean:.3f} below PCA-50 (3x more dims)")
    print(f"  Gap to full expression: {all_mean - z_mean:.3f} (cost of compression)")


if __name__ == "__main__":
    main()
