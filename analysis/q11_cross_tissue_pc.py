#!/usr/bin/env python3
"""Q11 — Cross-tissue PC validation: project blood PCs onto spleen, liver, lung, muscle.

Tests whether each blood PC encodes universal biology or blood-specific quirk:

  - PC1  (MYC ribosome / Reactome Innate Immunity)  → cell composition in any tissue
  - PC2  (PU.1 / UPR-stress)                          → blood-specific autopsy axis
  - PC3  (chromatin)                                  → likely universal
  - PC4  (mitochondrial respiration / TCF7 T-cell)    → ~muscle should be extreme PC4 high
  - PC5  (hypoxia / heme)                             → liver/lung/spleen vs blood
  - PC6  (endocytosis / monocyte vs lymphocyte)        → THE KEY TEST.
          If PC6 = monocyte fraction, spleen donors should score HIGH
          (myeloid-rich), muscle donors should score LOW or noise.

Pipeline:
  load each tissue → CPM/log2/filter (same as blood)
                  → align to blood shared_genes
                  → standardise with blood's scaler
                  → project onto blood PCA basis (V_blood^T x)
                  → compare per-tissue distributions of PC1..PC10
  Also re-fits PCA per tissue and reports loading |cos| vs blood PCs.

Run:  python q11_cross_tissue_pc.py
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from lib_data import align_to_shared, load_gtex_blood, standardise  # noqa: E402
from lib_model import load_trained  # noqa: E402

CHECKPOINT = "/Users/rls/ecs271/data/models/cross_modality_vae.pt"
DATA_DIR = Path("/Users/rls/ecs271/data/bulk")
RESULTS = ROOT / "results"
OUT = ROOT / "figures"

CPM_THRESHOLD = 1.0
MIN_SAMPLE_FRAC = 0.10

TISSUE_FILES = {
    "blood":   "gtex_v11_whole_blood.gct.gz",   # reference
    "spleen":  "gtex_v10_spleen.gct.gz",
    "liver":   "gtex_v10_liver.gct.gz",
    "lung":    "gtex_v10_lung.gct.gz",
    "muscle":  "gtex_v10_muscle_skeletal.gct.gz",
}
TISSUE_COLOR = {
    "blood":  "#f78166",
    "spleen": "#3fb950",
    "liver":  "#a371f7",
    "lung":   "#58a6ff",
    "muscle": "#f0883e",
}


def load_tissue_gct(path: Path):
    """Return (samples × genes_filt) log2(CPM+1), gene_symbols, sample_ids."""
    df = pd.read_csv(path, sep="\t", skiprows=2, compression="gzip")
    expr_raw = df.iloc[:, 2:].values.astype(np.float64)  # (genes, samples)
    gene_names = df["Description"].values.astype(str)
    n_samples = expr_raw.shape[1]
    lib = expr_raw.sum(axis=0)
    cpm = expr_raw / lib * 1e6
    min_s = max(1, int(MIN_SAMPLE_FRAC * n_samples))
    keep = (cpm > CPM_THRESHOLD).sum(axis=1) >= min_s
    expr_filt = expr_raw[keep]
    names_filt = gene_names[keep]
    lib2 = expr_filt.sum(axis=0, keepdims=True)
    cpm2 = expr_filt / lib2 * 1e6
    expr_log = np.log2(cpm2 + 1).T.astype(np.float32)
    sample_ids = np.asarray(df.columns[2:])
    return expr_log, names_filt, sample_ids


def main():
    print("Loading blood reference (PCA fit, scaler) ...")
    _, shared_genes, scaler_mean, scaler_std = load_trained(CHECKPOINT, device="cpu")
    expr_log_blood, gene_names_blood = load_gtex_blood()
    expr_aligned_blood, _, _ = align_to_shared(expr_log_blood, gene_names_blood, shared_genes)
    X_blood = standardise(expr_aligned_blood, scaler_mean, scaler_std)
    pca_blood = PCA(n_components=50).fit(X_blood)
    V_blood = pca_blood.components_

    # Project blood onto its own PCs (for reference)
    pc_scores = {"blood": pca_blood.transform(X_blood)}
    n_samples = {"blood": X_blood.shape[0]}
    self_match: list[dict] = []   # |cos| of each tissue's PC vs blood PC

    print("\nProcessing other tissues:")
    for tissue, fname in TISSUE_FILES.items():
        if tissue == "blood":
            continue
        path = DATA_DIR / fname
        print(f"\n  [{tissue}] loading {path.name} ...")
        expr_log_t, gene_names_t, _ = load_tissue_gct(path)
        print(f"    raw post-filter: {expr_log_t.shape[0]} samples × {expr_log_t.shape[1]:,} genes")
        aligned_t, _, _ = align_to_shared(expr_log_t, gene_names_t, shared_genes)
        # Standardise using BLOOD's scaler so we're in the same numerical space
        X_t = standardise(aligned_t, scaler_mean, scaler_std)
        n_samples[tissue] = X_t.shape[0]

        # (a) project onto blood PCs
        pc_scores[tissue] = X_t @ V_blood.T

        # (b) re-fit PCA on this tissue and compare PC1..PC20 loadings
        n_pcs_t = min(20, X_t.shape[0] - 1)
        pca_t = PCA(n_components=n_pcs_t).fit(X_t)
        V_t = pca_t.components_
        cos_full = V_blood[: n_pcs_t] @ V_t.T
        # for each blood PC, max |cos| with any of this tissue's PCs (Hungarian-lite)
        for k in range(n_pcs_t):
            absrow = np.abs(cos_full[k])
            j = int(np.argmax(absrow))
            self_match.append({
                "tissue": tissue,
                "blood_pc": k + 1,
                "best_tissue_pc": j + 1,
                "abs_cos": float(absrow[j]),
                "blood_pc_var": float(pca_blood.explained_variance_ratio_[k]),
                "tissue_pc_var": float(pca_t.explained_variance_ratio_[j]),
            })

    match_df = pd.DataFrame(self_match)
    match_df.to_csv(RESULTS / "q11_cross_tissue_pc_match.csv", index=False)
    print(f"\nWrote {RESULTS / 'q11_cross_tissue_pc_match.csv'}")

    # Print top-10 PC replication summary
    print("\nBlood PC -> best-matching cross-tissue PC, |cos|:")
    print(f"{'PC':<5}", end="")
    for tissue in ["spleen", "liver", "lung", "muscle"]:
        print(f"{tissue:>10}", end="")
    print()
    for k in range(10):
        print(f"PC{k+1:<3}", end=" ")
        for tissue in ["spleen", "liver", "lung", "muscle"]:
            sub = match_df[(match_df["tissue"] == tissue) & (match_df["blood_pc"] == k + 1)]
            if not sub.empty:
                print(f"{sub.iloc[0]['abs_cos']:>10.3f}", end="")
            else:
                print(f"{'-':>10}", end="")
        print()

    # ── Per-tissue PC score distributions ─────────────────────────────────
    rows = []
    for tissue, scores in pc_scores.items():
        for k in range(min(10, scores.shape[1])):
            for s in scores[:, k]:
                rows.append({"tissue": tissue, "pc": k + 1, "score": float(s)})
    score_df = pd.DataFrame(rows)
    score_df.to_csv(RESULTS / "q11_pc_scores_per_tissue.csv", index=False)

    # Compute per-tissue per-PC mean ± std
    summary = score_df.groupby(["tissue", "pc"])["score"].agg(["mean", "std", "count"]).reset_index()
    summary.to_csv(RESULTS / "q11_pc_score_summary_per_tissue.csv", index=False)
    print("\nMean PC1, PC2, PC4, PC6 score per tissue (projected onto blood PCs):")
    for k in (1, 2, 4, 6):
        sub = summary[summary["pc"] == k]
        line = f"  PC{k}: "
        for _, r in sub.iterrows():
            line += f"{r['tissue']}={r['mean']:+6.2f}±{r['std']:5.2f} (n={int(r['count'])})  "
        print(line)

    # ── Figure: per-tissue distributions for selected PCs ─────────────────
    plt.style.use("dark_background")
    pcs_to_plot = [1, 2, 3, 4, 5, 6]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    for ax, k in zip(axes, pcs_to_plot):
        for tissue, scores in pc_scores.items():
            ax.hist(scores[:, k - 1], bins=40, density=True, alpha=0.55,
                    color=TISSUE_COLOR[tissue], label=f"{tissue} (n={n_samples[tissue]})")
        ax.set_title(f"PC{k} score distribution by tissue (blood PCA basis)")
        ax.set_xlabel(f"PC{k} score (donor projected onto blood eigenvector)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "q11_pc_scores_by_tissue.png", dpi=150)
    plt.close(fig)
    print(f"\nWrote {OUT / 'q11_pc_scores_by_tissue.png'}")

    # ── Figure: |cos| heatmap over blood PCs × tissues ────────────────────
    pivot = match_df.pivot(index="blood_pc", columns="tissue", values="abs_cos")
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    pivot_show = pivot.iloc[:20]
    im = ax.imshow(pivot_show.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(pivot_show.index)))
    ax.set_yticklabels([f"PC{i}" for i in pivot_show.index])
    ax.set_xticks(range(len(pivot_show.columns)))
    ax.set_xticklabels(pivot_show.columns)
    ax.set_title("Q11: Blood PC vs cross-tissue PC loading similarity (|cos|)")
    for i in range(pivot_show.shape[0]):
        for j in range(pivot_show.shape[1]):
            v = pivot_show.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.5 else "black")
    plt.colorbar(im, ax=ax, label="|cos|")
    fig.tight_layout()
    fig.savefig(OUT / "q11_cross_tissue_pc_match.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT / 'q11_cross_tissue_pc_match.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
