#!/usr/bin/env python3
"""Q9 — Cross-cohort PC validation on GSE279480.

GSE279480 (Smithmyer 2025) has 1021 samples across ~70 donors, multiple
time points, and 4 stimulation conditions per donor (LPS / Null / Poly I:C
/ SEB). We use the *Null* (unstimulated baseline) samples — closest to
healthy peripheral blood without any handling stress — to test whether
the GTEx-derived PC structure replicates in an independent cohort.

What we test:
  (a) Project GSE279480 Null samples onto GTEx PCs (after gene-symbol
      alignment + scaler match). If GTEx PCs are biology, scores should
      land in a comparable range. If they're cohort-specific quirks,
      scores will be wildly off.
  (b) Re-fit PCA on the GSE279480 Null matrix. Compare GSE PC_k loadings
      to GTEx PC_k loadings via |cos similarity|. PCs that capture
      universal blood biology will match across cohorts; PCs that are
      cohort-specific will be uncorrelated.
  (c) Compare metadata correlations: GSE has Sex, age cohort, CMV status,
      donor — check whether GSE PCs that match GTEx PCs also correlate with
      the matching biology covariates.

Run:  python q9_gse279480_cohort.py
"""
from __future__ import annotations

import gzip
import sys
from collections import Counter
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
GSE_DIR = Path("/Users/rls/Desktop/programming-projects/single-cell/bulk-project/data/GSE279480")
RESULTS = ROOT / "results"
OUT = ROOT / "figures"

# ── Symbol ↔ Ensembl mapping comes from the GTEx GCT itself ───────────────
def gtex_ensembl_to_symbol() -> dict[str, str]:
    """Return {ENSG (no version) : symbol} from the GTEx v11 GCT header."""
    df = pd.read_csv("/Users/rls/ecs271/data/bulk/gtex_v11_whole_blood.gct.gz",
                     sep="\t", skiprows=2, compression="gzip", usecols=["Name", "Description"])
    mapping = {}
    for ensg, sym in zip(df["Name"].astype(str), df["Description"].astype(str)):
        ensg_no_v = ensg.split(".")[0]
        if ensg_no_v not in mapping:
            mapping[ensg_no_v] = sym
    return mapping


def load_gse279480_metadata() -> pd.DataFrame:
    matrix = GSE_DIR / "GSE279480_series_matrix.txt.gz"
    rows: dict[str, list[list[str]]] = {}
    with gzip.open(matrix, "rt") as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"): break
            if not line.startswith("!Sample_"): continue
            parts = line.rstrip("\n").split("\t")
            rows.setdefault(parts[0], []).append([p.strip('"') for p in parts[1:]])
    meta = pd.DataFrame({
        "gsm": rows["!Sample_geo_accession"][0],
        "title": rows["!Sample_title"][0],
        "lib": rows["!Sample_description"][0],
    })
    for row in rows.get("!Sample_characteristics_ch1", []):
        keys = [c.split(":", 1)[0].strip() for c in row if ":" in c]
        if not keys: continue
        key = Counter(keys).most_common(1)[0][0]
        meta[key] = [c.split(":", 1)[1].strip() if ":" in c else "" for c in row]
    return meta


def main():
    print("Loading GTEx PCA reference + scaler ...")
    _, shared_genes, scaler_mean, scaler_std = load_trained(CHECKPOINT, device="cpu")
    expr_log, gene_names = load_gtex_blood()
    expr_aligned, _, _ = align_to_shared(expr_log, gene_names, shared_genes)
    X_gtex = standardise(expr_aligned, scaler_mean, scaler_std)
    pca_gtex = PCA(n_components=50).fit(X_gtex)
    V_gtex = pca_gtex.components_  # (50, p)
    print(f"  GTEx: {X_gtex.shape}")

    print("\nLoading GSE279480 metadata + counts ...")
    meta = load_gse279480_metadata()
    null_meta = meta[meta["stimulation"] == "Null"].copy()
    print(f"  Null samples: {len(null_meta)} (out of {len(meta)} total)")
    print(f"  Donors: {null_meta['donor'].nunique()}")
    print(f"  Sex distribution: {null_meta['Sex'].value_counts().to_dict()}")
    print(f"  Age cohort: {null_meta['age cohort'].value_counts().to_dict()}")

    # Read full counts (gene rows × sample cols), filter to Null samples
    print("Reading full counts file ...")
    counts = pd.read_csv(GSE_DIR / "GSE279480_P441_genecounts.csv.gz", index_col=0)
    print(f"  counts: {counts.shape[0]:,} genes × {counts.shape[1]:,} samples")

    # Match GSM → lib via meta['lib'] (e.g. 'lib73151')
    null_libs = set(null_meta["lib"].astype(str))
    keep_cols = [c for c in counts.columns if c in null_libs]
    counts_null = counts[keep_cols].copy()
    print(f"  Null counts: {counts_null.shape}")

    # Build Ensembl → symbol map and rename rows
    print("Mapping Ensembl → symbol via GTEx GCT ...")
    e2s = gtex_ensembl_to_symbol()
    counts_null.index = counts_null.index.astype(str).str.split(".").str[0]
    keep_genes = counts_null.index.isin(e2s)
    counts_null = counts_null.loc[keep_genes].copy()
    counts_null.index = [e2s[g] for g in counts_null.index]
    # collapse duplicate symbols by sum
    counts_null = counts_null.groupby(level=0).sum()
    print(f"  symbols mapped + dedup: {counts_null.shape}")

    # log2(CPM + 1) — same normalization as GTEx loader
    expr_raw = counts_null.values.astype(np.float64)  # (genes, samples)
    lib_size = expr_raw.sum(axis=0, keepdims=True)
    cpm = expr_raw / np.maximum(lib_size, 1) * 1e6
    expr_log_gse = np.log2(cpm + 1).T.astype(np.float32)  # (samples, genes)
    gene_names_gse = np.asarray(counts_null.index)

    # Align to GTEx shared genes
    aligned_gse, _, _ = align_to_shared(expr_log_gse, gene_names_gse, shared_genes)
    print(f"  GSE Null aligned: {aligned_gse.shape}")
    X_gse = standardise(aligned_gse, scaler_mean, scaler_std)

    # ── (a) Project onto GTEx PCs ─────────────────────────────────────────
    pc_gse_in_gtex = X_gse @ V_gtex.T   # (n_samples, 50) — coords in GTEx PC basis
    pc_gtex_self = X_gtex @ V_gtex.T

    # ── (b) Re-fit PCA on GSE Null and compare loadings ───────────────────
    n_pcs_gse = min(50, X_gse.shape[0] - 1)
    pca_gse = PCA(n_components=n_pcs_gse).fit(X_gse)
    V_gse = pca_gse.components_   # (n_pcs_gse, p)

    cos_full = V_gtex[: n_pcs_gse] @ V_gse.T
    # for each GTEx PC, find best-matching GSE PC via max |cos|
    best = np.argmax(np.abs(cos_full), axis=1)
    best_cos = cos_full[np.arange(n_pcs_gse), best]
    df_match = pd.DataFrame({
        "gtex_pc": np.arange(1, n_pcs_gse + 1),
        "gtex_var_explained": pca_gtex.explained_variance_ratio_[:n_pcs_gse],
        "gse_var_explained": pca_gse.explained_variance_ratio_,
        "best_gse_pc": best + 1,
        "abs_cos": np.abs(best_cos),
        "signed_cos": best_cos,
    })
    df_match.to_csv(RESULTS / "q9_gtex_vs_gse_pc_match.csv", index=False)

    print("\nGTEx PC -> best matching GSE Null PC (top 10):")
    for _, row in df_match.head(10).iterrows():
        print(f"  GTEx PC{int(row.gtex_pc):>2} (var={row.gtex_var_explained:.3f}) -> "
              f"GSE PC{int(row.best_gse_pc):>2}  |cos|={row.abs_cos:.3f}")

    # ── (c) GSE metadata correlation with GTEx-PC scores of GSE samples ───
    gse_meta_aligned = null_meta.set_index("lib").loc[keep_cols].reset_index()
    age_map = {f"BR{i}": int(i) for i in range(1, 5)}  # BR1..BR4 = age cohort enums
    gse_meta_aligned["AGE_cat"] = gse_meta_aligned["age cohort"].map(age_map)
    sex_map = {"Male": 0, "Female": 1, "M": 0, "F": 1}
    gse_meta_aligned["SEX_bin"] = gse_meta_aligned["Sex"].map(sex_map)
    cmv_map = {"Negative": 0, "Positive": 1, "negative": 0, "positive": 1}
    gse_meta_aligned["CMV_bin"] = gse_meta_aligned["cmv status"].map(cmv_map)

    from scipy.stats import spearmanr
    cor_rows = []
    for col in ["AGE_cat", "SEX_bin", "CMV_bin"]:
        v = pd.to_numeric(gse_meta_aligned[col], errors="coerce").values
        for k in range(min(20, n_pcs_gse)):
            x = pc_gse_in_gtex[:, k]
            mask = np.isfinite(v) & np.isfinite(x)
            if mask.sum() < 30: continue
            r, _ = spearmanr(v[mask], x[mask])
            cor_rows.append({"meta": col, "gtex_pc": k + 1, "rho": float(r)})
    cor_df = pd.DataFrame(cor_rows)
    cor_df.to_csv(RESULTS / "q9_gse_metadata_in_gtex_basis.csv", index=False)

    # ── Figure ────────────────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # (i) GTEx PCs vs best GSE PC, by |cos|
    axes[0].plot(df_match["gtex_pc"], df_match["abs_cos"], color="#3fb950", lw=1.4, marker="o", ms=3)
    axes[0].axhline(0.5, color="#7d8590", ls=":", lw=0.8)
    axes[0].axhline(0.7, color="#f78166", ls=":", lw=0.8)
    axes[0].set_xlabel("GTEx PC"); axes[0].set_ylabel("|cos| with best-matching GSE PC")
    axes[0].set_title("Q9a: PC loading replication (GTEx → GSE279480 Null)")
    axes[0].set_ylim(0, 1.05)

    # (ii) PC1 vs PC2 score scatter overlaying both cohorts
    axes[1].scatter(pc_gtex_self[:, 0], pc_gtex_self[:, 1], s=8, alpha=0.4,
                    color="#58a6ff", label=f"GTEx (n={X_gtex.shape[0]})")
    axes[1].scatter(pc_gse_in_gtex[:, 0], pc_gse_in_gtex[:, 1], s=22, alpha=0.85,
                    color="#f78166", label=f"GSE279480 Null (n={X_gse.shape[0]})", zorder=5)
    axes[1].set_xlabel("PC1 score (GTEx basis)")
    axes[1].set_ylabel("PC2 score (GTEx basis)")
    axes[1].set_title("Q9b: GSE Null projected onto GTEx PCs")
    axes[1].legend()

    # (iii) GSE metadata heatmap
    if not cor_df.empty:
        pivot = cor_df.pivot(index="meta", columns="gtex_pc", values="rho")
        im = axes[2].imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        axes[2].set_yticks(range(len(pivot.index)))
        axes[2].set_yticklabels(pivot.index)
        axes[2].set_xticks(range(pivot.shape[1]))
        axes[2].set_xticklabels(pivot.columns)
        axes[2].set_title("Q9c: GSE metadata vs GTEx-basis PC scores (Spearman ρ)")
        plt.colorbar(im, ax=axes[2])
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.iloc[i, j]
                if pd.notna(v) and abs(v) > 0.15:
                    axes[2].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT / "q9_gse279480_cohort.png", dpi=150)
    plt.close(fig)
    print(f"\nWrote {OUT / 'q9_gse279480_cohort.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
