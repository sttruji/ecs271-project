#!/usr/bin/env python3
"""Q4 — How well do PCs encode biological information?

Two halves, both on the same standardised log2(CPM+1) GTEx whole-blood matrix:

(a) **Pathway / GO / KEGG enrichment of PC loadings.**
    For each of PC1..PC5, take the top 200 positive-loading and top 200
    negative-loading genes (by |loading|) and submit each list to the
    Enrichr REST API against:
      - GO_Biological_Process_2023
      - KEGG_2021_Human
    Saves the top-15 terms per (PC, direction) and a markdown summary.

(b) **Metadata correlation.**
    Joins each donor to the GTEx v10 annotations:
      - subject phenotypes:  AGE (decade band → midpoint), SEX, DTHHRDY
      - sample attributes:   SMRIN, SMTSISCH (post-mortem ischemia time, min),
                             SMCENTER (sequencing center), SMNABTCH (NA batch),
                             SMGEBTCH (gene-expr batch), SMRDLGTH
    For each PC, computes Spearman ρ with continuous metadata and
    one-way-ANOVA F-statistic / explained-variance for categorical
    (SEX, SMCENTER, batch). Saves a heatmap of |ρ| and the full table.

Run:  python q4_pcs_vs_biology.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, spearmanr
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from lib_data import align_to_shared, load_gtex_blood, standardise  # noqa: E402
from lib_model import load_trained  # noqa: E402

CHECKPOINT = "/Users/rls/ecs271/data/models/cross_modality_vae.pt"
ANNOT = Path("/Users/rls/ecs271/data/annotations")
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True, parents=True)
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True, parents=True)

ENRICHR = "https://maayanlab.cloud/Enrichr"
LIBRARIES = ["GO_Biological_Process_2023", "KEGG_2021_Human"]
N_PCS_FOR_ENRICHMENT = 5
TOP_GENES_PER_DIRECTION = 200


# ── GTEx sample / subject ID parsing ──────────────────────────────────────
def sample_to_subject(sample_id: str) -> str:
    parts = sample_id.split("-")
    return "-".join(parts[:2])  # GTEX-1117F-0005-... → GTEX-1117F


def age_to_midpoint(age_band: str) -> float:
    if not isinstance(age_band, str) or "-" not in age_band:
        return np.nan
    a, b = age_band.split("-")
    try:
        return (int(a) + int(b)) / 2
    except ValueError:
        return np.nan


# ── Enrichr ───────────────────────────────────────────────────────────────
def enrichr_submit(genes: list[str], description: str) -> dict:
    import requests

    payload = {
        "list": (None, "\n".join(genes)),
        "description": (None, description),
    }
    r = requests.post(f"{ENRICHR}/addList", files=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def enrichr_query(user_list_id: str, library: str, top_k: int = 15) -> pd.DataFrame:
    import requests

    r = requests.get(
        f"{ENRICHR}/enrich",
        params={"userListId": user_list_id, "backgroundType": library},
        timeout=60,
    )
    r.raise_for_status()
    rows = r.json().get(library, [])
    cols = ["rank", "term", "p", "z", "combined_score", "overlap_genes", "adj_p", "old_p", "old_adj_p"]
    out = pd.DataFrame(rows, columns=cols)
    out["overlap_genes"] = out["overlap_genes"].apply(lambda g: ";".join(g) if isinstance(g, list) else g)
    return out.sort_values("p").head(top_k)


# ── main ──────────────────────────────────────────────────────────────────
def main() -> int:
    print("Loading shared-genes / scaler from trained-VAE checkpoint ...")
    _, shared_genes, scaler_mean, scaler_std = load_trained(CHECKPOINT, device="cpu")

    print("Loading GTEx whole blood ...")
    expr_log, gene_names = load_gtex_blood()
    expr_aligned, _, _ = align_to_shared(expr_log, gene_names, shared_genes)
    expr_scaled = standardise(expr_aligned, scaler_mean, scaler_std)
    print(f"  shape: {expr_scaled.shape}")

    # Sample IDs from GCT column headers
    import gzip

    with gzip.open("/Users/rls/ecs271/data/bulk/gtex_v11_whole_blood.gct.gz", "rt") as fh:
        for _ in range(2):
            fh.readline()
        header = fh.readline().rstrip("\n").split("\t")
    sample_ids = header[2:]
    assert len(sample_ids) == expr_scaled.shape[0], (len(sample_ids), expr_scaled.shape[0])
    print(f"  sample IDs: first 3: {sample_ids[:3]}")

    # ── PCA on the standardised matrix ────────────────────────────────────
    print("\nFitting PCA-50 on standardised log2(CPM+1) ...")
    pca = PCA(n_components=50).fit(expr_scaled)
    pc_scores = pca.transform(expr_scaled)  # (803, 50)
    var_ratio = pca.explained_variance_ratio_
    print("  PC1..5 var ratio:", [f"{v:.3f}" for v in var_ratio[:5]])
    print("  cumulative:", [f"{v:.3f}" for v in np.cumsum(var_ratio)[:5]])

    # ── Metadata join ─────────────────────────────────────────────────────
    print("\nLoading GTEx annotations ...")
    sub = pd.read_csv(ANNOT / "GTEx_v10_Annotations_SubjectPhenotypesDS.txt", sep="\t")
    samp = pd.read_csv(
        ANNOT / "GTEx_v10_Annotations_SampleAttributesDS.txt",
        sep="\t",
        low_memory=False,
    )

    df = pd.DataFrame({"SAMPID": sample_ids})
    df["SUBJID"] = df["SAMPID"].apply(sample_to_subject)
    df = df.merge(sub, on="SUBJID", how="left")
    df = df.merge(
        samp[["SAMPID", "SMRIN", "SMTSISCH", "SMCENTER", "SMNABTCH", "SMGEBTCH", "SMRDLGTH"]],
        on="SAMPID",
        how="left",
    )
    df["AGE_mid"] = df["AGE"].apply(age_to_midpoint)
    print(f"  joined: {len(df)} rows, missing AGE: {df['AGE'].isna().sum()}, "
          f"missing SMTSISCH: {df['SMTSISCH'].isna().sum()}, missing SMRIN: {df['SMRIN'].isna().sum()}")

    # Continuous metadata vs PCs
    continuous_cols = ["AGE_mid", "DTHHRDY", "SMRIN", "SMTSISCH", "SMRDLGTH"]
    pc_cols = [f"PC{i+1}" for i in range(50)]
    pc_df = pd.DataFrame(pc_scores, columns=pc_cols)
    pc_df["SAMPID"] = df["SAMPID"].values
    df_full = df.merge(pc_df, on="SAMPID", how="left")

    rho_table = pd.DataFrame(index=continuous_cols, columns=pc_cols, dtype=float)
    for col in continuous_cols:
        v = pd.to_numeric(df_full[col], errors="coerce").values
        for pc in pc_cols:
            x = df_full[pc].values
            mask = np.isfinite(v) & np.isfinite(x)
            if mask.sum() < 30:
                rho_table.loc[col, pc] = np.nan
                continue
            rho, _ = spearmanr(v[mask], x[mask])
            rho_table.loc[col, pc] = rho

    # Categorical (one-way ANOVA explained variance ω²-style)
    cat_cols = ["SEX", "SMCENTER", "SMNABTCH", "SMGEBTCH"]
    cat_table = pd.DataFrame(index=cat_cols, columns=pc_cols, dtype=float)
    for col in cat_cols:
        groups_lookup = df_full[col].fillna("NA").astype(str).values
        unique_g = sorted(set(groups_lookup))
        # cap to <= 30 groups so ANOVA is meaningful (batch column has hundreds otherwise)
        if len(unique_g) > 30:
            top = pd.Series(groups_lookup).value_counts().head(30).index.tolist()
            mask_keep = np.isin(groups_lookup, top)
            kept_groups = groups_lookup[mask_keep]
            kept_unique = sorted(set(kept_groups))
        else:
            mask_keep = np.ones_like(groups_lookup, dtype=bool)
            kept_groups = groups_lookup
            kept_unique = unique_g
        for pc in pc_cols:
            x = df_full[pc].values[mask_keep]
            buckets = [x[kept_groups == g] for g in kept_unique if (kept_groups == g).sum() > 1]
            if len(buckets) < 2:
                cat_table.loc[col, pc] = np.nan
                continue
            try:
                f, _ = f_oneway(*buckets)
                # convert F to a crude η² ≈ F * (k-1) / (F * (k-1) + (n-k))
                k = len(buckets)
                n = sum(b.size for b in buckets)
                eta2 = (f * (k - 1)) / (f * (k - 1) + (n - k))
                cat_table.loc[col, pc] = eta2
            except Exception:
                cat_table.loc[col, pc] = np.nan

    rho_table.to_csv(RESULTS / "q4_pc_metadata_spearman.csv")
    cat_table.to_csv(RESULTS / "q4_pc_metadata_eta2.csv")
    print("\nTop |ρ| of any PC for each continuous variable:")
    for col in continuous_cols:
        vals = rho_table.loc[col].abs()
        if vals.notna().any():
            best = vals.idxmax()
            print(f"  {col:<10} max |ρ| = {vals.max():.3f}  at {best}")

    print("\nTop η² of any PC for each categorical variable:")
    for col in cat_cols:
        vals = cat_table.loc[col]
        if vals.notna().any():
            best = vals.idxmax()
            print(f"  {col:<10} max η² = {vals.max():.3f}  at {best}")

    # Top PCs vs PC1, PC2, PC3 explanatory power
    print("\nFor PC1, PC2, PC3 — best-explaining metadata:")
    for pc in ("PC1", "PC2", "PC3", "PC4", "PC5"):
        cont = rho_table[pc].abs()
        cat = cat_table[pc]
        print(f"  {pc}:  cont best = {cont.idxmax() if cont.notna().any() else 'NA'} "
              f"(|ρ|={cont.max():.3f}),  cat best = {cat.idxmax() if cat.notna().any() else 'NA'} "
              f"(η²={cat.max():.3f})")

    # ── Heatmap ───────────────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 1, figsize=(13, 5.5), gridspec_kw={"height_ratios": [1, 0.8]})

    n_show = 15
    sub_rho = rho_table[pc_cols[:n_show]].astype(float)
    im0 = axes[0].imshow(sub_rho.values, aspect="auto", cmap="RdBu_r", vmin=-0.7, vmax=0.7)
    axes[0].set_yticks(range(len(continuous_cols)))
    axes[0].set_yticklabels(continuous_cols)
    axes[0].set_xticks(range(n_show))
    axes[0].set_xticklabels(pc_cols[:n_show], rotation=0)
    axes[0].set_title("Q4: Spearman ρ — PC vs continuous metadata")
    for i, col in enumerate(continuous_cols):
        for j, pc in enumerate(pc_cols[:n_show]):
            v = sub_rho.iloc[i, j]
            if pd.notna(v) and abs(v) > 0.15:
                axes[0].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                             color="white" if abs(v) > 0.45 else "black")
    plt.colorbar(im0, ax=axes[0], label="ρ")

    sub_cat = cat_table[pc_cols[:n_show]].astype(float)
    im1 = axes[1].imshow(sub_cat.values, aspect="auto", cmap="viridis", vmin=0, vmax=0.5)
    axes[1].set_yticks(range(len(cat_cols)))
    axes[1].set_yticklabels(cat_cols)
    axes[1].set_xticks(range(n_show))
    axes[1].set_xticklabels(pc_cols[:n_show])
    axes[1].set_title("PC vs categorical metadata (one-way ANOVA η²)")
    for i, col in enumerate(cat_cols):
        for j, pc in enumerate(pc_cols[:n_show]):
            v = sub_cat.iloc[i, j]
            if pd.notna(v) and v > 0.05:
                axes[1].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                             color="white" if v > 0.3 else "black")
    plt.colorbar(im1, ax=axes[1], label="η²")

    fig.tight_layout()
    fig.savefig(OUT / "q4_metadata_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"\nWrote {OUT / 'q4_metadata_heatmap.png'}")

    # ── Enrichr enrichment for PC1..PC5 ───────────────────────────────────
    enrichment_summary = []
    components = pca.components_  # (50, n_genes)
    n_genes = components.shape[1]
    print(f"\nEnrichr enrichment for top {N_PCS_FOR_ENRICHMENT} PCs ...")
    for k in range(N_PCS_FOR_ENRICHMENT):
        loadings = components[k]
        order = np.argsort(loadings)
        bot = order[:TOP_GENES_PER_DIRECTION]   # most negative
        top = order[-TOP_GENES_PER_DIRECTION:]  # most positive
        for direction, idx in [("pos", top), ("neg", bot)]:
            genes = [str(g) for g in shared_genes[idx]]
            tag = f"PC{k+1}_{direction}"
            try:
                sub_resp = enrichr_submit(genes, tag)
                lid = sub_resp["userListId"]
                for lib in LIBRARIES:
                    try:
                        tbl = enrichr_query(lid, lib, top_k=15)
                        tbl["pc"] = f"PC{k+1}"
                        tbl["direction"] = direction
                        tbl["library"] = lib
                        tbl.to_csv(RESULTS / f"q4_enrich_{tag}_{lib}.csv", index=False)
                        if not tbl.empty:
                            enrichment_summary.append({
                                "pc": f"PC{k+1}",
                                "direction": direction,
                                "library": lib,
                                "top_term": tbl.iloc[0]["term"],
                                "top_adj_p": float(tbl.iloc[0]["adj_p"]),
                                "top_combined": float(tbl.iloc[0]["combined_score"]),
                                "n_terms_returned": len(tbl),
                            })
                            print(f"  {tag} {lib}: {tbl.iloc[0]['term']}  adj-p={float(tbl.iloc[0]['adj_p']):.2e}")
                        time.sleep(0.6)
                    except Exception as exc:
                        print(f"  !! {tag} {lib} failed: {exc}")
            except Exception as exc:
                print(f"  !! submit {tag} failed: {exc}")

    pd.DataFrame(enrichment_summary).to_csv(RESULTS / "q4_enrichment_summary.csv", index=False)
    print(f"Wrote {RESULTS / 'q4_enrichment_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
