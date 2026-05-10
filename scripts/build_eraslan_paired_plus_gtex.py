"""Build Eraslan paired + ALL GTEx bulk for the same 8 tissues.

Adds ~600-800 unpaired bulk samples per tissue for the encoder to see massive
donor variation. The paired data for the flip loss stays the same (16 Eraslan
donors), but the encoder is trained on 4000+ bulk samples total.

Output: /Users/rls/ecs271/data/sc/eraslan/eraslan_paired_plus_gtex.npz
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.data import load_shared_genes  # noqa: E402

TISSUE_BULK = {
    "skeletalmuscle": "/Users/rls/ecs271/data/bulk/gtex_v10_muscle_skeletal.gct.gz",
    "lung": "/Users/rls/ecs271/data/bulk/gtex_v10_lung.gct.gz",
    "breast": "/Users/rls/ecs271/data/bulk/gene_reads_v10_breast_mammary_tissue.gct.gz",
    "esophagusmucosa": "/Users/rls/ecs271/data/bulk/gene_reads_v10_esophagus_mucosa.gct.gz",
    "esophagusmuscularis": "/Users/rls/ecs271/data/bulk/gene_reads_v10_esophagus_muscularis.gct.gz",
    "heart": "/Users/rls/ecs271/data/bulk/gene_reads_v10_heart_atrial_appendage.gct.gz",
    "prostate": "/Users/rls/ecs271/data/bulk/gene_reads_v10_prostate.gct.gz",
    "skin": "/Users/rls/ecs271/data/bulk/gene_reads_v10_skin_not_sun_exposed_suprapubic.gct.gz",
}
ERASLAN = "/Users/rls/ecs271/data/sc/eraslan/GTEx_8_tissues_snRNAseq_atlas.h5ad"
OUT = "/Users/rls/ecs271/data/sc/eraslan/eraslan_paired_plus_gtex.npz"


def _align(expr_log: np.ndarray, gene_names: np.ndarray, shared_genes: np.ndarray):
    name_to_idx = {n.upper(): i for i, n in enumerate(gene_names)}
    out = np.zeros((expr_log.shape[0], len(shared_genes)), dtype=np.float32)
    for j, g in enumerate(shared_genes):
        idx = name_to_idx.get(str(g).upper())
        if idx is not None:
            out[:, j] = expr_log[:, idx]
    return out


def _load_gtex_bulk_all(path, shared_genes, sm, sd, eraslan_donors):
    """Read whole GTEx GCT, return per-donor log2(CPM+1) standardized matrix.
    Marks each row as 'eraslan' or 'extra' based on donor."""
    df = pd.read_csv(path, sep="\t", skiprows=2, compression="gzip")
    expr_raw = df.iloc[:, 2:].values.astype(np.float64)
    gene_names = df["Description"].values.astype(str)
    sample_ids = list(df.columns[2:])
    donor_for_sample = ["-".join(s.split("-")[:2]) for s in sample_ids]
    df_t = pd.DataFrame(expr_raw.T, index=donor_for_sample)
    avg = df_t.groupby(level=0).mean().values
    donor_order = list(df_t.groupby(level=0).mean().index)
    lib = avg.sum(axis=1, keepdims=True)
    log_cpm = np.log2(avg / np.where(lib == 0, 1, lib) * 1e6 + 1).astype(np.float32)
    aligned = _align(log_cpm, gene_names, shared_genes)
    scaled = ((aligned - sm) / sd).astype(np.float32)
    is_eraslan = np.array([d in eraslan_donors for d in donor_order])
    return scaled, donor_order, is_eraslan


def main():
    shared_genes, sm, sd = load_shared_genes()
    print(f"shared genes: {len(shared_genes)}")

    print(f"\nLoading Eraslan {ERASLAN} …")
    a = ad.read_h5ad(ERASLAN)
    print(f"  {a.shape}")
    eraslan_donors = sorted(a.obs["Participant ID"].unique().tolist())
    print(f"  16 Eraslan donors")

    # === BULK side: ALL GTEx donors ===
    print("\n=== Building per-(donor, tissue) GTEx bulk for ALL donors ===")
    bulk_x_list, bulk_donor_list, bulk_tissue_list, bulk_is_eraslan = [], [], [], []
    for tissue, p in TISSUE_BULK.items():
        scaled, donors, is_e = _load_gtex_bulk_all(p, shared_genes, sm, sd, set(eraslan_donors))
        n_e = int(is_e.sum())
        print(f"  {tissue}: {len(donors)} donors total ({n_e} Eraslan, {len(donors)-n_e} extra)")
        bulk_x_list.append(scaled)
        bulk_donor_list.extend(donors)
        bulk_tissue_list.extend([tissue] * len(donors))
        bulk_is_eraslan.extend(is_e.tolist())
    bulk_x = np.vstack(bulk_x_list)
    bulk_donor = np.asarray(bulk_donor_list, dtype=str)
    bulk_tissue = np.asarray(bulk_tissue_list, dtype=str)
    bulk_is_eraslan = np.asarray(bulk_is_eraslan, dtype=bool)
    print(f"\n  Total bulk samples: {bulk_x.shape}  ({bulk_is_eraslan.sum()} Eraslan paired, "
          f"{(~bulk_is_eraslan).sum()} extra)")

    # === SN side: same as before ===
    print("\n=== Building per-(donor, tissue, broad celltype) sn pseudobulks ===")
    X = a.X if sp.issparse(a.X) else sp.csr_matrix(a.X)
    if not isinstance(X, sp.csr_matrix):
        X = X.tocsr()
    sn_gene_names = np.asarray(a.var_names, dtype=str)
    obs = a.obs[["Participant ID", "tissue", "Broad cell type"]].astype(str)
    obs["row"] = np.arange(len(obs))
    grouped = obs.groupby(["Participant ID", "tissue", "Broad cell type"])["row"].apply(list)
    sn_x_list, sn_donor_list, sn_tissue_list, sn_celltype_list = [], [], [], []
    for (donor, tissue, ct), rows in grouped.items():
        if len(rows) < 30:
            continue
        sub = X[np.asarray(rows), :]
        bulk = np.asarray(sub.sum(axis=0)).ravel()
        lib = bulk.sum()
        if lib == 0: continue
        log_cpm = np.log2(bulk / lib * 1e6 + 1).astype(np.float32)
        sn_x_list.append(log_cpm[None, :])
        sn_donor_list.append(donor); sn_tissue_list.append(tissue); sn_celltype_list.append(ct)
    grouped2 = obs.groupby(["Participant ID", "tissue"])["row"].apply(list)
    for (donor, tissue), rows in grouped2.items():
        if len(rows) < 30: continue
        sub = X[np.asarray(rows), :]
        bulk = np.asarray(sub.sum(axis=0)).ravel()
        lib = bulk.sum()
        if lib == 0: continue
        log_cpm = np.log2(bulk / lib * 1e6 + 1).astype(np.float32)
        sn_x_list.append(log_cpm[None, :])
        sn_donor_list.append(donor); sn_tissue_list.append(tissue); sn_celltype_list.append("ALL")
    sn_x = np.vstack(sn_x_list)
    sn_aligned = _align(sn_x, sn_gene_names, shared_genes)
    sn_scaled = ((sn_aligned - sm) / sd).astype(np.float32)
    print(f"  sn pseudobulks: {sn_scaled.shape}")

    np.savez(OUT,
             bulk_x=bulk_x, bulk_donor=bulk_donor, bulk_tissue=bulk_tissue,
             bulk_is_eraslan=bulk_is_eraslan,
             sn_x=sn_scaled,
             sn_donor=np.asarray(sn_donor_list, dtype=str),
             sn_tissue=np.asarray(sn_tissue_list, dtype=str),
             sn_celltype=np.asarray(sn_celltype_list, dtype=str),
             shared_genes=shared_genes.astype(str))
    print(f"\n  saved → {OUT}")


if __name__ == "__main__":
    main()
