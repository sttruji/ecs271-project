"""Build Eraslan paired training set: per-(donor, tissue) bulk + per-(donor,
tissue, broad_celltype) sn pseudobulks. Outputs:

  /Users/rls/ecs271/data/sc/eraslan/eraslan_paired.npz
    bulk_x          (n_bulk, 11374) standardised log2(CPM+1) GTEx bulk
    bulk_donor      (n_bulk,)
    bulk_tissue     (n_bulk,)
    sn_x            (n_sn, 11374)   standardised log2(CPM+1) sn pseudobulk
    sn_donor        (n_sn,)
    sn_tissue       (n_sn,)
    sn_celltype     (n_sn,)         "all" for donor-level groups
    shared_genes    (11374,)

Uses the GTEx blood checkpoint's shared_genes / scaler so we can keep the
same architecture in Run 14.
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

# Eraslan tissue → GTEx bulk file path
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
OUT = "/Users/rls/ecs271/data/sc/eraslan/eraslan_paired.npz"
MIN_CELLS = 30


def _align(expr_log: np.ndarray, gene_names: np.ndarray, shared_genes: np.ndarray):
    name_to_idx = {n.upper(): i for i, n in enumerate(gene_names)}
    out = np.zeros((expr_log.shape[0], len(shared_genes)), dtype=np.float32)
    for j, g in enumerate(shared_genes):
        idx = name_to_idx.get(str(g).upper())
        if idx is not None:
            out[:, j] = expr_log[:, idx]
    return out


def _load_gtex_bulk(path: str, donor_filter: set[str], shared_genes, sm, sd):
    """Read a GTEx GCT, return per-donor log2(CPM+1) standardised matrix.

    Note: GTEx sample IDs look like GTEX-1HSMQ-0526-SM-XXXX; we collapse to the
    GTEX-XXXXX donor and average if multiple aliquots."""
    df = pd.read_csv(path, sep="\t", skiprows=2, compression="gzip")
    expr_raw = df.iloc[:, 2:].values.astype(np.float64)
    gene_names = df["Description"].values.astype(str)
    sample_ids = list(df.columns[2:])
    donor_for_sample = ["-".join(s.split("-")[:2]) for s in sample_ids]
    keep_cols = [i for i, d in enumerate(donor_for_sample) if d in donor_filter]
    if not keep_cols:
        return None, None
    sub = expr_raw[:, keep_cols]
    sub_donors = [donor_for_sample[i] for i in keep_cols]
    # Average per donor (some donors have replicate aliquots)
    df_t = pd.DataFrame(sub.T, index=sub_donors)
    avg = df_t.groupby(level=0).mean().values
    donor_order = list(df_t.groupby(level=0).mean().index)
    # CPM + log
    lib = avg.sum(axis=1, keepdims=True)
    log_cpm = np.log2(avg / np.where(lib == 0, 1, lib) * 1e6 + 1).astype(np.float32)
    aligned = _align(log_cpm, gene_names, shared_genes)
    scaled = ((aligned - sm) / sd).astype(np.float32)
    return scaled, donor_order


def main():
    shared_genes, sm, sd = load_shared_genes()
    print(f"shared genes: {len(shared_genes)}")

    print(f"\nLoading Eraslan {ERASLAN} …")
    a = ad.read_h5ad(ERASLAN)
    print(f"  {a.shape}")
    eraslan_donors = sorted(a.obs["Participant ID"].unique().tolist())
    print(f"  {len(eraslan_donors)} Eraslan donors: {eraslan_donors}")

    # === BULK side ===
    print("\n=== Building per-(donor, tissue) GTEx bulk ===")
    bulk_x_list, bulk_donor_list, bulk_tissue_list = [], [], []
    for tissue, p in TISSUE_BULK.items():
        scaled, donors = _load_gtex_bulk(p, set(eraslan_donors), shared_genes, sm, sd)
        if scaled is None:
            print(f"  {tissue}: 0 Eraslan donors found")
            continue
        print(f"  {tissue}: {len(donors)} donors → {donors}")
        bulk_x_list.append(scaled)
        bulk_donor_list.extend(donors)
        bulk_tissue_list.extend([tissue] * len(donors))
    bulk_x = np.vstack(bulk_x_list)
    bulk_donor = np.asarray(bulk_donor_list, dtype=str)
    bulk_tissue = np.asarray(bulk_tissue_list, dtype=str)
    print(f"\n  Total bulk samples: {bulk_x.shape}")

    # === SN side ===
    print("\n=== Building per-(donor, tissue, broad celltype) sn pseudobulks ===")
    X = a.X if sp.issparse(a.X) else sp.csr_matrix(a.X)
    if not isinstance(X, sp.csr_matrix):
        X = X.tocsr()
    sn_gene_names = np.asarray(a.var_names, dtype=str)

    obs = a.obs[["Participant ID", "tissue", "Broad cell type"]].astype(str)
    obs["row"] = np.arange(len(obs))
    grouped = obs.groupby(["Participant ID", "tissue", "Broad cell type"])["row"].apply(list)
    sn_x_list, sn_donor_list, sn_tissue_list, sn_celltype_list = [], [], [], []
    n_kept = 0; n_skipped = 0
    for (donor, tissue, ct), rows in grouped.items():
        if len(rows) < MIN_CELLS:
            n_skipped += 1; continue
        sub = X[np.asarray(rows), :]
        bulk = np.asarray(sub.sum(axis=0)).ravel()
        lib = bulk.sum()
        if lib == 0: continue
        log_cpm = np.log2(bulk / lib * 1e6 + 1).astype(np.float32)
        sn_x_list.append(log_cpm[None, :])
        sn_donor_list.append(donor); sn_tissue_list.append(tissue); sn_celltype_list.append(ct)
        n_kept += 1
    print(f"  built {n_kept} per-(donor,tissue,celltype) pseudobulks (skipped {n_skipped} for <{MIN_CELLS} cells)")

    # Also add per-(donor, tissue) "all-celltype" pseudobulks
    grouped2 = obs.groupby(["Participant ID", "tissue"])["row"].apply(list)
    for (donor, tissue), rows in grouped2.items():
        if len(rows) < MIN_CELLS:
            continue
        sub = X[np.asarray(rows), :]
        bulk = np.asarray(sub.sum(axis=0)).ravel()
        lib = bulk.sum()
        if lib == 0: continue
        log_cpm = np.log2(bulk / lib * 1e6 + 1).astype(np.float32)
        sn_x_list.append(log_cpm[None, :])
        sn_donor_list.append(donor); sn_tissue_list.append(tissue); sn_celltype_list.append("ALL")
        n_kept += 1
    print(f"  +per-(donor,tissue) ALL groups: total now {n_kept}")

    sn_x = np.vstack(sn_x_list)
    sn_aligned = _align(sn_x, sn_gene_names, shared_genes)
    sn_scaled = ((sn_aligned - sm) / sd).astype(np.float32)
    sn_donor = np.asarray(sn_donor_list, dtype=str)
    sn_tissue = np.asarray(sn_tissue_list, dtype=str)
    sn_celltype = np.asarray(sn_celltype_list, dtype=str)
    print(f"\n  sn_x: {sn_scaled.shape}")

    np.savez(OUT,
             bulk_x=bulk_x, bulk_donor=bulk_donor, bulk_tissue=bulk_tissue,
             sn_x=sn_scaled, sn_donor=sn_donor, sn_tissue=sn_tissue, sn_celltype=sn_celltype,
             shared_genes=shared_genes.astype(str))
    print(f"\n  saved → {OUT}")


if __name__ == "__main__":
    main()
