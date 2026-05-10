"""Build HCA per-donor-celltype pseudobulks from BL_standard_design.h5ad.

Outputs `/Users/rls/ecs271/data/sc/hca_celltype_pseudobulk.npz` with:
  expr_log     (n_pseudobulk, 11374)  log2(CPM+1) aligned to shared_genes
  expr_scaled  (n_pseudobulk, 11374)  standardised by the GTEx scaler
  donor        (n_pseudobulk,)        donor id
  celltype     (n_pseudobulk,)        cell-type label
  n_cells      (n_pseudobulk,)        number of cells aggregated

Skips groups with fewer than min_cells cells.
"""
from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.data import load_shared_genes  # noqa: E402

H5AD = "/Users/rls/ecs271/data/sc/hca_raw/BL_standard_design.h5ad"
OUT = "/Users/rls/ecs271/data/sc/hca_celltype_pseudobulk.npz"
MIN_CELLS = 30
DONOR_COL = "Donor"
ANNO_COL = "anno"


def main():
    print(f"loading {H5AD} …")
    a = ad.read_h5ad(H5AD)  # in memory; ~2 GB
    print(f"  cells={a.n_obs} genes={a.n_vars}")
    # Use raw counts if available, else .X
    if a.raw is not None:
        X = a.raw.X
        gene_names = np.asarray(a.raw.var_names, dtype=str)
        print(f"  using raw.X ({X.shape}, type={type(X).__name__})")
    else:
        X = a.X
        gene_names = np.asarray(a.var_names, dtype=str)
        print(f"  using .X ({X.shape})")
    # Force CSR for fast row sums
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    elif not isinstance(X, sp.csr_matrix):
        X = X.tocsr()

    obs = a.obs[[DONOR_COL, ANNO_COL]].astype(str)
    print("  donors:", sorted(obs[DONOR_COL].unique()))
    print("  cell types:", sorted(obs[ANNO_COL].unique()))

    groups: dict[tuple[str, str], list[int]] = {}
    for i, (d, c) in enumerate(zip(obs[DONOR_COL].values, obs[ANNO_COL].values)):
        groups.setdefault((d, c), []).append(i)

    rows = []
    donors = []
    celltypes = []
    n_cells_list = []
    for (d, c), idx in sorted(groups.items()):
        if len(idx) < MIN_CELLS:
            continue
        idx_arr = np.asarray(idx)
        sub = X[idx_arr, :]
        bulk = np.asarray(sub.sum(axis=0)).ravel()
        rows.append(bulk)
        donors.append(d)
        celltypes.append(c)
        n_cells_list.append(len(idx))
        print(f"  ({d:>3}, {c:<30}) n_cells={len(idx)}")

    counts = np.vstack(rows).astype(np.float64)
    donors_arr = np.asarray(donors, dtype=str)
    celltypes_arr = np.asarray(celltypes, dtype=str)
    n_cells_arr = np.asarray(n_cells_list, dtype=np.int64)
    print(f"\n  built {counts.shape[0]} pseudobulks")

    # log-CPM
    lib = counts.sum(axis=1, keepdims=True)
    log_cpm = np.log2(counts / lib * 1e6 + 1).astype(np.float32)

    # Align to shared_genes
    shared_genes, scaler_mean, scaler_std = load_shared_genes()
    name_to_idx = {n.upper(): i for i, n in enumerate(gene_names)}
    expr_log = np.zeros((log_cpm.shape[0], len(shared_genes)), dtype=np.float32)
    found = 0
    for j, g in enumerate(shared_genes):
        idx = name_to_idx.get(str(g).upper())
        if idx is not None:
            expr_log[:, j] = log_cpm[:, idx]
            found += 1
    print(f"  aligned to shared_genes: {found}/{len(shared_genes)} found")

    expr_scaled = ((expr_log - scaler_mean) / scaler_std).astype(np.float32)
    np.savez(
        OUT,
        expr_log=expr_log,
        expr_scaled=expr_scaled,
        donor=donors_arr,
        celltype=celltypes_arr,
        n_cells=n_cells_arr,
        shared_genes=shared_genes.astype(str),
    )
    print(f"  saved → {OUT}")


if __name__ == "__main__":
    main()
