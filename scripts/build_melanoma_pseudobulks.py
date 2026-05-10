"""Build melanoma PBMC pseudobulks for training Run 13.

The held-out 7 paired donors (YUALOE, YUFURL, YUHERN, YUHONEY, YUNANCY, YUROD,
YUTORY) stay out — they're for Q21 Test 2.

Outputs to /Users/rls/ecs271/data/sc/:
  melanoma_bulk_train.npz       (n_donors, 11374)  log2(TPM+1) standardised
  melanoma_sc_pb_train.npz      (n_pb,      11374) per-donor and per-(donor,cluster)
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.data import load_shared_genes  # noqa: E402

PAIRED_HELDOUT = {"YUALOE", "YUFURL", "YUHERN", "YUHONEY", "YUNANCY", "YUROD", "YUTORY"}

BULK_TPM = "/Users/rls/ecs271/data/paired_test2/gse186143_bulk_pbmc/GSE186143_bulk_TPM_samples.txt.gz"
SC_BATCHES = [
    ("/Users/rls/ecs271/data/paired_test2/gse189125_sc_pbmc/GSE189125_5prime_scRNAseq_seqbatchA_counts.txt.gz",
     "/Users/rls/ecs271/data/paired_test2/gse189125_sc_pbmc/GSE189125_5prime_scRNAseq_seqbatchA_metadata.txt.gz",
     "patient ID"),
    ("/Users/rls/ecs271/data/paired_test2/gse189125_sc_pbmc/GSE189125_5prime_scRNAseq_seqbatchB_counts.txt.gz",
     "/Users/rls/ecs271/data/paired_test2/gse189125_sc_pbmc/GSE189125_5prime_scRNAseq_seqbatchB_metadata.txt.gz",
     "patient.ID"),
]
OUT_BULK = "/Users/rls/ecs271/data/sc/melanoma_bulk_train.npz"
OUT_SC = "/Users/rls/ecs271/data/sc/melanoma_sc_pb_train.npz"


def _align(expr_log: np.ndarray, gene_names: np.ndarray, shared_genes: np.ndarray):
    name_to_idx = {n.upper(): i for i, n in enumerate(gene_names)}
    out = np.zeros((expr_log.shape[0], len(shared_genes)), dtype=np.float32)
    for j, g in enumerate(shared_genes):
        idx = name_to_idx.get(str(g).upper())
        if idx is not None:
            out[:, j] = expr_log[:, idx]
    return out


def main():
    shared_genes, scaler_mean, scaler_std = load_shared_genes()
    print(f"shared genes: {len(shared_genes)}")

    print("\n[bulk] reading GSE186143 …")
    df = pd.read_csv(BULK_TPM, sep="\t", index_col=0)
    keep = [c for c in df.columns if c not in PAIRED_HELDOUT]
    print(f"  bulk donors total={df.shape[1]}, kept (non-Test2)={len(keep)}")
    sub = df[keep].T.values.astype(np.float64)
    expr_log = np.log2(sub + 1).astype(np.float32)
    bulk_aligned = _align(expr_log, np.asarray(df.index, dtype=str), shared_genes)
    bulk_scaled = ((bulk_aligned - scaler_mean) / scaler_std).astype(np.float32)
    np.savez(OUT_BULK,
             expr_scaled=bulk_scaled,
             donor=np.asarray(keep, dtype=str),
             shared_genes=shared_genes.astype(str))
    print(f"  saved → {OUT_BULK}  ({bulk_scaled.shape})")

    print("\n[sc] streaming GSE189125 batches → per-(donor, cluster) pseudobulks")
    pseudo: dict[tuple[str, str], np.ndarray] = {}
    gene_names_sc = None
    for counts_path, meta_path, donor_col in SC_BATCHES:
        print(f"  loading {Path(counts_path).name}", flush=True)
        md = pd.read_csv(meta_path, sep="\t").set_index("barcode")
        cluster_col = "unsupervised cell cluster" if "unsupervised cell cluster" in md.columns else None
        donor_per_cell = md[donor_col]
        cluster_per_cell = md[cluster_col] if cluster_col else None

        with gzip.open(counts_path, "rt") as f:
            header = f.readline().rstrip().split()
            d_arr = donor_per_cell.reindex(header)
            keep_mask = d_arr.notna() & ~d_arr.isin(PAIRED_HELDOUT)
            keep_idx = np.where(keep_mask.values)[0].astype(np.int64)
            print(f"    cells in batch: {len(header)}; kept: {len(keep_idx)}", flush=True)
            if len(keep_idx) == 0:
                continue
            keep_donors = d_arr.values[keep_idx]
            if cluster_per_cell is not None:
                keep_clusters = cluster_per_cell.reindex(header).values[keep_idx]
            else:
                keep_clusters = np.array(["all"] * len(keep_idx))
            keys = [(str(d), str(c)) for d, c in zip(keep_donors, keep_clusters)]
            uniq_keys = sorted(set(keys))
            key_to_idx = {k: i for i, k in enumerate(uniq_keys)}
            local_idx = np.array([key_to_idx[k] for k in keys], dtype=np.int64)
            print(f"    groups in batch: {len(uniq_keys)}", flush=True)

            gene_names_local = []
            rows_buffer: list[np.ndarray] = []
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) < 2:
                    continue
                gene_names_local.append(parts[0])
                vals = np.asarray(parts[1:], dtype=np.float32)
                kept_vals = vals[keep_idx]
                row = np.zeros(len(uniq_keys), dtype=np.float32)
                np.add.at(row, local_idx, kept_vals)
                rows_buffer.append(row)
            arr = np.stack(rows_buffer)
            if gene_names_sc is None:
                gene_names_sc = np.asarray(gene_names_local, dtype=str)
            for j, k in enumerate(uniq_keys):
                v = arr[:, j].astype(np.float64)
                pseudo[k] = pseudo.get(k, np.zeros_like(v)) + v

    keys_final = sorted(pseudo.keys())
    counts = np.stack([pseudo[k] for k in keys_final])  # (n_groups, n_genes)
    # Drop tiny groups (< 50 cells worth — proxy via lib size threshold)
    lib = counts.sum(axis=1)
    keep_groups = lib > 1e5
    print(f"  total groups built: {len(keys_final)}; kept (lib > 1e5): {keep_groups.sum()}")
    counts = counts[keep_groups]
    keys_kept = [keys_final[i] for i in range(len(keys_final)) if keep_groups[i]]
    log_cpm = np.log2(counts / counts.sum(axis=1, keepdims=True) * 1e6 + 1).astype(np.float32)
    sc_aligned = _align(log_cpm, gene_names_sc, shared_genes)
    sc_scaled = ((sc_aligned - scaler_mean) / scaler_std).astype(np.float32)
    donors = np.asarray([k[0] for k in keys_kept], dtype=str)
    clusters = np.asarray([k[1] for k in keys_kept], dtype=str)
    np.savez(OUT_SC,
             expr_scaled=sc_scaled,
             donor=donors,
             cluster=clusters,
             shared_genes=shared_genes.astype(str))
    print(f"  saved → {OUT_SC}  ({sc_scaled.shape})")
    print(f"  donor breakdown: {pd.Series(donors).value_counts().to_dict()}")


if __name__ == "__main__":
    main()
