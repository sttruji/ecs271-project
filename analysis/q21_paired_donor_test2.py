"""Q21 — Paired-donor Test 2 on the Run-10 disentangled VAE.

Loads paired bulk + sc PBMC samples from 7 melanoma donors that have BOTH
modalities (GSE186143 bulk + GSE189125 sc). For each paired donor:

  1. Encode the bulk → flip modality → decode → x_hat_sc
  2. Compute distances to all 7 sc pseudobulks
  3. Test 2 (the proposal's named test): does the FLIPPED bulk's NN among
     the 7 held-out sc pseudobulks pick the SAME-DONOR sc pseudobulk?
  4. Symmetric: encode sc → flip → decode → x_hat_bulk; NN among 7 bulks.

This is the first paired-data validation. Random chance for top-1 = 1/7 ≈ 14%.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.disentangled_vae import DisentangledVAE, DisentangledConfig
from analysis.q20_disentangled_vae import DEVICE, OUT
from pipeline.data import load_shared_genes  # noqa: E402

PAIRED_DONORS = ["YUALOE", "YUFURL", "YUHERN", "YUHONEY", "YUNANCY", "YUROD", "YUTORY"]
BULK_TPM = "/Users/rls/ecs271/data/paired_test2/gse186143_bulk_pbmc/GSE186143_bulk_TPM_samples.txt.gz"
SC_BATCHES = [
    ("/Users/rls/ecs271/data/paired_test2/gse189125_sc_pbmc/GSE189125_5prime_scRNAseq_seqbatchA_counts.txt.gz",
     "/Users/rls/ecs271/data/paired_test2/gse189125_sc_pbmc/GSE189125_5prime_scRNAseq_seqbatchA_metadata.txt.gz",
     "patient ID"),
    ("/Users/rls/ecs271/data/paired_test2/gse189125_sc_pbmc/GSE189125_5prime_scRNAseq_seqbatchB_counts.txt.gz",
     "/Users/rls/ecs271/data/paired_test2/gse189125_sc_pbmc/GSE189125_5prime_scRNAseq_seqbatchB_metadata.txt.gz",
     "patient.ID"),
]
import os
CHECKPOINT = OUT / os.environ.get("Q21_CKPT", "q20_run10_celltype_pb.pt")
RESULT = OUT / os.environ.get("Q21_OUT", "q21_paired_donor_test2.json")


def _align(expr_log: np.ndarray, gene_names: np.ndarray, shared_genes: np.ndarray):
    """Reorder expr_log to match shared_genes; missing genes filled with zeros."""
    name_to_idx = {n.upper(): i for i, n in enumerate(gene_names)}
    out = np.zeros((expr_log.shape[0], len(shared_genes)), dtype=np.float32)
    found = 0
    for j, g in enumerate(shared_genes):
        idx = name_to_idx.get(str(g).upper())
        if idx is not None:
            out[:, j] = expr_log[:, idx]
            found += 1
    return out, found


def load_bulk():
    df = pd.read_csv(BULK_TPM, sep="\t", index_col=0)
    print(f"  GSE186143 bulk: {df.shape} (genes x samples)")
    keep = [d for d in PAIRED_DONORS if d in df.columns]
    print(f"  paired donors found in bulk: {keep}")
    sub = df[keep].T.values.astype(np.float64)  # (donors, genes)
    expr_log = np.log2(sub + 1).astype(np.float32)  # TPM is already library-normalised
    gene_names = np.asarray(df.index, dtype=str)
    return expr_log, gene_names, keep


def load_sc_pseudobulk(donors_to_keep: list[str], n_subsample: int | None = None,
                        n_replicates: int = 1, seed: int = 0):
    """Sum sc counts per donor across both batches → pseudobulks per donor.

    n_subsample: if set, randomly subsample exactly this many cells per donor
      so all pseudobulks come from equal-sized cell pools (removes the
      magnitude/noise bias from cell-count imbalance).
    n_replicates: build this many independent subsampled pseudobulks per donor
      (gives a sense of within-donor variance under subsampling).
    """
    rng = np.random.default_rng(seed)

    # First pass: collect cell indices per donor
    cells_per_donor: dict[str, list[tuple[str, int]]] = {}  # donor → [(batch_path, col_idx)]
    headers: dict[str, list[str]] = {}
    for counts_path, meta_path, donor_col in SC_BATCHES:
        md = pd.read_csv(meta_path, sep="\t").set_index("barcode")
        donor_per_cell = md[donor_col]
        with gzip.open(counts_path, "rt") as f:
            header = f.readline().rstrip().split()
        headers[counts_path] = header
        d_arr = donor_per_cell.reindex(header)
        for col_idx, d in enumerate(d_arr.values):
            if d in donors_to_keep:
                cells_per_donor.setdefault(d, []).append((counts_path, col_idx))

    print(f"  cells per donor (full): "
          + ", ".join(f"{d}={len(v)}" for d, v in sorted(cells_per_donor.items())),
          flush=True)

    if n_subsample is None:
        # Use ALL cells per donor (legacy behavior)
        chosen_per_donor_per_rep = {d: [v] for d, v in cells_per_donor.items()}
    else:
        # Subsample exactly n_subsample cells per donor; n_replicates draws
        chosen_per_donor_per_rep = {}
        for d, cells in cells_per_donor.items():
            if len(cells) < n_subsample:
                print(f"    WARN: {d} only has {len(cells)} cells (< {n_subsample}); using all")
                chosen_per_donor_per_rep[d] = [cells]
            else:
                chosen_per_donor_per_rep[d] = [
                    [cells[i] for i in rng.choice(len(cells), n_subsample, replace=False)]
                    for _ in range(n_replicates)
                ]

    # Second pass: stream the count files, summing into per-(donor, rep) buckets
    pseudo: dict[tuple[str, int], np.ndarray] = {}
    gene_names_global = None

    cells_by_batch_donor_rep: dict[str, dict[tuple[str, int], list[int]]] = {}
    for d, reps in chosen_per_donor_per_rep.items():
        for r, cells in enumerate(reps):
            for batch_path, col_idx in cells:
                cells_by_batch_donor_rep.setdefault(batch_path, {}).setdefault((d, r), []).append(col_idx)

    for counts_path, meta_path, donor_col in SC_BATCHES:
        if counts_path not in cells_by_batch_donor_rep:
            continue
        groups_in_batch = cells_by_batch_donor_rep[counts_path]
        keys = sorted(groups_in_batch.keys())
        cols_per_key = [np.asarray(groups_in_batch[k], dtype=np.int64) for k in keys]
        all_cols = np.concatenate(cols_per_key)
        key_per_col = np.concatenate([np.full(len(c), i, dtype=np.int64)
                                       for i, c in enumerate(cols_per_key)])
        print(f"  loading {Path(counts_path).name}: kept {len(all_cols)} cells across "
              f"{len(keys)} (donor,rep) groups", flush=True)
        with gzip.open(counts_path, "rt") as f:
            f.readline()  # skip header (already parsed)
            gene_names_local = []
            rows_buffer: list[np.ndarray] = []
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) < 2:
                    continue
                gene_names_local.append(parts[0])
                vals = np.asarray(parts[1:], dtype=np.float32)
                kept_vals = vals[all_cols]
                row = np.zeros(len(keys), dtype=np.float32)
                np.add.at(row, key_per_col, kept_vals)
                rows_buffer.append(row)
            arr = np.stack(rows_buffer)
            if gene_names_global is None:
                gene_names_global = np.asarray(gene_names_local, dtype=str)
            for j, k in enumerate(keys):
                v = arr[:, j].astype(np.float64)
                pseudo[k] = pseudo.get(k, np.zeros_like(v)) + v

    keys_final = sorted(pseudo.keys())
    counts = np.stack([pseudo[k] for k in keys_final])
    lib = counts.sum(axis=1, keepdims=True)
    log_cpm = np.log2(counts / np.where(lib == 0, 1, lib) * 1e6 + 1).astype(np.float32)
    donors = [k[0] for k in keys_final]
    reps = [k[1] for k in keys_final]
    print(f"  sc pseudobulks built: {len(keys_final)} groups, lib sizes "
          f"{lib.ravel().astype(int).tolist()}")
    return log_cpm, gene_names_global, donors, reps


def _pearson_row(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a**2).sum() * (b**2).sum()) + 1e-12))


def _cosine(a, b):
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    shared_genes, scaler_mean, scaler_std = load_shared_genes()
    print(f"shared genes: {len(shared_genes)}")

    print("\n[1/3] Loading paired bulk …")
    bulk_log, bulk_genes, bulk_donor_order = load_bulk()
    bulk_aligned, bf = _align(bulk_log, bulk_genes, shared_genes)
    bulk_scaled = ((bulk_aligned - scaler_mean) / scaler_std).astype(np.float32)
    print(f"  aligned: {bf}/{len(shared_genes)} shared genes found in bulk")

    n_subsample = int(os.environ.get("Q21_SUBSAMPLE", "0")) or None
    n_replicates = int(os.environ.get("Q21_REPLICATES", "1"))
    print(f"\n[2/3] Loading paired sc → pseudobulk (subsample={n_subsample}, reps={n_replicates}) …")
    sc_log, sc_genes, sc_donor_order, sc_rep_order = load_sc_pseudobulk(
        bulk_donor_order, n_subsample=n_subsample, n_replicates=n_replicates)
    sc_aligned, sf = _align(sc_log, sc_genes, shared_genes)
    sc_scaled = ((sc_aligned - scaler_mean) / scaler_std).astype(np.float32)
    print(f"  aligned: {sf}/{len(shared_genes)} shared genes found in sc")

    # If multiple replicates per donor: average them into one sc per donor for the
    # NN test (or keep separately and treat as in-pool noise — here we average).
    if n_replicates > 1:
        from collections import defaultdict
        d_to_rows: dict[str, list[int]] = defaultdict(list)
        for i, d in enumerate(sc_donor_order):
            d_to_rows[d].append(i)
        avg_donors = sorted(d_to_rows.keys())
        sc_scaled = np.stack([sc_scaled[d_to_rows[d]].mean(axis=0) for d in avg_donors])
        sc_donor_order = avg_donors

    common = [d for d in bulk_donor_order if d in sc_donor_order]
    bulk_idx = [bulk_donor_order.index(d) for d in common]
    sc_idx = [sc_donor_order.index(d) for d in common]
    bulk_scaled = bulk_scaled[bulk_idx]
    sc_scaled = sc_scaled[sc_idx]
    print(f"\n  Aligned {len(common)} paired donors: {common}")

    print(f"\n[3/3] Loading Run-10 checkpoint and running flips …")
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    cfg = DisentangledConfig(**ckpt["config"]) if isinstance(ckpt["config"], dict) else ckpt["config"]
    model = DisentangledVAE(cfg).to(DEVICE).eval()
    model.load_state_dict(ckpt["state_dict"])

    bulk_t = torch.from_numpy(bulk_scaled).to(DEVICE)
    sc_t = torch.from_numpy(sc_scaled).to(DEVICE)

    with torch.no_grad():
        mu_m_b, _, mu_b_b, _ = model.encode(bulk_t)
        mu_m_s, _, mu_b_s, _ = model.encode(sc_t)
        bulk_mean_z0 = mu_m_b[:, 0].mean().item()
        sc_mean_z0 = mu_m_s[:, 0].mean().item()

        # Flip bulk → sc
        z_m_flip = mu_m_b.clone(); z_m_flip[:, 0] = sc_mean_z0
        x_hat_sc = model.decode(z_m_flip, mu_b_b).cpu().numpy()
        x_hat_bulk_orig = model.decode(mu_m_b, mu_b_b).cpu().numpy()

        # Flip sc → bulk
        z_m_flip_s = mu_m_s.clone(); z_m_flip_s[:, 0] = bulk_mean_z0
        x_hat_bulk = model.decode(z_m_flip_s, mu_b_s).cpu().numpy()
        x_hat_sc_orig = model.decode(mu_m_s, mu_b_s).cpu().numpy()

    print(f"  encoded z_meta[modality]: bulk_mean={bulk_mean_z0:.3f}, sc_mean={sc_mean_z0:.3f}")

    # === TEST 2 (the proposal's paired-donor test) ===
    n = len(common)
    sc_real = sc_scaled        # (n, G)
    bulk_real = bulk_scaled    # (n, G)

    # 2a. flipped-bulk → NN among the n real sc pseudobulks. Should pick same-donor.
    cos_flipB_to_realSC = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cos_flipB_to_realSC[i, j] = _cosine(x_hat_sc[i], sc_real[j])
    # NN = argmax cosine sim (= argmin cosine distance)
    top1_flipB_to_realSC = (np.argmax(cos_flipB_to_realSC, axis=1) == np.arange(n)).mean()

    # rank of true-pair / total
    ranks_b2s = []
    for i in range(n):
        order = np.argsort(-cos_flipB_to_realSC[i])  # descending
        rank = int(np.where(order == i)[0][0]) + 1
        ranks_b2s.append(rank)

    # Symmetric: flipped-sc → NN among real bulks.
    cos_flipS_to_realBULK = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cos_flipS_to_realBULK[i, j] = _cosine(x_hat_bulk[i], bulk_real[j])
    top1_flipS_to_realB = (np.argmax(cos_flipS_to_realBULK, axis=1) == np.arange(n)).mean()
    ranks_s2b = []
    for i in range(n):
        order = np.argsort(-cos_flipS_to_realBULK[i])
        rank = int(np.where(order == i)[0][0]) + 1
        ranks_s2b.append(rank)

    # Baseline: ORIGINAL (no flip) bulk reconstruction → NN among sc. Expected: 1/n.
    cos_origB_to_realSC = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cos_origB_to_realSC[i, j] = _cosine(x_hat_bulk_orig[i], sc_real[j])
    top1_origB_to_realSC = (np.argmax(cos_origB_to_realSC, axis=1) == np.arange(n)).mean()

    # Baseline: real bulk → NN among real sc (no model, raw cosine on transcriptomes)
    cos_realB_to_realSC = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cos_realB_to_realSC[i, j] = _cosine(bulk_real[i], sc_real[j])
    top1_realB_to_realSC = (np.argmax(cos_realB_to_realSC, axis=1) == np.arange(n)).mean()

    out = {
        "checkpoint": str(CHECKPOINT.name),
        "n_paired_donors": int(n),
        "donor_order": common,
        "test2_top1_flipBULK_to_realSC_NN_correct": float(top1_flipB_to_realSC),
        "test2_top1_flipSC_to_realBULK_NN_correct": float(top1_flipS_to_realB),
        "test2_baseline_top1_origBULK_to_realSC": float(top1_origB_to_realSC),
        "test2_baseline_top1_realBULK_to_realSC_no_model": float(top1_realB_to_realSC),
        "test2_random_chance_top1": float(1.0 / n),
        "test2_ranks_bulk_to_sc": ranks_b2s,
        "test2_mean_rank_bulk_to_sc": float(np.mean(ranks_b2s)),
        "test2_ranks_sc_to_bulk": ranks_s2b,
        "test2_mean_rank_sc_to_bulk": float(np.mean(ranks_s2b)),
        "encoded_z_meta_modality_bulk_mean": float(bulk_mean_z0),
        "encoded_z_meta_modality_sc_mean": float(sc_mean_z0),
    }
    print()
    for k, v in out.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif isinstance(v, int):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")

    print("\n--- cosine matrices (flipped→real, diag = same-donor) ---")
    print("flipBulk → realSC:")
    print(pd.DataFrame(cos_flipB_to_realSC, index=common, columns=common).round(3).to_string())
    print("\nflipSC → realBulk:")
    print(pd.DataFrame(cos_flipS_to_realBULK, index=common, columns=common).round(3).to_string())

    with RESULT.open("w") as f:
        json.dump({**out,
                   "cos_flipBulk_to_realSC": cos_flipB_to_realSC.tolist(),
                   "cos_flipSC_to_realBulk": cos_flipS_to_realBULK.tolist(),
                   "cos_origBulk_to_realSC": cos_origB_to_realSC.tolist(),
                   "cos_realBulk_to_realSC_no_model": cos_realB_to_realSC.tolist()}, f, indent=2)
    print(f"\nsaved → {RESULT}")


if __name__ == "__main__":
    main()
