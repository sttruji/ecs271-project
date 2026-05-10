"""Q23 — Bootstrap-rich paired-donor LOO fine-tuning of Run 13.

Bootstrap each Test-2 donor's cells into N=10 random 100-cell subsamples,
giving 70 sc pseudobulks total (10 per donor). Pair each with the same donor's
bulk. LOO fine-tuning: hold out 1 donor's 10 sc samples, train on the OTHER
6 donors × 10 = 60 paired samples per fold. Then test:
  - "donor-NN" accuracy: held-out bulk flipped to sc → does NN among 70 sc
    pool belong to the same donor? (random = 10/70 = 14.3%)
  - top-3 / top-10 / mean rank-of-first-same-donor metrics
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.disentangled_vae import (
    DisentangledVAE, DisentangledConfig, kl_with_free_bits, hsic_penalty,
)
from analysis.q20_disentangled_vae import DEVICE, OUT, build_data
from analysis.q21_paired_donor_test2 import load_bulk, _align, _cosine
from pipeline.data import load_shared_genes

PAIRED_DONORS = ["YUALOE", "YUFURL", "YUHERN", "YUHONEY", "YUNANCY", "YUROD", "YUTORY"]
SC_BATCHES = [
    ("/Users/rls/ecs271/data/paired_test2/gse189125_sc_pbmc/GSE189125_5prime_scRNAseq_seqbatchA_counts.txt.gz",
     "/Users/rls/ecs271/data/paired_test2/gse189125_sc_pbmc/GSE189125_5prime_scRNAseq_seqbatchA_metadata.txt.gz",
     "patient ID"),
    ("/Users/rls/ecs271/data/paired_test2/gse189125_sc_pbmc/GSE189125_5prime_scRNAseq_seqbatchB_counts.txt.gz",
     "/Users/rls/ecs271/data/paired_test2/gse189125_sc_pbmc/GSE189125_5prime_scRNAseq_seqbatchB_metadata.txt.gz",
     "patient.ID"),
]
INIT_CKPT = OUT / "q20_run13_melanoma_in_train.pt"
RESULT = OUT / "q23_bootstrap_loo.json"

N_REPS = 10           # bootstrap subsamples per donor
CELLS_PER_REP = 100   # cells per subsample (~smallest donor / 2)
EPOCHS_FT = 80
LR_FT = 1e-4
LAM_PAIRED = 5.0
LAM_RECON = 1.0
LAM_LEAK = 0.3


def build_bootstrap_sc(seed=0):
    """For each Test-2 donor, build N_REPS random CELLS_PER_REP-cell pseudobulks."""
    rng = np.random.default_rng(seed)
    cells_per_donor: dict[str, list[tuple[str, int]]] = {}
    for counts_path, meta_path, donor_col in SC_BATCHES:
        md = pd.read_csv(meta_path, sep="\t").set_index("barcode")
        donor_per_cell = md[donor_col]
        with gzip.open(counts_path, "rt") as f:
            header = f.readline().rstrip().split()
        d_arr = donor_per_cell.reindex(header)
        for col_idx, d in enumerate(d_arr.values):
            if d in PAIRED_DONORS:
                cells_per_donor.setdefault(d, []).append((counts_path, col_idx))
    print(f"  cells per donor: {[(d, len(v)) for d, v in sorted(cells_per_donor.items())]}")

    samples_per_donor_rep: dict[tuple[str, int], list[tuple[str, int]]] = {}
    for d, cells in cells_per_donor.items():
        for r in range(N_REPS):
            n_eff = min(CELLS_PER_REP, len(cells))
            sel = [cells[i] for i in rng.choice(len(cells), n_eff,
                                                replace=(len(cells) < CELLS_PER_REP))]
            samples_per_donor_rep[(d, r)] = sel

    # Stream count files, summing per (donor, rep)
    by_batch: dict[str, dict[tuple[str, int], list[int]]] = {}
    for k, sel in samples_per_donor_rep.items():
        for batch_path, col_idx in sel:
            by_batch.setdefault(batch_path, {}).setdefault(k, []).append(col_idx)

    pseudo: dict[tuple[str, int], np.ndarray] = {}
    gene_names = None
    for counts_path, _, _ in SC_BATCHES:
        if counts_path not in by_batch:
            continue
        groups = by_batch[counts_path]
        keys = sorted(groups.keys())
        cols_per_key = [np.asarray(groups[k], dtype=np.int64) for k in keys]
        all_cols = np.concatenate(cols_per_key)
        key_per_col = np.concatenate([np.full(len(c), i, dtype=np.int64)
                                       for i, c in enumerate(cols_per_key)])
        with gzip.open(counts_path, "rt") as f:
            f.readline()
            gn = []
            rows = []
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) < 2:
                    continue
                gn.append(parts[0])
                vals = np.asarray(parts[1:], dtype=np.float32)
                kept = vals[all_cols]
                row = np.zeros(len(keys), dtype=np.float32)
                np.add.at(row, key_per_col, kept)
                rows.append(row)
            arr = np.stack(rows)
            if gene_names is None:
                gene_names = np.asarray(gn, dtype=str)
            for j, k in enumerate(keys):
                v = arr[:, j].astype(np.float64)
                pseudo[k] = pseudo.get(k, np.zeros_like(v)) + v

    keys_final = sorted(pseudo.keys())
    counts = np.stack([pseudo[k] for k in keys_final])
    lib = counts.sum(axis=1, keepdims=True)
    log_cpm = np.log2(counts / np.where(lib == 0, 1, lib) * 1e6 + 1).astype(np.float32)
    donors = [k[0] for k in keys_final]
    reps = [k[1] for k in keys_final]
    return log_cpm, gene_names, donors, reps


def main():
    print(f"[Q23] init={INIT_CKPT.name} device={DEVICE} reps={N_REPS} cells={CELLS_PER_REP}")
    shared_genes, scaler_mean, scaler_std = load_shared_genes()

    bulk_log, bulk_genes, bulk_donors = load_bulk()
    bulk_aligned, _ = _align(bulk_log, bulk_genes, shared_genes)
    bulk_scaled = ((bulk_aligned - scaler_mean) / scaler_std).astype(np.float32)
    sc_log, sc_genes, sc_donors_per_row, sc_reps_per_row = build_bootstrap_sc()
    sc_aligned, _ = _align(sc_log, sc_genes, shared_genes)
    sc_scaled = ((sc_aligned - scaler_mean) / scaler_std).astype(np.float32)
    print(f"  built {len(sc_donors_per_row)} sc pseudobulks ({N_REPS} per donor)")

    common = sorted(set(bulk_donors) & set(sc_donors_per_row))
    n = len(common)
    print(f"  paired donors: {n} = {common}")
    bulk_idx = [bulk_donors.index(d) for d in common]
    bulk_paired = bulk_scaled[bulk_idx]   # (n, G)

    # sc indices per donor: list of row indices in sc_scaled belonging to donor d
    sc_rows_per_donor = {d: [i for i, dn in enumerate(sc_donors_per_row) if dn == d]
                         for d in common}

    # Precompute background data for class-mean targets
    print("  loading background data for class-mean targets …")
    data = build_data(include_melanoma=True)

    fold_results = []
    for held in range(n):
        train_donors = [d for d in common if d != common[held]]
        held_donor = common[held]
        print(f"\n=== fold {held+1}/{n}: hold out {held_donor} ===")

        # Reload model fresh
        ckpt = torch.load(INIT_CKPT, map_location=DEVICE, weights_only=False)
        cfg = DisentangledConfig(**ckpt["config"]) if isinstance(ckpt["config"], dict) else ckpt["config"]
        model = DisentangledVAE(cfg).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        opt = torch.optim.Adam(model.parameters(), lr=LR_FT)

        # Class-mean encoded targets from background
        model.eval()
        with torch.no_grad():
            t = torch.from_numpy(data["x_gtex"]).to(DEVICE)
            bulk_z0 = model.encode(t)[0][:, 0].mean().item()
            t = torch.from_numpy(data["x_hca"]).to(DEVICE)
            sc_z0 = model.encode(t)[0][:, 0].mean().item()
        print(f"    pretrain class means: bulk={bulk_z0:.2f}, sc={sc_z0:.2f}")

        # Build paired training set: for each train donor, all (bulk_d, sc_d_rep)
        train_X_b = []
        train_X_s = []
        for d in train_donors:
            d_bulk = bulk_paired[common.index(d)]
            for ri in sc_rows_per_donor[d]:
                train_X_b.append(d_bulk)
                train_X_s.append(sc_scaled[ri])
        Xb = torch.from_numpy(np.stack(train_X_b)).to(DEVICE)
        Xs = torch.from_numpy(np.stack(train_X_s)).to(DEVICE)
        print(f"    paired training pairs: {len(train_X_b)}")
        ones = torch.ones(Xb.size(0), device=DEVICE)
        zeros = torch.zeros(Xs.size(0), device=DEVICE)

        for ep in range(1, EPOCHS_FT + 1):
            model.train()
            x_hat_b, mu_m_b, lv_m_b, mu_b_b, lv_b_b, z_m_b, z_b_b = model(Xb)
            x_hat_s, mu_m_s, lv_m_s, mu_b_s, lv_b_s, z_m_s, z_b_s = model(Xs)
            recon = nn.functional.mse_loss(x_hat_b, Xb) + nn.functional.mse_loss(x_hat_s, Xs)
            kl_b = (kl_with_free_bits(mu_b_b, lv_b_b, 0.5)
                    + kl_with_free_bits(mu_b_s, lv_b_s, 0.5)) / Xb.size(0)
            pred_b_logit = model.heads[0](z_m_b[:, 0:1]).squeeze(-1)
            pred_s_logit = model.heads[0](z_m_s[:, 0:1]).squeeze(-1)
            sup = (nn.functional.binary_cross_entropy_with_logits(pred_b_logit, zeros)
                   + nn.functional.binary_cross_entropy_with_logits(pred_s_logit, ones))
            # Paired flip MSE
            z_m_flip_b = z_m_b.clone(); z_m_flip_b[:, 0] = sc_z0
            x_flip_b2s = model.decode(z_m_flip_b, z_b_b)
            paired_b2s = nn.functional.mse_loss(x_flip_b2s, Xs)
            z_m_flip_s = z_m_s.clone(); z_m_flip_s[:, 0] = bulk_z0
            x_flip_s2b = model.decode(z_m_flip_s, z_b_s)
            paired_s2b = nn.functional.mse_loss(x_flip_s2b, Xb)
            # HSIC leak (modality only)
            mod_col = torch.cat([torch.full((Xb.size(0), 1), 0.0, device=DEVICE),
                                 torch.full((Xs.size(0), 1), 1.0, device=DEVICE)])
            z_b_combined = torch.cat([z_b_b, z_b_s], dim=0)
            leak = hsic_penalty(z_b_combined, mod_col)

            loss = (LAM_RECON * recon + 1e-3 * kl_b + 1.0 * sup
                    + LAM_PAIRED * (paired_b2s + paired_s2b)
                    + LAM_LEAK * leak)
            opt.zero_grad(); loss.backward(); opt.step()
            if ep % 20 == 0 or ep == 1:
                print(f"    ep{ep:2d}  loss={loss.item():.3f}  recon={recon.item():.3f}  "
                      f"paired_b2s={paired_b2s.item():.3f}  paired_s2b={paired_s2b.item():.3f}",
                      flush=True)

        # Test: encode held-out donor's bulk → flip → NN among ALL 70 sc samples
        model.eval()
        with torch.no_grad():
            x_held_b = torch.from_numpy(bulk_paired[held:held+1]).to(DEVICE)
            mu_m_h, _, mu_b_h, _ = model.encode(x_held_b)
            z_m_flip = mu_m_h.clone(); z_m_flip[:, 0] = sc_z0
            x_hat_sc = model.decode(z_m_flip, mu_b_h).cpu().numpy()[0]

        cos_to_all = np.array([_cosine(x_hat_sc, sc_scaled[i]) for i in range(len(sc_scaled))])
        order = np.argsort(-cos_to_all)
        order_donors = [sc_donors_per_row[i] for i in order]
        # Rank of the FIRST same-donor sample
        rank_first_same = next(i for i, d in enumerate(order_donors) if d == held_donor) + 1
        # Top-K donor purity: fraction of top-K samples that belong to held-out donor
        topK_purity = {K: float(np.mean([d == held_donor for d in order_donors[:K]]))
                       for K in [1, 5, 10]}
        # Mean rank across all same-donor sc samples
        same_donor_ranks = [i + 1 for i, d in enumerate(order_donors) if d == held_donor]
        mean_rank_same = float(np.mean(same_donor_ranks))

        # Symmetric: pick a random sc rep of held-out donor, encode → flip to bulk → NN among 7 bulks
        # (less interesting but nice for symmetry)
        held_sc_rep = sc_rows_per_donor[held_donor][0]
        with torch.no_grad():
            x_held_s = torch.from_numpy(sc_scaled[held_sc_rep:held_sc_rep+1]).to(DEVICE)
            mu_m_s, _, mu_b_s, _ = model.encode(x_held_s)
            z_m_flip_s = mu_m_s.clone(); z_m_flip_s[:, 0] = bulk_z0
            x_hat_b = model.decode(z_m_flip_s, mu_b_s).cpu().numpy()[0]
        cos_to_bulks = np.array([_cosine(x_hat_b, bulk_paired[j]) for j in range(n)])
        rank_s2b = int(np.where(np.argsort(-cos_to_bulks) == held)[0][0]) + 1

        print(f"    held-out {held_donor}: rank-of-first-same-donor in 70-pool = {rank_first_same}")
        print(f"      top-1 purity: {topK_purity[1]:.2f}, top-5: {topK_purity[5]:.2f}, top-10: {topK_purity[10]:.2f}")
        print(f"      mean rank of all same-donor sc: {mean_rank_same:.2f}")
        print(f"      flipSC→bulk rank (single rep): {rank_s2b}/{n}")

        fold_results.append({
            "held_out": held_donor,
            "rank_first_same_in_70_pool": rank_first_same,
            "top1_purity": topK_purity[1],
            "top5_purity": topK_purity[5],
            "top10_purity": topK_purity[10],
            "mean_rank_same_donor": mean_rank_same,
            "flipSC_to_bulk_rank": rank_s2b,
        })

    # Aggregate
    arr_top1 = np.array([f["top1_purity"] for f in fold_results])
    arr_top5 = np.array([f["top5_purity"] for f in fold_results])
    arr_top10 = np.array([f["top10_purity"] for f in fold_results])
    arr_meanr = np.array([f["mean_rank_same_donor"] for f in fold_results])
    arr_rank1 = np.array([f["rank_first_same_in_70_pool"] for f in fold_results])
    arr_s2b = np.array([f["flipSC_to_bulk_rank"] for f in fold_results])
    print(f"\n=== LOO summary (random chance: top-1=14.3%, top-5=14.3%, top-10=14.3%) ===")
    print(f"  mean top-1 same-donor purity in flipped→sc NN: {arr_top1.mean():.3f}")
    print(f"  mean top-5: {arr_top5.mean():.3f}")
    print(f"  mean top-10: {arr_top10.mean():.3f}")
    print(f"  mean rank of first-same-donor in 70-pool: {arr_rank1.mean():.2f}  (random expected ~6.5)")
    print(f"  mean rank of all same-donor sc: {arr_meanr.mean():.2f}  (random expected ~35.5)")
    print(f"  mean flipSC→bulk rank: {arr_s2b.mean():.2f}/{n}  (random expected ~4)")

    with RESULT.open("w") as f:
        json.dump({
            "init_checkpoint": str(INIT_CKPT.name),
            "n_paired_donors": n,
            "n_reps_per_donor": N_REPS,
            "cells_per_rep": CELLS_PER_REP,
            "epochs_ft": EPOCHS_FT,
            "lr_ft": LR_FT,
            "lam_paired": LAM_PAIRED,
            "fold_results": fold_results,
            "summary": {
                "mean_top1": float(arr_top1.mean()),
                "mean_top5": float(arr_top5.mean()),
                "mean_top10": float(arr_top10.mean()),
                "mean_rank_first_same": float(arr_rank1.mean()),
                "mean_rank_all_same": float(arr_meanr.mean()),
                "mean_flipSC_to_bulk_rank": float(arr_s2b.mean()),
            },
        }, f, indent=2)
    print(f"saved → {RESULT}")


if __name__ == "__main__":
    main()
