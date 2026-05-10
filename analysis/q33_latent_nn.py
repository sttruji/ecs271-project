"""Q33 — Latent-space NN test (skip the decode step).

Key insight from Q32: raw cosine and PCA-50 hit 50%+ top-1; our VAE flip
hits 29%. The flip + decode is destroying donor signal.

This script: train Run-17 architecture (decomposed VAE), then for the test,
encode held-out bulk into z_bio, encode all sn samples into z_bio, and do
NN in z_bio space directly. NO flip, NO decode.

If z_bio NN >= raw cosine 0.50, the VAE encoder is at least preserving
donor signal. If it's also at chance, the encoder is the problem.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.disentangled_vae import (
    DisentangledConfig, kl_with_free_bits, hsic_penalty,
)
from analysis.q27_eraslan_run17_decompose import DecomposedVAE

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

PAIRED_NPZ = "/Users/rls/ecs271/data/sc/eraslan/eraslan_paired.npz"
OUT_DIR = ROOT / "analysis" / "results" / "q20_disentangled"

EPOCHS = 200
LR = 1e-3
LAM_PAIRED = 5.0
LAM_LEAK = 0.3
LAM_CYCLE = 0.3
LAM_SUP = 1.0
LAM_DONOR_ID = 5.0
BETA_BIO = 1e-3


def _cosine(a, b):
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    print(f"[Q33 latent-NN test] device={DEVICE}")
    d = np.load(PAIRED_NPZ, allow_pickle=True)
    bulk_x = d["bulk_x"].astype(np.float32)
    bulk_donor = np.asarray(d["bulk_donor"], dtype=str)
    bulk_tissue = np.asarray(d["bulk_tissue"], dtype=str)
    sn_x = d["sn_x"].astype(np.float32)
    sn_donor = np.asarray(d["sn_donor"], dtype=str)
    sn_tissue = np.asarray(d["sn_tissue"], dtype=str)
    print(f"  bulk: {bulk_x.shape}  sn: {sn_x.shape}")
    tissues = sorted(set(np.concatenate([bulk_tissue, sn_tissue])))
    tissue_to_idx = {t: i for i, t in enumerate(tissues)}

    sn_idx_per_dt = {}
    for i, (dn, ts) in enumerate(zip(sn_donor, sn_tissue)):
        sn_idx_per_dt.setdefault((dn, ts), []).append(i)
    bulk_idx_per_dt = {(dn, ts): i for i, (dn, ts) in enumerate(zip(bulk_donor, bulk_tissue))}
    paired_dts = [(dn, ts) for (dn, ts) in bulk_idx_per_dt if (dn, ts) in sn_idx_per_dt]

    fold_results = []
    for fold, (held_dn, held_ts) in enumerate(paired_dts):
        held_bulk_i = bulk_idx_per_dt[(held_dn, held_ts)]
        held_sn_rows = set(sn_idx_per_dt[(held_dn, held_ts)])

        train_bulk_mask = np.ones(len(bulk_x), dtype=bool); train_bulk_mask[held_bulk_i] = False
        train_sn_mask = np.array([i not in held_sn_rows for i in range(len(sn_x))])
        Xb_tr = bulk_x[train_bulk_mask]; donor_b_tr = bulk_donor[train_bulk_mask]; tissue_b_tr = bulk_tissue[train_bulk_mask]
        Xs_tr = sn_x[train_sn_mask]; donor_s_tr = sn_donor[train_sn_mask]; tissue_s_tr = sn_tissue[train_sn_mask]
        sn_by_dt_tr = {}
        for j, (dn, ts) in enumerate(zip(donor_s_tr, tissue_s_tr)):
            sn_by_dt_tr.setdefault((dn, ts), []).append(j)
        paired_pairs = []
        for i, (dn, ts) in enumerate(zip(donor_b_tr, tissue_b_tr)):
            for j in sn_by_dt_tr.get((dn, ts), []):
                paired_pairs.append((i, j))
        if not paired_pairs: continue

        bulk_pair_idx = np.array([p[0] for p in paired_pairs])
        sn_pair_idx = np.array([p[1] for p in paired_pairs])
        Xb_pair = torch.from_numpy(Xb_tr[bulk_pair_idx]).to(DEVICE)
        Xs_pair = torch.from_numpy(Xs_tr[sn_pair_idx]).to(DEVICE)
        m_b_pair_tis = torch.tensor([tissue_to_idx[t] for t in tissue_b_tr[bulk_pair_idx]],
                                    device=DEVICE, dtype=torch.float32) / max(len(tissues) - 1, 1)
        m_s_pair_tis = torch.tensor([tissue_to_idx[t] for t in tissue_s_tr[sn_pair_idx]],
                                    device=DEVICE, dtype=torch.float32) / max(len(tissues) - 1, 1)
        m_b_pair_mod = torch.zeros(Xb_pair.size(0), device=DEVICE)
        m_s_pair_mod = torch.ones(Xs_pair.size(0), device=DEVICE)

        model = DecomposedVAE(input_dim=bulk_x.shape[1], z_meta_dim=2, z_bio_dim=50).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)

        bulk_z0_target = -3.0; sc_z0_target = 3.0
        for ep in range(1, EPOCHS + 1):
            x_hat_b, mu_m_b, lv_m_b, mu_b_b, lv_b_b, z_m_b, z_b_b = model(Xb_pair)
            x_hat_s, mu_m_s, lv_m_s, mu_b_s, lv_b_s, z_m_s, z_b_s = model(Xs_pair)
            recon = nn.functional.mse_loss(x_hat_b, Xb_pair) + nn.functional.mse_loss(x_hat_s, Xs_pair)
            kl_b = (kl_with_free_bits(mu_b_b, lv_b_b, 0.5)
                    + kl_with_free_bits(mu_b_s, lv_b_s, 0.5)) / Xb_pair.size(0)
            pred_b = model.head_mod(z_m_b[:, 0:1]).squeeze(-1)
            pred_s = model.head_mod(z_m_s[:, 0:1]).squeeze(-1)
            sup_mod = (nn.functional.binary_cross_entropy_with_logits(pred_b, m_b_pair_mod)
                       + nn.functional.binary_cross_entropy_with_logits(pred_s, m_s_pair_mod))
            pred_t_b = model.head_tis(z_m_b[:, 1:2]).squeeze(-1)
            pred_t_s = model.head_tis(z_m_s[:, 1:2]).squeeze(-1)
            sup_tis = (nn.functional.mse_loss(pred_t_b, m_b_pair_tis)
                       + nn.functional.mse_loss(pred_t_s, m_s_pair_tis))
            z_m_flip_b = z_m_b.clone(); z_m_flip_b[:, 0] = sc_z0_target
            x_flip_b2s = model.decode(z_m_flip_b, z_b_b)
            paired_b2s = nn.functional.mse_loss(x_flip_b2s, Xs_pair)
            z_m_flip_s = z_m_s.clone(); z_m_flip_s[:, 0] = bulk_z0_target
            x_flip_s2b = model.decode(z_m_flip_s, z_b_s)
            paired_s2b = nn.functional.mse_loss(x_flip_s2b, Xb_pair)
            mu_m_re_b, _, mu_b_re_b, _ = model.encode(x_flip_b2s)
            mu_m_re_s, _, mu_b_re_s, _ = model.encode(x_flip_s2b)
            cyc = (nn.functional.mse_loss(mu_b_re_b, mu_b_b.detach())
                   + nn.functional.mse_loss(mu_b_re_s, mu_b_s.detach()))
            mod_col = torch.cat([torch.full((Xb_pair.size(0),1), 0.0, device=DEVICE),
                                 torch.full((Xs_pair.size(0),1), 1.0, device=DEVICE)])
            z_b_combined = torch.cat([z_b_b, z_b_s], dim=0)
            leak = hsic_penalty(z_b_combined, mod_col)
            donor_id = nn.functional.mse_loss(mu_b_b, mu_b_s)
            loss = (recon + BETA_BIO * kl_b + LAM_SUP * (sup_mod + sup_tis)
                    + LAM_PAIRED * (paired_b2s + paired_s2b)
                    + LAM_CYCLE * cyc + LAM_LEAK * leak + LAM_DONOR_ID * donor_id)
            opt.zero_grad(); loss.backward(); opt.step()

        # Test in LATENT space (no flip, no decode)
        model.eval()
        with torch.no_grad():
            x_held_b = torch.from_numpy(bulk_x[held_bulk_i:held_bulk_i+1]).to(DEVICE)
            _, _, mu_b_h, _ = model.encode(x_held_b)
            z_held = mu_b_h.cpu().numpy()[0]
            # Encode ALL sn samples
            sn_full = torch.from_numpy(sn_x).to(DEVICE)
            _, _, mu_b_sn_full, _ = model.encode(sn_full)
            z_sn_all = mu_b_sn_full.cpu().numpy()

        # Restrict pool to held tissue (matches Q24 test)
        pool_idx = np.where(sn_tissue == held_ts)[0]
        z_sn_pool = z_sn_all[pool_idx]
        cos_to_pool = np.array([_cosine(z_held, z_sn_pool[i]) for i in range(len(z_sn_pool))])
        order = np.argsort(-cos_to_pool)
        ordered_donors = sn_donor[pool_idx][order]
        rank_first_same = next((i for i, dn in enumerate(ordered_donors) if dn == held_dn), None)
        rank_first_same_1based = (rank_first_same + 1) if rank_first_same is not None else None
        n_same_in_pool = int((sn_donor[pool_idx] == held_dn).sum())
        topK = {K: float(np.mean(ordered_donors[:K] == held_dn)) for K in [1, 3, 5, 10]}
        print(f"  fold {fold+1}/{len(paired_dts)}: held=({held_dn}, {held_ts}); LATENT NN: "
              f"first-same-rank={rank_first_same_1based}; top-1={topK[1]:.2f}; top-3={topK[3]:.2f}",
              flush=True)

        fold_results.append({
            "held_donor": held_dn, "held_tissue": held_ts,
            "n_pool": len(pool_idx), "n_same_donor_in_pool": n_same_in_pool,
            "rank_first_same": rank_first_same_1based,
            "topK_purity": topK,
        })

    arr_top1 = np.array([f["topK_purity"][1] for f in fold_results])
    arr_top3 = np.array([f["topK_purity"][3] for f in fold_results])
    arr_top5 = np.array([f["topK_purity"][5] for f in fold_results])
    arr_top10 = np.array([f["topK_purity"][10] for f in fold_results])
    arr_rank1 = np.array([f["rank_first_same"] for f in fold_results])
    arr_n_same = np.array([f["n_same_donor_in_pool"] for f in fold_results])
    arr_n_pool = np.array([f["n_pool"] for f in fold_results])
    expected_top1_random = float(np.mean(arr_n_same / arr_n_pool))
    print(f"\n=== Q33 LATENT NN summary across {len(fold_results)} folds ===")
    print(f"  mean top-1: {arr_top1.mean():.3f}  (random ~{expected_top1_random:.3f})")
    print(f"  mean top-3: {arr_top3.mean():.3f}")
    print(f"  mean top-5: {arr_top5.mean():.3f}")
    print(f"  mean top-10: {arr_top10.mean():.3f}")
    print(f"  mean rank-first-same: {arr_rank1.mean():.2f}")

    out = {
        "config": {"epochs": EPOCHS, "method": "encode bulk + sn into z_bio, NN in z_bio space (NO flip)"},
        "fold_results": fold_results,
        "summary": {
            "mean_top1": float(arr_top1.mean()),
            "mean_top3": float(arr_top3.mean()),
            "mean_top5": float(arr_top5.mean()),
            "mean_top10": float(arr_top10.mean()),
            "mean_rank_first_same": float(arr_rank1.mean()),
            "expected_top1_random": expected_top1_random,
        },
    }
    with (OUT_DIR / "q33_latent_nn.json").open("w") as f:
        json.dump(out, f, indent=2)
    print(f"saved → {OUT_DIR / 'q33_latent_nn.json'}")


if __name__ == "__main__":
    main()
