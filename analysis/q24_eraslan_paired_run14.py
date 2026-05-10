"""Q24 — Run 14: train DisentangledVAE from scratch on Eraslan donor-paired
multi-tissue bulk + sn data.

z_meta = [modality (BCE), tissue_index (MSE)]   — 2 dims
z_bio  = 50 dims
Loss = recon + KL_freebits + sup(modality, tissue) + λ_paired·flip MSE
       + cycle losses + HSIC modality-only.

Test 2: per-(donor, tissue) leave-one-out — encode held-out bulk → flip → check
NN among held-out tissue's sn pseudobulks across all donors.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.disentangled_vae import (
    DisentangledVAE, DisentangledConfig, kl_with_free_bits, hsic_penalty,
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

PAIRED_NPZ = "/Users/rls/ecs271/data/sc/eraslan/eraslan_paired.npz"
OUT_DIR = ROOT / "analysis" / "results" / "q20_disentangled"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 200
BATCH = 32
LR = 1e-3
LAM_PAIRED = 5.0
LAM_LEAK = 0.3
LAM_CYCLE = 0.3
BETA_BIO = 1e-3
LAM_SUP = 1.0


def _cosine(a, b):
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    print(f"[Q24/Run14] device={DEVICE}")
    d = np.load(PAIRED_NPZ, allow_pickle=True)
    bulk_x = d["bulk_x"].astype(np.float32)
    bulk_donor = np.asarray(d["bulk_donor"], dtype=str)
    bulk_tissue = np.asarray(d["bulk_tissue"], dtype=str)
    sn_x = d["sn_x"].astype(np.float32)
    sn_donor = np.asarray(d["sn_donor"], dtype=str)
    sn_tissue = np.asarray(d["sn_tissue"], dtype=str)
    sn_celltype = np.asarray(d["sn_celltype"], dtype=str)
    print(f"  bulk: {bulk_x.shape}  sn: {sn_x.shape}")
    tissues = sorted(set(np.concatenate([bulk_tissue, sn_tissue])))
    tissue_to_idx = {t: i for i, t in enumerate(tissues)}
    print(f"  tissues ({len(tissues)}): {tissues}")
    donors = sorted(set(np.concatenate([bulk_donor, sn_donor])))
    print(f"  donors ({len(donors)}): {donors}")

    # === LOO test set: hold out 1 (donor, tissue) bulk + matching sn samples ===
    # Build paired (bulk, donor, tissue) → list of sn rows for same (donor, tissue)
    sn_idx_per_dt: dict[tuple[str, str], list[int]] = {}
    for i, (dn, ts) in enumerate(zip(sn_donor, sn_tissue)):
        sn_idx_per_dt.setdefault((dn, ts), []).append(i)
    bulk_idx_per_dt: dict[tuple[str, str], int] = {}
    for i, (dn, ts) in enumerate(zip(bulk_donor, bulk_tissue)):
        bulk_idx_per_dt[(dn, ts)] = i
    paired_dts = [(dn, ts) for (dn, ts) in bulk_idx_per_dt
                  if (dn, ts) in sn_idx_per_dt]
    print(f"  paired (donor, tissue) groups: {len(paired_dts)}")

    fold_results = []
    for fold, (held_dn, held_ts) in enumerate(paired_dts):
        # Held-out: this (donor, tissue) bulk + all sn rows for it
        held_bulk_i = bulk_idx_per_dt[(held_dn, held_ts)]
        held_sn_rows = set(sn_idx_per_dt[(held_dn, held_ts)])

        train_bulk_mask = np.ones(len(bulk_x), dtype=bool); train_bulk_mask[held_bulk_i] = False
        train_sn_mask = np.array([i not in held_sn_rows for i in range(len(sn_x))])
        Xb_tr = bulk_x[train_bulk_mask]; donor_b_tr = bulk_donor[train_bulk_mask]; tissue_b_tr = bulk_tissue[train_bulk_mask]
        Xs_tr = sn_x[train_sn_mask]; donor_s_tr = sn_donor[train_sn_mask]; tissue_s_tr = sn_tissue[train_sn_mask]
        # Build paired indices: for each train bulk i, find all train sn rows with same (donor, tissue)
        sn_by_dt_tr = {}
        for j, (dn, ts) in enumerate(zip(donor_s_tr, tissue_s_tr)):
            sn_by_dt_tr.setdefault((dn, ts), []).append(j)
        paired_pairs = []  # list of (bulk_idx_tr, sn_idx_tr)
        for i, (dn, ts) in enumerate(zip(donor_b_tr, tissue_b_tr)):
            for j in sn_by_dt_tr.get((dn, ts), []):
                paired_pairs.append((i, j))
        if not paired_pairs:
            print(f"  fold {fold+1}: no paired pairs available, skipping {held_dn}/{held_ts}")
            continue

        if fold == 0:
            print(f"\n  fold 1: paired pairs in train = {len(paired_pairs)}")

        # Make tensors
        bulk_pair_idx = np.array([p[0] for p in paired_pairs])
        sn_pair_idx = np.array([p[1] for p in paired_pairs])
        Xb_pair = torch.from_numpy(Xb_tr[bulk_pair_idx]).to(DEVICE)
        Xs_pair = torch.from_numpy(Xs_tr[sn_pair_idx]).to(DEVICE)
        m_b_pair_mod = torch.zeros(Xb_pair.size(0), device=DEVICE)
        m_s_pair_mod = torch.ones(Xs_pair.size(0), device=DEVICE)
        m_b_pair_tis = torch.tensor([tissue_to_idx[t] for t in tissue_b_tr[bulk_pair_idx]],
                                    device=DEVICE, dtype=torch.float32) / max(len(tissues) - 1, 1)
        m_s_pair_tis = torch.tensor([tissue_to_idx[t] for t in tissue_s_tr[sn_pair_idx]],
                                    device=DEVICE, dtype=torch.float32) / max(len(tissues) - 1, 1)

        # New model fresh
        cfg = DisentangledConfig(
            input_dim=bulk_x.shape[1],
            z_bio_dim=50,
            meta_fields=["modality", "tissue"],
            meta_dims=[1, 1],
            meta_kinds=["bce", "mse"],
        )
        model = DisentangledVAE(cfg).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)

        # Train (full-batch — small enough)
        model.train()
        ones = torch.ones(Xb_pair.size(0), device=DEVICE)
        zeros = torch.zeros(Xs_pair.size(0), device=DEVICE)

        bulk_z0_target = -3.0
        sc_z0_target = 3.0
        for ep in range(1, EPOCHS + 1):
            x_hat_b, mu_m_b, lv_m_b, mu_b_b, lv_b_b, z_m_b, z_b_b = model(Xb_pair)
            x_hat_s, mu_m_s, lv_m_s, mu_b_s, lv_b_s, z_m_s, z_b_s = model(Xs_pair)
            recon = nn.functional.mse_loss(x_hat_b, Xb_pair) + nn.functional.mse_loss(x_hat_s, Xs_pair)
            kl_b = (kl_with_free_bits(mu_b_b, lv_b_b, 0.5)
                    + kl_with_free_bits(mu_b_s, lv_b_s, 0.5)) / Xb_pair.size(0)
            # Modality head supervision
            pred_b = model.heads[0](z_m_b[:, 0:1]).squeeze(-1)
            pred_s = model.heads[0](z_m_s[:, 0:1]).squeeze(-1)
            sup_mod = (nn.functional.binary_cross_entropy_with_logits(pred_b, m_b_pair_mod)
                       + nn.functional.binary_cross_entropy_with_logits(pred_s, m_s_pair_mod))
            # Tissue head supervision (continuous index)
            pred_t_b = model.heads[1](z_m_b[:, 1:2]).squeeze(-1)
            pred_t_s = model.heads[1](z_m_s[:, 1:2]).squeeze(-1)
            sup_tis = (nn.functional.mse_loss(pred_t_b, m_b_pair_tis)
                       + nn.functional.mse_loss(pred_t_s, m_s_pair_tis))

            # Paired flip MSE
            z_m_flip_b = z_m_b.clone(); z_m_flip_b[:, 0] = sc_z0_target
            x_flip_b2s = model.decode(z_m_flip_b, z_b_b)
            paired_b2s = nn.functional.mse_loss(x_flip_b2s, Xs_pair)
            z_m_flip_s = z_m_s.clone(); z_m_flip_s[:, 0] = bulk_z0_target
            x_flip_s2b = model.decode(z_m_flip_s, z_b_s)
            paired_s2b = nn.functional.mse_loss(x_flip_s2b, Xb_pair)

            # Cycle: re-encode flipped, z_bio should match
            mu_m_re_b, _, mu_b_re_b, _ = model.encode(x_flip_b2s)
            mu_m_re_s, _, mu_b_re_s, _ = model.encode(x_flip_s2b)
            cyc = (nn.functional.mse_loss(mu_b_re_b, mu_b_b.detach())
                   + nn.functional.mse_loss(mu_b_re_s, mu_b_s.detach()))

            # HSIC modality-only
            mod_col = torch.cat([torch.full((Xb_pair.size(0),1), 0.0, device=DEVICE),
                                 torch.full((Xs_pair.size(0),1), 1.0, device=DEVICE)])
            z_b_combined = torch.cat([z_b_b, z_b_s], dim=0)
            leak = hsic_penalty(z_b_combined, mod_col)

            loss = (recon + BETA_BIO * kl_b + LAM_SUP * (sup_mod + sup_tis)
                    + LAM_PAIRED * (paired_b2s + paired_s2b)
                    + LAM_CYCLE * cyc + LAM_LEAK * leak)
            opt.zero_grad(); loss.backward(); opt.step()

            if fold == 0 and (ep % 25 == 0 or ep == 1):
                print(f"    ep{ep:3d}  loss={loss.item():.3f}  recon={recon.item():.3f}  "
                      f"paired_b2s={paired_b2s.item():.3f}  paired_s2b={paired_s2b.item():.3f}",
                      flush=True)

        # Test: encode held-out bulk → flip → NN among ALL sn samples for this tissue
        model.eval()
        with torch.no_grad():
            x_held_b = torch.from_numpy(bulk_x[held_bulk_i:held_bulk_i+1]).to(DEVICE)
            mu_m_h, _, mu_b_h, _ = model.encode(x_held_b)
            z_m_flip = mu_m_h.clone(); z_m_flip[:, 0] = sc_z0_target
            x_hat_sc = model.decode(z_m_flip, mu_b_h).cpu().numpy()[0]

        # Pool: all sn rows belonging to tissue `held_ts` (across donors)
        pool_idx = np.where(sn_tissue == held_ts)[0]
        cos_to_pool = np.array([_cosine(x_hat_sc, sn_x[i]) for i in pool_idx])
        order = np.argsort(-cos_to_pool)
        ordered_donors = sn_donor[pool_idx][order]
        rank_first_same = next((i for i, dn in enumerate(ordered_donors) if dn == held_dn), None)
        rank_first_same_1based = (rank_first_same + 1) if rank_first_same is not None else None
        n_same_in_pool = int((sn_donor[pool_idx] == held_dn).sum())
        n_pool = len(pool_idx)
        same_donor_ranks = [i + 1 for i, dn in enumerate(ordered_donors) if dn == held_dn]
        topK_purity = {}
        for K in [1, 3, 5, 10]:
            topK_purity[K] = float(np.mean(ordered_donors[:K] == held_dn)) if K <= len(ordered_donors) else None

        print(f"  fold {fold+1}/{len(paired_dts)}: held=({held_dn}, {held_ts}); "
              f"pool={n_pool} sn samples ({n_same_in_pool} same-donor); "
              f"first-same-rank={rank_first_same_1based}; "
              f"top-1 purity={topK_purity[1]:.2f}; mean rank={np.mean(same_donor_ranks):.2f}", flush=True)

        fold_results.append({
            "held_donor": held_dn,
            "held_tissue": held_ts,
            "n_pool": n_pool,
            "n_same_donor_in_pool": n_same_in_pool,
            "rank_first_same": rank_first_same_1based,
            "mean_rank_same_donor": float(np.mean(same_donor_ranks)),
            "topK_purity": topK_purity,
            "n_paired_train_pairs": len(paired_pairs),
        })

    # Aggregate
    arr_top1 = np.array([f["topK_purity"][1] for f in fold_results])
    arr_top3 = np.array([f["topK_purity"][3] for f in fold_results])
    arr_top5 = np.array([f["topK_purity"][5] for f in fold_results])
    arr_rank1 = np.array([f["rank_first_same"] for f in fold_results])
    arr_mean_rank = np.array([f["mean_rank_same_donor"] for f in fold_results])
    arr_n_same = np.array([f["n_same_donor_in_pool"] for f in fold_results])
    arr_n_pool = np.array([f["n_pool"] for f in fold_results])
    expected_top1_random = float(np.mean(arr_n_same / arr_n_pool))
    expected_rank1_random = float(np.mean((arr_n_pool - arr_n_same + 1) / (arr_n_same + 1)))
    print(f"\n=== LOO summary across {len(fold_results)} (donor, tissue) folds ===")
    print(f"  mean top-1 same-donor purity:  {arr_top1.mean():.3f}  (random ~{expected_top1_random:.3f})")
    print(f"  mean top-3:                    {arr_top3.mean():.3f}")
    print(f"  mean top-5:                    {arr_top5.mean():.3f}")
    print(f"  mean rank-of-first-same:       {arr_rank1.mean():.2f}  (random ~{expected_rank1_random:.2f})")
    print(f"  mean rank-of-all-same-donor:   {arr_mean_rank.mean():.2f}")

    out = {
        "n_folds": len(fold_results),
        "tissues": tissues,
        "donors": donors,
        "config": {
            "epochs": EPOCHS, "lr": LR, "lam_paired": LAM_PAIRED,
            "lam_leak": LAM_LEAK, "lam_cycle": LAM_CYCLE,
        },
        "fold_results": fold_results,
        "summary": {
            "mean_top1": float(arr_top1.mean()),
            "mean_top3": float(arr_top3.mean()),
            "mean_top5": float(arr_top5.mean()),
            "mean_rank_first_same": float(arr_rank1.mean()),
            "mean_rank_all_same": float(arr_mean_rank.mean()),
            "expected_top1_random": expected_top1_random,
            "expected_rank1_random": expected_rank1_random,
        },
    }
    with (OUT_DIR / "q24_eraslan_run14.json").open("w") as f:
        json.dump(out, f, indent=2)
    print(f"saved → {OUT_DIR / 'q24_eraslan_run14.json'}")


if __name__ == "__main__":
    main()
