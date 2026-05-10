"""Q30 / Run 20 — Single-tissue Eraslan prostate (or other tissue).

Focus on ONE tissue at a time. Prostate has the most paired data (4 donors,
45 sn pseudobulks). Skips the cross-tissue transfer problem entirely.

Two configurations:
- TRAIN_FROM_SCRATCH=True : fresh model trained ONLY on this tissue's data
- TRAIN_FROM_SCRATCH=False: fine-tune Run-17 (multi-tissue) on this tissue

LOO across the paired donors. Test 2 NN evaluation as before.
Pool: all 45 sn pseudobulks for the tissue. Random top-1 ≈ 12/45 = 27% for prostate.
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

PAIRED_NPZ = "/Users/rls/ecs271/data/sc/eraslan/eraslan_paired_plus_gtex.npz"
OUT_DIR = ROOT / "analysis" / "results" / "q20_disentangled"

TISSUE = os.environ.get("Q30_TISSUE", "prostate")
EPOCHS = int(os.environ.get("Q30_EPOCHS", "300"))
LR = float(os.environ.get("Q30_LR", "1e-3"))
LAM_PAIRED = 5.0
LAM_LEAK = 0.3
LAM_CYCLE = 0.3
LAM_SUP = 1.0
LAM_DONOR_ID = 5.0
LAM_RECON_EXTRA = 0.5
BETA_BIO = 1e-3


def _cosine(a, b):
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    print(f"[Q30/Run20 single-tissue] device={DEVICE}  tissue={TISSUE}  epochs={EPOCHS}")
    d = np.load(PAIRED_NPZ, allow_pickle=True)
    bulk_x = d["bulk_x"].astype(np.float32)
    bulk_donor = np.asarray(d["bulk_donor"], dtype=str)
    bulk_tissue = np.asarray(d["bulk_tissue"], dtype=str)
    bulk_is_eraslan = np.asarray(d["bulk_is_eraslan"], dtype=bool)
    sn_x = d["sn_x"].astype(np.float32)
    sn_donor = np.asarray(d["sn_donor"], dtype=str)
    sn_tissue = np.asarray(d["sn_tissue"], dtype=str)

    # Restrict everything to chosen tissue
    bulk_keep = bulk_tissue == TISSUE
    sn_keep = sn_tissue == TISSUE
    bulk_x = bulk_x[bulk_keep]; bulk_donor = bulk_donor[bulk_keep]
    bulk_is_eraslan = bulk_is_eraslan[bulk_keep]; bulk_tissue = bulk_tissue[bulk_keep]
    sn_x = sn_x[sn_keep]; sn_donor = sn_donor[sn_keep]; sn_tissue = sn_tissue[sn_keep]
    print(f"  {TISSUE} bulk: {bulk_x.shape}  ({bulk_is_eraslan.sum()} Eraslan, "
          f"{(~bulk_is_eraslan).sum()} extra GTEx)")
    print(f"  {TISSUE} sn:   {sn_x.shape}")

    paired_donors = sorted(set(bulk_donor[bulk_is_eraslan]) & set(sn_donor))
    print(f"  paired donors in this tissue: {paired_donors}")
    if len(paired_donors) < 2:
        print("  not enough paired donors for LOO"); return

    fold_results = []
    for fold, held_dn in enumerate(paired_donors):
        # Hold out this donor entirely (their bulk + all their sn pseudobulks)
        held_bulk_i = np.where((bulk_donor == held_dn) & bulk_is_eraslan)[0]
        held_bulk_i = int(held_bulk_i[0])
        held_sn_rows = set(np.where(sn_donor == held_dn)[0])

        train_bulk_mask = np.arange(len(bulk_x)) != held_bulk_i
        train_sn_mask = np.array([i not in held_sn_rows for i in range(len(sn_x))])
        Xb_all = bulk_x[train_bulk_mask]
        donor_b_all = bulk_donor[train_bulk_mask]
        is_eraslan_all = bulk_is_eraslan[train_bulk_mask]
        Xs_tr = sn_x[train_sn_mask]; donor_s_tr = sn_donor[train_sn_mask]

        # Build paired pairs: for each Eraslan-paired bulk donor, all their sn rows
        sn_by_donor = {}
        for j, dn in enumerate(donor_s_tr):
            sn_by_donor.setdefault(dn, []).append(j)
        paired_pairs = []
        for ii in np.where(is_eraslan_all)[0]:
            dn = donor_b_all[ii]
            for j in sn_by_donor.get(dn, []):
                paired_pairs.append((ii, j))
        if not paired_pairs:
            print(f"  fold {fold+1}: no paired pairs"); continue

        if fold == 0:
            print(f"\n  fold 1: paired pairs = {len(paired_pairs)}, total bulk = {len(Xb_all)}")

        bulk_pair_idx = np.array([p[0] for p in paired_pairs])
        sn_pair_idx = np.array([p[1] for p in paired_pairs])
        Xb_pair = torch.from_numpy(Xb_all[bulk_pair_idx]).to(DEVICE)
        Xs_pair = torch.from_numpy(Xs_tr[sn_pair_idx]).to(DEVICE)
        # Single tissue: all tissue indices = 0
        m_pair_tis_b = torch.zeros(Xb_pair.size(0), device=DEVICE)
        m_pair_tis_s = torch.zeros(Xs_pair.size(0), device=DEVICE)
        m_b_pair_mod = torch.zeros(Xb_pair.size(0), device=DEVICE)
        m_s_pair_mod = torch.ones(Xs_pair.size(0), device=DEVICE)

        extra_bulk_idx = np.where(~is_eraslan_all)[0]
        Xb_extra_full = torch.from_numpy(Xb_all[extra_bulk_idx]).to(DEVICE)

        model = DecomposedVAE(input_dim=bulk_x.shape[1], z_meta_dim=2, z_bio_dim=50).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)

        bulk_z0_target = -3.0
        sc_z0_target = 3.0
        rng = np.random.default_rng(SEED + fold)
        batch_size_extra = min(256, Xb_extra_full.size(0))

        for ep in range(1, EPOCHS + 1):
            x_hat_b, mu_m_b, lv_m_b, mu_b_b, lv_b_b, z_m_b, z_b_b = model(Xb_pair)
            x_hat_s, mu_m_s, lv_m_s, mu_b_s, lv_b_s, z_m_s, z_b_s = model(Xs_pair)
            extra_idx = torch.from_numpy(
                rng.choice(Xb_extra_full.size(0), batch_size_extra, replace=False)
            ).to(DEVICE)
            Xb_extra = Xb_extra_full[extra_idx]
            x_hat_e, mu_m_e, lv_m_e, mu_b_e, lv_b_e, z_m_e, z_b_e = model(Xb_extra)

            recon = (nn.functional.mse_loss(x_hat_b, Xb_pair)
                     + nn.functional.mse_loss(x_hat_s, Xs_pair)
                     + LAM_RECON_EXTRA * nn.functional.mse_loss(x_hat_e, Xb_extra))
            kl_b = (kl_with_free_bits(mu_b_b, lv_b_b, 0.5)
                    + kl_with_free_bits(mu_b_s, lv_b_s, 0.5)
                    + kl_with_free_bits(mu_b_e, lv_b_e, 0.5)) / Xb_pair.size(0)
            pred_b = model.head_mod(z_m_b[:, 0:1]).squeeze(-1)
            pred_s = model.head_mod(z_m_s[:, 0:1]).squeeze(-1)
            pred_e = model.head_mod(z_m_e[:, 0:1]).squeeze(-1)
            sup_mod = (nn.functional.binary_cross_entropy_with_logits(pred_b, m_b_pair_mod)
                       + nn.functional.binary_cross_entropy_with_logits(pred_s, m_s_pair_mod)
                       + nn.functional.binary_cross_entropy_with_logits(pred_e, torch.zeros(pred_e.size(0), device=DEVICE)))

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

            loss = (recon + BETA_BIO * kl_b + LAM_SUP * sup_mod
                    + LAM_PAIRED * (paired_b2s + paired_s2b)
                    + LAM_CYCLE * cyc + LAM_LEAK * leak
                    + LAM_DONOR_ID * donor_id)
            opt.zero_grad(); loss.backward(); opt.step()

            if fold == 0 and (ep % 50 == 0 or ep == 1):
                print(f"    ep{ep:3d}  loss={loss.item():.3f}  recon={recon.item():.3f}  "
                      f"paired_b2s={paired_b2s.item():.3f}  donor_id={donor_id.item():.4f}",
                      flush=True)

        # Test on held-out donor
        model.eval()
        with torch.no_grad():
            x_held_b = torch.from_numpy(bulk_x[held_bulk_i:held_bulk_i+1]).to(DEVICE)
            mu_m_h, _, mu_b_h, _ = model.encode(x_held_b)
            z_m_flip = mu_m_h.clone(); z_m_flip[:, 0] = sc_z0_target
            x_hat_sc = model.decode(z_m_flip, mu_b_h).cpu().numpy()[0]

        # Pool: ALL sn samples for this tissue (including held-out donor's)
        cos_to_pool = np.array([_cosine(x_hat_sc, sn_x[i]) for i in range(len(sn_x))])
        order = np.argsort(-cos_to_pool)
        ordered_donors = sn_donor[order]
        rank_first_same = next((i for i, dn in enumerate(ordered_donors) if dn == held_dn), None)
        rank_first_same_1based = (rank_first_same + 1) if rank_first_same is not None else None
        n_same_in_pool = int((sn_donor == held_dn).sum())
        same_donor_ranks = [i + 1 for i, dn in enumerate(ordered_donors) if dn == held_dn]
        topK = {K: float(np.mean(ordered_donors[:K] == held_dn)) for K in [1, 3, 5, 10]}

        print(f"  fold {fold+1}/{len(paired_donors)}: held={held_dn}; "
              f"first-same-rank={rank_first_same_1based}/{len(sn_x)}; "
              f"top-1={topK[1]:.2f}; top-3={topK[3]:.2f}; top-10={topK[10]:.2f}; "
              f"mean rank={np.mean(same_donor_ranks):.2f}", flush=True)

        fold_results.append({
            "held_donor": held_dn,
            "n_pool": int(len(sn_x)), "n_same_donor_in_pool": n_same_in_pool,
            "rank_first_same": rank_first_same_1based,
            "mean_rank_same_donor": float(np.mean(same_donor_ranks)),
            "topK_purity": topK,
            "n_paired_train_pairs": len(paired_pairs),
            "n_extra_bulk": int((~is_eraslan_all).sum()),
        })

    if not fold_results:
        return
    arr_top1 = np.array([f["topK_purity"][1] for f in fold_results])
    arr_top3 = np.array([f["topK_purity"][3] for f in fold_results])
    arr_top5 = np.array([f["topK_purity"][5] for f in fold_results])
    arr_top10 = np.array([f["topK_purity"][10] for f in fold_results])
    arr_rank1 = np.array([f["rank_first_same"] for f in fold_results])
    arr_n_same = np.array([f["n_same_donor_in_pool"] for f in fold_results])
    arr_n_pool = np.array([f["n_pool"] for f in fold_results])
    expected_top1_random = float(np.mean(arr_n_same / arr_n_pool))
    expected_rank1_random = float(np.mean((arr_n_pool - arr_n_same + 1) / (arr_n_same + 1)))
    print(f"\n=== Run 20 ({TISSUE}) LOO across {len(fold_results)} donors ===")
    print(f"  mean top-1: {arr_top1.mean():.3f}  (random ~{expected_top1_random:.3f})")
    print(f"  mean top-3: {arr_top3.mean():.3f}")
    print(f"  mean top-5: {arr_top5.mean():.3f}")
    print(f"  mean top-10: {arr_top10.mean():.3f}")
    print(f"  mean rank-of-first-same: {arr_rank1.mean():.2f}  (random ~{expected_rank1_random:.2f})")

    out = {
        "config": {"tissue": TISSUE, "epochs": EPOCHS, "lr": LR},
        "fold_results": fold_results,
        "summary": {
            "mean_top1": float(arr_top1.mean()),
            "mean_top3": float(arr_top3.mean()),
            "mean_top5": float(arr_top5.mean()),
            "mean_top10": float(arr_top10.mean()),
            "mean_rank_first_same": float(arr_rank1.mean()),
            "expected_top1_random": expected_top1_random,
            "expected_rank1_random": expected_rank1_random,
        },
    }
    out_path = OUT_DIR / f"q30_run20_single_tissue_{TISSUE}.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"saved → {out_path}")


if __name__ == "__main__":
    main()
