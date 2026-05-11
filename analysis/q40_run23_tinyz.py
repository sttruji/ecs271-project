"""Q40 / Run 23 — Tiny-z VAE: z_bio = 5 instead of 50.

CCA-5 hits 0.604; our VAE at z_bio=50 hits 0.333. The most direct lesson is
that 5 dims is enough — and 50 dims gave the supervised losses too much
room to fragment donor signal.

This is Run 17 (decomposed decoder) architecture but with z_bio_dim=5.
Everything else identical. Test on Eraslan paired multi-tissue LOO.

Also includes the latent-NN evaluation (Q33-style) since with 5 dims and
explicit paired alignment, both flip-decode AND latent-NN should work.
"""
from __future__ import annotations

import json
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
Z_BIO_DIM = 5    # ← THE CHANGE


def _cosine(a, b):
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _eval(scores, sd_pool, dn):
    order = np.argsort(-scores)
    ord_d = sd_pool[order]
    rank = next(j for j, d in enumerate(ord_d) if d == dn) + 1
    topK = {K: float(np.mean(ord_d[:K] == dn)) for K in [1, 3, 5, 10]}
    return rank, topK


def main():
    print(f"[Q40/Run23 tiny-z VAE] device={DEVICE}  z_bio_dim={Z_BIO_DIM}")
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
        if not paired_pairs:
            continue
        if fold == 0:
            print(f"\n  fold 1: paired pairs = {len(paired_pairs)}")

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

        model = DecomposedVAE(input_dim=bulk_x.shape[1], z_meta_dim=2, z_bio_dim=Z_BIO_DIM).to(DEVICE)
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
            if fold == 0 and (ep % 50 == 0 or ep == 1):
                print(f"    ep{ep:3d}  loss={loss.item():.3f}  recon={recon.item():.3f}  "
                      f"paired_b2s={paired_b2s.item():.3f}  donor_id={donor_id.item():.4f}",
                      flush=True)

        # Test in BOTH ways: (a) flip+decode, (b) latent-NN
        model.eval()
        with torch.no_grad():
            x_held_b = torch.from_numpy(bulk_x[held_bulk_i:held_bulk_i+1]).to(DEVICE)
            mu_m_h, _, mu_b_h, _ = model.encode(x_held_b)
            z_m_flip = mu_m_h.clone(); z_m_flip[:, 0] = sc_z0_target
            x_hat_sc = model.decode(z_m_flip, mu_b_h).cpu().numpy()[0]
            sn_full = torch.from_numpy(sn_x).to(DEVICE)
            _, _, mu_b_sn_full, _ = model.encode(sn_full)
            z_held = mu_b_h.cpu().numpy()[0]
            z_sn_all = mu_b_sn_full.cpu().numpy()

        pool_idx = np.where(sn_tissue == held_ts)[0]
        # Flip+decode evaluation
        cos_flip = np.array([_cosine(x_hat_sc, sn_x[i]) for i in pool_idx])
        rank_flip, topK_flip = _eval(cos_flip, sn_donor[pool_idx], held_dn)
        # Latent-NN evaluation
        cos_lat = np.array([_cosine(z_held, z_sn_all[i]) for i in pool_idx])
        rank_lat, topK_lat = _eval(cos_lat, sn_donor[pool_idx], held_dn)

        print(f"  fold {fold+1}/{len(paired_dts)}: held=({held_dn}, {held_ts});  "
              f"FLIP top-1={topK_flip[1]:.2f} rank={rank_flip}  |  "
              f"LATENT top-1={topK_lat[1]:.2f} rank={rank_lat}", flush=True)

        fold_results.append({
            "held_donor": held_dn, "held_tissue": held_ts,
            "n_pool": len(pool_idx), "n_same_donor_in_pool": int((sn_donor[pool_idx] == held_dn).sum()),
            "flip_rank_first_same": rank_flip, "flip_topK": topK_flip,
            "latent_rank_first_same": rank_lat, "latent_topK": topK_lat,
        })

    arr_flip_top1 = np.array([f["flip_topK"][1] for f in fold_results])
    arr_flip_top3 = np.array([f["flip_topK"][3] for f in fold_results])
    arr_flip_rank = np.array([f["flip_rank_first_same"] for f in fold_results])
    arr_lat_top1 = np.array([f["latent_topK"][1] for f in fold_results])
    arr_lat_top3 = np.array([f["latent_topK"][3] for f in fold_results])
    arr_lat_rank = np.array([f["latent_rank_first_same"] for f in fold_results])
    arr_n_same = np.array([f["n_same_donor_in_pool"] for f in fold_results])
    arr_n_pool = np.array([f["n_pool"] for f in fold_results])
    expected_top1 = float(np.mean(arr_n_same / arr_n_pool))
    print(f"\n=== Run 23 (tiny-z={Z_BIO_DIM}) LOO summary across {len(fold_results)} folds ===")
    print(f"  FLIP   top-1: {arr_flip_top1.mean():.3f}  top-3: {arr_flip_top3.mean():.3f}  rank1: {arr_flip_rank.mean():.2f}")
    print(f"  LATENT top-1: {arr_lat_top1.mean():.3f}  top-3: {arr_lat_top3.mean():.3f}  rank1: {arr_lat_rank.mean():.2f}")
    print(f"  random top-1: {expected_top1:.3f}")
    print(f"  Reference: CCA-5 = 0.604, ENS = 0.615, PCA-50 = 0.510")

    out = {"config": {"z_bio_dim": Z_BIO_DIM, "epochs": EPOCHS},
           "fold_results": fold_results,
           "summary": {
               "flip_top1": float(arr_flip_top1.mean()),
               "flip_top3": float(arr_flip_top3.mean()),
               "flip_rank_first_same": float(arr_flip_rank.mean()),
               "latent_top1": float(arr_lat_top1.mean()),
               "latent_top3": float(arr_lat_top3.mean()),
               "latent_rank_first_same": float(arr_lat_rank.mean()),
               "expected_top1_random": expected_top1,
           }}
    with (OUT_DIR / "q40_run23_tinyz.json").open("w") as f:
        json.dump(out, f, indent=2)
    print(f"saved → {OUT_DIR / 'q40_run23_tinyz.json'}")


if __name__ == "__main__":
    main()
