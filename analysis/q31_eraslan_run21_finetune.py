"""Q31 / Run 21 — Pretrain on Eraslan multi-tissue, fine-tune per-tissue.

Step 1: train ONE multi-tissue model on ALL Eraslan + GTEx data (no LOO held out)
        → single pretrained checkpoint.
Step 2: per held-out donor in target tissue, init from pretrained checkpoint
        and fine-tune ONLY on the target tissue's data (LOO over donors).

Hypothesis: pretraining on multi-tissue gives the encoder broad
donor-discriminative features; fine-tuning specializes for the test tissue.
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
    kl_with_free_bits, hsic_penalty,
)
from analysis.q27_eraslan_run17_decompose import DecomposedVAE

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

PAIRED_NPZ = "/Users/rls/ecs271/data/sc/eraslan/eraslan_paired_plus_gtex.npz"
OUT_DIR = ROOT / "analysis" / "results" / "q20_disentangled"
PRETRAIN_CKPT = OUT_DIR / "q31_pretrain_multi_tissue.pt"

TISSUE = os.environ.get("Q31_TISSUE", "prostate")
PRETRAIN_EPOCHS = int(os.environ.get("Q31_PRETRAIN_EPOCHS", "150"))
FINETUNE_EPOCHS = int(os.environ.get("Q31_FT_EPOCHS", "150"))
LR_PRETRAIN = 1e-3
LR_FT = 5e-4
LAM_PAIRED = 5.0
LAM_LEAK = 0.3
LAM_CYCLE = 0.3
LAM_SUP = 1.0
LAM_DONOR_ID = 5.0
LAM_RECON_EXTRA = 0.5
BETA_BIO = 1e-3


def _cosine(a, b):
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _make_pretrain_data(d):
    """All Eraslan + GTEx bulk + Eraslan sn (no LOO held out yet)."""
    bulk_x = d["bulk_x"].astype(np.float32)
    bulk_donor = np.asarray(d["bulk_donor"], dtype=str)
    bulk_tissue = np.asarray(d["bulk_tissue"], dtype=str)
    bulk_is_eraslan = np.asarray(d["bulk_is_eraslan"], dtype=bool)
    sn_x = d["sn_x"].astype(np.float32)
    sn_donor = np.asarray(d["sn_donor"], dtype=str)
    sn_tissue = np.asarray(d["sn_tissue"], dtype=str)
    return bulk_x, bulk_donor, bulk_tissue, bulk_is_eraslan, sn_x, sn_donor, sn_tissue


def _train_step(model, opt, Xb_pair, Xs_pair, Xb_extra, m_b_pair_mod, m_s_pair_mod,
                m_b_pair_tis, m_s_pair_tis, m_e_mod, m_e_tis,
                bulk_z0_target, sc_z0_target):
    x_hat_b, mu_m_b, lv_m_b, mu_b_b, lv_b_b, z_m_b, z_b_b = model(Xb_pair)
    x_hat_s, mu_m_s, lv_m_s, mu_b_s, lv_b_s, z_m_s, z_b_s = model(Xs_pair)
    if Xb_extra is not None and Xb_extra.size(0) > 0:
        x_hat_e, mu_m_e, lv_m_e, mu_b_e, lv_b_e, z_m_e, z_b_e = model(Xb_extra)
        recon_e = LAM_RECON_EXTRA * nn.functional.mse_loss(x_hat_e, Xb_extra)
        kl_e = kl_with_free_bits(mu_b_e, lv_b_e, 0.5) / Xb_pair.size(0)
        pred_e_mod = model.head_mod(z_m_e[:, 0:1]).squeeze(-1)
        pred_e_tis = model.head_tis(z_m_e[:, 1:2]).squeeze(-1)
        sup_mod_e = nn.functional.binary_cross_entropy_with_logits(pred_e_mod, m_e_mod)
        sup_tis_e = nn.functional.mse_loss(pred_e_tis, m_e_tis)
    else:
        recon_e = torch.tensor(0.0, device=DEVICE)
        kl_e = torch.tensor(0.0, device=DEVICE)
        sup_mod_e = torch.tensor(0.0, device=DEVICE)
        sup_tis_e = torch.tensor(0.0, device=DEVICE)

    recon = nn.functional.mse_loss(x_hat_b, Xb_pair) + nn.functional.mse_loss(x_hat_s, Xs_pair) + recon_e
    kl_b = ((kl_with_free_bits(mu_b_b, lv_b_b, 0.5)
             + kl_with_free_bits(mu_b_s, lv_b_s, 0.5)) / Xb_pair.size(0)) + kl_e

    pred_b = model.head_mod(z_m_b[:, 0:1]).squeeze(-1)
    pred_s = model.head_mod(z_m_s[:, 0:1]).squeeze(-1)
    sup_mod = (nn.functional.binary_cross_entropy_with_logits(pred_b, m_b_pair_mod)
               + nn.functional.binary_cross_entropy_with_logits(pred_s, m_s_pair_mod)
               + sup_mod_e)
    pred_t_b = model.head_tis(z_m_b[:, 1:2]).squeeze(-1)
    pred_t_s = model.head_tis(z_m_s[:, 1:2]).squeeze(-1)
    sup_tis = (nn.functional.mse_loss(pred_t_b, m_b_pair_tis)
               + nn.functional.mse_loss(pred_t_s, m_s_pair_tis)
               + sup_tis_e)

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
    return loss.item(), recon.item(), paired_b2s.item(), donor_id.item()


def pretrain():
    """Train one model on ALL Eraslan multi-tissue paired + extra GTEx bulk."""
    print(f"\n[PRETRAIN] {PRETRAIN_EPOCHS} epochs on full Eraslan multi-tissue + GTEx")
    d = np.load(PAIRED_NPZ, allow_pickle=True)
    bulk_x, bulk_donor, bulk_tissue, bulk_is_eraslan, sn_x, sn_donor, sn_tissue = _make_pretrain_data(d)
    tissues = sorted(set(np.concatenate([bulk_tissue, sn_tissue])))
    tissue_to_idx = {t: i for i, t in enumerate(tissues)}

    sn_by_dt = {}
    for j, (dn, ts) in enumerate(zip(sn_donor, sn_tissue)):
        sn_by_dt.setdefault((dn, ts), []).append(j)
    eraslan_idx = np.where(bulk_is_eraslan)[0]
    paired_pairs = []
    for ii in eraslan_idx:
        for j in sn_by_dt.get((bulk_donor[ii], bulk_tissue[ii]), []):
            paired_pairs.append((ii, j))
    print(f"  pretrain paired pairs: {len(paired_pairs)}")
    bulk_pair_idx = np.array([p[0] for p in paired_pairs])
    sn_pair_idx = np.array([p[1] for p in paired_pairs])
    Xb_pair = torch.from_numpy(bulk_x[bulk_pair_idx]).to(DEVICE)
    Xs_pair = torch.from_numpy(sn_x[sn_pair_idx]).to(DEVICE)
    extra_idx = np.where(~bulk_is_eraslan)[0]
    Xb_extra = torch.from_numpy(bulk_x[extra_idx]).to(DEVICE)
    m_extra_tis = torch.tensor([tissue_to_idx[t] for t in bulk_tissue[extra_idx]],
                               device=DEVICE, dtype=torch.float32) / max(len(tissues) - 1, 1)
    m_b_pair_tis = torch.tensor([tissue_to_idx[t] for t in bulk_tissue[bulk_pair_idx]],
                                device=DEVICE, dtype=torch.float32) / max(len(tissues) - 1, 1)
    m_s_pair_tis = torch.tensor([tissue_to_idx[t] for t in sn_tissue[sn_pair_idx]],
                                device=DEVICE, dtype=torch.float32) / max(len(tissues) - 1, 1)
    m_b_pair_mod = torch.zeros(Xb_pair.size(0), device=DEVICE)
    m_s_pair_mod = torch.ones(Xs_pair.size(0), device=DEVICE)

    model = DecomposedVAE(input_dim=bulk_x.shape[1], z_meta_dim=2, z_bio_dim=50).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR_PRETRAIN)
    rng = np.random.default_rng(SEED)
    batch_extra = min(512, Xb_extra.size(0))
    bulk_z0_target = -3.0; sc_z0_target = 3.0

    for ep in range(1, PRETRAIN_EPOCHS + 1):
        idx_e = torch.from_numpy(rng.choice(Xb_extra.size(0), batch_extra, replace=False)).to(DEVICE)
        Xb_e_batch = Xb_extra[idx_e]
        m_e_tis = m_extra_tis[idx_e]
        m_e_mod = torch.zeros(batch_extra, device=DEVICE)
        l, r, p, di = _train_step(model, opt, Xb_pair, Xs_pair, Xb_e_batch,
                                  m_b_pair_mod, m_s_pair_mod, m_b_pair_tis, m_s_pair_tis,
                                  m_e_mod, m_e_tis, bulk_z0_target, sc_z0_target)
        if ep % 25 == 0 or ep == 1:
            print(f"  pre-ep{ep:3d}  loss={l:.3f}  recon={r:.3f}  paired={p:.3f}  donor_id={di:.4f}", flush=True)

    torch.save({"state_dict": model.state_dict(),
                "tissue_to_idx": tissue_to_idx,
                "n_tissues": len(tissues)}, PRETRAIN_CKPT)
    print(f"  saved pretrain ckpt → {PRETRAIN_CKPT}")
    return tissue_to_idx, len(tissues)


def main():
    print(f"[Q31/Run21 pretrain+finetune] device={DEVICE}  tissue={TISSUE}")

    if not PRETRAIN_CKPT.exists():
        tissue_to_idx, n_tissues = pretrain()
    else:
        ckpt = torch.load(PRETRAIN_CKPT, map_location=DEVICE, weights_only=False)
        tissue_to_idx = ckpt["tissue_to_idx"]; n_tissues = ckpt["n_tissues"]
        print(f"  using existing pretrain ckpt {PRETRAIN_CKPT.name}")

    d = np.load(PAIRED_NPZ, allow_pickle=True)
    bulk_x, bulk_donor, bulk_tissue, bulk_is_eraslan, sn_x, sn_donor, sn_tissue = _make_pretrain_data(d)
    bulk_keep = bulk_tissue == TISSUE
    sn_keep = sn_tissue == TISSUE
    bulk_x_t = bulk_x[bulk_keep]; bulk_donor_t = bulk_donor[bulk_keep]
    bulk_is_eraslan_t = bulk_is_eraslan[bulk_keep]; bulk_tissue_t = bulk_tissue[bulk_keep]
    sn_x_t = sn_x[sn_keep]; sn_donor_t = sn_donor[sn_keep]; sn_tissue_t = sn_tissue[sn_keep]
    print(f"  {TISSUE} bulk: {bulk_x_t.shape}  sn: {sn_x_t.shape}")

    paired_donors = sorted(set(bulk_donor_t[bulk_is_eraslan_t]) & set(sn_donor_t))
    print(f"  paired donors: {paired_donors}")

    bulk_z0_target = -3.0; sc_z0_target = 3.0
    fold_results = []
    for fold, held_dn in enumerate(paired_donors):
        held_bulk_i = int(np.where((bulk_donor_t == held_dn) & bulk_is_eraslan_t)[0][0])
        held_sn_rows = set(np.where(sn_donor_t == held_dn)[0])
        train_bulk_mask = np.arange(len(bulk_x_t)) != held_bulk_i
        train_sn_mask = np.array([i not in held_sn_rows for i in range(len(sn_x_t))])
        Xb_all = bulk_x_t[train_bulk_mask]
        donor_b_all = bulk_donor_t[train_bulk_mask]
        is_eraslan_all = bulk_is_eraslan_t[train_bulk_mask]
        Xs_tr = sn_x_t[train_sn_mask]; donor_s_tr = sn_donor_t[train_sn_mask]

        sn_by_donor = {}
        for j, dn in enumerate(donor_s_tr):
            sn_by_donor.setdefault(dn, []).append(j)
        paired_pairs = []
        for ii in np.where(is_eraslan_all)[0]:
            dn = donor_b_all[ii]
            for j in sn_by_donor.get(dn, []):
                paired_pairs.append((ii, j))
        if not paired_pairs:
            continue
        if fold == 0:
            print(f"\n  fold 1: paired pairs = {len(paired_pairs)}")
        bulk_pair_idx = np.array([p[0] for p in paired_pairs])
        sn_pair_idx = np.array([p[1] for p in paired_pairs])
        Xb_pair = torch.from_numpy(Xb_all[bulk_pair_idx]).to(DEVICE)
        Xs_pair = torch.from_numpy(Xs_tr[sn_pair_idx]).to(DEVICE)
        m_b_pair_tis = torch.full((Xb_pair.size(0),), float(tissue_to_idx[TISSUE]) / max(n_tissues - 1, 1),
                                  device=DEVICE)
        m_s_pair_tis = m_b_pair_tis.clone()[:Xs_pair.size(0)]
        m_b_pair_mod = torch.zeros(Xb_pair.size(0), device=DEVICE)
        m_s_pair_mod = torch.ones(Xs_pair.size(0), device=DEVICE)
        extra_idx_t = np.where(~is_eraslan_all)[0]
        Xb_extra = torch.from_numpy(Xb_all[extra_idx_t]).to(DEVICE)
        m_e_tis = torch.full((Xb_extra.size(0),), float(tissue_to_idx[TISSUE]) / max(n_tissues - 1, 1),
                             device=DEVICE)
        m_e_mod = torch.zeros(Xb_extra.size(0), device=DEVICE)

        # Init from pretrain
        model = DecomposedVAE(input_dim=bulk_x.shape[1], z_meta_dim=2, z_bio_dim=50).to(DEVICE)
        ckpt = torch.load(PRETRAIN_CKPT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        opt = torch.optim.Adam(model.parameters(), lr=LR_FT)
        rng = np.random.default_rng(SEED + fold)
        batch_extra = min(256, Xb_extra.size(0))

        for ep in range(1, FINETUNE_EPOCHS + 1):
            idx_e = torch.from_numpy(
                rng.choice(Xb_extra.size(0), batch_extra, replace=False)
            ).to(DEVICE) if Xb_extra.size(0) > 0 else None
            Xb_e_batch = Xb_extra[idx_e] if idx_e is not None else None
            m_e_t = m_e_tis[idx_e] if idx_e is not None else None
            m_e_m = m_e_mod[idx_e] if idx_e is not None else None
            l, r, p, di = _train_step(model, opt, Xb_pair, Xs_pair, Xb_e_batch,
                                      m_b_pair_mod, m_s_pair_mod, m_b_pair_tis, m_s_pair_tis,
                                      m_e_m, m_e_t, bulk_z0_target, sc_z0_target)
            if fold == 0 and (ep % 25 == 0 or ep == 1):
                print(f"    ft-ep{ep:3d}  loss={l:.3f}  recon={r:.3f}  paired={p:.3f}  donor_id={di:.4f}",
                      flush=True)

        model.eval()
        with torch.no_grad():
            x_held_b = torch.from_numpy(bulk_x_t[held_bulk_i:held_bulk_i+1]).to(DEVICE)
            mu_m_h, _, mu_b_h, _ = model.encode(x_held_b)
            z_m_flip = mu_m_h.clone(); z_m_flip[:, 0] = sc_z0_target
            x_hat_sc = model.decode(z_m_flip, mu_b_h).cpu().numpy()[0]

        cos_to_pool = np.array([_cosine(x_hat_sc, sn_x_t[i]) for i in range(len(sn_x_t))])
        order = np.argsort(-cos_to_pool)
        ordered_donors = sn_donor_t[order]
        rank_first_same = next((i for i, dn in enumerate(ordered_donors) if dn == held_dn), None)
        rank_first_same_1based = (rank_first_same + 1) if rank_first_same is not None else None
        n_same_in_pool = int((sn_donor_t == held_dn).sum())
        same_donor_ranks = [i + 1 for i, dn in enumerate(ordered_donors) if dn == held_dn]
        topK = {K: float(np.mean(ordered_donors[:K] == held_dn)) for K in [1, 3, 5, 10]}
        print(f"  fold {fold+1}/{len(paired_donors)}: held={held_dn}; "
              f"first-same-rank={rank_first_same_1based}/{len(sn_x_t)}; "
              f"top-1={topK[1]:.2f}; top-3={topK[3]:.2f}; top-10={topK[10]:.2f}", flush=True)
        fold_results.append({
            "held_donor": held_dn, "n_pool": int(len(sn_x_t)), "n_same_donor_in_pool": n_same_in_pool,
            "rank_first_same": rank_first_same_1based,
            "mean_rank_same_donor": float(np.mean(same_donor_ranks)),
            "topK_purity": topK, "n_paired_train_pairs": len(paired_pairs),
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
    print(f"\n=== Run 21 ({TISSUE}) LOO across {len(fold_results)} donors ===")
    print(f"  mean top-1: {arr_top1.mean():.3f}  (random ~{expected_top1_random:.3f})")
    print(f"  mean top-3: {arr_top3.mean():.3f}")
    print(f"  mean top-5: {arr_top5.mean():.3f}")
    print(f"  mean top-10: {arr_top10.mean():.3f}")
    print(f"  mean rank-of-first-same: {arr_rank1.mean():.2f}  (random ~{expected_rank1_random:.2f})")

    out = {
        "config": {"tissue": TISSUE, "pretrain_epochs": PRETRAIN_EPOCHS,
                   "finetune_epochs": FINETUNE_EPOCHS,
                   "pretrain_lr": LR_PRETRAIN, "ft_lr": LR_FT},
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
    out_path = OUT_DIR / f"q31_run21_finetune_{TISSUE}.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"saved → {out_path}")


if __name__ == "__main__":
    main()
