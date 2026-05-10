"""Q22 — Leave-one-out paired fine-tuning of Run-13 with the 7 paired donors.

For each held-out donor i in {1..7}:
  1. Init from Run 13 weights
  2. Fine-tune 50 epochs on the OTHER 6 paired (bulk_d, sc_d) pairs with:
       - paired flip MSE: MSE( decode(z_meta_sc, encode(x_bulk_d).z_bio), x_sc_d )
       - paired reverse MSE: MSE( decode(z_meta_bulk, encode(x_sc_d).z_bio), x_bulk_d )
       - keep the other Run-10/13 losses (recon, KL, leak, cycle, sup) at lower weight
  3. Test on the held-out donor: NN among all 7 sc using the flipped bulk_i
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
from analysis.q20_disentangled_vae import DEVICE, OUT, build_data, _hca_pseudobulk_aligned, SEED
from analysis.q21_paired_donor_test2 import load_bulk, load_sc_pseudobulk, _align, _cosine
from pipeline.data import load_shared_genes

INIT_CKPT = OUT / "q20_run13_melanoma_in_train.pt"
RESULT = OUT / "q22_loo_paired_finetune.json"

EPOCHS_FT = 50
LR_FT = 1e-4   # lower lr for fine-tuning
LAM_PAIRED = 5.0  # heavy weight on paired-flip MSE
LAM_RECON = 1.0
LAM_LEAK = 0.3
LAM_CYCLE = 0.3


def main():
    print(f"[Q22] init={INIT_CKPT.name} device={DEVICE}")
    shared_genes, scaler_mean, scaler_std = load_shared_genes()

    # Paired data (7 donors)
    bulk_log, bulk_genes, bulk_donors = load_bulk()
    bulk_aligned, _ = _align(bulk_log, bulk_genes, shared_genes)
    bulk_scaled = ((bulk_aligned - scaler_mean) / scaler_std).astype(np.float32)
    sc_log, sc_genes, sc_donors_per_row, _ = load_sc_pseudobulk(bulk_donors,
                                                                  n_subsample=None,
                                                                  n_replicates=1)
    sc_aligned, _ = _align(sc_log, sc_genes, shared_genes)
    sc_scaled = ((sc_aligned - scaler_mean) / scaler_std).astype(np.float32)

    common = [d for d in bulk_donors if d in sc_donors_per_row]
    bulk_idx = [bulk_donors.index(d) for d in common]
    sc_idx = [sc_donors_per_row.index(d) for d in common]
    bulk_paired = bulk_scaled[bulk_idx]
    sc_paired = sc_scaled[sc_idx]
    n = len(common)
    print(f"  paired donors: {n} = {common}")

    # Pre-compute the encoded class-mean targets for modality flips. Use the
    # whole TRAINING distribution to define them (not just paired donors).
    print("  loading background training data to define class-mean flip targets …")
    data = build_data(include_melanoma=True)

    test2_ranks_b2s = []
    test2_ranks_s2b = []
    cos_b2s_all = np.zeros((n, n))
    cos_s2b_all = np.zeros((n, n))

    for held in range(n):
        train_idx = [i for i in range(n) if i != held]
        held_donor = common[held]
        print(f"\n=== fold {held+1}/{n}: hold out {held_donor} ===")

        # Reload model from checkpoint (fresh weights each fold)
        ckpt = torch.load(INIT_CKPT, map_location=DEVICE, weights_only=False)
        cfg = DisentangledConfig(**ckpt["config"]) if isinstance(ckpt["config"], dict) else ckpt["config"]
        model = DisentangledVAE(cfg).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        opt = torch.optim.Adam(model.parameters(), lr=LR_FT)

        # Empirical class-means in z_meta[0] using a fresh encode of the broad
        # training distribution.
        model.eval()
        with torch.no_grad():
            t = torch.from_numpy(data["x_gtex"]).to(DEVICE)
            mu_m, _, _, _ = model.encode(t)
            bulk_z0 = mu_m[:, 0].mean().item()
            t = torch.from_numpy(data["x_hca"]).to(DEVICE)
            mu_m_h, _, _, _ = model.encode(t)
            sc_z0 = mu_m_h[:, 0].mean().item()
        print(f"    pretrain class means: bulk={bulk_z0:.2f}, sc={sc_z0:.2f}")

        # Fine-tune on the 6 paired donors
        Xb = torch.from_numpy(bulk_paired[train_idx]).to(DEVICE)
        Xs = torch.from_numpy(sc_paired[train_idx]).to(DEVICE)
        # Modality labels for the paired training donors
        ones = torch.ones(len(train_idx), device=DEVICE)
        zeros = torch.zeros(len(train_idx), device=DEVICE)

        for ep in range(1, EPOCHS_FT + 1):
            model.train()
            # Forward bulk and sc together
            x_hat_b, mu_m_b, lv_m_b, mu_b_b, lv_b_b, z_m_b, z_b_b = model(Xb)
            x_hat_s, mu_m_s, lv_m_s, mu_b_s, lv_b_s, z_m_s, z_b_s = model(Xs)

            # Recon loss
            recon = nn.functional.mse_loss(x_hat_b, Xb) + nn.functional.mse_loss(x_hat_s, Xs)
            kl_b = (kl_with_free_bits(mu_b_b, lv_b_b, 0.5) + kl_with_free_bits(mu_b_s, lv_b_s, 0.5)) / Xb.size(0)

            # Modality-classification supervision on z_meta[0]
            pred_b_logit = model.heads[0](z_m_b[:, 0:1]).squeeze(-1)
            pred_s_logit = model.heads[0](z_m_s[:, 0:1]).squeeze(-1)
            sup = (nn.functional.binary_cross_entropy_with_logits(pred_b_logit, zeros)
                   + nn.functional.binary_cross_entropy_with_logits(pred_s_logit, ones))

            # PAIRED FLIP MSE — the new term
            # Flip bulk → sc: replace z_meta[0] with sc class-mean, decode → should match sc_paired
            z_m_flip_b = z_m_b.clone(); z_m_flip_b[:, 0] = sc_z0
            x_flip_b2s = model.decode(z_m_flip_b, z_b_b)
            paired_b2s = nn.functional.mse_loss(x_flip_b2s, Xs)
            # Flip sc → bulk
            z_m_flip_s = z_m_s.clone(); z_m_flip_s[:, 0] = bulk_z0
            x_flip_s2b = model.decode(z_m_flip_s, z_b_s)
            paired_s2b = nn.functional.mse_loss(x_flip_s2b, Xb)

            # HSIC leak (modality only) on a small batch
            mod_col = torch.cat([torch.full((Xb.size(0),1), 0.0, device=DEVICE),
                                 torch.full((Xs.size(0),1), 1.0, device=DEVICE)])
            z_b_combined = torch.cat([z_b_b, z_b_s], dim=0)
            leak = hsic_penalty(z_b_combined, mod_col) if z_b_combined.size(0) >= 4 else z_b_b.new_zeros(())

            loss = (LAM_RECON * recon
                    + 1e-3 * kl_b
                    + 1.0 * sup
                    + LAM_PAIRED * (paired_b2s + paired_s2b)
                    + LAM_LEAK * leak)

            opt.zero_grad(); loss.backward(); opt.step()
            if ep % 10 == 0 or ep == 1:
                print(f"    ep{ep:2d}  loss={loss.item():.3f}  recon={recon.item():.3f}  "
                      f"paired_b2s={paired_b2s.item():.3f}  paired_s2b={paired_s2b.item():.3f}",
                      flush=True)

        # Test on the held-out donor
        model.eval()
        with torch.no_grad():
            x_held_b = torch.from_numpy(bulk_paired[held:held+1]).to(DEVICE)
            x_held_s = torch.from_numpy(sc_paired[held:held+1]).to(DEVICE)
            mu_m_h, _, mu_b_h, _ = model.encode(x_held_b)
            mu_m_s_h, _, mu_b_s_h, _ = model.encode(x_held_s)
            z_m_flip = mu_m_h.clone(); z_m_flip[:, 0] = sc_z0
            x_hat_sc_held = model.decode(z_m_flip, mu_b_h).cpu().numpy()[0]
            z_m_flip_s = mu_m_s_h.clone(); z_m_flip_s[:, 0] = bulk_z0
            x_hat_bulk_held = model.decode(z_m_flip_s, mu_b_s_h).cpu().numpy()[0]

        # NN among ALL 7 real sc samples (including the held out donor's true sc)
        cos_b2s = np.array([_cosine(x_hat_sc_held, sc_paired[j]) for j in range(n)])
        cos_s2b = np.array([_cosine(x_hat_bulk_held, bulk_paired[j]) for j in range(n)])
        cos_b2s_all[held] = cos_b2s
        cos_s2b_all[held] = cos_s2b

        order_b2s = np.argsort(-cos_b2s)
        order_s2b = np.argsort(-cos_s2b)
        rank_b2s = int(np.where(order_b2s == held)[0][0]) + 1
        rank_s2b = int(np.where(order_s2b == held)[0][0]) + 1
        print(f"    held-out {held_donor}: flipBulk→sc rank = {rank_b2s}/{n}, "
              f"flipSC→bulk rank = {rank_s2b}/{n}")
        test2_ranks_b2s.append(rank_b2s)
        test2_ranks_s2b.append(rank_s2b)

    # Aggregate
    top1_b2s = float((np.array(test2_ranks_b2s) == 1).mean())
    top1_s2b = float((np.array(test2_ranks_s2b) == 1).mean())
    print(f"\n=== LOO summary ===")
    print(f"  flipBulk→realSC top-1: {top1_b2s:.3f}  (random {1/n:.3f})")
    print(f"  flipSC→realBulk top-1: {top1_s2b:.3f}")
    print(f"  mean rank b2s: {np.mean(test2_ranks_b2s):.2f}, s2b: {np.mean(test2_ranks_s2b):.2f}")

    out = {
        "init_checkpoint": str(INIT_CKPT.name),
        "n_paired_donors": int(n),
        "donor_order": common,
        "loo_top1_flipBulk_to_realSC": top1_b2s,
        "loo_top1_flipSC_to_realBulk": top1_s2b,
        "loo_mean_rank_b2s": float(np.mean(test2_ranks_b2s)),
        "loo_mean_rank_s2b": float(np.mean(test2_ranks_s2b)),
        "loo_ranks_b2s": test2_ranks_b2s,
        "loo_ranks_s2b": test2_ranks_s2b,
        "ft_epochs": EPOCHS_FT,
        "lr_ft": LR_FT,
        "lam_paired": LAM_PAIRED,
        "cos_b2s_held_to_all_sc": cos_b2s_all.tolist(),
        "cos_s2b_held_to_all_bulk": cos_s2b_all.tolist(),
    }
    with RESULT.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"saved → {RESULT}")


if __name__ == "__main__":
    main()
