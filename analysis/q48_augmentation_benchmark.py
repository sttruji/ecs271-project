"""Q48 — Downstream-utility benchmark: does flipping sn→bulk augment a bulk
classifier in a meaningful way?

This directly tests Proposal Goal #3: "jointly leveraging bulk and sc improves
downstream predictive performance vs single-modality baselines".

Setup: tissue classification from bulk gene expression, evaluated on Eraslan
paired-donor bulk samples via leave-one-bulk-out CV.

Three training conditions:
  (A) BASELINE: real bulk only (92 Eraslan samples, leave 1 out per fold)
  (B) RAW SN: real bulk + raw sn pseudobulks (no flipping, just concatenate)
  (C) OUR FLIP: real bulk + VAE-flipped sn pseudobulks (sn→bulk via flip)

If (C) > (A): flipping helps (proposal works).
If (B) > (A) but (C) ≈ (B): just adding sn helps; flipping is irrelevant.
If (C) > (B): flipping ALSO removes the cross-modality domain shift.

Comparable to: scVI's data-augmentation benchmarks, BulkFormer pretraining
extensions, scGPT's perturbation-prediction transfer tests.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, top_k_accuracy_score
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.disentangled_vae import kl_with_free_bits, hsic_penalty
from analysis.q45_run27_tech_aware import TechAwareVAE, info_nce, TECH_MAP, _log_depth_proxy

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

PAIRED_NPZ = "/Users/rls/ecs271/data/sc/eraslan/eraslan_paired.npz"
OUT_DIR = ROOT / "analysis" / "results" / "q20_disentangled"

EPOCHS = 200
LR = 1e-3
LAM_PAIRED = 5.0; LAM_LEAK = 0.3; LAM_CYCLE = 0.3
LAM_SUP = 1.0; LAM_DONOR_ID = 5.0
LAM_NCE_LATENT = 5.0; NCE_TAU = 0.1
BETA_BIO = 1e-3
Z_BIO_DIM = 5


def _train_tech_vae(Xb_pair, Xs_pair, m_b_tis, m_s_tis, m_b_tech, m_s_tech,
                     m_b_depth, m_s_depth, n_genes, bulk_z0, sc_z0):
    model = TechAwareVAE(input_dim=n_genes, z_bio_dim=Z_BIO_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    m_b_mod = torch.zeros(Xb_pair.size(0), device=DEVICE)
    m_s_mod = torch.ones(Xs_pair.size(0), device=DEVICE)
    for ep in range(1, EPOCHS + 1):
        x_hat_b, mu_m_b, lv_m_b, mu_b_b, lv_b_b, z_m_b, z_b_b = model(Xb_pair)
        x_hat_s, mu_m_s, lv_m_s, mu_b_s, lv_b_s, z_m_s, z_b_s = model(Xs_pair)
        recon = nn.functional.mse_loss(x_hat_b, Xb_pair) + nn.functional.mse_loss(x_hat_s, Xs_pair)
        kl = (kl_with_free_bits(mu_b_b, lv_b_b, 0.5) + kl_with_free_bits(mu_b_s, lv_b_s, 0.5)) / Xb_pair.size(0)
        sup_mod = (nn.functional.binary_cross_entropy_with_logits(model.head_mod(z_m_b[:, 0:1]).squeeze(-1), m_b_mod)
                   + nn.functional.binary_cross_entropy_with_logits(model.head_mod(z_m_s[:, 0:1]).squeeze(-1), m_s_mod))
        sup_tis = (nn.functional.mse_loss(model.head_tis(z_m_b[:, 1:2]).squeeze(-1), m_b_tis)
                   + nn.functional.mse_loss(model.head_tis(z_m_s[:, 1:2]).squeeze(-1), m_s_tis))
        td = model.tech_dim
        sup_tech = (nn.functional.cross_entropy(model.head_tech(z_m_b[:, 2:2+td]), m_b_tech)
                    + nn.functional.cross_entropy(model.head_tech(z_m_s[:, 2:2+td]), m_s_tech))
        sup_depth = (nn.functional.mse_loss(model.head_depth(z_m_b[:, 2+td:2+td+1]).squeeze(-1), m_b_depth)
                     + nn.functional.mse_loss(model.head_depth(z_m_s[:, 2+td:2+td+1]).squeeze(-1), m_s_depth))
        z_m_flip_b = z_m_b.clone(); z_m_flip_b[:, 0] = sc_z0
        sn_tech_id_t = torch.full((Xb_pair.size(0),), TECH_MAP["10x_chromium"], dtype=torch.long, device=DEVICE)
        z_m_flip_b[:, 2:2+td] = model.tech_embed(sn_tech_id_t)
        z_m_flip_b[:, 2+td] = m_s_depth.mean()
        x_flip_b2s = model.decode(z_m_flip_b, z_b_b)
        paired_b2s = nn.functional.mse_loss(x_flip_b2s, Xs_pair)
        z_m_flip_s = z_m_s.clone(); z_m_flip_s[:, 0] = bulk_z0
        bulk_tech_id_t = torch.full((Xs_pair.size(0),), TECH_MAP["bulk_illumina"], dtype=torch.long, device=DEVICE)
        z_m_flip_s[:, 2:2+td] = model.tech_embed(bulk_tech_id_t)
        z_m_flip_s[:, 2+td] = m_b_depth.mean()
        x_flip_s2b = model.decode(z_m_flip_s, z_b_s)
        paired_s2b = nn.functional.mse_loss(x_flip_s2b, Xb_pair)
        nce_lat = info_nce(mu_b_b, mu_b_s, tau=NCE_TAU)
        mu_m_re_b, _, mu_b_re_b, _ = model.encode(x_flip_b2s)
        mu_m_re_s, _, mu_b_re_s, _ = model.encode(x_flip_s2b)
        cyc = nn.functional.mse_loss(mu_b_re_b, mu_b_b.detach()) + nn.functional.mse_loss(mu_b_re_s, mu_b_s.detach())
        mod_col = torch.cat([torch.full((Xb_pair.size(0),1), 0.0, device=DEVICE),
                             torch.full((Xs_pair.size(0),1), 1.0, device=DEVICE)])
        z_b_combined = torch.cat([z_b_b, z_b_s], dim=0)
        leak = hsic_penalty(z_b_combined, mod_col)
        donor_id = nn.functional.mse_loss(mu_b_b, mu_b_s)
        loss = (recon + BETA_BIO * kl + LAM_SUP * (sup_mod + sup_tis + sup_tech + sup_depth)
                + LAM_PAIRED * (paired_b2s + paired_s2b)
                + LAM_CYCLE * cyc + LAM_LEAK * leak + LAM_DONOR_ID * donor_id
                + LAM_NCE_LATENT * nce_lat)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def main():
    print(f"[Q48 augmentation benchmark] device={DEVICE}")
    d = np.load(PAIRED_NPZ, allow_pickle=True)
    bulk_x = d["bulk_x"].astype(np.float32)
    bulk_donor = np.asarray(d["bulk_donor"], dtype=str)
    bulk_tissue = np.asarray(d["bulk_tissue"], dtype=str)
    sn_x = d["sn_x"].astype(np.float32)
    sn_donor = np.asarray(d["sn_donor"], dtype=str)
    sn_tissue = np.asarray(d["sn_tissue"], dtype=str)
    print(f"  bulk: {bulk_x.shape}  sn: {sn_x.shape}")
    tissues = sorted(set(bulk_tissue) | set(sn_tissue))
    tissue_to_idx = {t: i for i, t in enumerate(tissues)}
    n_tissues = len(tissues)
    print(f"  tissues: {tissues}")

    bulk_tech = np.full(len(bulk_x), TECH_MAP["bulk_illumina"], dtype=np.int64)
    sn_tech = np.full(len(sn_x), TECH_MAP["10x_chromium"], dtype=np.int64)
    bulk_depth = _log_depth_proxy(bulk_x)
    sn_depth = _log_depth_proxy(sn_x)
    all_depth = np.concatenate([bulk_depth, sn_depth])
    depth_mean, depth_std = all_depth.mean(), all_depth.std() + 1e-8

    # Train one VAE on ALL paired data
    print("\n=== Training TechAwareVAE on all paired data ===")
    sn_idx_per_dt = {}
    for i, (dn, ts) in enumerate(zip(sn_donor, sn_tissue)):
        sn_idx_per_dt.setdefault((dn, ts), []).append(i)
    paired_pairs = []
    for i, (dn, ts) in enumerate(zip(bulk_donor, bulk_tissue)):
        for j in sn_idx_per_dt.get((dn, ts), []):
            paired_pairs.append((i, j))
    bp = np.array([p[0] for p in paired_pairs])
    sp = np.array([p[1] for p in paired_pairs])
    Xb_pair = torch.from_numpy(bulk_x[bp]).to(DEVICE)
    Xs_pair = torch.from_numpy(sn_x[sp]).to(DEVICE)
    m_b_tis = torch.tensor([tissue_to_idx[t] for t in bulk_tissue[bp]],
                           device=DEVICE, dtype=torch.float32) / max(n_tissues - 1, 1)
    m_s_tis = torch.tensor([tissue_to_idx[t] for t in sn_tissue[sp]],
                           device=DEVICE, dtype=torch.float32) / max(n_tissues - 1, 1)
    m_b_tech = torch.from_numpy(bulk_tech[bp]).to(DEVICE)
    m_s_tech = torch.from_numpy(sn_tech[sp]).to(DEVICE)
    m_b_depth = torch.from_numpy(((bulk_depth[bp] - depth_mean) / depth_std).astype(np.float32)).to(DEVICE)
    m_s_depth = torch.from_numpy(((sn_depth[sp] - depth_mean) / depth_std).astype(np.float32)).to(DEVICE)
    model = _train_tech_vae(Xb_pair, Xs_pair, m_b_tis, m_s_tis, m_b_tech, m_s_tech,
                              m_b_depth, m_s_depth, bulk_x.shape[1],
                              bulk_z0=-3.0, sc_z0=3.0)
    print("  trained.")

    # Generate "synthetic bulk" by flipping sn -> bulk
    model.eval()
    sn_flipped = np.zeros_like(sn_x)
    chunk = 64
    with torch.no_grad():
        for i in range(0, len(sn_x), chunk):
            x = torch.from_numpy(sn_x[i:i+chunk]).to(DEVICE)
            mu_m, _, mu_b, _ = model.encode(x)
            z_m_flip = mu_m.clone()
            z_m_flip[:, 0] = -3.0
            td = model.tech_dim
            bulk_tech_id_t = torch.full((mu_m.size(0),), TECH_MAP["bulk_illumina"], dtype=torch.long, device=DEVICE)
            z_m_flip[:, 2:2+td] = model.tech_embed(bulk_tech_id_t)
            z_m_flip[:, 2+td] = m_b_depth.mean()
            sn_flipped[i:i+chunk] = model.decode(z_m_flip, mu_b).cpu().numpy()

    # === Downstream task: predict tissue from bulk ===
    # Leave-one-bulk-out CV over the 92 Eraslan bulk samples
    print("\n=== Downstream: predict tissue from bulk (LOO over 92 bulk samples) ===")
    y_bulk = np.array([tissue_to_idx[t] for t in bulk_tissue])
    y_sn = np.array([tissue_to_idx[t] for t in sn_tissue])
    n_bulk = len(bulk_x)

    conditions = {
        "A_real_bulk_only": {"X_extra": None, "y_extra": None},
        "B_real_bulk_plus_raw_sn": {"X_extra": sn_x, "y_extra": y_sn},
        "C_real_bulk_plus_flipped_sn": {"X_extra": sn_flipped, "y_extra": y_sn},
    }
    accuracies = {k: {"bal_acc": [], "top1": [], "top3": []} for k in conditions}

    # SPEED: project everything to PCA-50 once (fit on bulk only).
    print("\n  fitting PCA-50 for speed (projection only, no leak)…")
    pca = PCA(n_components=50).fit(bulk_x)
    bulk_pca = pca.transform(bulk_x)
    sn_pca = pca.transform(sn_x)
    sn_flipped_pca = pca.transform(sn_flipped)
    print(f"  bulk_pca {bulk_pca.shape}  sn_pca {sn_pca.shape}  sn_flipped_pca {sn_flipped_pca.shape}")

    conditions_pca = {
        "A_real_bulk_only": {"X_extra": None, "y_extra": None},
        "B_real_bulk_plus_raw_sn": {"X_extra": sn_pca, "y_extra": y_sn},
        "C_real_bulk_plus_flipped_sn": {"X_extra": sn_flipped_pca, "y_extra": y_sn},
    }

    for fold, held_out in enumerate(range(n_bulk)):
        train_mask = np.ones(n_bulk, dtype=bool); train_mask[held_out] = False
        X_train_real = bulk_pca[train_mask]
        y_train_real = y_bulk[train_mask]
        X_test = bulk_pca[held_out:held_out+1]
        y_test = y_bulk[held_out:held_out+1]

        for cond_name, cond in conditions_pca.items():
            if cond["X_extra"] is None:
                X_tr = X_train_real; y_tr = y_train_real
            else:
                X_tr = np.vstack([X_train_real, cond["X_extra"]])
                y_tr = np.concatenate([y_train_real, cond["y_extra"]])
            if len(np.unique(y_tr)) < 2: continue
            clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
            try:
                clf.fit(X_tr, y_tr)
                pred = clf.predict(X_test)
                proba = clf.predict_proba(X_test)
                accuracies[cond_name]["top1"].append(int(pred[0] == y_test[0]))
                top3 = np.argsort(-proba[0])[:3]
                accuracies[cond_name]["top3"].append(int(y_test[0] in top3))
            except Exception:
                pass
        if fold % 20 == 0:
            print(f"  fold {fold+1}/{n_bulk}", flush=True)

    print("\n=== Final accuracy comparison ===")
    print(f"{'condition':<32} {'top-1':>8} {'top-3':>8}  (n_train_bulk={n_bulk-1})")
    summary = {}
    for cond_name in conditions:
        top1 = float(np.mean(accuracies[cond_name]["top1"]))
        top3 = float(np.mean(accuracies[cond_name]["top3"]))
        summary[cond_name] = {"top1": top1, "top3": top3,
                              "n_train_extra": (0 if conditions[cond_name]["X_extra"] is None
                                                else len(conditions[cond_name]["X_extra"]))}
        print(f"  {cond_name:<32} {top1:>8.3f} {top3:>8.3f}")

    out = {"summary": summary, "n_bulk": int(n_bulk), "n_sn": int(len(sn_x)),
           "tissues": tissues}
    with (OUT_DIR / "q48_augmentation_benchmark.json").open("w") as f:
        json.dump(out, f, indent=2)
    print(f"saved → {OUT_DIR / 'q48_augmentation_benchmark.json'}")


if __name__ == "__main__":
    main()
