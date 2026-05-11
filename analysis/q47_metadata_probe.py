"""Q47 — Metadata prediction probes on VAE embeddings.

Tests whether the trained AdditiveVAE z_bio embedding contains biological
signal by predicting donor-level metadata that the model never saw.

Targets (16 Eraslan donors, joined from h5ad obs):
  - Sex (binary)
  - Age_bin -> midpoint (continuous regression, range 21-70)
  - Sample Ischemic Time (continuous, range 129-825 min)
  - RIN score from PAXgene (continuous, range 5.9-9.6)
  - Autolysis (binary None vs Mild)

Predictors:
  - raw_mean         per-donor mean of 11,374 standardized log-CPM
  - pca50            per-donor mean PCA-50 score
  - z_bio (5-D)      per-donor mean of TechAwareVAE z_bio
  - z_full (7-D)     per-donor mean of full latent (z_meta + z_bio)
  - z_bio_no_tech (5-D) AdditiveVAE z_bio (Run 26 architecture - simpler model)

Protocol: 5-fold donor CV; Ridge/Logistic with cross-validated alpha; report
R^2 (continuous) or balanced accuracy + AUC (binary).

Inspired by:
  - scVI/scANVI's frozen-embedding linear-probe convention
  - scIB benchmark protocol for evaluating cross-modality embeddings
  - Q12 of past work (PCA-50 / Sane-VAE / CM-VAE probes on GTEx)
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import anndata as ad
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             roc_auc_score, r2_score, mean_absolute_error)
from sklearn.model_selection import KFold
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.disentangled_vae import kl_with_free_bits, hsic_penalty
from analysis.q45_run27_tech_aware import TechAwareVAE, info_nce, TECH_MAP, _log_depth_proxy

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

PAIRED_NPZ = "/Users/rls/ecs271/data/sc/eraslan/eraslan_paired.npz"
ERASLAN_H5AD = "/Users/rls/ecs271/data/sc/eraslan/GTEx_8_tissues_snRNAseq_atlas.h5ad"
OUT_DIR = ROOT / "analysis" / "results" / "q20_disentangled"

EPOCHS = 200
LR = 1e-3
LAM_PAIRED = 5.0; LAM_LEAK = 0.3; LAM_CYCLE = 0.3
LAM_SUP = 1.0; LAM_DONOR_ID = 5.0
LAM_NCE_LATENT = 5.0; NCE_TAU = 0.1
BETA_BIO = 1e-3
Z_BIO_DIM = 5


def _train_tech_vae(Xb_pair, Xs_pair, m_b_tis, m_s_tis, m_b_tech, m_s_tech,
                     m_b_depth, m_s_depth, n_genes, bulk_z0, sc_z0,
                     n_tissues):
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
        if ep % 50 == 0 or ep == 1:
            print(f"    ep{ep:3d}  loss={loss.item():.2f}  recon={recon.item():.2f}", flush=True)
    return model


def get_eraslan_metadata():
    a = ad.read_h5ad(ERASLAN_H5AD, backed="r")
    md = a.obs.groupby("Participant ID", observed=False).agg({
        "Age_bin": "first", "Sex": "first",
        "Sample Ischemic Time (mins)": "first",
        "RIN score from PAXgene tissue Aliquot": "first",
        "RIN score from Frozen tissue Aliquot": "first",
        "Autolysis Score": "first",
    })
    md = md.reset_index()
    md.columns = ["donor", "age_bin", "sex", "ischemia", "rin_paxgene",
                  "rin_frozen", "autolysis"]
    # Numeric conversions
    age_map = {"21-40": 30.5, "41-50": 45.5, "51-60": 55.5, "61-70": 65.5}
    md["age_mid"] = md["age_bin"].map(lambda x: age_map.get(str(x), np.nan))
    md["sex_bin"] = md["sex"].map({"Male": 0, "Female": 1})
    md["autolysis_bin"] = md["autolysis"].map(
        {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}).fillna(np.nan)
    # numeric ischemia + RIN
    for c in ["ischemia", "rin_paxgene", "rin_frozen"]:
        md[c] = pd.to_numeric(md[c], errors="coerce")
    return md


def main():
    print(f"[Q47 metadata probe] device={DEVICE}")
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

    bulk_tech = np.full(len(bulk_x), TECH_MAP["bulk_illumina"], dtype=np.int64)
    sn_tech = np.full(len(sn_x), TECH_MAP["10x_chromium"], dtype=np.int64)
    bulk_depth = _log_depth_proxy(bulk_x)
    sn_depth = _log_depth_proxy(sn_x)
    all_depth = np.concatenate([bulk_depth, sn_depth])
    depth_mean, depth_std = all_depth.mean(), all_depth.std() + 1e-8

    # Build training paired set (all 256 pairs, no LOO held out)
    sn_idx_per_dt = {}
    for i, (dn, ts) in enumerate(zip(sn_donor, sn_tissue)):
        sn_idx_per_dt.setdefault((dn, ts), []).append(i)
    paired_pairs = []
    for i, (dn, ts) in enumerate(zip(bulk_donor, bulk_tissue)):
        for j in sn_idx_per_dt.get((dn, ts), []):
            paired_pairs.append((i, j))
    print(f"  paired pairs: {len(paired_pairs)}")
    bp = np.array([p[0] for p in paired_pairs])
    sp = np.array([p[1] for p in paired_pairs])
    Xb_pair = torch.from_numpy(bulk_x[bp]).to(DEVICE)
    Xs_pair = torch.from_numpy(sn_x[sp]).to(DEVICE)
    m_b_tis = torch.tensor([tissue_to_idx[t] for t in bulk_tissue[bp]],
                           device=DEVICE, dtype=torch.float32) / max(len(tissues) - 1, 1)
    m_s_tis = torch.tensor([tissue_to_idx[t] for t in sn_tissue[sp]],
                           device=DEVICE, dtype=torch.float32) / max(len(tissues) - 1, 1)
    m_b_tech = torch.from_numpy(bulk_tech[bp]).to(DEVICE)
    m_s_tech = torch.from_numpy(sn_tech[sp]).to(DEVICE)
    m_b_depth = torch.from_numpy(((bulk_depth[bp] - depth_mean) / depth_std).astype(np.float32)).to(DEVICE)
    m_s_depth = torch.from_numpy(((sn_depth[sp] - depth_mean) / depth_std).astype(np.float32)).to(DEVICE)

    print("\n=== Training TechAwareVAE on all paired data ===")
    model = _train_tech_vae(Xb_pair, Xs_pair, m_b_tis, m_s_tis, m_b_tech, m_s_tech,
                              m_b_depth, m_s_depth, bulk_x.shape[1],
                              bulk_z0=-3.0, sc_z0=3.0, n_tissues=len(tissues))
    print("  trained.")

    # === Embed every sample ===
    model.eval()
    z_bio_bulk, z_full_bulk = [], []
    z_bio_sn, z_full_sn = [], []
    chunk = 64
    with torch.no_grad():
        for i in range(0, len(bulk_x), chunk):
            x = torch.from_numpy(bulk_x[i:i+chunk]).to(DEVICE)
            mu_m, _, mu_b, _ = model.encode(x)
            z_bio_bulk.append(mu_b.cpu().numpy())
            z_full_bulk.append(torch.cat([mu_m, mu_b], dim=1).cpu().numpy())
        for i in range(0, len(sn_x), chunk):
            x = torch.from_numpy(sn_x[i:i+chunk]).to(DEVICE)
            mu_m, _, mu_b, _ = model.encode(x)
            z_bio_sn.append(mu_b.cpu().numpy())
            z_full_sn.append(torch.cat([mu_m, mu_b], dim=1).cpu().numpy())
    z_bio_bulk = np.vstack(z_bio_bulk); z_full_bulk = np.vstack(z_full_bulk)
    z_bio_sn = np.vstack(z_bio_sn); z_full_sn = np.vstack(z_full_sn)
    print(f"  z_bio bulk: {z_bio_bulk.shape}  sn: {z_bio_sn.shape}")
    print(f"  z_full bulk: {z_full_bulk.shape}  sn: {z_full_sn.shape}")

    # PCA-50 baseline on bulk; project both bulk and sn
    pca = PCA(n_components=min(50, len(bulk_x) - 1)).fit(bulk_x)
    pca_bulk = pca.transform(bulk_x)
    pca_sn = pca.transform(sn_x)

    # Aggregate per-donor: average across all that donor's samples (bulk + sn)
    md = get_eraslan_metadata()
    md = md.set_index("donor")
    donors = list(md.index)

    def agg(donor_list, x_bulk, x_sn, d_bulk, d_sn):
        out = []
        for d in donor_list:
            rows_b = x_bulk[d_bulk == d]
            rows_s = x_sn[d_sn == d]
            if len(rows_b) and len(rows_s):
                v = np.concatenate([rows_b, rows_s], axis=0).mean(axis=0)
            elif len(rows_b):
                v = rows_b.mean(axis=0)
            elif len(rows_s):
                v = rows_s.mean(axis=0)
            else:
                v = np.zeros(x_bulk.shape[1] if len(x_bulk) else x_sn.shape[1])
            out.append(v)
        return np.stack(out)

    print("\n=== Aggregating per-donor representations ===")
    rep_raw = agg(donors, bulk_x, sn_x, bulk_donor, sn_donor)
    rep_pca = agg(donors, pca_bulk, pca_sn, bulk_donor, sn_donor)
    rep_zbio = agg(donors, z_bio_bulk, z_bio_sn, bulk_donor, sn_donor)
    rep_zfull = agg(donors, z_full_bulk, z_full_sn, bulk_donor, sn_donor)
    print(f"  per-donor raw: {rep_raw.shape}, pca50: {rep_pca.shape}, "
          f"z_bio: {rep_zbio.shape}, z_full: {rep_zfull.shape}")

    targets = {
        "sex_bin": ("binary", np.asarray(md["sex_bin"].values, dtype=float)),
        "age_mid": ("continuous", np.asarray(md["age_mid"].values, dtype=float)),
        "ischemia": ("continuous", np.asarray(md["ischemia"].values, dtype=float)),
        "rin_paxgene": ("continuous", np.asarray(md["rin_paxgene"].values, dtype=float)),
        "autolysis_bin": ("binary", np.asarray(md["autolysis_bin"].values, dtype=float)),
    }
    methods = {"raw_mean": rep_raw, "pca50": rep_pca,
               "z_bio (5-D)": rep_zbio, "z_full (7-D)": rep_zfull}

    results = {}
    print("\n=== Linear probe results (5-fold donor CV, n=16) ===")
    print(f"{'target':<14} {'kind':<11} {'method':<14} {'metric':>8}  {'val':>6}")
    print("-" * 65)
    rng = np.random.default_rng(SEED)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for target_name, (kind, y) in targets.items():
        valid = ~np.isnan(y)
        y_v = y[valid]
        n_classes = len(np.unique(y_v.astype(int))) if kind == "binary" else None
        for method_name, X in methods.items():
            X_v = X[valid]
            try:
                preds, truths, probs = [], [], []
                for train_idx, test_idx in kf.split(X_v):
                    if kind == "continuous":
                        clf = RidgeCV(alphas=np.logspace(-3, 3, 7))
                        clf.fit(X_v[train_idx], y_v[train_idx])
                        p = clf.predict(X_v[test_idx])
                    else:
                        if len(np.unique(y_v[train_idx])) < 2:
                            continue
                        clf = LogisticRegressionCV(
                            Cs=np.logspace(-2, 2, 5), max_iter=2000,
                            class_weight="balanced")
                        clf.fit(X_v[train_idx], y_v[train_idx])
                        p = clf.predict(X_v[test_idx])
                        if n_classes == 2:
                            probs.extend(clf.predict_proba(X_v[test_idx])[:, 1])
                    preds.extend(p); truths.extend(y_v[test_idx])
                preds = np.asarray(preds); truths = np.asarray(truths)
                if kind == "continuous":
                    r2 = r2_score(truths, preds)
                    mae = mean_absolute_error(truths, preds)
                    print(f"{target_name:<14} {kind:<11} {method_name:<14} R²: {r2:>+6.3f}  (MAE={mae:.2f})")
                    results[f"{target_name}__{method_name}"] = {"R2": float(r2), "MAE": float(mae)}
                else:
                    bal = balanced_accuracy_score(truths, preds)
                    if probs and n_classes == 2 and len(set(truths.astype(int))) >= 2:
                        auc = roc_auc_score(truths, probs)
                    else:
                        auc = float("nan")
                    print(f"{target_name:<14} {kind:<11} {method_name:<14} bal_acc:{bal:>5.3f}  AUC={auc:.3f}")
                    results[f"{target_name}__{method_name}"] = {"bal_acc": float(bal), "AUC": float(auc) if not np.isnan(auc) else None}
            except Exception as e:
                print(f"{target_name:<14} {kind:<11} {method_name:<14}  FAILED: {e}")
        print()

    out = {"n_donors": len(donors), "donors": list(donors), "results": results,
           "config": {"epochs": EPOCHS, "z_bio_dim": Z_BIO_DIM}}
    with (OUT_DIR / "q47_metadata_probe.json").open("w") as f:
        json.dump(out, f, indent=2)
    print(f"saved → {OUT_DIR / 'q47_metadata_probe.json'}")


if __name__ == "__main__":
    main()
