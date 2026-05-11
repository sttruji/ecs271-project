"""Q49 — Augmentation benchmark on AGE prediction (unsaturated downstream task).

Q48 showed bulk alone hits 100% on tissue classification — saturated. Try
AGE regression: harder, n=16 unique donors with 4 age bins, much less
saturable.

Same three conditions as Q48, predicting age_mid from PCA-50 bulk.
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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.q45_run27_tech_aware import TECH_MAP, _log_depth_proxy
from analysis.q47_metadata_probe import _train_tech_vae, get_eraslan_metadata

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

PAIRED_NPZ = "/Users/rls/ecs271/data/sc/eraslan/eraslan_paired.npz"
OUT_DIR = ROOT / "analysis" / "results" / "q20_disentangled"


def main():
    print(f"[Q49 age regression augmentation] device={DEVICE}")
    d = np.load(PAIRED_NPZ, allow_pickle=True)
    bulk_x = d["bulk_x"].astype(np.float32)
    bulk_donor = np.asarray(d["bulk_donor"], dtype=str)
    bulk_tissue = np.asarray(d["bulk_tissue"], dtype=str)
    sn_x = d["sn_x"].astype(np.float32)
    sn_donor = np.asarray(d["sn_donor"], dtype=str)
    sn_tissue = np.asarray(d["sn_tissue"], dtype=str)
    tissues = sorted(set(np.concatenate([bulk_tissue, sn_tissue])))
    tissue_to_idx = {t: i for i, t in enumerate(tissues)}

    md = get_eraslan_metadata().set_index("donor")
    age_map = md["age_mid"].to_dict()
    age_bulk = np.array([age_map.get(d) for d in bulk_donor], dtype=float)
    age_sn = np.array([age_map.get(d) for d in sn_donor], dtype=float)
    print(f"  bulk: {bulk_x.shape} (with age)  sn: {sn_x.shape}")

    # Train VAE
    bulk_tech = np.full(len(bulk_x), TECH_MAP["bulk_illumina"], dtype=np.int64)
    sn_tech = np.full(len(sn_x), TECH_MAP["10x_chromium"], dtype=np.int64)
    bulk_depth = _log_depth_proxy(bulk_x)
    sn_depth = _log_depth_proxy(sn_x)
    all_depth = np.concatenate([bulk_depth, sn_depth])
    depth_mean, depth_std = all_depth.mean(), all_depth.std() + 1e-8

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
                           device=DEVICE, dtype=torch.float32) / max(len(tissues) - 1, 1)
    m_s_tis = torch.tensor([tissue_to_idx[t] for t in sn_tissue[sp]],
                           device=DEVICE, dtype=torch.float32) / max(len(tissues) - 1, 1)
    m_b_tech = torch.from_numpy(bulk_tech[bp]).to(DEVICE)
    m_s_tech = torch.from_numpy(sn_tech[sp]).to(DEVICE)
    m_b_depth = torch.from_numpy(((bulk_depth[bp] - depth_mean) / depth_std).astype(np.float32)).to(DEVICE)
    m_s_depth = torch.from_numpy(((sn_depth[sp] - depth_mean) / depth_std).astype(np.float32)).to(DEVICE)
    print("\n  training VAE…")
    model = _train_tech_vae(Xb_pair, Xs_pair, m_b_tis, m_s_tis, m_b_tech, m_s_tech,
                              m_b_depth, m_s_depth, bulk_x.shape[1],
                              bulk_z0=-3.0, sc_z0=3.0, n_tissues=len(tissues))
    print("  trained.")

    model.eval()
    sn_flipped = np.zeros_like(sn_x)
    chunk = 64
    with torch.no_grad():
        for i in range(0, len(sn_x), chunk):
            x = torch.from_numpy(sn_x[i:i+chunk]).to(DEVICE)
            mu_m, _, mu_b, _ = model.encode(x)
            z_m_flip = mu_m.clone(); z_m_flip[:, 0] = -3.0
            td = model.tech_dim
            bulk_tech_id_t = torch.full((mu_m.size(0),), TECH_MAP["bulk_illumina"], dtype=torch.long, device=DEVICE)
            z_m_flip[:, 2:2+td] = model.tech_embed(bulk_tech_id_t)
            z_m_flip[:, 2+td] = m_b_depth.mean()
            sn_flipped[i:i+chunk] = model.decode(z_m_flip, mu_b).cpu().numpy()

    pca = PCA(n_components=50).fit(bulk_x)
    bulk_pca = pca.transform(bulk_x)
    sn_pca = pca.transform(sn_x)
    sn_flipped_pca = pca.transform(sn_flipped)

    # LOO over UNIQUE DONORS (not bulk samples) to avoid leak through tissue replicates
    unique_donors = sorted(set(bulk_donor))
    print(f"\n=== Age regression LOO over {len(unique_donors)} donors ===")
    age_donor = {d: age_map[d] for d in unique_donors}

    results = {"A_real_bulk": [], "B_plus_raw_sn": [], "C_plus_flipped_sn": []}
    for held_donor in unique_donors:
        # All test samples are this donor's bulk
        test_mask = bulk_donor == held_donor
        train_mask = ~test_mask
        X_train_real = bulk_pca[train_mask]
        y_train_real = age_bulk[train_mask]
        X_test = bulk_pca[test_mask]
        y_test = age_bulk[test_mask]
        if not np.all(np.isfinite(y_train_real)): continue

        # Held-out donor's sn samples are also held out from augmentation
        sn_train_mask = sn_donor != held_donor

        for cond_name, X_extra, y_extra in [
            ("A_real_bulk", None, None),
            ("B_plus_raw_sn", sn_pca[sn_train_mask], age_sn[sn_train_mask]),
            ("C_plus_flipped_sn", sn_flipped_pca[sn_train_mask], age_sn[sn_train_mask]),
        ]:
            if X_extra is None:
                X_tr = X_train_real; y_tr = y_train_real
            else:
                X_tr = np.vstack([X_train_real, X_extra])
                y_tr = np.concatenate([y_train_real, y_extra])
                valid = np.isfinite(y_tr)
                X_tr = X_tr[valid]; y_tr = y_tr[valid]
            reg = Ridge(alpha=1.0).fit(X_tr, y_tr)
            preds = reg.predict(X_test)
            results[cond_name].extend(list(zip(preds.tolist(), y_test.tolist())))

    print(f"{'condition':<28} {'R²':>8} {'MAE':>8}  n={sum(len(v) for v in results.values())//3}")
    summary = {}
    for c, preds_truths in results.items():
        preds = np.array([p for p, t in preds_truths])
        truths = np.array([t for p, t in preds_truths])
        r2 = r2_score(truths, preds)
        mae = mean_absolute_error(truths, preds)
        summary[c] = {"R2": float(r2), "MAE": float(mae)}
        print(f"  {c:<28} {r2:>+8.3f} {mae:>8.2f}")

    with (OUT_DIR / "q49_augmentation_age.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved → {OUT_DIR / 'q49_augmentation_age.json'}")


if __name__ == "__main__":
    main()
