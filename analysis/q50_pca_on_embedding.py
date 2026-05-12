"""Q50 — PCA on the learned z_full embedding. Does whitening help?

Hypothesis: z_full has supervised dims with very different variances
(modality ±5 vs tissue 0-1 vs tech ~1). Cosine and Ridge probes are
dominated by the high-variance dim. PCA on z_full across samples
equalizes variance (whitening), which should help any cosine/linear
downstream task.

Test: re-run Q47 metadata probes and Q21-style donor-NN test with
z_full vs PCA-projected z_full for k in {2,3,4,5,6,7}.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, r2_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import KFold
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


def _cosine(a, b):
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    print(f"[Q50 PCA on embedding] device={DEVICE}")
    d = np.load(PAIRED_NPZ, allow_pickle=True)
    bulk_x = d["bulk_x"].astype(np.float32)
    bulk_donor = np.asarray(d["bulk_donor"], dtype=str)
    bulk_tissue = np.asarray(d["bulk_tissue"], dtype=str)
    sn_x = d["sn_x"].astype(np.float32)
    sn_donor = np.asarray(d["sn_donor"], dtype=str)
    sn_tissue = np.asarray(d["sn_tissue"], dtype=str)
    tissues = sorted(set(np.concatenate([bulk_tissue, sn_tissue])))
    tissue_to_idx = {t: i for i, t in enumerate(tissues)}

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

    print("\n=== Training VAE ===")
    model = _train_tech_vae(Xb_pair, Xs_pair, m_b_tis, m_s_tis, m_b_tech, m_s_tech,
                              m_b_depth, m_s_depth, bulk_x.shape[1],
                              bulk_z0=-3.0, sc_z0=3.0, n_tissues=len(tissues))
    print("  trained.")

    # Embed all samples
    model.eval()
    z_full_bulk, z_bio_bulk = [], []
    z_full_sn, z_bio_sn = [], []
    chunk = 64
    with torch.no_grad():
        for i in range(0, len(bulk_x), chunk):
            x = torch.from_numpy(bulk_x[i:i+chunk]).to(DEVICE)
            mu_m, _, mu_b, _ = model.encode(x)
            z_full_bulk.append(torch.cat([mu_m, mu_b], dim=1).cpu().numpy())
            z_bio_bulk.append(mu_b.cpu().numpy())
        for i in range(0, len(sn_x), chunk):
            x = torch.from_numpy(sn_x[i:i+chunk]).to(DEVICE)
            mu_m, _, mu_b, _ = model.encode(x)
            z_full_sn.append(torch.cat([mu_m, mu_b], dim=1).cpu().numpy())
            z_bio_sn.append(mu_b.cpu().numpy())
    z_full_bulk = np.vstack(z_full_bulk); z_bio_bulk = np.vstack(z_bio_bulk)
    z_full_sn = np.vstack(z_full_sn); z_bio_sn = np.vstack(z_bio_sn)
    print(f"  z_full bulk: {z_full_bulk.shape}, z_bio: {z_bio_bulk.shape}")
    print(f"  z_full sn: {z_full_sn.shape}, z_bio: {z_bio_sn.shape}")

    # Show variance per dim
    z_full_all = np.vstack([z_full_bulk, z_full_sn])
    print(f"\n  z_full per-dim variance (across all {len(z_full_all)} samples):")
    print(f"  {[f'{v:.3f}' for v in z_full_all.var(axis=0)]}")
    print(f"  ratio max/min: {z_full_all.var(axis=0).max() / (z_full_all.var(axis=0).min() + 1e-9):.1f}x")

    # === DONOR-NN TEST: z_full vs PCA(z_full, k) for various k ===
    print(f"\n=== Donor-NN test on Eraslan paired LOO (per-tissue) ===")
    sn_idx_per_dt2 = {}
    for i, (dn, ts) in enumerate(zip(sn_donor, sn_tissue)):
        sn_idx_per_dt2.setdefault((dn, ts), []).append(i)
    bulk_idx_per_dt = {(dn, ts): i for i, (dn, ts) in enumerate(zip(bulk_donor, bulk_tissue))}
    paired_dts = [(dn, ts) for (dn, ts) in bulk_idx_per_dt if (dn, ts) in sn_idx_per_dt2]

    # Methods: raw z_full, PCA(z_full, k), z_bio, PCA(z_bio, k), whitened versions
    method_results = {}
    for k in [2, 3, 4, 5, 6, 7]:
        for variant in ["zfull", "zfull_pca", "zfull_pca_white", "zbio", "zbio_pca_white"]:
            if variant == "zfull" and k != 7: continue
            if variant == "zbio" and k != 5: continue
            if variant == "zfull_pca" and k > 7: continue
            if variant == "zfull_pca_white" and k > 7: continue
            if variant == "zbio_pca_white" and k > 5: continue
            method_results[f"{variant}_k{k}"] = {"top1": [], "top3": [], "rank1": []}

    for fold_idx, (held_dn, held_ts) in enumerate(paired_dts):
        held_bulk_i = bulk_idx_per_dt[(held_dn, held_ts)]
        held_sn_rows = set(sn_idx_per_dt2[(held_dn, held_ts)])

        # Restrict pool to held tissue
        pool_idx = np.where(sn_tissue == held_ts)[0]
        sn_donor_pool = sn_donor[pool_idx]

        # Fit PCA on TRAIN samples only (avoid leak)
        train_b = np.ones(len(z_full_bulk), dtype=bool); train_b[held_bulk_i] = False
        train_s = np.array([i not in held_sn_rows for i in range(len(z_full_sn))])
        z_full_train = np.vstack([z_full_bulk[train_b], z_full_sn[train_s]])
        z_bio_train = np.vstack([z_bio_bulk[train_b], z_bio_sn[train_s]])

        for variant_k, _ in method_results.items():
            variant, k_str = variant_k.rsplit("_k", 1); k = int(k_str)
            if variant == "zfull":
                q = z_full_bulk[held_bulk_i]
                pool_z = z_full_sn[pool_idx]
            elif variant == "zfull_pca":
                pca = PCA(n_components=k, whiten=False).fit(z_full_train)
                q = pca.transform(z_full_bulk[held_bulk_i:held_bulk_i+1])[0]
                pool_z = pca.transform(z_full_sn[pool_idx])
            elif variant == "zfull_pca_white":
                pca = PCA(n_components=k, whiten=True).fit(z_full_train)
                q = pca.transform(z_full_bulk[held_bulk_i:held_bulk_i+1])[0]
                pool_z = pca.transform(z_full_sn[pool_idx])
            elif variant == "zbio":
                q = z_bio_bulk[held_bulk_i]
                pool_z = z_bio_sn[pool_idx]
            elif variant == "zbio_pca_white":
                pca = PCA(n_components=k, whiten=True).fit(z_bio_train)
                q = pca.transform(z_bio_bulk[held_bulk_i:held_bulk_i+1])[0]
                pool_z = pca.transform(z_bio_sn[pool_idx])
            cos = np.array([_cosine(q, pool_z[i]) for i in range(len(pool_z))])
            order = np.argsort(-cos)
            ord_d = sn_donor_pool[order]
            rank = next(j for j, dn in enumerate(ord_d) if dn == held_dn) + 1
            method_results[variant_k]["rank1"].append(rank)
            method_results[variant_k]["top1"].append(float(np.mean(ord_d[:1] == held_dn)))
            method_results[variant_k]["top3"].append(float(np.mean(ord_d[:3] == held_dn)))

    print(f"\n{'method':<25} {'top-1':>7} {'top-3':>7} {'rank':>7}")
    print("-" * 55)
    summary = {}
    sorted_methods = sorted(method_results.items(),
                              key=lambda x: -np.mean(x[1]["top1"]))
    for m, vals in sorted_methods:
        t1 = float(np.mean(vals["top1"]))
        t3 = float(np.mean(vals["top3"]))
        rk = float(np.mean(vals["rank1"]))
        summary[m] = {"top1": t1, "top3": t3, "rank": rk}
        print(f"  {m:<25} {t1:>7.3f} {t3:>7.3f} {rk:>7.2f}")

    # === METADATA PROBE: same comparison ===
    print(f"\n=== Q47-style metadata probe (5-fold donor CV, n=16) ===")
    md = get_eraslan_metadata().set_index("donor")
    donors = list(md.index)
    age_map = {"21-40": 30.5, "41-50": 45.5, "51-60": 55.5, "61-70": 65.5}

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

    rep_zfull = agg(donors, z_full_bulk, z_full_sn, bulk_donor, sn_donor)
    rep_zbio = agg(donors, z_bio_bulk, z_bio_sn, bulk_donor, sn_donor)
    # PCA-whitened on training fold variation
    pca_zfull_w = PCA(n_components=7, whiten=True).fit(rep_zfull)
    rep_zfull_white = pca_zfull_w.transform(rep_zfull)
    pca_zbio_w = PCA(n_components=5, whiten=True).fit(rep_zbio)
    rep_zbio_white = pca_zbio_w.transform(rep_zbio)

    targets_def = {
        "sex_bin": ("binary", np.asarray(md["sex_bin"].values, dtype=float)),
        "age_mid": ("continuous", np.asarray(md["age_mid"].values, dtype=float)),
        "ischemia": ("continuous", np.asarray(md["ischemia"].values, dtype=float)),
        "rin_paxgene": ("continuous", np.asarray(md["rin_paxgene"].values, dtype=float)),
        "autolysis_bin": ("binary", np.asarray(md["autolysis_bin"].values, dtype=float)),
    }

    methods = {"z_full (7)": rep_zfull, "z_full_white (7)": rep_zfull_white,
               "z_bio (5)": rep_zbio, "z_bio_white (5)": rep_zbio_white}

    rng = np.random.default_rng(SEED)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    print(f"{'target':<14} {'kind':<11} {'method':<22} {'metric':>10}")
    print("-" * 60)
    probe_results = {}
    for target_name, (kind, y) in targets_def.items():
        valid = ~np.isnan(y); y_v = y[valid]
        for method_name, X in methods.items():
            X_v = X[valid]
            preds, truths, probs = [], [], []
            for train_idx, test_idx in kf.split(X_v):
                if kind == "continuous":
                    clf = RidgeCV(alphas=np.logspace(-3, 3, 7))
                    clf.fit(X_v[train_idx], y_v[train_idx])
                    p = clf.predict(X_v[test_idx])
                else:
                    if len(np.unique(y_v[train_idx])) < 2: continue
                    clf = LogisticRegressionCV(Cs=np.logspace(-2, 2, 5), max_iter=2000,
                                                class_weight="balanced")
                    clf.fit(X_v[train_idx], y_v[train_idx])
                    p = clf.predict(X_v[test_idx])
                    probs.extend(clf.predict_proba(X_v[test_idx])[:, 1] if clf.predict_proba(X_v[test_idx]).shape[1] == 2 else [0.5]*len(test_idx))
                preds.extend(p); truths.extend(y_v[test_idx])
            preds = np.asarray(preds); truths = np.asarray(truths)
            if kind == "continuous":
                r2 = r2_score(truths, preds)
                mae = mean_absolute_error(truths, preds)
                print(f"{target_name:<14} {kind:<11} {method_name:<22} R²:{r2:>+6.3f} MAE={mae:.2f}")
                probe_results[f"{target_name}__{method_name}"] = {"R2": float(r2), "MAE": float(mae)}
            else:
                bal = balanced_accuracy_score(truths, preds)
                if probs and len(set(truths.astype(int))) >= 2:
                    try:
                        auc = roc_auc_score(truths, probs)
                    except: auc = float("nan")
                else: auc = float("nan")
                print(f"{target_name:<14} {kind:<11} {method_name:<22} bal:{bal:.3f} AUC={auc:.3f}")
                probe_results[f"{target_name}__{method_name}"] = {
                    "bal_acc": float(bal), "AUC": float(auc) if not np.isnan(auc) else None}
        print()

    out = {"donor_nn": summary, "metadata_probe": probe_results,
           "z_full_var": z_full_all.var(axis=0).tolist()}
    with (OUT_DIR / "q50_pca_on_embedding.json").open("w") as f:
        json.dump(out, f, indent=2)
    print(f"saved → {OUT_DIR / 'q50_pca_on_embedding.json'}")


if __name__ == "__main__":
    main()
