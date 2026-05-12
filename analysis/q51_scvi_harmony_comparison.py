"""Q51 — Compare scVI + Harmony embeddings against ours on the SAME harness.

User wants to know: how well do popular off-the-shelf embeddings work
compared to our tech-aware VAE? Run the same Q50 evaluation:
  - donor-NN LOO test on Eraslan paired (per-tissue)
  - Q47-style metadata probes (Sex/Age/Ischemia/RIN/Autolysis)
on three baselines vs ours.

Baselines:
  1. scVI (scvi-tools 1.4) — n_latent=12, gene_likelihood='normal'
     (matches our data scale: log-CPM standardized, continuous)
  2. Harmony (harmonypy 2.0) — batch correction on PCA-50, batch=tech
  3. Raw PCA-50 (no correction) — sanity baseline

Ours:
  - z_full (12-D from Run 27 tech-aware VAE)
  - z_bio (5-D)
  - z_full_pca_white_k=5 (Q50 best variant for donor-NN)

Note: scVI is designed for raw counts; we feed log-CPM with normal
likelihood, which is the most natural drop-in for our preprocessing.
This is documented in scvi-tools as a supported mode for continuous data.
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
import scvi
import harmonypy
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, r2_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.q45_run27_tech_aware import TECH_MAP, _log_depth_proxy
from analysis.q47_metadata_probe import _train_tech_vae, get_eraslan_metadata

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
scvi.settings.seed = SEED

PAIRED_NPZ = "/Users/rls/ecs271/data/sc/eraslan/eraslan_paired.npz"
OUT_DIR = ROOT / "analysis" / "results" / "q20_disentangled"
N_LATENT = 12  # match our tech-aware VAE's z_full size


def _cosine(a, b):
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def train_scvi(bulk_x, sn_x, bulk_tech_str, sn_tech_str, n_latent=12, n_epochs=200):
    """Train scVI with normal likelihood on log-CPM data, batch=tech.

    Our data is standardized log-CPM (continuous, has negatives). scVI's
    default encoder applies log(1+x) which would produce NaN on negatives.
    Pass log_variational=False to skip the inner log; also set
    use_observed_lib_size=False since we don't have meaningful library sizes
    after standardization.
    """
    X = np.vstack([bulk_x, sn_x]).astype(np.float32)
    batch = np.concatenate([bulk_tech_str, sn_tech_str])
    obs = pd.DataFrame({"tech": batch})
    adata = ad.AnnData(X=X, obs=obs)
    scvi.model.SCVI.setup_anndata(adata, batch_key="tech")
    # gene_likelihood='normal' for continuous log-CPM
    # log_variational=False so encoder doesn't apply log(1+x) to negative-valued std-log-CPM
    model = scvi.model.SCVI(adata, n_latent=n_latent, n_hidden=128, n_layers=1,
                             gene_likelihood="normal", dropout_rate=0.1,
                             use_observed_lib_size=False, log_variational=False)
    print(f"  scVI: training {n_epochs} epochs on {X.shape} (n_latent={n_latent})…")
    model.train(max_epochs=n_epochs, batch_size=64, plan_kwargs={"lr": 1e-3},
                  early_stopping=False, check_val_every_n_epoch=50, accelerator="cpu")
    z = model.get_latent_representation()
    z_bulk = z[: len(bulk_x)]
    z_sn = z[len(bulk_x):]
    return z_bulk, z_sn


def run_harmony(bulk_x, sn_x, bulk_tech_str, sn_tech_str, n_pcs=50):
    """PCA-50 on combined data, then harmonypy batch correction by tech."""
    X = np.vstack([bulk_x, sn_x]).astype(np.float32)
    print(f"  Harmony: fitting PCA({n_pcs}) on {X.shape}…")
    pca = PCA(n_components=n_pcs).fit(X)
    pcs = pca.transform(X)
    batch = np.concatenate([bulk_tech_str, sn_tech_str])
    meta = pd.DataFrame({"tech": batch})
    print(f"  Harmony: running batch correction (batch=tech)…")
    ho = harmonypy.run_harmony(pcs, meta, vars_use=["tech"], max_iter_harmony=20,
                                 verbose=False)
    z = ho.Z_corr.T  # (n_samples, n_pcs)
    z_bulk = z[: len(bulk_x)]
    z_sn = z[len(bulk_x):]
    return z_bulk, z_sn, pcs[: len(bulk_x)], pcs[len(bulk_x):]


def main():
    print(f"[Q51 scVI + Harmony comparison] device={DEVICE}")
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
    bulk_tech_str = np.array(["bulk_illumina"] * len(bulk_x))
    sn_tech_str = np.array(["10x_chromium"] * len(sn_x))
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

    # =========================================================
    # 1. OUR VAE (Run 27 tech-aware)
    # =========================================================
    print("\n=== [1/3] Training our tech-aware VAE (Run 27) ===")
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
    model = _train_tech_vae(Xb_pair, Xs_pair, m_b_tis, m_s_tis, m_b_tech, m_s_tech,
                              m_b_depth, m_s_depth, bulk_x.shape[1],
                              bulk_z0=-3.0, sc_z0=3.0, n_tissues=len(tissues))
    model.eval()
    z_full_bulk, z_bio_bulk, z_full_sn, z_bio_sn = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(bulk_x), 64):
            x = torch.from_numpy(bulk_x[i:i+64]).to(DEVICE)
            mu_m, _, mu_b, _ = model.encode(x)
            z_full_bulk.append(torch.cat([mu_m, mu_b], dim=1).cpu().numpy())
            z_bio_bulk.append(mu_b.cpu().numpy())
        for i in range(0, len(sn_x), 64):
            x = torch.from_numpy(sn_x[i:i+64]).to(DEVICE)
            mu_m, _, mu_b, _ = model.encode(x)
            z_full_sn.append(torch.cat([mu_m, mu_b], dim=1).cpu().numpy())
            z_bio_sn.append(mu_b.cpu().numpy())
    z_full_bulk = np.vstack(z_full_bulk); z_bio_bulk = np.vstack(z_bio_bulk)
    z_full_sn = np.vstack(z_full_sn); z_bio_sn = np.vstack(z_bio_sn)
    print(f"  ours: z_full={z_full_bulk.shape[1]}, z_bio={z_bio_bulk.shape[1]}")

    # =========================================================
    # 2. scVI
    # =========================================================
    print("\n=== [2/3] Training scVI ===")
    z_scvi_bulk, z_scvi_sn = train_scvi(bulk_x, sn_x, bulk_tech_str, sn_tech_str,
                                          n_latent=N_LATENT, n_epochs=200)
    print(f"  scvi: z_scvi={z_scvi_bulk.shape[1]}")

    # =========================================================
    # 3. Harmony + raw PCA-50
    # =========================================================
    print("\n=== [3/3] Harmony + raw PCA-50 ===")
    z_harm_bulk, z_harm_sn, z_pca_bulk, z_pca_sn = run_harmony(
        bulk_x, sn_x, bulk_tech_str, sn_tech_str, n_pcs=50)
    print(f"  harmony: z_harm={z_harm_bulk.shape[1]}, raw_pca={z_pca_bulk.shape[1]}")

    # =========================================================
    # DONOR-NN TEST on all embeddings
    # =========================================================
    print(f"\n=== Donor-NN LOO test on Eraslan paired (per-tissue) ===")
    bulk_idx_per_dt = {(dn, ts): i for i, (dn, ts) in enumerate(zip(bulk_donor, bulk_tissue))}
    paired_dts = [(dn, ts) for (dn, ts) in bulk_idx_per_dt if (dn, ts) in sn_idx_per_dt]

    embeddings = {
        "ours_zfull (12)": (z_full_bulk, z_full_sn, None),
        "ours_zbio (5)": (z_bio_bulk, z_bio_sn, None),
        "ours_zfull_pca_white_k5": (z_full_bulk, z_full_sn, ("whiten", 5)),
        "scVI (12)": (z_scvi_bulk, z_scvi_sn, None),
        "scVI_white (12)": (z_scvi_bulk, z_scvi_sn, ("whiten", N_LATENT)),
        "harmony_pca50": (z_harm_bulk, z_harm_sn, None),
        "raw_pca50": (z_pca_bulk, z_pca_sn, None),
    }

    method_nn = {name: {"top1": [], "top3": [], "rank": []} for name in embeddings}

    for (held_dn, held_ts) in paired_dts:
        held_bulk_i = bulk_idx_per_dt[(held_dn, held_ts)]
        held_sn_rows = set(sn_idx_per_dt[(held_dn, held_ts)])
        pool_idx = np.where(sn_tissue == held_ts)[0]
        sn_donor_pool = sn_donor[pool_idx]

        for name, (zb, zs, transform) in embeddings.items():
            train_b = np.ones(len(zb), dtype=bool); train_b[held_bulk_i] = False
            train_s = np.array([i not in held_sn_rows for i in range(len(zs))])
            z_train = np.vstack([zb[train_b], zs[train_s]])
            if transform is None:
                q = zb[held_bulk_i]
                pool_z = zs[pool_idx]
            elif transform[0] == "whiten":
                k = transform[1]
                pca = PCA(n_components=k, whiten=True).fit(z_train)
                q = pca.transform(zb[held_bulk_i:held_bulk_i+1])[0]
                pool_z = pca.transform(zs[pool_idx])
            cos = np.array([_cosine(q, pool_z[i]) for i in range(len(pool_z))])
            order = np.argsort(-cos)
            ord_d = sn_donor_pool[order]
            rank = next(j for j, dn in enumerate(ord_d) if dn == held_dn) + 1
            method_nn[name]["rank"].append(rank)
            method_nn[name]["top1"].append(float(np.mean(ord_d[:1] == held_dn)))
            method_nn[name]["top3"].append(float(np.mean(ord_d[:3] == held_dn)))

    print(f"\n{'method':<30} {'top-1':>7} {'top-3':>7} {'rank':>7}")
    print("-" * 60)
    nn_summary = {}
    for m, vals in sorted(method_nn.items(), key=lambda x: -np.mean(x[1]["top1"])):
        t1 = float(np.mean(vals["top1"]))
        t3 = float(np.mean(vals["top3"]))
        rk = float(np.mean(vals["rank"]))
        nn_summary[m] = {"top1": t1, "top3": t3, "rank": rk}
        print(f"  {m:<30} {t1:>7.3f} {t3:>7.3f} {rk:>7.2f}")

    # =========================================================
    # METADATA PROBE on per-donor aggregated reps
    # =========================================================
    print(f"\n=== Q47-style metadata probe (5-fold donor CV, n=16) ===")
    md = get_eraslan_metadata().set_index("donor")
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

    rep_ours_zfull = agg(donors, z_full_bulk, z_full_sn, bulk_donor, sn_donor)
    rep_ours_zbio = agg(donors, z_bio_bulk, z_bio_sn, bulk_donor, sn_donor)
    rep_scvi = agg(donors, z_scvi_bulk, z_scvi_sn, bulk_donor, sn_donor)
    rep_harmony = agg(donors, z_harm_bulk, z_harm_sn, bulk_donor, sn_donor)
    rep_rawpca = agg(donors, z_pca_bulk, z_pca_sn, bulk_donor, sn_donor)

    targets_def = {
        "sex_bin": ("binary", np.asarray(md["sex_bin"].values, dtype=float)),
        "age_mid": ("continuous", np.asarray(md["age_mid"].values, dtype=float)),
        "ischemia": ("continuous", np.asarray(md["ischemia"].values, dtype=float)),
        "rin_paxgene": ("continuous", np.asarray(md["rin_paxgene"].values, dtype=float)),
        "autolysis_bin": ("binary", np.asarray(md["autolysis_bin"].values, dtype=float)),
    }
    methods = {
        "ours_zfull (12)": rep_ours_zfull,
        "ours_zbio (5)": rep_ours_zbio,
        "scVI (12)": rep_scvi,
        "harmony_pca50": rep_harmony,
        "raw_pca50": rep_rawpca,
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    print(f"{'target':<14} {'kind':<11} {'method':<24} {'metric':>10}")
    print("-" * 65)
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
                    pp = clf.predict_proba(X_v[test_idx])
                    probs.extend(pp[:, 1] if pp.shape[1] == 2 else [0.5]*len(test_idx))
                preds.extend(p); truths.extend(y_v[test_idx])
            preds = np.asarray(preds); truths = np.asarray(truths)
            if kind == "continuous":
                r2 = r2_score(truths, preds); mae = mean_absolute_error(truths, preds)
                print(f"{target_name:<14} {kind:<11} {method_name:<24} R²:{r2:>+6.3f} MAE={mae:.2f}")
                probe_results[f"{target_name}__{method_name}"] = {"R2": float(r2), "MAE": float(mae)}
            else:
                bal = balanced_accuracy_score(truths, preds)
                if probs and len(set(truths.astype(int))) >= 2:
                    try: auc = roc_auc_score(truths, probs)
                    except: auc = float("nan")
                else: auc = float("nan")
                print(f"{target_name:<14} {kind:<11} {method_name:<24} bal:{bal:.3f} AUC={auc:.3f}")
                probe_results[f"{target_name}__{method_name}"] = {
                    "bal_acc": float(bal), "AUC": float(auc) if not np.isnan(auc) else None}
        print()

    out = {"donor_nn": nn_summary, "metadata_probe": probe_results,
           "n_latent": N_LATENT}
    with (OUT_DIR / "q51_scvi_harmony_comparison.json").open("w") as f:
        json.dump(out, f, indent=2)
    print(f"saved → {OUT_DIR / 'q51_scvi_harmony_comparison.json'}")


if __name__ == "__main__":
    main()
