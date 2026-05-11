"""Q41 — Extended ensemble: all 5 Q39 methods + Run 23 FLIP + Run 23 LATENT.

Run 23's flip wins and latent wins are largely DIFFERENT folds from CCA-5 /
PCA-50 / etc. — so adding them as ensemble members should push above Q39's
0.615 ceiling.

This is a re-run of the ensemble logic plus a tiny-z VAE training step
inside the LOO loop.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy import stats
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.disentangled_vae import kl_with_free_bits, hsic_penalty
from analysis.q27_eraslan_run17_decompose import DecomposedVAE

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

PAIRED_NPZ = "/Users/rls/ecs271/data/sc/eraslan/eraslan_paired.npz"
OUT_DIR = ROOT / "analysis" / "results" / "q20_disentangled"

VAE_EPOCHS = 200
LR = 1e-3
LAM_PAIRED = 5.0; LAM_LEAK = 0.3; LAM_CYCLE = 0.3
LAM_SUP = 1.0; LAM_DONOR_ID = 5.0; BETA_BIO = 1e-3
Z_BIO_DIM = 5


def _cosine_matrix(A, B):
    A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A_n @ B_n.T


def _spearman_matrix(A, B):
    A_r = stats.rankdata(A, axis=1).astype(np.float32)
    B_r = stats.rankdata(B, axis=1).astype(np.float32)
    return _cosine_matrix(A_r - A_r.mean(axis=1, keepdims=True),
                          B_r - B_r.mean(axis=1, keepdims=True))


def _ftest_top_k(X, donor_labels, k=500):
    uniq = np.unique(donor_labels)
    if len(uniq) < 2: return np.arange(X.shape[1])
    grand_mean = X.mean(axis=0, keepdims=True)
    ss_b = np.zeros(X.shape[1]); ss_w = np.zeros(X.shape[1])
    for d in uniq:
        idx = np.where(np.array(donor_labels) == d)[0]
        if len(idx) < 1: continue
        gm = X[idx].mean(axis=0, keepdims=True)
        ss_b += len(idx) * ((gm - grand_mean) ** 2)[0]
        ss_w += ((X[idx] - gm) ** 2).sum(axis=0)
    df_b = max(len(uniq) - 1, 1); df_w = max(X.shape[0] - len(uniq), 1)
    f = (ss_b / df_b) / (ss_w / df_w + 1e-12)
    return np.argsort(-np.where(np.isfinite(f), f, 0.0))[:k]


def _eval(scores, sd_t, paired):
    ranks, topK = [], {1: [], 3: [], 5: [], 10: []}
    for i, dn in enumerate(paired):
        order = np.argsort(-scores[i])
        ord_d = sd_t[order]
        rank = next(j for j, d in enumerate(ord_d) if d == dn) + 1
        ranks.append(rank)
        for K in topK: topK[K].append(float(np.mean(ord_d[:K] == dn)))
    return {"mean_rank_first_same": float(np.mean(ranks)),
            "mean_top1": float(np.mean(topK[1])),
            "mean_top3": float(np.mean(topK[3])),
            "mean_top5": float(np.mean(topK[5])),
            "mean_top10": float(np.mean(topK[10]))}


def _train_tinyz_vae(Xb_pair, Xs_pair, m_b_tis, m_s_tis, n_genes, sc_z0, bulk_z0):
    model = DecomposedVAE(input_dim=n_genes, z_meta_dim=2, z_bio_dim=Z_BIO_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    m_b_mod = torch.zeros(Xb_pair.size(0), device=DEVICE)
    m_s_mod = torch.ones(Xs_pair.size(0), device=DEVICE)
    for ep in range(1, VAE_EPOCHS + 1):
        x_hat_b, mu_m_b, lv_m_b, mu_b_b, lv_b_b, z_m_b, z_b_b = model(Xb_pair)
        x_hat_s, mu_m_s, lv_m_s, mu_b_s, lv_b_s, z_m_s, z_b_s = model(Xs_pair)
        recon = nn.functional.mse_loss(x_hat_b, Xb_pair) + nn.functional.mse_loss(x_hat_s, Xs_pair)
        kl = (kl_with_free_bits(mu_b_b, lv_b_b, 0.5) + kl_with_free_bits(mu_b_s, lv_b_s, 0.5)) / Xb_pair.size(0)
        sup_mod = (nn.functional.binary_cross_entropy_with_logits(model.head_mod(z_m_b[:, 0:1]).squeeze(-1), m_b_mod)
                   + nn.functional.binary_cross_entropy_with_logits(model.head_mod(z_m_s[:, 0:1]).squeeze(-1), m_s_mod))
        sup_tis = (nn.functional.mse_loss(model.head_tis(z_m_b[:, 1:2]).squeeze(-1), m_b_tis)
                   + nn.functional.mse_loss(model.head_tis(z_m_s[:, 1:2]).squeeze(-1), m_s_tis))
        z_m_flip_b = z_m_b.clone(); z_m_flip_b[:, 0] = sc_z0
        x_flip_b2s = model.decode(z_m_flip_b, z_b_b)
        paired_b2s = nn.functional.mse_loss(x_flip_b2s, Xs_pair)
        z_m_flip_s = z_m_s.clone(); z_m_flip_s[:, 0] = bulk_z0
        x_flip_s2b = model.decode(z_m_flip_s, z_b_s)
        paired_s2b = nn.functional.mse_loss(x_flip_s2b, Xb_pair)
        mu_m_re_b, _, mu_b_re_b, _ = model.encode(x_flip_b2s)
        mu_m_re_s, _, mu_b_re_s, _ = model.encode(x_flip_s2b)
        cyc = nn.functional.mse_loss(mu_b_re_b, mu_b_b.detach()) + nn.functional.mse_loss(mu_b_re_s, mu_b_s.detach())
        mod_col = torch.cat([torch.full((Xb_pair.size(0),1), 0.0, device=DEVICE),
                             torch.full((Xs_pair.size(0),1), 1.0, device=DEVICE)])
        z_b_combined = torch.cat([z_b_b, z_b_s], dim=0)
        leak = hsic_penalty(z_b_combined, mod_col)
        donor_id = nn.functional.mse_loss(mu_b_b, mu_b_s)
        loss = (recon + BETA_BIO * kl + LAM_SUP * (sup_mod + sup_tis)
                + LAM_PAIRED * (paired_b2s + paired_s2b) + LAM_CYCLE * cyc
                + LAM_LEAK * leak + LAM_DONOR_ID * donor_id)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def main():
    d = np.load(PAIRED_NPZ, allow_pickle=True)
    bulk_x = d["bulk_x"].astype(np.float32)
    bulk_donor = np.asarray(d["bulk_donor"], dtype=str)
    bulk_tissue = np.asarray(d["bulk_tissue"], dtype=str)
    sn_x = d["sn_x"].astype(np.float32)
    sn_donor = np.asarray(d["sn_donor"], dtype=str)
    sn_tissue = np.asarray(d["sn_tissue"], dtype=str)
    print(f"bulk: {bulk_x.shape}  sn: {sn_x.shape}")
    tissues = sorted(set(np.concatenate([bulk_tissue, sn_tissue])))
    tissue_to_idx = {t: i for i, t in enumerate(tissues)}

    methods = ["raw_cosine", "pca50", "spearman", "ftest_int_pca50", "cca_5",
               "vae_flip", "vae_latent",
               "ENS_5_old", "ENS_7_all", "ENS_top4"]
    summary = {}

    bulk_z0_target = -3.0; sc_z0_target = 3.0

    for tissue in sorted(set(bulk_tissue)):
        b_mask = bulk_tissue == tissue; s_mask = sn_tissue == tissue
        Xb_t = bulk_x[b_mask]; bd_t = bulk_donor[b_mask]
        Xs_t = sn_x[s_mask]; sd_t = sn_donor[s_mask]
        paired = sorted(set(bd_t) & set(sd_t))
        if len(paired) < 2: continue
        method_scores = {m: [] for m in methods}
        print(f"\n--- {tissue} ({len(paired)} paired donors, {len(Xs_t)} sn pool) ---", flush=True)

        for held_dn in paired:
            held_idx = np.where(bd_t == held_dn)[0][0]
            x_query = Xb_t[held_idx]
            train_b_idx = np.array([i for i in range(len(Xb_t)) if i != held_idx])
            held_sn_idx = set(np.where(sd_t == held_dn)[0].tolist())
            train_s_idx = np.array([i for i in range(len(Xs_t)) if i not in held_sn_idx])

            # Standard scores
            s_raw = _cosine_matrix(x_query[None, :], Xs_t)[0]
            method_scores["raw_cosine"].append(s_raw)
            n_comp = min(50, len(train_b_idx))
            pca = PCA(n_components=n_comp).fit(Xb_t[train_b_idx])
            s_pca = _cosine_matrix(pca.transform(x_query[None, :]), pca.transform(Xs_t))[0]
            method_scores["pca50"].append(s_pca)
            s_sp = _spearman_matrix(x_query[None, :], Xs_t)[0]
            method_scores["spearman"].append(s_sp)

            from collections import Counter
            cross_sn = sn_x[sn_tissue != tissue]
            cross_donor = sn_donor[sn_tissue != tissue]
            cross_tis = sn_tissue[sn_tissue != tissue]
            counter = Counter()
            for ts in set(cross_tis):
                m = cross_tis == ts
                if m.sum() < 4: continue
                top_k = _ftest_top_k(cross_sn[m], cross_donor[m], k=500)
                for g in top_k: counter[g] += 1
            common = np.array([g for g, c in counter.items() if c >= 2]) \
                     if any(c >= 2 for c in counter.values()) else np.arange(min(500, bulk_x.shape[1]))
            n_comp = min(50, len(train_b_idx), len(common))
            pca_c = PCA(n_components=n_comp).fit(Xb_t[train_b_idx][:, common])
            s_ftpca = _cosine_matrix(pca_c.transform(x_query[common][None, :]),
                                      pca_c.transform(Xs_t[:, common]))[0]
            method_scores["ftest_int_pca50"].append(s_ftpca)

            # CCA-5
            train_X, train_Y = [], []
            for bi in train_b_idx:
                dn = bd_t[bi]
                same_sn = [j for j in train_s_idx if sd_t[j] == dn]
                for j in same_sn:
                    train_X.append(Xb_t[bi]); train_Y.append(Xs_t[j])
            if len(train_X) >= 5:
                X_t = np.stack(train_X); Y_t = np.stack(train_Y)
                n_comp = min(5, len(train_X) - 1, X_t.shape[1])
                try:
                    cca = CCA(n_components=n_comp, max_iter=500)
                    cca.fit(X_t, Y_t)
                    x_q_score, _ = cca.transform(x_query[None, :], np.zeros((1, Y_t.shape[1])))
                    _, sn_score = cca.transform(np.zeros((Xs_t.shape[0], X_t.shape[1])), Xs_t)
                    s_cca = _cosine_matrix(x_q_score, sn_score)[0]
                except Exception:
                    s_cca = np.zeros(len(Xs_t))
            else:
                s_cca = np.zeros(len(Xs_t))
            method_scores["cca_5"].append(s_cca)

            # === VAE: train per-fold using ALL paired data across ALL tissues ===
            # (matches Q40's setup; per-tissue VAE would be too small)
            paired_pairs_all = []
            for ii in range(len(bulk_donor)):
                if (bulk_donor[ii], bulk_tissue[ii]) == (held_dn, tissue): continue
                # find sn samples with same (donor, tissue)
                for j in range(len(sn_donor)):
                    if j in (set(np.where((sn_donor == held_dn) & (sn_tissue == tissue))[0].tolist())):
                        continue
                    if sn_donor[j] == bulk_donor[ii] and sn_tissue[j] == bulk_tissue[ii]:
                        paired_pairs_all.append((ii, j))
            if len(paired_pairs_all) < 5:
                method_scores["vae_flip"].append(np.zeros(len(Xs_t)))
                method_scores["vae_latent"].append(np.zeros(len(Xs_t)))
            else:
                bp_idx = np.array([p[0] for p in paired_pairs_all])
                sp_idx = np.array([p[1] for p in paired_pairs_all])
                Xb_pair = torch.from_numpy(bulk_x[bp_idx]).to(DEVICE)
                Xs_pair = torch.from_numpy(sn_x[sp_idx]).to(DEVICE)
                m_b_tis = torch.tensor([tissue_to_idx[t] for t in bulk_tissue[bp_idx]],
                                       device=DEVICE, dtype=torch.float32) / max(len(tissues) - 1, 1)
                m_s_tis = torch.tensor([tissue_to_idx[t] for t in sn_tissue[sp_idx]],
                                       device=DEVICE, dtype=torch.float32) / max(len(tissues) - 1, 1)
                model = _train_tinyz_vae(Xb_pair, Xs_pair, m_b_tis, m_s_tis,
                                          bulk_x.shape[1], sc_z0_target, bulk_z0_target)
                model.eval()
                with torch.no_grad():
                    x_held_b = torch.from_numpy(x_query[None, :]).to(DEVICE)
                    mu_m_h, _, mu_b_h, _ = model.encode(x_held_b)
                    z_m_flip = mu_m_h.clone(); z_m_flip[:, 0] = sc_z0_target
                    x_hat_sc = model.decode(z_m_flip, mu_b_h).cpu().numpy()[0]
                    sn_full = torch.from_numpy(Xs_t).to(DEVICE)
                    _, _, mu_b_sn_full, _ = model.encode(sn_full)
                    z_held = mu_b_h.cpu().numpy()[0]
                    z_sn_all = mu_b_sn_full.cpu().numpy()
                s_flip = np.array([np.dot(x_hat_sc, Xs_t[i]) /
                                    (np.linalg.norm(x_hat_sc) * np.linalg.norm(Xs_t[i]) + 1e-12)
                                    for i in range(len(Xs_t))])
                s_lat = np.array([np.dot(z_held, z_sn_all[i]) /
                                   (np.linalg.norm(z_held) * np.linalg.norm(z_sn_all[i]) + 1e-12)
                                   for i in range(len(Xs_t))])
                method_scores["vae_flip"].append(s_flip)
                method_scores["vae_latent"].append(s_lat)

            print(f"    fold held={held_dn}: scoring done", flush=True)

            # ENSEMBLES via average rank
            def to_ranks(s): return stats.rankdata(-s)
            r_raw, r_pca, r_sp = to_ranks(s_raw), to_ranks(s_pca), to_ranks(s_sp)
            r_ftpca, r_cca = to_ranks(s_ftpca), to_ranks(s_cca)
            r_vflip = to_ranks(method_scores["vae_flip"][-1])
            r_vlat = to_ranks(method_scores["vae_latent"][-1])
            ens5 = (r_raw + r_pca + r_sp + r_ftpca + r_cca) / 5
            ens7 = (r_raw + r_pca + r_sp + r_ftpca + r_cca + r_vflip + r_vlat) / 7
            ens_top4 = (r_pca + r_ftpca + r_cca + r_vlat) / 4
            method_scores["ENS_5_old"].append(-ens5)
            method_scores["ENS_7_all"].append(-ens7)
            method_scores["ENS_top4"].append(-ens_top4)

        summary[tissue] = {"n_paired": len(paired), "n_pool": len(Xs_t)}
        for m in methods:
            if len(method_scores[m]) == len(paired):
                summary[tissue][m] = _eval(np.stack(method_scores[m]), sd_t, paired)

    # Print
    print(f"\n{'tissue':<22} {'n':>2} | " + " ".join(f"{m[:11]:>11}" for m in methods))
    print(f"{'top-1':<22} {'':>2}-+-" + "-+-".join("-" * 11 for _ in methods))
    for t, s in summary.items():
        row = f"{t:<22} {s['n_paired']:>2} | "
        row += " ".join(f"{s.get(m, {'mean_top1': 0})['mean_top1']:>11.3f}" for m in methods)
        print(row)
    print("\n" + "=" * 100)
    print(f"{'OVERALL top-1':<22} {'':>2} | " + " ".join(
        f"{np.mean([s.get(m, {'mean_top1': 0})['mean_top1'] for s in summary.values()]):>11.3f}" for m in methods))
    print(f"{'OVERALL top-3':<22} {'':>2} | " + " ".join(
        f"{np.mean([s.get(m, {'mean_top3': 0})['mean_top3'] for s in summary.values()]):>11.3f}" for m in methods))
    print(f"{'OVERALL rank1':<22} {'':>2} | " + " ".join(
        f"{np.mean([s.get(m, {'mean_rank_first_same': 0})['mean_rank_first_same'] for s in summary.values()]):>11.2f}" for m in methods))

    out = {"summary": summary,
           "overall": {m: {
               "mean_top1": float(np.mean([s.get(m, {"mean_top1": 0})["mean_top1"] for s in summary.values()])),
               "mean_top3": float(np.mean([s.get(m, {"mean_top3": 0})["mean_top3"] for s in summary.values()])),
               "mean_rank_first_same": float(np.mean([s.get(m, {"mean_rank_first_same": 0})["mean_rank_first_same"] for s in summary.values()])),
           } for m in methods}}
    with (OUT_DIR / "q41_ensemble_extended.json").open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved → {OUT_DIR / 'q41_ensemble_extended.json'}")


if __name__ == "__main__":
    main()
