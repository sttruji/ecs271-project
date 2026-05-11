"""Q42 — Exhaustive ensemble subset ablation on Q41 outputs.

Loads Q41's per-fold per-method scores from disk, tries all 2-7 subsets of
the 7 methods, computes overall top-1 / top-3 / mean-rank.

Reports the best subset and whether VAE methods help.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "analysis" / "results" / "q20_disentangled"

# Re-run the same Q41 logic but save intermediate per-fold scores so we can
# do exhaustive subset analysis without re-training.

# For now: just re-run Q41 with all subsets evaluated and report best.
# Faster: extract per-fold per-method scores from Q41 by re-evaluating.

# Easier path: since Q41 was deterministic and the scores stable, just run
# the same flow but compute all subset ensembles inline.

import warnings
warnings.filterwarnings("ignore")
import torch
from torch import nn
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from collections import Counter

from models.disentangled_vae import kl_with_free_bits, hsic_penalty
from analysis.q27_eraslan_run17_decompose import DecomposedVAE

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

PAIRED_NPZ = "/Users/rls/ecs271/data/sc/eraslan/eraslan_paired.npz"

VAE_EPOCHS = 200
LR = 1e-3
LAM_PAIRED = 5.0; LAM_LEAK = 0.3; LAM_CYCLE = 0.3
LAM_SUP = 1.0; LAM_DONOR_ID = 5.0; BETA_BIO = 1e-3
Z_BIO_DIM = 5

bulk_z0_target = -3.0; sc_z0_target = 3.0

METHOD_NAMES = ["raw_cosine", "pca50", "spearman", "ftest_int_pca50",
                "cca_5", "vae_flip", "vae_latent"]


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


def _train_tinyz_vae(Xb_pair, Xs_pair, m_b_tis, m_s_tis, n_genes):
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
        z_m_flip_b = z_m_b.clone(); z_m_flip_b[:, 0] = sc_z0_target
        x_flip_b2s = model.decode(z_m_flip_b, z_b_b)
        paired_b2s = nn.functional.mse_loss(x_flip_b2s, Xs_pair)
        z_m_flip_s = z_m_s.clone(); z_m_flip_s[:, 0] = bulk_z0_target
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
    print(f"bulk: {bulk_x.shape}  sn: {sn_x.shape}", flush=True)
    tissues = sorted(set(np.concatenate([bulk_tissue, sn_tissue])))
    tissue_to_idx = {t: i for i, t in enumerate(tissues)}

    # Collect per-fold per-method scores: scores_per_fold[(tissue, donor)] = {method: scores_array}
    all_fold_data = []  # list of (tissue, held_dn, sd_pool, scores_per_method)

    for tissue in sorted(set(bulk_tissue)):
        b_mask = bulk_tissue == tissue; s_mask = sn_tissue == tissue
        Xb_t = bulk_x[b_mask]; bd_t = bulk_donor[b_mask]
        Xs_t = sn_x[s_mask]; sd_t = sn_donor[s_mask]
        paired = sorted(set(bd_t) & set(sd_t))
        if len(paired) < 2: continue
        print(f"--- {tissue} ({len(paired)} donors, {len(Xs_t)} sn) ---", flush=True)

        for held_dn in paired:
            held_idx = np.where(bd_t == held_dn)[0][0]
            x_query = Xb_t[held_idx]
            train_b_idx = np.array([i for i in range(len(Xb_t)) if i != held_idx])
            held_sn_idx = set(np.where(sd_t == held_dn)[0].tolist())
            train_s_idx = np.array([i for i in range(len(Xs_t)) if i not in held_sn_idx])

            scores = {}
            scores["raw_cosine"] = _cosine_matrix(x_query[None, :], Xs_t)[0]
            n_comp = min(50, len(train_b_idx))
            pca = PCA(n_components=n_comp).fit(Xb_t[train_b_idx])
            scores["pca50"] = _cosine_matrix(pca.transform(x_query[None, :]), pca.transform(Xs_t))[0]
            scores["spearman"] = _spearman_matrix(x_query[None, :], Xs_t)[0]

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
            scores["ftest_int_pca50"] = _cosine_matrix(pca_c.transform(x_query[common][None, :]),
                                                       pca_c.transform(Xs_t[:, common]))[0]

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
                    scores["cca_5"] = _cosine_matrix(x_q_score, sn_score)[0]
                except Exception:
                    scores["cca_5"] = np.zeros(len(Xs_t))
            else:
                scores["cca_5"] = np.zeros(len(Xs_t))

            paired_pairs_all = []
            for ii in range(len(bulk_donor)):
                if (bulk_donor[ii], bulk_tissue[ii]) == (held_dn, tissue): continue
                for j in range(len(sn_donor)):
                    if j in (set(np.where((sn_donor == held_dn) & (sn_tissue == tissue))[0].tolist())):
                        continue
                    if sn_donor[j] == bulk_donor[ii] and sn_tissue[j] == bulk_tissue[ii]:
                        paired_pairs_all.append((ii, j))
            if len(paired_pairs_all) < 5:
                scores["vae_flip"] = np.zeros(len(Xs_t))
                scores["vae_latent"] = np.zeros(len(Xs_t))
            else:
                bp_idx = np.array([p[0] for p in paired_pairs_all])
                sp_idx = np.array([p[1] for p in paired_pairs_all])
                Xb_pair = torch.from_numpy(bulk_x[bp_idx]).to(DEVICE)
                Xs_pair = torch.from_numpy(sn_x[sp_idx]).to(DEVICE)
                m_b_tis = torch.tensor([tissue_to_idx[t] for t in bulk_tissue[bp_idx]],
                                       device=DEVICE, dtype=torch.float32) / max(len(tissues) - 1, 1)
                m_s_tis = torch.tensor([tissue_to_idx[t] for t in sn_tissue[sp_idx]],
                                       device=DEVICE, dtype=torch.float32) / max(len(tissues) - 1, 1)
                model = _train_tinyz_vae(Xb_pair, Xs_pair, m_b_tis, m_s_tis, bulk_x.shape[1])
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
                scores["vae_flip"] = _cosine_matrix(x_hat_sc[None, :], Xs_t)[0]
                scores["vae_latent"] = _cosine_matrix(z_held[None, :], z_sn_all)[0]

            all_fold_data.append((tissue, held_dn, sd_t.copy(), scores))
            print(f"  fold {tissue}/{held_dn} scored", flush=True)

    # Now exhaustively try all subsets of methods (size 2-7)
    print("\n=== Exhaustive subset ensemble search ===")
    all_subsets = []
    for k in range(2, 8):
        for combo in combinations(METHOD_NAMES, k):
            all_subsets.append(combo)
    print(f"  {len(all_subsets)} subsets of size 2-7")

    def to_ranks(s): return stats.rankdata(-s)

    subset_results = []
    for combo in all_subsets:
        all_top1, all_top3, all_rank1 = [], [], []
        for tissue, held_dn, sd_t, scores in all_fold_data:
            ranks_per_method = [to_ranks(scores[m]) for m in combo]
            ens = np.mean(ranks_per_method, axis=0)
            order = np.argsort(ens)  # smallest rank = best
            ord_d = sd_t[order]
            r = next(j for j, d in enumerate(ord_d) if d == held_dn) + 1
            all_rank1.append(r)
            all_top1.append(float(np.mean(ord_d[:1] == held_dn)))
            all_top3.append(float(np.mean(ord_d[:3] == held_dn)))
        subset_results.append({
            "methods": combo,
            "n_methods": len(combo),
            "top1": float(np.mean(all_top1)),
            "top3": float(np.mean(all_top3)),
            "rank1": float(np.mean(all_rank1)),
        })

    # Sort by top-1
    subset_results.sort(key=lambda r: -r["top1"])
    print(f"\n--- Top 15 ensembles by top-1 ---")
    for r in subset_results[:15]:
        has_vae = "VAE" if any("vae" in m for m in r["methods"]) else "   "
        print(f"  top-1={r['top1']:.3f}  top-3={r['top3']:.3f}  rank={r['rank1']:.2f}  "
              f"{has_vae}  k={r['n_methods']}  {','.join(m[:4] for m in r['methods'])}")
    # Best single method
    print(f"\n--- Single methods ---")
    for m in METHOD_NAMES:
        all_top1 = []
        for tissue, held_dn, sd_t, scores in all_fold_data:
            order = np.argsort(-scores[m])
            ord_d = sd_t[order]
            all_top1.append(float(np.mean(ord_d[:1] == held_dn)))
        print(f"  {m:<20} top-1={np.mean(all_top1):.3f}")

    out = {
        "top_subsets": subset_results[:30],
        "all_fold_data": [{"tissue": t, "held_dn": dn, "sd_t": list(sd.tolist()),
                            "scores": {m: list(s.tolist()) for m, s in sc.items()}}
                           for t, dn, sd, sc in all_fold_data],
    }
    with (OUT_DIR / "q42_subset_ablation.json").open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved → {OUT_DIR / 'q42_subset_ablation.json'}")


if __name__ == "__main__":
    main()
