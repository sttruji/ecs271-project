"""Q39 — Ensemble of all methods. Averaged-rank fusion.

For each method we have a per-query similarity score over the sn pool.
Convert to per-pool RANK (1=best NN, ... N=worst), then average ranks
across methods. Lowest-rank pool member is the final NN choice.

Methods to include (by best individual top-1 from Q35/Q36):
  cca_5:                  0.604
  pca50:                  0.510
  ftest_intersect_pca50:  0.510
  raw_cosine:             0.500
  spearman:               0.458
  pca_then_cca:           0.281 (skip, too noisy)
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAIRED_NPZ = "/Users/rls/ecs271/data/sc/eraslan/eraslan_paired.npz"
OUT_DIR = ROOT / "analysis" / "results" / "q20_disentangled"


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
    if len(uniq) < 2:
        return np.arange(X.shape[1])
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


def main():
    d = np.load(PAIRED_NPZ, allow_pickle=True)
    bulk_x = d["bulk_x"].astype(np.float32)
    bulk_donor = np.asarray(d["bulk_donor"], dtype=str)
    bulk_tissue = np.asarray(d["bulk_tissue"], dtype=str)
    sn_x = d["sn_x"].astype(np.float32)
    sn_donor = np.asarray(d["sn_donor"], dtype=str)
    sn_tissue = np.asarray(d["sn_tissue"], dtype=str)

    methods = ["raw_cosine", "pca50", "spearman", "ftest_int_pca50", "cca_5",
               "ENS_avg_rank_5", "ENS_avg_rank_top3"]
    summary = {}

    for tissue in sorted(set(bulk_tissue)):
        b_mask = bulk_tissue == tissue; s_mask = sn_tissue == tissue
        Xb_t = bulk_x[b_mask]; bd_t = bulk_donor[b_mask]
        Xs_t = sn_x[s_mask]; sd_t = sn_donor[s_mask]
        paired = sorted(set(bd_t) & set(sd_t))
        if len(paired) < 2: continue
        method_scores = {m: [] for m in methods}

        for held_dn in paired:
            held_idx = np.where(bd_t == held_dn)[0][0]
            x_query = Xb_t[held_idx]
            train_b_idx = np.array([i for i in range(len(Xb_t)) if i != held_idx])
            held_sn_idx = set(np.where(sd_t == held_dn)[0].tolist())
            train_s_idx = np.array([i for i in range(len(Xs_t)) if i not in held_sn_idx])

            # 1. raw_cosine
            s_raw = _cosine_matrix(x_query[None, :], Xs_t)[0]
            method_scores["raw_cosine"].append(s_raw)

            # 2. pca50
            n_comp = min(50, len(train_b_idx))
            pca = PCA(n_components=n_comp).fit(Xb_t[train_b_idx])
            s_pca = _cosine_matrix(pca.transform(x_query[None, :]), pca.transform(Xs_t))[0]
            method_scores["pca50"].append(s_pca)

            # 3. spearman
            s_sp = _spearman_matrix(x_query[None, :], Xs_t)[0]
            method_scores["spearman"].append(s_sp)

            # 4. ftest_intersect_pca50 (cross-tissue donor-discriminative gene set)
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

            # 5. CCA-5
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

            # ── ENSEMBLES via average rank ──
            # Convert each method's scores to ranks (lower rank = better, since we want argmax score)
            def to_ranks(s):
                # rank 1 = highest score
                return stats.rankdata(-s)
            r_raw = to_ranks(s_raw); r_pca = to_ranks(s_pca); r_sp = to_ranks(s_sp)
            r_ftpca = to_ranks(s_ftpca); r_cca = to_ranks(s_cca)
            ens5 = (r_raw + r_pca + r_sp + r_ftpca + r_cca) / 5
            method_scores["ENS_avg_rank_5"].append(-ens5)  # negate so argmax → smallest rank
            ens3 = (r_pca + r_ftpca + r_cca) / 3
            method_scores["ENS_avg_rank_top3"].append(-ens3)

        summary[tissue] = {"n_paired": len(paired), "n_pool": len(Xs_t)}
        for m in methods:
            scores = method_scores[m]
            if len(scores) == len(paired):
                summary[tissue][m] = _eval(np.stack(scores), sd_t, paired)

    # Print
    print(f"\n{'tissue':<22} {'n_d':>3} | " + " ".join(f"{m[:14]:>14}" for m in methods))
    print(f"{'top-1':<22} {'':>3}-+-" + "-+-".join("-" * 14 for _ in methods))
    for t, s in summary.items():
        row = f"{t:<22} {s['n_paired']:>3} | "
        row += " ".join(f"{s.get(m, {'mean_top1': 0})['mean_top1']:>14.3f}" for m in methods)
        print(row)
    print("\n" + "=" * 80)
    print(f"{'OVERALL top-1':<22} {'':>3} | " + " ".join(
        f"{np.mean([s.get(m, {'mean_top1': 0})['mean_top1'] for s in summary.values()]):>14.3f}" for m in methods))
    print(f"{'OVERALL top-3':<22} {'':>3} | " + " ".join(
        f"{np.mean([s.get(m, {'mean_top3': 0})['mean_top3'] for s in summary.values()]):>14.3f}" for m in methods))
    print(f"{'OVERALL top-5':<22} {'':>3} | " + " ".join(
        f"{np.mean([s.get(m, {'mean_top5': 0})['mean_top5'] for s in summary.values()]):>14.3f}" for m in methods))
    print(f"{'OVERALL rank1':<22} {'':>3} | " + " ".join(
        f"{np.mean([s.get(m, {'mean_rank_first_same': 0})['mean_rank_first_same'] for s in summary.values()]):>14.2f}" for m in methods))

    out = {"summary": summary,
           "overall": {m: {
               "mean_top1": float(np.mean([s.get(m, {"mean_top1": 0})["mean_top1"] for s in summary.values()])),
               "mean_top3": float(np.mean([s.get(m, {"mean_top3": 0})["mean_top3"] for s in summary.values()])),
               "mean_rank_first_same": float(np.mean([s.get(m, {"mean_rank_first_same": 0})["mean_rank_first_same"] for s in summary.values()])),
           } for m in methods}}
    with (OUT_DIR / "q39_ensemble.json").open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved → {OUT_DIR / 'q39_ensemble.json'}")


if __name__ == "__main__":
    main()
