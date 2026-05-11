"""Q35 — Extended no-training baselines on Eraslan paired Test 2.

Methods (all per-tissue LOO):
  raw_cosine     baseline #1: cosine on raw 11,374 standardized log-CPM
  pca50          baseline #2: cosine on PCA-50 of training bulk
  ftest_500      F-test selects top-500 donor-discriminative genes using
                 ALL training-sn samples (donor as factor); cosine on
                 those 500. NOT used at inference for production — this
                 is a validation that donor signal is recoverable when
                 you focus on the right features.
  ftest_pca50    F-test top-500 → PCA-50 (all-genes-PCA but on the
                 donor-discriminative subset)
  spearman       per-tissue Spearman correlation (rank-cosine)

Reports per-tissue and overall top-K and rank-of-first-same.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
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
    """Vectorized per-gene F-statistic with donor as factor. Returns top-k gene indices.

    F = (MS_between / MS_within) where
      MS_between = Σ_d n_d (mean_d - mean_total)^2 / (k - 1)
      MS_within  = Σ_d Σ_{i in d} (X_i - mean_d)^2 / (N - k)
    """
    uniq = np.unique(donor_labels)
    n_groups = len(uniq)
    if n_groups < 2:
        return np.arange(X.shape[1])
    N = X.shape[0]
    grand_mean = X.mean(axis=0, keepdims=True)
    ss_between = np.zeros(X.shape[1])
    ss_within = np.zeros(X.shape[1])
    for d in uniq:
        idx = np.where(np.array(donor_labels) == d)[0]
        if len(idx) < 1: continue
        gm = X[idx].mean(axis=0, keepdims=True)
        ss_between += len(idx) * ((gm - grand_mean) ** 2)[0]
        ss_within += ((X[idx] - gm) ** 2).sum(axis=0)
    df_b = max(n_groups - 1, 1)
    df_w = max(N - n_groups, 1)
    f_stats = (ss_between / df_b) / (ss_within / df_w + 1e-12)
    f_stats = np.where(np.isfinite(f_stats), f_stats, 0.0)
    return np.argsort(-f_stats)[:k]


def _eval_method(scores, sd_t, paired_donors):
    """For each paired_donor, compute rank-of-first-same and topK."""
    ranks, topK = [], {1: [], 3: [], 5: [], 10: []}
    n_same_per = []
    for i, dn in enumerate(paired_donors):
        cos = scores[i]
        order = np.argsort(-cos)
        ord_donors = sd_t[order]
        rank = next(j for j, d in enumerate(ord_donors) if d == dn) + 1
        ranks.append(rank)
        for K in topK:
            topK[K].append(float(np.mean(ord_donors[:K] == dn)))
        n_same_per.append(int((sd_t == dn).sum()))
    return {
        "mean_rank_first_same": float(np.mean(ranks)),
        "ranks": ranks,
        "mean_top1": float(np.mean(topK[1])),
        "mean_top3": float(np.mean(topK[3])),
        "mean_top5": float(np.mean(topK[5])),
        "mean_top10": float(np.mean(topK[10])),
        "n_same_avg": float(np.mean(n_same_per)),
    }


def main():
    d = np.load(PAIRED_NPZ, allow_pickle=True)
    bulk_x = d["bulk_x"].astype(np.float32)
    bulk_donor = np.asarray(d["bulk_donor"], dtype=str)
    bulk_tissue = np.asarray(d["bulk_tissue"], dtype=str)
    sn_x = d["sn_x"].astype(np.float32)
    sn_donor = np.asarray(d["sn_donor"], dtype=str)
    sn_tissue = np.asarray(d["sn_tissue"], dtype=str)
    print(f"bulk: {bulk_x.shape}  sn: {sn_x.shape}")

    summary = {}
    for tissue in sorted(set(bulk_tissue)):
        b_mask = bulk_tissue == tissue
        s_mask = sn_tissue == tissue
        Xb_t = bulk_x[b_mask]; bd_t = bulk_donor[b_mask]
        Xs_t = sn_x[s_mask]; sd_t = sn_donor[s_mask]
        paired = sorted(set(bd_t) & set(sd_t))
        if len(paired) < 2: continue

        # Per-method per-fold scoring
        method_scores = {m: [] for m in
                         ["raw_cosine", "pca50", "ftest_500", "ftest_pca50",
                          "spearman", "ftest_spearman", "ftest_intersect_pca50"]}

        for held_dn in paired:
            held_idx = np.where(bd_t == held_dn)[0][0]
            x_query = Xb_t[held_idx]
            train_b_idx = np.arange(len(Xb_t))
            train_b_idx = train_b_idx[train_b_idx != held_idx]
            held_sn_idx = np.where(sd_t == held_dn)[0]
            train_s_idx = np.array([i for i in range(len(Xs_t)) if i not in set(held_sn_idx)])

            # 1. Raw cosine
            method_scores["raw_cosine"].append(_cosine_matrix(x_query[None, :], Xs_t)[0])

            # 2. PCA-50 fit on training bulk only
            n_comp = min(50, len(train_b_idx))
            pca = PCA(n_components=n_comp).fit(Xb_t[train_b_idx])
            x_q_pca = pca.transform(x_query[None, :])
            sn_pca = pca.transform(Xs_t)
            method_scores["pca50"].append(_cosine_matrix(x_q_pca, sn_pca)[0])

            # 3. F-test on training sn samples (donor as factor)
            top_k_genes = _ftest_top_k(Xs_t[train_s_idx], sd_t[train_s_idx], k=500)
            method_scores["ftest_500"].append(
                _cosine_matrix(x_query[top_k_genes][None, :], Xs_t[:, top_k_genes])[0])

            # 4. F-test gene set + PCA-50 (PCA fit on those 500 genes)
            X_b_filt = Xb_t[train_b_idx][:, top_k_genes]
            n_comp = min(50, len(train_b_idx), len(top_k_genes))
            pca_f = PCA(n_components=n_comp).fit(X_b_filt)
            x_q_pca_f = pca_f.transform(x_query[top_k_genes][None, :])
            sn_pca_f = pca_f.transform(Xs_t[:, top_k_genes])
            method_scores["ftest_pca50"].append(_cosine_matrix(x_q_pca_f, sn_pca_f)[0])

            # 5. Spearman on raw genes
            method_scores["spearman"].append(_spearman_matrix(x_query[None, :], Xs_t)[0])

            # 6. Spearman on F-test top genes
            method_scores["ftest_spearman"].append(
                _spearman_matrix(x_query[top_k_genes][None, :], Xs_t[:, top_k_genes])[0])

            # 7. F-test intersect across all training tissues (validate generalization)
            #    For each TISSUE in training (excluding current), run F-test on its sn samples
            #    Take genes that appear in F-test top-500 in 2+ training tissues.
            cross_tissue_train_sn_donors = sn_donor[sn_tissue != tissue]
            cross_tissue_train_sn = sn_x[sn_tissue != tissue]
            cross_tissue_train_tissues = sn_tissue[sn_tissue != tissue]
            from collections import Counter
            gene_counter = Counter()
            for ts in set(cross_tissue_train_tissues):
                m = cross_tissue_train_tissues == ts
                if m.sum() < 4: continue
                top_k = _ftest_top_k(cross_tissue_train_sn[m], cross_tissue_train_sn_donors[m], k=500)
                for g in top_k:
                    gene_counter[g] += 1
            common_genes = [g for g, c in gene_counter.items() if c >= 2]
            if not common_genes:
                common_genes = list(range(min(500, bulk_x.shape[1])))
            common_genes = np.array(common_genes)
            n_comp = min(50, len(train_b_idx), len(common_genes))
            X_b_filt = Xb_t[train_b_idx][:, common_genes]
            pca_c = PCA(n_components=n_comp).fit(X_b_filt)
            x_q_pca_c = pca_c.transform(x_query[common_genes][None, :])
            sn_pca_c = pca_c.transform(Xs_t[:, common_genes])
            method_scores["ftest_intersect_pca50"].append(_cosine_matrix(x_q_pca_c, sn_pca_c)[0])

        summary[tissue] = {
            "n_paired": len(paired),
            "n_pool": len(Xs_t),
            "donors": paired,
        }
        for method, scores_list in method_scores.items():
            summary[tissue][method] = _eval_method(np.stack(scores_list), sd_t, paired)

    # Print per-tissue table
    methods = ["raw_cosine", "pca50", "ftest_500", "ftest_pca50", "spearman",
               "ftest_spearman", "ftest_intersect_pca50"]
    print(f"\n{'tissue':<22} {'n_d':>3} | " + " ".join(f"{m[:14]:>14}" for m in methods))
    print(f"{'top-1':<22} {'':>3}-+-" + "-+-".join("-" * 14 for _ in methods))
    for t, s in summary.items():
        row = f"{t:<22} {s['n_paired']:>3} | "
        row += " ".join(f"{s[m]['mean_top1']:>14.3f}" for m in methods)
        print(row)
    print(f"{'':<22} {'':>3} | " + " ".join(f"{'':>14}" for m in methods))
    print(f"{'OVERALL top-1':<22} {'':>3} | " + " ".join(
        f"{np.mean([s[m]['mean_top1'] for s in summary.values()]):>14.3f}" for m in methods))
    print(f"{'OVERALL top-3':<22} {'':>3} | " + " ".join(
        f"{np.mean([s[m]['mean_top3'] for s in summary.values()]):>14.3f}" for m in methods))
    print(f"{'OVERALL rank1':<22} {'':>3} | " + " ".join(
        f"{np.mean([s[m]['mean_rank_first_same'] for s in summary.values()]):>14.2f}" for m in methods))

    # Print which genes the F-test selects (averaged across tissues, just for interpretability)
    print(f"\n=== F-test top genes (validation that test is solvable with right features) ===")
    print(f"  see q35_ftest_genes_per_tissue.json for per-tissue lists")

    out = {
        "summary": summary,
        "overall": {m: {
            "mean_top1": float(np.mean([s[m]["mean_top1"] for s in summary.values()])),
            "mean_top3": float(np.mean([s[m]["mean_top3"] for s in summary.values()])),
            "mean_top5": float(np.mean([s[m]["mean_top5"] for s in summary.values()])),
            "mean_rank_first_same": float(np.mean([s[m]["mean_rank_first_same"] for s in summary.values()])),
        } for m in methods},
    }
    with (OUT_DIR / "q35_baselines_extended.json").open("w") as f:
        json.dump(out, f, indent=2)
    print(f"saved → {OUT_DIR / 'q35_baselines_extended.json'}")


if __name__ == "__main__":
    main()
