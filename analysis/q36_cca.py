"""Q36 — CCA (Canonical Correlation Analysis) encoder for cross-modality NN.

CCA learns linear projections of bulk and sn into a shared low-dim space
that maximizes their cross-modality correlation on paired training data.
At inference, project held-out bulk + all sn into the CCA space, do cosine NN.

This is the closest to a "scVI/scANVI" approach: explicit cross-modality
alignment via supervised paired data, no decoder, no flip.

Per-tissue LOO over the 4 (or 3) paired Eraslan donors.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
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


def _eval_per_donor(scores, sd_t, paired_donors):
    ranks = []; topK = {1: [], 3: [], 5: [], 10: []}
    for i, dn in enumerate(paired_donors):
        order = np.argsort(-scores[i])
        ord_d = sd_t[order]
        rank = next(j for j, d in enumerate(ord_d) if d == dn) + 1
        ranks.append(rank)
        for K in topK:
            topK[K].append(float(np.mean(ord_d[:K] == dn)))
    return {
        "ranks": ranks,
        "mean_rank_first_same": float(np.mean(ranks)),
        "mean_top1": float(np.mean(topK[1])),
        "mean_top3": float(np.mean(topK[3])),
        "mean_top5": float(np.mean(topK[5])),
        "mean_top10": float(np.mean(topK[10])),
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
    methods = ["cca_5", "cca_10", "cca_25", "pca_then_cca"]

    for tissue in sorted(set(bulk_tissue)):
        b_mask = bulk_tissue == tissue
        s_mask = sn_tissue == tissue
        Xb_t = bulk_x[b_mask]; bd_t = bulk_donor[b_mask]
        Xs_t = sn_x[s_mask]; sd_t = sn_donor[s_mask]
        paired = sorted(set(bd_t) & set(sd_t))
        if len(paired) < 2: continue

        method_scores = {m: [] for m in methods}

        for held_dn in paired:
            held_idx = np.where(bd_t == held_dn)[0][0]
            x_query = Xb_t[held_idx]
            train_b_idx = np.arange(len(Xb_t))
            train_b_idx = train_b_idx[train_b_idx != held_idx]
            held_sn_idx = np.where(sd_t == held_dn)[0]
            train_s_idx = np.array([i for i in range(len(Xs_t)) if i not in set(held_sn_idx)])

            # Build PAIRED training set: for each training bulk donor, pair with
            # one of their training sn samples (just take the first).
            # CCA needs equal-length paired arrays.
            train_pairs_X, train_pairs_Y = [], []
            for bi in train_b_idx:
                dn = bd_t[bi]
                same_sn = [j for j in train_s_idx if sd_t[j] == dn]
                if not same_sn: continue
                # Use ALL same-donor sn samples (each pair has same bulk vector)
                for j in same_sn:
                    train_pairs_X.append(Xb_t[bi])
                    train_pairs_Y.append(Xs_t[j])
            if len(train_pairs_X) < 3: continue
            X_train = np.stack(train_pairs_X)
            Y_train = np.stack(train_pairs_Y)

            for n_comp_label, n_comp_actual in [("cca_5", 5), ("cca_10", 10), ("cca_25", 25)]:
                n_comp = min(n_comp_actual, len(train_pairs_X) - 1, X_train.shape[1])
                if n_comp < 1:
                    method_scores[n_comp_label].append(np.zeros(len(Xs_t)))
                    continue
                try:
                    cca = CCA(n_components=n_comp, max_iter=500)
                    cca.fit(X_train, Y_train)
                    x_q_score, _ = cca.transform(x_query[None, :], np.zeros((1, Y_train.shape[1])))
                    _, sn_score = cca.transform(np.zeros((Xs_t.shape[0], X_train.shape[1])), Xs_t)
                    method_scores[n_comp_label].append(_cosine_matrix(x_q_score, sn_score)[0])
                except Exception:
                    method_scores[n_comp_label].append(np.zeros(len(Xs_t)))

            # PCA then CCA: PCA each modality to 50 dims first (denoising)
            try:
                n_pca = min(50, X_train.shape[0] - 1, X_train.shape[1])
                pca_b = PCA(n_components=n_pca).fit(X_train)
                pca_s = PCA(n_components=n_pca).fit(Y_train)
                X_train_pca = pca_b.transform(X_train)
                Y_train_pca = pca_s.transform(Y_train)
                n_comp = min(10, X_train_pca.shape[0] - 1, X_train_pca.shape[1])
                cca = CCA(n_components=n_comp, max_iter=500)
                cca.fit(X_train_pca, Y_train_pca)
                x_q_pca = pca_b.transform(x_query[None, :])
                sn_pca = pca_s.transform(Xs_t)
                x_q_score, _ = cca.transform(x_q_pca, np.zeros((1, Y_train_pca.shape[1])))
                _, sn_score = cca.transform(np.zeros((sn_pca.shape[0], X_train_pca.shape[1])), sn_pca)
                method_scores["pca_then_cca"].append(_cosine_matrix(x_q_score, sn_score)[0])
            except Exception:
                method_scores["pca_then_cca"].append(np.zeros(len(Xs_t)))

        summary[tissue] = {"n_paired": len(paired), "n_pool": len(Xs_t)}
        for m, scores in method_scores.items():
            if len(scores) == len(paired):
                summary[tissue][m] = _eval_per_donor(np.stack(scores), sd_t, paired)

    print(f"\n{'tissue':<22} {'n_d':>3} | " + " ".join(f"{m:>14}" for m in methods))
    print(f"{'top-1':<22} {'':>3}-+-" + "-+-".join("-" * 14 for _ in methods))
    for t, s in summary.items():
        row = f"{t:<22} {s['n_paired']:>3} | "
        row += " ".join(f"{s.get(m, {'mean_top1': 0})['mean_top1']:>14.3f}" for m in methods)
        print(row)
    print(f"{'OVERALL top-1':<22} {'':>3} | " + " ".join(
        f"{np.mean([s.get(m, {'mean_top1': 0})['mean_top1'] for s in summary.values()]):>14.3f}" for m in methods))
    print(f"{'OVERALL top-3':<22} {'':>3} | " + " ".join(
        f"{np.mean([s.get(m, {'mean_top3': 0})['mean_top3'] for s in summary.values()]):>14.3f}" for m in methods))
    print(f"{'OVERALL rank1':<22} {'':>3} | " + " ".join(
        f"{np.mean([s.get(m, {'mean_rank_first_same': 0})['mean_rank_first_same'] for s in summary.values()]):>14.2f}" for m in methods))

    out = {"summary": summary,
           "overall": {m: {
               "mean_top1": float(np.mean([s.get(m, {"mean_top1": 0})["mean_top1"] for s in summary.values()])),
               "mean_top3": float(np.mean([s.get(m, {"mean_top3": 0})["mean_top3"] for s in summary.values()])),
               "mean_rank_first_same": float(np.mean([s.get(m, {"mean_rank_first_same": 0})["mean_rank_first_same"] for s in summary.values()])),
           } for m in methods}}
    with (OUT_DIR / "q36_cca.json").open("w") as f:
        json.dump(out, f, indent=2)
    print(f"saved → {OUT_DIR / 'q36_cca.json'}")


if __name__ == "__main__":
    main()
