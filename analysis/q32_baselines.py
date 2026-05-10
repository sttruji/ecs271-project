"""Q32 — Sanity-check baselines on Eraslan paired Test 2.

Are we even ABOVE chance with simple methods that have no learned model?
- raw cosine: encode bulk in raw gene-space, NN among raw sn samples
- PCA-50: project both into PCA-50 of training bulk
- per-tissue LOO (no fine-tuning, no flip)

If even these are at chance, the dataset has no donor-discriminative signal
across modalities. If they work, the VAE is the bottleneck.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAIRED_NPZ = "/Users/rls/ecs271/data/sc/eraslan/eraslan_paired.npz"
OUT_DIR = ROOT / "analysis" / "results" / "q20_disentangled"


def _cosine(a, b):
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    d = np.load(PAIRED_NPZ, allow_pickle=True)
    bulk_x = d["bulk_x"].astype(np.float32)
    bulk_donor = np.asarray(d["bulk_donor"], dtype=str)
    bulk_tissue = np.asarray(d["bulk_tissue"], dtype=str)
    sn_x = d["sn_x"].astype(np.float32)
    sn_donor = np.asarray(d["sn_donor"], dtype=str)
    sn_tissue = np.asarray(d["sn_tissue"], dtype=str)
    print(f"bulk: {bulk_x.shape}  sn: {sn_x.shape}")

    # Per-tissue analysis
    summary = {}
    for tissue in sorted(set(bulk_tissue)):
        b_mask = bulk_tissue == tissue
        s_mask = sn_tissue == tissue
        Xb_t = bulk_x[b_mask]; bd_t = bulk_donor[b_mask]
        Xs_t = sn_x[s_mask]; sd_t = sn_donor[s_mask]
        paired = sorted(set(bd_t) & set(sd_t))
        if len(paired) < 2: continue

        ranks_raw, ranks_pca = [], []
        topK_raw = {1: [], 3: [], 5: [], 10: []}
        topK_pca = {1: [], 3: [], 5: [], 10: []}

        for held_dn in paired:
            held_idx = np.where(bd_t == held_dn)[0][0]
            x_query = Xb_t[held_idx]

            # Raw cosine bulk → sn
            cos_raw = np.array([_cosine(x_query, Xs_t[i]) for i in range(len(Xs_t))])
            order = np.argsort(-cos_raw)
            same_rank = next(i for i, dn in enumerate(sd_t[order]) if dn == held_dn) + 1
            ranks_raw.append(same_rank)
            for K in topK_raw:
                topK_raw[K].append(float(np.mean(sd_t[order[:K]] == held_dn)))

            # PCA-50 of training bulk (LOO), then project both bulk and sn
            train_bulk_idx = np.arange(len(Xb_t))
            # Don't use held-out bulk for PCA fit (avoid leak)
            train_bulk_idx = train_bulk_idx[train_bulk_idx != held_idx]
            n_comp = min(50, len(train_bulk_idx))
            pca = PCA(n_components=n_comp).fit(Xb_t[train_bulk_idx])
            x_q_pca = pca.transform(x_query[None, :])[0]
            sn_pca = pca.transform(Xs_t)
            cos_pca = np.array([_cosine(x_q_pca, sn_pca[i]) for i in range(len(sn_pca))])
            order = np.argsort(-cos_pca)
            same_rank = next(i for i, dn in enumerate(sd_t[order]) if dn == held_dn) + 1
            ranks_pca.append(same_rank)
            for K in topK_pca:
                topK_pca[K].append(float(np.mean(sd_t[order[:K]] == held_dn)))

        n_pool = len(Xs_t)
        # Average n_same per donor across this tissue's paired donors
        n_same_per = [int((sd_t == d).sum()) for d in paired]
        expected_top1 = float(np.mean([n / n_pool for n in n_same_per]))
        expected_rank1 = float(np.mean([(n_pool - n + 1) / (n + 1) for n in n_same_per]))

        summary[tissue] = {
            "n_paired_donors": len(paired),
            "n_pool": n_pool,
            "n_same_in_pool_avg": float(np.mean(n_same_per)),
            "raw_cosine": {
                "mean_rank_first_same": float(np.mean(ranks_raw)),
                "mean_top1": float(np.mean(topK_raw[1])),
                "mean_top3": float(np.mean(topK_raw[3])),
                "mean_top5": float(np.mean(topK_raw[5])),
                "mean_top10": float(np.mean(topK_raw[10])),
            },
            "pca50": {
                "mean_rank_first_same": float(np.mean(ranks_pca)),
                "mean_top1": float(np.mean(topK_pca[1])),
                "mean_top3": float(np.mean(topK_pca[3])),
                "mean_top5": float(np.mean(topK_pca[5])),
                "mean_top10": float(np.mean(topK_pca[10])),
            },
            "random_top1": expected_top1,
            "random_rank1": expected_rank1,
        }

    print(f"\n=== Sanity baselines per tissue (held-out donor's bulk → NN among ALL tissue sn samples) ===\n")
    print(f"{'tissue':<22} {'n_d':>3} {'raw_top1':>10} {'pca_top1':>10} {'random':>8} | "
          f"{'raw_rank1':>10} {'pca_rank1':>10} {'random':>8}")
    for t, s in summary.items():
        print(f"{t:<22} {s['n_paired_donors']:>3} "
              f"{s['raw_cosine']['mean_top1']:>10.3f} {s['pca50']['mean_top1']:>10.3f} {s['random_top1']:>8.3f} | "
              f"{s['raw_cosine']['mean_rank_first_same']:>10.2f} {s['pca50']['mean_rank_first_same']:>10.2f} "
              f"{s['random_rank1']:>8.2f}")

    overall_raw_top1 = float(np.mean([s['raw_cosine']['mean_top1'] for s in summary.values()]))
    overall_pca_top1 = float(np.mean([s['pca50']['mean_top1'] for s in summary.values()]))
    overall_random = float(np.mean([s['random_top1'] for s in summary.values()]))
    overall_raw_rank = float(np.mean([s['raw_cosine']['mean_rank_first_same'] for s in summary.values()]))
    overall_pca_rank = float(np.mean([s['pca50']['mean_rank_first_same'] for s in summary.values()]))
    overall_random_rank = float(np.mean([s['random_rank1'] for s in summary.values()]))
    print(f"\n{'OVERALL':<22} {'  ':>3} {overall_raw_top1:>10.3f} {overall_pca_top1:>10.3f} {overall_random:>8.3f} | "
          f"{overall_raw_rank:>10.2f} {overall_pca_rank:>10.2f} {overall_random_rank:>8.2f}")

    summary["_overall"] = {
        "raw_top1": overall_raw_top1, "pca_top1": overall_pca_top1, "random_top1": overall_random,
        "raw_rank_first_same": overall_raw_rank, "pca_rank_first_same": overall_pca_rank,
        "random_rank_first_same": overall_random_rank,
    }
    with (OUT_DIR / "q32_baselines.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved → {OUT_DIR / 'q32_baselines.json'}")


if __name__ == "__main__":
    main()
