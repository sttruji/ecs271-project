#!/usr/bin/env python3
"""Q8 — Bootstrap stability of PCs.

Donor-level resampling with replacement, 50 bootstraps. For each
bootstrap, refit PCA, then for each original PC_k find the bootstrap
PC with maximum |cos similarity| to the original loading vector. Report:

  - median, 5th, 95th percentile of |cos| per PC
  - sign-flip rate (sign of cos before taking abs)
  - first PC where median |cos| < 0.5 (heuristic stability cutoff)

A stable PC has median |cos| close to 1; an unstable / noise PC bounces
around the unit sphere with much lower median |cos| and high variance.

Run:  python q8_bootstrap_stability.py [--n-boot 50] [--n-pcs 100]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from lib_data import align_to_shared, load_gtex_blood, standardise  # noqa: E402
from lib_model import load_trained  # noqa: E402

CHECKPOINT = "/Users/rls/ecs271/data/models/cross_modality_vae.pt"
RESULTS = ROOT / "results"
OUT = ROOT / "figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=50)
    ap.add_argument("--n-pcs", type=int, default=100)
    args = ap.parse_args()

    print("Loading data ...")
    _, shared_genes, scaler_mean, scaler_std = load_trained(CHECKPOINT, device="cpu")
    expr_log, gene_names = load_gtex_blood()
    expr_aligned, _, _ = align_to_shared(expr_log, gene_names, shared_genes)
    X = standardise(expr_aligned, scaler_mean, scaler_std)
    n, p = X.shape
    print(f"  {n} donors x {p} genes")

    print(f"Fitting reference PCA (top {args.n_pcs} PCs) ...")
    pca_ref = PCA(n_components=args.n_pcs).fit(X)
    V_ref = pca_ref.components_  # (n_pcs, p)

    print(f"Running {args.n_boot} bootstrap resamples ...")
    rng = np.random.default_rng(0)
    cos_matrix = np.zeros((args.n_boot, args.n_pcs), dtype=np.float32)
    sign_matrix = np.zeros((args.n_boot, args.n_pcs), dtype=np.int8)

    for b in range(args.n_boot):
        idx = rng.choice(n, size=n, replace=True)
        Xb = X[idx]
        # The training scaler was fit on the original X; for bootstrap we re-centre
        # to the bootstrap mean to avoid mean-shift artefacts in the PCs.
        Xb = Xb - Xb.mean(axis=0, keepdims=True)
        pca_b = PCA(n_components=args.n_pcs).fit(Xb)
        V_b = pca_b.components_  # (n_pcs, p)
        # For each original PC_k, find best-matching bootstrap PC by max |cos|
        # (some bootstrap PCs swap order, so we don't assume diagonal alignment)
        cos_full = V_ref @ V_b.T   # (n_pcs, n_pcs)
        # Greedy match: for each ref PC, take the bootstrap PC with max |cos|
        # not yet taken. With 100 PCs this is OK without Hungarian.
        used = np.zeros(args.n_pcs, dtype=bool)
        for k in range(args.n_pcs):
            absrow = np.abs(cos_full[k])
            absrow[used] = -1
            j = int(np.argmax(absrow))
            used[j] = True
            cos_matrix[b, k] = absrow[j]
            sign_matrix[b, k] = int(np.sign(cos_full[k, j]))
        if (b + 1) % 10 == 0:
            print(f"  bootstrap {b+1}/{args.n_boot}")

    # Aggregate
    median = np.median(cos_matrix, axis=0)
    p5 = np.percentile(cos_matrix, 5, axis=0)
    p95 = np.percentile(cos_matrix, 95, axis=0)
    flip_rate = (sign_matrix == -1).mean(axis=0)

    df = pd.DataFrame({
        "pc": np.arange(1, args.n_pcs + 1),
        "var_explained": pca_ref.explained_variance_ratio_,
        "median_abs_cos": median,
        "p5_abs_cos": p5,
        "p95_abs_cos": p95,
        "sign_flip_rate": flip_rate,
    })
    df.to_csv(RESULTS / "q8_bootstrap_stability.csv", index=False)
    print(f"\nWrote {RESULTS / 'q8_bootstrap_stability.csv'}")

    # Stability cutoffs
    first_under_50 = int(np.argmax(median < 0.5)) + 1 if (median < 0.5).any() else None
    first_under_70 = int(np.argmax(median < 0.7)) + 1 if (median < 0.7).any() else None
    print(f"  median |cos| < 0.7 first at PC{first_under_70}")
    print(f"  median |cos| < 0.5 first at PC{first_under_50}")
    print(f"  PC1-10 stability:  median |cos| = {median[:10]}")
    print(f"  PC30-40 stability: median |cos| = {median[29:39]}")

    plt.style.use("dark_background")
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.fill_between(df["pc"], df["p5_abs_cos"], df["p95_abs_cos"], color="#3fb950", alpha=0.3, label="5%-95%")
    ax.plot(df["pc"], df["median_abs_cos"], color="#3fb950", lw=1.6, label="median")
    ax.axhline(0.5, color="#7d8590", ls=":", lw=0.8, label="|cos|=0.5")
    ax.axhline(0.7, color="#f78166", ls=":", lw=0.8, label="|cos|=0.7")
    if first_under_50:
        ax.axvline(first_under_50, color="#7d8590", ls="--", lw=1)
    if first_under_70:
        ax.axvline(first_under_70, color="#f78166", ls="--", lw=1)
    ax.set_xlabel("PC index"); ax.set_ylabel("|cos similarity| of PC loading vs bootstrap")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Q8: Bootstrap stability of PCs (n_boot = {args.n_boot})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "q8_bootstrap_stability.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT / 'q8_bootstrap_stability.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
