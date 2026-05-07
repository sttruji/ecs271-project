#!/usr/bin/env python3
"""Q7 — Validate PC biological meaningfulness, focused PC6 deep-dive.

Two analyses in one:

(A) Horn's parallel analysis  +  Marchenko–Pastur edge
    For each of n_perm permutations, columns of the standardised matrix
    are independently shuffled and PCA is re-fit. PCs whose eigenvalues
    exceed the 99th percentile of permuted eigenvalues are above the
    column-permutation noise floor. We also compute the MP edge
    λ_+ = σ² (1 + √(p/n))² as an analytical reference.

(B) PC6 (endocytosis) deep-dive
    - Print the top 20 positive- and negative-loading genes — useful for
      checking whether they're cell-type markers (monocytes, dendritic).
    - Correlate PC6 score with EVERY numeric column in
      GTEx_v10_Annotations_SampleAttributesDS.txt (degradation, library
      size, fragment-length stats, mapping rates, etc.) — surfaces any
      technical covariate that drives the PC.

Run:  python q7_pc_validation.py [--n-perm 30]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from lib_data import align_to_shared, load_gtex_blood, standardise  # noqa: E402
from lib_model import load_trained  # noqa: E402

CHECKPOINT = "/Users/rls/ecs271/data/models/cross_modality_vae.pt"
ANNOT = Path("/Users/rls/ecs271/data/annotations")
RESULTS = ROOT / "results"
OUT = ROOT / "figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=30,
                    help="permutations for Horn's analysis (default 30)")
    ap.add_argument("--max-k", type=int, default=300,
                    help="how many PCs to compare (default 300, covers > K_95)")
    args = ap.parse_args()

    print("Loading shared genes / scaler / GTEx ...")
    _, shared_genes, scaler_mean, scaler_std = load_trained(CHECKPOINT, device="cpu")
    expr_log, gene_names = load_gtex_blood()
    expr_aligned, _, _ = align_to_shared(expr_log, gene_names, shared_genes)
    expr_scaled = standardise(expr_aligned, scaler_mean, scaler_std)
    n, p = expr_scaled.shape
    print(f"  n_donors = {n}, n_genes = {p}")

    # ── Real PCA ──────────────────────────────────────────────────────────
    print("Fitting real PCA ...")
    n_comp = min(n - 1, args.max_k)
    pca = PCA(n_components=n_comp).fit(expr_scaled)
    eig_real = pca.explained_variance_

    # ── Horn's parallel analysis ──────────────────────────────────────────
    rng = np.random.default_rng(0)
    print(f"Running Horn's parallel analysis with {args.n_perm} permutations (per-column shuffle) ...")
    perm_eigs = np.zeros((args.n_perm, n_comp), dtype=np.float32)
    X = expr_scaled.copy()
    for i in range(args.n_perm):
        Xp = np.empty_like(X)
        for c in range(p):
            Xp[:, c] = rng.permutation(X[:, c])
        pca_p = PCA(n_components=n_comp).fit(Xp)
        perm_eigs[i] = pca_p.explained_variance_
        if (i + 1) % 5 == 0:
            print(f"  perm {i+1}/{args.n_perm}")
    perm_p99 = np.quantile(perm_eigs, 0.99, axis=0)
    perm_p50 = np.quantile(perm_eigs, 0.50, axis=0)

    # MP edge for whitened gaussian noise: σ²=1 (we standardised)
    mp_edge = (1 + np.sqrt(p / n)) ** 2

    above_p99 = int((eig_real > perm_p99).sum())
    above_mp = int((eig_real > mp_edge).sum())
    print(f"\nReal eigenvalue λ_1 = {eig_real[0]:.1f}; permutation p99 floor for PC1 = {perm_p99[0]:.3f}")
    print(f"PCs with eigenvalue above the per-PC permutation 99th percentile: {above_p99}")
    print(f"PCs with eigenvalue above the Marchenko-Pastur edge λ_+ = {mp_edge:.3f}: {above_mp}")
    print(f"For reference: K_95 = 251 (PCs needed for >=95% cum var).")

    # Save horn table
    horn = pd.DataFrame({
        "pc": np.arange(1, n_comp + 1),
        "eig_real": eig_real,
        "perm_p50": perm_p50,
        "perm_p99": perm_p99,
        "above_p99": eig_real > perm_p99,
    })
    horn.to_csv(RESULTS / "q7_horn_parallel.csv", index=False)
    print(f"Wrote {RESULTS / 'q7_horn_parallel.csv'}")

    plt.style.use("dark_background")
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(horn["pc"], horn["eig_real"], color="#3fb950", lw=1.5, label="real eigenvalues")
    ax.fill_between(horn["pc"], 0, perm_p99, color="#7d8590", alpha=0.4, label="permutation p99 (noise floor)")
    ax.axhline(mp_edge, color="#f78166", ls="--", lw=1, label=f"MP edge λ_+ = {mp_edge:.2f}")
    ax.axvline(above_p99, color="#58a6ff", ls=":", lw=1.5, label=f"last PC > p99 = PC{above_p99}")
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_xlabel("PC index"); ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Q7: Horn's parallel analysis — {above_p99}/{n_comp} PCs above per-PC permutation noise floor")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "q7_horn_parallel.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT / 'q7_horn_parallel.png'}")

    # ── PC6 deep-dive ─────────────────────────────────────────────────────
    print("\n--- PC6 deep-dive ---")
    pc6_loadings = pca.components_[5]  # PC index 5 = PC6
    order = np.argsort(pc6_loadings)
    bot = order[:20]
    top = order[-20:][::-1]
    print(f"\nTop 20 +loading genes (drive PC6 up):")
    for i in top:
        print(f"  {str(shared_genes[i]):<20s}  loading = {pc6_loadings[i]:+.4f}")
    print(f"\nTop 20 -loading genes (drive PC6 down):")
    for i in bot:
        print(f"  {str(shared_genes[i]):<20s}  loading = {pc6_loadings[i]:+.4f}")

    # ── Sample IDs + metadata join ────────────────────────────────────────
    import gzip
    with gzip.open("/Users/rls/ecs271/data/bulk/gtex_v11_whole_blood.gct.gz", "rt") as fh:
        for _ in range(2):
            fh.readline()
        sample_ids = np.asarray(fh.readline().rstrip("\n").split("\t")[2:])

    samp = pd.read_csv(ANNOT / "GTEx_v10_Annotations_SampleAttributesDS.txt", sep="\t", low_memory=False)
    df = pd.DataFrame({"SAMPID": sample_ids})
    df = df.merge(samp, on="SAMPID", how="left")
    pc_scores = pca.transform(expr_scaled)
    df["PC6"] = pc_scores[:, 5]
    df["PC2"] = pc_scores[:, 1]  # for comparison
    df["PC1"] = pc_scores[:, 0]

    # Spearman with every numeric column
    rows = []
    for col in samp.columns:
        if col in ("SAMPID",):
            continue
        v = pd.to_numeric(df[col], errors="coerce").values
        if np.isfinite(v).sum() < 100:
            continue
        for pc in ("PC1", "PC2", "PC6"):
            x = df[pc].values
            mask = np.isfinite(v) & np.isfinite(x)
            if mask.sum() < 50:
                continue
            r, pval = spearmanr(v[mask], x[mask])
            rows.append({"col": col, "pc": pc, "rho": float(r), "p": float(pval), "n": int(mask.sum())})
    cor = pd.DataFrame(rows)
    cor.to_csv(RESULTS / "q7_pc6_metadata_correlations.csv", index=False)

    print("\nTop 15 |ρ| metadata covariates for PC6:")
    pc6_cor = cor[cor["pc"] == "PC6"].assign(absrho=lambda d: d["rho"].abs()).sort_values("absrho", ascending=False)
    for _, r in pc6_cor.head(15).iterrows():
        print(f"  {r['col']:<14s}  ρ = {r['rho']:+.3f}  (p={r['p']:.1e}, n={r['n']})")

    print("\nFor comparison — top 10 |ρ| for PC2 (stress axis, well-understood):")
    pc2_cor = cor[cor["pc"] == "PC2"].assign(absrho=lambda d: d["rho"].abs()).sort_values("absrho", ascending=False)
    for _, r in pc2_cor.head(10).iterrows():
        print(f"  {r['col']:<14s}  ρ = {r['rho']:+.3f}  (p={r['p']:.1e}, n={r['n']})")

    print("\n--- Summary numbers ---")
    print(f"  Horn p99: {above_p99} PCs above per-PC permutation noise floor")
    print(f"  MP edge λ_+ ({mp_edge:.2f}): {above_mp} PCs above analytical noise floor")
    print(f"  K_95 (cum var >= 0.95): 251 PCs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
