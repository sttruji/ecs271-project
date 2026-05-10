#!/usr/bin/env python3
"""Q18 — Empirical bio-dimensionality of GTEx whole blood.

THESIS (user's framing): the biological signal in bulk RNA-seq lives in
a much smaller subspace than the 11k-gene measurement.  We test it by
sweeping PCA-N at fine-grained N ∈ {1..500} and asking: at what N does
adding more dimensions stop materially improving held-out reconstruction?
That N is the empirical bio-dim.

Linear-MMSE-optimal here means PCA — Q17 confirmed that nonlinear
encoders/decoders trained from scratch *underperform* PCA on this
standardised log-CPM bulk regime (the data is approximately Gaussian
and PCA is the optimum).  So the PCA-N curve is the right question.

Three complementary metrics, all on the same 161-donor held-out test
set (same seed=0 80/20 split as every other Q).

  •  R²                      — fraction of variance explained
  •  frac<1σ                 — fraction of (sample, gene) predictions
                              with absolute error < 1 std (the user's
                              within-1σ criterion)
  •  per-gene resid std max  — worst-case gene reconstruction tightness

We also report the **knee** by two criteria:
  •  First N at which incrementing N adds < 0.10 pct-pts to frac<1σ
  •  First N within 0.5 pct-pts of the asymptote (PCA-500)

Run:  python q18_bio_dim.py
"""
from __future__ import annotations

import json
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
FIGS = ROOT / "figures"
RESULTS.mkdir(exist_ok=True, parents=True)
FIGS.mkdir(exist_ok=True, parents=True)

SEED = 0
DIMS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64,
        80, 96, 128, 160, 192, 256, 320, 400, 500]


def main() -> int:
    print("[1/3] Loading GTEx + setup ...")
    _, shared_genes, scaler_mean, scaler_std = load_trained(CHECKPOINT, device="cpu")
    expr_log, gene_names = load_gtex_blood()
    expr_aligned, _, _ = align_to_shared(expr_log, gene_names, shared_genes)
    X = standardise(expr_aligned, scaler_mean, scaler_std)
    n_samples, n_genes = X.shape
    print(f"  X: {X.shape}  (donors × genes)")

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_samples)
    n_test = int(round(0.2 * n_samples))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    X_train, X_test = X[train_idx], X[test_idx]

    print(f"\n[2/3] PCA-N sweep, N ∈ {DIMS} ...")
    rows = []
    for N in DIMS:
        if N >= X_train.shape[0]:
            continue
        pca = PCA(n_components=N).fit(X_train)
        X_hat = pca.inverse_transform(pca.transform(X_test))
        err = X_test - X_hat
        var_total = float(X_test.var())
        cum_var_train = float(pca.explained_variance_ratio_.sum())
        m = {
            "N": N,
            "cum_var_train": cum_var_train,
            "r2_test": float(1 - err.var() / var_total),
            "mse_test": float((err ** 2).mean()),
            "frac_within_1std": float((np.abs(err) < 1.0).mean()),
            "frac_within_0.5std": float((np.abs(err) < 0.5).mean()),
            "frac_within_2std": float((np.abs(err) < 2.0).mean()),
            "max_abs_err": float(np.abs(err).max()),
            "per_gene_resid_std_p99": float(np.percentile(err.std(0), 99)),
            "per_gene_resid_std_max": float(err.std(0).max()),
        }
        rows.append(m)
        print(f"  PCA-{N:>3}  cumvar={cum_var_train:.3f}  R²={m['r2_test']:.3f}  "
              f"frac<1σ={m['frac_within_1std']:.3%}  "
              f"frac<0.5σ={m['frac_within_0.5std']:.3%}  "
              f"max|err|={m['max_abs_err']:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "q18_bio_dim.csv", index=False)
    asymptote = df["frac_within_1std"].iloc[-1]
    df["pp_gain_per_dim"] = df["frac_within_1std"].diff() * 100 / df["N"].diff()
    df["pp_below_asymptote"] = (asymptote - df["frac_within_1std"]) * 100

    # Knee detection
    knee_per_dim = df[df["pp_gain_per_dim"] < 0.10]
    knee_at_per_dim = (int(knee_per_dim["N"].iloc[0])
                       if not knee_per_dim.empty else None)
    knee_to_asym = df[df["pp_below_asymptote"] < 0.5]
    knee_at_asym = (int(knee_to_asym["N"].iloc[0])
                    if not knee_to_asym.empty else None)

    summary = {
        "asymptote_frac_within_1std": asymptote,
        "asymptote_at_N": int(df["N"].iloc[-1]),
        "knee_first_N_with_per_dim_gain_below_0.10pp": knee_at_per_dim,
        "knee_first_N_within_0.5pp_of_asymptote": knee_at_asym,
        "rows": rows,
    }
    (RESULTS / "q18_bio_dim.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  Asymptote frac<1σ = {asymptote:.3%}  at N = {int(df['N'].iloc[-1])}")
    print(f"  Knee  (per-dim gain < 0.1 pct-pts):  N = {knee_at_per_dim}")
    print(f"  Knee  (within 0.5 pct-pts of asymptote): N = {knee_at_asym}")
    print(f"  Wrote {RESULTS / 'q18_bio_dim.csv'}, {RESULTS / 'q18_bio_dim.json'}")

    # Plot
    print("\n[3/3] Drawing figure ...")
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    fig.suptitle("Q18 — Bio-dimensionality of GTEx whole blood "
                 "(PCA-N held-out test, n=161 donors × 11,374 genes)",
                 fontsize=12)

    Ns = df["N"].values
    f1 = df["frac_within_1std"].values * 100
    f05 = df["frac_within_0.5std"].values * 100
    r2 = df["r2_test"].values

    # (a) within-1σ vs N
    ax = axes[0]
    ax.plot(Ns, f1, "-o", color="#3fb950", lw=2, label="frac<1σ (target metric)")
    ax.plot(Ns, f05, "--s", color="#58a6ff", lw=1.5, alpha=0.7,
            label="frac<0.5σ (tighter)")
    ax.axhline(99, color="#f78166", ls="--", lw=0.8, label="user target 99%")
    if knee_at_asym is not None:
        ax.axvline(knee_at_asym, color="#7d8590", ls=":", lw=0.8,
                   label=f"knee N={knee_at_asym} (within 0.5pp asymptote)")
    ax.set_xscale("log")
    ax.set_xlabel("Latent dim N")
    ax.set_ylabel("Test predictions (%)")
    ax.set_title(f"(a) Reconstruction tightness vs N\n"
                 f"asymptote frac<1σ = {asymptote:.2%} (PCA-{int(df['N'].iloc[-1])})")
    ax.set_ylim(50, 100)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=8)

    # (b) R² + cumulative variance
    ax = axes[1]
    ax.plot(Ns, r2, "-o", color="#3fb950", lw=2, label="held-out R²")
    ax.plot(Ns, df["cum_var_train"].values, "--s", color="#58a6ff",
            lw=1.5, alpha=0.7, label="train cum-var ratio")
    ax.set_xscale("log")
    ax.set_xlabel("Latent dim N")
    ax.set_ylabel("Variance explained")
    ax.set_title("(b) R² and cumulative variance vs N")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=9)

    # (c) per-dim gain — locate the knee precisely
    ax = axes[2]
    pp_gain = df["pp_gain_per_dim"].fillna(np.nan).values
    valid = np.isfinite(pp_gain)
    ax.plot(Ns[valid], pp_gain[valid], "-o", color="#3fb950", lw=2,
            label="frac<1σ gain per +1 dim")
    ax.axhline(0.10, color="#f78166", ls="--", lw=0.8,
               label="0.1 pct-pt knee threshold")
    if knee_at_per_dim is not None:
        ax.axvline(knee_at_per_dim, color="#7d8590", ls=":", lw=0.8,
                   label=f"knee N={knee_at_per_dim}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Latent dim N")
    ax.set_ylabel("Δ frac<1σ (pct-pts) per +1 dim")
    ax.set_title("(c) Marginal gain per added dim — locates the knee")
    ax.grid(alpha=0.2, which="both")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGS / "q18_bio_dim.png", dpi=150)
    plt.close(fig)
    print(f"  Wrote {FIGS / 'q18_bio_dim.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
