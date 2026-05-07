#!/usr/bin/env python3
"""Q1 — How well does the trained cross-modality VAE reconstruct GTEx bulk?

Compares three reconstructions of the standardised log2(CPM+1) GTEx whole-blood
matrix on a 80/20 donor split (seed=0):

  • VAE (use the bulk encoder, sample z = mu, decode)
  • PCA-K (truncated PCA on the train donors), K in {16, 50, 200}
  • Identity / mean baseline (returns zeros — the standardised mean)

Reports per-donor Pearson r, per-gene Pearson r, MSE, and overall
explained-variance R^2 on held-out donors. Saves per-method JSON +
side-by-side scatter / hexbin figures.

Run:  python q1_reconstruction.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from lib_data import align_to_shared, load_gtex_blood, standardise  # noqa: E402
from lib_model import load_trained  # noqa: E402

CHECKPOINT = "/Users/rls/ecs271/data/models/cross_modality_vae.pt"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True, parents=True)
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True, parents=True)


def metric_block(true: np.ndarray, pred: np.ndarray) -> dict:
    diff = true - pred
    mse = float((diff ** 2).mean())
    mae = float(np.abs(diff).mean())
    var_total = float(true.var())
    var_resid = float(diff.var())
    r2 = float(1.0 - var_resid / var_total) if var_total > 0 else float("nan")
    per_sample_r = []
    for i in range(true.shape[0]):
        a = true[i] - true[i].mean()
        b = pred[i] - pred[i].mean()
        denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
        per_sample_r.append(float((a * b).sum() / denom) if denom else 0.0)
    per_gene_r = []
    for j in range(true.shape[1]):
        a = true[:, j] - true[:, j].mean()
        b = pred[:, j] - pred[:, j].mean()
        denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
        per_gene_r.append(float((a * b).sum() / denom) if denom else 0.0)
    per_sample_r = np.asarray(per_sample_r)
    per_gene_r = np.asarray(per_gene_r)
    return {
        "mse": mse,
        "mae": mae,
        "r2_overall": r2,
        "per_sample_r_mean": float(per_sample_r.mean()),
        "per_sample_r_med": float(np.median(per_sample_r)),
        "per_gene_r_mean": float(per_gene_r.mean()),
        "per_gene_r_med": float(np.median(per_gene_r)),
        "per_sample_r_min": float(per_sample_r.min()),
        "per_gene_r_min": float(per_gene_r.min()),
    }, per_sample_r, per_gene_r


def main() -> int:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading trained VAE checkpoint ...")
    model, shared_genes, scaler_mean, scaler_std = load_trained(CHECKPOINT, device=device)
    n_genes = len(shared_genes)
    print(f"  shared_genes: {n_genes:,}  latent_dim: {model.latent_dim}")

    print("Loading GTEx whole blood ...")
    expr_log, gene_names = load_gtex_blood()
    print(f"  GTEx (post-filter): {expr_log.shape[0]} donors x {expr_log.shape[1]:,} genes")

    expr_aligned, found, missing = align_to_shared(expr_log, gene_names, shared_genes)
    print(f"  aligned to checkpoint genes: {found}/{n_genes} found, {len(missing)} missing (mean-imputed)")

    expr_scaled = standardise(expr_aligned, scaler_mean, scaler_std)
    print(f"  scaled matrix: {expr_scaled.shape}  (mean ~ {expr_scaled.mean():.3g}, std ~ {expr_scaled.std():.3g})")

    rng = np.random.default_rng(0)
    n = expr_scaled.shape[0]
    perm = rng.permutation(n)
    n_test = int(round(0.2 * n))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    train = expr_scaled[train_idx]
    test = expr_scaled[test_idx]
    print(f"  split: train={len(train_idx)}  test={len(test_idx)}")

    # ── VAE forward (use posterior mean μ) ────────────────────────────────
    print("\nVAE reconstruction ...")
    with torch.no_grad():
        x_test = torch.from_numpy(test).to(device)
        mu, _ = model.enc_bulk(x_test)
        recon_vae = model.decoder(mu).cpu().numpy()

    vae_metrics, vae_sample_r, vae_gene_r = metric_block(test, recon_vae)
    print("  VAE :", vae_metrics)

    # ── PCA baselines (fit on train, project test → reconstruct) ─────────
    pca_results = {}
    for k in (16, 50, 200, 500):
        if k >= train.shape[0]:
            continue
        pca = PCA(n_components=k).fit(train)
        recon_k = pca.inverse_transform(pca.transform(test))
        m, sr, gr = metric_block(test, recon_k)
        pca_results[f"pca_{k}"] = m
        print(f"  PCA-{k:>3}: {m}")

    # ── Mean baseline (zeros after scaling) ───────────────────────────────
    mean_recon = np.zeros_like(test)
    mean_metrics, _, _ = metric_block(test, mean_recon)
    print("  MEAN:", mean_metrics)

    # ── Save ──────────────────────────────────────────────────────────────
    out = {
        "n_donors_total": int(n),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_genes": int(n_genes),
        "missing_genes_imputed": len(missing),
        "vae": vae_metrics,
        "mean_baseline": mean_metrics,
        **pca_results,
    }
    (RESULTS / "q1_reconstruction.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {RESULTS / 'q1_reconstruction.json'}")

    # ── Figure: bar of R² across methods ──────────────────────────────────
    plt.style.use("dark_background")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    methods = ["mean", "pca_16", "pca_50", "pca_200", "vae"]
    if "pca_500" in pca_results:
        methods = ["mean", "pca_16", "pca_50", "pca_200", "pca_500", "vae"]
    labels = [m.replace("_", "-").upper() for m in methods]
    r2 = []
    for m in methods:
        if m == "mean":
            r2.append(mean_metrics["r2_overall"])
        elif m == "vae":
            r2.append(vae_metrics["r2_overall"])
        else:
            r2.append(pca_results[m]["r2_overall"])
    colors = ["#7d8590"] + ["#58a6ff"] * (len(methods) - 2) + ["#3fb950"]
    ax[0].bar(labels, r2, color=colors)
    ax[0].set_ylabel("Held-out R² on standardised log2(CPM+1)")
    ax[0].set_title("Q1: Reconstruction of GTEx bulk (held-out 20%)")
    ax[0].axhline(0, color="#7d8590", lw=0.8)
    for i, v in enumerate(r2):
        ax[0].text(i, v + 0.01 * np.sign(v), f"{v:.3f}", ha="center", fontsize=9)

    # per-gene Pearson r distribution
    bins = np.linspace(-0.2, 1.0, 60)
    ax[1].hist(vae_gene_r, bins=bins, alpha=0.7, color="#3fb950", label=f"VAE  (med={np.median(vae_gene_r):.2f})")
    if "pca_50" in pca_results:
        # rerun PCA-50 to get its per-gene r for plotting
        pca50 = PCA(n_components=50).fit(train)
        recon50 = pca50.inverse_transform(pca50.transform(test))
        _, _, pca50_gene_r = metric_block(test, recon50)
        ax[1].hist(pca50_gene_r, bins=bins, alpha=0.5, color="#58a6ff", label=f"PCA-50 (med={np.median(pca50_gene_r):.2f})")
    ax[1].set_xlabel("Per-gene Pearson r (test donors)")
    ax[1].set_ylabel("Genes")
    ax[1].set_title("Per-gene reconstruction quality")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "q1_reconstruction.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT / 'q1_reconstruction.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
