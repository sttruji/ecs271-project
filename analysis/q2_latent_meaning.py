#!/usr/bin/env python3
"""Q2 — Does the VAE latent encode meaningful biology?

Probes:
  (a) Latent activity per dim — variance across donors. A healthy VAE has
      most dims with var ~ 1 (close to prior). Posterior collapse ⇒ var ~ 0.
  (b) PCA of latent — should show 1 or 2 strong axes if biology is encoded
      (we expect the State-A / State-B handling-stress axis from
      HEALTHY_STATE_v1.md §2 to project clearly into z if it survived).
  (c) Correlation of each latent dim with bulk PC1, PC2, PC3 (these are
      the *known* biological axes from the bulk healthy-state work). If
      latent dim k correlates with PC1, biology is preserved.
  (d) Anchor-gene linear probe — fit a logistic regression on z to
      predict the State-A / State-B label using the 2-gene anchor
      (DDIT4 high ⇒ State-B per HEALTHY_STATE §2).

Run:  python q2_latent_meaning.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from lib_data import align_to_shared, load_gtex_blood, standardise  # noqa: E402
from lib_model import load_trained  # noqa: E402

CHECKPOINT = "/Users/rls/ecs271/data/models/cross_modality_vae.pt"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True, parents=True)
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True, parents=True)


def per_donor_state_label(expr_log: np.ndarray, gene_names: np.ndarray) -> np.ndarray:
    """Apply the 2-gene anchor rule (DDIT4 / FRAT1, both High ⇒ State-B).

    From HEALTHY_STATE_v1.md §2: 'DDIT4 (HIGH in B), FRAT1 (HIGH in B)'.
    Use a per-gene median split ⇒ donor in State-B if both genes are above
    their median expression.
    Returns 1 = State-B (handling-stress), 0 = State-A.
    """
    gn_upper = np.array([g.upper() for g in gene_names])
    ddit4 = np.where(gn_upper == "DDIT4")[0]
    frat1 = np.where(gn_upper == "FRAT1")[0]
    if len(ddit4) == 0 or len(frat1) == 0:
        # Fall back: just use DDIT4
        ddit4 = np.where(gn_upper == "DDIT4")[0]
        return (expr_log[:, ddit4[0]] > np.median(expr_log[:, ddit4[0]])).astype(int)
    a = expr_log[:, ddit4[0]] > np.median(expr_log[:, ddit4[0]])
    b = expr_log[:, frat1[0]] > np.median(expr_log[:, frat1[0]])
    return (a & b).astype(int)


def main() -> int:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading trained VAE checkpoint ...")
    model, shared_genes, scaler_mean, scaler_std = load_trained(CHECKPOINT, device=device)

    print("Loading GTEx whole blood ...")
    expr_log, gene_names = load_gtex_blood()
    expr_aligned, _, _ = align_to_shared(expr_log, gene_names, shared_genes)
    expr_scaled = standardise(expr_aligned, scaler_mean, scaler_std)

    # ── encode all 803 donors ─────────────────────────────────────────────
    with torch.no_grad():
        z = model.enc_bulk(torch.from_numpy(expr_scaled).to(device))[0].cpu().numpy()
    print(f"  z shape: {z.shape}, var-per-dim mean: {z.var(0).mean():.4g}")

    # (a) latent dim variance
    var_per_dim = z.var(0)
    active = int((var_per_dim > 0.01).sum())
    near_collapsed = int((var_per_dim < 1e-3).sum())
    print(f"  dims with var > 0.01 ('active'): {active}/64")
    print(f"  dims with var < 1e-3 ('near-collapsed'): {near_collapsed}/64")

    # (b) PCA of z
    pca_z = PCA(n_components=min(10, z.shape[1])).fit(z)
    print(f"  PC1-PC2 variance explained in z: {pca_z.explained_variance_ratio_[0]:.3f}, {pca_z.explained_variance_ratio_[1]:.3f}")

    # (c) correlate latent dims with bulk PCs
    pca_x = PCA(n_components=10).fit(expr_scaled)
    bulk_pcs = pca_x.transform(expr_scaled)  # (803, 10)
    corr = np.zeros((z.shape[1], 10))
    for i in range(z.shape[1]):
        for j in range(10):
            a = z[:, i] - z[:, i].mean()
            b = bulk_pcs[:, j] - bulk_pcs[:, j].mean()
            denom = np.sqrt((a * a).sum() * (b * b).sum())
            corr[i, j] = (a * b).sum() / denom if denom > 0 else 0.0
    max_abs_per_pc = np.abs(corr).max(0)
    print("  max |r| of any latent dim with bulk PC1..PC5:", [f"{v:.3f}" for v in max_abs_per_pc[:5]])

    # (d) anchor-gene linear probe
    state_labels = per_donor_state_label(expr_log, gene_names)
    print(f"  state-B fraction (DDIT4 & FRAT1 above median): {state_labels.mean():.3f}")

    # Use z to predict state — cross-validated AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    if z.var() > 1e-10:
        clf = LogisticRegression(max_iter=2000, C=1.0)
        aucs = cross_val_score(clf, z, state_labels, cv=cv, scoring="roc_auc")
        z_auc = float(aucs.mean())
    else:
        z_auc = 0.5  # collapsed: nothing to learn from
    # As a control: use the bulk PCs directly
    clf2 = LogisticRegression(max_iter=2000, C=1.0)
    aucs_pc = cross_val_score(clf2, bulk_pcs, state_labels, cv=cv, scoring="roc_auc")
    pc_auc = float(aucs_pc.mean())
    print(f"  state-classification 5-fold AUC:  z={z_auc:.3f}  bulk-PCs={pc_auc:.3f}")

    out = {
        "n_active_dims": active,
        "n_near_collapsed_dims": near_collapsed,
        "z_var_per_dim_mean": float(var_per_dim.mean()),
        "z_var_per_dim_max": float(var_per_dim.max()),
        "z_var_per_dim_top5": [float(v) for v in np.sort(var_per_dim)[::-1][:5]],
        "z_pc1_explained": float(pca_z.explained_variance_ratio_[0]),
        "z_pc2_explained": float(pca_z.explained_variance_ratio_[1]),
        "max_abs_corr_z_with_bulkPCs": [float(v) for v in max_abs_per_pc],
        "anchor_state_classification_auc": {"z": z_auc, "bulk_pcs": pc_auc},
        "state_b_fraction": float(state_labels.mean()),
    }
    (RESULTS / "q2_latent_meaning.json").write_text(json.dumps(out, indent=2))

    # ── figure ────────────────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].bar(np.arange(64), np.sort(var_per_dim)[::-1], color="#3fb950")
    axes[0].axhline(0.01, color="#f78166", ls="--", lw=1, label="active threshold (0.01)")
    axes[0].axhline(1.0, color="#7d8590", ls=":", lw=1, label="N(0,1) prior")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Latent dim (sorted)")
    axes[0].set_ylabel("Variance across 803 donors")
    axes[0].set_title(f"Q2a: Latent activity\n{active}/64 active dims  → posterior collapse" if active < 5 else "Q2a: Latent activity per dim")
    axes[0].legend(fontsize=8)

    im = axes[1].imshow(np.abs(corr).T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axes[1].set_xlabel("Latent dim")
    axes[1].set_ylabel("Bulk PC")
    axes[1].set_title("Q2c: |corr(latent dim, bulk PC)|\n(should be ≥ 0.5 for some PCs if biology survives)")
    axes[1].set_yticks(range(10))
    axes[1].set_yticklabels([f"PC{i+1}" for i in range(10)])
    plt.colorbar(im, ax=axes[1])

    axes[2].bar(["z (latent)", "bulk PC1-10"], [z_auc, pc_auc], color=["#3fb950", "#58a6ff"])
    axes[2].axhline(0.5, color="#7d8590", ls="--", lw=1, label="chance")
    axes[2].set_ylim(0.4, 1.0)
    axes[2].set_ylabel("5-fold CV AUC")
    axes[2].set_title("Q2d: State-B (stress) classification\nfrom DDIT4 & FRAT1 anchor labels")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(OUT / "q2_latent_meaning.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {RESULTS / 'q2_latent_meaning.json'}")
    print(f"Wrote {OUT / 'q2_latent_meaning.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
