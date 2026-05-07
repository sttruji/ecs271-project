#!/usr/bin/env python3
"""Q14 — Pathway-aware structured-decoder VAE (variant B with soft sparsity).

DESIGN.  Same encoder as Sane-VAE (Q3/Q12), but the decoder is *masked* by
MSigDB Hallmark gene sets so each latent dim is anchored to a small set
of biologically interpretable pathways.

  Encoder:        x (n_genes) → 1024 → 512 → (μ, log σ²)   [identical to Sane-VAE]
  Decoder:        z (latent_dim) → linear W (n_clusters × latent_dim)
                                 → cluster scores (n_clusters)
                                 → x̂ = M @ cluster_scores + bias_per_gene
                  where M (n_genes × n_clusters) is FIXED at gene-set
                  membership; gene g writes from every cluster k it
                  belongs to.

SOFT SPARSITY.  An L1 penalty on W encourages each latent dim to write to
~3–5 dominant clusters rather than all 51.  Latent k's "meaning" can then
be read off as the top-3 clusters by |W[:, k]|.

CLUSTERS.  50 MSigDB Hallmark sets (v2024.1.Hs.symbols) + 1 OTHER bucket
catching the ~25% of shared genes that aren't in any hallmark.  This
guarantees every gene has at least one cluster path.

LATENT DIM.  50 — roughly one per hallmark, with the soft-sparsity term
deciding which latents merge multiple clusters.

LOSS.   total = MSE(x, x̂) + β·KL + λ_L1·‖W‖₁
        β = 1e-3 (matches Sane-VAE; the only knob preventing collapse).
        λ_L1 = 1e-3 (small enough not to crush the signal).

EVAL.
  (1) Held-out reconstruction R² on the GTEx 80/20 split (Q1/Q12 baseline).
  (2) Latent activity (active dims, var-per-dim).
  (3) Q12-style linear probes for AGE_mid / DTHHRDY / SMRIN / SMTSISCH
      / SEX / SMCENTER on the trained latent. Compare to Q12's PCA-50,
      Random-64, CM-VAE μ, Sane-VAE μ.
  (4) Interpretability — for each latent dim k, the top-3 hallmarks by
      |W[:, k]|.

Run:  python q14_structured_vae.py
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from lib_data import align_to_shared, load_gtex_blood, standardise  # noqa: E402
from lib_model import load_trained  # noqa: E402

CHECKPOINT = "/Users/rls/ecs271/data/models/cross_modality_vae.pt"
ANNOT = Path("/Users/rls/ecs271/data/annotations")
GMT_PATH = Path("/Users/rls/ecs271/data/genesets/h.all.v2024.1.Hs.symbols.gmt")

RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
RESULTS.mkdir(exist_ok=True, parents=True)
FIGS.mkdir(exist_ok=True, parents=True)

SEED = 0
LATENT_DIM = 50
EPOCHS = 200
BATCH = 64
LR = 1e-3
BETA = 1e-3
L1_LAMBDA = 1e-3


# ── GMT loader + cluster mask ─────────────────────────────────────────────
def load_gmt(path: Path) -> dict[str, list[str]]:
    sets: dict[str, list[str]] = {}
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name, _url, *genes = parts
            sets[name] = [g.upper() for g in genes if g]
    return sets


def build_cluster_mask(
    shared_genes: np.ndarray, gene_sets: dict[str, list[str]]
) -> tuple[np.ndarray, list[str]]:
    """Return M of shape (n_genes, n_clusters) with binary membership +
    cluster_names.  Adds a final 'OTHER' cluster covering shared_genes
    not in any hallmark set."""
    upper = np.array([str(g).upper() for g in shared_genes])
    cluster_names = list(gene_sets.keys())
    n_genes = len(upper)
    n_clust = len(cluster_names) + 1  # +1 OTHER bucket
    M = np.zeros((n_genes, n_clust), dtype=np.float32)
    set_to_idx = {n: i for i, n in enumerate(cluster_names)}
    in_any = np.zeros(n_genes, dtype=bool)
    for k, name in enumerate(cluster_names):
        gset = set(gene_sets[name])
        idx = np.where(np.isin(upper, list(gset)))[0]
        if len(idx):
            M[idx, k] = 1.0
            in_any[idx] = True
    other = ~in_any
    M[other, -1] = 1.0
    cluster_names = cluster_names + ["OTHER"]
    return M, cluster_names


# ── Model ────────────────────────────────────────────────────────────────
class StructuredVAE(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_clusters: int,
        cluster_mask: np.ndarray,
        latent_dim: int = LATENT_DIM,
    ):
        super().__init__()
        # Encoder identical to Sane-VAE
        self.enc_trunk = nn.Sequential(
            nn.Linear(n_genes, 1024), nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2), nn.Dropout(0.1),
            nn.Linear(1024, 512), nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2), nn.Dropout(0.1),
        )
        self.mu = nn.Linear(512, latent_dim)
        self.logv = nn.Linear(512, latent_dim)
        # Structured decoder
        self.latent_to_cluster = nn.Linear(latent_dim, n_clusters, bias=False)
        self.gene_bias = nn.Parameter(torch.zeros(n_genes))
        # Fixed gene-cluster mask (n_genes, n_clusters)
        self.register_buffer(
            "gene_mask", torch.from_numpy(cluster_mask).float()
        )

    def encode(self, x):
        h = self.enc_trunk(x)
        return self.mu(h), self.logv(h).clamp(-10, 4)

    def decode(self, z):
        cluster_scores = self.latent_to_cluster(z)         # (B, n_clusters)
        # x̂[B, g] = bias[g] + Σ_k mask[g, k] * cluster_score[B, k]
        x_hat = cluster_scores @ self.gene_mask.T + self.gene_bias
        return x_hat, cluster_scores

    def forward(self, x):
        mu, logv = self.encode(x)
        if self.training:
            z = mu + (0.5 * logv).exp() * torch.randn_like(mu)
        else:
            z = mu
        x_hat, cluster_scores = self.decode(z)
        return x_hat, mu, logv, cluster_scores


# ── Training loop ─────────────────────────────────────────────────────────
def train_structured(
    train_t: torch.Tensor,
    test_t: torch.Tensor,
    cluster_mask: np.ndarray,
    *,
    device: str,
    epochs: int = EPOCHS,
    lr: float = LR,
    batch: int = BATCH,
    beta: float = BETA,
    l1: float = L1_LAMBDA,
) -> tuple[StructuredVAE, dict]:
    torch.manual_seed(SEED)
    n_genes = cluster_mask.shape[0]
    n_clusters = cluster_mask.shape[1]
    model = StructuredVAE(n_genes, n_clusters, cluster_mask).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loader = DataLoader(
        TensorDataset(train_t.to(device)),
        batch_size=batch, shuffle=True, drop_last=True,
    )
    history = {"train_recon": [], "test_recon": [], "kl": [], "l1": []}
    for ep in range(1, epochs + 1):
        model.train()
        ep_recon = ep_kl = ep_l1 = 0.0
        nb = 0
        for (xb,) in loader:
            opt.zero_grad()
            xh, mu, logv, _ = model(xb)
            recon = F.mse_loss(xh, xb)
            kl = -0.5 * (1 + logv - mu.pow(2) - logv.exp()).sum(-1).mean()
            l1_pen = model.latent_to_cluster.weight.abs().mean()
            loss = recon + beta * kl + l1 * l1_pen
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_recon += float(recon)
            ep_kl += float(kl)
            ep_l1 += float(l1_pen)
            nb += 1
        model.eval()
        with torch.no_grad():
            xh, _, _, _ = model(test_t.to(device))
            te_recon = float(F.mse_loss(xh, test_t.to(device)))
        history["train_recon"].append(ep_recon / nb)
        history["test_recon"].append(te_recon)
        history["kl"].append(ep_kl / nb)
        history["l1"].append(ep_l1 / nb)
        if ep == 1 or ep % 25 == 0 or ep == epochs:
            print(
                f"  [Struct-VAE] ep {ep:>3}/{epochs}  "
                f"train MSE={history['train_recon'][-1]:.4f}  "
                f"test MSE={te_recon:.4f}  KL={history['kl'][-1]:.3f}  "
                f"L1={history['l1'][-1]:.4f}"
            )
    return model, history


def encode_struct(model: StructuredVAE, x: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(x.astype(np.float32)).to(device)
        h = model.enc_trunk(xt)
        mu = model.mu(h)
    return mu.cpu().numpy()


# ── Probes (lifted from Q12) ──────────────────────────────────────────────
def cv_ridge_r2(z: np.ndarray, y: np.ndarray, *, k: int = 5) -> dict:
    mask = np.isfinite(y) & np.isfinite(z).all(axis=1)
    if mask.sum() < 30:
        return {"r2": float("nan"), "r2_std": float("nan"), "n": int(mask.sum())}
    zk = StandardScaler().fit_transform(z[mask])
    yk = y[mask]
    cv = KFold(n_splits=k, shuffle=True, random_state=SEED)
    scores = cross_val_score(Ridge(alpha=1.0), zk, yk, cv=cv, scoring="r2")
    return {"r2": float(scores.mean()), "r2_std": float(scores.std()),
            "n": int(mask.sum())}


def cv_logreg(z: np.ndarray, y: np.ndarray, *, k: int = 5,
              min_per_class: int = 5) -> dict:
    finite = np.isfinite(z).all(axis=1)
    y_arr = pd.Series(y).where(pd.Series(y).notna()).astype("object")
    valid = finite & y_arr.notna().values
    if valid.sum() < 30:
        return {"bal_acc": float("nan"), "macro_auc": float("nan"),
                "n": int(valid.sum()), "n_classes": 0}
    zv = StandardScaler().fit_transform(z[valid])
    yv = y_arr.values[valid].astype(str)
    classes, counts = np.unique(yv, return_counts=True)
    keep = classes[counts >= min_per_class]
    keep_mask = np.isin(yv, keep)
    if keep_mask.sum() < 30 or len(keep) < 2:
        return {"bal_acc": float("nan"), "macro_auc": float("nan"),
                "n": int(keep_mask.sum()), "n_classes": int(len(keep))}
    zv, yv = zv[keep_mask], yv[keep_mask]
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                             multi_class="auto")
    bal = cross_val_score(clf, zv, yv, cv=cv, scoring="balanced_accuracy")
    auc_metric = "roc_auc_ovr_weighted" if len(keep) > 2 else "roc_auc"
    try:
        auc = cross_val_score(clf, zv, yv, cv=cv, scoring=auc_metric)
        macro_auc = float(auc.mean())
    except Exception:
        macro_auc = float("nan")
    return {"bal_acc": float(bal.mean()), "macro_auc": macro_auc,
            "n": int(keep_mask.sum()), "n_classes": int(len(keep))}


# ── Metadata join (lifted from Q12) ───────────────────────────────────────
def sample_to_subject(sample_id: str) -> str:
    return "-".join(sample_id.split("-")[:2])


def age_to_midpoint(age_band: str) -> float:
    if not isinstance(age_band, str) or "-" not in age_band:
        return float("nan")
    a, b = age_band.split("-")
    try:
        return (int(a) + int(b)) / 2
    except ValueError:
        return float("nan")


def load_gtex_metadata(sample_ids: list[str]) -> pd.DataFrame:
    sub = pd.read_csv(ANNOT / "GTEx_v10_Annotations_SubjectPhenotypesDS.txt", sep="\t")
    samp = pd.read_csv(
        ANNOT / "GTEx_v10_Annotations_SampleAttributesDS.txt",
        sep="\t", low_memory=False,
    )
    df = pd.DataFrame({"SAMPID": sample_ids})
    df["SUBJID"] = df["SAMPID"].apply(sample_to_subject)
    df = df.merge(sub, on="SUBJID", how="left")
    df = df.merge(
        samp[["SAMPID", "SMRIN", "SMTSISCH", "SMCENTER",
              "SMNABTCH", "SMGEBTCH", "SMRDLGTH"]],
        on="SAMPID", how="left",
    )
    df["AGE_mid"] = df["AGE"].apply(age_to_midpoint)
    return df


# ── Main ──────────────────────────────────────────────────────────────────
def main() -> int:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"Device: {device}")

    # GTEx + scaler (same shared_genes basis as Q12)
    print("\n[1/5] Loading GTEx + shared-gene basis ...")
    _, shared_genes, scaler_mean, scaler_std = load_trained(CHECKPOINT, device="cpu")
    expr_log, gene_names = load_gtex_blood()
    expr_aligned, _, _ = align_to_shared(expr_log, gene_names, shared_genes)
    X = standardise(expr_aligned, scaler_mean, scaler_std)
    n_samples, n_genes = X.shape
    print(f"  X: {X.shape}")

    # Hallmark cluster mask
    print("\n[2/5] Building Hallmark cluster mask ...")
    gene_sets = load_gmt(GMT_PATH)
    print(f"  loaded {len(gene_sets)} hallmark sets")
    M, cluster_names = build_cluster_mask(shared_genes, gene_sets)
    coverage = (M.sum(axis=1) > 0).mean()
    other_count = int(M[:, -1].sum())
    cluster_sizes = M.sum(axis=0).astype(int)
    print(f"  M shape: {M.shape}  (genes × clusters incl. OTHER)")
    print(f"  gene coverage by ≥1 hallmark: {1 - other_count/n_genes:.1%}")
    print(f"  OTHER bucket size: {other_count} / {n_genes}")
    print(f"  cluster size — min={cluster_sizes.min()}  "
          f"med={int(np.median(cluster_sizes))}  max={cluster_sizes.max()}")

    # 80/20 split (same seed as Q3/Q12)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_samples)
    n_test = int(round(0.2 * n_samples))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    train_t = torch.from_numpy(X[train_idx])
    test_t = torch.from_numpy(X[test_idx])

    # Train (or load cached)
    ckpt_path = RESULTS / "q14_structured_vae.pt"
    if ckpt_path.exists():
        print(f"\n[3/5] Loading cached structured VAE from {ckpt_path}")
        cached = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = StructuredVAE(n_genes, M.shape[1], M).to(device)
        model.load_state_dict(cached["state_dict"])
        hist = cached.get("history", {})
        r2_holdout = float(cached.get("holdout_r2", float("nan")))
        print(f"  cached held-out R² = {r2_holdout:.3f}")
    else:
        print(f"\n[3/5] Training Structured VAE  (latent={LATENT_DIM}, "
              f"β={BETA}, λ_L1={L1_LAMBDA}, epochs={EPOCHS}) ...")
        model, hist = train_structured(train_t, test_t, M, device=device)
        model.eval()
        with torch.no_grad():
            xh, _, _, _ = model(test_t.to(device))
        pred_te = xh.cpu().numpy()
        true_te = X[test_idx]
        diff = true_te - pred_te
        r2_holdout = float(1 - diff.var() / true_te.var())
        print(f"  held-out R² = {r2_holdout:.3f}")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "shared_genes": shared_genes,
                "scaler_mean": scaler_mean,
                "scaler_std": scaler_std,
                "cluster_mask": M,
                "cluster_names": cluster_names,
                "latent_dim": LATENT_DIM,
                "history": hist,
                "holdout_r2": r2_holdout,
                "beta": BETA,
                "l1_lambda": L1_LAMBDA,
            },
            ckpt_path,
        )
        print(f"  saved {ckpt_path}")

    # Encode all 803 donors
    Z = encode_struct(model, X, device=device)
    var_per = Z.var(0)
    n_active = int((var_per > 0.01).sum())
    print(f"  Z: {Z.shape}  active>0.01: {n_active}/{LATENT_DIM}  "
          f"mean var={var_per.mean():.4f}")

    # ── (4/5) Metadata probes ─────────────────────────────────────────────
    print("\n[4/5] Running metadata probes (same set as Q12) ...")
    # Sample IDs
    with gzip.open("/Users/rls/ecs271/data/bulk/gtex_v11_whole_blood.gct.gz", "rt") as fh:
        for _ in range(2):
            fh.readline()
        header = fh.readline().rstrip("\n").split("\t")
    sample_ids = header[2:]
    df = load_gtex_metadata(sample_ids)

    cont_targets = ["AGE_mid", "DTHHRDY", "SMRIN", "SMTSISCH", "SMRDLGTH"]
    cat_targets = ["SEX", "SMCENTER"]
    rows = []
    for t in cont_targets:
        y = pd.to_numeric(df[t], errors="coerce").values
        res = cv_ridge_r2(Z, y)
        rows.append({"latent": "Struct-VAE", "target": t,
                     "metric": "R2", "value": res["r2"], "std": res["r2_std"],
                     "n": res["n"], "n_classes": np.nan})
        print(f"  Struct-VAE  {t:<10}  R² = {res['r2']:+.3f} ± {res['r2_std']:.3f}  (n={res['n']})")
    for t in cat_targets:
        y = df[t].values
        res = cv_logreg(Z, y)
        rows.append({"latent": "Struct-VAE", "target": t,
                     "metric": "bal_acc", "value": res["bal_acc"],
                     "std": float("nan"), "n": res["n"],
                     "n_classes": res["n_classes"]})
        rows.append({"latent": "Struct-VAE", "target": t,
                     "metric": "macro_auc", "value": res["macro_auc"],
                     "std": float("nan"), "n": res["n"],
                     "n_classes": res["n_classes"]})
        print(f"  Struct-VAE  {t:<10}  balAcc={res['bal_acc']:.3f}  "
              f"AUC={res['macro_auc']:.3f}  (n={res['n']}, classes={res['n_classes']})")

    pd.DataFrame(rows).to_csv(RESULTS / "q14_metadata_probe.csv", index=False)

    # ── (5/5) Interpretability — top-3 clusters per latent dim ────────────
    print("\n[5/5] Latent → cluster attribution (top-3 by |W|) ...")
    W = model.latent_to_cluster.weight.detach().cpu().numpy()  # (n_clusters, latent_dim)
    n_clust = W.shape[0]
    interp_rows = []
    for k in range(LATENT_DIM):
        wk = W[:, k]
        order = np.argsort(-np.abs(wk))
        top3 = [(cluster_names[i], float(wk[i])) for i in order[:3]]
        interp_rows.append({
            "latent_dim": k,
            "latent_var": float(var_per[k]),
            "top1_cluster": top3[0][0],
            "top1_w": top3[0][1],
            "top2_cluster": top3[1][0],
            "top2_w": top3[1][1],
            "top3_cluster": top3[2][0],
            "top3_w": top3[2][1],
            # how concentrated this latent's contribution is
            "frac_top3_l1": float(
                (np.abs(wk[order[:3]]).sum()) / (np.abs(wk).sum() + 1e-9)
            ),
        })
    interp_df = pd.DataFrame(interp_rows).sort_values(
        "latent_var", ascending=False
    )
    interp_df.to_csv(RESULTS / "q14_latent_to_cluster.csv", index=False)
    print(f"  Saved {RESULTS / 'q14_latent_to_cluster.csv'}")

    print("\nTop 12 most-active latent dims, with their dominant clusters:")
    for _, r in interp_df.head(12).iterrows():
        print(
            f"  z[{int(r['latent_dim']):>2}]  "
            f"var={r['latent_var']:.3f}  "
            f"top3 L1-share={r['frac_top3_l1']:.2f}  "
            f"|  {r['top1_cluster']}({r['top1_w']:+.2f})  "
            f"{r['top2_cluster']}({r['top2_w']:+.2f})  "
            f"{r['top3_cluster']}({r['top3_w']:+.2f})"
        )

    summary = {
        "holdout_r2": r2_holdout,
        "n_active_dims": n_active,
        "mean_top3_l1_share": float(interp_df["frac_top3_l1"].mean()),
        "median_top3_l1_share": float(interp_df["frac_top3_l1"].median()),
        "probes": rows,
        "config": {
            "latent_dim": LATENT_DIM,
            "n_clusters": n_clust,
            "beta": BETA,
            "l1_lambda": L1_LAMBDA,
            "epochs": EPOCHS,
        },
    }
    (RESULTS / "q14_structured_vae.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"\nMean top-3 cluster L1-share per latent dim: "
          f"{summary['mean_top3_l1_share']:.2%}")
    print(f"  (1.0 = latent writes only to 3 clusters; "
          f"3/51≈0.06 = totally diffuse)")

    # ── Figure ────────────────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Panel 1: training curves
    ep = np.arange(1, len(hist.get("train_recon", [])) + 1)
    if len(ep):
        axes[0].plot(ep, hist["train_recon"], color="#3fb950", label="train MSE")
        axes[0].plot(ep, hist["test_recon"], color="#3fb950", ls="--",
                     label="test MSE")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("MSE on standardised log2(CPM+1)")
        axes[0].set_title(f"Q14a — Structured VAE training\n"
                          f"held-out R²={r2_holdout:.3f}, active "
                          f"{n_active}/{LATENT_DIM}")
        axes[0].legend(fontsize=9)

    # Panel 2: |W| heatmap, latents (rows) × clusters (cols)
    # Show only the top-30 most-active latents and clusters with any signal
    top_lat = interp_df["latent_dim"].head(30).values
    cluster_max = np.abs(W).max(axis=1)
    keep_clust = np.argsort(-cluster_max)[:30]
    sub_W = W[keep_clust][:, top_lat]
    im = axes[1].imshow(sub_W, aspect="auto", cmap="RdBu_r",
                        vmin=-np.abs(sub_W).max(), vmax=np.abs(sub_W).max())
    axes[1].set_yticks(range(len(keep_clust)))
    axes[1].set_yticklabels(
        [cluster_names[i].replace("HALLMARK_", "")[:25] for i in keep_clust],
        fontsize=7,
    )
    axes[1].set_xticks(range(len(top_lat)))
    axes[1].set_xticklabels([f"z{i}" for i in top_lat], fontsize=7,
                            rotation=90)
    axes[1].set_title("Q14b — Latent → cluster decoder weights\n"
                      "(top 30 most-active latents × top 30 cluster rows)")
    plt.colorbar(im, ax=axes[1], label="W[k,j]")

    fig.tight_layout()
    fig.savefig(FIGS / "q14_structured_vae.png", dpi=150)
    plt.close(fig)
    print(f"  Wrote {FIGS / 'q14_structured_vae.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
