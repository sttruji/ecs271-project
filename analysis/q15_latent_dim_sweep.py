#!/usr/bin/env python3
"""Q15 — Latent-dimensionality sweep for the Sane VAE.

Question.  How does reconstruction quality and metadata recoverability change
as we shrink or grow the latent bottleneck?  Q12/Q13/Q14 used a fixed 64-D
(VAEs) or 50-D (PCA) latent.  Here we sweep the same Sane-VAE backbone
(β = 1e-3, MLP encoder/decoder, 200 epochs Adam(1e-3), batch=64) across
latent_dim ∈ {4, 8, 16, 32, 64, 128, 256}, and compare to PCA-N for each N.

Metrics tracked at each dim.
  • Held-out reconstruction R² (Sane-VAE μ vs PCA-N at the same N)
  • Active latent dims (var > 0.01)
  • AGE_mid linear-probe R²  (5-fold CV ridge probe, same as Q12)
  • SEX linear-probe macro-AUC  (5-fold CV logistic probe, same as Q12)

Caching.  Each (latent_dim, model) checkpoint is saved to
analysis/results/q15_sweep_d{N}.pt and reloaded on rerun, so iterating on
the figure / probes does not require retraining.

Outputs.
  results/q15_sweep.csv              tidy table
  results/q15_sweep.json             compact summary
  figures/q15_latent_dim_sweep.png   2x2 panel
  results/q15_sweep_d{N}.pt          per-dim VAE checkpoint cache

Run:  python q15_latent_dim_sweep.py
"""
from __future__ import annotations

import gzip
import json
import sys
import time
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
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from lib_data import align_to_shared, load_gtex_blood, standardise  # noqa: E402
from lib_model import load_trained  # noqa: E402

CHECKPOINT = "/Users/rls/ecs271/data/models/cross_modality_vae.pt"
ANNOT = Path("/Users/rls/ecs271/data/annotations")
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
RESULTS.mkdir(exist_ok=True, parents=True)
FIGS.mkdir(exist_ok=True, parents=True)

LATENT_DIMS = [4, 8, 16, 32, 64, 128, 256]
SEED = 0
EPOCHS = 200
LR = 1e-3
BATCH = 64
BETA = 1e-3


# ── Sane VAE — same arch as Q3/Q12 but parameterised by latent_dim ───────
class SaneVAE(nn.Module):
    def __init__(self, n_genes: int, latent_dim: int):
        super().__init__()
        self.enc_trunk = nn.Sequential(
            nn.Linear(n_genes, 1024), nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2), nn.Dropout(0.1),
            nn.Linear(1024, 512), nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2), nn.Dropout(0.1),
        )
        self.mu = nn.Linear(512, latent_dim)
        self.logv = nn.Linear(512, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2), nn.Dropout(0.1),
            nn.Linear(512, 1024), nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2), nn.Dropout(0.1),
            nn.Linear(1024, n_genes),
        )

    def forward(self, x):
        h = self.enc_trunk(x)
        mu = self.mu(h)
        logv = self.logv(h).clamp(-10, 4)
        if self.training:
            z = mu + (0.5 * logv).exp() * torch.randn_like(mu)
        else:
            z = mu
        return self.decoder(z), mu, logv


def train_one(
    train_t: torch.Tensor, test_t: torch.Tensor, n_genes: int,
    latent_dim: int, *, device: str, epochs: int = EPOCHS,
    beta: float = BETA, lr: float = LR, batch: int = BATCH,
) -> tuple[SaneVAE, dict]:
    torch.manual_seed(SEED)
    model = SaneVAE(n_genes, latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loader = DataLoader(
        TensorDataset(train_t.to(device)),
        batch_size=batch, shuffle=True, drop_last=True,
    )
    history = {"train_recon": [], "test_recon": [], "kl": []}
    for ep in range(1, epochs + 1):
        model.train()
        ep_recon = ep_kl = 0.0
        nb = 0
        for (xb,) in loader:
            opt.zero_grad()
            xh, mu, logv = model(xb)
            recon = F.mse_loss(xh, xb)
            kl = -0.5 * (1 + logv - mu.pow(2) - logv.exp()).sum(-1).mean()
            loss = recon + beta * kl
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_recon += float(recon)
            ep_kl += float(kl)
            nb += 1
        model.eval()
        with torch.no_grad():
            xh, _, _ = model(test_t.to(device))
            te_recon = float(F.mse_loss(xh, test_t.to(device)))
        history["train_recon"].append(ep_recon / nb)
        history["test_recon"].append(te_recon)
        history["kl"].append(ep_kl / nb)
    return model, history


def encode_mu(model: SaneVAE, x: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(x.astype(np.float32)).to(device)
        h = model.enc_trunk(xt)
        mu = model.mu(h)
    return mu.cpu().numpy()


# ── Probes (same as Q12) ──────────────────────────────────────────────────
def cv_ridge_r2(z: np.ndarray, y: np.ndarray, *, k: int = 5) -> float:
    mask = np.isfinite(y) & np.isfinite(z).all(axis=1)
    if mask.sum() < 30:
        return float("nan")
    zk = StandardScaler().fit_transform(z[mask])
    cv = KFold(n_splits=k, shuffle=True, random_state=SEED)
    scores = cross_val_score(Ridge(alpha=1.0), zk, y[mask], cv=cv, scoring="r2")
    return float(scores.mean())


def cv_logreg_auc(z: np.ndarray, y: np.ndarray, *, k: int = 5) -> float:
    finite = np.isfinite(z).all(axis=1)
    y_arr = pd.Series(y).where(pd.Series(y).notna()).astype("object")
    valid = finite & y_arr.notna().values
    if valid.sum() < 30:
        return float("nan")
    zv = StandardScaler().fit_transform(z[valid])
    yv = y_arr.values[valid].astype(str)
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    metric = "roc_auc_ovr_weighted" if len(np.unique(yv)) > 2 else "roc_auc"
    try:
        return float(cross_val_score(clf, zv, yv, cv=cv, scoring=metric).mean())
    except Exception:
        return float("nan")


# ── Metadata join (same as Q12) ───────────────────────────────────────────
def sample_to_subject(s: str) -> str:
    return "-".join(s.split("-")[:2])


def age_to_midpoint(b: str) -> float:
    if not isinstance(b, str) or "-" not in b:
        return float("nan")
    a, c = b.split("-")
    try:
        return (int(a) + int(c)) / 2
    except ValueError:
        return float("nan")


def load_gtex_metadata(sample_ids: list[str]) -> pd.DataFrame:
    sub = pd.read_csv(ANNOT / "GTEx_v10_Annotations_SubjectPhenotypesDS.txt", sep="\t")
    df = pd.DataFrame({"SAMPID": sample_ids})
    df["SUBJID"] = df["SAMPID"].apply(sample_to_subject)
    df = df.merge(sub, on="SUBJID", how="left")
    df["AGE_mid"] = df["AGE"].apply(age_to_midpoint)
    return df


# ── Main ──────────────────────────────────────────────────────────────────
def main() -> int:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"Device: {device}")

    print("\n[1/4] Loading GTEx + setup ...")
    _, shared_genes, scaler_mean, scaler_std = load_trained(CHECKPOINT, device="cpu")
    expr_log, gene_names = load_gtex_blood()
    expr_aligned, _, _ = align_to_shared(expr_log, gene_names, shared_genes)
    X = standardise(expr_aligned, scaler_mean, scaler_std)
    n_samples, n_genes = X.shape
    print(f"  X: {X.shape}")

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_samples)
    n_test = int(round(0.2 * n_samples))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    X_train, X_test = X[train_idx], X[test_idx]

    # Sample IDs (for metadata join)
    with gzip.open("/Users/rls/ecs271/data/bulk/gtex_v11_whole_blood.gct.gz", "rt") as fh:
        for _ in range(2):
            fh.readline()
        header = fh.readline().rstrip("\n").split("\t")
    sample_ids = header[2:]
    df_meta = load_gtex_metadata(sample_ids)
    age_y = pd.to_numeric(df_meta["AGE_mid"], errors="coerce").values
    sex_y = df_meta["SEX"].astype(str).values

    # ── (2/4) PCA-N reference for each latent_dim in the sweep ───────────
    print("\n[2/4] Computing PCA references for each N ...")
    pca_results = {}
    for N in LATENT_DIMS:
        pca = PCA(n_components=min(N, X_train.shape[0] - 1),
                  random_state=SEED).fit(X_train)
        Z_train = pca.transform(X_train)
        Z_test = pca.transform(X_test)
        X_test_hat = pca.inverse_transform(Z_test)
        diff = X_test - X_test_hat
        pca_r2 = float(1 - diff.var() / X_test.var())
        # probes on full-set PCA for comparability with Q12
        Z_all = pca.transform(X)
        age_r2 = cv_ridge_r2(Z_all, age_y)
        sex_auc = cv_logreg_auc(Z_all, sex_y)
        pca_results[N] = {
            "r2": pca_r2, "age_r2": age_r2, "sex_auc": sex_auc,
            "active_dims": int(min(N, X_train.shape[0] - 1)),  # PCA dims always active
        }
        print(f"  PCA-{N:>3}  recon R²={pca_r2:.3f}  AGE R²={age_r2:+.3f}  SEX AUC={sex_auc:.3f}")

    # ── (3/4) Sane VAE sweep ────────────────────────────────────────────
    print("\n[3/4] Sane VAE sweep ...")
    train_t = torch.from_numpy(X_train)
    test_t = torch.from_numpy(X_test)
    vae_results = {}
    for N in LATENT_DIMS:
        ckpt_path = RESULTS / f"q15_sweep_d{N}.pt"
        if ckpt_path.exists():
            print(f"  [d={N}] loading cached {ckpt_path.name}")
            cached = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = SaneVAE(n_genes, N).to(device)
            model.load_state_dict(cached["state_dict"])
            history = cached.get("history", {})
            r2_holdout = float(cached.get("holdout_r2", float("nan")))
        else:
            t0 = time.time()
            print(f"  [d={N}] training (β={BETA}, epochs={EPOCHS}, batch={BATCH}) ...")
            model, history = train_one(
                train_t, test_t, n_genes, N, device=device,
            )
            model.eval()
            with torch.no_grad():
                xh, _, _ = model(test_t.to(device))
            pred_te = xh.cpu().numpy()
            diff = X_test - pred_te
            r2_holdout = float(1 - diff.var() / X_test.var())
            elapsed = time.time() - t0
            print(f"  [d={N}]   recon R²={r2_holdout:.3f}  "
                  f"final train MSE={history['train_recon'][-1]:.3f}  "
                  f"final test MSE={history['test_recon'][-1]:.3f}  "
                  f"({elapsed:.0f}s)")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "shared_genes": shared_genes,
                    "scaler_mean": scaler_mean, "scaler_std": scaler_std,
                    "latent_dim": N,
                    "history": history,
                    "holdout_r2": r2_holdout,
                },
                ckpt_path,
            )
        # encode all 803 donors and probe
        Z_all = encode_mu(model, X, device=device)
        var_per = Z_all.var(0)
        active = int((var_per > 0.01).sum())
        age_r2 = cv_ridge_r2(Z_all, age_y)
        sex_auc = cv_logreg_auc(Z_all, sex_y)
        vae_results[N] = {
            "r2": r2_holdout,
            "age_r2": age_r2,
            "sex_auc": sex_auc,
            "active_dims": active,
            "active_frac": active / N,
            "mean_var": float(var_per.mean()),
            "final_train_mse": (history.get("train_recon", [float("nan")])[-1]
                                if history else float("nan")),
            "final_test_mse": (history.get("test_recon", [float("nan")])[-1]
                               if history else float("nan")),
        }
        print(f"  [d={N}]   active {active}/{N} ({active/N:.0%}), "
              f"AGE R²={age_r2:+.3f}, SEX AUC={sex_auc:.3f}")

    # ── (4/4) Save tidy table + figure ─────────────────────────────────
    print("\n[4/4] Writing outputs ...")
    rows = []
    for N in LATENT_DIMS:
        for kind, src in [("PCA", pca_results[N]), ("Sane-VAE", vae_results[N])]:
            rows.append({
                "latent_dim": N, "kind": kind,
                "recon_r2": src["r2"],
                "age_r2": src["age_r2"],
                "sex_auc": src["sex_auc"],
                "active_dims": src.get("active_dims", N),
                "active_frac": (src.get("active_dims", N) / N),
                "final_train_mse": src.get("final_train_mse", float("nan")),
                "final_test_mse": src.get("final_test_mse", float("nan")),
            })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "q15_sweep.csv", index=False)
    (RESULTS / "q15_sweep.json").write_text(json.dumps({
        "latent_dims": LATENT_DIMS,
        "config": {"epochs": EPOCHS, "beta": BETA, "lr": LR, "batch": BATCH},
        "rows": rows,
    }, indent=2))
    print(f"  Wrote {RESULTS / 'q15_sweep.csv'}")

    # ── Figure ──────────────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Q15 — Latent-dimension sweep  (Sane VAE β=1e-3 vs PCA, "
                 "GTEx whole blood n=803)", fontsize=12)

    xs = LATENT_DIMS

    # (a) recon R²
    ax = axes[0, 0]
    pca_r2 = [pca_results[N]["r2"] for N in xs]
    vae_r2 = [vae_results[N]["r2"] for N in xs]
    ax.plot(xs, pca_r2, "-o", color="#58a6ff", label="PCA-N (linear)", lw=2)
    ax.plot(xs, vae_r2, "-s", color="#3fb950", label="Sane VAE μ", lw=2)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Latent dim")
    ax.set_ylabel("Held-out reconstruction R²")
    ax.set_title("(a) Reconstruction quality vs latent dim")
    ax.set_xticks(xs)
    ax.set_xticklabels(xs)
    ax.axhline(0.86, color="#7d8590", ls=":", lw=0.6)  # PCA-50 of Q12
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    for x, v in zip(xs, vae_r2):
        ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="#3fb950")

    # (b) Active dims (VAE only)
    ax = axes[0, 1]
    active = [vae_results[N]["active_dims"] for N in xs]
    frac = [vae_results[N]["active_frac"] for N in xs]
    ax.bar([str(N) for N in xs], active, color="#3fb950", alpha=0.7,
           label="active dims (var > 0.01)")
    ax.plot([str(N) for N in xs], xs, "-o", color="#7d8590",
            label="total dims (= N)")
    ax.set_xlabel("Latent dim")
    ax.set_ylabel("Number of dims")
    ax.set_title("(b) Active dims vs latent capacity\n"
                 "(VAEs collapse extra capacity once N > sample diversity)")
    ax.legend(loc="upper left")
    for x, (a, f) in enumerate(zip(active, frac)):
        ax.text(x, a + 1, f"{f:.0%}", ha="center", fontsize=8)

    # (c) AGE_mid probe
    ax = axes[1, 0]
    pca_age = [pca_results[N]["age_r2"] for N in xs]
    vae_age = [vae_results[N]["age_r2"] for N in xs]
    ax.plot(xs, pca_age, "-o", color="#58a6ff", label="PCA-N")
    ax.plot(xs, vae_age, "-s", color="#3fb950", label="Sane VAE μ")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(xs)
    ax.set_xlabel("Latent dim")
    ax.set_ylabel("AGE_mid linear-probe R²")
    ax.set_title("(c) AGE_mid recovery vs latent dim")
    ax.axhline(0.0, color="#7d8590", lw=0.6)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)

    # (d) SEX probe
    ax = axes[1, 1]
    pca_sex = [pca_results[N]["sex_auc"] for N in xs]
    vae_sex = [vae_results[N]["sex_auc"] for N in xs]
    ax.plot(xs, pca_sex, "-o", color="#58a6ff", label="PCA-N")
    ax.plot(xs, vae_sex, "-s", color="#3fb950", label="Sane VAE μ")
    ax.axhline(0.5, color="#7d8590", ls=":", lw=0.6, label="chance (AUC=0.5)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(xs)
    ax.set_xlabel("Latent dim")
    ax.set_ylabel("SEX classifier macro-AUC")
    ax.set_title("(d) SEX recovery vs latent dim")
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIGS / "q15_latent_dim_sweep.png", dpi=150)
    plt.close(fig)
    print(f"  Wrote {FIGS / 'q15_latent_dim_sweep.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
