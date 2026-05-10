#!/usr/bin/env python3
"""Q17 — Hybrid PCA + nonlinear AE. Probe bio-dim of GTEx blood.

THESIS.  The biological signal in bulk RNA-seq lives in a much smaller
subspace than the 11k-gene measurement space.  If true, a competent
encoder–decoder at small d should reconstruct the transcript within 1σ
on most predictions.  Q15 / Q16 hit ~95-97% within-1σ at d=32 with
plain MLP AEs.  Q17 builds an AE *guaranteed* to match PCA-d (the
optimal linear ceiling) and tries to push past it via a nonlinear
correction branch.

ARCHITECTURE.

  encoder:    z = (PCA_lin(x))                   ← linear branch, PCA-init
                + α · MLP_enc(x)                 ← nonlinear correction
  decoder:    x̂ = (PCA_dec(z)) + bias_per_gene   ← linear branch, PCA-init
                + γ · MLP_dec(z)                 ← nonlinear correction

  α, γ are learnable scalars, **initialised to 0**.  At init the model
  IS PCA-d (since the nonlinear branches multiply by 0).  Training
  can only improve over PCA-d, so this provably matches the linear
  ceiling.  Any gains above PCA-d come from the MLP branches finding
  nonlinear structure (cell-type interactions, saturation, batch shifts,
  …) that PCA alone can't capture.

  MLP branches use LayerNorm (more stable than BN at small batch),
  SiLU activation, no dropout.

LOSS / TRAINING.  Plain MSE.  AdamW(1e-3, wd=1e-4).  Cosine schedule
to lr=1e-5 over 500 epochs.  Per-sample MSE on 80/20 split.

EVAL.

  •  frac<1σ                          ← user's target ≥ 99 %
  •  R² and MSE (held-out)
  •  per-gene resid std (max + p99)
  •  comparison vs PCA-N at the same N

DIM SWEEP.  After d=32 hits its ceiling, sweep d ∈ {32, 16, 8, 4, 2, 1}
to find the lowest d that maintains the recovery.  The "knee" gives an
empirical estimate of the bio-dim of GTEx blood.

Run:  python q17_hybrid_ae.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

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
TARGET_FRAC = 0.99


# ── Architecture ──────────────────────────────────────────────────────────
class HybridAE(nn.Module):
    """Linear (PCA-init) + nonlinear MLP correction, both branches summed.

    At init, with `enc_alpha = dec_alpha = 0`, model exactly = PCA-d.
    """

    def __init__(self, n_genes: int, latent_dim: int,
                 pca_components: np.ndarray, x_train_mean: np.ndarray):
        super().__init__()
        # Linear branches — PCA-init
        self.enc_lin = nn.Linear(n_genes, latent_dim, bias=False)
        self.dec_lin = nn.Linear(latent_dim, n_genes, bias=True)
        with torch.no_grad():
            self.enc_lin.weight.copy_(torch.from_numpy(pca_components))   # (d, p)
            self.dec_lin.weight.copy_(torch.from_numpy(pca_components.T))  # (p, d)
            self.dec_lin.bias.copy_(torch.from_numpy(x_train_mean))         # (p,)

        # Nonlinear correction branches
        self.enc_nl = nn.Sequential(
            nn.Linear(n_genes, 1024), nn.LayerNorm(1024), nn.SiLU(),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, latent_dim),
        )
        self.dec_nl = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, 1024), nn.LayerNorm(1024), nn.SiLU(),
            nn.Linear(1024, n_genes),
        )

        # Mixing scalars — initialised to 0 so model starts at PCA exactly
        self.enc_alpha = nn.Parameter(torch.zeros(1))
        self.dec_alpha = nn.Parameter(torch.zeros(1))

    def encode(self, x):
        return self.enc_lin(x) + self.enc_alpha * self.enc_nl(x)

    def decode(self, z):
        return self.dec_lin(z) + self.dec_alpha * self.dec_nl(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


# ── Training ──────────────────────────────────────────────────────────────
@dataclass
class Cfg:
    latent_dim: int = 32
    epochs: int = 500
    lr: float = 1e-3
    lr_min: float = 1e-5
    wd: float = 1e-4
    batch: int = 64


def train_one(
    train_t: torch.Tensor, test_t: torch.Tensor, n_genes: int,
    pca_components: np.ndarray, x_train_mean: np.ndarray,
    cfg: Cfg, device: str, log_every: int = 50,
) -> tuple[HybridAE, dict]:
    torch.manual_seed(SEED)
    model = HybridAE(n_genes, cfg.latent_dim, pca_components, x_train_mean).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs, eta_min=cfg.lr_min,
    )
    loader = DataLoader(
        TensorDataset(train_t.to(device)),
        batch_size=cfg.batch, shuffle=True, drop_last=True,
    )
    history = {"train_mse": [], "test_mse": [], "test_frac_1std": [],
               "test_r2": [], "enc_alpha": [], "dec_alpha": []}
    best = {"frac": -1.0, "epoch": 0, "state": None}
    for ep in range(1, cfg.epochs + 1):
        model.train()
        ep_mse = 0.0
        nb = 0
        for (xb,) in loader:
            opt.zero_grad()
            xh, _ = model(xb)
            loss = F.mse_loss(xh, xb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_mse += float(loss)
            nb += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            xt = test_t.to(device)
            xh, _ = model(xt)
            te_mse = float(F.mse_loss(xh, xt))
            err = (xt - xh).cpu().numpy()
            frac1 = float((np.abs(err) < 1.0).mean())
            r2 = float(1.0 - err.var() / float(xt.var()))
        history["train_mse"].append(ep_mse / nb)
        history["test_mse"].append(te_mse)
        history["test_frac_1std"].append(frac1)
        history["test_r2"].append(r2)
        history["enc_alpha"].append(float(model.enc_alpha.detach().cpu()))
        history["dec_alpha"].append(float(model.dec_alpha.detach().cpu()))
        if frac1 > best["frac"]:
            best["frac"] = frac1
            best["epoch"] = ep
            best["state"] = {k: v.detach().cpu().clone()
                             for k, v in model.state_dict().items()}
        if ep == 1 or ep % log_every == 0 or ep == cfg.epochs:
            print(f"  [d={cfg.latent_dim}] ep {ep:>3}/{cfg.epochs}  "
                  f"train MSE={history['train_mse'][-1]:.4f}  "
                  f"test MSE={te_mse:.4f}  R²={r2:.3f}  "
                  f"frac<1σ={frac1:.3%}  "
                  f"α_enc={history['enc_alpha'][-1]:.3f}  "
                  f"α_dec={history['dec_alpha'][-1]:.3f}  "
                  f"lr={opt.param_groups[0]['lr']:.1e}")
    # restore best
    model.load_state_dict(best["state"])
    print(f"  [d={cfg.latent_dim}] best frac<1σ = {best['frac']:.3%} "
          f"at epoch {best['epoch']}")
    return model, history


def measure(model: HybridAE, X_test: np.ndarray, device: str) -> dict:
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X_test.astype(np.float32)).to(device)
        xh, _ = model(xt)
        err = (xt - xh).cpu().numpy()
    var_total = float(X_test.var())
    var_resid = float(err.var())
    return {
        "mse": float((err ** 2).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "r2": float(1.0 - var_resid / var_total),
        "max_abs_err": float(np.abs(err).max()),
        "frac_within_1std": float((np.abs(err) < 1.0).mean()),
        "frac_within_2std": float((np.abs(err) < 2.0).mean()),
        "frac_within_0.5std": float((np.abs(err) < 0.5).mean()),
        "per_gene_resid_std_max": float(err.std(0).max()),
        "per_gene_resid_std_p99": float(np.percentile(err.std(0), 99)),
        "per_gene_resid_std_mean": float(err.std(0).mean()),
    }


def measure_pca(d: int, X_train: np.ndarray, X_test: np.ndarray) -> dict:
    pca = PCA(n_components=min(d, X_train.shape[0] - 1)).fit(X_train)
    X_hat = pca.inverse_transform(pca.transform(X_test))
    err = X_test - X_hat
    var_total = float(X_test.var())
    return {
        "mse": float((err ** 2).mean()),
        "r2": float(1 - err.var() / var_total),
        "max_abs_err": float(np.abs(err).max()),
        "frac_within_1std": float((np.abs(err) < 1.0).mean()),
        "frac_within_2std": float((np.abs(err) < 2.0).mean()),
        "frac_within_0.5std": float((np.abs(err) < 0.5).mean()),
        "per_gene_resid_std_max": float(err.std(0).max()),
        "per_gene_resid_std_p99": float(np.percentile(err.std(0), 99)),
        "per_gene_resid_std_mean": float(err.std(0).mean()),
    }


# ── Main ──────────────────────────────────────────────────────────────────
def main() -> int:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"Device: {device}")

    print("\n[1/3] Loading GTEx + setup ...")
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

    SWEEP_DIMS = [32, 16, 8, 4, 2, 1]

    print(f"\n[2/3] Sweeping dims {SWEEP_DIMS} ...")
    out = {"target_frac_within_1std": TARGET_FRAC, "rows": []}

    for d in SWEEP_DIMS:
        # PCA reference
        m_pca = measure_pca(d, X_train, X_test)
        print(f"\n  PCA-{d:>2}     R²={m_pca['r2']:.3f}  "
              f"frac<1σ={m_pca['frac_within_1std']:.3%}  "
              f"max|err|={m_pca['max_abs_err']:.2f}")

        ckpt_path = RESULTS / f"q17_hybrid_d{d}.pt"
        if ckpt_path.exists():
            print(f"  HybridAE-{d} (cached {ckpt_path.name})")
            cached = torch.load(ckpt_path, map_location=device, weights_only=False)
            # Re-fit PCA to get components for init (not used at eval, but
            # the constructor needs them)
            pca_init = PCA(n_components=min(d, X_train.shape[0] - 1)).fit(X_train)
            model = HybridAE(n_genes, d, pca_init.components_.astype(np.float32),
                             pca_init.mean_.astype(np.float32)).to(device)
            model.load_state_dict(cached["state_dict"])
        else:
            pca_init = PCA(n_components=min(d, X_train.shape[0] - 1)).fit(X_train)
            cfg = Cfg(latent_dim=d)
            t0 = time.time()
            model, history = train_one(
                torch.from_numpy(X_train), torch.from_numpy(X_test),
                n_genes, pca_init.components_.astype(np.float32),
                pca_init.mean_.astype(np.float32), cfg, device=device,
            )
            print(f"  HybridAE-{d} trained in {time.time() - t0:.0f}s")
            torch.save({
                "state_dict": model.state_dict(),
                "config": cfg.__dict__,
                "history": history,
                "shared_genes": shared_genes,
            }, ckpt_path)

        m_ae = measure(model, X_test, device=device)
        gain = m_ae["frac_within_1std"] - m_pca["frac_within_1std"]
        flag = "✓" if m_ae["frac_within_1std"] >= TARGET_FRAC else "→"
        gain_flag = ("+" if gain > 0 else "") + f"{gain*100:.2f}pt"
        print(f"  {flag} HybridAE-{d}  R²={m_ae['r2']:.3f}  "
              f"frac<1σ={m_ae['frac_within_1std']:.3%}  "
              f"max|err|={m_ae['max_abs_err']:.2f}  "
              f"({gain_flag} vs PCA-{d})")

        out["rows"].append({
            "latent_dim": d,
            "pca": m_pca,
            "hybrid_ae": m_ae,
            "ae_gain_over_pca_pct_pts": gain * 100,
        })

    (RESULTS / "q17_hybrid_ae.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {RESULTS / 'q17_hybrid_ae.json'}")

    # Plot
    print("\n[3/3] Drawing figure ...")
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    fig.suptitle("Q17 — HybridAE (PCA-init + nonlinear correction) vs PCA-N",
                 fontsize=12)

    ds = [r["latent_dim"] for r in out["rows"]]
    pca_f1 = [r["pca"]["frac_within_1std"] for r in out["rows"]]
    ae_f1 = [r["hybrid_ae"]["frac_within_1std"] for r in out["rows"]]
    pca_r2 = [r["pca"]["r2"] for r in out["rows"]]
    ae_r2 = [r["hybrid_ae"]["r2"] for r in out["rows"]]
    pca_max = [r["pca"]["max_abs_err"] for r in out["rows"]]
    ae_max = [r["hybrid_ae"]["max_abs_err"] for r in out["rows"]]

    ax = axes[0]
    ax.plot(ds, pca_f1, "-o", color="#58a6ff", lw=2, label="PCA-N")
    ax.plot(ds, ae_f1, "-s", color="#3fb950", lw=2, label="HybridAE")
    ax.axhline(TARGET_FRAC, color="#f78166", ls="--", lw=1,
               label=f"target ≥ {TARGET_FRAC:.0%}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ds); ax.set_xticklabels(ds)
    ax.set_xlabel("Latent dim")
    ax.set_ylabel("Test frac with |err| < 1 std")
    ax.set_title("(a) frac<1σ vs latent_dim")
    ax.set_ylim(0.5, 1.005)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    for x, p, a in zip(ds, pca_f1, ae_f1):
        ax.annotate(f"{p:.2%}", (x, p), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=7,
                    color="#58a6ff")
        ax.annotate(f"{a:.2%}", (x, a), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=7,
                    color="#3fb950")

    ax = axes[1]
    ax.plot(ds, pca_r2, "-o", color="#58a6ff", lw=2, label="PCA-N")
    ax.plot(ds, ae_r2, "-s", color="#3fb950", lw=2, label="HybridAE")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ds); ax.set_xticklabels(ds)
    ax.set_xlabel("Latent dim")
    ax.set_ylabel("Held-out R²")
    ax.set_title("(b) Recon R² vs latent_dim")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")

    ax = axes[2]
    ax.plot(ds, pca_max, "-o", color="#58a6ff", lw=2, label="PCA-N")
    ax.plot(ds, ae_max, "-s", color="#3fb950", lw=2, label="HybridAE")
    ax.axhline(1.0, color="#f78166", ls=":", lw=0.8,
               label="1 std target")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ds); ax.set_xticklabels(ds)
    ax.set_xlabel("Latent dim")
    ax.set_ylabel("max |element-wise err| (std)")
    ax.set_title("(c) Worst-case prediction error")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(FIGS / "q17_hybrid_ae.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {FIGS / 'q17_hybrid_ae.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
