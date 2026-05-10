#!/usr/bin/env python3
"""Q16 — Push reconstruction to "within-1-std" at d=32, then minimise d.

OBJECTIVE.  Build an autoencoder whose held-out reconstruction is good enough
that ≥99 % of (sample, gene) test predictions land within 1 standard deviation
of the truth (data is per-gene standardised, so 1 std = 1.0 in our units).
Once we hit ≥99 %, sweep latent_dim downward and find the smallest d that
still passes.

ITERATION STRATEGY.  Three architecture rungs of increasing capacity, all
deterministic AEs (no KL — Q15 / Q3 already showed KL costs reconstruction
on this dataset).  Training runs add capacity until the criterion is met,
then we hold the architecture fixed and shrink d.

  v1 — "Sane-AE wider":  11374 → 2048 → 1024 → 512 → d → 512 → 1024 → 2048 → 11374
                          BN, LeakyReLU, no dropout, MSE only,
                          AdamW(1e-3, wd=1e-4), cosine schedule, 400 epochs.
  v2 — "Wide+deep":       11374 → 4096 → 2048 → 1024 → 512 → d → 512 → 1024 → 2048 → 4096 → 11374
                          (used only if v1 < 99 %)
  v3 — "Residual":        v2 backbone with input/output residual streams
                          (used only if v2 < 99 %)

EVAL.  At convergence we compute on the held-out 161-donor test set:
  •  per-element MSE           (lower = better)
  •  R² (overall variance)
  •  frac_within_1std          (target ≥ 0.99) ← the success criterion
  •  per-gene std of residuals (max across genes — secondary tightness check)

Caching.  Each (architecture, d) checkpoint saved at
  results/q16_{arch}_d{N}.pt
so reruns / dim-sweep skip retraining.

Run:  python q16_perfect_recovery.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
TARGET_FRAC = 0.99   # 99 % of test predictions within 1 std
TARGET_NAME = "frac_within_1std"


# ── Architectures ─────────────────────────────────────────────────────────
def _block(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(i, o), nn.BatchNorm1d(o), nn.LeakyReLU(0.2),
    )


class WideAE(nn.Module):
    """v1: Sane-AE wider.  ~30 M params at d=32."""
    def __init__(self, n_genes: int, latent_dim: int):
        super().__init__()
        h = [n_genes, 2048, 1024, 512]
        self.encoder = nn.Sequential(
            *[_block(h[i], h[i + 1]) for i in range(len(h) - 1)],
            nn.Linear(h[-1], latent_dim),
        )
        d = [latent_dim, 512, 1024, 2048]
        self.decoder = nn.Sequential(
            *[_block(d[i], d[i + 1]) for i in range(len(d) - 1)],
            nn.Linear(d[-1], n_genes),
        )

    def encode(self, x): return self.encoder(x)
    def decode(self, z): return self.decoder(z)
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class WideDeepAE(nn.Module):
    """v2: Wider+deeper.  ~75 M params at d=32."""
    def __init__(self, n_genes: int, latent_dim: int):
        super().__init__()
        h = [n_genes, 4096, 2048, 1024, 512]
        self.encoder = nn.Sequential(
            *[_block(h[i], h[i + 1]) for i in range(len(h) - 1)],
            nn.Linear(h[-1], latent_dim),
        )
        d = [latent_dim, 512, 1024, 2048, 4096]
        self.decoder = nn.Sequential(
            *[_block(d[i], d[i + 1]) for i in range(len(d) - 1)],
            nn.Linear(d[-1], n_genes),
        )

    def encode(self, x): return self.encoder(x)
    def decode(self, z): return self.decoder(z)
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class ResidualAE(nn.Module):
    """v3: Wide+deep backbone with residual short-circuits inside each stage."""

    class _Res(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim)
            self.bn1 = nn.BatchNorm1d(dim)
            self.fc2 = nn.Linear(dim, dim)
            self.bn2 = nn.BatchNorm1d(dim)

        def forward(self, x):
            h = F.leaky_relu(self.bn1(self.fc1(x)), 0.2)
            h = self.bn2(self.fc2(h))
            return F.leaky_relu(x + h, 0.2)

    def __init__(self, n_genes: int, latent_dim: int):
        super().__init__()
        self.in_proj = _block(n_genes, 4096)
        self.enc = nn.Sequential(
            self._Res(4096), _block(4096, 2048),
            self._Res(2048), _block(2048, 1024),
            self._Res(1024), _block(1024, 512),
            nn.Linear(512, latent_dim),
        )
        self.dec_proj = _block(latent_dim, 512)
        self.dec = nn.Sequential(
            _block(512, 1024), self._Res(1024),
            _block(1024, 2048), self._Res(2048),
            _block(2048, 4096), self._Res(4096),
            nn.Linear(4096, n_genes),
        )

    def encode(self, x):
        return self.enc(self.in_proj(x))

    def decode(self, z):
        return self.dec(self.dec_proj(z))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


ARCHS = {"WideAE": WideAE, "WideDeepAE": WideDeepAE, "ResidualAE": ResidualAE}


# ── Training ──────────────────────────────────────────────────────────────
@dataclass
class TrainCfg:
    arch: str = "WideAE"
    latent_dim: int = 32
    epochs: int = 400
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch: int = 64
    cosine: bool = True
    log_every: int = 25


def train(
    train_t: torch.Tensor, test_t: torch.Tensor, n_genes: int,
    cfg: TrainCfg, device: str,
) -> tuple[nn.Module, dict]:
    torch.manual_seed(SEED)
    model = ARCHS[cfg.arch](n_genes, cfg.latent_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
             if cfg.cosine else None)
    loader = DataLoader(
        TensorDataset(train_t.to(device)),
        batch_size=cfg.batch, shuffle=True, drop_last=True,
    )
    history = {"train_mse": [], "test_mse": [], "test_frac_1std": [],
               "test_r2": [], "lr": []}
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
        if sched is not None:
            sched.step()
        model.eval()
        with torch.no_grad():
            x_t = test_t.to(device)
            xh, _ = model(x_t)
            te_mse = float(F.mse_loss(xh, x_t))
            err = (x_t - xh).cpu().numpy()
            frac_1std = float((np.abs(err) < 1.0).mean())
            r2 = float(1.0 - err.var() / float(x_t.var()))
        history["train_mse"].append(ep_mse / nb)
        history["test_mse"].append(te_mse)
        history["test_frac_1std"].append(frac_1std)
        history["test_r2"].append(r2)
        history["lr"].append(opt.param_groups[0]["lr"])
        if ep == 1 or ep % cfg.log_every == 0 or ep == cfg.epochs:
            print(f"  [{cfg.arch} d={cfg.latent_dim}] ep {ep:>3}/{cfg.epochs}  "
                  f"train MSE={history['train_mse'][-1]:.4f}  "
                  f"test MSE={te_mse:.4f}  R²={r2:.3f}  "
                  f"frac<1σ={frac_1std:.3%}  lr={history['lr'][-1]:.1e}")
    return model, history


def measure(model, X_test: np.ndarray, device: str) -> dict:
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X_test.astype(np.float32)).to(device)
        xh, _ = model(xt)
        err = (xt - xh).cpu().numpy()
    n_total = err.size
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
        "per_gene_resid_std_mean": float(err.std(0).mean()),
        "per_gene_resid_std_p99": float(np.percentile(err.std(0), 99)),
        "n_pred_total": int(n_total),
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
    train_t = torch.from_numpy(X_train)
    test_t = torch.from_numpy(X_test)

    # ── Stage 1: iterate architectures at d=32 ──────────────────────────
    print("\n[2/3] Iterating architectures at d=32 until "
          f"frac<1σ ≥ {TARGET_FRAC:.0%} ...")
    iter_log = []
    winner = None
    for arch in ["WideAE", "WideDeepAE", "ResidualAE"]:
        ckpt_path = RESULTS / f"q16_{arch}_d32.pt"
        cfg = TrainCfg(arch=arch, latent_dim=32, epochs=400)
        if ckpt_path.exists():
            print(f"\n  === {arch}, d=32 (cached {ckpt_path.name}) ===")
            cached = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = ARCHS[arch](n_genes, 32).to(device)
            model.load_state_dict(cached["state_dict"])
            history = cached.get("history", {})
        else:
            print(f"\n  === Training {arch}, d=32 ===")
            t0 = time.time()
            model, history = train(train_t, test_t, n_genes, cfg, device=device)
            elapsed = time.time() - t0
            print(f"  done ({elapsed:.0f}s)")
            torch.save({
                "state_dict": model.state_dict(),
                "config": cfg.__dict__,
                "history": history,
                "shared_genes": shared_genes,
            }, ckpt_path)
        m = measure(model, X_test, device=device)
        iter_log.append({"arch": arch, "latent_dim": 32, **m})
        print(f"  → R²={m['r2']:.3f}  frac<1σ={m['frac_within_1std']:.3%}  "
              f"max|err|={m['max_abs_err']:.2f}  "
              f"max per-gene σ_err={m['per_gene_resid_std_max']:.2f}")
        if m["frac_within_1std"] >= TARGET_FRAC:
            winner = (arch, model, m)
            print(f"  ✓ {arch} hits target ({m['frac_within_1std']:.3%} ≥ {TARGET_FRAC:.0%})")
            break

    if winner is None:
        # use the best of the three even if below target
        best_arch, best_model = None, None
        best_score = -1.0
        for entry, arch in zip(iter_log, ["WideAE", "WideDeepAE", "ResidualAE"]):
            if entry["frac_within_1std"] > best_score:
                best_score = entry["frac_within_1std"]
                best_arch = entry["arch"]
        # reload best
        cached = torch.load(RESULTS / f"q16_{best_arch}_d32.pt",
                            map_location=device, weights_only=False)
        best_model = ARCHS[best_arch](n_genes, 32).to(device)
        best_model.load_state_dict(cached["state_dict"])
        winner = (best_arch, best_model, iter_log[-1])
        print(f"\n  No arch hit {TARGET_FRAC:.0%}; using best ({best_arch} "
              f"at {best_score:.3%}) for the dim sweep")

    winner_arch = winner[0]

    # ── Stage 2: dim sweep with the winning architecture ────────────────
    print(f"\n[3/3] Dim sweep with {winner_arch} ...")
    SWEEP_DIMS = [32, 16, 12, 8, 4, 2, 1]
    sweep_log = []
    for d in SWEEP_DIMS:
        ckpt_path = RESULTS / f"q16_{winner_arch}_d{d}.pt"
        cfg = TrainCfg(arch=winner_arch, latent_dim=d, epochs=400)
        if ckpt_path.exists():
            print(f"\n  === {winner_arch}, d={d} (cached) ===")
            cached = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = ARCHS[winner_arch](n_genes, d).to(device)
            model.load_state_dict(cached["state_dict"])
            history = cached.get("history", {})
        else:
            print(f"\n  === Training {winner_arch}, d={d} ===")
            t0 = time.time()
            model, history = train(train_t, test_t, n_genes, cfg, device=device)
            print(f"  done ({time.time() - t0:.0f}s)")
            torch.save({
                "state_dict": model.state_dict(),
                "config": cfg.__dict__,
                "history": history,
                "shared_genes": shared_genes,
            }, ckpt_path)
        m = measure(model, X_test, device=device)
        sweep_log.append({"arch": winner_arch, "latent_dim": d, **m})
        flag = "✓" if m["frac_within_1std"] >= TARGET_FRAC else "✗"
        print(f"  {flag} d={d:>3}  R²={m['r2']:.3f}  "
              f"frac<1σ={m['frac_within_1std']:.3%}  "
              f"max per-gene σ_err={m['per_gene_resid_std_max']:.2f}")

    # ── Save + plot ─────────────────────────────────────────────────────
    full_log = {
        "iter_log_at_d32": iter_log,
        "winner_arch": winner_arch,
        "dim_sweep": sweep_log,
        "target_frac_within_1std": TARGET_FRAC,
    }
    (RESULTS / "q16_perfect_recovery.json").write_text(json.dumps(full_log, indent=2))
    print(f"\nWrote {RESULTS / 'q16_perfect_recovery.json'}")

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    fig.suptitle(f"Q16 — Push to within-1σ recovery, then minimise d "
                 f"({winner_arch})", fontsize=12)

    # (a) iter_log @ d=32: arch comparison
    ax = axes[0]
    archs = [e["arch"] for e in iter_log]
    fracs = [e["frac_within_1std"] for e in iter_log]
    r2s = [e["r2"] for e in iter_log]
    bars = ax.bar(range(len(archs)), fracs, color="#3fb950")
    ax.axhline(TARGET_FRAC, color="#f78166", ls="--", lw=1,
               label=f"target ≥ {TARGET_FRAC:.0%}")
    ax.set_xticks(range(len(archs)))
    ax.set_xticklabels(archs, rotation=15)
    ax.set_ylabel("Test frac with |err| < 1 std")
    ax.set_title(f"(a) Architecture sweep at d=32")
    ax.set_ylim(0.85, 1.005)
    for i, (f, r) in enumerate(zip(fracs, r2s)):
        ax.text(i, f + 0.001, f"{f:.2%}\nR²={r:.2f}",
                ha="center", fontsize=8)
    ax.legend(loc="lower right")

    # (b) dim sweep frac
    ax = axes[1]
    ds = [e["latent_dim"] for e in sweep_log]
    fracs = [e["frac_within_1std"] for e in sweep_log]
    r2s = [e["r2"] for e in sweep_log]
    ax.plot(ds, fracs, "-o", color="#3fb950", lw=2, label="frac < 1σ")
    ax.axhline(TARGET_FRAC, color="#f78166", ls="--", lw=1,
               label=f"target ≥ {TARGET_FRAC:.0%}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ds)
    ax.set_xticklabels(ds)
    ax.set_xlabel("Latent dim")
    ax.set_ylabel("Test frac with |err| < 1 std")
    ax.set_title(f"(b) {winner_arch} — frac<1σ vs latent_dim")
    ax.set_ylim(0.5, 1.005)
    ax.grid(alpha=0.2)
    for x, v in zip(ds, fracs):
        ax.annotate(f"{v:.2%}", (x, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.legend(loc="lower right")

    # (c) dim sweep R²
    ax = axes[2]
    ax.plot(ds, r2s, "-s", color="#58a6ff", lw=2, label="recon R²")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ds)
    ax.set_xticklabels(ds)
    ax.set_xlabel("Latent dim")
    ax.set_ylabel("Held-out R²")
    ax.set_title(f"(c) {winner_arch} — recon R² vs latent_dim")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2)
    for x, v in zip(ds, r2s):
        ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGS / "q16_perfect_recovery.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {FIGS / 'q16_perfect_recovery.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
