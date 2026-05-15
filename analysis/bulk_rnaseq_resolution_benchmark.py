"""
Bulk RNA-seq fold-change resolution benchmark.

Derived from Marioni 2008 (Poisson technical noise).
See bulk_rnaseq_resolution_benchmark.md for full derivation and provenance.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset


# ── core math ─────────────────────────────────────────────────────────────────

def poisson_log2fc_sd(mu: np.ndarray) -> np.ndarray:
    """SD of log2 fold-change between two Poisson replicates at mean count mu."""
    return np.sqrt(2.0 / mu) / np.log(2)


def poisson_tolerance(mu: np.ndarray, z: float = 1.96) -> np.ndarray:
    """
    95% CI half-width (log2 units) for fold-change between two Poisson replicates.

    tolerance = z * sqrt(2/mu) / ln(2)  ≈  4.0 / sqrt(mu)  for z=1.96
    """
    return z * poisson_log2fc_sd(mu)


def nb_log2fc_sd(mu: np.ndarray, phi: float) -> np.ndarray:
    """
    SD of log2 fold-change for negative-binomial model with dispersion phi.

    Asymptote as mu→∞ is sqrt(2*phi)/ln(2) — the biological replicate floor.
    """
    return np.sqrt(2.0 / mu + 2.0 * phi) / np.log(2)


def asymptotic_fold_change(phi: float, z: float = 1.96) -> float:
    """
    Asymptotic (high-count) fold-change resolution for dispersion phi.

    Returns the 95% CI fold-change floor: 2^(z * sqrt(2*phi) / ln(2))
    """
    if phi == 0:
        return 1.0  # no floor under pure Poisson
    log2_hw = z * np.sqrt(2.0 * phi) / np.log(2)
    return float(2 ** log2_hw)


# ── benchmark evaluation ───────────────────────────────────────────────────────

COUNT_BINS = [0, 10, 100, 1000, np.inf]
BIN_LABELS = ["<10", "10–100", "100–1000", ">1000"]


def evaluate(
    pred_log2fc: np.ndarray,
    true_log2fc: np.ndarray,
    mu: np.ndarray,
    z: float = 1.96,
) -> dict:
    """
    Evaluate model predictions against the Marioni-derived Poisson resolution criterion.

    Parameters
    ----------
    pred_log2fc : (n,) array   — model's predicted log2 fold-changes
    true_log2fc : (n,) array   — measured log2 fold-changes
    mu          : (n,) array   — mean observed counts per gene
    z           : float        — z-score for confidence interval (default 1.96 → 95%)

    Returns
    -------
    dict with keys:
        within_resolution   : bool array, True where |error| ≤ tolerance
        accuracy            : overall fraction within resolution
        mae                 : mean absolute log2 FC error (unweighted)
        mae_weighted        : count-weighted mean absolute error
        by_bin              : DataFrame with per-count-bin accuracy and MAE
    """
    pred_log2fc = np.asarray(pred_log2fc, dtype=float)
    true_log2fc = np.asarray(true_log2fc, dtype=float)
    mu = np.asarray(mu, dtype=float)

    if not (pred_log2fc.shape == true_log2fc.shape == mu.shape):
        raise ValueError("pred_log2fc, true_log2fc, and mu must have the same shape")

    tol = poisson_tolerance(mu, z=z)
    abs_err = np.abs(pred_log2fc - true_log2fc)
    within = abs_err <= tol

    # count-weighted MAE — gives more weight to high-count, high-confidence genes
    weights = mu / mu.sum()
    mae_weighted = float(np.sum(weights * abs_err))

    # per-bin stats
    bin_idx = np.digitize(mu, COUNT_BINS) - 1
    bin_idx = np.clip(bin_idx, 0, len(BIN_LABELS) - 1)

    rows = []
    for b, label in enumerate(BIN_LABELS):
        mask = bin_idx == b
        if mask.sum() == 0:
            rows.append(dict(count_bin=label, n=0, accuracy=np.nan, mae=np.nan))
        else:
            rows.append(dict(
                count_bin=label,
                n=int(mask.sum()),
                accuracy=float(within[mask].mean()),
                mae=float(abs_err[mask].mean()),
            ))

    by_bin = pd.DataFrame(rows).set_index("count_bin")

    return dict(
        within_resolution=within,
        accuracy=float(within.mean()),
        mae=float(abs_err.mean()),
        mae_weighted=mae_weighted,
        by_bin=by_bin,
    )


def null_baseline(true_log2fc: np.ndarray, mu: np.ndarray, z: float = 1.96) -> dict:
    """
    Evaluate the null predictor (always predicts 0 log2 FC).

    Scores True only for genes that are not differentially expressed
    within measurement noise — i.e., |true_log2fc| ≤ tolerance.
    """
    zeros = np.zeros_like(true_log2fc, dtype=float)
    return evaluate(zeros, true_log2fc, mu, z=z)


# ── reference table ────────────────────────────────────────────────────────────

def resolution_table() -> pd.DataFrame:
    """Return the count-dependent resolution table from the benchmark document."""
    mu_vals = [10, 50, 100, 500, 1_000, 10_000, 100_000]
    mu = np.array(mu_vals, dtype=float)
    sd = poisson_log2fc_sd(mu)
    hw = poisson_tolerance(mu)
    fc = 2.0 ** hw
    return pd.DataFrame({
        "count_mu": mu_vals,
        "sd_log2fc": np.round(sd, 4),
        "ci95_halfwidth_log2": np.round(hw, 4),
        "fold_change_resolution": np.round(fc, 3),
    }).set_index("count_mu")


def dispersion_floor_table() -> pd.DataFrame:
    """Return the asymptotic fold-change floor table for various dispersions."""
    phi_vals = [0, 0.005, 0.014, 0.024, 0.05, 0.10]
    labels = [
        "0 (pure Poisson)",
        "0.005 (hypothetical tight)",
        "0.014 (yeast Δsnf2, Gierliński)",
        "0.024 (yeast WT, Gierliński)",
        "0.05 (inbred mouse, typical)",
        "0.10 (outbred human, typical)",
    ]
    floors = []
    for phi in phi_vals:
        f = asymptotic_fold_change(phi)
        floors.append(f if f > 1.0 else float("inf"))

    return pd.DataFrame({
        "phi": phi_vals,
        "context": labels,
        "asymptotic_fc_resolution": floors,
    }).set_index("phi")


# ── model comparison ──────────────────────────────────────────────────────────

ROOT   = Path(__file__).resolve().parents[1]
CKPT   = "/Users/rls/Desktop/programming-projects/single-cell/bulk-project/analysis/14_cross_modality_vae/cross_modality_vae.pt"
OUT    = ROOT / "analysis" / "results" / "bulk_resolution_benchmark"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED, N_LATENT, META_DIM = 0, 12, 4
EPOCHS, BATCH, LR = 300, 64, 1e-3
BETA, FREE_BITS, BETA_WARMUP = 1e-3, 0.1, 100


def _zscore_col(series):
    v = pd.to_numeric(series, errors="coerce").values.astype(float)
    z = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-8)
    return np.where(np.isfinite(z), z, 0.0).astype(np.float32)


def build_meta_matrix(meta_df):
    """(n, 4): [SMTSISCH_z, DTHHRDY_z, AGE_mid_z, SEX_bin].  NaN → 0."""
    isch = _zscore_col(meta_df["SMTSISCH"])
    dthh = _zscore_col(meta_df["DTHHRDY"])
    age  = _zscore_col(meta_df["AGE_mid"])
    sex_raw = pd.to_numeric(meta_df["SEX"], errors="coerce").values
    sex  = np.where(np.isfinite(sex_raw), (sex_raw - 1).astype(np.float32), 0.5)
    return np.column_stack([isch, dthh, age, sex]).astype(np.float32)


def fc_resolution_score(
    model_name: str,
    x_hat_logcpm_A: np.ndarray,
    x_hat_logcpm_B: np.ndarray,
    true_fc: np.ndarray,
    mu: np.ndarray,
) -> dict:
    """Evaluate model's reconstructed FC against Marioni resolution criterion."""
    pred_fc = x_hat_logcpm_A.mean(0) - x_hat_logcpm_B.mean(0)
    r = evaluate(pred_fc, true_fc, mu)
    return {
        "model":    model_name,
        "accuracy": r["accuracy"],
        "mae":      r["mae"],
        "mae_w":    r["mae_weighted"],
        "by_bin":   r["by_bin"].to_dict(),
    }


def _mlp(dims):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [nn.LayerNorm(dims[i + 1]), nn.GELU()]
    return nn.Sequential(*layers)


class _StandardVAE(nn.Module):
    name = "standard_vae"
    def __init__(self, n_genes):
        super().__init__()
        self.encoder = _mlp([n_genes, 512, 256])
        self.mu_h = nn.Linear(256, N_LATENT)
        self.lv_h = nn.Linear(256, N_LATENT)
        self.decoder = _mlp([N_LATENT, 256, 512, n_genes])
    def encode(self, x):
        h = self.encoder(x); return self.mu_h(h), self.lv_h(h)
    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + (0.5 * lv).exp() * torch.randn_like(mu)
        return self.decoder(z), mu, lv
    def elbo(self, x):
        xh, mu, lv = self.forward(x)
        recon = F.mse_loss(xh, x)
        kl = (-0.5*(1+lv-mu.pow(2)-lv.exp())).clamp(min=FREE_BITS).sum(-1).mean()
        return recon + BETA*kl, {"recon": recon.item(), "kl": kl.item()}
    def reconstruct(self, x):
        with torch.no_grad():
            mu, _ = self.encode(x); return self.decoder(mu)


def _train(model, X_tr, tag, M_tr=None, warmup=0):
    model.to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    Xt = torch.from_numpy(X_tr.astype(np.float32))
    ds = TensorDataset(Xt, torch.from_numpy(M_tr.astype(np.float32))) if M_tr is not None \
         else TensorDataset(Xt)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)
    for ep in range(1, EPOCHS + 1):
        beta_ep = BETA * min(1.0, ep / warmup) if warmup else BETA
        model.train()
        for batch in dl:
            xb = batch[0].to(DEVICE)
            if M_tr is not None:
                mb = batch[1].to(DEVICE)
                loss, _ = model.elbo(xb, mb) if hasattr(model, "meta_dim") \
                           else model.elbo_beta(xb, beta_ep) if hasattr(model, "elbo_beta") \
                           else model.elbo(xb)
            elif hasattr(model, "elbo_beta"):
                loss, _ = model.elbo_beta(xb, beta_ep)
            else:
                loss, _ = model.elbo(xb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        if ep % 100 == 0 or ep == 1:
            print(f"  [{tag}] ep {ep}/{EPOCHS}")


def run_model_comparison():
    """Train all four models, evaluate each on Marioni FC resolution benchmark."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ROOT))
    from pipeline.data import load_gtex_blood, load_metadata, load_shared_genes
    from pipeline.latent import metadata_linear_probe
    from models.drvi_model import AdditiveVAE, AdditiveVAEConfig
    from models.meta_injection_vae import MetaInjectionVAE, MetaInjectionConfig

    print("Loading data ...")
    gtex = load_gtex_blood(checkpoint_path=CKPT)
    meta = load_metadata(gtex.sample_ids)
    _, scaler_mean, scaler_std = load_shared_genes(CKPT)
    X_sc = gtex.expr_scaled                      # standardised  (n, G)
    X_lc = gtex.expr_aligned                     # log2(CPM+1)   (n, G)
    M    = build_meta_matrix(meta)
    n_genes = X_sc.shape[1]

    # 80/20 split (deterministic)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X_sc))
    n_test = int(0.2 * len(X_sc))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    X_tr, X_te = X_sc[train_idx], X_sc[test_idx]
    M_tr, M_te = M[train_idx],   M[test_idx]

    # FC benchmark split: high vs low SMTSISCH (top/bottom quartile)
    isch = pd.to_numeric(meta["SMTSISCH"], errors="coerce").values
    hi = isch > np.nanpercentile(isch, 75)
    lo = isch < np.nanpercentile(isch, 25)
    true_fc = X_lc[hi].mean(0) - X_lc[lo].mean(0)           # true log2 FC
    mu_cpm  = 2 ** ((X_lc[hi].mean(0) + X_lc[lo].mean(0)) / 2) - 1  # approx CPM

    def _unscale(x_hat_sc):
        return x_hat_sc * scaler_std + scaler_mean            # → log2(CPM+1)

    def _recon_logcpm(model, X_sc_subset, meta_subset=None):
        dev = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            xt = torch.from_numpy(X_sc_subset.astype(np.float32)).to(dev)
            if meta_subset is not None:
                mt = torch.from_numpy(meta_subset.astype(np.float32)).to(dev)
                mu, _ = model.encode(xt)
                xh = model.decode(mu, mt).cpu().numpy()
            elif hasattr(model, "reconstruct"):
                xh = model.reconstruct(xt).cpu().numpy()
            else:
                mu, _ = model.encode(xt)
                xh = model.decoder(mu).cpu().numpy()
        return _unscale(xh)

    scores = {}

    # ── null baseline (no model, raw data) ───────────────────────────────
    r_null = null_baseline(true_fc, mu_cpm)
    scores["raw_data"] = {"accuracy": r_null["accuracy"], "mae": r_null["mae"],
                          "mae_w": r_null["mae_weighted"]}
    print(f"\nnull baseline (raw data): acc={r_null['accuracy']:.3f}  mae={r_null['mae']:.4f}")

    # ── PCA-50 ────────────────────────────────────────────────────────────
    print("\n── PCA-50 ──")
    pca = PCA(n_components=50, random_state=SEED).fit(X_tr)
    xh_hi_pca = _unscale(pca.inverse_transform(pca.transform(X_sc[hi])))
    xh_lo_pca = _unscale(pca.inverse_transform(pca.transform(X_sc[lo])))
    s = fc_resolution_score("pca50", xh_hi_pca, xh_lo_pca, true_fc, mu_cpm)
    scores["pca50"] = s
    print(f"  acc={s['accuracy']:.3f}  mae={s['mae']:.4f}")
    z_pca = pca.transform(X_sc).astype(np.float32)
    scores["pca50"]["probes"] = metadata_linear_probe(z_pca, meta)

    # ── Standard VAE ─────────────────────────────────────────────────────
    print(f"\n── Standard VAE K={N_LATENT} ──")
    std_vae = _StandardVAE(n_genes)
    _train(std_vae, X_tr, "std")
    std_vae.eval()
    xh_hi_std = _recon_logcpm(std_vae, X_sc[hi])
    xh_lo_std = _recon_logcpm(std_vae, X_sc[lo])
    s = fc_resolution_score("standard_vae", xh_hi_std, xh_lo_std, true_fc, mu_cpm)
    scores["standard_vae"] = s
    print(f"  acc={s['accuracy']:.3f}  mae={s['mae']:.4f}")
    with torch.no_grad():
        z_std = std_vae.encode(
            torch.from_numpy(X_sc.astype(np.float32)).to(DEVICE))[0].cpu().numpy()
    scores["standard_vae"]["probes"] = metadata_linear_probe(z_std, meta)

    # ── Additive VAE ─────────────────────────────────────────────────────
    print(f"\n── Additive VAE K={N_LATENT} ──")
    add_vae = AdditiveVAE(AdditiveVAEConfig(
        input_dim=n_genes, n_latent=N_LATENT,
        encoder_hidden=(512, 256), decoder_hidden=32,
        beta=BETA, free_bits=FREE_BITS))
    _train(add_vae, X_tr, "add", warmup=BETA_WARMUP)
    add_vae.eval()
    xh_hi_add = _recon_logcpm(add_vae, X_sc[hi])
    xh_lo_add = _recon_logcpm(add_vae, X_sc[lo])
    s = fc_resolution_score("additive_vae", xh_hi_add, xh_lo_add, true_fc, mu_cpm)
    scores["additive_vae"] = s
    print(f"  acc={s['accuracy']:.3f}  mae={s['mae']:.4f}")
    z_add = add_vae.encode_np(X_sc)
    scores["additive_vae"]["probes"] = metadata_linear_probe(z_add, meta)

    # ── MetaInjection VAE ─────────────────────────────────────────────────
    print(f"\n── MetaInjection VAE K={N_LATENT} ──")
    mi_vae = MetaInjectionVAE(MetaInjectionConfig(
        input_dim=n_genes, meta_dim=META_DIM, z_bio_dim=N_LATENT,
        encoder_hidden=(512, 256), beta=BETA, free_bits=FREE_BITS))
    mi_vae.meta_dim = META_DIM  # tag for _train dispatcher
    # Override elbo to match dispatcher signature
    orig_elbo = mi_vae.elbo
    mi_vae.elbo = lambda x, m: orig_elbo(x, m)
    _train(mi_vae, X_tr, "mi", M_tr=M_tr)
    mi_vae.meta_dim = META_DIM  # restore after training
    mi_vae.eval()
    xh_hi_mi = _recon_logcpm(mi_vae, X_sc[hi], meta_subset=M[hi])
    xh_lo_mi = _recon_logcpm(mi_vae, X_sc[lo], meta_subset=M[lo])
    s = fc_resolution_score("meta_injection_vae", xh_hi_mi, xh_lo_mi, true_fc, mu_cpm)
    scores["meta_injection_vae"] = s
    print(f"  acc={s['accuracy']:.3f}  mae={s['mae']:.4f}")
    z_mi = mi_vae.encode_np(X_sc)
    scores["meta_injection_vae"]["probes"] = metadata_linear_probe(z_mi, meta)

    # ── MetaInjection flip: encode lo-isch samples, inject hi-isch metadata ──
    print("\n── MetaInjection VAE — ischemia flip (lo→hi) ──")
    M_lo_flipped = M[lo].copy()
    M_lo_flipped[:, 0] = M[hi][:, 0].mean()   # replace SMTSISCH_z with hi mean
    xh_flip_mi = _recon_logcpm(mi_vae, X_sc[lo], meta_subset=M_lo_flipped)
    s_flip = fc_resolution_score(
        "meta_inj_flip(lo→hi)", xh_hi_mi, xh_flip_mi, true_fc, mu_cpm)
    scores["meta_inj_flip"] = s_flip
    print(f"  acc={s_flip['accuracy']:.3f}  mae={s_flip['mae']:.4f}")

    # ── summary table ─────────────────────────────────────────────────────
    print("\n" + "═" * 78)
    print(f"{'':>24}  {'FC acc':>7}  {'FC MAE':>7}  {'SMTSISCH R²':>12}  {'DTHHRDY acc':>12}")
    print("─" * 78)
    for mname, r in scores.items():
        fc_acc = r.get("accuracy", float("nan"))
        fc_mae = r.get("mae",      float("nan"))
        p = r.get("probes", {})
        isch_r2 = p.get("SMTSISCH", {}).get("cv_r2",           float("nan"))
        dthh_ac = p.get("DTHHRDY",  {}).get("cv_balanced_acc", float("nan"))
        print(f"{mname:<24}  {fc_acc:>7.3f}  {fc_mae:>7.4f}  "
              f"{isch_r2:>12.3f}  {dthh_ac:>12.3f}")

    # serialise (strip non-JSON-safe entries)
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items() if k != "within_resolution"}
        if isinstance(obj, (np.floating, np.integer)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    (OUT / "results.json").write_text(json.dumps(_clean(scores), indent=2))
    print(f"\nSaved → {OUT}/results.json")
    return scores


# ── CLI demo ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Poisson resolution table (Marioni 2008) ===")
    print(resolution_table().to_string())
    print()
    print("=== Dispersion floor table (NB model) ===")
    print(dispersion_floor_table().to_string())
    print()

    # toy example
    rng = np.random.default_rng(0)
    n = 500
    mu_example = 10 ** rng.uniform(1, 4, n)              # log-uniform 10–10000
    true_fc = rng.normal(0, 1, n)                         # random true FC
    # simulate a "model" with moderate noise
    pred_fc = true_fc + rng.normal(0, 0.3, n)

    result = evaluate(pred_fc, true_fc, mu_example)
    baseline = null_baseline(true_fc, mu_example)

    print("=== Toy example: model vs null baseline ===")
    print(f"Model  — accuracy: {result['accuracy']:.3f}  MAE: {result['mae']:.4f}")
    print(f"Null   — accuracy: {baseline['accuracy']:.3f}  MAE: {baseline['mae']:.4f}")
    print()
    print("Model by count bin:")
    print(result["by_bin"].to_string())
    print()

    # ── model comparison on real GTEx data ─────────────────────────────────
    print("=== Model comparison — ischemia FC resolution (GTEx whole blood) ===")
    run_model_comparison()
