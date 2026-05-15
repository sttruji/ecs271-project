"""Q54 — Ischemia-residual encoder with FiLM decoder.

Root cause of Q53 failures:
  - HBB (mu_cpm=70755) fails FC resolution because the encoder captures the
    ischemia signal in z_bio, introducing a systematic bias for hi vs lo groups.
  - Flip test >1000-count at 21%: same cause — z_bio carries distributed ischemia.

Fix: subtract the linear ischemia effect from X before encoding.
  X_resid = X_scaled - beta_isch * SMTSISCH_z

Encoder then sees ischemia-corrected expression; decoder reconstructs FULL X_scaled
using z_bio + meta (including SMTSISCH).  The decoder is forced to learn the
ischemia→expression mapping from the meta pathway because z_bio contains no
ischemia information by construction.

Expected improvements vs Q53:
  - FC >1000-count: 100% (HBB fixed — ischemia effect in meta pathway)
  - Flip SMTSISCH >1000-count: >>21% (z_bio ischemia-free)
  - DTHHRDY flip: fixed (use explicit class comparison, not percentile)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.data import load_gtex_blood, load_metadata, load_shared_genes
from pipeline.latent import metadata_linear_probe
from analysis.bulk_rnaseq_resolution_benchmark import evaluate, poisson_tolerance
from models.meta_injection_vae import FiLMMetaInjectionVAE, FiLMMetaInjectionConfig

CKPT   = "/Users/rls/Desktop/programming-projects/single-cell/bulk-project/analysis/14_cross_modality_vae/cross_modality_vae.pt"
OUT    = ROOT / "analysis" / "results" / "q54_residual_encoding"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED   = 0
META_DIM = 4
torch.manual_seed(SEED)
np.random.seed(SEED)

# Single config — residual encoding removes the main source of failure
TRAIN_CFG = dict(
    z_bio_dim=16, decoder_hidden=(512, 512), epochs=400,
    beta=5e-4, lambda_tc=0.0, beta_warmup=120, tc_warmup=0,
)


# ── metadata matrix ────────────────────────────────────────────────────────

def _zscore_col(series: pd.Series) -> np.ndarray:
    v = pd.to_numeric(series, errors="coerce").values.astype(float)
    z = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-8)
    return np.where(np.isfinite(z), z, 0.0).astype(np.float32)


def build_meta_matrix(meta_df: pd.DataFrame) -> np.ndarray:
    isch = _zscore_col(meta_df["SMTSISCH"])
    dthh = _zscore_col(meta_df["DTHHRDY"])
    age  = _zscore_col(meta_df["AGE_mid"])
    sex_raw = pd.to_numeric(meta_df["SEX"], errors="coerce").values
    sex  = np.where(np.isfinite(sex_raw), (sex_raw - 1).astype(np.float32), 0.5)
    return np.column_stack([isch, dthh, age, sex]).astype(np.float32)


# ── ischemia residual preprocessing ───────────────────────────────────────

class IschemiaResidualizer:
    """Fit linear ischemia effect on training data; subtract from any split.

    Model: X_scaled[i, g] ≈ alpha[g] + beta_isch[g] * SMTSISCH_z[i]

    Only SMTSISCH_z is regressed out.  Other metadata (age, sex, DTHHRDY)
    are NOT removed — they stay in z_bio and can be recovered by probes.
    """

    def __init__(self):
        self.beta_isch: np.ndarray | None = None
        self.s_mean: float = 0.0

    def fit(self, X_tr: np.ndarray, isch_z_tr: np.ndarray) -> None:
        """OLS fit per-gene ischemia slope on training data."""
        s = isch_z_tr.astype(np.float64)
        self.s_mean = float(s.mean())
        sc = s - self.s_mean                      # centred
        denom = float(sc @ sc) + 1e-12
        self.beta_isch = (sc @ X_tr.astype(np.float64) / denom).astype(np.float32)
        print(f"  Ischemia residualization fitted: "
              f"|beta_isch| mean={np.abs(self.beta_isch).mean():.4f}  "
              f"max={np.abs(self.beta_isch).max():.4f}")
        print(f"  Genes with |beta|>0.1: {(np.abs(self.beta_isch)>0.1).sum()}")

    def transform(self, X_sc: np.ndarray, isch_z: np.ndarray) -> np.ndarray:
        """Return X with linear ischemia effect subtracted."""
        s = isch_z.astype(np.float64) - self.s_mean
        delta = np.outer(s, self.beta_isch.astype(np.float64))
        return (X_sc.astype(np.float64) - delta).astype(np.float32)

    def ischemia_effect(self, isch_z: np.ndarray) -> np.ndarray:
        """Return the ischemia-effect component for given ischemia values."""
        s = isch_z.astype(np.float64) - self.s_mean
        return np.outer(s, self.beta_isch.astype(np.float64)).astype(np.float32)


# ── training ───────────────────────────────────────────────────────────────

def train_model(model: FiLMMetaInjectionVAE,
                X_tr_resid: np.ndarray,
                X_tr_full: np.ndarray,
                M_tr: np.ndarray,
                cfg: dict) -> list[dict]:
    """Train encoder on residuals; decoder target is full X_scaled."""
    epochs      = cfg["epochs"]
    beta_target = cfg["beta"]
    lambda_tc   = cfg["lambda_tc"]
    beta_warmup = cfg["beta_warmup"]

    model.to(DEVICE).train()
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    Xr = torch.from_numpy(X_tr_resid.astype(np.float32))   # encoder input
    Xf = torch.from_numpy(X_tr_full.astype(np.float32))    # decoder target
    Mt = torch.from_numpy(M_tr.astype(np.float32))
    dl = DataLoader(TensorDataset(Xr, Xf, Mt), batch_size=64, shuffle=True)

    log = []
    for ep in range(1, epochs + 1):
        beta_ep = beta_target * min(1.0, ep / max(1, beta_warmup))
        model.train()
        ep_loss = ep_recon = ep_kl = 0.0
        for xr_b, xf_b, mb in dl:
            xr_b = xr_b.to(DEVICE)
            xf_b = xf_b.to(DEVICE)
            mb   = mb.to(DEVICE)

            mu, lv = model.encode(xr_b)
            z = mu + (0.5 * lv).exp() * torch.randn_like(mu)
            x_hat = model.decode(z, mb)

            recon = torch.nn.functional.mse_loss(x_hat, xf_b)
            kl_pd = -0.5 * (1.0 + lv - mu.pow(2) - lv.exp())
            kl    = kl_pd.clamp(min=model.cfg.free_bits).sum(-1).mean()
            loss  = recon + beta_ep * kl

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss  += loss.item()
            ep_recon += recon.item()
            ep_kl    += kl.item()
        sched.step()

        if ep % 100 == 0 or ep == 1:
            nb = len(dl)
            print(f"  ep {ep:4d}/{epochs}  β={beta_ep:.1e}  "
                  f"recon={ep_recon/nb:.4f}  kl={ep_kl/nb:.4f}")
            log.append({"epoch": ep, "beta": beta_ep,
                        "recon": ep_recon / nb, "kl": ep_kl / nb})
    return log


# ── encode helper (uses residualiser at inference) ─────────────────────────

def encode_with_resid(model: FiLMMetaInjectionVAE,
                      X_sc: np.ndarray, isch_z: np.ndarray,
                      resid: IschemiaResidualizer) -> np.ndarray:
    """Encode ischemia-residual expression to z_bio."""
    X_resid = resid.transform(X_sc, isch_z)
    return model.encode_np(X_resid)


# ── FC resolution benchmark ────────────────────────────────────────────────

def evaluate_fc_resolution(model, X_sc, M, X_lc, meta_df,
                            scaler_mean, scaler_std,
                            resid: IschemiaResidualizer) -> dict:
    isch = pd.to_numeric(meta_df["SMTSISCH"], errors="coerce").values
    hi = isch > np.nanpercentile(isch, 75)
    lo = isch < np.nanpercentile(isch, 25)

    true_fc = X_lc[hi].mean(0) - X_lc[lo].mean(0)
    mu_cpm  = 2 ** ((X_lc[hi].mean(0) + X_lc[lo].mean(0)) / 2) - 1

    def _recon(mask):
        X_r = resid.transform(X_sc[mask], M[mask, 0])
        dev = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            xt = torch.from_numpy(X_r.astype(np.float32)).to(dev)
            mt = torch.from_numpy(M[mask].astype(np.float32)).to(dev)
            mu_, _ = model.encode(xt)
            xh = model.decode(mu_, mt).cpu().numpy()
        return xh * scaler_std + scaler_mean

    xh_hi = _recon(hi)
    xh_lo = _recon(lo)
    pred_fc = xh_hi.mean(0) - xh_lo.mean(0)
    result  = evaluate(pred_fc, true_fc, mu_cpm)

    # Identify failing high-count genes
    tol     = poisson_tolerance(mu_cpm)
    err     = np.abs(pred_fc - true_fc)
    hi_cnt  = mu_cpm > 1000
    fail    = hi_cnt & (err > tol)
    failing = [(gtex_genes[i] if 'gtex_genes' in dir() else f"gene_{i}", float(mu_cpm[i]),
                float(true_fc[i]), float(pred_fc[i]), float(err[i]), float(tol[i]))
               for i in np.where(fail)[0]]

    return {
        "overall":   result["accuracy"],
        "mae":       result["mae"],
        "by_bin":    {k: float(v) for k, v in result["by_bin"]["accuracy"].items()},
        "n_failing_hi_count": int(fail.sum()),
        "failing_genes": failing,
    }


# ── roundtrip ──────────────────────────────────────────────────────────────

def roundtrip_test(model, X_sc, M, X_lc, scaler_std, scaler_mean,
                   resid: IschemiaResidualizer) -> dict:
    X_resid = resid.transform(X_sc, M[:, 0])
    dev = next(model.parameters()).device
    model.eval()
    xh_sc = np.zeros_like(X_sc)
    bs = 64
    for s in range(0, len(X_sc), bs):
        e = min(s + bs, len(X_sc))
        with torch.no_grad():
            xt = torch.from_numpy(X_resid[s:e].astype(np.float32)).to(dev)
            mt = torch.from_numpy(M[s:e].astype(np.float32)).to(dev)
            mu_, _ = model.encode(xt)
            xh_sc[s:e] = model.decode(mu_, mt).cpu().numpy()

    X_lc_approx = X_sc * scaler_std + scaler_mean
    xh_lc       = xh_sc * scaler_std + scaler_mean

    ss_res  = ((X_lc_approx - xh_lc) ** 2).sum(1)
    ss_tot  = ((X_lc_approx - X_lc_approx.mean(1, keepdims=True)) ** 2).sum(1)
    r2s     = 1.0 - ss_res / (ss_tot + 1e-12)
    corr_g  = np.array([np.corrcoef(X_lc_approx[:, g], xh_lc[:, g])[0, 1]
                        for g in range(X_sc.shape[1])])
    corr_g  = np.where(np.isfinite(corr_g), corr_g, 0.0)
    return {
        "r2_per_sample_mean":   float(r2s.mean()),
        "r2_per_sample_median": float(np.median(r2s)),
        "r2_per_sample_p5":     float(np.percentile(r2s, 5)),
        "per_gene_r_mean":      float(corr_g.mean()),
        "per_gene_r_median":    float(np.median(corr_g)),
    }


# ── flip test (fixed: handles discrete metadata) ───────────────────────────

def flip_test_continuous(model, X_sc, M, meta_df, X_lc,
                         scaler_mean, scaler_std,
                         resid: IschemiaResidualizer,
                         col: str, dim_idx: int,
                         lo_pct: float = 25, hi_pct: float = 75) -> dict:
    """Flip test for continuous metadata (ischemia, age)."""
    raw = pd.to_numeric(meta_df[col], errors="coerce").values
    hi_mask = raw > np.nanpercentile(raw, hi_pct)
    lo_mask = raw < np.nanpercentile(raw, lo_pct)
    if hi_mask.sum() == 0 or lo_mask.sum() == 0:
        return {"dim": col, "overall": float("nan"), "error": "empty group"}

    true_fc = X_lc[hi_mask].mean(0) - X_lc[lo_mask].mean(0)
    mu_cpm  = 2 ** ((X_lc[hi_mask].mean(0) + X_lc[lo_mask].mean(0)) / 2) - 1

    def _recon(mask, M_override=None):
        X_r = resid.transform(X_sc[mask], M[mask, 0])
        dev = next(model.parameters()).device
        model.eval()
        meta_in = M_override if M_override is not None else M[mask]
        with torch.no_grad():
            xt = torch.from_numpy(X_r.astype(np.float32)).to(dev)
            mt = torch.from_numpy(meta_in.astype(np.float32)).to(dev)
            mu_, _ = model.encode(xt)
            xh = model.decode(mu_, mt).cpu().numpy()
        return xh * scaler_std + scaler_mean

    xh_hi = _recon(hi_mask)

    # Flip: take lo-group z_bio, inject hi-group mean for this meta dim
    M_lo_flip = M[lo_mask].copy()
    M_lo_flip[:, dim_idx] = M[hi_mask][:, dim_idx].mean()
    xh_lo_flip = _recon(lo_mask, M_override=M_lo_flip)

    pred_fc = xh_hi.mean(0) - xh_lo_flip.mean(0)
    r = evaluate(pred_fc, true_fc, mu_cpm)
    return {
        "dim":     col,
        "overall": r["accuracy"],
        "mae":     r["mae"],
        "by_bin":  {k: float(v) for k, v in r["by_bin"]["accuracy"].items()},
        "n_hi":    int(hi_mask.sum()),
        "n_lo":    int(lo_mask.sum()),
    }


def flip_test_dthhrdy(model, X_sc, M, meta_df, X_lc,
                      scaler_mean, scaler_std,
                      resid: IschemiaResidualizer) -> dict:
    """Flip test for DTHHRDY: compare class 0 (violent/sudden) vs class 4 (slow illness)."""
    raw = pd.to_numeric(meta_df["DTHHRDY"], errors="coerce").values
    class_a, class_b = 0, 4   # most biologically distinct
    mask_a = raw == class_a
    mask_b = raw == class_b

    print(f"  DTHHRDY class 0: {mask_a.sum()} samples, class 4: {mask_b.sum()} samples")
    if mask_a.sum() < 5 or mask_b.sum() < 5:
        return {"dim": "DTHHRDY", "overall": float("nan"),
                "error": f"too few: class0={mask_a.sum()}, class4={mask_b.sum()}"}

    true_fc = X_lc[mask_b].mean(0) - X_lc[mask_a].mean(0)   # slow vs violent
    mu_cpm  = 2 ** ((X_lc[mask_b].mean(0) + X_lc[mask_a].mean(0)) / 2) - 1

    def _recon(mask, M_override=None):
        X_r = resid.transform(X_sc[mask], M[mask, 0])
        dev = next(model.parameters()).device
        model.eval()
        meta_in = M_override if M_override is not None else M[mask]
        with torch.no_grad():
            xt = torch.from_numpy(X_r.astype(np.float32)).to(dev)
            mt = torch.from_numpy(meta_in.astype(np.float32)).to(dev)
            mu_, _ = model.encode(xt)
            xh = model.decode(mu_, mt).cpu().numpy()
        return xh * scaler_std + scaler_mean

    xh_b = _recon(mask_b)

    # Flip: class_a z_bio + class_b metadata
    M_a_flip = M[mask_a].copy()
    M_a_flip[:, 1] = M[mask_b][:, 1].mean()   # DTHHRDY_z at dim_idx=1
    xh_a_flip = _recon(mask_a, M_override=M_a_flip)

    pred_fc = xh_b.mean(0) - xh_a_flip.mean(0)
    r = evaluate(pred_fc, true_fc, mu_cpm)
    return {
        "dim":     "DTHHRDY",
        "overall": r["accuracy"],
        "mae":     r["mae"],
        "by_bin":  {k: float(v) for k, v in r["by_bin"]["accuracy"].items()},
        "class_a": int(class_a),
        "class_b": int(class_b),
        "n_a":     int(mask_a.sum()),
        "n_b":     int(mask_b.sum()),
    }


# ── disentanglement ────────────────────────────────────────────────────────

def disentanglement_report(z: np.ndarray, meta_df: pd.DataFrame) -> dict:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold, cross_val_score

    probes = metadata_linear_probe(z, meta_df)

    def _per_dim_r2(col: str) -> list[float]:
        v = pd.to_numeric(meta_df[col], errors="coerce").values
        mask = np.isfinite(v)
        if mask.sum() < 20:
            return [float("nan")] * z.shape[1]
        y  = v[mask]
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        return [float(cross_val_score(Ridge(alpha=1.0),
                                      StandardScaler().fit_transform(z[mask, k:k+1]),
                                      y, cv=kf, scoring="r2").mean())
                for k in range(z.shape[1])]

    isch_per_dim = _per_dim_r2("SMTSISCH")
    age_per_dim  = _per_dim_r2("AGE_mid")
    var_per_dim  = z.var(0).tolist()

    def _mig(per_dim: list) -> float:
        s = sorted([r for r in per_dim if np.isfinite(r)], reverse=True)
        return s[0] - s[1] if len(s) >= 2 else float("nan")

    return {
        "probes":          probes,
        "isch_per_dim_r2": isch_per_dim,
        "age_per_dim_r2":  age_per_dim,
        "isch_mig_gap":    _mig(isch_per_dim),
        "age_mig_gap":     _mig(age_per_dim),
        "var_per_dim":     [float(v) for v in var_per_dim],
        "n_active_dims":   int((np.array(var_per_dim) > 0.01).sum()),
    }


# ── biological analysis ────────────────────────────────────────────────────

def biological_analysis(model, z, gene_names, meta_ref, out_dir: Path) -> dict:
    loadings = model.gene_loadings(meta_ref)   # (K, G)
    np.save(out_dir / "gene_loadings.npy", loadings)
    pd.DataFrame(loadings.T, index=gene_names,
                 columns=[f"z{k+1}" for k in range(model.n_latent)]
                 ).to_csv(out_dir / "gene_loadings.csv")

    top_genes: dict = {}
    for k in range(model.n_latent):
        ld = loadings[k]
        order = np.argsort(ld)
        top_genes[f"z{k+1}"] = {
            "top_pos": [str(gene_names[i]) for i in order[-5:][::-1]],
            "top_neg": [str(gene_names[i]) for i in order[:5]],
        }

    enrich_summary: list[dict] = []
    try:
        from pipeline.enrichment import enrich_dim_loadings, DEFAULT_LIBRARIES
        var_per_dim = z.var(0)
        active_dims = np.argsort(var_per_dim)[::-1][:3]
        for k in active_dims:
            ld = loadings[k]
            try:
                tables = enrich_dim_loadings(ld, gene_names, top_n=200,
                                             libraries=DEFAULT_LIBRARIES,
                                             description=f"q54_z{k+1}")
                ed = out_dir / "enrichment"; ed.mkdir(exist_ok=True)
                for direction, lib_dict in tables.items():
                    for lib, tbl in lib_dict.items():
                        if lib.startswith("_") or tbl.empty: continue
                        tbl.to_csv(ed / f"z{k+1}__{direction}__{lib}.csv", index=False)
                        if "term" in tbl.columns:
                            enrich_summary.append({
                                "dim": int(k)+1, "direction": direction,
                                "library": lib, "top_term": str(tbl.iloc[0]["term"]),
                                "adj_p": float(tbl.iloc[0].get("adj_p", float("nan"))),
                            })
            except Exception as e:
                print(f"  [enrichment z{k+1}] {e}")
    except ImportError:
        pass

    return {"top_genes_per_dim": top_genes, "enrichment": enrich_summary}


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print("Loading GTEx blood …")
    gtex         = load_gtex_blood(checkpoint_path=CKPT)
    meta         = load_metadata(gtex.sample_ids)
    _, sc_mean, sc_std = load_shared_genes(CKPT)
    X_sc         = gtex.expr_scaled      # standardised (n, G)
    X_lc         = gtex.expr_aligned     # log2(CPM+1)  (n, G)
    M            = build_meta_matrix(meta)
    n_genes      = X_sc.shape[1]
    meta_ref     = M.mean(0)
    gene_names   = gtex.shared_genes

    # Make gene_names accessible in nested function scope
    global gtex_genes
    gtex_genes = gene_names

    # 80/20 split
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X_sc))
    n_test = int(0.2 * len(X_sc))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    X_tr, M_tr = X_sc[train_idx], M[train_idx]

    # ── Ischemia residualization ───────────────────────────────────────────
    print("\n── Ischemia residualization ──")
    resid = IschemiaResidualizer()
    resid.fit(X_tr, M_tr[:, 0])
    X_tr_resid = resid.transform(X_tr, M_tr[:, 0])
    print(f"  Max residual change (HBB): checking ...")
    hbb_idx = np.where(gene_names == "HBB")[0]
    if len(hbb_idx):
        hbb_beta = float(resid.beta_isch[hbb_idx[0]])
        print(f"  HBB beta_isch = {hbb_beta:.4f}  "
              f"(ischemia explains {hbb_beta**2:.1%} of HBB variance per unit SMTSISCH)")

    # ── Train model ────────────────────────────────────────────────────────
    print(f"\n── Training FiLM-ResidualEncoder K={TRAIN_CFG['z_bio_dim']} "
          f"dec={TRAIN_CFG['decoder_hidden']} ep={TRAIN_CFG['epochs']} ──")
    model_cfg = FiLMMetaInjectionConfig(
        input_dim      = n_genes,
        meta_dim       = META_DIM,
        z_bio_dim      = TRAIN_CFG["z_bio_dim"],
        decoder_hidden = TRAIN_CFG["decoder_hidden"],
        meta_embed_dim = 128,
        beta           = TRAIN_CFG["beta"],
        free_bits      = 0.1,
        lambda_tc      = TRAIN_CFG["lambda_tc"],
    )
    model = FiLMMetaInjectionVAE(model_cfg)
    log   = train_model(model, X_tr_resid, X_tr, M_tr, TRAIN_CFG)

    # ── Step 3: FC resolution ──────────────────────────────────────────────
    print("\n── Step 3: FC resolution ──")
    fc = evaluate_fc_resolution(model, X_sc, M, X_lc, meta, sc_mean, sc_std, resid)
    print(f"  overall={fc['overall']:.4f}  >1000-count={fc['by_bin'].get('>1000',float('nan')):.4f}")
    for bin_name, acc in fc["by_bin"].items():
        print(f"    {bin_name:>8}: {acc:.4f}")
    if fc["failing_genes"]:
        for g_info in fc["failing_genes"]:
            g, mu, tfc, pfc, err, tol = g_info
            print(f"  FAIL: {g}  mu={mu:.0f}  tol={tol:.4f}  true={tfc:.4f}  pred={pfc:.4f}  err={err:.4f}")
    else:
        print("  ✓ ALL high-count genes pass!")

    # ── Step 4a: roundtrip ─────────────────────────────────────────────────
    print("\n── Step 4a: Roundtrip ──")
    rt = roundtrip_test(model, X_sc, M, X_lc, sc_std, sc_mean, resid)
    print(f"  per-sample R² mean={rt['r2_per_sample_mean']:.4f}  "
          f"median={rt['r2_per_sample_median']:.4f}  p5={rt['r2_per_sample_p5']:.4f}")
    print(f"  per-gene r  mean={rt['per_gene_r_mean']:.4f}  "
          f"median={rt['per_gene_r_median']:.4f}")

    # ── Step 4b: flip tests ────────────────────────────────────────────────
    print("\n── Step 4b: Flip tests ──")
    flip_results = []
    for col, dim_idx, label in [("SMTSISCH", 0, "ischemia"),
                                 ("AGE_mid",  2, "age")]:
        fr = flip_test_continuous(model, X_sc, M, meta, X_lc,
                                  sc_mean, sc_std, resid, col, dim_idx)
        flip_results.append(fr)
        print(f"  {label:<12}  overall={fr['overall']:.4f}  "
              f">1000={fr['by_bin'].get('>1000', float('nan')):.4f}  MAE={fr.get('mae', float('nan')):.4f}")

    fr_dth = flip_test_dthhrdy(model, X_sc, M, meta, X_lc, sc_mean, sc_std, resid)
    flip_results.append(fr_dth)
    print(f"  DTHHRDY       overall={fr_dth['overall']:.4f}  "
          f">1000={fr_dth['by_bin'].get('>1000', float('nan')):.4f}  MAE={fr_dth.get('mae', float('nan')):.4f}")

    # ── Step 5: disentanglement ────────────────────────────────────────────
    print("\n── Step 5: Disentanglement ──")
    z_bio = encode_with_resid(model, X_sc, M[:, 0], resid)
    dis = disentanglement_report(z_bio, meta)
    p   = dis["probes"]
    print(f"  SMTSISCH R²  = {p['SMTSISCH']['cv_r2']:.3f}  "
          f"(should be LOW — ischemia in meta pathway)")
    print(f"  AGE_mid  R²  = {p['AGE_mid']['cv_r2']:.3f}")
    print(f"  DTHHRDY acc  = {p['DTHHRDY']['cv_balanced_acc']:.3f}  (chance 0.20)")
    print(f"  active dims  = {dis['n_active_dims']} / {model.n_latent}")
    print(f"  ischemia MIG gap = {dis['isch_mig_gap']:.3f}")
    print("  per-dim ischemia R²:", [f"{r:.2f}" for r in dis["isch_per_dim_r2"]])
    print("  var per dim:       ", [f"{v:.2f}" for v in dis["var_per_dim"]])

    # ── Step 6: biological analysis ────────────────────────────────────────
    print("\n── Step 6: Biological analysis ──")
    bio = biological_analysis(model, z_bio, gene_names, meta_ref, OUT)
    for k, info in bio["top_genes_per_dim"].items():
        print(f"  {k}  pos={info['top_pos'][:3]}  neg={info['top_neg'][:3]}")
    if bio["enrichment"]:
        print("  Top enrichment terms:")
        for e in bio["enrichment"][:6]:
            print(f"    z{e['dim']} {e['direction']}  {e['library']}: "
                  f"{e['top_term']} (adj_p={e['adj_p']:.3e})")

    # ── Save ───────────────────────────────────────────────────────────────
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    results = {
        "fc_resolution":   _clean(fc),
        "roundtrip":       _clean(rt),
        "flip":            _clean(flip_results),
        "disentanglement": _clean(dis),
        "bio_top_genes":   _clean(bio["top_genes_per_dim"]),
        "enrichment":      _clean(bio["enrichment"]),
        "ischemia_beta_hbb": float(resid.beta_isch[hbb_idx[0]]) if len(hbb_idx) else None,
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))

    torch.save({
        "state_dict": model.state_dict(),
        "cfg": {"input_dim": n_genes, "meta_dim": META_DIM,
                "z_bio_dim": model.n_latent,
                "decoder_hidden": list(model.cfg.decoder_hidden),
                "meta_embed_dim": model.cfg.meta_embed_dim,
                "beta": model.cfg.beta, "free_bits": model.cfg.free_bits,
                "lambda_tc": model.cfg.lambda_tc},
        "residualizer": {"beta_isch": resid.beta_isch.tolist(), "s_mean": resid.s_mean},
    }, OUT / "film_resid_vae.pt")

    print(f"\nSaved → {OUT}/")
    print("\n══ SUMMARY ══")
    print(f"FC overall       = {fc['overall']:.4f}")
    print(f"FC >1000-count   = {fc['by_bin'].get('>1000', float('nan')):.4f}")
    print(f"Roundtrip R²     = {rt['r2_per_sample_median']:.4f}  (median per sample)")
    for fr in flip_results:
        d = fr["dim"]
        print(f"Flip {d:<12} = {fr['overall']:.4f}  >1000ct={fr['by_bin'].get('>1000', float('nan')):.4f}")
    print(f"SMTSISCH R² in z = {p['SMTSISCH']['cv_r2']:.3f}  "
          f"(was 0.446 in Q53; should be lower)")
    print(f"Active dims      = {dis['n_active_dims']} / {model.n_latent}")


if __name__ == "__main__":
    main()
