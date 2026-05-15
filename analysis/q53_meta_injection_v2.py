"""Q53 — FiLM-conditioned MetaInjection VAE: reconstruction, flip, disentanglement.

Goals:
  Step 3 — FC fold-change resolution ≥ 99.9% overall; 100% on >1000-count genes.
            Iterate z_bio_dim / epochs until threshold is met.
  Step 4 — Roundtrip test (encode → decode with same meta) and flip test
            (swap one metadata dimension; compare reconstructed FC to ground truth).
  Step 5 — Disentanglement: TC penalty pushes z_bio dims to be independent;
            per-dim linear probes show which dim captures which biological factor.
  Step 6 — Biological analysis: per-dim gene loadings (Jacobian at z=0) and
            enrichment via the existing pipeline.

Architecture (FiLMMetaInjectionVAE):
  Encoder: x → [512,256] → z_bio (K-dim)
  Meta embedder: meta(4) → [128,128] → meta_emb
  Decoder: z_bio, FiLM(meta_emb) at each hidden layer → x̂
  Loss: MSE + β·KL + λ_tc·TC  (no metadata prediction loss in encoder)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.data import load_gtex_blood, load_metadata, load_shared_genes
from pipeline.latent import metadata_linear_probe, evaluate_latent_meta
from analysis.bulk_rnaseq_resolution_benchmark import (
    evaluate, null_baseline, poisson_tolerance
)
from models.meta_injection_vae import FiLMMetaInjectionVAE, FiLMMetaInjectionConfig

CKPT   = "/Users/rls/Desktop/programming-projects/single-cell/bulk-project/analysis/14_cross_modality_vae/cross_modality_vae.pt"
OUT    = ROOT / "analysis" / "results" / "q53_meta_injection_v2"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED   = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── hyper-parameter sweep (tried in order; first to hit FC target wins) ────
META_DIM = 4

CONFIGS = [
    dict(z_bio_dim=12, decoder_hidden=(256, 512), epochs=500,
         beta=1e-3, lambda_tc=0.05, beta_warmup=100, tc_warmup=200),
    dict(z_bio_dim=16, decoder_hidden=(256, 512), epochs=500,
         beta=1e-3, lambda_tc=0.05, beta_warmup=100, tc_warmup=200),
    dict(z_bio_dim=16, decoder_hidden=(512, 512), epochs=700,
         beta=5e-4, lambda_tc=0.02, beta_warmup=150, tc_warmup=300),
    dict(z_bio_dim=20, decoder_hidden=(512, 512), epochs=700,
         beta=5e-4, lambda_tc=0.0,  beta_warmup=150, tc_warmup=0),
]

FC_TARGET_OVERALL  = 0.999
FC_TARGET_HI_COUNT = 1.0   # 100% on >1000-count bin


# ── metadata matrix ────────────────────────────────────────────────────────

def _zscore_col(series: pd.Series) -> np.ndarray:
    v = pd.to_numeric(series, errors="coerce").values.astype(float)
    z = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-8)
    return np.where(np.isfinite(z), z, 0.0).astype(np.float32)


def build_meta_matrix(meta_df: pd.DataFrame) -> np.ndarray:
    """(n, 4): [SMTSISCH_z, DTHHRDY_z, AGE_mid_z, SEX_bin].  NaN → 0."""
    isch = _zscore_col(meta_df["SMTSISCH"])
    dthh = _zscore_col(meta_df["DTHHRDY"])
    age  = _zscore_col(meta_df["AGE_mid"])
    sex_raw = pd.to_numeric(meta_df["SEX"], errors="coerce").values
    sex  = np.where(np.isfinite(sex_raw), (sex_raw - 1).astype(np.float32), 0.5)
    return np.column_stack([isch, dthh, age, sex]).astype(np.float32)


# ── training ───────────────────────────────────────────────────────────────

def train_model(model: FiLMMetaInjectionVAE,
                X_tr: np.ndarray, M_tr: np.ndarray,
                cfg: dict) -> list[dict]:
    epochs      = cfg["epochs"]
    beta_target = cfg["beta"]
    lambda_tc   = cfg["lambda_tc"]
    beta_warmup = cfg["beta_warmup"]
    tc_warmup   = cfg["tc_warmup"]

    model.to(DEVICE).train()
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    Xt = torch.from_numpy(X_tr.astype(np.float32))
    Mt = torch.from_numpy(M_tr.astype(np.float32))
    dl = DataLoader(TensorDataset(Xt, Mt), batch_size=64, shuffle=True)

    log = []
    for ep in range(1, epochs + 1):
        beta_ep = beta_target * min(1.0, ep / max(1, beta_warmup))
        tc_ep   = lambda_tc   * min(1.0, ep / max(1, tc_warmup)) if tc_warmup > 0 else lambda_tc

        model.train()
        ep_recon = ep_kl = ep_tc = 0.0
        for xb, mb in dl:
            xb, mb = xb.to(DEVICE), mb.to(DEVICE)
            loss, parts = model.elbo(xb, mb, beta=beta_ep, lambda_tc=tc_ep)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_recon += parts["recon"]
            ep_kl    += parts["kl"]
            ep_tc    += parts["tc"]
        sched.step()

        if ep % 100 == 0 or ep == 1:
            nb = len(dl)
            print(f"  ep {ep:4d}/{epochs}  β={beta_ep:.1e}  λ_tc={tc_ep:.2e}  "
                  f"recon={ep_recon/nb:.4f}  kl={ep_kl/nb:.4f}  tc={ep_tc/nb:.4f}")
            log.append({"epoch": ep, "beta": beta_ep, "lambda_tc": tc_ep,
                        "recon": ep_recon / nb, "kl": ep_kl / nb, "tc": ep_tc / nb})
    return log


# ── Step 3 — FC resolution benchmark ──────────────────────────────────────

def _unscale(x_hat_sc: np.ndarray, scaler_mean: np.ndarray,
             scaler_std: np.ndarray) -> np.ndarray:
    return x_hat_sc * scaler_std + scaler_mean


def recon_logcpm(model: FiLMMetaInjectionVAE,
                 X_sc: np.ndarray, M: np.ndarray,
                 scaler_mean: np.ndarray, scaler_std: np.ndarray) -> np.ndarray:
    """Encode + decode a batch, return log2(CPM+1) reconstructions."""
    dev = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X_sc.astype(np.float32)).to(dev)
        mt = torch.from_numpy(M.astype(np.float32)).to(dev)
        mu, _ = model.encode(xt)
        xh = model.decode(mu, mt).cpu().numpy()
    return _unscale(xh, scaler_mean, scaler_std)


def evaluate_fc_resolution(model, X_sc, M, X_lc, meta_df,
                            scaler_mean, scaler_std) -> dict:
    """Evaluate FC resolution using Marioni Poisson criterion."""
    isch = pd.to_numeric(meta_df["SMTSISCH"], errors="coerce").values
    hi = isch > np.nanpercentile(isch, 75)
    lo = isch < np.nanpercentile(isch, 25)

    true_fc = X_lc[hi].mean(0) - X_lc[lo].mean(0)
    mu_cpm  = 2 ** ((X_lc[hi].mean(0) + X_lc[lo].mean(0)) / 2) - 1

    xh_hi = recon_logcpm(model, X_sc[hi], M[hi], scaler_mean, scaler_std)
    xh_lo = recon_logcpm(model, X_sc[lo], M[lo], scaler_mean, scaler_std)
    pred_fc = xh_hi.mean(0) - xh_lo.mean(0)

    result = evaluate(pred_fc, true_fc, mu_cpm)

    return {
        "overall":    result["accuracy"],
        "mae":        result["mae"],
        "by_bin":     {k: float(v) for k, v in result["by_bin"]["accuracy"].items()},
        "true_fc":    true_fc,
        "pred_fc":    pred_fc,
        "mu_cpm":     mu_cpm,
        "hi_mask":    hi,
        "lo_mask":    lo,
    }


# ── Step 4a — roundtrip test ───────────────────────────────────────────────

def roundtrip_test(model: FiLMMetaInjectionVAE,
                   X_sc: np.ndarray, M: np.ndarray,
                   scaler_mean: np.ndarray, scaler_std: np.ndarray) -> dict:
    """Encode then decode with same meta; compute per-sample R² in log-CPM space."""
    xh_sc = np.zeros_like(X_sc)
    dev = next(model.parameters()).device
    model.eval()
    bs = 64
    for start in range(0, len(X_sc), bs):
        end  = min(start + bs, len(X_sc))
        with torch.no_grad():
            xt = torch.from_numpy(X_sc[start:end].astype(np.float32)).to(dev)
            mt = torch.from_numpy(M[start:end].astype(np.float32)).to(dev)
            mu, _ = model.encode(xt)
            xh_sc[start:end] = model.decode(mu, mt).cpu().numpy()

    X_lc_approx  = _unscale(X_sc,    scaler_mean, scaler_std)
    xh_lc        = _unscale(xh_sc,   scaler_mean, scaler_std)

    # Per-sample R²
    ss_res  = ((X_lc_approx - xh_lc) ** 2).sum(axis=1)
    ss_tot  = ((X_lc_approx - X_lc_approx.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    r2_per_sample = 1.0 - ss_res / (ss_tot + 1e-12)

    # Per-gene Pearson r across samples (measures how well expression variation is preserved)
    corr_per_gene = np.array([
        np.corrcoef(X_lc_approx[:, g], xh_lc[:, g])[0, 1]
        for g in range(X_sc.shape[1])
    ])
    corr_per_gene = np.where(np.isfinite(corr_per_gene), corr_per_gene, 0.0)

    return {
        "r2_per_sample_mean":   float(r2_per_sample.mean()),
        "r2_per_sample_median": float(np.median(r2_per_sample)),
        "r2_per_sample_p5":     float(np.percentile(r2_per_sample, 5)),
        "per_gene_r_mean":      float(corr_per_gene.mean()),
        "per_gene_r_median":    float(np.median(corr_per_gene)),
        "per_gene_r_p5":        float(np.percentile(corr_per_gene, 5)),
    }


# ── Step 4b — flip test ────────────────────────────────────────────────────

def flip_test(model: FiLMMetaInjectionVAE,
              X_sc: np.ndarray, M: np.ndarray,
              meta_df: pd.DataFrame,
              X_lc: np.ndarray,
              scaler_mean: np.ndarray, scaler_std: np.ndarray,
              dim_name: str, dim_idx: int,
              lo_pct: float = 25, hi_pct: float = 75) -> dict:
    """Flip one meta dimension (lo→hi) and evaluate FC recovery.

    Takes the lo-group samples, encodes them, then decodes with the hi-group
    mean for dim_idx injected into their metadata.  Compares reconstructed
    FC against ground-truth FC between hi and lo groups.
    """
    col_vals = pd.to_numeric(meta_df[dim_name], errors="coerce").values
    # Normalised position in M[:,dim_idx]
    hi_mask = col_vals > np.nanpercentile(col_vals, hi_pct)
    lo_mask = col_vals < np.nanpercentile(col_vals, lo_pct)

    true_fc = X_lc[hi_mask].mean(0) - X_lc[lo_mask].mean(0)
    mu_cpm  = 2 ** ((X_lc[hi_mask].mean(0) + X_lc[lo_mask].mean(0)) / 2) - 1

    # Normal reconstructions (no flip)
    xh_hi = recon_logcpm(model, X_sc[hi_mask], M[hi_mask], scaler_mean, scaler_std)

    # Flipped: take lo-group z_bio, inject hi-group mean for this meta dim
    M_lo_flipped = M[lo_mask].copy()
    M_lo_flipped[:, dim_idx] = M[hi_mask][:, dim_idx].mean()
    xh_lo_flip = recon_logcpm(model, X_sc[lo_mask], M_lo_flipped, scaler_mean, scaler_std)

    pred_fc = xh_hi.mean(0) - xh_lo_flip.mean(0)
    result  = evaluate(pred_fc, true_fc, mu_cpm)

    return {
        "dim":       dim_name,
        "overall":   result["accuracy"],
        "mae":       result["mae"],
        "by_bin":    {k: float(v) for k, v in result["by_bin"]["accuracy"].items()},
        "n_hi":      int(hi_mask.sum()),
        "n_lo":      int(lo_mask.sum()),
    }


# ── Step 5 — disentanglement ───────────────────────────────────────────────

def disentanglement_report(z: np.ndarray, meta_df: pd.DataFrame) -> dict:
    """Per-dim linear probes + dim-wise metadata Spearman correlations."""
    probes = metadata_linear_probe(z, meta_df)
    n_dims = z.shape[1]

    # Per-dim Ridge R² for each metadata variable
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold, cross_val_score

    def _per_dim_probe(col: str):
        v = pd.to_numeric(meta_df[col], errors="coerce").values
        mask = np.isfinite(v)
        if mask.sum() < 20:
            return [float("nan")] * n_dims
        y  = v[mask]
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        r2s = []
        for k in range(n_dims):
            zk = StandardScaler().fit_transform(z[mask, k:k+1])
            sc = cross_val_score(Ridge(alpha=1.0), zk, y, cv=kf, scoring="r2")
            r2s.append(float(sc.mean()))
        return r2s

    isch_per_dim = _per_dim_probe("SMTSISCH")
    age_per_dim  = _per_dim_probe("AGE_mid")

    # MIG-like score: best dim R² minus 2nd-best
    def _mig_gap(per_dim: list[float]) -> float:
        sorted_r2 = sorted([r for r in per_dim if np.isfinite(r)], reverse=True)
        if len(sorted_r2) < 2:
            return float("nan")
        return sorted_r2[0] - sorted_r2[1]

    # Activity: variance per dim
    var_per_dim = z.var(0).tolist()

    return {
        "probes":            probes,
        "isch_per_dim_r2":   isch_per_dim,
        "age_per_dim_r2":    age_per_dim,
        "isch_mig_gap":      _mig_gap(isch_per_dim),
        "age_mig_gap":       _mig_gap(age_per_dim),
        "var_per_dim":       [float(v) for v in var_per_dim],
        "n_active_dims":     int((np.array(var_per_dim) > 0.01).sum()),
    }


# ── Step 6 — biological analysis ──────────────────────────────────────────

def biological_analysis(model: FiLMMetaInjectionVAE,
                        z: np.ndarray,
                        gene_names: np.ndarray,
                        meta_ref: np.ndarray,
                        out_dir: Path) -> dict:
    """Gene loadings per z_bio dim + pathway enrichment for top dims."""
    loadings = model.gene_loadings(meta_ref)   # (K, G)
    np.save(out_dir / "gene_loadings.npy", loadings)
    pd.DataFrame(loadings.T, index=gene_names,
                 columns=[f"z{k+1}" for k in range(model.n_latent)]
                 ).to_csv(out_dir / "gene_loadings.csv")

    # Top-5 genes per dim (positive and negative loading)
    top_genes: dict[str, dict] = {}
    for k in range(model.n_latent):
        ld = loadings[k]
        order = np.argsort(ld)
        top_genes[f"z{k+1}"] = {
            "top_pos": [str(gene_names[i]) for i in order[-5:][::-1]],
            "top_neg": [str(gene_names[i]) for i in order[:5]],
            "loading_max": float(ld.max()),
            "loading_min": float(ld.min()),
        }

    # Pathway enrichment (optional — skip gracefully if offline)
    enrich_summary: list[dict] = []
    try:
        from pipeline.enrichment import enrich_dim_loadings, DEFAULT_LIBRARIES
        var_per_dim = z.var(0)
        active_dims = np.argsort(var_per_dim)[::-1][:3]   # top-3 by variance
        for k in active_dims:
            ld = loadings[k]
            try:
                tables = enrich_dim_loadings(
                    ld, gene_names, top_n=200,
                    libraries=DEFAULT_LIBRARIES,
                    description=f"film_mi_vae_z{k+1}",
                )
                enrich_dir = out_dir / "enrichment"
                enrich_dir.mkdir(exist_ok=True)
                for direction, lib_dict in tables.items():
                    for lib, tbl in lib_dict.items():
                        if lib.startswith("_") or tbl.empty:
                            continue
                        tbl.to_csv(enrich_dir / f"z{k+1}__{direction}__{lib}.csv",
                                   index=False)
                        if "term" in tbl.columns:
                            enrich_summary.append({
                                "dim": int(k) + 1,
                                "direction": direction,
                                "library": lib,
                                "top_term": str(tbl.iloc[0]["term"]),
                                "adj_p": float(tbl.iloc[0].get("adj_p", float("nan"))),
                            })
            except Exception as e:
                print(f"  [enrichment z{k+1}] {e}")
    except ImportError:
        print("  [enrichment] pipeline.enrichment not available")

    return {"top_genes_per_dim": top_genes, "enrichment": enrich_summary}


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print("Loading GTEx blood …")
    gtex          = load_gtex_blood(checkpoint_path=CKPT)
    meta          = load_metadata(gtex.sample_ids)
    _, scaler_mean, scaler_std = load_shared_genes(CKPT)
    X_sc          = gtex.expr_scaled      # standardised (n, G)
    X_lc          = gtex.expr_aligned     # log2(CPM+1)  (n, G)
    M             = build_meta_matrix(meta)
    n_genes       = X_sc.shape[1]
    meta_ref      = M.mean(axis=0)        # reference meta for Jacobian

    # 80/20 deterministic split
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X_sc))
    n_test = int(0.2 * len(X_sc))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    X_tr, M_tr = X_sc[train_idx], M[train_idx]

    # ── Step 3: iterate configs until FC target is met ─────────────────────
    best_model, best_fc, best_cfg_idx = None, None, None

    for cfg_idx, cfg in enumerate(CONFIGS):
        print(f"\n═══ Config {cfg_idx+1}/{len(CONFIGS)}: K={cfg['z_bio_dim']} "
              f"dec={cfg['decoder_hidden']} ep={cfg['epochs']} "
              f"β={cfg['beta']} λ_tc={cfg['lambda_tc']} ═══")

        model_cfg = FiLMMetaInjectionConfig(
            input_dim     = n_genes,
            meta_dim      = META_DIM,
            z_bio_dim     = cfg["z_bio_dim"],
            decoder_hidden= cfg["decoder_hidden"],
            meta_embed_dim= 128,
            beta          = cfg["beta"],
            free_bits     = 0.1,
            lambda_tc     = cfg["lambda_tc"],
        )
        model = FiLMMetaInjectionVAE(model_cfg)
        log   = train_model(model, X_tr, M_tr, cfg)

        fc = evaluate_fc_resolution(model, X_sc, M, X_lc, meta, scaler_mean, scaler_std)
        hi_cnt_acc = fc["by_bin"].get(">1000", 0.0)
        print(f"  → FC overall={fc['overall']:.4f}  >1000-count={hi_cnt_acc:.4f}  "
              f"MAE={fc['mae']:.5f}")

        if best_model is None or fc["overall"] > (best_fc["overall"] if best_fc else 0):
            best_model, best_fc, best_cfg_idx = model, fc, cfg_idx

        if fc["overall"] >= FC_TARGET_OVERALL and hi_cnt_acc >= FC_TARGET_HI_COUNT:
            print(f"  ✓ FC targets met at config {cfg_idx+1}!")
            break
    else:
        print(f"\n  ⚠ FC target not fully met; using best config ({best_cfg_idx+1}) "
              f"with overall={best_fc['overall']:.4f}, "
              f">1000={best_fc['by_bin'].get('>1000',0):.4f}")

    model = best_model
    fc    = best_fc
    print(f"\nFinal FC scores: overall={fc['overall']:.4f}  by bin:")
    for bin_name, acc in fc["by_bin"].items():
        print(f"  {bin_name:>8}: {acc:.4f}")

    # ── z_bio embeddings for all samples ──────────────────────────────────
    z_bio = model.encode_np(X_sc)

    # ── Step 4a: roundtrip test ────────────────────────────────────────────
    print("\n── Step 4a: Roundtrip test ──")
    rt = roundtrip_test(model, X_sc, M, scaler_mean, scaler_std)
    print(f"  per-sample R² mean={rt['r2_per_sample_mean']:.4f}  "
          f"median={rt['r2_per_sample_median']:.4f}  p5={rt['r2_per_sample_p5']:.4f}")
    print(f"  per-gene r  mean={rt['per_gene_r_mean']:.4f}  "
          f"median={rt['per_gene_r_median']:.4f}  p5={rt['per_gene_r_p5']:.4f}")

    # ── Step 4b: flip tests ────────────────────────────────────────────────
    print("\n── Step 4b: Flip tests ──")
    meta_dims = [
        ("SMTSISCH", 0, "ischemia time"),
        ("DTHHRDY",  1, "death circumstances"),
        ("AGE_mid",  2, "age"),
    ]
    flip_results: list[dict] = []
    for col, dim_idx, label in meta_dims:
        fr = flip_test(model, X_sc, M, meta, X_lc,
                       scaler_mean, scaler_std, col, dim_idx)
        flip_results.append(fr)
        print(f"  {label:<24}  overall={fr['overall']:.4f}  "
              f">1000={fr['by_bin'].get('>1000', float('nan')):.4f}  "
              f"MAE={fr['mae']:.4f}")

    # ── Step 5: disentanglement ────────────────────────────────────────────
    print("\n── Step 5: Disentanglement ──")
    dis = disentanglement_report(z_bio, meta)
    probes = dis["probes"]
    print(f"  SMTSISCH R²   = {probes['SMTSISCH']['cv_r2']:.3f}  "
          f"(ideal: low, ischemia handled by meta)")
    print(f"  AGE_mid  R²   = {probes['AGE_mid']['cv_r2']:.3f}")
    print(f"  DTHHRDY  acc  = {probes['DTHHRDY']['cv_balanced_acc']:.3f}  (chance 0.20)")
    print(f"  active dims   = {dis['n_active_dims']} / {model.n_latent}")
    print(f"  ischemia MIG gap = {dis['isch_mig_gap']:.3f}")
    print(f"  age      MIG gap = {dis['age_mig_gap']:.3f}")
    print("  per-dim ischemia R²:", [f"{r:.2f}" for r in dis["isch_per_dim_r2"]])
    print("  var per dim:       ", [f"{v:.2f}" for v in dis["var_per_dim"]])

    # ── Step 6: biological analysis ────────────────────────────────────────
    print("\n── Step 6: Biological analysis ──")
    bio = biological_analysis(model, z_bio, gtex.shared_genes, meta_ref, OUT)
    for k, info in bio["top_genes_per_dim"].items():
        print(f"  {k}  pos={info['top_pos'][:3]}  neg={info['top_neg'][:3]}")
    if bio["enrichment"]:
        print("  Top enrichment terms:")
        for e in bio["enrichment"][:6]:
            print(f"    z{e['dim']} {e['direction']}  {e['library']}: "
                  f"{e['top_term']} (adj_p={e['adj_p']:.3e})")

    # ── save everything ────────────────────────────────────────────────────
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()
                    if not isinstance(v, np.ndarray)}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    results = {
        "fc_resolution":      _clean({k: v for k, v in fc.items()
                                      if k not in ("true_fc","pred_fc","mu_cpm",
                                                   "hi_mask","lo_mask")}),
        "roundtrip":          _clean(rt),
        "flip":               _clean(flip_results),
        "disentanglement":    _clean(dis),
        "bio_top_genes":      _clean(bio["top_genes_per_dim"]),
        "enrichment":         _clean(bio["enrichment"]),
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))

    # Save trained model
    torch.save({
        "state_dict": model.state_dict(),
        "cfg": {
            "input_dim":      n_genes,
            "meta_dim":       META_DIM,
            "z_bio_dim":      model.n_latent,
            "decoder_hidden": list(model.cfg.decoder_hidden),
            "meta_embed_dim": model.cfg.meta_embed_dim,
            "beta":           model.cfg.beta,
            "free_bits":      model.cfg.free_bits,
            "lambda_tc":      model.cfg.lambda_tc,
        },
    }, OUT / "film_meta_vae.pt")

    print(f"\nAll results saved to {OUT}/")
    print("\n══ SUMMARY ══")
    print(f"FC resolution   overall = {fc['overall']:.4f}")
    print(f"FC resolution  >1000ct  = {fc['by_bin'].get('>1000',float('nan')):.4f}")
    print(f"Roundtrip R²   (sample) = {rt['r2_per_sample_median']:.4f}")
    for fr in flip_results:
        print(f"Flip {fr['dim']:<12} overall = {fr['overall']:.4f}  "
              f">1000ct = {fr['by_bin'].get('>1000',float('nan')):.4f}")
    print(f"z_bio active dims = {dis['n_active_dims']} / {model.n_latent}")


if __name__ == "__main__":
    main()
