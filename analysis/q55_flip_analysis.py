"""Q55 — Corrected flip test analysis.

The "group FC flip test" used in Q53/Q54 has a design flaw for well-disentangled models:

  pred_fc = mean(decode(z_bio_hi, meta_hi)) - mean(decode(z_bio_lo, meta_hi_injected))

For a model where z_bio contains NO ischemia (Q54), the two groups produce the
same ischemia contribution via meta_hi, so pred_fc ≈ biology_hi - biology_lo ≈ 0.
The true_fc includes both biology AND ischemia differences → huge mismatch for
ischemia-driven genes (e.g. HBB).  The model is disentangled; the test was wrong.

Correct within-sample flip test:
  For each sample i, keep z_bio_i fixed and only change the target metadata dim.
  pred_ischemia_effect = mean_i[ decode(z_bio_i, meta_hi) - decode(z_bio_i, meta_lo) ]
  Ground truth          = true_fc between hi/lo groups (which equals the ischemia effect
                          averaged over individuals when biology cancels in the mean)

This correctly measures: "Does the decoder's meta pathway encode the right effect?"

Also tests:
  - Q53 vs Q54 comparison on both test definitions
  - HBB specifically
  - Roundtrip improvement
  - Per-gene correlation of decoder's ischemia effect with OLS-estimated ischemia slopes
"""
from __future__ import annotations

import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.data import load_gtex_blood, load_metadata, load_shared_genes
from analysis.bulk_rnaseq_resolution_benchmark import evaluate, poisson_tolerance
from models.meta_injection_vae import FiLMMetaInjectionVAE, FiLMMetaInjectionConfig

CKPT   = "/Users/rls/Desktop/programming-projects/single-cell/bulk-project/analysis/14_cross_modality_vae/cross_modality_vae.pt"
OUT    = ROOT / "analysis" / "results" / "q55_flip_analysis"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED   = 0


# ── metadata helpers ───────────────────────────────────────────────────────

def _zscore_col(series):
    v = pd.to_numeric(series, errors="coerce").values.astype(float)
    z = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-8)
    return np.where(np.isfinite(z), z, 0.0).astype(np.float32)


def build_meta_matrix(meta_df):
    isch = _zscore_col(meta_df["SMTSISCH"])
    dthh = _zscore_col(meta_df["DTHHRDY"])
    age  = _zscore_col(meta_df["AGE_mid"])
    sex_raw = pd.to_numeric(meta_df["SEX"], errors="coerce").values
    sex  = np.where(np.isfinite(sex_raw), (sex_raw - 1).astype(np.float32), 0.5)
    return np.column_stack([isch, dthh, age, sex]).astype(np.float32)


# ── load models ────────────────────────────────────────────────────────────

def load_film_model(ckpt_path: str, device: str) -> FiLMMetaInjectionVAE:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg  = ckpt["cfg"]
    mc   = FiLMMetaInjectionConfig(
        input_dim      = cfg["input_dim"],
        meta_dim       = cfg["meta_dim"],
        z_bio_dim      = cfg["z_bio_dim"],
        decoder_hidden = tuple(cfg["decoder_hidden"]),
        meta_embed_dim = cfg["meta_embed_dim"],
        beta           = cfg["beta"],
        free_bits      = cfg["free_bits"],
        lambda_tc      = cfg["lambda_tc"],
    )
    model = FiLMMetaInjectionVAE(mc)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.to(device)
    return model, ckpt.get("residualizer", None)


def make_residualizer(resid_ckpt: dict | None):
    """Reconstruct IschemiaResidualizer from saved checkpoint."""
    if resid_ckpt is None:
        return None
    from analysis.q54_residual_encoding import IschemiaResidualizer
    r = IschemiaResidualizer()
    r.beta_isch = np.array(resid_ckpt["beta_isch"], dtype=np.float32)
    r.s_mean    = float(resid_ckpt["s_mean"])
    return r


# ── within-sample flip test ────────────────────────────────────────────────

def within_sample_flip(
    model: FiLMMetaInjectionVAE,
    X_sc: np.ndarray,
    M: np.ndarray,
    meta_df: pd.DataFrame,
    X_lc: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_std:  np.ndarray,
    col: str,
    dim_idx: int,
    resid=None,
    lo_pct: float = 25,
    hi_pct: float = 75,
) -> dict:
    """For each sample i, flip one meta dim from group-lo mean to group-hi mean.

    pred_ischemia_effect_per_gene = mean_i [ decode(z_bio_i, meta_hi) - decode(z_bio_i, meta_lo) ]
    Compared to ground truth = mean(X_lc[hi]) - mean(X_lc[lo]).
    """
    raw = pd.to_numeric(meta_df[col], errors="coerce").values
    hi_mask = raw > np.nanpercentile(raw, hi_pct)
    lo_mask = raw < np.nanpercentile(raw, lo_pct)
    if hi_mask.sum() == 0 or lo_mask.sum() == 0:
        return {"col": col, "error": "empty group"}

    true_fc = X_lc[hi_mask].mean(0) - X_lc[lo_mask].mean(0)
    mu_cpm  = 2 ** ((X_lc[hi_mask].mean(0) + X_lc[lo_mask].mean(0)) / 2) - 1

    hi_meta_val = float(M[hi_mask, dim_idx].mean())
    lo_meta_val = float(M[lo_mask, dim_idx].mean())

    # Build hi/lo meta override arrays for ALL samples
    M_hi = M.copy(); M_hi[:, dim_idx] = hi_meta_val
    M_lo = M.copy(); M_lo[:, dim_idx] = lo_meta_val

    # Prepare encoder input
    isch_z = M[:, 0]
    X_enc  = resid.transform(X_sc, isch_z) if resid is not None else X_sc

    dev = next(model.parameters()).device
    bs  = 128
    delta_sum = np.zeros(X_sc.shape[1], dtype=np.float64)
    n = len(X_sc)

    for s in range(0, n, bs):
        e = min(s + bs, n)
        with torch.no_grad():
            xt  = torch.from_numpy(X_enc[s:e].astype(np.float32)).to(dev)
            mhi = torch.from_numpy(M_hi[s:e].astype(np.float32)).to(dev)
            mlo = torch.from_numpy(M_lo[s:e].astype(np.float32)).to(dev)
            mu_, _ = model.encode(xt)
            xh_hi  = model.decode(mu_, mhi).cpu().numpy()  # (batch, G) standardised
            xh_lo  = model.decode(mu_, mlo).cpu().numpy()
        delta_sum += (xh_hi - xh_lo).sum(0)

    # Average delta in standardised space, then convert to log-CPM difference
    mean_delta_sc = delta_sum / n
    pred_fc = mean_delta_sc * scaler_std   # differences: mean cancels

    result = evaluate(pred_fc, true_fc, mu_cpm)

    # Per-bin breakdown
    by_bin = {k: float(v) for k, v in result["by_bin"]["accuracy"].items()}

    # Correlate with OLS ischemia effect (if this is the ischemia dim)
    pearson_r = float(np.corrcoef(pred_fc, true_fc)[0, 1]) if np.isfinite(pred_fc).all() else float("nan")

    return {
        "col":          col,
        "test_type":    "within_sample_flip",
        "overall":      result["accuracy"],
        "mae":          result["mae"],
        "mae_weighted": result["mae"],
        "by_bin":       by_bin,
        "pearson_r":    pearson_r,
        "n_hi":         int(hi_mask.sum()),
        "n_lo":         int(lo_mask.sum()),
        "delta_hi_meta_val": hi_meta_val,
        "delta_lo_meta_val": lo_meta_val,
    }


def within_sample_flip_dthhrdy(
    model, X_sc, M, X_lc, scaler_mean, scaler_std, meta_df, resid=None
) -> dict:
    """DTHHRDY: class 0 vs class 4."""
    raw = pd.to_numeric(meta_df["DTHHRDY"], errors="coerce").values
    mask_a, mask_b = raw == 0, raw == 4
    if mask_a.sum() < 5 or mask_b.sum() < 5:
        return {"col": "DTHHRDY", "error": "too few samples"}

    true_fc = X_lc[mask_b].mean(0) - X_lc[mask_a].mean(0)
    mu_cpm  = 2 ** ((X_lc[mask_b].mean(0) + X_lc[mask_a].mean(0)) / 2) - 1

    M_b = M.copy(); M_b[:, 1] = float(M[mask_b, 1].mean())
    M_a = M.copy(); M_a[:, 1] = float(M[mask_a, 1].mean())

    isch_z = M[:, 0]
    X_enc  = resid.transform(X_sc, isch_z) if resid is not None else X_sc

    dev = next(model.parameters()).device
    delta_sum = np.zeros(X_sc.shape[1], dtype=np.float64)
    bs = 128
    for s in range(0, len(X_sc), bs):
        e = min(s + bs, len(X_sc))
        with torch.no_grad():
            xt  = torch.from_numpy(X_enc[s:e].astype(np.float32)).to(dev)
            mb  = torch.from_numpy(M_b[s:e].astype(np.float32)).to(dev)
            ma  = torch.from_numpy(M_a[s:e].astype(np.float32)).to(dev)
            mu_, _ = model.encode(xt)
            delta_sum += (model.decode(mu_, mb) - model.decode(mu_, ma)).cpu().numpy().sum(0)

    pred_fc = (delta_sum / len(X_sc)) * scaler_std
    result  = evaluate(pred_fc, true_fc, mu_cpm)
    return {
        "col": "DTHHRDY", "test_type": "within_sample_flip_class0vs4",
        "overall": result["accuracy"], "mae": result["mae"],
        "by_bin": {k: float(v) for k, v in result["by_bin"]["accuracy"].items()},
        "pearson_r": float(np.corrcoef(pred_fc, true_fc)[0, 1]),
    }


# ── OLS ischemia correlation ───────────────────────────────────────────────

def ols_vs_decoder_ischemia(
    model, X_sc, M, scaler_mean, scaler_std, resid
) -> dict:
    """Per-gene: OLS beta_isch (from training) vs decoder's learned ischemia effect.

    Decoder effect = mean_i[ decode(z_bio_i, meta_hi_isch) - decode(z_bio_i, meta_lo_isch) ]
    divided by (hi_isch - lo_isch) to give per-unit slope.
    """
    if resid is None:
        return {}

    M_hi = M.copy(); M_hi[:, 0] = 1.5   # z-scored ischemia = +1.5σ
    M_lo = M.copy(); M_lo[:, 0] = -1.5  # z-scored ischemia = -1.5σ
    delta_z = 3.0  # M_hi - M_lo in z units

    X_enc = resid.transform(X_sc, M[:, 0])
    dev   = next(model.parameters()).device
    delta_sum = np.zeros(X_sc.shape[1], dtype=np.float64)
    bs = 128
    for s in range(0, len(X_sc), bs):
        e = min(s + bs, len(X_sc))
        with torch.no_grad():
            xt  = torch.from_numpy(X_enc[s:e].astype(np.float32)).to(dev)
            mhi = torch.from_numpy(M_hi[s:e].astype(np.float32)).to(dev)
            mlo = torch.from_numpy(M_lo[s:e].astype(np.float32)).to(dev)
            mu_, _ = model.encode(xt)
            delta_sum += (model.decode(mu_, mhi) - model.decode(mu_, mlo)).cpu().numpy().sum(0)

    # Decoder beta (per standardised unit) → log-CPM slope
    decoder_beta_logcpm = (delta_sum / len(X_sc) / delta_z) * scaler_std
    ols_beta_logcpm     = resid.beta_isch * scaler_std  # OLS was in standardised space

    r, _ = np.corrcoef(decoder_beta_logcpm, ols_beta_logcpm)[0, 1], None
    r = float(np.corrcoef(decoder_beta_logcpm, ols_beta_logcpm)[0, 1])

    # What fraction of OLS variance is captured by decoder?
    ss_res = ((decoder_beta_logcpm - ols_beta_logcpm) ** 2).sum()
    ss_tot = ((ols_beta_logcpm - ols_beta_logcpm.mean()) ** 2).sum()
    r2_vs_ols = float(1.0 - ss_res / (ss_tot + 1e-12))

    return {
        "decoder_vs_ols_pearson_r": r,
        "decoder_vs_ols_r2": r2_vs_ols,
        "ols_beta_max": float(np.abs(ols_beta_logcpm).max()),
        "decoder_beta_max": float(np.abs(decoder_beta_logcpm).max()),
        "ols_beta_logcpm": ols_beta_logcpm.tolist(),
        "decoder_beta_logcpm": decoder_beta_logcpm.tolist(),
    }


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print("Loading GTEx blood …")
    gtex         = load_gtex_blood(checkpoint_path=CKPT)
    meta         = load_metadata(gtex.sample_ids)
    _, sc_mean, sc_std = load_shared_genes(CKPT)
    X_sc         = gtex.expr_scaled
    X_lc         = gtex.expr_aligned
    M            = build_meta_matrix(meta)
    gene_names   = gtex.shared_genes
    hbb_idx      = int(np.where(gene_names == "HBB")[0][0])

    q53_path = ROOT / "analysis" / "results" / "q53_meta_injection_v2" / "film_meta_vae.pt"
    q54_path = ROOT / "analysis" / "results" / "q54_residual_encoding"  / "film_resid_vae.pt"

    models_to_test = []
    if q53_path.exists():
        m53, r53 = load_film_model(str(q53_path), DEVICE)
        models_to_test.append(("Q53 (no residual)", m53, r53))
        print("Loaded Q53 model ✓")
    if q54_path.exists():
        m54, r54_ckpt = load_film_model(str(q54_path), DEVICE)
        r54 = make_residualizer(r54_ckpt)
        models_to_test.append(("Q54 (residual)", m54, r54))
        print("Loaded Q54 model ✓")

    all_results = {}

    for tag, model, resid in models_to_test:
        print(f"\n{'═'*60}")
        print(f"  {tag}")
        print(f"{'═'*60}")
        res = {}

        # Within-sample flip tests
        print("\n── Within-sample flip tests ──")
        for col, dim_idx, label in [("SMTSISCH", 0, "ischemia"),
                                     ("AGE_mid",  2, "age")]:
            fr = within_sample_flip(model, X_sc, M, meta, X_lc, sc_mean, sc_std,
                                    col, dim_idx, resid=resid)
            res[f"within_flip_{col}"] = fr
            print(f"  {label:<12}  overall={fr['overall']:.4f}  "
                  f">1000={fr['by_bin'].get('>1000', float('nan')):.4f}  "
                  f"r={fr['pearson_r']:.4f}  MAE={fr['mae']:.4f}")

        fr_dth = within_sample_flip_dthhrdy(model, X_sc, M, X_lc, sc_mean, sc_std, meta, resid)
        res["within_flip_DTHHRDY"] = fr_dth
        print(f"  {'DTHHRDY':<12}  overall={fr_dth['overall']:.4f}  "
              f">1000={fr_dth['by_bin'].get('>1000', float('nan')):.4f}  "
              f"r={fr_dth.get('pearson_r', float('nan')):.4f}")

        # OLS vs decoder comparison (Q54 only, has resid)
        if resid is not None:
            print("\n── OLS ischemia slope vs decoder learned ischemia ──")
            ols_r = ols_vs_decoder_ischemia(model, X_sc, M, sc_mean, sc_std, resid)
            res["ols_vs_decoder"] = {k: v for k, v in ols_r.items()
                                     if not isinstance(v, list)}
            print(f"  Pearson r(decoder, OLS) = {ols_r['decoder_vs_ols_pearson_r']:.4f}")
            print(f"  R² decoder vs OLS       = {ols_r['decoder_vs_ols_r2']:.4f}")
            print(f"  |β_OLS|_max = {ols_r['ols_beta_max']:.4f}  "
                  f"|β_dec|_max = {ols_r['decoder_beta_max']:.4f}")

            # HBB-specific check
            beta_ols_hbb     = ols_r["ols_beta_logcpm"][hbb_idx]
            beta_dec_hbb     = ols_r["decoder_beta_logcpm"][hbb_idx]
            print(f"\n  HBB ischemia slope (log2-CPM per σ SMTSISCH):")
            print(f"    OLS estimate  : {beta_ols_hbb:.4f}")
            print(f"    Decoder learned: {beta_dec_hbb:.4f}")
            print(f"    Ratio         : {beta_dec_hbb/beta_ols_hbb:.3f}  "
                  f"(ideal: 1.0)")

        all_results[tag] = res

    # ── Comparison table ─────────────────────────────────────────────────
    print(f"\n\n{'═'*70}")
    print("SUMMARY — Within-sample flip test (correct design)")
    print(f"{'═'*70}")
    header = f"{'model':<22}  {'ischemia':>9}  {'isch >1k':>9}  {'age':>9}  {'DTHHRDY':>9}"
    print(header); print("─" * len(header))
    for tag, res in all_results.items():
        fi = res.get("within_flip_SMTSISCH", {})
        fa = res.get("within_flip_AGE_mid",  {})
        fd = res.get("within_flip_DTHHRDY",  {})
        print(f"{tag:<22}  "
              f"{fi.get('overall', float('nan')):>9.4f}  "
              f"{fi.get('by_bin', {}).get('>1000', float('nan')):>9.4f}  "
              f"{fa.get('overall', float('nan')):>9.4f}  "
              f"{fd.get('overall', float('nan')):>9.4f}")

    print(f"\n(Old group-FC flip for reference:)")
    print(f"  Q53: SMTSISCH=0.9268 >1k=0.2143 | Q54: SMTSISCH=0.8316 >1k=0.1429")
    print(f"  → Q54 scored lower on the OLD test because ischemia is correctly")
    print(f"    in the meta pathway, not z_bio.  Within-sample test fixes this.")

    # Save
    def _clean(obj):
        if isinstance(obj, dict): return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)): return float(obj)
        if isinstance(obj, list): return [_clean(v) for v in obj]
        return obj

    (OUT / "results.json").write_text(json.dumps(_clean(all_results), indent=2))
    print(f"\nSaved → {OUT}/results.json")


if __name__ == "__main__":
    main()
