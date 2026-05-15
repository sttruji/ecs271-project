"""Q54b — Q54 with count-weighted MSE loss.

Root cause of HBB still failing:
  The decoder learned only 62% of HBB's ischemia effect (0.296 vs OLS 0.479 log-CPM/σ).
  With uniform MSE, the 11374-gene output distributes capacity equally; HBB gets 1/11374 of
  the gradient signal for its precise ischemia fit.

Fix: weight each gene's reconstruction loss by its average count (capped at 10×).
  This aligns the training objective with the FC resolution benchmark criterion:
    - Tolerance ∝ 1/sqrt(count) → tighter for high-count genes
    - Count-weighted MSE ∝ count → more gradient for high-count genes
    - HBB at 70754 CPM gets weight = 10 (max cap)

Everything else identical to Q54: ischemia-residual encoder, FiLM decoder, K=16.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.data import load_gtex_blood, load_metadata, load_shared_genes
from pipeline.latent import metadata_linear_probe
from analysis.bulk_rnaseq_resolution_benchmark import evaluate, poisson_tolerance
from analysis.q54_residual_encoding import (
    IschemiaResidualizer, build_meta_matrix, roundtrip_test,
    flip_test_continuous, flip_test_dthhrdy, disentanglement_report,
    biological_analysis, encode_with_resid
)
from models.meta_injection_vae import FiLMMetaInjectionVAE, FiLMMetaInjectionConfig

CKPT   = "/Users/rls/Desktop/programming-projects/single-cell/bulk-project/analysis/14_cross_modality_vae/cross_modality_vae.pt"
OUT    = ROOT / "analysis" / "results" / "q54b_count_weighted"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED   = 0
META_DIM = 4
torch.manual_seed(SEED)
np.random.seed(SEED)

TRAIN_CFG = dict(
    z_bio_dim=16, decoder_hidden=(512, 512), epochs=600,
    beta=5e-4, lambda_tc=0.0, beta_warmup=150, tc_warmup=0,
)
COUNT_WEIGHT_CAP = 10.0   # max weight ratio for any gene


def compute_gene_weights(X_lc: np.ndarray) -> np.ndarray:
    """Count-based gene weights aligned with FC resolution tolerance.

    weight[g] = clip(avg_cpm[g] / 100, 1, cap) / mean(...)
    Normalised to mean=1 so total loss scale is unchanged.
    """
    avg_logcpm = X_lc.mean(0)                         # (G,) average log2(CPM+1)
    avg_cpm    = 2.0 ** avg_logcpm - 1.0
    raw_w      = np.clip(avg_cpm / 100.0, 1.0, COUNT_WEIGHT_CAP)
    return (raw_w / raw_w.mean()).astype(np.float32)


def train_model(model: FiLMMetaInjectionVAE,
                X_tr_resid: np.ndarray,
                X_tr_full: np.ndarray,
                M_tr: np.ndarray,
                gene_weights: np.ndarray,
                cfg: dict) -> list[dict]:
    epochs      = cfg["epochs"]
    beta_target = cfg["beta"]
    beta_warmup = cfg["beta_warmup"]

    model.to(DEVICE).train()
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    Xr = torch.from_numpy(X_tr_resid.astype(np.float32))
    Xf = torch.from_numpy(X_tr_full.astype(np.float32))
    Mt = torch.from_numpy(M_tr.astype(np.float32))
    Wt = torch.from_numpy(gene_weights).to(DEVICE)     # (G,) gene weights

    dl = DataLoader(TensorDataset(Xr, Xf, Mt), batch_size=64, shuffle=True)

    log = []
    for ep in range(1, epochs + 1):
        beta_ep = beta_target * min(1.0, ep / max(1, beta_warmup))
        model.train()
        ep_recon = ep_kl = 0.0
        for xr_b, xf_b, mb in dl:
            xr_b = xr_b.to(DEVICE)
            xf_b = xf_b.to(DEVICE)
            mb   = mb.to(DEVICE)

            mu, lv = model.encode(xr_b)
            z      = mu + (0.5 * lv).exp() * torch.randn_like(mu)
            x_hat  = model.decode(z, mb)

            # Count-weighted MSE: each gene weighted by its average count proxy
            recon = ((x_hat - xf_b).pow(2) * Wt).mean()
            kl_pd = -0.5 * (1.0 + lv - mu.pow(2) - lv.exp())
            kl    = kl_pd.clamp(min=model.cfg.free_bits).sum(-1).mean()
            loss  = recon + beta_ep * kl

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_recon += recon.item()
            ep_kl    += kl.item()
        sched.step()

        if ep % 100 == 0 or ep == 1:
            nb = len(dl)
            print(f"  ep {ep:4d}/{epochs}  β={beta_ep:.1e}  "
                  f"recon_wt={ep_recon/nb:.5f}  kl={ep_kl/nb:.4f}")
            log.append({"epoch": ep, "beta": beta_ep,
                        "recon_weighted": ep_recon / nb, "kl": ep_kl / nb})
    return log


def evaluate_fc_resolution(model, X_sc, M, X_lc, meta_df,
                            sc_mean, sc_std, resid, gene_names) -> dict:
    isch = pd.to_numeric(meta_df["SMTSISCH"], errors="coerce").values
    hi   = isch > np.nanpercentile(isch, 75)
    lo   = isch < np.nanpercentile(isch, 25)
    true_fc = X_lc[hi].mean(0) - X_lc[lo].mean(0)
    mu_cpm  = 2 ** ((X_lc[hi].mean(0) + X_lc[lo].mean(0)) / 2) - 1

    def _recon(mask):
        X_r = resid.transform(X_sc[mask], M[mask, 0])
        dev = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            xt  = torch.from_numpy(X_r.astype(np.float32)).to(dev)
            mt  = torch.from_numpy(M[mask].astype(np.float32)).to(dev)
            mu_, _ = model.encode(xt)
            xh = model.decode(mu_, mt).cpu().numpy()
        return xh * sc_std + sc_mean

    pred_fc = _recon(hi).mean(0) - _recon(lo).mean(0)
    result  = evaluate(pred_fc, true_fc, mu_cpm)
    tol     = poisson_tolerance(mu_cpm)
    err     = np.abs(pred_fc - true_fc)
    hi_cnt  = mu_cpm > 1000
    fail    = hi_cnt & (err > tol)

    failing = [(str(gene_names[i]), float(mu_cpm[i]), float(true_fc[i]),
                float(pred_fc[i]), float(err[i]), float(tol[i]))
               for i in np.where(fail)[0]]

    return {
        "overall":   result["accuracy"],
        "mae":       result["mae"],
        "by_bin":    {k: float(v) for k, v in result["by_bin"]["accuracy"].items()},
        "n_failing_hi_count": int(fail.sum()),
        "failing_genes": failing,
    }


def main():
    print("Loading GTEx blood …")
    gtex         = load_gtex_blood(checkpoint_path=CKPT)
    meta         = load_metadata(gtex.sample_ids)
    _, sc_mean, sc_std = load_shared_genes(CKPT)
    X_sc         = gtex.expr_scaled
    X_lc         = gtex.expr_aligned
    M            = build_meta_matrix(meta)
    n_genes      = X_sc.shape[1]
    gene_names   = gtex.shared_genes
    meta_ref     = M.mean(0)

    # 80/20 split
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X_sc))
    n_test = int(0.2 * len(X_sc))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    X_tr, M_tr, X_lc_tr = X_sc[train_idx], M[train_idx], X_lc[train_idx]

    # Gene weights (from all samples — count distribution is stable)
    gene_weights = compute_gene_weights(X_lc)
    hbb_idx = np.where(gene_names == "HBB")[0][0]
    print(f"Gene weights — HBB: {gene_weights[hbb_idx]:.2f}  "
          f"mean: {gene_weights.mean():.2f}  "
          f"top-5 genes: {[(str(gene_names[i]), f'{gene_weights[i]:.1f}') for i in np.argsort(gene_weights)[-5:]]}")

    # Ischemia residualization
    print("\n── Ischemia residualization ──")
    resid = IschemiaResidualizer()
    resid.fit(X_tr, M_tr[:, 0])
    X_tr_resid = resid.transform(X_tr, M_tr[:, 0])

    # Train
    print(f"\n── Training K={TRAIN_CFG['z_bio_dim']} ep={TRAIN_CFG['epochs']} "
          f"count-weighted MSE ──")
    model_cfg = FiLMMetaInjectionConfig(
        input_dim=n_genes, meta_dim=META_DIM,
        z_bio_dim=TRAIN_CFG["z_bio_dim"],
        decoder_hidden=TRAIN_CFG["decoder_hidden"],
        meta_embed_dim=128, beta=TRAIN_CFG["beta"],
        free_bits=0.1, lambda_tc=0.0,
    )
    model = FiLMMetaInjectionVAE(model_cfg)
    log   = train_model(model, X_tr_resid, X_tr, M_tr, gene_weights, TRAIN_CFG)

    # Step 3: FC resolution
    print("\n── Step 3: FC resolution ──")
    fc = evaluate_fc_resolution(model, X_sc, M, X_lc, meta, sc_mean, sc_std, resid, gene_names)
    print(f"  overall={fc['overall']:.4f}  >1000={fc['by_bin'].get('>1000',float('nan')):.4f}")
    for b, a in fc["by_bin"].items():
        print(f"    {b:>8}: {a:.4f}")
    if fc["failing_genes"]:
        for g, mu, tfc, pfc, err, tol in fc["failing_genes"]:
            print(f"  FAIL: {g}  mu={mu:.0f}  tol={tol:.4f}  "
                  f"true={tfc:.4f}  pred={pfc:.4f}  err={err:.4f}")
    else:
        print("  ✓ ALL high-count genes pass! FC = 100%")

    # Step 4a: roundtrip
    print("\n── Step 4a: Roundtrip ──")
    rt = roundtrip_test(model, X_sc, M, X_lc, sc_std, sc_mean, resid)
    print(f"  R² median={rt['r2_per_sample_median']:.4f}  "
          f"mean={rt['r2_per_sample_mean']:.4f}")

    # Step 4b: flip (within-sample, from Q55 design)
    print("\n── Step 4b: Within-sample flip tests ──")
    from analysis.q55_flip_analysis import within_sample_flip, within_sample_flip_dthhrdy
    flip_results = []
    for col, dim_idx, label in [("SMTSISCH", 0, "ischemia"), ("AGE_mid", 2, "age")]:
        fr = within_sample_flip(model, X_sc, M, meta, X_lc, sc_mean, sc_std,
                                col, dim_idx, resid=resid)
        flip_results.append(fr)
        print(f"  {label:<12}  overall={fr['overall']:.4f}  "
              f">1000={fr['by_bin'].get('>1000', float('nan')):.4f}  "
              f"r={fr.get('pearson_r', float('nan')):.4f}")
    fr_dth = within_sample_flip_dthhrdy(model, X_sc, M, X_lc, sc_mean, sc_std, meta, resid)
    flip_results.append(fr_dth)
    print(f"  {'DTHHRDY':<12}  overall={fr_dth['overall']:.4f}  "
          f">1000={fr_dth['by_bin'].get('>1000', float('nan')):.4f}")

    # Step 5: disentanglement
    print("\n── Step 5: Disentanglement ──")
    z_bio = encode_with_resid(model, X_sc, M[:, 0], resid)
    dis   = disentanglement_report(z_bio, meta)
    p     = dis["probes"]
    print(f"  SMTSISCH R²  = {p['SMTSISCH']['cv_r2']:.3f}")
    print(f"  AGE_mid  R²  = {p['AGE_mid']['cv_r2']:.3f}")
    print(f"  DTHHRDY acc  = {p['DTHHRDY']['cv_balanced_acc']:.3f}")
    print(f"  active dims  = {dis['n_active_dims']} / {model.n_latent}")
    print("  per-dim ischemia R²:", [f"{r:.2f}" for r in dis["isch_per_dim_r2"]])

    # Step 6: bio analysis
    print("\n── Step 6: Biological analysis ──")
    bio = biological_analysis(model, z_bio, gene_names, meta_ref, OUT)
    for k, info in bio["top_genes_per_dim"].items():
        print(f"  {k}  pos={info['top_pos'][:3]}  neg={info['top_neg'][:3]}")

    # Save
    def _clean(obj):
        if isinstance(obj, dict): return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)): return float(obj)
        if isinstance(obj, list): return [_clean(v) for v in obj]
        return obj

    results = {
        "fc_resolution":   _clean(fc),
        "roundtrip":       _clean(rt),
        "flip":            _clean(flip_results),
        "disentanglement": _clean(dis),
        "bio_top_genes":   _clean(bio["top_genes_per_dim"]),
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
        "gene_weights": gene_weights.tolist(),
    }, OUT / "film_resid_vae_weighted.pt")

    print(f"\nSaved → {OUT}/")
    print("\n══ SUMMARY ══")
    print(f"FC overall     = {fc['overall']:.4f}")
    print(f"FC >1000-count = {fc['by_bin'].get('>1000', float('nan')):.4f}")
    print(f"Roundtrip R²   = {rt['r2_per_sample_median']:.4f}")
    for fr in flip_results:
        d = fr.get("col", fr.get("dim", "?"))
        print(f"Flip {d:<12} = {fr['overall']:.4f}  >1000={fr['by_bin'].get('>1000', float('nan')):.4f}")
    print(f"SMTSISCH R² z  = {p['SMTSISCH']['cv_r2']:.3f}")


if __name__ == "__main__":
    main()
