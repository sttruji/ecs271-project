"""High-level evaluation runner — give it a model, get a full report."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .data import GTExBlood, load_gtex_blood, load_metadata
from .enrichment import DEFAULT_LIBRARIES, enrich_dim_loadings
from .latent import (correlate_z_with_pcs, evaluate_latent_activity,
                     evaluate_latent_meta, state_classification_auc)
from .reconstruction import evaluate_reconstruction


@dataclass
class EvalConfig:
    name: str = "model"
    out_dir: Path = field(default_factory=lambda: Path("eval_output"))
    train_test_split_seed: int = 0
    test_frac: float = 0.2
    pca_ranks: tuple[int, ...] = (16, 50, 200, 500)
    enrich_top_n_dims: int = 3
    enrich_top_genes: int = 200
    enrich_libraries: tuple[str, ...] = DEFAULT_LIBRARIES
    skip_enrichment: bool = False
    checkpoint_path: Optional[str] = None


def _split(n: int, test_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(round(test_frac * n))
    return perm[n_test:], perm[:n_test]


def run_evaluation(model, cfg: EvalConfig, *, gtex: Optional[GTExBlood] = None) -> dict:
    """Run the full evaluation suite against the GTEx whole-blood dataset.

    Outputs (written to cfg.out_dir):
      - reconstruction.json   — model + PCA + mean R²/MSE
      - latent_activity.json  — collapse diagnostic
      - z_pc_corr.npy         — |corr| matrix between latent and bulk PC1-10
      - state_auc.json        — anchor-gene state classification
      - metadata_spearman.csv — Spearman ρ between metadata and latent dims
      - metadata_eta2.csv     — categorical η² between metadata and latent dims
      - enrichment/<dim>__<direction>__<library>.csv  — Enrichr tables
      - summary.json          — top-line numbers, easy to consume

    Returns the summary dict.
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    enrich_dir = cfg.out_dir / "enrichment"
    enrich_dir.mkdir(exist_ok=True)

    if gtex is None:
        print("[1/6] Loading GTEx blood ...")
        gtex = load_gtex_blood(checkpoint_path=cfg.checkpoint_path)

    train_idx, test_idx = _split(gtex.n_donors, cfg.test_frac, cfg.train_test_split_seed)
    expr_train = gtex.expr_scaled[train_idx]
    expr_test = gtex.expr_scaled[test_idx]

    # ── Q1 ────────────────────────────────────────────────────────────────
    print("[2/6] Reconstruction quality ...")
    recon = evaluate_reconstruction(
        model, expr_train, expr_test, pca_ranks=cfg.pca_ranks
    )
    (cfg.out_dir / "reconstruction.json").write_text(json.dumps(recon, indent=2))

    # ── Q2 ────────────────────────────────────────────────────────────────
    print("[3/6] Latent activity / collapse diagnostic ...")
    z_all = model.encode(gtex.expr_scaled)
    activity = evaluate_latent_activity(z_all)
    (cfg.out_dir / "latent_activity.json").write_text(json.dumps(activity, indent=2))

    print("[4/6] Latent vs bulk PCs ...")
    corr = correlate_z_with_pcs(z_all, gtex.expr_scaled, n_pcs=10)
    np.save(cfg.out_dir / "z_pc_corr.npy", corr)

    state = state_classification_auc(z_all, gtex.expr_log, gtex.gene_names)
    (cfg.out_dir / "state_auc.json").write_text(json.dumps(state, indent=2))

    # ── Q4 ────────────────────────────────────────────────────────────────
    print("[5/6] Metadata correlation ...")
    metadata = load_metadata(gtex.sample_ids)
    metadata.to_csv(cfg.out_dir / "metadata.csv", index=False)
    meta = evaluate_latent_meta(z_all, metadata)
    meta["spearman"].to_csv(cfg.out_dir / "metadata_spearman.csv")
    meta["eta2"].to_csv(cfg.out_dir / "metadata_eta2.csv")

    # ── Optional: Enrichr on the most-active dims ─────────────────────────
    enrichment_summary: list[dict] = []
    if not cfg.skip_enrichment:
        print(f"[6/6] Enrichr on top-{cfg.enrich_top_n_dims} active dims ...")
        var_per_dim = z_all.var(0)
        # Skip dims with vanishing variance — Enrichment on collapsed dims is meaningless
        active = np.where(var_per_dim > 1e-4)[0]
        if len(active) == 0:
            print("  no active dims — skipping enrichment")
        else:
            top_dims = active[np.argsort(var_per_dim[active])[-cfg.enrich_top_n_dims:][::-1]]
            # We need a mapping from latent dim k → gene loading vector. The
            # cleanest *model-agnostic* approach is the decoder-Jacobian at z=0,
            # i.e. ∂x̂/∂z_k. Use a finite-difference probe.
            for dim_idx in top_dims:
                loading = _finite_diff_loading(model, dim_idx, gtex.n_shared_genes)
                tag = f"z{int(dim_idx) + 1}"
                tables = enrich_dim_loadings(
                    loading,
                    gtex.shared_genes,
                    top_n=cfg.enrich_top_genes,
                    libraries=cfg.enrich_libraries,
                    description=f"{cfg.name}_{tag}",
                )
                for direction, lib_dict in tables.items():
                    for lib, tbl in lib_dict.items():
                        if lib.startswith("_"):
                            continue
                        path = enrich_dir / f"{tag}__{direction}__{lib}.csv"
                        tbl.to_csv(path, index=False)
                        if not tbl.empty and "term" in tbl.columns:
                            row = tbl.iloc[0]
                            enrichment_summary.append({
                                "dim": int(dim_idx) + 1,
                                "direction": direction,
                                "library": lib,
                                "top_term": str(row["term"]),
                                "top_adj_p": float(row["adj_p"]),
                            })
            pd.DataFrame(enrichment_summary).to_csv(
                cfg.out_dir / "enrichment_summary.csv", index=False)

    # ── consumable summary ────────────────────────────────────────────────
    summary = _build_summary(cfg, recon, activity, state, meta, enrichment_summary, corr)
    (cfg.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _finite_diff_loading(model, dim_idx: int, n_genes: int, eps: float = 1e-2) -> np.ndarray:
    """Decoder-Jacobian column at z=0 for the requested latent dim.

    Falls back to *zero* if the model doesn't have a decode method.
    """
    if not hasattr(model, "decode"):
        return np.zeros(n_genes, dtype=np.float32)
    z0 = np.zeros((1, model.latent_dim), dtype=np.float32)
    zk = z0.copy()
    zk[0, dim_idx] = eps
    x0 = model.decode(z0)[0]
    xk = model.decode(zk)[0]
    return ((xk - x0) / eps).astype(np.float32)


def _build_summary(
    cfg: EvalConfig,
    recon: dict,
    activity: dict,
    state: dict,
    meta: dict,
    enrichment_summary: list[dict],
    corr: np.ndarray,
) -> dict:
    rho_cont = meta["spearman"].astype(float)
    eta_cat = meta["eta2"].astype(float)
    summary_meta = {}
    for col in rho_cont.index:
        vals = rho_cont.loc[col].abs()
        if vals.notna().any():
            summary_meta[col] = {
                "max_abs_rho": float(vals.max()),
                "best_dim": str(vals.idxmax()),
            }
    cat_meta = {}
    for col in eta_cat.index:
        vals = eta_cat.loc[col]
        if vals.notna().any():
            cat_meta[col] = {
                "max_eta2": float(vals.max()),
                "best_dim": str(vals.idxmax()),
            }
    return {
        "name": cfg.name,
        "reconstruction": {k: {kk: vv for kk, vv in v.items()
                                if kk in ("mse", "r2_overall", "per_sample_r_mean", "per_gene_r_mean")}
                           for k, v in recon.items()},
        "latent_activity": activity,
        "state_b_classification": state,
        "metadata_continuous": summary_meta,
        "metadata_categorical": cat_meta,
        "max_corr_z_with_bulk_pc1_10": [float(v) for v in corr.max(axis=0)],
        "enrichment_top_terms": enrichment_summary,
    }
