"""Q59 — Push past 34% R² by adding sub-cell-type markers.

Q57 showed that 7 broad cell-type proxies + 8 clinical/technical variables
together explained 34% of z_bio variance.  Cell types of blood are more
heterogeneous than 7 categories suggest:

  T_cell → CD4_naive, CD4_memory, CD8_naive, CD8_cytotoxic, Treg
  B_cell → B_naive, B_memory, Plasmablast
  monocyte → classical, non-classical
  NK_cell → CD56bright, CD56dim
  neutrophil → mature, immature
  Plus state markers: activation, exhaustion, proliferation

This script defines ~20 sub-cell-type / state marker sets, computes scores,
and reruns the Q57/Q58 bidirectional analysis to see how high R² can go.

We also run Q58-style deconvolution comparison at the new granularity to
see if z_bio handles fine-grained populations better than coarse ones.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.data import load_gtex_blood, load_metadata, load_shared_genes
from models.meta_injection_vae import FiLMMetaInjectionVAE, FiLMMetaInjectionConfig
from analysis.q57_cell_composition import build_meta_matrix, build_base_meta, score_cell_types as score_broad

CKPT = "/Users/rls/Desktop/programming-projects/single-cell/bulk-project/analysis/14_cross_modality_vae/cross_modality_vae.pt"
Q54B = ROOT / "analysis" / "results" / "q54b_count_weighted" / "film_resid_vae_weighted.pt"
OUT  = ROOT / "analysis" / "results" / "q59_subcell_types"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 0


# ──────────────────────────────────────────────────────────────────────────
# Sub-cell-type + state markers (curated from PanglaoDB, Azimuth, Tabula Sapiens)
# ──────────────────────────────────────────────────────────────────────────

SUBCELL_MARKERS: dict[str, list[str]] = {
    # T cell subsets
    "CD4_naive":      ["CD4", "CCR7", "SELL", "LEF1", "TCF7", "BACH2", "S100A11"],
    "CD4_memory":     ["CD4", "S100A4", "IL7R", "AQP3", "KLRB1", "ITGB1"],
    "CD8_naive":      ["CD8A", "CD8B", "CCR7", "SELL", "LEF1", "TCF7"],
    "CD8_cytotoxic":  ["CD8A", "CD8B", "GZMA", "GZMB", "GZMK", "GZMH", "PRF1", "NKG7", "CCL5"],
    "Treg":           ["CD4", "FOXP3", "IL2RA", "CTLA4", "IKZF2", "TIGIT"],

    # B cell subsets
    "B_naive":        ["CD19", "MS4A1", "IGHD", "TCL1A", "BACH2"],
    "B_memory":       ["CD19", "MS4A1", "CD27", "S100A10", "AIM2"],
    "Plasmablast":    ["XBP1", "MZB1", "JCHAIN", "IGHG1", "IGHG2", "CD38", "PRDM1"],

    # Monocyte subsets
    "Mono_classical":     ["CD14", "S100A12", "VCAN", "FCN1", "MNDA", "S100A8", "S100A9"],
    "Mono_nonclassical":  ["FCGR3A", "MTSS1", "LST1", "CDKN1C", "CSF1R", "MS4A7"],

    # NK subsets
    "NK_bright":  ["NCAM1", "GZMK", "XCL1", "XCL2", "IL7R", "SELL"],
    "NK_dim":     ["FCGR3A", "GZMB", "KLRD1", "NKG7", "PRF1", "CX3CR1"],

    # Granulocyte subsets
    "Neut_mature":   ["FCGR3B", "CXCR2", "FPR1", "ALPL", "S100A12"],
    "Neut_immature": ["MPO", "ELANE", "DEFA1", "PRTN3", "CTSG", "AZU1"],
    "Eosinophil":    ["SIGLEC8", "CCR3", "IL5RA", "EPX", "PRG2"],
    "Basophil":      ["HDC", "ENPP3", "MS4A2", "CPA3"],

    # Dendritic cells
    "cDC1":  ["CLEC9A", "BATF3", "IRF8", "XCR1"],
    "cDC2":  ["CD1C", "CLEC10A", "FCER1A", "CLEC4A"],
    "pDC":   ["LILRA4", "IL3RA", "CLEC4C", "IRF7", "TCF4"],

    # Cell states (not types)
    "Activation":      ["CD69", "NR4A1", "EGR1", "FOS", "JUN", "IER2"],
    "Exhaustion":      ["PDCD1", "LAG3", "TIGIT", "HAVCR2", "TOX"],
    "Proliferation":   ["MKI67", "TOP2A", "PCNA", "CCNA2", "CCNB1", "BIRC5"],
    "Type1_IFN":       ["ISG15", "IFI6", "IFI27", "OAS1", "OAS2", "MX1", "IFIT1", "IFITM3"],
    "Inflammation":    ["IL1B", "IL6", "TNF", "CXCL8", "CCL2", "PTGS2"],
}


def score_subcell_types(X_lc: np.ndarray, gene_names: list[str]):
    upper_to_idx = {g.upper(): i for i, g in enumerate(gene_names)}
    names = list(SUBCELL_MARKERS.keys())
    used: dict[str, int] = {}
    scores = np.zeros((X_lc.shape[0], len(names)), dtype=np.float32)
    for j, t in enumerate(names):
        idx = [upper_to_idx[m.upper()] for m in SUBCELL_MARKERS[t]
               if m.upper() in upper_to_idx]
        used[t] = len(idx)
        if idx:
            scores[:, j] = X_lc[:, idx].mean(axis=1)
    z = (scores - scores.mean(0)) / (scores.std(0) + 1e-8)
    return z.astype(np.float32), names, used


def load_z_bio(X_sc, M4):
    ckpt = torch.load(Q54B, map_location="cpu")
    cfg  = ckpt["cfg"]
    mc   = FiLMMetaInjectionConfig(
        input_dim=cfg["input_dim"], meta_dim=cfg["meta_dim"],
        z_bio_dim=cfg["z_bio_dim"], decoder_hidden=tuple(cfg["decoder_hidden"]),
        meta_embed_dim=cfg["meta_embed_dim"], beta=cfg["beta"],
        free_bits=cfg["free_bits"], lambda_tc=cfg["lambda_tc"],
    )
    m = FiLMMetaInjectionVAE(mc).eval()
    m.load_state_dict(ckpt["state_dict"])
    beta_isch = np.array(ckpt["residualizer"]["beta_isch"], dtype=np.float32)
    s_mean    = float(ckpt["residualizer"]["s_mean"])
    X_resid   = X_sc - np.outer(M4[:, 0] - s_mean, beta_isch)
    with torch.no_grad():
        return m.encode(torch.from_numpy(X_resid.astype(np.float32)))[0].numpy()


def direction_A(X_meta, z):
    """Joint Ridge: predict z from metadata, 5-fold CV."""
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    z_sc = StandardScaler().fit_transform(z)
    x_sc = StandardScaler().fit_transform(X_meta)
    per_dim = []
    for k in range(z.shape[1]):
        sc = cross_val_score(Ridge(alpha=1.0), x_sc, z_sc[:, k],
                             cv=kf, scoring="r2")
        per_dim.append(float(sc.mean()))
    preds = np.zeros_like(z_sc)
    for tr, te in kf.split(x_sc):
        preds[te] = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(x_sc[tr], z_sc[tr]).predict(x_sc[te])
    ss_res = ((z_sc - preds) ** 2).sum()
    ss_tot = ((z_sc - z_sc.mean(0)) ** 2).sum()
    r2_joint = float(1 - ss_res / ss_tot)
    return {"r2_joint": r2_joint, "r2_per_dim": per_dim,
            "r2_max_dim": float(np.max(per_dim))}


def direction_B(z, X_meta, names):
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    z_sc = StandardScaler().fit_transform(z)
    rows = []
    for i, name in enumerate(names):
        y = X_meta[:, i]
        if y.std() < 1e-8:
            continue
        sc = cross_val_score(Ridge(alpha=1.0), z_sc, y, cv=kf, scoring="r2")
        # best single dim
        per_dim = [float(cross_val_score(Ridge(alpha=1.0), z_sc[:, k:k+1], y,
                                          cv=kf, scoring="r2").mean())
                   for k in range(z.shape[1])]
        rows.append({"variable": name, "full_z_r2": float(sc.mean()),
                     "full_z_r2_std": float(sc.std()),
                     "best_dim": int(np.argmax(per_dim)) + 1,
                     "best_dim_r2": float(np.max(per_dim)),
                     "per_dim_r2": per_dim})
    return rows


def main():
    print("Loading data ...")
    gtex = load_gtex_blood(checkpoint_path=CKPT)
    meta = load_metadata(gtex.sample_ids)
    M4   = build_meta_matrix(meta)
    X_lc = gtex.expr_aligned
    X_sc = gtex.expr_scaled
    gene_names = list(gtex.shared_genes)

    print("Computing sub-cell-type marker scores ...")
    sub_scores, sub_names, used = score_subcell_types(X_lc, gene_names)
    print(f"  {len(sub_names)} sub-types / states scored")
    found_counts = {n: f"{used[n]}/{len(SUBCELL_MARKERS[n])}" for n in sub_names}
    for n in sub_names:
        print(f"    {n:<22} markers: {found_counts[n]}")

    # Cross-correlations between sub-types
    cc = pd.DataFrame(np.corrcoef(sub_scores.T), index=sub_names, columns=sub_names)
    cc.to_csv(OUT / "subtype_correlations.csv")

    print("Encoding z_bio ...")
    z_bio = load_z_bio(X_sc, M4)

    # Base metadata + broad cell types (from Q57)
    base_meta, base_names = build_base_meta(meta)
    broad_scores, broad_names, _ = score_broad(X_lc, gene_names)

    # Three nested metadata matrices
    matrices = {
        "base_9":        (base_meta, base_names),
        "base+broad_16": (np.column_stack([base_meta, broad_scores]),
                          base_names + broad_names),
        "base+broad+sub": (np.column_stack([base_meta, broad_scores, sub_scores]),
                            base_names + broad_names + sub_names),
    }

    print("\n══ Direction A: metadata → z_bio (joint Ridge R²) ══")
    direction_A_results = {}
    for label, (M, names) in matrices.items():
        r = direction_A(M, z_bio)
        direction_A_results[label] = r
        print(f"  {label:<18} n={M.shape[1]:>3}  joint R²={r['r2_joint']:.3f}  "
              f"max-dim R²={r['r2_max_dim']:.3f}")

    delta_total = (direction_A_results["base+broad+sub"]["r2_joint"]
                   - direction_A_results["base_9"]["r2_joint"])
    delta_sub = (direction_A_results["base+broad+sub"]["r2_joint"]
                 - direction_A_results["base+broad_16"]["r2_joint"])
    print(f"\n  Δ from sub-cell-types alone:  +{delta_sub:.3f}")
    print(f"  Δ total (base → base+broad+sub): +{delta_total:.3f}")

    print("\n══ Direction B: z_bio → each variable ══")
    M_full, names_full = matrices["base+broad+sub"]
    b_rows = direction_B(z_bio, M_full, names_full)
    b_df = pd.DataFrame(b_rows).sort_values("full_z_r2", ascending=False)

    # Tag whether each is base / broad / sub
    def _tag(n):
        if n in base_names:  return "base"
        if n in broad_names: return "broad"
        return "sub/state"
    b_df["category"] = b_df["variable"].apply(_tag)

    print(f"\n{'variable':<22}  {'category':<10}  {'full-z R²':>10}  best_dim")
    print("─" * 60)
    for _, row in b_df.iterrows():
        print(f"{row['variable']:<22}  {row['category']:<10}  "
              f"{row['full_z_r2']:>10.3f}  z{int(row['best_dim']):<2}")

    # Per-dim best variable
    print("\n══ Per-dim 'top variable' assignment ══")
    K = z_bio.shape[1]
    per_dim_best = []
    for k in range(K):
        # find variable with best per-dim R² for this dim
        scores = [(r["variable"], r["per_dim_r2"][k]) for r in b_rows]
        scores.sort(key=lambda x: x[1], reverse=True)
        per_dim_best.append({"dim": f"z{k+1}",
                             "top1": scores[0][0], "top1_r2": scores[0][1],
                             "top2": scores[1][0], "top2_r2": scores[1][1],
                             "top3": scores[2][0], "top3_r2": scores[2][1]})
    pdb_df = pd.DataFrame(per_dim_best)
    print(pdb_df[["dim", "top1", "top1_r2", "top2", "top2_r2"]].to_string(index=False))

    # Save
    pdb_df.to_csv(OUT / "per_dim_top_variables.csv", index=False)
    b_df.to_csv(OUT / "direction_B_full.csv", index=False)
    full_atlas = pd.DataFrame(
        np.array([r["per_dim_r2"] for r in b_rows]),
        index=[r["variable"] for r in b_rows],
        columns=[f"z{k+1}" for k in range(K)],
    )
    full_atlas.to_csv(OUT / "latent_atlas_full.csv")

    def _clean(o):
        if isinstance(o, dict): return {k: _clean(v) for k,v in o.items()}
        if isinstance(o, (np.floating, np.integer)): return float(o)
        if isinstance(o, list): return [_clean(v) for v in o]
        return o
    json_out = {
        "direction_A":          _clean(direction_A_results),
        "direction_B":          _clean(b_rows),
        "per_dim_assignment":   pdb_df.to_dict(orient="records"),
        "delta_joint_from_subcells": float(delta_sub),
        "delta_joint_total":         float(delta_total),
        "n_subtypes_scored":         len(sub_names),
    }
    (OUT / "results.json").write_text(json.dumps(json_out, indent=2))

    print(f"\nSaved → {OUT}/")
    print("\n" + "═" * 60)
    print("SUMMARY")
    print("═" * 60)
    print(f"Joint R²(metadata → z_bio):")
    for k, v in direction_A_results.items():
        print(f"  {k:<18}  R²={v['r2_joint']:.3f}")
    print(f"\nTotal metadata variables tested: {M_full.shape[1]}")
    print(f"Best single variable (z→meta):    "
          f"{b_df.iloc[0]['variable']} = {b_df.iloc[0]['full_z_r2']:.3f} R²")


if __name__ == "__main__":
    main()
