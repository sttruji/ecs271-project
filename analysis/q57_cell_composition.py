"""Q57 — Add cell-composition proxies to the metadata matrix, rerun Q56.

In Q56, 9 GTEx metadata variables explained only 4% of z_bio variance jointly.
The dominant unmeasured driver of inter-donor blood RNA-seq variance is *cell
composition* — the relative abundance of neutrophils, T cells, B cells, NK
cells, monocytes, erythroid remnants and platelets in each sample.

Real deconvolution (CIBERSORT/CIBERSORTx/EPIC/xCell) requires either an R install
or a web upload. A well-validated quick proxy is **marker-gene scoring**:
average the expression of canonical markers for each cell type and use that as
a continuous abundance estimate. This is the same idea as Seurat's
`AddModuleScore` / scanpy's `score_genes` and is the basis of MCP-counter.

We add 7 cell-type proxies to the existing 9 metadata variables → 16 features,
then rerun the bidirectional R² analysis from Q56.

Expected: large jump in joint R² (predicted: 0.04 → 0.3-0.5) if cell composition
is the dominant unmeasured driver. If R² stays flat, the remaining variance is
something else (genetic, environmental, noise).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.data import load_gtex_blood, load_metadata, load_shared_genes
from models.meta_injection_vae import FiLMMetaInjectionVAE, FiLMMetaInjectionConfig

CKPT   = "/Users/rls/Desktop/programming-projects/single-cell/bulk-project/analysis/14_cross_modality_vae/cross_modality_vae.pt"
Q54B   = ROOT / "analysis" / "results" / "q54b_count_weighted" / "film_resid_vae_weighted.pt"
OUT    = ROOT / "analysis" / "results" / "q57_cell_composition"
OUT.mkdir(parents=True, exist_ok=True)
SEED   = 0


# ──────────────────────────────────────────────────────────────────────────
# Canonical blood cell-type markers (from PanglaoDB + LM22 consensus)
# ──────────────────────────────────────────────────────────────────────────

CELL_TYPE_MARKERS: dict[str, list[str]] = {
    "neutrophil": [
        "MPO", "ELANE", "S100A8", "S100A9", "CTSG", "PRTN3", "FCGR3B",
        "CXCR2", "FPR1", "CSF3R", "LCN2", "CAMP", "MMP9", "LTF",
        "DEFA1", "DEFA3", "DEFA4",
    ],
    "T_cell": [
        "CD3D", "CD3E", "CD3G", "CD2", "CD8A", "CD8B", "CD4", "TRAC",
        "TRBC1", "TRBC2", "IL7R", "TCF7", "LEF1", "LCK", "ITK",
    ],
    "B_cell": [
        "CD19", "MS4A1", "CD79A", "CD79B", "BANK1", "PAX5", "TCL1A",
        "IGHM", "IGHD", "IGKC", "IGLC2", "CD22", "BLK", "BLNK",
    ],
    "NK_cell": [
        "NCAM1", "GNLY", "NKG7", "KLRD1", "KLRF1", "KLRB1", "PRF1",
        "GZMA", "GZMB", "GZMH", "FCGR3A", "XCL1", "XCL2",
    ],
    "monocyte": [
        "CD14", "LYZ", "FCN1", "VCAN", "CD163", "S100A12", "MAFB",
        "CSF1R", "MARCO", "CST3", "CLEC10A", "FCGR1A",
    ],
    "erythroid": [   # RBC remnants / immature reticulocytes
        "HBA1", "HBA2", "HBB", "ALAS2", "GYPA", "GYPB", "EPB42",
        "ANK1", "KEL", "SLC4A1", "HBM", "HEMGN", "AHSP",
    ],
    "platelet": [
        "PPBP", "PF4", "ITGA2B", "GP9", "GP1BA", "GP1BB", "TUBB1",
        "SELP", "MYL9", "CLU", "TREML1", "SH3BGRL2",
    ],
}


# ──────────────────────────────────────────────────────────────────────────
# Cell-type scoring
# ──────────────────────────────────────────────────────────────────────────

def score_cell_types(
    X_logcpm: np.ndarray,
    gene_names: list[str],
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    """For each cell type, compute mean log-CPM of its markers per sample.

    Returns (scores, type_names, n_markers_used) where:
      scores   shape (n_samples, n_types) — z-scored across samples
      n_markers_used[type] = how many of the canonical markers were present
    """
    upper_to_idx = {g.upper(): i for i, g in enumerate(gene_names)}
    type_names = list(CELL_TYPE_MARKERS.keys())
    n_markers_used: dict[str, int] = {}

    scores_raw = np.zeros((X_logcpm.shape[0], len(type_names)), dtype=np.float32)

    for j, t in enumerate(type_names):
        markers = CELL_TYPE_MARKERS[t]
        idx = [upper_to_idx[m.upper()] for m in markers if m.upper() in upper_to_idx]
        n_markers_used[t] = len(idx)
        if len(idx) == 0:
            continue
        # Mean log2(CPM+1) across present markers, per sample
        scores_raw[:, j] = X_logcpm[:, idx].mean(axis=1)

    # Z-score across samples so the scale matches the other meta variables
    scores_z = (scores_raw - scores_raw.mean(0)) / (scores_raw.std(0) + 1e-8)
    return scores_z.astype(np.float32), type_names, n_markers_used


# ──────────────────────────────────────────────────────────────────────────
# Existing 9-var metadata (unchanged from Q56)
# ──────────────────────────────────────────────────────────────────────────

def build_base_meta(meta_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    from sklearn.preprocessing import LabelEncoder

    def _z(s):
        v = pd.to_numeric(s, errors="coerce").values.astype(float)
        z = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-8)
        return np.where(np.isfinite(z), z, 0.0).astype(np.float32)

    def _cat(s):
        v = s.fillna("MISSING").astype(str).values
        codes = LabelEncoder().fit_transform(v).astype(float)
        return ((codes - codes.mean()) / (codes.std() + 1e-8)).astype(np.float32)

    features = {}
    for col in ["SMTSISCH", "SMRIN", "AGE_mid", "DTHHRDY"]:
        if col in meta_df.columns:
            features[col] = _z(meta_df[col])

    if "SEX" in meta_df.columns:
        sex = pd.to_numeric(meta_df["SEX"], errors="coerce").values.astype(float)
        features["SEX"] = np.where(np.isfinite(sex), sex - 1.5, 0.0).astype(np.float32)

    for col in ["SMCENTER", "SMNABTCH", "SMGEBTCH"]:
        if col in meta_df.columns:
            features[col + "_code"] = _cat(meta_df[col])

    X = np.column_stack(list(features.values())).astype(np.float32)
    return X, list(features.keys())


# ──────────────────────────────────────────────────────────────────────────
# Load z_bio
# ──────────────────────────────────────────────────────────────────────────

def load_z_bio(X_sc: np.ndarray, M4: np.ndarray) -> np.ndarray:
    ckpt = torch.load(Q54B, map_location="cpu")
    cfg  = ckpt["cfg"]
    mc   = FiLMMetaInjectionConfig(
        input_dim=cfg["input_dim"], meta_dim=cfg["meta_dim"],
        z_bio_dim=cfg["z_bio_dim"], decoder_hidden=tuple(cfg["decoder_hidden"]),
        meta_embed_dim=cfg["meta_embed_dim"], beta=cfg["beta"],
        free_bits=cfg["free_bits"], lambda_tc=cfg["lambda_tc"],
    )
    model = FiLMMetaInjectionVAE(mc).eval()
    model.load_state_dict(ckpt["state_dict"])

    beta_isch = np.array(ckpt["residualizer"]["beta_isch"], dtype=np.float32)
    s_mean    = float(ckpt["residualizer"]["s_mean"])
    X_resid   = X_sc - np.outer(M4[:, 0] - s_mean, beta_isch)
    with torch.no_grad():
        return model.encode(torch.from_numpy(X_resid.astype(np.float32)))[0].numpy()


def build_meta_matrix(meta_df: pd.DataFrame) -> np.ndarray:
    """Same 4-dim matrix used by the trained model."""
    def _z(s):
        v = pd.to_numeric(s, errors="coerce").values.astype(float)
        z = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-8)
        return np.where(np.isfinite(z), z, 0.0).astype(np.float32)
    isch = _z(meta_df["SMTSISCH"]); dthh = _z(meta_df["DTHHRDY"])
    age  = _z(meta_df["AGE_mid"])
    sex  = pd.to_numeric(meta_df["SEX"], errors="coerce").values
    sex  = np.where(np.isfinite(sex), (sex - 1).astype(np.float32), 0.5)
    return np.column_stack([isch, dthh, age, sex]).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────
# Bidirectional analysis (Q56 logic, expanded matrix)
# ──────────────────────────────────────────────────────────────────────────

def direction_A_meta_to_z(X_meta: np.ndarray, z: np.ndarray) -> dict:
    """metadata → z_bio.  Ridge (linear) + Random Forest (non-linear)."""
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    z_sc = StandardScaler().fit_transform(z)
    x_sc = StandardScaler().fit_transform(X_meta)

    # Per-dim Ridge
    per_dim_ridge = []
    for k in range(z.shape[1]):
        sc = cross_val_score(Ridge(alpha=1.0), x_sc, z_sc[:, k], cv=kf, scoring="r2")
        per_dim_ridge.append(float(sc.mean()))

    # Joint Ridge (predict all dims, evaluate jointly)
    preds = np.zeros_like(z_sc)
    for tr, te in kf.split(x_sc):
        rcv = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(x_sc[tr], z_sc[tr])
        preds[te] = rcv.predict(x_sc[te])
    ss_res = ((z_sc - preds) ** 2).sum()
    ss_tot = ((z_sc - z_sc.mean(0)) ** 2).sum()
    r2_joint_ridge = float(1 - ss_res / ss_tot)

    # Per-dim Random Forest
    per_dim_rf = []
    for k in range(z.shape[1]):
        sc = cross_val_score(
            RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
            x_sc, z_sc[:, k], cv=kf, scoring="r2"
        )
        per_dim_rf.append(float(sc.mean()))

    return {
        "ridge_r2_joint":     r2_joint_ridge,
        "ridge_r2_per_dim":   per_dim_ridge,
        "ridge_r2_mean_dim":  float(np.mean(per_dim_ridge)),
        "ridge_r2_max_dim":   float(np.max(per_dim_ridge)),
        "rf_r2_per_dim":      per_dim_rf,
        "rf_r2_mean_dim":     float(np.mean(per_dim_rf)),
        "rf_r2_max_dim":      float(np.max(per_dim_rf)),
    }


def direction_B_z_to_meta(z: np.ndarray, X_meta: np.ndarray,
                           names: list[str]) -> dict:
    """z_bio → each metadata variable.  Ridge per variable."""
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    z_sc = StandardScaler().fit_transform(z)
    out = {}
    for i, name in enumerate(names):
        y = X_meta[:, i]
        if not np.isfinite(y).any() or y.std() < 1e-8:
            continue
        mask = np.isfinite(y)
        sc = cross_val_score(Ridge(alpha=1.0), z_sc[mask], y[mask],
                             cv=kf, scoring="r2")
        # Per-dim
        per_dim = []
        for k in range(z.shape[1]):
            s = cross_val_score(Ridge(alpha=1.0), z_sc[mask, k:k+1], y[mask],
                                cv=kf, scoring="r2")
            per_dim.append(float(s.mean()))
        out[name] = {
            "full_z_r2":    float(sc.mean()),
            "full_z_r2_std": float(sc.std()),
            "per_dim_r2":   per_dim,
            "best_dim":     int(np.nanargmax(per_dim)),
            "best_dim_r2":  float(np.nanmax(per_dim)),
        }
    return out


# ──────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────

def main():
    print("Loading GTEx blood ...")
    gtex     = load_gtex_blood(checkpoint_path=CKPT)
    meta     = load_metadata(gtex.sample_ids)
    M4       = build_meta_matrix(meta)
    gene_names = list(gtex.shared_genes)

    print(f"  n_donors={len(gtex.sample_ids)}  n_genes={len(gene_names)}")

    # ── Compute cell composition proxies ───────────────────────────────
    print("\n── Computing cell-type marker scores ──")
    cell_scores, cell_names, n_used = score_cell_types(
        gtex.expr_aligned, gene_names
    )
    for ct in cell_names:
        markers_in_set = CELL_TYPE_MARKERS[ct]
        print(f"  {ct:<12}  markers found: {n_used[ct]:>2} / {len(markers_in_set)}")

    print(f"  cell_scores shape: {cell_scores.shape}")
    print(f"  cross-correlations between cell types:")
    cc = pd.DataFrame(np.corrcoef(cell_scores.T), index=cell_names, columns=cell_names)
    print(cc.round(2).to_string())

    # ── Load z_bio ────────────────────────────────────────────────────
    print("\n── Loading Q54b model and encoding z_bio ──")
    z_bio = load_z_bio(gtex.expr_scaled, M4)
    print(f"  z_bio shape: {z_bio.shape}")

    # ── Build full metadata matrix (base + cell composition) ─────────
    base_meta, base_names = build_base_meta(meta)
    print(f"\nBase metadata variables ({len(base_names)}): {base_names}")
    print(f"Cell-type variables ({len(cell_names)}):  {cell_names}")

    full_meta = np.column_stack([base_meta, cell_scores])
    full_names = base_names + cell_names
    print(f"Combined: {full_meta.shape}")

    # ── Direction A: metadata → z_bio ─────────────────────────────────
    print("\n══ Direction A: metadata → z_bio ══")
    print("\nBaseline (9 base variables only):")
    a_base = direction_A_meta_to_z(base_meta, z_bio)
    print(f"  Ridge joint R²:    {a_base['ridge_r2_joint']:.4f}")
    print(f"  Ridge mean-dim R²: {a_base['ridge_r2_mean_dim']:.4f}")
    print(f"  Ridge max-dim R²:  {a_base['ridge_r2_max_dim']:.4f}")
    print(f"  RF max-dim R²:     {a_base['rf_r2_max_dim']:.4f}")

    print("\nWith cell composition (9 + 7 = 16 variables):")
    a_full = direction_A_meta_to_z(full_meta, z_bio)
    print(f"  Ridge joint R²:    {a_full['ridge_r2_joint']:.4f}  "
          f"(Δ = +{a_full['ridge_r2_joint'] - a_base['ridge_r2_joint']:.4f})")
    print(f"  Ridge mean-dim R²: {a_full['ridge_r2_mean_dim']:.4f}  "
          f"(Δ = +{a_full['ridge_r2_mean_dim'] - a_base['ridge_r2_mean_dim']:.4f})")
    print(f"  Ridge max-dim R²:  {a_full['ridge_r2_max_dim']:.4f}  "
          f"(Δ = +{a_full['ridge_r2_max_dim'] - a_base['ridge_r2_max_dim']:.4f})")
    print(f"  RF max-dim R²:     {a_full['rf_r2_max_dim']:.4f}  "
          f"(Δ = +{a_full['rf_r2_max_dim'] - a_base['rf_r2_max_dim']:.4f})")
    print(f"\n  Per-dim Ridge R² (cell-comp included):")
    for k, r in enumerate(a_full["ridge_r2_per_dim"]):
        marker = "★" if r > 0.3 else " "
        print(f"    {marker} z{k+1:<2}: Ridge={r:>.3f}  RF={a_full['rf_r2_per_dim'][k]:.3f}")

    # ── Direction B: z_bio → individual cell types ────────────────────
    print("\n══ Direction B: z_bio → each metadata variable ══")
    b_full = direction_B_z_to_meta(z_bio, full_meta, full_names)
    print(f"\n{'variable':<18}  {'full-z R²':>10}  {'best dim':>10}  {'best dim R²':>12}")
    print("─" * 55)
    rows = sorted(b_full.items(), key=lambda kv: kv[1]["full_z_r2"], reverse=True)
    for name, r in rows:
        flag = "★ NEW" if name in cell_names else "     "
        print(f"{flag} {name:<13}  {r['full_z_r2']:>10.3f}  z{r['best_dim']+1:>9}  "
              f"{r['best_dim_r2']:>12.3f}")

    # ── Save ─────────────────────────────────────────────────────────
    def _clean(obj):
        if isinstance(obj, dict): return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)): return float(obj)
        if isinstance(obj, list): return [_clean(v) for v in obj]
        return obj

    results = {
        "direction_A_base_meta":   _clean(a_base),
        "direction_A_with_cells":  _clean(a_full),
        "direction_B_with_cells":  _clean(b_full),
        "cell_type_markers_used":  n_used,
        "delta_joint_r2":          float(a_full["ridge_r2_joint"] - a_base["ridge_r2_joint"]),
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))

    # latent atlas: full_meta × z_dim
    atlas = pd.DataFrame(
        np.array([b_full[n]["per_dim_r2"] for n in full_names]),
        index=full_names,
        columns=[f"z{k+1}" for k in range(z_bio.shape[1])],
    )
    atlas.to_csv(OUT / "latent_atlas_full.csv")
    print(f"\nSaved → {OUT}/")
    print(f"  results.json + latent_atlas_full.csv (16 × {z_bio.shape[1]} heatmap)")

    print("\n" + "═" * 60)
    print("SUMMARY")
    print("═" * 60)
    print(f"Joint R² (metadata → z_bio):")
    print(f"  9 base variables:                {a_base['ridge_r2_joint']:.3f}")
    print(f"  16 (base + cell composition):    {a_full['ridge_r2_joint']:.3f}")
    print(f"  Improvement:                     +{a_full['ridge_r2_joint'] - a_base['ridge_r2_joint']:.3f}")


if __name__ == "__main__":
    main()
