"""Q61 — Does adding NMF-12 interpretable basis improve PULSAR-style lupus classification?

Setup:
  - Data: lupus_subsampled_uce_adata.h5ad (315k PBMC cells, 261 donors, binary label)
  - Foundation model representation: mean-pooled UCE embeddings per donor (1280-d)
    (UCE = Universal Cell Embeddings, the frozen backbone that feeds into PULSAR)
  - Augmentation: NMF-12 coordinates computed from pseudobulk cell-type marker scores
    (same 24 sub-cell-type markers as Q59/Q60, projected through NMF from Q60)

Question:
  Does adding NMF-12 features to UCE donor embeddings improve lupus classification?

Three-way comparison:
  A. UCE mean-pool only             (1280-d)
  B. NMF-12 features only           (12-d)
  C. UCE mean-pool + NMF-12         (1292-d)
  D. UCE mean-pool + sub-cell 24    (1304-d, un-compressed version of NMF-12)

Probe: LogisticRegression + Ridge (5-fold stratified CV, accuracy + macro-F1)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
H5AD = Path("/tmp/lupus_demo.h5ad")
OUT  = ROOT / "analysis" / "results" / "q61_pulsar_nmf"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42

# ──────────────────────────────────────────────────────────────────────────
# Sub-cell-type markers (same as Q59 — we redefine to keep this self-contained)
# ──────────────────────────────────────────────────────────────────────────
SUBCELL_MARKERS: dict[str, list[str]] = {
    "CD4_naive":      ["CD4", "CCR7", "SELL", "LEF1", "TCF7", "BACH2"],
    "CD4_memory":     ["CD4", "S100A4", "IL7R", "AQP3", "KLRB1", "ITGB1"],
    "CD8_naive":      ["CD8A", "CD8B", "CCR7", "SELL", "LEF1", "TCF7"],
    "CD8_cytotoxic":  ["CD8A", "CD8B", "GZMA", "GZMB", "GZMK", "PRF1", "NKG7"],
    "Treg":           ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TIGIT"],
    "B_naive":        ["CD19", "MS4A1", "IGHD", "TCL1A"],
    "B_memory":       ["CD19", "MS4A1", "CD27", "AIM2"],
    "Plasmablast":    ["XBP1", "MZB1", "JCHAIN", "PRDM1"],
    "Mono_classical": ["CD14", "S100A12", "VCAN", "FCN1", "S100A8", "S100A9"],
    "Mono_nonclass":  ["FCGR3A", "MTSS1", "LST1", "CDKN1C", "MS4A7"],
    "NK_bright":      ["NCAM1", "GZMK", "XCL1", "XCL2", "IL7R"],
    "NK_dim":         ["FCGR3A", "GZMB", "KLRD1", "NKG7", "PRF1"],
    "Neut_mature":    ["FCGR3B", "CXCR2", "FPR1", "S100A12"],
    "Neut_immature":  ["MPO", "ELANE", "DEFA1", "PRTN3", "AZU1"],
    "Eosinophil":     ["CCR3", "IL5RA", "EPX", "PRG2"],
    "Basophil":       ["HDC", "ENPP3", "MS4A2", "CPA3"],
    "cDC1":           ["CLEC9A", "BATF3", "IRF8", "XCR1"],
    "cDC2":           ["CD1C", "CLEC10A", "FCER1A"],
    "pDC":            ["LILRA4", "IL3RA", "CLEC4C", "IRF7"],
    "Activation":     ["CD69", "NR4A1", "EGR1", "FOS", "JUN"],
    "Exhaustion":     ["PDCD1", "LAG3", "TIGIT", "HAVCR2", "TOX"],
    "Proliferation":  ["MKI67", "TOP2A", "PCNA", "CCNA2"],
    "Type1_IFN":      ["ISG15", "IFI6", "IFI27", "OAS1", "MX1", "IFIT1"],
    "Inflammation":   ["IL1B", "IL6", "TNF", "CXCL8", "CCL2", "PTGS2"],
}


def load_h5ad_data(h5ad_path: Path):
    """Load obs metadata, UCE embeddings, and raw expression from H5AD."""
    print(f"Loading {h5ad_path} ...")
    f = h5py.File(h5ad_path, "r")

    # UCE embeddings (cells × 1280)
    X_uce = f["obsm"]["X_uce"][:]
    print(f"  UCE: {X_uce.shape}")

    # Obs metadata: donor_id, disease
    def _decode_cat(g):
        cats = np.array(g["categories"][:])
        codes = np.array(g["codes"][:])
        return np.array([cats[c].decode() if codes[c] >= 0 else "" for c in codes])

    donor_ids = _decode_cat(f["obs"]["donor_id"])
    disease   = _decode_cat(f["obs"]["disease"])

    # Gene names — stored as gene symbols in feature_name (Ensembl IDs in _index)
    fn = f["var"]["feature_name"]
    if hasattr(fn, "keys") and "categories" in fn:
        cats  = [g.decode() for g in fn["categories"][:]]
        codes = np.array(fn["codes"][:])
        gene_names = [cats[c] if c >= 0 else "" for c in codes]
    else:
        gene_names = [g.decode() for g in fn[:]]

    # Raw counts matrix — use raw/X (X is z-scored in this H5AD)
    from scipy.sparse import csr_matrix
    n_cells = len(donor_ids)
    if "raw" in f and "X" in f["raw"]:
        raw_X = f["raw"]["X"]
        # Gene names for raw (may differ from main var)
        raw_var = f["raw"]["var"]
        if "_index" in raw_var:
            raw_ensg = [g.decode() for g in raw_var["_index"][:]]
        else:
            raw_ensg = gene_names
        # Map raw genes to gene symbols via main var feature_name matching on Ensembl
        # raw var has same Ensembl IDs, use main gene_names (already decoded as symbols)
        data    = raw_X["data"][:]
        indices = raw_X["indices"][:]
        indptr  = raw_X["indptr"][:]
        n_raw_genes = len(raw_ensg)
        X_raw = csr_matrix((data.astype(np.float32), indices, indptr),
                           shape=(n_cells, n_raw_genes))
        # Use raw gene names mapped from main var feature_name
        # raw var has same genes as main var (checked by length)
        if n_raw_genes == len(gene_names):
            pass  # gene_names already loaded with symbols
        print(f"  X_raw from raw/X (sparse): {X_raw.shape}")
    elif "data" in f["X"]:
        data    = f["X"]["data"][:]
        indices = f["X"]["indices"][:]
        indptr  = f["X"]["indptr"][:]
        X_raw = csr_matrix((data.astype(np.float32), indices, indptr),
                           shape=(n_cells, len(gene_names)))
        print(f"  X_raw (sparse): {X_raw.shape}")
    else:
        X_raw = None
        print("  X_raw: not found, skipping")

    f.close()
    return X_uce, donor_ids, disease, gene_names, X_raw


def compute_pseudobulk_marker_scores(X_raw, donor_ids, gene_names):
    """
    For each donor: sum raw counts per gene, log-normalise to 1e4 CPM,
    then compute mean log-CPM for each marker gene set.
    """
    print("Computing pseudobulk marker scores ...")
    upper_to_idx = {g.upper(): i for i, g in enumerate(gene_names)}
    sub_names = list(SUBCELL_MARKERS.keys())

    unique_donors = sorted(set(donor_ids))
    n_donors = len(unique_donors)
    donor_to_row = {d: i for i, d in enumerate(unique_donors)}

    scores = np.zeros((n_donors, len(sub_names)), dtype=np.float32)

    for donor in unique_donors:
        mask = donor_ids == donor
        row  = donor_to_row[donor]
        # pseudobulk: sum across cells
        counts = np.asarray(X_raw[mask].sum(axis=0)).flatten()  # (n_genes,)
        total  = counts.sum()
        if total == 0:
            continue
        # log-CPM normalise (ensure non-negative counts first)
        counts = np.maximum(counts, 0)
        lc = np.log1p(counts / (total + 1) * 1e4)
        for j, sname in enumerate(sub_names):
            idx = [upper_to_idx[m.upper()] for m in SUBCELL_MARKERS[sname]
                   if m.upper() in upper_to_idx]
            if idx:
                scores[row, j] = lc[idx].mean()

    # z-score across donors; replace NaN (zero-variance columns) with 0
    std = scores.std(0)
    scores = (scores - scores.mean(0)) / np.where(std > 1e-8, std, 1.0)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores, sub_names, [sorted(set(donor_ids))[i] for i in range(n_donors)]


def pool_uce_per_donor(X_uce, donor_ids):
    """Mean-pool 1280-d UCE embeddings per donor."""
    print("Pooling UCE embeddings per donor ...")
    unique_donors = sorted(set(donor_ids))
    donor_to_row = {d: i for i, d in enumerate(unique_donors)}
    n_donors = len(unique_donors)
    emb = np.zeros((n_donors, X_uce.shape[1]), dtype=np.float32)
    counts = np.zeros(n_donors, dtype=int)
    for i, d in enumerate(donor_ids):
        row = donor_to_row[d]
        emb[row] += X_uce[i]
        counts[row] += 1
    emb /= counts[:, None].clip(1)
    return emb, unique_donors


def get_donor_labels(donor_list, donor_ids, disease):
    """Binary label: 1=lupus, 0=normal for each donor in donor_list."""
    donor_to_disease = {}
    for d, dis in zip(donor_ids, disease):
        donor_to_disease[d] = dis
    labels = np.array([1 if "lupus" in donor_to_disease[d].lower() else 0
                        for d in donor_list])
    return labels


def fit_nmf_on_scores(scores, k=12):
    """Fit NMF on the sub-cell-type scores and return coordinates + model."""
    shifted = scores - scores.min(0)
    nmf = NMF(n_components=k, random_state=SEED, max_iter=1000)
    H = nmf.fit_transform(shifted)
    return H, nmf


def cv_classify(X, y, label, n_splits=5):
    """5-fold stratified CV, LogisticRegression + Ridge."""
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    X_sc = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED, C=0.1)
    cv_res = cross_validate(clf, X_sc, y, cv=kf,
                            scoring=["accuracy", "f1_macro"], return_train_score=False)
    acc  = float(cv_res["test_accuracy"].mean())
    f1   = float(cv_res["test_f1_macro"].mean())
    sacc = float(cv_res["test_accuracy"].std())
    sf1  = float(cv_res["test_f1_macro"].std())
    print(f"  {label:<35}  dim={X.shape[1]:>5}  acc={acc:.3f}±{sacc:.3f}  f1={f1:.3f}±{sf1:.3f}")
    return {"label": label, "dim": X.shape[1], "acc": acc, "f1": f1, "acc_std": sacc, "f1_std": sf1}


def make_plot(results):
    labels = [r["label"] for r in results]
    accs   = [r["acc"] for r in results]
    f1s    = [r["f1"] for r in results]
    accs_e = [r["acc_std"] for r in results]
    f1s_e  = [r["f1_std"] for r in results]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, accs, w, yerr=accs_e, capsize=4, label="Accuracy", color="steelblue", alpha=0.85)
    ax.bar(x + w/2, f1s,  w, yerr=f1s_e,  capsize=4, label="Macro-F1", color="orange",   alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("5-fold CV score")
    ax.set_title("PULSAR (UCE) + NMF-12 augmentation — Lupus classification")
    ax.legend()
    ax.set_ylim(0.3, 1.0)
    plt.tight_layout()
    plt.savefig(OUT / "pulsar_nmf_lupus.png", dpi=150)
    plt.close()
    print(f"\nPlot saved: {OUT}/pulsar_nmf_lupus.png")


def main():
    X_uce, donor_ids, disease, gene_names, X_raw = load_h5ad_data(H5AD)

    # Pool UCE per donor
    uce_emb, donor_list = pool_uce_per_donor(X_uce, donor_ids)
    del X_uce  # free memory

    # Labels
    y = get_donor_labels(donor_list, donor_ids, disease)
    print(f"\nDonors: {len(donor_list)}  |  lupus: {y.sum()}  normal: {(y==0).sum()}")

    # Cell-type marker scores
    if X_raw is not None:
        sub_scores, sub_names, score_donors = compute_pseudobulk_marker_scores(
            X_raw, donor_ids, gene_names)
        del X_raw  # free memory
        # Align to donor_list order
        score_donor_to_row = {d: i for i, d in enumerate(score_donors)}
        sub_scores_aligned = np.array([sub_scores[score_donor_to_row[d]] for d in donor_list])
    else:
        print("WARNING: no expression matrix found, skipping NMF features")
        sub_scores_aligned = None

    # NMF-12 on sub-cell-type scores
    if sub_scores_aligned is not None:
        nmf12_coords, nmf_model = fit_nmf_on_scores(sub_scores_aligned, k=12)
        print(f"NMF-12 coords: {nmf12_coords.shape}")

    # ── Comparisons ──
    print("\n" + "═"*70)
    print("LUPUS CLASSIFICATION — 5-fold CV (n=261 donors)")
    print("═"*70)

    results = []
    results.append(cv_classify(uce_emb,                                          y, "A. UCE mean-pool (1280d)"))
    if sub_scores_aligned is not None:
        results.append(cv_classify(nmf12_coords,                                 y, "B. NMF-12 only (12d)"))
        results.append(cv_classify(sub_scores_aligned,                           y, "C. Sub-cell-24 only (24d)"))
        results.append(cv_classify(np.hstack([uce_emb, nmf12_coords]),           y, "D. UCE + NMF-12 (1292d)"))
        results.append(cv_classify(np.hstack([uce_emb, sub_scores_aligned]),     y, "E. UCE + sub-cell-24 (1304d)"))

    # Delta
    if len(results) >= 4:
        base_acc = results[0]["acc"]
        aug_acc  = results[3]["acc"]
        base_f1  = results[0]["f1"]
        aug_f1   = results[3]["f1"]
        print(f"\n  Δ accuracy (UCE+NMF-12 vs UCE alone): {aug_acc - base_acc:+.3f}")
        print(f"  Δ macro-F1 (UCE+NMF-12 vs UCE alone): {aug_f1  - base_f1:+.3f}")

    make_plot(results)

    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nAll results → {OUT}/")


if __name__ == "__main__":
    main()
