"""Bulk RNA-seq loading + alignment to a target gene set.

Used by every Q1/Q2/Q3 analysis so we have one canonical preprocessing path:

  GTEx GCT  →  CPM filter (>1 in >=10% donors)  →  log2(CPM+1)
            →  align genes to the trained-VAE shared_genes list
            →  apply trained-VAE feature-wise scaler (mean / std)
            →  (samples, n_genes) float32 ready for the encoder
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data"))
import paths as shared_paths  # noqa: E402

CPM_THRESHOLD = 1.0
MIN_SAMPLE_FRAC = 0.1


def _cpm_log(expr_raw: np.ndarray) -> np.ndarray:
    lib = expr_raw.sum(axis=0, keepdims=True)
    cpm = expr_raw / lib * 1e6
    return np.log2(cpm + 1)


def _filter_genes(expr_raw: np.ndarray, gene_names: np.ndarray):
    n_samples = expr_raw.shape[1]
    lib = expr_raw.sum(axis=0)
    cpm = expr_raw / lib * 1e6
    min_s = max(1, int(MIN_SAMPLE_FRAC * n_samples))
    keep = (cpm > CPM_THRESHOLD).sum(axis=1) >= min_s
    return expr_raw[keep], gene_names[keep]


def load_gtex_blood():
    """(samples, genes_after_filter) log2(CPM+1), gene-symbol array."""
    df = pd.read_csv(shared_paths.GTEX_WHOLE_BLOOD, sep="\t", skiprows=2, compression="gzip")
    expr_raw = df.iloc[:, 2:].values.astype(np.float64)  # (genes, samples)
    gene_names = df["Description"].values.astype(str)
    expr_filt, names_filt = _filter_genes(expr_raw, gene_names)
    expr_log = _cpm_log(expr_filt)  # (genes, samples)
    return expr_log.T.astype(np.float32), names_filt


def align_to_shared(expr: np.ndarray, gene_names: np.ndarray, shared_genes: np.ndarray) -> np.ndarray:
    """Pick / reorder columns of `expr` so they match `shared_genes` exactly.

    shared_genes is the (n_shared,) list saved in the trained checkpoint;
    the model expects features in that order.
    """
    name_to_idx: dict[str, int] = {}
    for i, n in enumerate(gene_names):
        key = n.upper()
        if key not in name_to_idx:
            name_to_idx[key] = i

    out = np.full((expr.shape[0], len(shared_genes)), np.nan, dtype=np.float32)
    found = 0
    missing: list[str] = []
    for j, g in enumerate(shared_genes):
        idx = name_to_idx.get(str(g).upper())
        if idx is None:
            missing.append(str(g))
            continue
        out[:, j] = expr[:, idx]
        found += 1
    if missing:
        # Fill genes that aren't in GTEx with the column mean (=0 after scaling)
        col_mean = np.nanmean(out, axis=0)
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
        nan_cols = np.where(~np.isfinite(out).all(axis=0))[0]
        for c in nan_cols:
            out[:, c] = col_mean[c]
    return out, found, missing


def standardise(expr: np.ndarray, scaler_mean: np.ndarray, scaler_std: np.ndarray) -> np.ndarray:
    return ((expr - scaler_mean.astype(np.float32)) / scaler_std.astype(np.float32)).astype(np.float32)
