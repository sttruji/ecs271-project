"""GTEx whole-blood loader + metadata join.

Provides:
  - GTExBlood dataclass: (expr_log, gene_names, sample_ids, expr_scaled)
  - load_metadata(sample_ids) -> pd.DataFrame keyed by SAMPID with columns
       SUBJID, AGE_mid, SEX, DTHHRDY, SMRIN, SMTSISCH, SMCENTER, SMNABTCH, SMGEBTCH, SMRDLGTH
  - load_shared_genes(checkpoint_path) -> (np.ndarray, mean, std) used by the
       trained cross-modality VAE so a new model can be evaluated in the same
       gene space.
"""
from __future__ import annotations

import gzip
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

SHARED_DATA = Path("/Users/rls/ecs271/data")
sys.path.insert(0, str(SHARED_DATA))
import paths as shared_paths  # noqa: E402

CPM_THRESHOLD = 1.0
MIN_SAMPLE_FRAC = 0.10


# ── data classes ──────────────────────────────────────────────────────────
@dataclass
class GTExBlood:
    expr_log: np.ndarray         # (n_donors, n_genes_filtered)  log2(CPM+1)
    gene_names: np.ndarray       # (n_genes_filtered,)  gene symbols
    sample_ids: np.ndarray       # (n_donors,)         GTEX-XXXX-XXXX-...
    expr_aligned: np.ndarray     # (n_donors, n_shared_genes)  reordered to shared_genes
    shared_genes: np.ndarray     # (n_shared_genes,)
    expr_scaled: np.ndarray      # (n_donors, n_shared_genes)  standardised by checkpoint scaler

    @property
    def n_donors(self) -> int:
        return self.expr_log.shape[0]

    @property
    def n_shared_genes(self) -> int:
        return self.shared_genes.size


# ── loaders ──────────────────────────────────────────────────────────────
def _cpm_log(expr_raw: np.ndarray) -> np.ndarray:
    lib = expr_raw.sum(axis=0, keepdims=True)
    return np.log2(expr_raw / lib * 1e6 + 1)


def _filter_genes(expr_raw: np.ndarray, gene_names: np.ndarray):
    n_samples = expr_raw.shape[1]
    lib = expr_raw.sum(axis=0)
    cpm = expr_raw / lib * 1e6
    min_s = max(1, int(MIN_SAMPLE_FRAC * n_samples))
    keep = (cpm > CPM_THRESHOLD).sum(axis=1) >= min_s
    return expr_raw[keep], gene_names[keep]


def _read_gct_sample_ids(gct_path: Path) -> np.ndarray:
    with gzip.open(gct_path, "rt") as fh:
        for _ in range(2):
            fh.readline()
        header = fh.readline().rstrip("\n").split("\t")
    return np.asarray(header[2:])


def _align(expr: np.ndarray, gene_names: np.ndarray, target_genes: np.ndarray):
    name_to_idx: dict[str, int] = {}
    for i, n in enumerate(gene_names):
        k = str(n).upper()
        if k not in name_to_idx:
            name_to_idx[k] = i
    out = np.full((expr.shape[0], len(target_genes)), np.nan, dtype=np.float32)
    for j, g in enumerate(target_genes):
        idx = name_to_idx.get(str(g).upper())
        if idx is None:
            continue
        out[:, j] = expr[:, idx]
    # mean-impute missing columns
    if np.isnan(out).any():
        col_mean = np.nanmean(out, axis=0)
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
        for c in np.where(~np.isfinite(out).all(axis=0))[0]:
            out[:, c] = col_mean[c]
    return out


def load_shared_genes(checkpoint_path: Optional[str] = None):
    """Load the (shared_genes, scaler_mean, scaler_std) from a saved checkpoint.

    Defaults to the cross-modality VAE checkpoint. A model trained in a
    different gene space should provide its own arrays.
    """
    import torch

    if checkpoint_path is None:
        checkpoint_path = str(shared_paths.CROSS_MODALITY_VAE)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return (
        np.asarray(ckpt["shared_genes"]),
        np.asarray(ckpt["scaler_mean"], dtype=np.float32),
        np.asarray(ckpt["scaler_std"], dtype=np.float32),
    )


def load_gtex_blood(
    shared_genes: Optional[np.ndarray] = None,
    scaler_mean: Optional[np.ndarray] = None,
    scaler_std: Optional[np.ndarray] = None,
    checkpoint_path: Optional[str] = None,
) -> GTExBlood:
    """Load + filter + align + scale GTEx whole blood.

    If shared_genes / scaler are not given, they are read from the checkpoint
    at `checkpoint_path` (or the default cross-modality VAE checkpoint).
    """
    if shared_genes is None or scaler_mean is None or scaler_std is None:
        shared_genes, scaler_mean, scaler_std = load_shared_genes(checkpoint_path)

    df = pd.read_csv(shared_paths.GTEX_WHOLE_BLOOD, sep="\t", skiprows=2, compression="gzip")
    expr_raw = df.iloc[:, 2:].values.astype(np.float64)  # (genes, samples)
    gene_names = df["Description"].values.astype(str)
    expr_filt, names_filt = _filter_genes(expr_raw, gene_names)
    expr_log = _cpm_log(expr_filt).T.astype(np.float32)  # (samples, genes_filt)

    expr_aligned = _align(expr_log, names_filt, shared_genes)
    expr_scaled = ((expr_aligned - scaler_mean) / scaler_std).astype(np.float32)
    sample_ids = _read_gct_sample_ids(shared_paths.GTEX_WHOLE_BLOOD)

    return GTExBlood(
        expr_log=expr_log,
        gene_names=names_filt,
        sample_ids=sample_ids,
        expr_aligned=expr_aligned,
        shared_genes=shared_genes,
        expr_scaled=expr_scaled,
    )


# ── metadata ─────────────────────────────────────────────────────────────
ANNOT = SHARED_DATA / "annotations"


def _age_to_midpoint(age_band) -> float:
    if not isinstance(age_band, str) or "-" not in age_band:
        return float("nan")
    a, b = age_band.split("-")
    try:
        return (int(a) + int(b)) / 2
    except ValueError:
        return float("nan")


def load_metadata(sample_ids: np.ndarray) -> pd.DataFrame:
    """Return a DataFrame indexed by SAMPID (in the same order as sample_ids)
    with the standard set of GTEx covariates joined in."""
    sub = pd.read_csv(ANNOT / "GTEx_v10_Annotations_SubjectPhenotypesDS.txt", sep="\t")
    samp = pd.read_csv(
        ANNOT / "GTEx_v10_Annotations_SampleAttributesDS.txt",
        sep="\t",
        low_memory=False,
    )
    df = pd.DataFrame({"SAMPID": sample_ids})
    df["SUBJID"] = df["SAMPID"].apply(lambda s: "-".join(s.split("-")[:2]))
    df = df.merge(sub, on="SUBJID", how="left")
    df = df.merge(
        samp[["SAMPID", "SMRIN", "SMTSISCH", "SMCENTER", "SMNABTCH", "SMGEBTCH", "SMRDLGTH"]],
        on="SAMPID",
        how="left",
    )
    df["AGE_mid"] = df["AGE"].apply(_age_to_midpoint)
    return df.set_index("SAMPID").loc[list(sample_ids)].reset_index()
