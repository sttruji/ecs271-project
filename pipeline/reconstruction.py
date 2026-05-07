"""Reconstruction-quality evaluation for a LatentModel."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class ReconstructionResult:
    name: str
    mse: float
    mae: float
    r2_overall: float
    per_sample_r_mean: float
    per_sample_r_med: float
    per_sample_r_min: float
    per_gene_r_mean: float
    per_gene_r_med: float
    per_gene_r_min: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _metric_block(true: np.ndarray, pred: np.ndarray, name: str) -> ReconstructionResult:
    diff = true - pred
    var_total = float(true.var())
    var_resid = float(diff.var())
    r2 = float(1.0 - var_resid / var_total) if var_total > 0 else float("nan")
    psr, pgr = _pearson_per_axis(true, pred)
    return ReconstructionResult(
        name=name,
        mse=float((diff ** 2).mean()),
        mae=float(np.abs(diff).mean()),
        r2_overall=r2,
        per_sample_r_mean=float(psr.mean()),
        per_sample_r_med=float(np.median(psr)),
        per_sample_r_min=float(psr.min()),
        per_gene_r_mean=float(pgr.mean()),
        per_gene_r_med=float(np.median(pgr)),
        per_gene_r_min=float(pgr.min()),
    )


def _pearson_per_axis(true: np.ndarray, pred: np.ndarray):
    psr = []
    for i in range(true.shape[0]):
        a = true[i] - true[i].mean()
        b = pred[i] - pred[i].mean()
        d = float(np.sqrt((a * a).sum() * (b * b).sum()))
        psr.append(float((a * b).sum() / d) if d else 0.0)
    pgr = []
    for j in range(true.shape[1]):
        a = true[:, j] - true[:, j].mean()
        b = pred[:, j] - pred[:, j].mean()
        d = float(np.sqrt((a * a).sum() * (b * b).sum()))
        pgr.append(float((a * b).sum() / d) if d else 0.0)
    return np.asarray(psr), np.asarray(pgr)


def evaluate_reconstruction(
    model,
    expr_train: np.ndarray,
    expr_test: np.ndarray,
    *,
    pca_ranks: tuple[int, ...] = (16, 50, 200),
    include_mean_baseline: bool = True,
) -> dict:
    """Evaluate `model` against PCA baselines + the mean baseline.

    `model` must implement encode(x) and decode(z) returning numpy arrays.
    Returns {name: ReconstructionResult.to_dict()}, including:
      - "<model.name>"
      - "pca_<k>" for each k in pca_ranks where k < n_train
      - "mean_baseline" if include_mean_baseline
    """
    out: dict[str, dict] = {}
    z_test = model.encode(expr_test)
    pred = model.decode(z_test)
    out[model.name] = _metric_block(expr_test, pred, model.name).to_dict()

    for k in pca_ranks:
        if k >= expr_train.shape[0]:
            continue
        pca = PCA(n_components=k).fit(expr_train)
        recon_k = pca.inverse_transform(pca.transform(expr_test))
        out[f"pca_{k}"] = _metric_block(expr_test, recon_k, f"pca_{k}").to_dict()

    if include_mean_baseline:
        zero = np.zeros_like(expr_test)
        out["mean_baseline"] = _metric_block(expr_test, zero, "mean_baseline").to_dict()

    return out
