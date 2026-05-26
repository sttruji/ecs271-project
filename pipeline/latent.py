"""Latent-space probes — activity, biology correlation, metadata correlation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler


def evaluate_latent_activity(z: np.ndarray, var_threshold: float = 0.01) -> dict:
    """Posterior-collapse / activity diagnostic.

    Returns:
        n_active_dims: how many dims have variance > threshold
        var_per_dim_top5: the 5 highest-variance latent dims
        var_total: total variance summed across dims
    """
    var_per_dim = z.var(0)
    return {
        "latent_dim": int(z.shape[1]),
        "n_active_dims": int((var_per_dim > var_threshold).sum()),
        "n_near_collapsed_dims": int((var_per_dim < 1e-3).sum()),
        "var_per_dim_top5": [float(v) for v in np.sort(var_per_dim)[::-1][:5]],
        "var_per_dim_mean": float(var_per_dim.mean()),
        "var_per_dim_total": float(var_per_dim.sum()),
    }


def correlate_z_with_pcs(z: np.ndarray, expr: np.ndarray, n_pcs: int = 10) -> np.ndarray:
    """|corr(z_dim_k, bulk_PC_j)| for k in [0, latent_dim) × j in [0, n_pcs)."""
    bulk_pcs = PCA(n_components=n_pcs).fit_transform(expr)
    out = np.zeros((z.shape[1], n_pcs))
    for k in range(z.shape[1]):
        for j in range(n_pcs):
            a = z[:, k] - z[:, k].mean()
            b = bulk_pcs[:, j] - bulk_pcs[:, j].mean()
            d = float(np.sqrt((a * a).sum() * (b * b).sum()))
            out[k, j] = abs((a * b).sum() / d) if d else 0.0
    return out


def evaluate_latent_meta(
    z: np.ndarray,
    metadata: pd.DataFrame,
    *,
    continuous_cols: tuple[str, ...] = ("AGE_mid", "DTHHRDY", "SMRIN", "SMTSISCH", "SMRDLGTH"),
    categorical_cols: tuple[str, ...] = ("SEX", "SMCENTER", "SMNABTCH", "SMGEBTCH"),
    max_groups_for_anova: int = 30,
) -> dict[str, pd.DataFrame]:
    """Spearman ρ (continuous) and η² (categorical) of metadata against latent dims.

    Returns:
        {"spearman": DataFrame[continuous × latent_dim_k],
         "eta2":     DataFrame[categorical × latent_dim_k]}
    """
    cols = [f"z{k+1}" for k in range(z.shape[1])]
    rho = pd.DataFrame(index=continuous_cols, columns=cols, dtype=float)
    for col in continuous_cols:
        v = pd.to_numeric(metadata[col], errors="coerce").values
        for k, c in enumerate(cols):
            x = z[:, k]
            mask = np.isfinite(v) & np.isfinite(x)
            if mask.sum() < 30:
                rho.loc[col, c] = float("nan")
                continue
            r, _ = spearmanr(v[mask], x[mask])
            rho.loc[col, c] = r

    eta = pd.DataFrame(index=categorical_cols, columns=cols, dtype=float)
    for col in categorical_cols:
        groups = metadata[col].fillna("NA").astype(str).values
        if len(set(groups)) > max_groups_for_anova:
            top = pd.Series(groups).value_counts().head(max_groups_for_anova).index.tolist()
            mask_keep = np.isin(groups, top)
            groups = groups[mask_keep]
        else:
            mask_keep = np.ones_like(groups, dtype=bool)
        for k, c in enumerate(cols):
            x = z[mask_keep, k]
            buckets = [x[groups == g] for g in sorted(set(groups)) if (groups == g).sum() > 1]
            if len(buckets) < 2:
                eta.loc[col, c] = float("nan")
                continue
            try:
                f_stat, _ = f_oneway(*buckets)
                k_g, n_obs = len(buckets), sum(b.size for b in buckets)
                eta.loc[col, c] = (f_stat * (k_g - 1)) / (f_stat * (k_g - 1) + (n_obs - k_g))
            except Exception:
                eta.loc[col, c] = float("nan")
    return {"spearman": rho, "eta2": eta}


def state_classification_auc(
    z: np.ndarray,
    expr_log: np.ndarray,
    gene_names: np.ndarray,
    anchor_genes: tuple[str, ...] = ("DDIT4", "FRAT1"),
    *,
    cv_splits: int = 5,
    seed: int = 0,
) -> dict:
    """Linear-probe AUC for predicting State-A / State-B (handling-stress) labels
    derived from anchor-gene median splits — ALL anchors above median ⇒ State-B."""
    upper = np.array([str(g).upper() for g in gene_names])
    masks = []
    for g in anchor_genes:
        idx = np.where(upper == g.upper())[0]
        if not len(idx):
            continue
        v = expr_log[:, idx[0]]
        masks.append(v > np.median(v))
    if not masks:
        return {"auc": float("nan"), "note": "no anchor genes found"}
    label = masks[0].copy()
    for m in masks[1:]:
        label = label & m
    label = label.astype(int)

    if z.var() < 1e-12:
        return {"auc": 0.5, "note": "z is collapsed; auc set to chance",
                "state_b_fraction": float(label.mean())}
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    aucs = cross_val_score(clf, z, label, cv=cv, scoring="roc_auc")
    return {
        "auc": float(aucs.mean()),
        "auc_std": float(aucs.std()),
        "state_b_fraction": float(label.mean()),
        "anchors": list(anchor_genes),
    }


def metadata_linear_probe(
    z: np.ndarray,
    metadata: pd.DataFrame,
    *,
    cv_splits: int = 5,
    seed: int = 0,
) -> dict:
    """Cross-validated linear probes for two dominant GTEx metadata factors.

    SMTSISCH → Ridge regression, CV R² (continuous ischemia time in minutes)
    DTHHRDY  → LogisticRegression OvR, CV balanced-accuracy (5-class Hardy scale)

    Both use StandardScaler on z before fitting so the probe is scale-invariant.
    A collapsed z (all variance < 1e-12) returns chance-level results immediately.

    Returns:
        {"SMTSISCH": {"cv_r2": float, "cv_r2_std": float, "n": int},
         "DTHHRDY":  {"cv_balanced_acc": float, "cv_balanced_acc_std": float,
                      "n": int, "n_classes": int, "chance": float}}
    """
    results: dict = {}

    collapsed_reg = {"cv_r2": 0.0, "cv_r2_std": 0.0, "n": 0, "note": "z collapsed"}
    if float(z.var()) < 1e-12:
        return {
            "SMTSISCH": collapsed_reg,
            "AGE_mid":  collapsed_reg,
            "DTHHRDY":  {"cv_balanced_acc": 0.2, "cv_balanced_acc_std": 0.0,
                         "n": 0, "n_classes": 5, "chance": 0.2, "note": "z collapsed"},
        }

    def _ridge_probe(col: str) -> dict:
        v = pd.to_numeric(metadata[col], errors="coerce").values
        mask = np.isfinite(v)
        if mask.sum() < cv_splits * 2:
            return {"cv_r2": float("nan"), "cv_r2_std": float("nan"),
                    "n": int(mask.sum()), "note": "too few samples"}
        z_s = StandardScaler().fit_transform(z[mask])
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
        scores = cross_val_score(Ridge(alpha=1.0), z_s, v[mask], cv=kf, scoring="r2")
        return {"cv_r2": float(scores.mean()), "cv_r2_std": float(scores.std()),
                "n": int(mask.sum())}

    # ── continuous regressions ────────────────────────────────────────────
    for col in ("SMTSISCH", "AGE_mid"):
        results[col] = _ridge_probe(col)

    # ── DTHHRDY (classification) ─────────────────────────────────────────
    hardy = pd.to_numeric(metadata["DTHHRDY"], errors="coerce").values
    mask2 = np.isfinite(hardy)
    y_h = hardy[mask2].astype(int)
    classes = np.unique(y_h)
    if len(classes) >= 2 and mask2.sum() >= cv_splits * len(classes):
        z_h = StandardScaler().fit_transform(z[mask2])
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
        scores2 = cross_val_score(
            LogisticRegression(max_iter=2000, C=1.0),
            z_h, y_h, cv=skf, scoring="balanced_accuracy",
        )
        results["DTHHRDY"] = {
            "cv_balanced_acc": float(scores2.mean()),
            "cv_balanced_acc_std": float(scores2.std()),
            "n": int(mask2.sum()),
            "n_classes": int(len(classes)),
            "chance": float(1 / len(classes)),
        }
    else:
        results["DTHHRDY"] = {
            "cv_balanced_acc": float("nan"), "cv_balanced_acc_std": float("nan"),
            "n": int(mask2.sum()), "note": "too few samples or classes",
        }

    return results
