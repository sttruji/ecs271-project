#!/usr/bin/env python3
"""Q13 — Time-of-day / circadian probe on GSE223613 ("TrACES of Time").

Why this dataset.  Q12 capped at AGE R² = 0.22 on GTEx and could not probe
"sleep" or "circadian" axes at all (GTEx has no time-of-collection). The
Pösel et al. 2023 GSE223613 cohort is purpose-built for that probe:

  • 10 healthy donors (A..J), ages 19–31, both sexes, whole blood
  • Sampled 8 times per donor across the day (08, 11, 14, 17, 20, 23, 02, 05h)
  • 80 RNA-seq libraries, single-center, single-platform (Illumina)
  • Public counts (GSE223613_counts.txt.gz, gene-symbol indexed, ~33k genes)

What we test.

  (1) **Time-of-day recovery.**  Given a latent vector z for a sample,
      predict the sampling hour. Treated two ways:
        - cyclic regression on (sin(2π h/24), cos(2π h/24)) — better fit
          for circular targets; report mean R² over both components.
        - 8-class classification on the discrete clock label — report
          balanced accuracy + macro AUC.
  (2) **Participant identity (donor fingerprint).**  10-class probe.
      A perfect score implies the latent encodes a donor-specific signature
      that survives across 8 sampling times — i.e. the encoder picks up
      stable inter-individual axes (genetics, baseline state, lifestyle).
  (3) **Sex.**  Sanity check — should be near-perfect from PCA on 50 PCs
      that include XIST etc., harder for compressed VAE latents (Q12).
  (4) **Age in years (19–31 only).**  Won't recover much (range too narrow)
      but kept for symmetry with Q12.

Latents compared (same as Q12).
  PCA-50 (GTEx-fit), Random Gauss-64 (GTEx-fit), Cross-mod VAE μ
  (collapsed), Sane VAE μ (Q12 checkpoint).

The encoders were trained / fit on GTEx whole blood. Applying them to
GSE223613 is a true cross-cohort test.

Run:  python q13_gse223613_circadian_probe.py
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from lib_data import align_to_shared, load_gtex_blood, standardise  # noqa: E402
from lib_model import load_trained  # noqa: E402

# Reuse the Sane-VAE class + cached checkpoint from Q12
from q12_metadata_probe import SaneVAE, encode_with_sane_vae  # noqa: E402

CHECKPOINT = "/Users/rls/ecs271/data/models/cross_modality_vae.pt"
GSE_DIR = Path("/Users/rls/ecs271/data/bulk/GSE223613")
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
RESULTS.mkdir(exist_ok=True, parents=True)
FIGS.mkdir(exist_ok=True, parents=True)

SEED = 0
LATENT_DIM = 64


# ── Load GSE223613 metadata from the series matrix ────────────────────────
def load_gse223613_metadata() -> pd.DataFrame:
    matrix = GSE_DIR / "GSE223613_series_matrix.txt.gz"
    rows: dict[str, list[list[str]]] = {}
    with gzip.open(matrix, "rt") as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                break
            if not line.startswith("!Sample_"):
                continue
            parts = line.rstrip("\n").split("\t")
            rows.setdefault(parts[0], []).append([p.strip('"') for p in parts[1:]])

    meta = pd.DataFrame(
        {
            "gsm": rows["!Sample_geo_accession"][0],
            "title": rows["!Sample_title"][0],
            "lib": rows["!Sample_description"][0],
        }
    )
    # Each !Sample_characteristics_ch1 line is one characteristic.
    for row in rows.get("!Sample_characteristics_ch1", []):
        keys = [c.split(":", 1)[0].strip() for c in row if ":" in c]
        if not keys:
            continue
        key = Counter(keys).most_common(1)[0][0]
        meta[key] = [c.split(":", 1)[1].strip() if ":" in c else "" for c in row]
    return meta


# ── Load counts + align genes to GTEx shared-gene basis ───────────────────
def load_gse223613_aligned(
    shared_genes: np.ndarray, scaler_mean: np.ndarray, scaler_std: np.ndarray
) -> tuple[np.ndarray, pd.DataFrame]:
    meta = load_gse223613_metadata()
    counts_path = GSE_DIR / "GSE223613_counts.txt.gz"

    # Counts are tab-separated, first col = gene symbol, header = sample lib IDs
    counts = pd.read_csv(counts_path, sep="\t", index_col=0, compression="gzip")
    counts.index = counts.index.astype(str)
    print(f"  raw counts: {counts.shape[0]:,} genes × {counts.shape[1]:,} samples")

    # Map sample columns (lib IDs in series matrix `lib` field) to row order
    lib_to_gsm = dict(zip(meta["lib"], meta["gsm"]))
    keep_cols = [c for c in counts.columns if c in lib_to_gsm]
    print(f"  matched lib → gsm: {len(keep_cols)} / {counts.shape[1]} cols")
    counts = counts[keep_cols]

    # Re-order metadata to match column order of counts
    meta_aligned = meta.set_index("lib").loc[keep_cols].reset_index()

    # Collapse duplicate gene symbols (rare) by summing
    counts = counts.groupby(level=0).sum()

    # log2(CPM + 1) using per-sample library size
    expr_raw = counts.values.astype(np.float64)        # (genes, samples)
    lib_size = expr_raw.sum(axis=0, keepdims=True)
    cpm = expr_raw / np.maximum(lib_size, 1) * 1e6
    expr_log = np.log2(cpm + 1).T.astype(np.float32)   # (samples, genes)
    gene_names = np.asarray(counts.index)

    # Align to GTEx shared-gene basis + apply same scaler
    aligned, n_found, missing = align_to_shared(expr_log, gene_names, shared_genes)
    print(f"  aligned to shared genes: {n_found} found, {len(missing)} filled w/ mean")
    X = standardise(aligned, scaler_mean, scaler_std)
    print(f"  X (samples × shared_genes): {X.shape}")
    return X, meta_aligned


# ── Probes ────────────────────────────────────────────────────────────────
def cv_ridge_r2(z: np.ndarray, y: np.ndarray, *, k: int = 5) -> dict:
    mask = np.isfinite(y) & np.isfinite(z).all(axis=1)
    if mask.sum() < 30:
        return {"r2": float("nan"), "r2_std": float("nan"), "n": int(mask.sum())}
    zk = StandardScaler().fit_transform(z[mask])
    yk = y[mask]
    cv = KFold(n_splits=k, shuffle=True, random_state=SEED)
    scores = cross_val_score(Ridge(alpha=1.0), zk, yk, cv=cv, scoring="r2")
    return {
        "r2": float(scores.mean()),
        "r2_std": float(scores.std()),
        "n": int(mask.sum()),
    }


def cv_logreg(
    z: np.ndarray, y: np.ndarray, *, k: int = 5, min_per_class: int = 5
) -> dict:
    finite = np.isfinite(z).all(axis=1)
    y_arr = pd.Series(y).where(pd.Series(y).notna()).astype("object")
    valid = finite & y_arr.notna().values
    if valid.sum() < 30:
        return {
            "bal_acc": float("nan"),
            "macro_auc": float("nan"),
            "n": int(valid.sum()),
            "n_classes": 0,
        }
    zv = StandardScaler().fit_transform(z[valid])
    yv = y_arr.values[valid].astype(str)
    classes, counts = np.unique(yv, return_counts=True)
    keep = classes[counts >= min_per_class]
    keep_mask = np.isin(yv, keep)
    if keep_mask.sum() < 30 or len(keep) < 2:
        return {
            "bal_acc": float("nan"),
            "macro_auc": float("nan"),
            "n": int(keep_mask.sum()),
            "n_classes": int(len(keep)),
        }
    zv, yv = zv[keep_mask], yv[keep_mask]
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", multi_class="auto")
    bal = cross_val_score(clf, zv, yv, cv=cv, scoring="balanced_accuracy")
    auc_metric = "roc_auc_ovr_weighted" if len(keep) > 2 else "roc_auc"
    try:
        auc = cross_val_score(clf, zv, yv, cv=cv, scoring=auc_metric)
        macro_auc = float(auc.mean())
    except Exception:
        macro_auc = float("nan")
    return {
        "bal_acc": float(bal.mean()),
        "bal_acc_std": float(bal.std()),
        "macro_auc": macro_auc,
        "n": int(keep_mask.sum()),
        "n_classes": int(len(keep)),
    }


def lodo_cyclic_mae(
    z: np.ndarray, hours: np.ndarray, donors: np.ndarray
) -> dict:
    """Leave-one-DONOR-out cyclic time-of-day prediction.

    Trains on 9 donors, tests on the held-out donor. With 8 timepoints
    per donor, the test fold sees all 8 hours so the probe is testing
    whether the *circadian axis* generalises across people, not whether
    we can memorise a person's transcriptome.
    """
    valid = np.isfinite(hours) & np.isfinite(z).all(axis=1)
    z, hours, donors = z[valid], hours[valid], donors[valid]
    theta = 2 * np.pi * hours / 24.0
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    zk = StandardScaler().fit_transform(z)
    deltas = []
    for d in np.unique(donors):
        train = donors != d
        test = donors == d
        if train.sum() < 10 or test.sum() < 1:
            continue
        m_sin = Ridge(alpha=1.0).fit(zk[train], sin_t[train]).predict(zk[test])
        m_cos = Ridge(alpha=1.0).fit(zk[train], cos_t[train]).predict(zk[test])
        pred_theta = np.arctan2(m_sin, m_cos)
        delta = (pred_theta - theta[test] + np.pi) % (2 * np.pi) - np.pi
        deltas.append(delta)
    if not deltas:
        return {"lodo_mae_hours": float("nan"), "lodo_n_donors": 0}
    delta = np.concatenate(deltas)
    return {
        "lodo_mae_hours": float(np.mean(np.abs(delta)) * 24.0 / (2 * np.pi)),
        "lodo_n_donors": int(len(np.unique(donors))),
    }


def lodo_age_r2(z: np.ndarray, age: np.ndarray, donors: np.ndarray) -> dict:
    """LODO age prediction. Each donor has one fixed age; on held-out donor
    we have to extrapolate age from their transcriptome — true generalisation."""
    valid = np.isfinite(age) & np.isfinite(z).all(axis=1)
    z, age, donors = z[valid], age[valid], donors[valid]
    zk = StandardScaler().fit_transform(z)
    preds = np.full(len(age), np.nan)
    for d in np.unique(donors):
        train = donors != d
        test = donors == d
        preds[test] = Ridge(alpha=1.0).fit(zk[train], age[train]).predict(zk[test])
    var_total = float(age.var())
    var_resid = float(((age - preds) ** 2).mean())
    return {
        "lodo_age_r2": float(1 - var_resid / var_total) if var_total > 0 else float("nan"),
        "lodo_age_mae": float(np.abs(age - preds).mean()),
        "lodo_n_donors": int(len(np.unique(donors))),
    }


def lodo_sex_auc(
    z: np.ndarray, sex: np.ndarray, donors: np.ndarray
) -> dict:
    """LODO sex classification. Tests whether sex axes extrapolate across donors."""
    valid = np.isfinite(sex) & np.isfinite(z).all(axis=1)
    z, sex, donors = z[valid], sex[valid].astype(int), donors[valid]
    zk = StandardScaler().fit_transform(z)
    correct = 0
    total = 0
    from sklearn.metrics import roc_auc_score
    all_y, all_p = [], []
    for d in np.unique(donors):
        train = donors != d
        test = donors == d
        if len(np.unique(sex[train])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(zk[train], sex[train])
        proba = clf.predict_proba(zk[test])[:, 1]
        pred = (proba > 0.5).astype(int)
        correct += int((pred == sex[test]).sum())
        total += int(test.sum())
        all_y.extend(sex[test].tolist())
        all_p.extend(proba.tolist())
    try:
        auc = float(roc_auc_score(all_y, all_p)) if len(set(all_y)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    return {
        "lodo_sex_acc": float(correct / total) if total else float("nan"),
        "lodo_sex_auc": auc,
        "lodo_n_donors": int(len(np.unique(donors))),
    }


def cv_cyclic_r2(z: np.ndarray, hours: np.ndarray, *, k: int = 5) -> dict:
    """Cyclic time-of-day regression: target = (sin θ, cos θ), θ = 2π h/24.

    Reports mean R² over the two components (sin and cos) and MAE in
    hours of the predicted angle.
    """
    valid = np.isfinite(hours) & np.isfinite(z).all(axis=1)
    if valid.sum() < 30:
        return {"r2_cyclic": float("nan"), "mae_hours": float("nan"), "n": int(valid.sum())}
    theta = 2 * np.pi * hours[valid] / 24.0
    sin_t = np.sin(theta).astype(np.float64)
    cos_t = np.cos(theta).astype(np.float64)
    zv = StandardScaler().fit_transform(z[valid])
    cv = KFold(n_splits=k, shuffle=True, random_state=SEED)
    r2_sin = cross_val_score(Ridge(alpha=1.0), zv, sin_t, cv=cv, scoring="r2")
    r2_cos = cross_val_score(Ridge(alpha=1.0), zv, cos_t, cv=cv, scoring="r2")
    # held-out predictions to compute angular MAE
    from sklearn.model_selection import cross_val_predict
    sin_hat = cross_val_predict(Ridge(alpha=1.0), zv, sin_t, cv=cv)
    cos_hat = cross_val_predict(Ridge(alpha=1.0), zv, cos_t, cv=cv)
    pred_theta = np.arctan2(sin_hat, cos_hat)
    delta = (pred_theta - theta + np.pi) % (2 * np.pi) - np.pi
    mae_hours = float(np.mean(np.abs(delta)) * 24.0 / (2 * np.pi))
    return {
        "r2_cyclic": float((r2_sin.mean() + r2_cos.mean()) / 2),
        "r2_sin": float(r2_sin.mean()),
        "r2_cos": float(r2_cos.mean()),
        "mae_hours": mae_hours,
        "n": int(valid.sum()),
    }


def main() -> int:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"Device: {device}")

    # ── Load GTEx + GTEx-fit baselines (PCA-50, Rand-64, CM-VAE, Sane-VAE) ──
    print("\n[1/4] Loading GTEx + fitting GTEx-basis projections ...")
    cm_vae, shared_genes, scaler_mean, scaler_std = load_trained(CHECKPOINT, device="cpu")
    expr_log, gene_names = load_gtex_blood()
    expr_aligned, _, _ = align_to_shared(expr_log, gene_names, shared_genes)
    X_gtex = standardise(expr_aligned, scaler_mean, scaler_std)
    n_genes = X_gtex.shape[1]
    print(f"  GTEx X: {X_gtex.shape}")

    pca = PCA(n_components=50, random_state=SEED).fit(X_gtex)
    rp = GaussianRandomProjection(n_components=LATENT_DIM, random_state=SEED).fit(X_gtex)

    # Sane-VAE from Q12 cache
    sane_ckpt_path = RESULTS / "q12_sane_vae.pt"
    cached = torch.load(sane_ckpt_path, map_location=device, weights_only=False)
    sane = SaneVAE(n_genes=n_genes, latent_dim=LATENT_DIM).to(device)
    sane.load_state_dict(cached["state_dict"])
    sane.eval()
    print(f"  Sane VAE loaded from {sane_ckpt_path} (R²={cached.get('holdout_r2', float('nan')):.3f})")

    # ── Encode GSE223613 ────────────────────────────────────────────────
    print("\n[2/4] Loading + aligning GSE223613 ...")
    X_gse, gse_meta = load_gse223613_aligned(shared_genes, scaler_mean, scaler_std)

    # Parse + cast metadata
    gse_meta["hour"] = (
        gse_meta["sampling time"]
        .astype(str)
        .str.replace("h", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace(".00", "", regex=False)
        .astype(float)
    )
    gse_meta["participant_id"] = gse_meta["participant"].astype(str)
    gse_meta["sex_bin"] = (
        gse_meta["Sex"].astype(str).str.strip().str.lower().map(
            {"male": 0, "female": 1, "m": 0, "f": 1}
        )
    )
    gse_meta["age_yrs"] = pd.to_numeric(gse_meta["age"], errors="coerce")

    print("  Metadata coverage:")
    print(f"    hour: unique={sorted(gse_meta['hour'].unique())}  "
          f"n_finite={gse_meta['hour'].notna().sum()}")
    print(f"    participant: unique={sorted(gse_meta['participant_id'].unique())}  "
          f"counts={gse_meta['participant_id'].value_counts().to_dict()}")
    print(f"    sex: {gse_meta['Sex'].value_counts().to_dict()}")
    print(f"    age: range={gse_meta['age_yrs'].min():.0f}-"
          f"{gse_meta['age_yrs'].max():.0f}, mean={gse_meta['age_yrs'].mean():.1f}")

    # ── Build 4 latents on GSE223613 samples ───────────────────────────
    Z_pca = pca.transform(X_gse).astype(np.float32)
    Z_rand = rp.transform(X_gse).astype(np.float32)
    with torch.no_grad():
        Z_cm = cm_vae.enc_bulk(torch.from_numpy(X_gse).float())[0].cpu().numpy()
    Z_sane = encode_with_sane_vae(sane, X_gse, device=device)

    for name, Z in [("PCA-50", Z_pca), ("Rand-64", Z_rand),
                    ("CM-VAE", Z_cm), ("Sane-VAE", Z_sane)]:
        v = Z.var(0)
        print(
            f"  {name:<10}  shape={Z.shape}  "
            f"active>0.01={int((v > 0.01).sum())}/{Z.shape[1]}  "
            f"mean var={v.mean():.4f}"
        )

    # ── Probes ──────────────────────────────────────────────────────────
    print("\n[3/4] Running probes (5-fold CV) ...")
    latents = [
        ("PCA-50", Z_pca),
        ("Rand-64", Z_rand),
        ("CM-VAE", Z_cm),
        ("Sane-VAE", Z_sane),
    ]
    rows = []
    for lname, Z in latents:
        # (a) Cyclic time-of-day regression
        cyc = cv_cyclic_r2(Z, gse_meta["hour"].values)
        rows.append({
            "latent": lname, "target": "hour (cyclic)", "metric": "r2_cyclic",
            "value": cyc["r2_cyclic"], "n": cyc["n"], "n_classes": np.nan,
        })
        rows.append({
            "latent": lname, "target": "hour (cyclic)", "metric": "mae_hours",
            "value": cyc["mae_hours"], "n": cyc["n"], "n_classes": np.nan,
        })
        # (b) 8-class clock probe
        hour_str = gse_meta["hour"].astype(str).values
        cls = cv_logreg(Z, hour_str)
        rows.append({
            "latent": lname, "target": "hour (8-class)", "metric": "bal_acc",
            "value": cls["bal_acc"], "n": cls["n"], "n_classes": cls["n_classes"],
        })
        rows.append({
            "latent": lname, "target": "hour (8-class)", "metric": "macro_auc",
            "value": cls["macro_auc"], "n": cls["n"], "n_classes": cls["n_classes"],
        })
        # (c) Participant identity (10-class)
        pid = gse_meta["participant_id"].values
        prt = cv_logreg(Z, pid)
        rows.append({
            "latent": lname, "target": "participant", "metric": "bal_acc",
            "value": prt["bal_acc"], "n": prt["n"], "n_classes": prt["n_classes"],
        })
        rows.append({
            "latent": lname, "target": "participant", "metric": "macro_auc",
            "value": prt["macro_auc"], "n": prt["n"], "n_classes": prt["n_classes"],
        })
        # (d) Sex
        sex_y = gse_meta["sex_bin"].values
        sx = cv_logreg(Z, sex_y)
        rows.append({
            "latent": lname, "target": "sex", "metric": "bal_acc",
            "value": sx["bal_acc"], "n": sx["n"], "n_classes": sx["n_classes"],
        })
        rows.append({
            "latent": lname, "target": "sex", "metric": "macro_auc",
            "value": sx["macro_auc"], "n": sx["n"], "n_classes": sx["n_classes"],
        })
        # (e) Age (continuous, 19-31)
        age_y = gse_meta["age_yrs"].values
        agr = cv_ridge_r2(Z, age_y)
        rows.append({
            "latent": lname, "target": "age (yrs)", "metric": "R2",
            "value": agr["r2"], "n": agr["n"], "n_classes": np.nan,
        })

        print(
            f"  {lname:<10}  cyclic R²={cyc['r2_cyclic']:+.3f}  "
            f"|  hour-MAE={cyc['mae_hours']:.2f}h  "
            f"|  hour-AUC={cls['macro_auc']:.3f}  "
            f"|  participant-AUC={prt['macro_auc']:.3f}  "
            f"|  sex-AUC={sx['macro_auc']:.3f}  "
            f"|  age R²={agr['r2']:+.3f}"
        )

    # ── Leave-one-donor-out (honest cross-individual probes) ─────────────
    # 5-fold random CV above leaks donors across folds (each donor has 8
    # samples, every fold sees most donors). The LODO numbers below are
    # the actually-honest measure of "given an unseen donor's
    # transcriptome, what can we recover?".
    print("\n[3b/4] LODO probes (each fold = one donor held out) ...")
    donors_arr = gse_meta["participant_id"].values
    for lname, Z in latents:
        ld_cyc = lodo_cyclic_mae(Z, gse_meta["hour"].values, donors_arr)
        ld_age = lodo_age_r2(Z, gse_meta["age_yrs"].values, donors_arr)
        ld_sex = lodo_sex_auc(Z, gse_meta["sex_bin"].values, donors_arr)
        rows.append({
            "latent": lname, "target": "hour (LODO cyclic)",
            "metric": "mae_hours", "value": ld_cyc["lodo_mae_hours"],
            "n": int(len(donors_arr)), "n_classes": np.nan,
        })
        rows.append({
            "latent": lname, "target": "age (LODO)",
            "metric": "R2", "value": ld_age["lodo_age_r2"],
            "n": int(len(donors_arr)), "n_classes": np.nan,
        })
        rows.append({
            "latent": lname, "target": "age (LODO)",
            "metric": "mae_yrs", "value": ld_age["lodo_age_mae"],
            "n": int(len(donors_arr)), "n_classes": np.nan,
        })
        rows.append({
            "latent": lname, "target": "sex (LODO)",
            "metric": "auc", "value": ld_sex["lodo_sex_auc"],
            "n": int(len(donors_arr)), "n_classes": 2,
        })
        print(
            f"  {lname:<10}  LODO hour-MAE={ld_cyc['lodo_mae_hours']:.2f}h  "
            f"|  LODO age R²={ld_age['lodo_age_r2']:+.3f} "
            f"(MAE={ld_age['lodo_age_mae']:.1f} yrs)  "
            f"|  LODO sex-AUC={ld_sex['lodo_sex_auc']:.3f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "q13_gse223613_probe.csv", index=False)
    summary = {
        "n_samples": int(X_gse.shape[0]),
        "n_donors": int(gse_meta["participant_id"].nunique()),
        "n_timepoints": int(gse_meta["hour"].nunique()),
        "rows": rows,
    }
    (RESULTS / "q13_gse223613_probe.json").write_text(json.dumps(summary, indent=2))
    print(f"  Wrote {RESULTS / 'q13_gse223613_probe.csv'}")
    print(f"  Wrote {RESULTS / 'q13_gse223613_probe.json'}")

    # ── Figure ──────────────────────────────────────────────────────────
    print("\n[4/4] Drawing figure ...")
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # Panel 1: bar — main metrics
    targets = [
        ("hour (cyclic)", "r2_cyclic", "Hour\n(cyclic R²)"),
        ("hour (8-class)", "macro_auc", "Hour\n(8-class AUC)"),
        ("participant", "macro_auc", "Participant\n(10-class AUC)"),
        ("sex", "macro_auc", "Sex\n(AUC)"),
        ("age (yrs)", "R2", "Age R²\n(19-31 yrs)"),
    ]
    n_t = len(targets)
    n_l = len(latents)
    bar_w = 0.18
    xs = np.arange(n_t)
    colors = {"PCA-50": "#58a6ff", "Rand-64": "#7d8590",
              "CM-VAE": "#f78166", "Sane-VAE": "#3fb950"}
    for k, (lname, _) in enumerate(latents):
        ys = []
        for t, m, _ in targets:
            v = df[(df.latent == lname) & (df.target == t) & (df.metric == m)]["value"]
            ys.append(float(v.iloc[0]) if len(v) else 0.0)
        axes[0].bar(xs + (k - (n_l - 1) / 2) * bar_w, ys, width=bar_w,
                    label=lname, color=colors.get(lname, "#999"))
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([lab for _, _, lab in targets], fontsize=8)
    axes[0].axhline(0.5, color="#7d8590", ls=":", lw=0.5, label="AUC chance (0.5)")
    axes[0].axhline(0.0, color="#7d8590", lw=0.5)
    axes[0].set_ylim(-0.5, 1.05)
    axes[0].set_ylabel("Probe metric")
    axes[0].set_title("Q13a — Cross-cohort metadata recovery on GSE223613 (n=80)")
    axes[0].legend(fontsize=8, loc="lower right")

    # Panel 2: time-of-day MAE bar
    mae_vals = []
    for lname, _ in latents:
        v = df[(df.latent == lname) & (df.metric == "mae_hours")]["value"]
        mae_vals.append(float(v.iloc[0]) if len(v) else float("nan"))
    axes[1].bar([n for n, _ in latents], mae_vals,
                color=[colors[n] for n, _ in latents])
    axes[1].axhline(6.0, color="#7d8590", ls=":", lw=0.5,
                    label="random circular guess ≈ 6h")
    axes[1].set_ylabel("Hour-prediction MAE (held-out)")
    axes[1].set_title("Q13b — Time-of-day MAE (cyclic regression)")
    axes[1].legend(fontsize=8)
    for i, v in enumerate(mae_vals):
        axes[1].text(i, v + 0.1, f"{v:.2f}h", ha="center", fontsize=9)

    # Panel 3: scatter PC1 vs PC2 colored by hour, marker by participant
    color_for_hour = plt.get_cmap("twilight")((gse_meta["hour"].values % 24) / 24.0)
    markers = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">"]
    pid_to_marker = {p: markers[i % len(markers)]
                     for i, p in enumerate(sorted(gse_meta["participant_id"].unique()))}
    for p in sorted(gse_meta["participant_id"].unique()):
        m = gse_meta["participant_id"] == p
        axes[2].scatter(
            Z_pca[m, 0], Z_pca[m, 1],
            c=color_for_hour[m], marker=pid_to_marker[p],
            s=70, edgecolors="#222", linewidths=0.6, label=p,
        )
    axes[2].set_xlabel("PCA-50 PC1 (GTEx-fit)")
    axes[2].set_ylabel("PCA-50 PC2 (GTEx-fit)")
    axes[2].set_title("Q13c — GSE223613 in GTEx PC space\n"
                      "(color = hour, marker = donor)")

    fig.tight_layout()
    fig.savefig(FIGS / "q13_gse223613_probe.png", dpi=150)
    plt.close(fig)
    print(f"  Wrote {FIGS / 'q13_gse223613_probe.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
