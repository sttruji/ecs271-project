#!/usr/bin/env python3
"""Q12 — Metadata probing of unsupervised latents.

GOAL.  Test whether donor metadata (age, sex, ischemia, RIN, Hardy class,
sequencing center, CMV status, ...) is implicitly *recoverable* from
unsupervised representations of bulk RNA-seq, even though no model has seen
these labels during training.  This is a textbook "linear probe" evaluation:
we freeze the encoder, train a 5-fold-CV linear classifier/regressor from the
latent vector to each metadata target, and report a held-out metric.

LATENTS COMPARED (all on the same 803 GTEx whole-blood donors).

  •  PCA-50            — linear baseline; same input matrix.
  •  Random Gauss-64   — null projection; preserves no signal beyond
                        what fits in 64 random linear combinations.
  •  Cross-mod VAE μ   — the existing posterior-collapsed checkpoint
                        (REPORT Q1: held-out R² = -3.27, all dims var <
                        1e-3).  Expected to fail almost everywhere ⇒
                        confirms the probe is honest.
  •  Sane VAE μ        — newly trained here, same arch as REPORT Q3
                        (β = 1e-3, no adversarial term, 64-D, MLP).
                        Reaches R² ≈ 0.79 on held-out reconstruction.

PROBES.

  •  Continuous targets — Ridge regression, 5-fold KFold, mean R² over
     held-out folds.  Targets: AGE_mid (years, bracket midpoint),
     DTHHRDY (0-4 ordinal Hardy class), SMRIN (RNA integrity, 1-10),
     SMTSISCH (post-mortem ischemia time in minutes), SMRDLGTH (read
     length).
  •  Categorical targets — Logistic regression (multinomial), 5-fold
     StratifiedKFold, mean balanced accuracy + mean macro-AUC.  Targets:
     SEX (M/F), SMCENTER (sequencing-center ID, ~5 levels).

CROSS-COHORT GENERALIZATION.  Encode GSE279480 Null-stim samples through
the trained Sane-VAE encoder (gene-symbol-aligned + same scaler as GTEx)
and probe Sex / age cohort (BR1-BR4) / CMV status from those latents.
Tests whether GTEx-learned representations preserve donor-level metadata
in an independent cohort.

OUTPUTS.

  results/q12_metadata_probe.json   per-latent / per-target metric table
  results/q12_metadata_probe.csv    same, flat tidy table
  figures/q12_metadata_probe.png    summary heatmap + bars
  results/q12_sane_vae.pt           Sane-VAE checkpoint (encoder weights
                                    + shared_genes + scaler)

Run:  python q12_metadata_probe.py
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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from lib_data import align_to_shared, load_gtex_blood, standardise  # noqa: E402
from lib_model import load_trained  # noqa: E402

CHECKPOINT = "/Users/rls/ecs271/data/models/cross_modality_vae.pt"
ANNOT = Path("/Users/rls/ecs271/data/annotations")
GSE_DIR = Path(
    "/Users/rls/Desktop/programming-projects/single-cell/bulk-project/data/GSE279480"
)
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
RESULTS.mkdir(exist_ok=True, parents=True)
FIGS.mkdir(exist_ok=True, parents=True)

SEED = 0
LATENT_DIM = 64
SANE_VAE_EPOCHS = 200
SANE_VAE_BETA = 1e-3
SANE_VAE_LR = 1e-3
SANE_VAE_BATCH = 64


# ── Sane VAE — same arch as REPORT Q3 ─────────────────────────────────────
class SaneVAE(nn.Module):
    def __init__(self, n_genes: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.enc_trunk = nn.Sequential(
            nn.Linear(n_genes, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
        )
        self.mu = nn.Linear(512, latent_dim)
        self.logv = nn.Linear(512, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(1024, n_genes),
        )

    def forward(self, x):
        h = self.enc_trunk(x)
        mu = self.mu(h)
        logv = self.logv(h).clamp(-10, 4)
        if self.training:
            z = mu + (0.5 * logv).exp() * torch.randn_like(mu)
        else:
            z = mu
        return self.decoder(z), mu, logv


def train_sane_vae(
    train_t: torch.Tensor,
    test_t: torch.Tensor,
    n_genes: int,
    *,
    device: str,
    beta: float = SANE_VAE_BETA,
    epochs: int = SANE_VAE_EPOCHS,
    lr: float = SANE_VAE_LR,
    batch: int = SANE_VAE_BATCH,
) -> tuple[SaneVAE, dict]:
    torch.manual_seed(SEED)
    model = SaneVAE(n_genes=n_genes, latent_dim=LATENT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loader = DataLoader(
        TensorDataset(train_t.to(device)),
        batch_size=batch,
        shuffle=True,
        drop_last=True,
    )
    history = {"train_recon": [], "test_recon": [], "kl": []}
    for ep in range(1, epochs + 1):
        model.train()
        ep_recon = 0.0
        ep_kl = 0.0
        n_batches = 0
        for (xb,) in loader:
            opt.zero_grad()
            xh, mu, logv = model(xb)
            recon = F.mse_loss(xh, xb)
            kl = -0.5 * (1 + logv - mu.pow(2) - logv.exp()).sum(-1).mean()
            loss = recon + beta * kl
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_recon += float(recon)
            ep_kl += float(kl)
            n_batches += 1
        model.eval()
        with torch.no_grad():
            xh, _, _ = model(test_t.to(device))
            te_recon = float(F.mse_loss(xh, test_t.to(device)))
        history["train_recon"].append(ep_recon / n_batches)
        history["test_recon"].append(te_recon)
        history["kl"].append(ep_kl / n_batches)
        if ep % 25 == 0 or ep == 1 or ep == epochs:
            print(
                f"  [Sane-VAE] ep {ep:>3}/{epochs}  "
                f"train MSE={history['train_recon'][-1]:.4f}  "
                f"test MSE={te_recon:.4f}  KL={history['kl'][-1]:.3f}"
            )
    return model, history


def encode_with_sane_vae(model: SaneVAE, x: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(x.astype(np.float32)).to(device)
        h = model.enc_trunk(xt)
        mu = model.mu(h)
    return mu.cpu().numpy()


# ── GTEx metadata join ────────────────────────────────────────────────────
def sample_to_subject(sample_id: str) -> str:
    return "-".join(sample_id.split("-")[:2])


def age_to_midpoint(age_band: str) -> float:
    if not isinstance(age_band, str) or "-" not in age_band:
        return float("nan")
    a, b = age_band.split("-")
    try:
        return (int(a) + int(b)) / 2
    except ValueError:
        return float("nan")


def load_gtex_metadata(sample_ids: list[str]) -> pd.DataFrame:
    sub = pd.read_csv(ANNOT / "GTEx_v10_Annotations_SubjectPhenotypesDS.txt", sep="\t")
    samp = pd.read_csv(
        ANNOT / "GTEx_v10_Annotations_SampleAttributesDS.txt",
        sep="\t",
        low_memory=False,
    )
    df = pd.DataFrame({"SAMPID": sample_ids})
    df["SUBJID"] = df["SAMPID"].apply(sample_to_subject)
    df = df.merge(sub, on="SUBJID", how="left")
    df = df.merge(
        samp[
            [
                "SAMPID",
                "SMRIN",
                "SMTSISCH",
                "SMCENTER",
                "SMNABTCH",
                "SMGEBTCH",
                "SMRDLGTH",
            ]
        ],
        on="SAMPID",
        how="left",
    )
    df["AGE_mid"] = df["AGE"].apply(age_to_midpoint)
    return df


# ── GSE279480 metadata ────────────────────────────────────────────────────
def load_gse_metadata() -> pd.DataFrame:
    matrix = GSE_DIR / "GSE279480_series_matrix.txt.gz"
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
    for row in rows.get("!Sample_characteristics_ch1", []):
        keys = [c.split(":", 1)[0].strip() for c in row if ":" in c]
        if not keys:
            continue
        key = Counter(keys).most_common(1)[0][0]
        meta[key] = [c.split(":", 1)[1].strip() if ":" in c else "" for c in row]
    return meta


def gtex_ensembl_to_symbol() -> dict[str, str]:
    df = pd.read_csv(
        "/Users/rls/ecs271/data/bulk/gtex_v11_whole_blood.gct.gz",
        sep="\t",
        skiprows=2,
        compression="gzip",
        usecols=["Name", "Description"],
    )
    mapping: dict[str, str] = {}
    for ensg, sym in zip(df["Name"].astype(str), df["Description"].astype(str)):
        ensg_no_v = ensg.split(".")[0]
        if ensg_no_v not in mapping:
            mapping[ensg_no_v] = sym
    return mapping


def load_gse_aligned_to_gtex(
    shared_genes: np.ndarray, scaler_mean: np.ndarray, scaler_std: np.ndarray
):
    meta = load_gse_metadata()
    null_meta = meta[meta["stimulation"] == "Null"].copy()
    counts = pd.read_csv(GSE_DIR / "GSE279480_P441_genecounts.csv.gz", index_col=0)
    null_libs = set(null_meta["lib"].astype(str))
    keep_cols = [c for c in counts.columns if c in null_libs]
    counts_null = counts[keep_cols].copy()
    e2s = gtex_ensembl_to_symbol()
    counts_null.index = counts_null.index.astype(str).str.split(".").str[0]
    counts_null = counts_null.loc[counts_null.index.isin(e2s)].copy()
    counts_null.index = [e2s[g] for g in counts_null.index]
    counts_null = counts_null.groupby(level=0).sum()
    expr_raw = counts_null.values.astype(np.float64)
    lib_size = expr_raw.sum(axis=0, keepdims=True)
    cpm = expr_raw / np.maximum(lib_size, 1) * 1e6
    expr_log_gse = np.log2(cpm + 1).T.astype(np.float32)
    aligned, _, _ = align_to_shared(
        expr_log_gse, np.asarray(counts_null.index), shared_genes
    )
    X_gse = standardise(aligned, scaler_mean, scaler_std)
    gse_meta_aligned = (
        null_meta.set_index("lib").loc[keep_cols].reset_index()
    )
    age_map = {f"BR{i}": int(i) for i in range(1, 5)}
    gse_meta_aligned["AGE_cat"] = gse_meta_aligned["age cohort"].map(age_map)
    # Lowercase the inputs to be robust to capitalization (this cohort uses
    # 'male'/'female'; q9 mapped the capitalized form, which yielded all-NaN.)
    gse_meta_aligned["SEX_bin"] = (
        gse_meta_aligned["Sex"].astype(str).str.strip().str.lower().map(
            {"male": 0, "female": 1, "m": 0, "f": 1}
        )
    )
    gse_meta_aligned["CMV_bin"] = (
        gse_meta_aligned["cmv status"].astype(str).str.strip().str.lower().map(
            {"negative": 0, "positive": 1, "neg": 0, "pos": 1}
        )
    )
    return X_gse, gse_meta_aligned


# ── Probes ────────────────────────────────────────────────────────────────
def cv_ridge_r2(z: np.ndarray, y: np.ndarray, *, k: int = 5) -> dict:
    """5-fold mean R² of Ridge(z) → y on samples where y is finite."""
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


def cv_logreg_balacc(
    z: np.ndarray, y: np.ndarray, *, k: int = 5, min_per_class: int = 5
) -> dict:
    """5-fold StratifiedKFold mean balanced accuracy + macro-AUC of logistic
    regression z → y."""
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
    keep_classes = classes[counts >= min_per_class]
    keep_mask = np.isin(yv, keep_classes)
    if keep_mask.sum() < 30 or len(keep_classes) < 2:
        return {
            "bal_acc": float("nan"),
            "macro_auc": float("nan"),
            "n": int(keep_mask.sum()),
            "n_classes": int(len(keep_classes)),
        }
    zv = zv[keep_mask]
    yv = yv[keep_mask]
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    clf = LogisticRegression(
        max_iter=2000, C=1.0, solver="lbfgs", multi_class="auto"
    )
    bal = cross_val_score(clf, zv, yv, cv=cv, scoring="balanced_accuracy")
    auc_metric = "roc_auc_ovr_weighted" if len(keep_classes) > 2 else "roc_auc"
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
        "n_classes": int(len(keep_classes)),
    }


# ── main ──────────────────────────────────────────────────────────────────
def main() -> int:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"Device: {device}")

    # GTEx ----------------------------------------------------------------
    print("\n[1/6] Loading GTEx whole blood + scaler from cross-mod VAE checkpoint ...")
    cm_vae, shared_genes, scaler_mean, scaler_std = load_trained(CHECKPOINT, device="cpu")
    expr_log, gene_names = load_gtex_blood()
    expr_aligned, _, _ = align_to_shared(expr_log, gene_names, shared_genes)
    X_gtex = standardise(expr_aligned, scaler_mean, scaler_std)
    n_genes = X_gtex.shape[1]
    n_samples = X_gtex.shape[0]
    print(f"  GTEx X: {X_gtex.shape}  (donors × genes)")

    # Sample IDs from GCT header
    with gzip.open("/Users/rls/ecs271/data/bulk/gtex_v11_whole_blood.gct.gz", "rt") as fh:
        for _ in range(2):
            fh.readline()
        header = fh.readline().rstrip("\n").split("\t")
    sample_ids = header[2:]
    assert len(sample_ids) == n_samples

    # ── Train / test split for the VAE ───────────────────────────────────
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_samples)
    n_test = int(round(0.2 * n_samples))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    train_t = torch.from_numpy(X_gtex[train_idx])
    test_t = torch.from_numpy(X_gtex[test_idx])

    # ── Train Sane VAE on GTEx (or load cached checkpoint) ───────────────
    sane_ckpt_path = RESULTS / "q12_sane_vae.pt"
    if sane_ckpt_path.exists():
        print(
            f"\n[2/6] Loading cached Sane VAE checkpoint {sane_ckpt_path} "
            "(delete the file to force retraining)"
        )
        cached = torch.load(sane_ckpt_path, map_location=device, weights_only=False)
        sane = SaneVAE(n_genes=n_genes, latent_dim=LATENT_DIM).to(device)
        sane.load_state_dict(cached["state_dict"])
        hist = cached.get("history", {})
        r2_holdout = float(cached.get("holdout_r2", float("nan")))
        print(f"  Sane VAE held-out R² (cached) = {r2_holdout:.3f}")
    else:
        print(
            f"\n[2/6] Training Sane VAE  (β={SANE_VAE_BETA}, "
            f"latent={LATENT_DIM}, epochs={SANE_VAE_EPOCHS}, "
            f"batch={SANE_VAE_BATCH}) ..."
        )
        sane, hist = train_sane_vae(
            train_t, test_t, n_genes=n_genes, device=device
        )
        sane.eval()
        with torch.no_grad():
            xh_te, _, _ = sane(test_t.to(device))
        pred_te = xh_te.cpu().numpy()
        true_te = X_gtex[test_idx]
        diff = true_te - pred_te
        r2_holdout = float(1 - diff.var() / true_te.var())
        print(f"  Sane VAE held-out R² = {r2_holdout:.3f}")
        torch.save(
            {
                "state_dict": sane.state_dict(),
                "shared_genes": shared_genes,
                "scaler_mean": scaler_mean,
                "scaler_std": scaler_std,
                "latent_dim": LATENT_DIM,
                "n_genes": n_genes,
                "history": hist,
                "holdout_r2": r2_holdout,
            },
            sane_ckpt_path,
        )
        print(f"  Saved {sane_ckpt_path}")

    # ── Build all latents on the full 803-donor GTEx matrix ──────────────
    print("\n[3/6] Building latents on all 803 GTEx donors ...")

    # PCA-50 on standardised X
    pca = PCA(n_components=50, random_state=SEED).fit(X_gtex)
    Z_pca = pca.transform(X_gtex).astype(np.float32)

    # Random Gaussian projection to 64-D
    rp = GaussianRandomProjection(n_components=LATENT_DIM, random_state=SEED).fit(X_gtex)
    Z_rand = rp.transform(X_gtex).astype(np.float32)

    # Cross-mod VAE μ (collapsed)
    cm_vae.eval()
    with torch.no_grad():
        Z_cm = cm_vae.enc_bulk(torch.from_numpy(X_gtex).float())[0].cpu().numpy()

    # Sane VAE μ
    Z_sane = encode_with_sane_vae(sane, X_gtex, device=device)

    # Diagnostics
    for name, Z in [("PCA-50", Z_pca), ("Rand-64", Z_rand),
                    ("CM-VAE μ", Z_cm), ("Sane-VAE μ", Z_sane)]:
        v = Z.var(0)
        print(
            f"  {name:<12}  shape={Z.shape}  "
            f"active>0.01={int((v > 0.01).sum())}/{Z.shape[1]}  "
            f"mean var={v.mean():.4f}"
        )

    # ── Metadata join ────────────────────────────────────────────────────
    print("\n[4/6] Joining GTEx metadata ...")
    df = load_gtex_metadata(sample_ids)
    cont_targets = ["AGE_mid", "DTHHRDY", "SMRIN", "SMTSISCH", "SMRDLGTH"]
    cat_targets = ["SEX", "SMCENTER"]
    for t in cont_targets:
        v = pd.to_numeric(df[t], errors="coerce")
        print(f"  {t:<10}  n_finite={v.notna().sum()}  "
              f"min={v.min():.2f}  max={v.max():.2f}  "
              f"mean={v.mean():.2f}")
    for t in cat_targets:
        vc = df[t].dropna().value_counts()
        print(f"  {t:<10}  n={vc.sum()}  classes={dict(vc.head(6))}")

    # ── Probes on each latent × each metadata target ─────────────────────
    print("\n[5/6] Running 5-fold CV linear probes ...")
    latents = [
        ("PCA-50", Z_pca),
        ("Rand-64", Z_rand),
        ("CM-VAE", Z_cm),
        ("Sane-VAE", Z_sane),
    ]
    rows = []
    for latent_name, Z in latents:
        for t in cont_targets:
            y = pd.to_numeric(df[t], errors="coerce").values
            res = cv_ridge_r2(Z, y)
            rows.append(
                {
                    "cohort": "GTEx",
                    "latent": latent_name,
                    "target": t,
                    "kind": "continuous",
                    "metric": "R2",
                    "value": res["r2"],
                    "std": res.get("r2_std", float("nan")),
                    "n": res["n"],
                    "n_classes": np.nan,
                }
            )
            print(
                f"  GTEx  {latent_name:<10} {t:<10}  R² = {res['r2']:+.3f} "
                f"± {res.get('r2_std', float('nan')):.3f}  (n={res['n']})"
            )
        for t in cat_targets:
            y = df[t].values
            res = cv_logreg_balacc(Z, y)
            rows.append(
                {
                    "cohort": "GTEx",
                    "latent": latent_name,
                    "target": t,
                    "kind": "categorical",
                    "metric": "bal_acc",
                    "value": res["bal_acc"],
                    "std": res.get("bal_acc_std", float("nan")),
                    "n": res["n"],
                    "n_classes": res["n_classes"],
                }
            )
            rows.append(
                {
                    "cohort": "GTEx",
                    "latent": latent_name,
                    "target": t,
                    "kind": "categorical",
                    "metric": "macro_auc",
                    "value": res["macro_auc"],
                    "std": float("nan"),
                    "n": res["n"],
                    "n_classes": res["n_classes"],
                }
            )
            print(
                f"  GTEx  {latent_name:<10} {t:<10}  "
                f"balAcc = {res['bal_acc']:.3f}  AUC = {res['macro_auc']:.3f}  "
                f"(n={res['n']}, classes={res['n_classes']})"
            )

    # ── GSE279480 cross-cohort probe ─────────────────────────────────────
    print("\n[6/6] Encoding GSE279480 Null samples + probing ...")
    X_gse, gse_meta = load_gse_aligned_to_gtex(shared_genes, scaler_mean, scaler_std)
    print(f"  GSE Null aligned X: {X_gse.shape}  (samples × genes)")

    Z_pca_gse = pca.transform(X_gse).astype(np.float32)  # GTEx-fit PCA
    Z_rand_gse = rp.transform(X_gse).astype(np.float32)  # GTEx-fit RP
    with torch.no_grad():
        Z_cm_gse = cm_vae.enc_bulk(torch.from_numpy(X_gse).float())[0].cpu().numpy()
    Z_sane_gse = encode_with_sane_vae(sane, X_gse, device=device)

    # NB: "age cohort" in GSE279480 only has 2 levels (BR1, BR2), so we
    # treat it as a categorical probe rather than ridge regression on the
    # ordinal code. CMV/Sex are obviously categorical.
    gse_targets = [
        ("AGE_cat", "categorical"),
        ("SEX_bin", "categorical"),
        ("CMV_bin", "categorical"),
    ]
    gse_latents = [
        ("PCA-50", Z_pca_gse),
        ("Rand-64", Z_rand_gse),
        ("CM-VAE", Z_cm_gse),
        ("Sane-VAE", Z_sane_gse),
    ]
    for latent_name, Z in gse_latents:
        for t, kind in gse_targets:
            if kind == "continuous":
                y = pd.to_numeric(gse_meta[t], errors="coerce").values
                res = cv_ridge_r2(Z, y)
                rows.append(
                    {
                        "cohort": "GSE279480",
                        "latent": latent_name,
                        "target": t,
                        "kind": kind,
                        "metric": "R2",
                        "value": res["r2"],
                        "std": res.get("r2_std", float("nan")),
                        "n": res["n"],
                        "n_classes": np.nan,
                    }
                )
                print(
                    f"  GSE   {latent_name:<10} {t:<10}  R² = {res['r2']:+.3f}  "
                    f"(n={res['n']})"
                )
            else:
                y = gse_meta[t].values
                res = cv_logreg_balacc(Z, y)
                for metric in ("bal_acc", "macro_auc"):
                    rows.append(
                        {
                            "cohort": "GSE279480",
                            "latent": latent_name,
                            "target": t,
                            "kind": kind,
                            "metric": metric,
                            "value": res[metric],
                            "std": res.get("bal_acc_std", float("nan"))
                            if metric == "bal_acc"
                            else float("nan"),
                            "n": res["n"],
                            "n_classes": res["n_classes"],
                        }
                    )
                print(
                    f"  GSE   {latent_name:<10} {t:<10}  "
                    f"balAcc = {res['bal_acc']:.3f}  "
                    f"AUC = {res['macro_auc']:.3f}  (n={res['n']})"
                )

    # ── Save flat tidy results ───────────────────────────────────────────
    df_out = pd.DataFrame(rows)
    df_out.to_csv(RESULTS / "q12_metadata_probe.csv", index=False)
    summary = {
        "sane_vae_holdout_r2": r2_holdout,
        "sane_vae_active_dims": int((Z_sane.var(0) > 0.01).sum()),
        "cm_vae_active_dims": int((Z_cm.var(0) > 0.01).sum()),
        "n_gtex": int(n_samples),
        "n_gse_null": int(X_gse.shape[0]),
        "rows": rows,
    }
    (RESULTS / "q12_metadata_probe.json").write_text(json.dumps(summary, indent=2))
    print(f"  Wrote {RESULTS / 'q12_metadata_probe.csv'}")
    print(f"  Wrote {RESULTS / 'q12_metadata_probe.json'}")

    # ── Figure ───────────────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.5))

    # GTEx panel — heatmap of metric per latent × target (R² for continuous,
    # macro AUC for categorical)
    pivot_rows = []
    for latent_name, _ in latents:
        for t in cont_targets:
            v = df_out[
                (df_out.cohort == "GTEx")
                & (df_out.latent == latent_name)
                & (df_out.target == t)
                & (df_out.metric == "R2")
            ]["value"].iloc[0]
            pivot_rows.append({"latent": latent_name, "target": t,
                               "value": v, "metric": "R²"})
        for t in cat_targets:
            v = df_out[
                (df_out.cohort == "GTEx")
                & (df_out.latent == latent_name)
                & (df_out.target == t)
                & (df_out.metric == "macro_auc")
            ]["value"].iloc[0]
            pivot_rows.append({"latent": latent_name, "target": t,
                               "value": v, "metric": "AUC"})
    pv = pd.DataFrame(pivot_rows).pivot(
        index="latent", columns="target", values="value"
    )
    pv = pv.reindex([n for n, _ in latents])
    pv = pv[cont_targets + cat_targets]
    im = axes[0].imshow(pv.values, aspect="auto", cmap="RdBu_r",
                        vmin=-0.3, vmax=0.9)
    axes[0].set_xticks(range(pv.shape[1]))
    axes[0].set_xticklabels(pv.columns, rotation=20, ha="right")
    axes[0].set_yticks(range(pv.shape[0]))
    axes[0].set_yticklabels(pv.index)
    axes[0].set_title("Q12a — GTEx metadata recovery (R² for continuous, "
                      "macro-AUC for categorical)")
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            if pd.notna(v):
                axes[0].text(
                    j, i, f"{v:+.2f}" if v < 0 else f"{v:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > 0.5 else "black",
                )
    plt.colorbar(im, ax=axes[0])

    # GSE panel — bar grouped by target
    gse_targets_names = [t for t, _ in gse_targets]
    n_t = len(gse_targets_names)
    n_l = len(gse_latents)
    bar_w = 0.18
    xs = np.arange(n_t)
    colors = {"PCA-50": "#58a6ff", "Rand-64": "#7d8590",
              "CM-VAE": "#f78166", "Sane-VAE": "#3fb950"}
    for k, (lname, _) in enumerate(gse_latents):
        ys = []
        for t, kind in gse_targets:
            metric = "R2" if kind == "continuous" else "macro_auc"
            v = df_out[
                (df_out.cohort == "GSE279480")
                & (df_out.latent == lname)
                & (df_out.target == t)
                & (df_out.metric == metric)
            ]["value"].iloc[0]
            ys.append(v if pd.notna(v) else 0.0)
        axes[1].bar(xs + (k - (n_l - 1) / 2) * bar_w, ys, width=bar_w,
                    label=lname, color=colors.get(lname, "#999"))
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(["AGE_cat (AUC)", "SEX (AUC)", "CMV (AUC)"])
    axes[1].axhline(0.5, color="#7d8590", ls=":", lw=0.5,
                    label="chance (AUC=0.5)")
    axes[1].set_title("Q12b — GSE279480 cross-cohort metadata probe (encoder "
                      "trained on GTEx, applied to independent cohort)")
    axes[1].set_ylabel("Macro-AUC")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(FIGS / "q12_metadata_probe.png", dpi=150)
    plt.close(fig)
    print(f"  Wrote {FIGS / 'q12_metadata_probe.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
