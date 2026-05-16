"""Q56 — Metadata ↔ Embedding bidirectional mapping.

Hypothesis: z_bio encodes biological "states" that are predictable from
donor/sample metadata, and conversely each z_bio dim corresponds to a
measurable biological or technical factor.

This experiment tests both directions:
  A. metadata → z_bio  (can metadata predict the embedding?)
  B. z_bio → metadata  (which dims drive which metadata variables?)

For direction A, we train a small MLP and compare to Ridge (linear baseline).
  High R² means: knowing who a donor is (their metadata profile) tells you
  what biological state they're in.

For direction B, we run per-variable cross-validated Ridge probes on the full
z_bio and per-dimension (to build a "metadata atlas" of the latent space).

GTEx metadata used (expanded from our usual 4 to all available variables):
  Continuous:  SMTSISCH, SMRIN, SMRDLGTH, AGE_mid, DTHHRDY, SMMPPD (reads)
  Categorical: SEX, SMCENTER (collection site), SMNABTCH (NA isolation batch)
  Derived:     SMTSISCH_z, SMRIN_z (z-scored versions used in the model)

Candidate additions for future work (not in GTEx):
  - Hour of blood draw (circadian)
  - BMI, sleep duration, diet, exercise (lifestyle)
  - Country/ancestry (population structure)
  See discussion at bottom of file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.data import load_gtex_blood, load_metadata, load_shared_genes
from models.meta_injection_vae import FiLMMetaInjectionVAE, FiLMMetaInjectionConfig

CKPT   = "/Users/rls/Desktop/programming-projects/single-cell/bulk-project/analysis/14_cross_modality_vae/cross_modality_vae.pt"
Q54B   = ROOT / "analysis" / "results" / "q54b_count_weighted" / "film_resid_vae_weighted.pt"
OUT    = ROOT / "analysis" / "results" / "q56_metadata_embedding"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED   = 0


# ── expanded metadata matrix ───────────────────────────────────────────────

def build_expanded_meta(meta_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build a richer metadata matrix using all available GTEx variables.

    Returns (X_meta, feature_names) where X_meta is (n, F) float32.
    Missing values → 0.  Categorical variables → z-scored integer codes.
    """
    rows = []

    def _zscore(series: pd.Series) -> np.ndarray:
        v = pd.to_numeric(series, errors="coerce").values.astype(float)
        z = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-8)
        return np.where(np.isfinite(z), z, 0.0).astype(np.float32)

    def _encode_cat(series: pd.Series) -> np.ndarray:
        """Label-encode a categorical, z-score the integer codes."""
        v = series.fillna("MISSING").astype(str).values
        le = LabelEncoder()
        codes = le.fit_transform(v).astype(float)
        z = (codes - codes.mean()) / (codes.std() + 1e-8)
        return z.astype(np.float32)

    features = {}

    # Continuous clinical / sample metadata
    for col in ["SMTSISCH", "SMRIN", "SMRDLGTH", "AGE_mid", "DTHHRDY"]:
        if col in meta_df.columns:
            features[col] = _zscore(meta_df[col])

    # Sex (binary, but z-scored for uniformity)
    if "SEX" in meta_df.columns:
        sex = pd.to_numeric(meta_df["SEX"], errors="coerce").values.astype(float)
        features["SEX"] = np.where(np.isfinite(sex), sex - 1.5, 0.0).astype(np.float32)

    # Categorical batch / site variables
    for col in ["SMCENTER", "SMNABTCH", "SMGEBTCH"]:
        if col in meta_df.columns:
            features[col + "_code"] = _encode_cat(meta_df[col])

    X = np.column_stack([v for v in features.values()]).astype(np.float32)
    names = list(features.keys())
    return X, names


# ── load z_bio ─────────────────────────────────────────────────────────────

def load_z_bio(X_sc: np.ndarray, M4: np.ndarray) -> np.ndarray:
    """Load Q54b model and encode X_sc → z_bio using ischemia residualization."""
    ckpt = torch.load(Q54B, map_location="cpu")
    cfg  = ckpt["cfg"]
    mc   = FiLMMetaInjectionConfig(
        input_dim      = cfg["input_dim"],
        meta_dim       = cfg["meta_dim"],
        z_bio_dim      = cfg["z_bio_dim"],
        decoder_hidden = tuple(cfg["decoder_hidden"]),
        meta_embed_dim = cfg["meta_embed_dim"],
        beta           = cfg["beta"],
        free_bits      = cfg["free_bits"],
        lambda_tc      = cfg["lambda_tc"],
    )
    model = FiLMMetaInjectionVAE(mc)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(DEVICE)

    resid_ckpt = ckpt["residualizer"]
    from analysis.q54_residual_encoding import IschemiaResidualizer
    resid = IschemiaResidualizer()
    resid.beta_isch = np.array(resid_ckpt["beta_isch"], dtype=np.float32)
    resid.s_mean    = float(resid_ckpt["s_mean"])

    X_resid = resid.transform(X_sc, M4[:, 0])
    with torch.no_grad():
        xt = torch.from_numpy(X_resid.astype(np.float32)).to(DEVICE)
        mu, _ = model.encode(xt)
    return mu.cpu().numpy()


# ── Direction B: z_bio → metadata (linear probes) ─────────────────────────

def probe_z_to_meta(z: np.ndarray, meta_df: pd.DataFrame,
                    X_meta: np.ndarray, meta_names: list[str]) -> dict:
    """For each metadata variable, cross-validated Ridge R² on full z_bio.
    Also fits per-dimension probes to find which z dim drives which variable.
    """
    results = {}
    n_dims  = z.shape[1]
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    z_sc = StandardScaler().fit_transform(z)

    continuous = [n for n in meta_names
                  if n not in ("SMCENTER_code", "SMNABTCH_code", "SMGEBTCH_code", "SEX")]
    categorical = ["SEX"] + [n for n in meta_names if n.endswith("_code")]

    print("  z_bio → metadata (Ridge / LR probes):")
    for i, name in enumerate(meta_names):
        y = X_meta[:, i]
        mask = np.isfinite(y)
        if mask.sum() < 20:
            continue
        y_m = y[mask]
        z_m = z_sc[mask]

        is_cat = name in categorical or name.endswith("_code")

        if is_cat:
            y_int = np.round(y_m).astype(int)
            n_cls = len(np.unique(y_int))
            if n_cls < 2:
                continue
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            try:
                sc = cross_val_score(
                    LogisticRegression(max_iter=500, C=1.0),
                    z_m, y_int, cv=skf, scoring="balanced_accuracy"
                )
                r2_full = float(sc.mean())
                r2_std  = float(sc.std())
                chance  = 1.0 / n_cls
                metric  = "balanced_acc"
            except Exception:
                continue
        else:
            sc = cross_val_score(Ridge(alpha=1.0), z_m, y_m, cv=kf, scoring="r2")
            r2_full = float(sc.mean())
            r2_std  = float(sc.std())
            chance  = 0.0
            metric  = "r2"

        # Per-dim probe
        per_dim = []
        for k in range(n_dims):
            zk = z_sc[mask, k:k+1]
            if is_cat:
                try:
                    s = cross_val_score(
                        LogisticRegression(max_iter=300, C=1.0),
                        zk, y_int, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                        scoring="balanced_accuracy"
                    )
                    per_dim.append(float(s.mean()))
                except Exception:
                    per_dim.append(float("nan"))
            else:
                s = cross_val_score(Ridge(alpha=1.0), zk, y_m,
                                    cv=kf, scoring="r2")
                per_dim.append(float(s.mean()))

        valid = [x for x in per_dim if np.isfinite(x)]
        sorted_pd = sorted(valid, reverse=True)
        mig_gap = sorted_pd[0] - sorted_pd[1] if len(sorted_pd) >= 2 else float("nan")
        best_dim = int(np.nanargmax(per_dim))

        results[name] = {
            "metric":    metric,
            "full_z":    r2_full,
            "full_z_std": r2_std,
            "chance":    chance,
            "above_chance": r2_full > chance + 0.02,
            "per_dim":   per_dim,
            "best_dim":  best_dim,
            "best_dim_score": sorted_pd[0] if sorted_pd else float("nan"),
            "mig_gap":   mig_gap,
        }

        above_flag = "✓" if results[name]["above_chance"] else " "
        print(f"  {above_flag} {name:<18} full={r2_full:>6.3f}±{r2_std:.3f}  "
              f"(chance={chance:.2f})  best_dim=z{best_dim+1}  "
              f"mig_gap={mig_gap:.3f}")

    return results


# ── Direction A: metadata → z_bio ─────────────────────────────────────────

class MetaToZ(nn.Module):
    """Small MLP: metadata vector → z_bio prediction."""
    def __init__(self, n_meta: int, n_z: int, hidden: tuple[int, ...] = (64, 64)):
        super().__init__()
        dims = [n_meta] + list(hidden) + [n_z]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [nn.LayerNorm(dims[i + 1]), nn.GELU()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_meta_to_z(X_meta: np.ndarray, z: np.ndarray,
                    epochs: int = 300) -> tuple[nn.Module, list[dict]]:
    """Train MLP: metadata → z_bio."""
    n_meta, n_z = X_meta.shape[1], z.shape[1]
    model = MetaToZ(n_meta, n_z).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    Xm = torch.from_numpy(X_meta.astype(np.float32))
    Zz = torch.from_numpy(z.astype(np.float32))
    dl = DataLoader(TensorDataset(Xm, Zz), batch_size=64, shuffle=True)

    log = []
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for xb, zb in dl:
            xb, zb = xb.to(DEVICE), zb.to(DEVICE)
            loss = nn.functional.mse_loss(model(xb), zb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += loss.item()
        sched.step()
        if ep % 100 == 0 or ep == 1:
            log.append({"epoch": ep, "mse": ep_loss / len(dl)})
            print(f"  [meta→z] ep {ep}/{epochs}  mse={ep_loss/len(dl):.4f}")

    return model, log


def cv_meta_to_z(X_meta: np.ndarray, z: np.ndarray) -> dict:
    """Cross-validated evaluation: metadata → z_bio.

    Uses both Ridge (linear) and MLP (non-linear) predictors.
    Returns R² per z dimension and overall.
    """
    results = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    z_sc = StandardScaler().fit_transform(z)
    x_sc = StandardScaler().fit_transform(X_meta)

    # ── Ridge (linear baseline) ───────────────────────────────────────────
    per_dim_ridge = []
    for k in range(z.shape[1]):
        sc = cross_val_score(Ridge(alpha=1.0), x_sc, z_sc[:, k],
                             cv=kf, scoring="r2")
        per_dim_ridge.append(float(sc.mean()))

    # Multi-output: predict all z dims jointly with one Ridge
    from sklearn.linear_model import RidgeCV
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])
    preds = np.zeros_like(z_sc)
    for tr, te in kf.split(x_sc):
        ridge.fit(x_sc[tr], z_sc[tr])
        preds[te] = ridge.predict(x_sc[te])
    ss_res = ((z_sc - preds) ** 2).sum()
    ss_tot = ((z_sc - z_sc.mean(0)) ** 2).sum()
    r2_joint_ridge = float(1 - ss_res / ss_tot)

    results["ridge"] = {
        "r2_per_dim":  per_dim_ridge,
        "r2_joint":    r2_joint_ridge,
        "r2_mean_dim": float(np.mean(per_dim_ridge)),
    }
    print(f"  Ridge: joint R²={r2_joint_ridge:.3f}  "
          f"per-dim mean={np.mean(per_dim_ridge):.3f}  "
          f"max={max(per_dim_ridge):.3f}")

    # ── Random Forest (non-linear) ───────────────────────────────────────
    per_dim_rf = []
    for k in range(z.shape[1]):
        sc = cross_val_score(
            RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
            x_sc, z_sc[:, k], cv=kf, scoring="r2"
        )
        per_dim_rf.append(float(sc.mean()))

    results["random_forest"] = {
        "r2_per_dim":  per_dim_rf,
        "r2_mean_dim": float(np.mean(per_dim_rf)),
        "r2_max_dim":  float(max(per_dim_rf)),
    }
    print(f"  RF:    per-dim mean={np.mean(per_dim_rf):.3f}  "
          f"max={max(per_dim_rf):.3f}")

    return results


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print("Loading data ...")
    gtex     = load_gtex_blood(checkpoint_path=CKPT)
    meta     = load_metadata(gtex.sample_ids)
    X_sc     = gtex.expr_scaled

    # Build 4-dim meta for model (to pass to residualiser)
    from analysis.q54_residual_encoding import build_meta_matrix
    M4 = build_meta_matrix(meta)

    # Load z_bio from best model (Q54b)
    print("Encoding z_bio with Q54b model ...")
    z_bio = load_z_bio(X_sc, M4)
    print(f"  z_bio shape: {z_bio.shape}  "
          f"active dims (var>0.01): {(z_bio.var(0) > 0.01).sum()}")

    # Build expanded metadata
    X_meta, meta_names = build_expanded_meta(meta)
    print(f"Expanded metadata: {len(meta_names)} variables  {X_meta.shape}")
    print(f"  Variables: {meta_names}")

    # ── Direction B: z_bio → metadata ─────────────────────────────────────
    print("\n══ Direction B: z_bio → metadata probes ══")
    b_results = probe_z_to_meta(z_bio, meta, X_meta, meta_names)

    # ── Direction A: metadata → z_bio ─────────────────────────────────────
    print("\n══ Direction A: metadata → z_bio ══")
    a_results = cv_meta_to_z(X_meta, z_bio)

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n══ Summary: which metadata variables are encoded in z_bio? ══")
    print(f"{'variable':<20}  {'z→meta':>7}  {'above chance':>13}  {'best dim':>8}  {'MIG gap':>8}")
    print("─" * 65)
    rows = sorted(b_results.items(), key=lambda kv: kv[1]["full_z"], reverse=True)
    for name, r in rows:
        fc = r["full_z"]
        chance = r["chance"]
        above  = "✓" if r["above_chance"] else "–"
        print(f"{name:<20}  {fc:>7.3f}  {above:>13}  z{r['best_dim']+1:>7}  "
              f"{r['mig_gap']:>8.3f}")

    print(f"\n── Metadata → z_bio (can metadata PREDICT the embedding?) ──")
    r_ridge = a_results["ridge"]
    r_rf    = a_results["random_forest"]
    print(f"  Ridge R² (joint, 5-fold CV):  {r_ridge['r2_joint']:.3f}")
    print(f"  Ridge R² (per-dim mean):      {r_ridge['r2_mean_dim']:.3f}")
    print(f"  RF R² (per-dim mean):         {r_rf['r2_mean_dim']:.3f}")
    print(f"  RF R² (per-dim max):          {r_rf['r2_max_dim']:.3f}")
    print()
    print(f"  Best dim Ridge R²: ", [(f"z{k+1}", f"{v:.3f}")
          for k, v in enumerate(r_ridge["r2_per_dim"])])

    print(f"\n  Interpretation:")
    joint_r2 = r_ridge['r2_joint']
    if joint_r2 < 0.1:
        print("    Metadata explains <10% of z_bio — most biology is unmeasured.")
    elif joint_r2 < 0.3:
        print("    Metadata explains 10-30% of z_bio — real signal but large unmeasured component.")
    elif joint_r2 < 0.6:
        print("    Metadata explains 30-60% of z_bio — metadata is a meaningful predictor of state.")
    else:
        print("    Metadata explains >60% of z_bio — the embedding is strongly metadata-driven.")

    # Save
    def _clean(obj):
        if isinstance(obj, dict): return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)): return float(obj)
        if isinstance(obj, list): return [_clean(v) for v in obj]
        return obj

    results = {
        "direction_B_z_to_meta":     _clean(b_results),
        "direction_A_meta_to_z":     _clean(a_results),
        "metadata_variables":        meta_names,
        "z_bio_active_dims":         int((z_bio.var(0) > 0.01).sum()),
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {OUT}/results.json")

    # Latent atlas: heatmap data (z_dim × metadata)
    atlas_rows = []
    for name, r in b_results.items():
        atlas_rows.append([float(x) for x in r["per_dim"]])
    atlas = pd.DataFrame(
        atlas_rows,
        index=list(b_results.keys()),
        columns=[f"z{k+1}" for k in range(z_bio.shape[1])]
    )
    atlas.to_csv(OUT / "latent_atlas.csv")
    print(f"Latent atlas (metadata × z_dim) → {OUT}/latent_atlas.csv")


if __name__ == "__main__":
    main()


# ──────────────────────────────────────────────────────────────────────────
# DISCUSSION: Datasets for richer metadata
# ──────────────────────────────────────────────────────────────────────────
#
# GTEx (current):
#   Variables: SMTSISCH, SMRIN, SMRDLGTH, AGE, SEX, DTHHRDY, SMCENTER, batch vars
#   Missing:   Lifestyle (sleep, diet, exercise), hour of draw, population ancestry,
#              BMI (some donors), medication history
#
# Better datasets for lifestyle metadata + blood RNA-seq:
#
# 1. ROSMAP (Rush Alzheimer's Disease Center, dbGaP phs000048)
#    - Brain tissue RNA-seq (not blood, but rich metadata)
#    - Has: cognitive test scores, sleep quality (Pittsburgh Sleep Quality Index),
#           diet (Mediterranean diet score), physical activity, depression, BMI
#    - ~1200 donors, many longitudinal
#    - Best for: aging states, cognitive reserve, sleep-expression links
#
# 2. GTEx v9 extended phenotype file
#    - Some donors have BMI, cause of death details, medical history (MHHTN, MHDBTS etc.)
#    - Access via dbGaP phs000424
#    - Not yet integrated in our pipeline
#
# 3. INTERVAL (Cambridge, UK)
#    - 50,000 blood donors with lifestyle questionnaire + some RNA-seq (subset)
#    - Has: exercise frequency, diet, sleep, alcohol, smoking, BMI
#    - RNA-seq available for subset (~2000)
#    - https://www.intervalstudy.org.uk
#
# 4. eQTLGen (Westra et al. 2015)
#    - 31,684 blood samples; primarily eQTL; limited lifestyle metadata
#    - But massive N → good for population structure effects
#
# 5. All of Us (NIH)
#    - Diverse US population; BMI, lifestyle, ancestry
#    - Whole-genome sequencing available; RNA-seq in progress
#    - Best for: diversity + lifestyle, but RNA-seq not yet widely accessible
#
# 6. UK Biobank blood RNA-seq (olink + some bulk)
#    - 500,000 participants; very rich lifestyle metadata
#    - RNA-seq available for subset ~5000
#    - Best for: sleep, diet, exercise, occupation, socioeconomic
#
# For circadian (hour of draw): ENCODE RNA-seq time series, or
#   Möller-Levet et al. 2013 (blood RNA-seq at multiple time points after
#   sleep restriction) — small N but directly tests circadian hypothesis.
#
# Recommended next step (Q57):
#   Download GTEx v8 extended phenotype file (phs000424.v8.p2),
#   merge MHHTN, MHDBTS, MHSMKSTS, BMI, ETHNCTY, DTHCOD into our metadata,
#   rerun Q56 with ~20 variables instead of 9.
