"""Q47 — Disentangled VAE at 11,374 shared genes with capacity-weighted head punishment.

New features vs Q20:
  1. Loads the local 11,374-gene matrix (data/processed_11k/).
  2. Per-field MLP prediction heads (Linear→GELU→Linear) on each z_meta slot.
  3. Capacity-weighted KL for z_meta: slots that fail to predict their metadata
     are penalised (KL weight rises), earning capacity only through prediction.
  4. z_bio capped at 32 dims (only ~30 PCs carry stable signal per Q7/Q8).
  5. Flip test + round-trip validation on the 20 % held-out GTEx set.

Loss:
  L = MSE(x, x̂)
    + β_bio · KL(z_bio; free-bits)         # z_bio: capped at 32 dims
    + β_meta · Σ_k w_k · KL_k(z_meta)     # w_k = 1 + λ_cap·head_loss_k (punish)
    + λ_sup · Σ_k masked_head_loss_k       # auxiliary heads, NaN-masked
    + λ_leak · HSIC(z_bio, modality)       # keep modality out of z_bio
    + λ_cycle · MSE(re-encode(flip).z_bio, z_bio)
    + λ_cycle_meta · BCE(re-encode(flip).mod_head, flipped_label)

Metadata definition (m vector, 4 fields × 2 z_meta dims each = 8 z_meta dims):
  m[:,0] = modality     (0=bulk, 1=sc/HCA)    → BCE head
  m[:,1] = ischemia     (z-scored SMTSISCH)   → MSE head
  m[:,2] = sex          (0/1)                 → BCE head
  m[:,3] = dthhrdy      (z-scored DTHHRDY)   → MSE head
  NaN where the annotation is unavailable; the head loss is masked out.

Usage:
  # First rebuild the 11K gene matrix (one-time, ~20 min on first run):
  python scripts/build_shared_gene_matrix.py

  # Train with default settings (Run-10 hyperparams from Q20 report):
  python analysis/q47_disentangled_train.py

  # Tune hyperparameters:
  python analysis/q47_disentangled_train.py \\
      --epochs 200 --tag run1 --lam_cap 2.0 --lam_cycle 0.3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.disentangled_vae import (
    DisentangledConfig,
    DisentangledVAE,
    capacity_weighted_kl_meta,
    kl_with_free_bits,
    kl_per_dim,
    supervised_loss,
    hsic_penalty,
)

OUT = ROOT / "analysis" / "results" / "q47_disentangled"
OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "analysis" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Data loading — all paths are local to the project.
# ---------------------------------------------------------------------------
DATA_11K = ROOT / "data" / "processed_11k"
DATA_2K = ROOT / "data" / "processed"          # fallback if 11K not built yet

# GTEx annotation paths: try local project first, then a sibling project cache
# (GTEx v8 annotations — sufficient for SMTSISCH/SEX/DTHHRDY coverage on
# the v11 cohort; any missing annotations become NaN and are masked in the loss).
_ANNOT_CANDIDATES = [
    ROOT / "data" / "annotations" / "GTEx_v10_Annotations_SampleAttributesDS.txt",
    ROOT / "data" / "annotations" / "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt",
    Path("/Users/stevotrujillo/Desktop/jepa_genotype_2_phenotype/test_data_cache/"
         "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"),
]
_SUBJ_CANDIDATES = [
    ROOT / "data" / "annotations" / "GTEx_v10_Annotations_SubjectPhenotypesDS.txt",
    ROOT / "data" / "annotations" / "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt",
    Path("/Users/stevotrujillo/Desktop/jepa_genotype_2_phenotype/test_data_cache/"
         "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt"),
]


def _find_first(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _load_data_dir() -> Path:
    """Pick the best available processed-data directory."""
    if (DATA_11K / "bulk_log_cpm.npy").exists():
        print(f"  [data] Using 11K gene matrix from {DATA_11K}")
        return DATA_11K
    print(
        f"  [data] WARNING: 11K matrix not found at {DATA_11K}\n"
        f"         Run: python scripts/build_shared_gene_matrix.py\n"
        f"         Falling back to 2K matrix at {DATA_2K}"
    )
    return DATA_2K


def _hca_counts_to_log_cpm(counts: np.ndarray) -> np.ndarray:
    """Convert raw pseudobulk count matrix → log2(CPM+1).

    counts: (n_pseudobulks, n_genes) raw read counts.
    """
    lib = counts.sum(axis=1, keepdims=True)
    lib = np.maximum(lib, 1.0)
    return np.log2(counts / lib * 1e6 + 1).astype(np.float32)


def _load_gtex_metadata(sample_ids: np.ndarray) -> pd.DataFrame:
    """Load GTEx subject + sample annotations.  Returns DataFrame with columns
    SEX (0/1), SMTSISCH (float), DTHHRDY (float), SMRIN (float).

    Falls back gracefully to all-NaN when annotation files are not available.
    """
    annot_path = _find_first(_ANNOT_CANDIDATES)
    subj_path = _find_first(_SUBJ_CANDIDATES)

    if annot_path is None or subj_path is None:
        print(
            "  [metadata] GTEx annotation files not found — running in modality-only mode.\n"
            "  Ischemia / sex / hardy-class heads will have NaN targets (masked in loss).\n"
            "  To enable full metadata: copy GTEx v10 annotations to data/annotations/"
        )
        df = pd.DataFrame({"SAMPID": sample_ids})
        for col in ["SEX", "SMTSISCH", "DTHHRDY", "SMRIN"]:
            df[col] = np.nan
        return df.set_index("SAMPID").loc[list(sample_ids)].reset_index()

    print(f"  [metadata] Loading annotations from {annot_path.name} + {subj_path.name}")
    subj = pd.read_csv(subj_path, sep="\t")
    samp = pd.read_csv(annot_path, sep="\t", low_memory=False)

    df = pd.DataFrame({"SAMPID": sample_ids})
    df["SUBJID"] = df["SAMPID"].apply(lambda s: "-".join(s.split("-")[:2]))
    df = df.merge(subj[["SUBJID", "SEX", "DTHHRDY"]], on="SUBJID", how="left")
    cols = ["SAMPID"] + [c for c in ["SMRIN", "SMTSISCH"] if c in samp.columns]
    df = df.merge(samp[cols], on="SAMPID", how="left")
    return df.set_index("SAMPID").loc[list(sample_ids)].reset_index()


def build_data(data_dir: Path | None = None) -> dict:
    """Load GTEx bulk + HCA pseudobulks → train-ready arrays.

    Returns:
      x_gtex:      (n_gtex, n_genes) standardised log-CPM
      m_gtex:      (n_gtex, 4)  [modality=0, ischemia_z, sex, dthhrdy_z]
      id_gtex:     (n_gtex,) SAMPID strings
      x_hca:       (n_hca, n_genes) standardised log-CPM
      m_hca:       (n_hca, 4)  [modality=1, isan=NaN, sex=NaN, dthhrdy=NaN]
      id_hca:      (n_hca,) "donor|celltype" strings
      shared_genes, scaler_mean, scaler_std (all fitted on GTEx only)
      isch_mean, isch_std, dthh_mean, dthh_std (for un-z-scoring)
    """
    if data_dir is None:
        data_dir = _load_data_dir()

    bulk_log = np.load(data_dir / "bulk_log_cpm.npy").astype(np.float32)  # (n, genes)
    sample_ids = np.loadtxt(data_dir / "bulk_sample_ids.txt", dtype=str)
    genes_df = pd.read_csv(data_dir / "gene_metadata.tsv", sep="\t")
    shared_genes = genes_df["gene_symbol"].values.astype(str)

    # GTEx metadata
    md = _load_gtex_metadata(sample_ids)
    sex_b = md["SEX"].apply(
        lambda v: float(v) - 1.0 if pd.notna(v) else np.nan
    ).values.astype(np.float32)  # gtex encodes 1=male,2=female → 0/1

    isch = md["SMTSISCH"].astype(float).values if "SMTSISCH" in md else np.full(len(sample_ids), np.nan)
    dthh = md["DTHHRDY"].astype(float).values if "DTHHRDY" in md else np.full(len(sample_ids), np.nan)

    isch_mean = float(np.nanmean(isch)) if np.any(np.isfinite(isch)) else 0.0
    isch_std = float(np.nanstd(isch)) if np.any(np.isfinite(isch)) else 1.0
    dthh_mean = float(np.nanmean(dthh)) if np.any(np.isfinite(dthh)) else 0.0
    dthh_std = float(np.nanstd(dthh)) if np.any(np.isfinite(dthh)) else 1.0

    isch_z = np.where(np.isfinite(isch), (isch - isch_mean) / isch_std, np.nan).astype(np.float32)
    dthh_z = np.where(np.isfinite(dthh), (dthh - dthh_mean) / dthh_std, np.nan).astype(np.float32)

    m_gtex = np.column_stack([
        np.zeros(len(sample_ids), dtype=np.float32),  # modality = 0 (bulk)
        isch_z,
        sex_b,
        dthh_z,
    ]).astype(np.float32)

    # HCA pseudobulk: raw counts → log-CPM
    pb_counts = np.load(data_dir / "hca_pseudobulk_counts_by_donor_celltype.npy")
    hca_log = _hca_counts_to_log_cpm(pb_counts.astype(np.float64))  # (n_hca, genes)
    pb_meta = pd.read_csv(data_dir / "hca_pseudobulk_donor_celltype_metadata.tsv", sep="\t")
    donor_col = pb_meta.columns[0]
    ct_col = pb_meta.columns[1]
    hca_ids = np.asarray(
        [f"{d}|{c}" for d, c in zip(pb_meta[donor_col], pb_meta[ct_col])],
        dtype=str,
    )

    # Fit scaler on GTEx train split (first 80% after shuffle) to avoid data
    # leakage. We use all GTEx here and the caller re-standardises on train-only
    # after the split — but we pre-compute a reasonable scaler now for alignment.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(sample_ids))
    n_val = int(0.2 * len(sample_ids))
    train_idx = perm[n_val:]

    scaler_mean = bulk_log[train_idx].mean(axis=0, keepdims=True)
    scaler_std = np.maximum(bulk_log[train_idx].std(axis=0, keepdims=True), 1e-8)

    x_gtex = ((bulk_log - scaler_mean) / scaler_std).astype(np.float32)
    x_hca = ((hca_log - scaler_mean) / scaler_std).astype(np.float32)

    m_hca = np.full((len(hca_ids), 4), np.nan, dtype=np.float32)
    m_hca[:, 0] = 1.0  # modality = 1 (sc)

    n_meta_known = int(np.sum(np.isfinite(m_gtex[:, 1:])))
    print(
        f"  [data] GTEx: {len(sample_ids)} samples × {bulk_log.shape[1]} genes  "
        f"(metadata-known fields: {n_meta_known}/{len(sample_ids)*3})\n"
        f"  [data] HCA:  {len(hca_ids)} pseudobulks × {hca_log.shape[1]} genes"
    )

    return {
        "x_gtex": x_gtex,
        "m_gtex": m_gtex,
        "id_gtex": sample_ids,
        "x_hca": x_hca,
        "m_hca": m_hca,
        "id_hca": hca_ids,
        "shared_genes": shared_genes,
        "scaler_mean": scaler_mean.squeeze(0),
        "scaler_std": scaler_std.squeeze(0),
        "isch_mean": isch_mean,
        "isch_std": isch_std,
        "dthh_mean": dthh_mean,
        "dthh_std": dthh_std,
        "_perm": perm,   # same permutation used for scaler; training loop re-uses it
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    model: DisentangledVAE,
    x_train: np.ndarray,
    m_train: np.ndarray,
    x_val: np.ndarray,
    m_val: np.ndarray,
    *,
    epochs: int = 200,
    batch: int = 64,
    lr: float = 1e-3,
    beta_bio: float = 1e-3,    # low β: don't chase perfect reconstruction
    beta_meta: float = 1e-4,
    lam_sup: float = 1.0,
    lam_leak: float = 0.3,
    free_bits: float = 0.5,
    lam_cycle: float = 0.3,
    lam_cycle_meta: float | None = None,
    lam_cap: float | None = None,  # override config default
    # Per-field cap weights.  If provided, overrides lam_cap for each field.
    # Default: full punishment for modality/ischemia/sex; softer for DTHHRDY
    # (noisy ordinal signal; uniform punishment traps it in a vicious cycle).
    lam_cap_per_field: list[float] | None = None,
) -> dict:
    """Train the model; returns history dict."""
    if lam_cycle_meta is None:
        lam_cycle_meta = lam_cycle
    if lam_cap is None:
        lam_cap = model.config.lam_cap
    if lam_cap_per_field is None:
        # Default: same as lam_cap for modality/ischemia/sex, reduced for DTHHRDY.
        lam_cap_per_field = [lam_cap, lam_cap, lam_cap, lam_cap * 0.3]

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(m_train)),
        batch_size=batch,
        shuffle=True,
        drop_last=False,
    )
    x_val_t = torch.from_numpy(x_val).to(DEVICE)
    m_val_t = torch.from_numpy(m_val).to(DEVICE)
    kinds = model.config.meta_kinds
    meta_dims = model.config.meta_dims

    history: dict = {
        "epoch": [], "train_loss": [], "val_recon": [],
        "val_active_bio": [], "head_acc": [], "val_active_meta": [],
    }

    for ep in range(1, epochs + 1):
        model.train()
        ep_losses: list[float] = []
        for x_b, m_b in loader:
            x_b = x_b.to(DEVICE)
            m_b = m_b.to(DEVICE)

            x_hat, mu_m, lv_m, mu_b, lv_b, z_m, z_b = model(x_b)

            # Reconstruction — not the dominant objective (beta_bio is small).
            recon = nn.functional.mse_loss(x_hat, x_b)

            # z_bio KL with free-bits floor capping unused dims.
            kl_b = kl_with_free_bits(mu_b, lv_b, free_bits) / x_b.size(0)

            # Supervised auxiliary heads (NaN-masked per field).
            preds = model.head_predictions(z_m)
            sup, _, per_field_losses = supervised_loss(
                preds, [m_b[:, i] for i in range(m_b.size(1))], kinds
            )

            # Capacity-weighted KL for z_meta: punish dims that fail prediction.
            kl_m = capacity_weighted_kl_meta(
                mu_m, lv_m, per_field_losses, meta_dims, lam_cap, lam_cap_per_field
            )

            # HSIC leak penalty: keep modality OUT of z_bio only.
            mod_col = m_b[:, 0:1]
            row_ok = torch.isfinite(mod_col[:, 0])
            leak = (
                hsic_penalty(z_b[row_ok], mod_col[row_ok])
                if row_ok.sum() >= 8
                else z_b.new_zeros(())
            )

            # Cycle consistency: flip z_meta[modality] and re-encode.
            cycle_bio = z_b.new_zeros(())
            cycle_meta = z_b.new_zeros(())
            if lam_cycle > 0 and row_ok.sum() >= 4:
                bulk_mask = mod_col[:, 0] < 0.5
                d_mod = meta_dims[0]
                # Full d_mod-dimensional class centroids as flip targets.
                # Setting the ENTIRE modality subspace (not just dim 0) ensures
                # the decoder sees a coherent class signal and Test B works.
                if bulk_mask.sum() > 0 and (~bulk_mask).sum() > 0:
                    bulk_centroid = z_m[:, :d_mod][bulk_mask].mean(dim=0).detach()  # (d_mod,)
                    sc_centroid   = z_m[:, :d_mod][~bulk_mask].mean(dim=0).detach()
                else:
                    # Fall back to zeros (prior centre) when only one class in batch.
                    bulk_centroid = z_m.new_zeros(d_mod)
                    sc_centroid   = z_m.new_zeros(d_mod)

                # Each sample flips to the OTHER class full centroid.
                target_centroid = torch.where(
                    bulk_mask.unsqueeze(1).expand(-1, d_mod),
                    sc_centroid.unsqueeze(0).expand(z_m.size(0), -1),
                    bulk_centroid.unsqueeze(0).expand(z_m.size(0), -1),
                )
                z_m_flip = z_m.clone()
                z_m_flip[:, :d_mod] = target_centroid

                x_flipped = model.decode(z_m_flip, z_b)
                mu_m_re, _, mu_b_re, _ = model.encode(x_flipped)

                # z_bio must survive the modality flip (biology preserved).
                cycle_bio = nn.functional.mse_loss(mu_b_re, mu_b.detach())

                # Modality head of re-encoded output should predict the FLIPPED class.
                flipped_label = torch.where(
                    bulk_mask, torch.ones_like(bulk_mask, dtype=torch.float32),
                    torch.zeros_like(bulk_mask, dtype=torch.float32)
                )
                logit_re = model.heads[0](mu_m_re[:, :d_mod]).squeeze(-1)
                cycle_meta = nn.functional.binary_cross_entropy_with_logits(
                    logit_re, flipped_label
                )

            loss = (
                recon
                + beta_bio * kl_b
                + beta_meta * kl_m
                + lam_sup * sup
                + lam_leak * leak
                + lam_cycle * cycle_bio
                + lam_cycle_meta * cycle_meta
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            ep_losses.append(loss.item())

        # --- validation ---
        model.eval()
        with torch.no_grad():
            x_hat_v, mu_m_v, lv_m_v, mu_b_v, _, z_m_v, _ = model(x_val_t)
            v_recon = nn.functional.mse_loss(x_hat_v, x_val_t).item()
            active_bio = int((mu_b_v.var(dim=0) > 0.01).sum().item())
            active_meta = int((mu_m_v.var(dim=0) > 0.01).sum().item())
            preds_v = model.head_predictions(z_m_v)
            head_metrics: list[float] = []
            for i, (pv, kind) in enumerate(zip(preds_v, kinds)):
                y = m_val_t[:, i]
                mk = torch.isfinite(y)
                if not mk.any():
                    head_metrics.append(float("nan"))
                    continue
                if kind == "bce":
                    acc = ((torch.sigmoid(pv[mk]) > 0.5).float() == y[mk]).float().mean().item()
                    head_metrics.append(acc)
                else:
                    sst = ((y[mk] - y[mk].mean()) ** 2).sum().item()
                    sse = ((pv[mk] - y[mk]) ** 2).sum().item()
                    head_metrics.append(1.0 - sse / max(sst, 1e-12))

        history["epoch"].append(ep)
        history["train_loss"].append(float(np.mean(ep_losses)))
        history["val_recon"].append(v_recon)
        history["val_active_bio"].append(active_bio)
        history["val_active_meta"].append(active_meta)
        history["head_acc"].append(head_metrics)

        if ep % 10 == 0 or ep in (1, epochs):
            heads_fmt = [f"{h:.2f}" if not np.isnan(h) else "nan" for h in head_metrics]
            print(
                f"ep{ep:3d}  train={np.mean(ep_losses):.4f}  "
                f"v_recon={v_recon:.4f}  "
                f"act_bio={active_bio}/{model.config.z_bio_dim}  "
                f"act_meta={active_meta}/{model.config.z_meta_dim}  "
                f"heads={heads_fmt}",
                flush=True,
            )

    return history


# ---------------------------------------------------------------------------
# Flip-test validation on held-out set
# ---------------------------------------------------------------------------
def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a**2).sum() * (b**2).sum()) + 1e-12))


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return 1.0 - an @ bn.T


def flip_test(
    model: DisentangledVAE,
    data: dict,
    val_idx: np.ndarray,
    *,
    k_top: int = 200,
    n_balance: int = 8,
) -> dict:
    """Metadata-flip test + round-trip test on held-out GTEx samples.

    Test A — modality flip (GTEx bulk → sc):
      Encode GTEx val bulk → flip z_meta[modality] to sc class mean →
      decode → NN-accuracy in a balanced pool of GTEx + HCA samples.
      A working model should get >> 50 % (baseline ≈ 5 %).

    Test B — reverse modality flip (HCA sc → bulk):
      Encode HCA pseudobulks → flip z_meta[modality] to bulk class mean →
      decode → NN should land in the GTEx cluster.

    Test C — round-trip (bulk → sc → bulk):
      Encode GTEx val → flip to sc → re-encode → flip back to bulk →
      Pearson correlation with the original input.  If biology survived the
      modality detour, round-trip Pearson should be > 0.7.
      Also: cosine similarity between the original z_bio and the re-encoded
      z_bio after the flip (z_bio cycle cosine ≥ 0.9 = biology preserved).
    """
    model.eval()
    x_gtex_val = data["x_gtex"][val_idx]
    x_hca = data["x_hca"]

    x_gtex_t = torch.from_numpy(x_gtex_val).to(DEVICE)
    x_hca_t = torch.from_numpy(x_hca).to(DEVICE)

    with torch.no_grad():
        mu_m_g, _, mu_b_g, _ = model.encode(x_gtex_t)
        mu_m_h, _, mu_b_h, _ = model.encode(x_hca_t)

        # Empirical class centroids over the full d_mod modality subspace.
        # Using all d_mod dims (not just dim 0) ensures the decoder sees a
        # coherent class signal in both flip directions (fixes Test B = 0.0).
        d_mod = model.config.meta_dims[0]
        bulk_centroid = mu_m_g[:, :d_mod].mean(dim=0)  # (d_mod,) on DEVICE
        sc_centroid   = mu_m_h[:, :d_mod].mean(dim=0)
        bulk_z0 = float(bulk_centroid[0].item())  # dim-0 scalar for logging only
        sc_z0   = float(sc_centroid[0].item())
        print(f"  z_meta[modality]: bulk_centroid[0]={bulk_z0:.3f}  sc_centroid[0]={sc_z0:.3f}")

        # Original reconstructions (no flip — sanity-check).
        x_orig = model.decode(mu_m_g, mu_b_g).cpu().numpy()

        # Flip bulk → sc: set entire d_mod subspace to sc centroid.
        z_m_g_flip = mu_m_g.clone()
        z_m_g_flip[:, :d_mod] = sc_centroid.unsqueeze(0)
        x_sc = model.decode(z_m_g_flip, mu_b_g).cpu().numpy()

        # Flip HCA sc → bulk: set entire d_mod subspace to bulk centroid.
        z_m_h_flip = mu_m_h.clone()
        z_m_h_flip[:, :d_mod] = bulk_centroid.unsqueeze(0)
        x_bulk_from_hca = model.decode(z_m_h_flip, mu_b_h).cpu().numpy()

    # --- Test A: balanced NN accuracy (GTEx→sc flip) ---
    rng = np.random.default_rng(0)
    n_v = len(val_idx)
    nn_corr_flip, nn_corr_orig = [], []
    for q in range(n_v):
        for _ in range(50):
            other = [j for j in range(n_v) if j != q]
            sample_g = rng.choice(other, size=min(n_balance, len(other)), replace=False)
            pool = np.concatenate([x_gtex_val[sample_g], x_hca], axis=0)
            labels = np.concatenate([
                np.zeros(len(sample_g), dtype=int),
                np.ones(len(x_hca), dtype=int),
            ])
            d_flip = _cosine_dist(x_sc[q:q+1], pool)[0]
            d_orig = _cosine_dist(x_orig[q:q+1], pool)[0]
            nn_corr_flip.append(int(labels[d_flip.argmin()] == 1))
            nn_corr_orig.append(int(labels[d_orig.argmin()] == 1))
    nn_acc_to_sc = float(np.mean(nn_corr_flip))
    nn_baseline = float(np.mean(nn_corr_orig))

    # --- Test B: balanced NN accuracy (HCA→bulk flip) ---
    nn_corr_h = []
    for q in range(len(x_hca)):
        other_h = [j for j in range(len(x_hca)) if j != q]
        for _ in range(50):
            sample_g = rng.choice(n_v, size=n_balance, replace=False)
            pool = np.concatenate([x_gtex_val[sample_g], x_hca[other_h]], axis=0)
            labels = np.concatenate([
                np.zeros(len(sample_g), dtype=int),
                np.ones(len(other_h), dtype=int),
            ])
            d = _cosine_dist(x_bulk_from_hca[q:q+1], pool)[0]
            nn_corr_h.append(int(labels[d.argmin()] == 0))
    nn_acc_to_bulk = float(np.mean(nn_corr_h))

    # --- Test C: round-trip (bulk→sc→bulk) ---
    with torch.no_grad():
        x_sc_t = torch.from_numpy(x_sc).to(DEVICE)
        mu_m_re, _, mu_b_re, _ = model.encode(x_sc_t)
        z_m_back = mu_m_re.clone()
        z_m_back[:, :d_mod] = bulk_centroid.unsqueeze(0)  # full centroid flip back
        x_round = model.decode(z_m_back, mu_b_re).cpu().numpy()

    round_corrs = [_pearson(x_round[i], x_gtex_val[i]) for i in range(n_v)]

    z_bio_orig_np = mu_b_g.cpu().numpy()
    z_bio_re_np = mu_b_re.cpu().numpy()
    z_bio_cosine = float(np.mean([
        float(
            (z_bio_orig_np[i] @ z_bio_re_np[i])
            / (np.linalg.norm(z_bio_orig_np[i]) * np.linalg.norm(z_bio_re_np[i]) + 1e-12)
        )
        for i in range(n_v)
    ]))

    # Biology-preservation Pearson (on non-modality-marker genes).
    bulk_mean = data["x_gtex"].mean(axis=0)
    sc_mean = data["x_hca"].mean(axis=0)
    lfc = np.abs(bulk_mean - sc_mean)
    meta_genes = np.argsort(lfc)[-k_top:]
    bio_mask = np.ones(x_gtex_val.shape[1], dtype=bool)
    bio_mask[meta_genes] = False
    bio_corrs = [_pearson(x_sc[i, bio_mask], x_gtex_val[i, bio_mask]) for i in range(n_v)]

    # Cosine-distance diagnostics.
    d_orig_g = _cosine_dist(x_orig, x_gtex_val).min(axis=1).mean()
    d_orig_h = _cosine_dist(x_orig, x_hca).min(axis=1).mean()
    d_flip_g = _cosine_dist(x_sc, x_gtex_val).min(axis=1).mean()
    d_flip_h = _cosine_dist(x_sc, x_hca).min(axis=1).mean()

    val_recon_mse = float(((x_orig - x_gtex_val) ** 2).mean())

    results = {
        "testA_nn_acc_flip_to_sc_BALANCED": nn_acc_to_sc,
        "testA_nn_baseline_orig_to_sc": nn_baseline,
        "testB_nn_acc_flip_to_bulk_HCA_BALANCED": nn_acc_to_bulk,
        "testC_round_trip_pearson": float(np.mean(round_corrs)),
        "testC_z_bio_cycle_cosine": z_bio_cosine,
        "biology_preservation_pearson": float(np.mean(bio_corrs)),
        "cosine_orig_to_gtex_pool": float(d_orig_g),
        "cosine_orig_to_hca_pool": float(d_orig_h),
        "cosine_flip_to_gtex_pool": float(d_flip_g),
        "cosine_flip_to_hca_pool": float(d_flip_h),
        "val_recon_mse": val_recon_mse,
        "n_val_gtex": int(n_v),
        "n_hca_pseudobulks": int(len(x_hca)),
        "z_meta_modality_bulk_mean": bulk_z0,
        "z_meta_modality_sc_mean": sc_z0,
    }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(
    epochs: int = 200,
    run_tag: str = "run1",
    data_dir: Path | None = None,
    **train_kwargs,
) -> dict:
    print(f"[Q47] Loading data — device={DEVICE}")
    data = build_data(data_dir)

    n_gtex = data["x_gtex"].shape[0]
    # Reuse the same permutation that was used to fit the scaler.
    perm = data["_perm"]
    n_val = int(0.2 * n_gtex)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    # Oversample HCA to balance the modalities.
    n_hca = data["x_hca"].shape[0]
    hca_repeat = max(1, len(train_idx) // n_hca)
    x_hca_rep = np.tile(data["x_hca"], (hca_repeat, 1))
    m_hca_rep = np.tile(data["m_hca"], (hca_repeat, 1))

    x_train = np.concatenate([data["x_gtex"][train_idx], x_hca_rep], axis=0)
    m_train = np.concatenate([data["m_gtex"][train_idx], m_hca_rep], axis=0)
    x_val = data["x_gtex"][val_idx]
    m_val = data["m_gtex"][val_idx]

    print(
        f"  train: GTEx {len(train_idx)} + HCA {n_hca}×{hca_repeat}={n_hca*hca_repeat}"
        f" = {x_train.shape[0]} samples\n"
        f"  val:   GTEx {len(val_idx)} samples  (20 % held-out)\n"
        f"  genes: {x_train.shape[1]}"
    )

    cfg = DisentangledConfig(input_dim=x_train.shape[1])
    model = DisentangledVAE(cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"  params: {n_params:,}  "
        f"latent: z_meta={cfg.z_meta_dim} ({cfg.meta_dims} per field) + z_bio={cfg.z_bio_dim}"
    )

    history = train(model, x_train, m_train, x_val, m_val, epochs=epochs, **train_kwargs)

    print("\n[Q47] Running flip tests on 20 % held-out set …")
    flip_results = flip_test(model, data, val_idx)
    for k, v in flip_results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    out = {
        "history": history,
        "flip": flip_results,
        "config": {
            "z_bio_dim": cfg.z_bio_dim,
            "z_meta_dim": cfg.z_meta_dim,
            "meta_dims": cfg.meta_dims,
            "meta_hidden": list(cfg.meta_hidden),
            "meta_fields": cfg.meta_fields,
            "epochs": epochs,
            "n_genes": x_train.shape[1],
            **train_kwargs,
        },
    }
    out_path = OUT / f"q47_{run_tag}.json"
    with out_path.open("w") as fh:
        json.dump(out, fh, indent=2)

    ckpt_path = OUT / f"q47_{run_tag}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg.__dict__,
            "shared_genes": data["shared_genes"],
            "scaler_mean": data["scaler_mean"],
            "scaler_std": data["scaler_std"],
        },
        ckpt_path,
    )
    print(f"\n  Saved: {out_path.name}  {ckpt_path.name}")
    return flip_results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train disentangled VAE at 11K genes.")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--tag", default="run1")
    p.add_argument("--data-dir", type=Path, default=None, help="Override data directory.")
    p.add_argument("--beta-bio", type=float, default=1e-3, dest="beta_bio")
    p.add_argument("--beta-meta", type=float, default=1e-4, dest="beta_meta")
    p.add_argument("--lam-sup", type=float, default=1.0, dest="lam_sup")
    p.add_argument("--lam-leak", type=float, default=0.3, dest="lam_leak")
    p.add_argument("--free-bits", type=float, default=0.5, dest="free_bits")
    p.add_argument("--lam-cycle", type=float, default=0.3, dest="lam_cycle")
    p.add_argument("--lam-cycle-meta", type=float, default=None, dest="lam_cycle_meta")
    p.add_argument("--lam-cap", type=float, default=1.0, dest="lam_cap",
                   help="Capacity-penalty multiplier for modality/ischemia/sex heads.")
    p.add_argument("--lam-cap-dthhrdy", type=float, default=0.3, dest="lam_cap_dthhrdy",
                   help="Cap weight for DTHHRDY head only (default 0.3, softer than others).")
    a = p.parse_args()
    lam_cap_per_field = [a.lam_cap, a.lam_cap, a.lam_cap, a.lam_cap_dthhrdy]
    main(
        epochs=a.epochs,
        run_tag=a.tag,
        data_dir=a.data_dir,
        beta_bio=a.beta_bio,
        beta_meta=a.beta_meta,
        lam_sup=a.lam_sup,
        lam_leak=a.lam_leak,
        free_bits=a.free_bits,
        lam_cycle=a.lam_cycle,
        lam_cycle_meta=a.lam_cycle_meta,
        lam_cap=a.lam_cap,
        lam_cap_per_field=lam_cap_per_field,
    )
