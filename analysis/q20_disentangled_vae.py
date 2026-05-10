"""Q20 — Disentangled VAE: train + run metadata-flip tests.

Loss: recon + β·KL_z_bio (free-bits) + γ·KL_z_meta + λ_sup·heads + λ_leak·HSIC.

Tests:
  1. synthetic flip (modality bulk→sc on GTEx held-out + ischemia 0→max):
       1.1  NN-correctness across (GTEx held-out + HCA)
       1.2  biology preservation: gene-correlation on non-metadata genes
  2. modality-flip on the few HCA donors: encode HCA, flip→bulk, NN should
     drift toward the GTEx cluster (since we have no donor-matched bulk).
"""
from __future__ import annotations

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
    kl_with_free_bits,
    kl_per_dim,
    supervised_loss,
    hsic_penalty,
)
from pipeline.data import load_gtex_blood, load_metadata, load_shared_genes  # noqa: E402

OUT = ROOT / "analysis" / "results" / "q20_disentangled"
OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "analysis" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)


# ── data ─────────────────────────────────────────────────────────────────
def _hca_pseudobulk_aligned(shared_genes: np.ndarray, scaler_mean, scaler_std,
                            kind: str = "celltype"):
    """Load HCA pseudobulk → log-CPM → align to shared_genes → standardise.

    kind="donor"     → 8 donor-level pseudobulks (legacy)
    kind="celltype"  → 113 (donor, celltype) pseudobulks built from
                       BL_standard_design.h5ad
    """
    if kind == "celltype":
        path = "/Users/rls/ecs271/data/sc/hca_celltype_pseudobulk.npz"
        d = np.load(path, allow_pickle=True)
        scaled = d["expr_scaled"].astype(np.float32)
        sample_ids = np.asarray([f"{donor}|{ct}" for donor, ct
                                 in zip(d["donor"], d["celltype"])], dtype=str)
        return scaled, sample_ids

    path = "/Users/rls/ecs271/data/sc/hca_blood_pseudobulk.npz"
    d = np.load(path, allow_pickle=True)
    expr = d["expr"].astype(np.float64)
    gene_names = np.asarray(d["gene_names"], dtype=str)
    sample_ids = np.asarray(d["sample_ids"], dtype=str)
    lib = expr.sum(axis=0, keepdims=True)
    log_cpm = np.log2(expr / lib * 1e6 + 1).T.astype(np.float32)
    name_to_idx = {n.upper(): i for i, n in enumerate(gene_names)}
    out = np.zeros((log_cpm.shape[0], len(shared_genes)), dtype=np.float32)
    for j, g in enumerate(shared_genes):
        idx = name_to_idx.get(str(g).upper())
        if idx is not None:
            out[:, j] = log_cpm[:, idx]
    scaled = ((out - scaler_mean) / scaler_std).astype(np.float32)
    return scaled, sample_ids


def build_data(include_melanoma: bool = False):
    g = load_gtex_blood()
    md = load_metadata(g.sample_ids)

    # GTEx metadata vector: [modality=0, ischemia, sex, dthhrdy]
    sex_b = md["SEX"].apply(lambda v: float(v) - 1 if pd.notna(v) else np.nan).values  # 1->0, 2->1
    isch = md["SMTSISCH"].astype(float).values
    isch_z = (isch - np.nanmean(isch)) / np.nanstd(isch)
    dthh = md["DTHHRDY"].astype(float).values
    dthh_z = (dthh - np.nanmean(dthh)) / np.nanstd(dthh)
    m_gtex = np.column_stack([
        np.zeros(len(g.sample_ids), dtype=np.float32),
        isch_z.astype(np.float32),
        sex_b.astype(np.float32),
        dthh_z.astype(np.float32),
    ])

    shared_genes, scaler_mean, scaler_std = load_shared_genes()
    hca_donor_x, hca_donor_ids = _hca_pseudobulk_aligned(
        shared_genes, scaler_mean, scaler_std, kind="donor")
    hca_ct_x, hca_ct_ids = _hca_pseudobulk_aligned(
        shared_genes, scaler_mean, scaler_std, kind="celltype")
    hca_x = np.concatenate([hca_donor_x, hca_ct_x], axis=0)
    hca_ids = np.concatenate([hca_donor_ids, hca_ct_ids])

    # Optional: melanoma PBMC bulk (59 donors) + sc per-(donor, cluster) pseudobulks
    # (82 groups from 9 donors, all NOT in Test-2 hold-out).
    extra_bulk_x = np.zeros((0, hca_x.shape[1]), dtype=np.float32)
    extra_bulk_ids = np.zeros((0,), dtype=str)
    extra_sc_x = np.zeros((0, hca_x.shape[1]), dtype=np.float32)
    extra_sc_ids = np.zeros((0,), dtype=str)
    if include_melanoma:
        d = np.load("/Users/rls/ecs271/data/sc/melanoma_bulk_train.npz", allow_pickle=True)
        extra_bulk_x = d["expr_scaled"].astype(np.float32)
        extra_bulk_ids = np.asarray([f"melB|{x}" for x in d["donor"]], dtype=str)
        d = np.load("/Users/rls/ecs271/data/sc/melanoma_sc_pb_train.npz", allow_pickle=True)
        extra_sc_x = d["expr_scaled"].astype(np.float32)
        extra_sc_ids = np.asarray(
            [f"melSC|{donor}|{cluster}" for donor, cluster in zip(d["donor"], d["cluster"])],
            dtype=str,
        )
        print(f"  melanoma extras: {extra_bulk_x.shape[0]} bulk + {extra_sc_x.shape[0]} sc")
    m_hca = np.full((hca_x.shape[0], 4), np.nan, dtype=np.float32)
    m_hca[:, 0] = 1.0  # modality=sc

    # Metadata for melanoma extras: modality known, GTEx-style covariates NaN
    m_extra_bulk = np.full((extra_bulk_x.shape[0], 4), np.nan, dtype=np.float32)
    m_extra_bulk[:, 0] = 0.0
    m_extra_sc = np.full((extra_sc_x.shape[0], 4), np.nan, dtype=np.float32)
    m_extra_sc[:, 0] = 1.0

    return {
        "x_gtex": g.expr_scaled, "m_gtex": m_gtex, "id_gtex": np.asarray(g.sample_ids),
        "x_hca": hca_x, "m_hca": m_hca, "id_hca": hca_ids,
        "x_extra_bulk": extra_bulk_x, "m_extra_bulk": m_extra_bulk, "id_extra_bulk": extra_bulk_ids,
        "x_extra_sc": extra_sc_x, "m_extra_sc": m_extra_sc, "id_extra_sc": extra_sc_ids,
        "shared_genes": shared_genes,
        "scaler_mean": scaler_mean, "scaler_std": scaler_std,
        "isch_mean": float(np.nanmean(isch)), "isch_std": float(np.nanstd(isch)),
        "dthh_mean": float(np.nanmean(dthh)), "dthh_std": float(np.nanstd(dthh)),
    }


# ── train ────────────────────────────────────────────────────────────────
def train(model, x_train, m_train, x_val, m_val, *, epochs=200, batch=64,
          lr=1e-3, beta_bio=1e-3, beta_meta=1e-4, lam_sup=1.0, lam_leak=0.1,
          free_bits=0.5, lam_cycle=0.0, lam_cycle_meta=None):
    if lam_cycle_meta is None:
        lam_cycle_meta = lam_cycle
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(m_train)),
        batch_size=batch, shuffle=True, drop_last=False,
    )
    x_val_t = torch.from_numpy(x_val).to(DEVICE)
    m_val_t = torch.from_numpy(m_val).to(DEVICE)
    history = {"epoch": [], "train_loss": [], "val_recon": [],
               "val_active_bio": [], "head_acc": []}
    kinds = model.config.meta_kinds

    for ep in range(1, epochs + 1):
        model.train()
        ep_losses = []
        for x_b, m_b in train_loader:
            x_b = x_b.to(DEVICE); m_b = m_b.to(DEVICE)
            x_hat, mu_m, lv_m, mu_b, lv_b, z_m, z_b = model(x_b)
            recon = nn.functional.mse_loss(x_hat, x_b)
            kl_b = kl_with_free_bits(mu_b, lv_b, free_bits) / x_b.size(0)
            kl_m = kl_per_dim(mu_m, lv_m).sum()
            preds = model.head_predictions(z_m)
            sup, _ = supervised_loss(preds, [m_b[:, i] for i in range(m_b.size(1))], kinds)

            # HSIC: leak ONLY on modality (col 0). The other metadata are bio
            # axes — z_bio should be free to redundantly encode them.
            mod_col = m_b[:, 0:1]
            row_ok = torch.isfinite(mod_col[:, 0])
            if row_ok.sum() >= 8:
                leak = hsic_penalty(z_b[row_ok], mod_col[row_ok])
            else:
                leak = z_b.new_zeros(())

            # Cycle consistency:
            #   - z_bio re-encoded after flip should match original z_bio
            #     (forces z_bio to be modality-invariant biology)
            #   - z_meta re-encoded modality should match the FLIPPED target
            #     (forces the decoder to actually change output under flip)
            if lam_cycle > 0 and row_ok.sum() >= 4:
                z_m_flip = z_m.clone()
                bulk_mask = mod_col[:, 0] < 0.5
                target_mod = torch.where(bulk_mask,
                                         torch.full_like(z_m_flip[:, 0], 5.0),
                                         torch.full_like(z_m_flip[:, 0], -5.0))
                z_m_flip[:, 0] = target_mod
                x_flipped = model.decode(z_m_flip, z_b)
                mu_m_recon, _, mu_b_recon, _ = model.encode(x_flipped)
                cycle_bio = nn.functional.mse_loss(mu_b_recon, mu_b.detach())
                # Force the re-encoded modality logit to be the flipped class.
                flipped_mod_label = torch.where(bulk_mask,
                                                torch.ones_like(target_mod),
                                                torch.zeros_like(target_mod))
                pred_logit_post = model.heads[0](mu_m_recon[:, 0:1]).squeeze(-1)
                cycle_meta = nn.functional.binary_cross_entropy_with_logits(
                    pred_logit_post, flipped_mod_label
                )
            else:
                cycle_bio = z_b.new_zeros(())
                cycle_meta = z_b.new_zeros(())

            loss = (recon + beta_bio * kl_b + beta_meta * kl_m
                    + lam_sup * sup + lam_leak * leak
                    + lam_cycle * cycle_bio + lam_cycle_meta * cycle_meta)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            x_hat_v, mu_m_v, lv_m_v, mu_b_v, lv_b_v, z_m_v, z_b_v = model(x_val_t)
            v_recon = nn.functional.mse_loss(x_hat_v, x_val_t).item()
            active_bio = (mu_b_v.var(dim=0) > 0.01).sum().item()
            preds_v = model.head_predictions(z_m_v)
            head_metrics = []
            for i, (p, kind) in enumerate(zip(preds_v, kinds)):
                y = m_val_t[:, i]; mk = torch.isfinite(y)
                if mk.sum() == 0:
                    head_metrics.append(np.nan); continue
                if kind == "bce":
                    pred_lab = (torch.sigmoid(p[mk]) > 0.5).float()
                    head_metrics.append((pred_lab == y[mk]).float().mean().item())
                else:
                    sst = ((y[mk] - y[mk].mean()) ** 2).sum()
                    sse = ((p[mk] - y[mk]) ** 2).sum()
                    head_metrics.append(1 - (sse / sst).item())
        history["epoch"].append(ep)
        history["train_loss"].append(float(np.mean(ep_losses)))
        history["val_recon"].append(v_recon)
        history["val_active_bio"].append(active_bio)
        history["head_acc"].append(head_metrics)
        if ep % 10 == 0 or ep == 1 or ep == epochs:
            print(f"ep{ep:3d}  train={np.mean(ep_losses):.4f}  v_recon={v_recon:.4f}  "
                  f"act_bio={active_bio}/{model.config.z_bio_dim}  "
                  f"heads={['%.2f'%h for h in head_metrics]}", flush=True)
    return history


# ── flip tests ───────────────────────────────────────────────────────────
def _r2(y_true, y_pred):
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    ss_res = ((y_true - y_pred) ** 2).sum()
    return float(1 - ss_res / max(float(ss_tot), 1e-12))


def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a**2).sum() * (b**2).sum()) + 1e-12))


def _cosine_dist(a, b):
    """a: (n,d), b: (m,d) → (n,m) cosine distance, scale-robust."""
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return 1 - an @ bn.T


def flip_test(model, data, val_idx, *, k_top=200, n_balance=8):
    """Run Test 1 + Test 2 with balanced NN-pools and cosine distance.

    Test 1.1 (balanced NN): for each query, draw {n_balance HCA + n_balance GTEx
      held-out} as the candidate pool, compute cosine NN. Repeat with reseeding.
      A working flip should push the NN from GTEx (baseline) toward HCA.

    Test 1.2 (biology preservation): on genes NOT in the top-k bulk-vs-sc lfc
      list, Pearson(flipped, original) should stay high.

    Test 1.3 (mean distance to each pool): mean cosine dist from x_hat to
      {GTEx pool, HCA pool}, before vs after flip — should swap direction.
    """
    model.eval()
    x_gtex_val_np = data["x_gtex"][val_idx]
    x_hca_np = data["x_hca"]
    x_gtex_val = torch.from_numpy(x_gtex_val_np).to(DEVICE)
    x_hca = torch.from_numpy(x_hca_np).to(DEVICE)

    with torch.no_grad():
        mu_m_g, _, mu_b_g, _ = model.encode(x_gtex_val)
        mu_m_h, _, mu_b_h, _ = model.encode(x_hca)
        # Empirical class-means in z_meta space — use them as flip targets
        # rather than arbitrary +/-5 magnitudes.
        bulk_mean_z0 = mu_m_g[:, 0].mean().item()
        sc_mean_z0 = mu_m_h[:, 0].mean().item()
        print(f"  encoded z_meta[modality]: bulk_mean={bulk_mean_z0:.3f} sc_mean={sc_mean_z0:.3f}")

        x_hat_orig = model.decode(mu_m_g, mu_b_g).cpu().numpy()
        z_m_flip = mu_m_g.clone(); z_m_flip[:, 0] = sc_mean_z0
        x_hat_sc = model.decode(z_m_flip, mu_b_g).cpu().numpy()

        # Ischemia flip: empirical high-ischemia direction
        bulk_isch = mu_m_g[:, 1]
        z_m_isch_flip = mu_m_g.clone(); z_m_isch_flip[:, 1] = bulk_isch.max().item()
        x_hat_isch = model.decode(z_m_isch_flip, mu_b_g).cpu().numpy()

        x_hat_hca_orig = model.decode(mu_m_h, mu_b_h).cpu().numpy()
        z_m_flip_h = mu_m_h.clone(); z_m_flip_h[:, 0] = bulk_mean_z0
        x_hat_bulk = model.decode(z_m_flip_h, mu_b_h).cpu().numpy()

    # Test 1.3: mean cosine distance to each pool
    d_orig_to_gtex = _cosine_dist(x_hat_orig, x_gtex_val_np)
    d_orig_to_hca = _cosine_dist(x_hat_orig, x_hca_np)
    d_flip_to_gtex = _cosine_dist(x_hat_sc, x_gtex_val_np)
    d_flip_to_hca = _cosine_dist(x_hat_sc, x_hca_np)
    # Use mean-of-min (NN distance) for a sharper signal
    nn_orig_to_gtex = d_orig_to_gtex.min(axis=1).mean()
    nn_orig_to_hca = d_orig_to_hca.min(axis=1).mean()
    nn_flip_to_gtex = d_flip_to_gtex.min(axis=1).mean()
    nn_flip_to_hca = d_flip_to_hca.min(axis=1).mean()

    # Test 1.1 (balanced NN): for each query, candidate pool = all 8 HCA +
    # 8 randomly drawn GTEx held-out (excluding self). Repeat 50 reps and average.
    rng = np.random.default_rng(0)
    nn_corr_flip_balanced = []
    nn_corr_orig_balanced = []
    n_v = len(val_idx)
    for q in range(n_v):
        for _ in range(50):
            other_g = np.array([j for j in range(n_v) if j != q])
            sample = rng.choice(other_g, size=min(n_balance, len(other_g)), replace=False)
            pool = np.concatenate([x_gtex_val_np[sample], x_hca_np], axis=0)
            mod = np.concatenate([np.zeros(len(sample), dtype=int),
                                  np.ones(len(x_hca_np), dtype=int)])
            d_flip = _cosine_dist(x_hat_sc[q:q+1], pool)[0]
            d_orig = _cosine_dist(x_hat_orig[q:q+1], pool)[0]
            nn_corr_flip_balanced.append(int(mod[d_flip.argmin()] == 1))
            nn_corr_orig_balanced.append(int(mod[d_orig.argmin()] == 1))
    nn_acc_flip_to_sc = float(np.mean(nn_corr_flip_balanced))
    nn_acc_orig_to_sc = float(np.mean(nn_corr_orig_balanced))

    # Symmetric for HCA→bulk: pool = all GTEx + 8 HCA (excluding self)
    nn_corr_flip_h = []
    for q in range(len(x_hca_np)):
        other_h = np.array([j for j in range(len(x_hca_np)) if j != q])
        for _ in range(50):
            sample_g = rng.choice(n_v, size=n_balance, replace=False)
            pool = np.concatenate([x_gtex_val_np[sample_g], x_hca_np[other_h]], axis=0)
            mod = np.concatenate([np.zeros(len(sample_g), dtype=int),
                                  np.ones(len(other_h), dtype=int)])
            d = _cosine_dist(x_hat_bulk[q:q+1], pool)[0]
            nn_corr_flip_h.append(int(mod[d.argmin()] == 0))
    nn_acc_flip_to_bulk = float(np.mean(nn_corr_flip_h))

    # Test 1.2: biology preservation
    bulk_mean = data["x_gtex"].mean(axis=0)
    sc_mean = data["x_hca"].mean(axis=0)
    lfc = np.abs(bulk_mean - sc_mean)
    meta_genes = np.argsort(lfc)[-k_top:]
    bio_mask = np.ones(x_gtex_val_np.shape[1], dtype=bool); bio_mask[meta_genes] = False
    bio_corrs = [_pearson(x_hat_sc[i, bio_mask], x_gtex_val_np[i, bio_mask])
                 for i in range(n_v)]
    meta_corrs = [_pearson(x_hat_sc[i, meta_genes], x_gtex_val_np[i, meta_genes])
                  for i in range(n_v)]

    # Recon quality (val) — log-MSE in scaled space
    val_recon_mse = float(((x_hat_orig - x_gtex_val_np) ** 2).mean())

    isch_drift = float(((x_hat_isch - x_hat_orig) ** 2).mean())
    flip_drift = float(((x_hat_sc - x_hat_orig) ** 2).mean())

    # Round-trip cycle test: GTEx → flip-to-sc → flip-back-to-bulk → should
    # match original GTEx. This is a paired-data-free biology preservation
    # metric: tests whether z_bio carries donor identity through the flip.
    with torch.no_grad():
        x_hat_sc_t = torch.from_numpy(x_hat_sc).to(DEVICE)
        mu_m_sc_re, _, mu_b_sc_re, _ = model.encode(x_hat_sc_t)
        z_m_back = mu_m_sc_re.clone()
        z_m_back[:, 0] = bulk_mean_z0   # flip back to bulk
        x_round = model.decode(z_m_back, mu_b_sc_re).cpu().numpy()
        # Per-sample correlation with original GTEx input
        round_corrs = [_pearson(x_round[i], x_gtex_val_np[i]) for i in range(n_v)]
        # z_bio cycle preservation
        z_bio_orig = mu_b_g.cpu().numpy()
        z_bio_re_sc = mu_b_sc_re.cpu().numpy()
        z_bio_cosine = float(np.mean([
            float((z_bio_orig[i] @ z_bio_re_sc[i])
                  / (np.linalg.norm(z_bio_orig[i]) * np.linalg.norm(z_bio_re_sc[i]) + 1e-12))
            for i in range(n_v)
        ]))

    return {
        "test1_1_nn_acc_flip_to_sc_BALANCED": nn_acc_flip_to_sc,
        "test1_1_nn_acc_orig_to_sc_BALANCED_baseline": nn_acc_orig_to_sc,
        "test1_1_nn_acc_flip_to_bulk_HCA_BALANCED": nn_acc_flip_to_bulk,
        "test1_2_biology_preservation_pearson_GTEx_v_flipGTEx": float(np.mean(bio_corrs)),
        "test1_2_metadata_genes_pearson_GTEx_v_flipGTEx": float(np.mean(meta_corrs)),
        "test1_3_mean_NN_cosine_orig_to_GTEx_pool": float(nn_orig_to_gtex),
        "test1_3_mean_NN_cosine_orig_to_HCA_pool": float(nn_orig_to_hca),
        "test1_3_mean_NN_cosine_flip_to_GTEx_pool": float(nn_flip_to_gtex),
        "test1_3_mean_NN_cosine_flip_to_HCA_pool": float(nn_flip_to_hca),
        "test1_4_round_trip_pearson_GTEx_v_flipped_back": float(np.mean(round_corrs)),
        "test1_4_z_bio_cycle_cosine": z_bio_cosine,
        "ischemia_flip_drift_mse": isch_drift,
        "modality_flip_drift_mse": flip_drift,
        "val_recon_mse": val_recon_mse,
        "n_val_gtex": int(n_v),
        "n_hca": int(x_hca_np.shape[0]),
        "n_meta_linked_genes_excluded": int(k_top),
    }


# ── main ─────────────────────────────────────────────────────────────────
def main(epochs=200, run_tag="run1", include_melanoma=False, **train_kwargs):
    print(f"[Q20] Loading data, device={DEVICE}, include_melanoma={include_melanoma}")
    data = build_data(include_melanoma=include_melanoma)
    n_gtex = data["x_gtex"].shape[0]
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_gtex)
    n_val = int(0.2 * n_gtex)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    # Oversample HCA to balance the modalities
    n_hca = data["x_hca"].shape[0]
    hca_repeat = max(1, len(train_idx) // n_hca)
    x_hca_rep = np.tile(data["x_hca"], (hca_repeat, 1))
    m_hca_rep = np.tile(data["m_hca"], (hca_repeat, 1))

    pieces_x = [data["x_gtex"][train_idx], x_hca_rep]
    pieces_m = [data["m_gtex"][train_idx], m_hca_rep]
    if include_melanoma:
        pieces_x += [data["x_extra_bulk"], data["x_extra_sc"]]
        pieces_m += [data["m_extra_bulk"], data["m_extra_sc"]]

    x_train = np.concatenate(pieces_x, axis=0)
    m_train = np.concatenate(pieces_m, axis=0)
    x_val = data["x_gtex"][val_idx]; m_val = data["m_gtex"][val_idx]
    print(f"  train: GTEx {len(train_idx)} + HCA {n_hca}×{hca_repeat}={n_hca*hca_repeat}"
          + (f" + melB {data['x_extra_bulk'].shape[0]} + melSC {data['x_extra_sc'].shape[0]}"
             if include_melanoma else "")
          + f" = {x_train.shape[0]}")
    print(f"  val:   GTEx {len(val_idx)}")

    cfg = DisentangledConfig(input_dim=x_train.shape[1])
    model = DisentangledVAE(cfg).to(DEVICE)
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}  "
          f"latent: z_meta={cfg.z_meta_dim} + z_bio={cfg.z_bio_dim}")

    history = train(model, x_train, m_train, x_val, m_val, epochs=epochs, **train_kwargs)

    print("\n[Q20] Running flip tests …")
    flip_results = flip_test(model, data, val_idx)
    for k, v in flip_results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    out_path = OUT / f"q20_{run_tag}.json"
    with out_path.open("w") as f:
        json.dump({"history": history, "flip": flip_results,
                   "config": {"z_bio_dim": cfg.z_bio_dim,
                              "z_meta_dim": cfg.z_meta_dim,
                              "epochs": epochs,
                              **train_kwargs}}, f, indent=2)
    ckpt_path = OUT / f"q20_{run_tag}.pt"
    torch.save({"state_dict": model.state_dict(),
                "config": cfg.__dict__,
                "shared_genes": data["shared_genes"]}, ckpt_path)
    print(f"  saved: {out_path.name} + {ckpt_path.name}")
    return flip_results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--tag", default="run1")
    p.add_argument("--beta_bio", type=float, default=1e-3)
    p.add_argument("--beta_meta", type=float, default=1e-4)
    p.add_argument("--lam_sup", type=float, default=1.0)
    p.add_argument("--lam_leak", type=float, default=0.1)
    p.add_argument("--free_bits", type=float, default=0.5)
    p.add_argument("--lam_cycle", type=float, default=0.0)
    p.add_argument("--lam_cycle_meta", type=float, default=None)
    p.add_argument("--include_melanoma", action="store_true")
    a = p.parse_args()
    main(epochs=a.epochs, run_tag=a.tag, include_melanoma=a.include_melanoma,
         beta_bio=a.beta_bio, beta_meta=a.beta_meta,
         lam_sup=a.lam_sup, lam_leak=a.lam_leak, free_bits=a.free_bits,
         lam_cycle=a.lam_cycle, lam_cycle_meta=a.lam_cycle_meta)
