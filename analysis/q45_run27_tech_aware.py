"""Q45 / Run 27 — Technology-aware AdditiveVAE.

Adds two new metadata dimensions inspired by scFM literature:

1. TECHNOLOGY (scGPT/CellPLM-style):
   z_meta now has a tech_idx dim with CE classifier head.
   Values: 0=bulk_illumina, 1=10x_chromium. In current data, tech and
   modality are 100% correlated; the slot exists for future multi-tech use.

2. LOG_DEPTH (scFoundation-style):
   z_meta has a continuous log10_total_counts dim with MSE head.
   This is what scFoundation's "read-depth recovery" captures.

Plus fix Run 26's broken contrastive: InfoNCE on 5-D LATENT z_bio (not 11k-D
decoded gene-space) — gradients work in low-dim cosine.

Test: BOTH flip-decode and latent-NN, Eraslan paired LOO.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.disentangled_vae import kl_with_free_bits, hsic_penalty

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

PAIRED_NPZ = "/Users/rls/ecs271/data/sc/eraslan/eraslan_paired.npz"
OUT_DIR = ROOT / "analysis" / "results" / "q20_disentangled"

EPOCHS = 200
LR = 1e-3
LAM_PAIRED = 5.0
LAM_LEAK = 0.3
LAM_CYCLE = 0.3
LAM_SUP = 1.0
LAM_DONOR_ID = 5.0
LAM_NCE_LATENT = 5.0   # NEW: contrastive on z_bio (5-D), where gradients work
NCE_TAU = 0.1
BETA_BIO = 1e-3
Z_BIO_DIM = 5
# Tech mapping (categorical)
TECH_MAP = {"bulk_illumina": 0, "10x_chromium": 1}
N_TECH = 2


def _mlp(dims, dropout=0.1):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class TechAwareVAE(nn.Module):
    """AdditiveVAE + tech-id embedding + log-depth metadata.

    z_meta = [modality, tissue, tech_embedded(4), log_depth]  -> total 7 dims
    """
    def __init__(self, input_dim, z_bio_dim=5, tech_dim=4, hidden=(1024, 512, 256)):
        super().__init__()
        h = list(hidden)
        self.tech_dim = tech_dim
        self.encoder = _mlp([input_dim] + h)
        z_meta_dim = 1 + 1 + tech_dim + 1   # modality + tissue + tech_emb + log_depth
        self.z_meta_dim = z_meta_dim
        self.mu_meta = nn.Linear(h[-1], z_meta_dim)
        self.lv_meta = nn.Linear(h[-1], z_meta_dim)
        self.mu_bio = nn.Linear(h[-1], z_bio_dim)
        self.lv_bio = nn.Linear(h[-1], z_bio_dim)
        # scGPT-style: explicit tech embedding lookup (for FLIPS, the decoder needs to use this directly)
        self.tech_embed = nn.Embedding(N_TECH, tech_dim)
        # Decoder: additive (template + residual)
        self.template_branch = _mlp([z_meta_dim, 128, 512, 1024, input_dim])
        self.residual_branch = _mlp([z_bio_dim, 64, 256, 1024, input_dim])
        # Heads on z_meta slots
        self.head_mod = nn.Linear(1, 1)
        self.head_tis = nn.Linear(1, 1)
        self.head_tech = nn.Linear(tech_dim, N_TECH)
        self.head_depth = nn.Linear(1, 1)

    @staticmethod
    def reparam(mu, lv):
        return mu + (0.5 * lv).exp() * torch.randn_like(mu)

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_meta(h), self.lv_meta(h), self.mu_bio(h), self.lv_bio(h)

    def decode(self, z_meta, z_bio):
        return self.template_branch(z_meta) + self.residual_branch(z_bio)

    def forward(self, x):
        mu_m, lv_m, mu_b, lv_b = self.encode(x)
        z_m = self.reparam(mu_m, lv_m); z_b = self.reparam(mu_b, lv_b)
        return self.decode(z_m, z_b), mu_m, lv_m, mu_b, lv_b, z_m, z_b


def info_nce(q, p, tau=0.1):
    qn = nn.functional.normalize(q, dim=1)
    pn = nn.functional.normalize(p, dim=1)
    logits_qp = qn @ pn.t() / tau
    logits_pq = pn @ qn.t() / tau
    target = torch.arange(q.size(0), device=q.device)
    return 0.5 * (nn.functional.cross_entropy(logits_qp, target)
                  + nn.functional.cross_entropy(logits_pq, target))


def _cosine(a, b):
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _eval(scores, sd_pool, dn):
    order = np.argsort(-scores)
    ord_d = sd_pool[order]
    rank = next(j for j, d in enumerate(ord_d) if d == dn) + 1
    topK = {K: float(np.mean(ord_d[:K] == dn)) for K in [1, 3, 5, 10]}
    return rank, topK


def _log_depth_proxy(x_scaled):
    """Proxy for log-depth from already-standardized log2(CPM+1) data.
    The mean of the standardized matrix is a coarse depth proxy (lower for
    sparser data). Z-score it across samples to get a stable input to the head."""
    return x_scaled.mean(axis=1, keepdims=False)


def main():
    print(f"[Q45/Run27 tech-aware] device={DEVICE}  z_bio={Z_BIO_DIM}  N_TECH={N_TECH}")
    d = np.load(PAIRED_NPZ, allow_pickle=True)
    bulk_x = d["bulk_x"].astype(np.float32)
    bulk_donor = np.asarray(d["bulk_donor"], dtype=str)
    bulk_tissue = np.asarray(d["bulk_tissue"], dtype=str)
    sn_x = d["sn_x"].astype(np.float32)
    sn_donor = np.asarray(d["sn_donor"], dtype=str)
    sn_tissue = np.asarray(d["sn_tissue"], dtype=str)
    print(f"  bulk: {bulk_x.shape}  sn: {sn_x.shape}")
    tissues = sorted(set(np.concatenate([bulk_tissue, sn_tissue])))
    tissue_to_idx = {t: i for i, t in enumerate(tissues)}

    # All bulk in our data is GTEx illumina; all sn is 10x.
    # In future cross-tech data, tech_idx would vary independently of modality.
    bulk_tech = np.full(len(bulk_x), TECH_MAP["bulk_illumina"], dtype=np.int64)
    sn_tech = np.full(len(sn_x), TECH_MAP["10x_chromium"], dtype=np.int64)

    # Depth proxy
    bulk_depth = _log_depth_proxy(bulk_x)
    sn_depth = _log_depth_proxy(sn_x)
    # Z-score across all training samples (we'll re-normalize per-fold)
    all_depth = np.concatenate([bulk_depth, sn_depth])
    depth_mean, depth_std = all_depth.mean(), all_depth.std() + 1e-8
    print(f"  depth proxy: bulk mean={bulk_depth.mean():.3f}  sn mean={sn_depth.mean():.3f}")

    sn_idx_per_dt = {}
    for i, (dn, ts) in enumerate(zip(sn_donor, sn_tissue)):
        sn_idx_per_dt.setdefault((dn, ts), []).append(i)
    bulk_idx_per_dt = {(dn, ts): i for i, (dn, ts) in enumerate(zip(bulk_donor, bulk_tissue))}
    paired_dts = [(dn, ts) for (dn, ts) in bulk_idx_per_dt if (dn, ts) in sn_idx_per_dt]

    fold_results = []
    for fold, (held_dn, held_ts) in enumerate(paired_dts):
        held_bulk_i = bulk_idx_per_dt[(held_dn, held_ts)]
        held_sn_rows = set(sn_idx_per_dt[(held_dn, held_ts)])

        train_bulk_mask = np.ones(len(bulk_x), dtype=bool); train_bulk_mask[held_bulk_i] = False
        train_sn_mask = np.array([i not in held_sn_rows for i in range(len(sn_x))])
        Xb_tr = bulk_x[train_bulk_mask]; donor_b_tr = bulk_donor[train_bulk_mask]
        tissue_b_tr = bulk_tissue[train_bulk_mask]; depth_b_tr = bulk_depth[train_bulk_mask]
        tech_b_tr = bulk_tech[train_bulk_mask]
        Xs_tr = sn_x[train_sn_mask]; donor_s_tr = sn_donor[train_sn_mask]
        tissue_s_tr = sn_tissue[train_sn_mask]; depth_s_tr = sn_depth[train_sn_mask]
        tech_s_tr = sn_tech[train_sn_mask]
        sn_by_dt_tr = {}
        for j, (dn, ts) in enumerate(zip(donor_s_tr, tissue_s_tr)):
            sn_by_dt_tr.setdefault((dn, ts), []).append(j)
        paired_pairs = []
        for i, (dn, ts) in enumerate(zip(donor_b_tr, tissue_b_tr)):
            for j in sn_by_dt_tr.get((dn, ts), []):
                paired_pairs.append((i, j))
        if not paired_pairs: continue

        bulk_pair_idx = np.array([p[0] for p in paired_pairs])
        sn_pair_idx = np.array([p[1] for p in paired_pairs])
        Xb_pair = torch.from_numpy(Xb_tr[bulk_pair_idx]).to(DEVICE)
        Xs_pair = torch.from_numpy(Xs_tr[sn_pair_idx]).to(DEVICE)
        m_b_tis = torch.tensor([tissue_to_idx[t] for t in tissue_b_tr[bulk_pair_idx]],
                                device=DEVICE, dtype=torch.float32) / max(len(tissues) - 1, 1)
        m_s_tis = torch.tensor([tissue_to_idx[t] for t in tissue_s_tr[sn_pair_idx]],
                                device=DEVICE, dtype=torch.float32) / max(len(tissues) - 1, 1)
        m_b_mod = torch.zeros(Xb_pair.size(0), device=DEVICE)
        m_s_mod = torch.ones(Xs_pair.size(0), device=DEVICE)
        m_b_tech = torch.from_numpy(tech_b_tr[bulk_pair_idx]).to(DEVICE)
        m_s_tech = torch.from_numpy(tech_s_tr[sn_pair_idx]).to(DEVICE)
        m_b_depth = torch.from_numpy(
            ((depth_b_tr[bulk_pair_idx] - depth_mean) / depth_std).astype(np.float32)).to(DEVICE)
        m_s_depth = torch.from_numpy(
            ((depth_s_tr[sn_pair_idx] - depth_mean) / depth_std).astype(np.float32)).to(DEVICE)

        model = TechAwareVAE(input_dim=bulk_x.shape[1], z_bio_dim=Z_BIO_DIM).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)

        bulk_z0_target = -3.0; sc_z0_target = 3.0
        for ep in range(1, EPOCHS + 1):
            x_hat_b, mu_m_b, lv_m_b, mu_b_b, lv_b_b, z_m_b, z_b_b = model(Xb_pair)
            x_hat_s, mu_m_s, lv_m_s, mu_b_s, lv_b_s, z_m_s, z_b_s = model(Xs_pair)
            recon = nn.functional.mse_loss(x_hat_b, Xb_pair) + nn.functional.mse_loss(x_hat_s, Xs_pair)
            kl_b = (kl_with_free_bits(mu_b_b, lv_b_b, 0.5)
                    + kl_with_free_bits(mu_b_s, lv_b_s, 0.5)) / Xb_pair.size(0)

            sup_mod = (nn.functional.binary_cross_entropy_with_logits(
                          model.head_mod(z_m_b[:, 0:1]).squeeze(-1), m_b_mod)
                       + nn.functional.binary_cross_entropy_with_logits(
                          model.head_mod(z_m_s[:, 0:1]).squeeze(-1), m_s_mod))
            sup_tis = (nn.functional.mse_loss(model.head_tis(z_m_b[:, 1:2]).squeeze(-1), m_b_tis)
                       + nn.functional.mse_loss(model.head_tis(z_m_s[:, 1:2]).squeeze(-1), m_s_tis))
            # NEW: tech head (CE on the tech_dim slot)
            tech_b_logits = model.head_tech(z_m_b[:, 2:2+model.tech_dim])
            tech_s_logits = model.head_tech(z_m_s[:, 2:2+model.tech_dim])
            sup_tech = (nn.functional.cross_entropy(tech_b_logits, m_b_tech)
                        + nn.functional.cross_entropy(tech_s_logits, m_s_tech))
            # NEW: depth head (MSE on the last slot)
            depth_b_pred = model.head_depth(z_m_b[:, 2+model.tech_dim:2+model.tech_dim+1]).squeeze(-1)
            depth_s_pred = model.head_depth(z_m_s[:, 2+model.tech_dim:2+model.tech_dim+1]).squeeze(-1)
            sup_depth = (nn.functional.mse_loss(depth_b_pred, m_b_depth)
                         + nn.functional.mse_loss(depth_s_pred, m_s_depth))

            # Flip operations: flip BOTH modality AND tech (and depth)
            z_m_flip_b = z_m_b.clone()
            z_m_flip_b[:, 0] = sc_z0_target           # modality flip
            # tech flip: replace tech_dim slot with sc-mean tech embedding
            sn_tech_id_t = torch.full((Xb_pair.size(0),), TECH_MAP["10x_chromium"],
                                       dtype=torch.long, device=DEVICE)
            z_m_flip_b[:, 2:2+model.tech_dim] = model.tech_embed(sn_tech_id_t)
            # depth flip: shift to sn mean
            z_m_flip_b[:, 2+model.tech_dim] = m_s_depth.mean()
            x_flip_b2s = model.decode(z_m_flip_b, z_b_b)
            paired_b2s = nn.functional.mse_loss(x_flip_b2s, Xs_pair)

            z_m_flip_s = z_m_s.clone()
            z_m_flip_s[:, 0] = bulk_z0_target
            bulk_tech_id_t = torch.full((Xs_pair.size(0),), TECH_MAP["bulk_illumina"],
                                          dtype=torch.long, device=DEVICE)
            z_m_flip_s[:, 2:2+model.tech_dim] = model.tech_embed(bulk_tech_id_t)
            z_m_flip_s[:, 2+model.tech_dim] = m_b_depth.mean()
            x_flip_s2b = model.decode(z_m_flip_s, z_b_s)
            paired_s2b = nn.functional.mse_loss(x_flip_s2b, Xb_pair)

            # NEW: contrastive on LATENT z_bio (low-dim, gradients work)
            nce_lat = info_nce(mu_b_b, mu_b_s, tau=NCE_TAU)

            # Cycle + leak + donor_id
            mu_m_re_b, _, mu_b_re_b, _ = model.encode(x_flip_b2s)
            mu_m_re_s, _, mu_b_re_s, _ = model.encode(x_flip_s2b)
            cyc = (nn.functional.mse_loss(mu_b_re_b, mu_b_b.detach())
                   + nn.functional.mse_loss(mu_b_re_s, mu_b_s.detach()))
            mod_col = torch.cat([torch.full((Xb_pair.size(0),1), 0.0, device=DEVICE),
                                 torch.full((Xs_pair.size(0),1), 1.0, device=DEVICE)])
            z_b_combined = torch.cat([z_b_b, z_b_s], dim=0)
            leak = hsic_penalty(z_b_combined, mod_col)
            donor_id = nn.functional.mse_loss(mu_b_b, mu_b_s)

            loss = (recon + BETA_BIO * kl_b
                    + LAM_SUP * (sup_mod + sup_tis + sup_tech + sup_depth)
                    + LAM_PAIRED * (paired_b2s + paired_s2b)
                    + LAM_CYCLE * cyc + LAM_LEAK * leak + LAM_DONOR_ID * donor_id
                    + LAM_NCE_LATENT * nce_lat)
            opt.zero_grad(); loss.backward(); opt.step()
            if fold == 0 and (ep % 50 == 0 or ep == 1):
                print(f"    ep{ep:3d}  loss={loss.item():.2f}  recon={recon.item():.2f}  "
                      f"paired={paired_b2s.item():.2f}  nce_lat={nce_lat.item():.2f}  "
                      f"sup_tech={sup_tech.item():.2f}  sup_depth={sup_depth.item():.3f}  "
                      f"donor_id={donor_id.item():.4f}", flush=True)

        # Test both flip and latent
        model.eval()
        with torch.no_grad():
            x_held_b = torch.from_numpy(bulk_x[held_bulk_i:held_bulk_i+1]).to(DEVICE)
            mu_m_h, _, mu_b_h, _ = model.encode(x_held_b)
            z_m_flip = mu_m_h.clone()
            z_m_flip[:, 0] = sc_z0_target
            sn_tech_id = torch.tensor([TECH_MAP["10x_chromium"]], device=DEVICE)
            z_m_flip[:, 2:2+model.tech_dim] = model.tech_embed(sn_tech_id)
            z_m_flip[:, 2+model.tech_dim] = m_s_depth.mean()
            x_hat_sc = model.decode(z_m_flip, mu_b_h).cpu().numpy()[0]
            sn_full = torch.from_numpy(sn_x).to(DEVICE)
            _, _, mu_b_sn_full, _ = model.encode(sn_full)
            z_held = mu_b_h.cpu().numpy()[0]
            z_sn_all = mu_b_sn_full.cpu().numpy()

        pool_idx = np.where(sn_tissue == held_ts)[0]
        cos_flip = np.array([_cosine(x_hat_sc, sn_x[i]) for i in pool_idx])
        rank_flip, topK_flip = _eval(cos_flip, sn_donor[pool_idx], held_dn)
        cos_lat = np.array([_cosine(z_held, z_sn_all[i]) for i in pool_idx])
        rank_lat, topK_lat = _eval(cos_lat, sn_donor[pool_idx], held_dn)

        print(f"  fold {fold+1}/{len(paired_dts)}: ({held_dn}, {held_ts});  "
              f"FLIP top-1={topK_flip[1]:.2f} rank={rank_flip}  |  "
              f"LATENT top-1={topK_lat[1]:.2f} rank={rank_lat}", flush=True)

        fold_results.append({
            "held_donor": held_dn, "held_tissue": held_ts,
            "n_pool": len(pool_idx), "n_same_donor_in_pool": int((sn_donor[pool_idx] == held_dn).sum()),
            "flip_rank_first_same": rank_flip, "flip_topK": topK_flip,
            "latent_rank_first_same": rank_lat, "latent_topK": topK_lat,
        })

    arr_flip_top1 = np.array([f["flip_topK"][1] for f in fold_results])
    arr_flip_top3 = np.array([f["flip_topK"][3] for f in fold_results])
    arr_flip_rank = np.array([f["flip_rank_first_same"] for f in fold_results])
    arr_lat_top1 = np.array([f["latent_topK"][1] for f in fold_results])
    arr_lat_top3 = np.array([f["latent_topK"][3] for f in fold_results])
    arr_lat_rank = np.array([f["latent_rank_first_same"] for f in fold_results])
    print(f"\n=== Run 27 tech-aware LOO summary across {len(fold_results)} folds ===")
    print(f"  FLIP   top-1: {arr_flip_top1.mean():.3f}  top-3: {arr_flip_top3.mean():.3f}  rank: {arr_flip_rank.mean():.2f}")
    print(f"  LATENT top-1: {arr_lat_top1.mean():.3f}  top-3: {arr_lat_top3.mean():.3f}  rank: {arr_lat_rank.mean():.2f}")
    print(f"  Reference: Run 23 (no tech)  FLIP 0.292, LATENT 0.333")
    print(f"             CCA-5 = 0.625, ENS-3 = 0.708")

    out = {"config": {"z_bio_dim": Z_BIO_DIM, "N_TECH": N_TECH,
                      "lam_nce_latent": LAM_NCE_LATENT, "tau": NCE_TAU,
                      "encoding": "scGPT-style tech-id + scFoundation-style depth"},
           "fold_results": fold_results,
           "summary": {
               "flip_top1": float(arr_flip_top1.mean()),
               "flip_top3": float(arr_flip_top3.mean()),
               "flip_rank_first_same": float(arr_flip_rank.mean()),
               "latent_top1": float(arr_lat_top1.mean()),
               "latent_top3": float(arr_lat_top3.mean()),
               "latent_rank_first_same": float(arr_lat_rank.mean()),
           }}
    with (OUT_DIR / "q45_run27_tech_aware.json").open("w") as f:
        json.dump(out, f, indent=2)
    print(f"saved → {OUT_DIR / 'q45_run27_tech_aware.json'}")


if __name__ == "__main__":
    main()
