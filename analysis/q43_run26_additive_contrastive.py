"""Q43 / Run 26 — Make vae_flip useful (proposal cornerstone).

Two architectural changes:

1. ADDITIVE DECODER:
   x_hat = template(z_meta) + residual(z_bio)
   Flipping z_meta changes template; residual (donor-specific) stays.
   Cleaner than Run 17's multiplicative gate.

2. CONTRASTIVE-ON-DECODED LOSS:
   For each (bulk_d, sn_d), decoded flip should have higher cosine to sn_d
   than to all other batch sn samples. Direct optimization of the test metric.

Tested: BOTH flip-decode (proposal goal) AND latent-NN, on Eraslan paired LOO.
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
LAM_NCE_FLIP = 10.0   # NEW: weight on contrastive-on-decoded
NCE_TAU = 0.1
BETA_BIO = 1e-3
Z_BIO_DIM = 5


def _mlp(dims, dropout=0.1):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class AdditiveVAE(nn.Module):
    """Decoder = template(z_meta) + residual(z_bio).

    The flip changes template; residual stays — guaranteeing donor info
    survives the flip by ARCHITECTURE, not just by training.
    """
    def __init__(self, input_dim, z_meta_dim=2, z_bio_dim=5, hidden=(1024, 512, 256)):
        super().__init__()
        h = list(hidden)
        self.encoder = _mlp([input_dim] + h)
        self.mu_meta = nn.Linear(h[-1], z_meta_dim)
        self.lv_meta = nn.Linear(h[-1], z_meta_dim)
        self.mu_bio = nn.Linear(h[-1], z_bio_dim)
        self.lv_bio = nn.Linear(h[-1], z_bio_dim)
        # Two parallel decoder branches
        self.template_branch = _mlp([z_meta_dim, 64, 256, 1024, input_dim])
        self.residual_branch = _mlp([z_bio_dim, 64, 256, 1024, input_dim])
        self.head_mod = nn.Linear(1, 1)
        self.head_tis = nn.Linear(1, 1)

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


def main():
    print(f"[Q43/Run26 additive+contrastive] device={DEVICE}  z_bio={Z_BIO_DIM}  lam_nce={LAM_NCE_FLIP}")
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
        Xb_tr = bulk_x[train_bulk_mask]; donor_b_tr = bulk_donor[train_bulk_mask]; tissue_b_tr = bulk_tissue[train_bulk_mask]
        Xs_tr = sn_x[train_sn_mask]; donor_s_tr = sn_donor[train_sn_mask]; tissue_s_tr = sn_tissue[train_sn_mask]
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
        m_b_pair_tis = torch.tensor([tissue_to_idx[t] for t in tissue_b_tr[bulk_pair_idx]],
                                    device=DEVICE, dtype=torch.float32) / max(len(tissues) - 1, 1)
        m_s_pair_tis = torch.tensor([tissue_to_idx[t] for t in tissue_s_tr[sn_pair_idx]],
                                    device=DEVICE, dtype=torch.float32) / max(len(tissues) - 1, 1)
        m_b_pair_mod = torch.zeros(Xb_pair.size(0), device=DEVICE)
        m_s_pair_mod = torch.ones(Xs_pair.size(0), device=DEVICE)

        model = AdditiveVAE(input_dim=bulk_x.shape[1], z_meta_dim=2, z_bio_dim=Z_BIO_DIM).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)

        bulk_z0_target = -3.0; sc_z0_target = 3.0
        for ep in range(1, EPOCHS + 1):
            x_hat_b, mu_m_b, lv_m_b, mu_b_b, lv_b_b, z_m_b, z_b_b = model(Xb_pair)
            x_hat_s, mu_m_s, lv_m_s, mu_b_s, lv_b_s, z_m_s, z_b_s = model(Xs_pair)
            recon = nn.functional.mse_loss(x_hat_b, Xb_pair) + nn.functional.mse_loss(x_hat_s, Xs_pair)
            kl_b = (kl_with_free_bits(mu_b_b, lv_b_b, 0.5)
                    + kl_with_free_bits(mu_b_s, lv_b_s, 0.5)) / Xb_pair.size(0)
            sup_mod = (nn.functional.binary_cross_entropy_with_logits(
                          model.head_mod(z_m_b[:, 0:1]).squeeze(-1), m_b_pair_mod)
                       + nn.functional.binary_cross_entropy_with_logits(
                          model.head_mod(z_m_s[:, 0:1]).squeeze(-1), m_s_pair_mod))
            sup_tis = (nn.functional.mse_loss(model.head_tis(z_m_b[:, 1:2]).squeeze(-1), m_b_pair_tis)
                       + nn.functional.mse_loss(model.head_tis(z_m_s[:, 1:2]).squeeze(-1), m_s_pair_tis))

            # Flip operations
            z_m_flip_b = z_m_b.clone(); z_m_flip_b[:, 0] = sc_z0_target
            x_flip_b2s = model.decode(z_m_flip_b, z_b_b)
            paired_b2s = nn.functional.mse_loss(x_flip_b2s, Xs_pair)
            z_m_flip_s = z_m_s.clone(); z_m_flip_s[:, 0] = bulk_z0_target
            x_flip_s2b = model.decode(z_m_flip_s, z_b_s)
            paired_s2b = nn.functional.mse_loss(x_flip_s2b, Xb_pair)

            # ── NEW: contrastive on decoded flip output ──
            # For paired batch: x_flip_b2s[i] should be cosine-closer to Xs_pair[i]
            # than to Xs_pair[j!=i].
            nce_b2s = info_nce(x_flip_b2s, Xs_pair, tau=NCE_TAU)
            nce_s2b = info_nce(x_flip_s2b, Xb_pair, tau=NCE_TAU)

            # Cycle + leak + donor_id (still useful)
            mu_m_re_b, _, mu_b_re_b, _ = model.encode(x_flip_b2s)
            mu_m_re_s, _, mu_b_re_s, _ = model.encode(x_flip_s2b)
            cyc = (nn.functional.mse_loss(mu_b_re_b, mu_b_b.detach())
                   + nn.functional.mse_loss(mu_b_re_s, mu_b_s.detach()))
            mod_col = torch.cat([torch.full((Xb_pair.size(0),1), 0.0, device=DEVICE),
                                 torch.full((Xs_pair.size(0),1), 1.0, device=DEVICE)])
            z_b_combined = torch.cat([z_b_b, z_b_s], dim=0)
            leak = hsic_penalty(z_b_combined, mod_col)
            donor_id = nn.functional.mse_loss(mu_b_b, mu_b_s)

            loss = (recon + BETA_BIO * kl_b + LAM_SUP * (sup_mod + sup_tis)
                    + LAM_PAIRED * (paired_b2s + paired_s2b)
                    + LAM_CYCLE * cyc + LAM_LEAK * leak + LAM_DONOR_ID * donor_id
                    + LAM_NCE_FLIP * (nce_b2s + nce_s2b))
            opt.zero_grad(); loss.backward(); opt.step()
            if fold == 0 and (ep % 50 == 0 or ep == 1):
                print(f"    ep{ep:3d}  loss={loss.item():.3f}  recon={recon.item():.3f}  "
                      f"paired={paired_b2s.item():.3f}  nce_b2s={nce_b2s.item():.3f}  "
                      f"donor_id={donor_id.item():.4f}", flush=True)

        # Test both flip and latent
        model.eval()
        with torch.no_grad():
            x_held_b = torch.from_numpy(bulk_x[held_bulk_i:held_bulk_i+1]).to(DEVICE)
            mu_m_h, _, mu_b_h, _ = model.encode(x_held_b)
            z_m_flip = mu_m_h.clone(); z_m_flip[:, 0] = sc_z0_target
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
    print(f"\n=== Run 26 LOO summary across {len(fold_results)} folds ===")
    print(f"  FLIP   top-1: {arr_flip_top1.mean():.3f}  top-3: {arr_flip_top3.mean():.3f}  rank: {arr_flip_rank.mean():.2f}")
    print(f"  LATENT top-1: {arr_lat_top1.mean():.3f}  top-3: {arr_lat_top3.mean():.3f}  rank: {arr_lat_rank.mean():.2f}")
    print(f"  Reference: VAE flip (Run 23) = 0.292, CCA-5 = 0.625, ENS-3 = 0.708")

    out = {"config": {"z_bio_dim": Z_BIO_DIM, "lam_nce": LAM_NCE_FLIP, "tau": NCE_TAU,
                      "decoder": "additive (template + residual)"},
           "fold_results": fold_results,
           "summary": {
               "flip_top1": float(arr_flip_top1.mean()),
               "flip_top3": float(arr_flip_top3.mean()),
               "flip_rank_first_same": float(arr_flip_rank.mean()),
               "latent_top1": float(arr_lat_top1.mean()),
               "latent_top3": float(arr_lat_top3.mean()),
               "latent_rank_first_same": float(arr_lat_rank.mean()),
           }}
    with (OUT_DIR / "q43_run26_additive_contrastive.json").open("w") as f:
        json.dump(out, f, indent=2)
    print(f"saved → {OUT_DIR / 'q43_run26_additive_contrastive.json'}")


if __name__ == "__main__":
    main()
