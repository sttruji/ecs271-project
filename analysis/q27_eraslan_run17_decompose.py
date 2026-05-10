"""Q27 / Run 17 — Eraslan paired with deconvolution-aware decoder.

Architectural fix: the decoder explicitly factors as
   x_hat = (1 - alpha_modality) * x_donor + alpha_modality * x_template

where:
- x_donor       = MLP(z_bio)               donor-specific shape (modality-agnostic)
- x_template    = small MLP(z_meta)        modality + tissue template
- alpha_modality = sigmoid scalar gated by z_meta[modality]

Idea: when we flip z_meta[modality], only x_template changes; x_donor is preserved.
This GUARANTEES donor-faithfulness through the flip if z_bio carries donor info.

Other losses identical to Run 15.
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
from models.disentangled_vae import (
    DisentangledConfig, kl_with_free_bits, hsic_penalty,
)

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
BETA_BIO = 1e-3


def _mlp(dims, dropout=0.1):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class DecomposedVAE(nn.Module):
    """Decoder = (1 - alpha) * donor_branch(z_bio) + alpha * template_branch(z_meta).

    alpha is a sigmoid-gated scalar driven by z_meta. When you flip z_meta,
    alpha and template_branch change; donor_branch stays put.
    """
    def __init__(self, input_dim, z_meta_dim=2, z_bio_dim=50, hidden=(1024, 512, 256)):
        super().__init__()
        self.z_meta_dim = z_meta_dim
        self.z_bio_dim = z_bio_dim
        h = list(hidden)
        self.encoder = _mlp([input_dim] + h)
        self.mu_meta = nn.Linear(h[-1], z_meta_dim)
        self.lv_meta = nn.Linear(h[-1], z_meta_dim)
        self.mu_bio = nn.Linear(h[-1], z_bio_dim)
        self.lv_bio = nn.Linear(h[-1], z_bio_dim)
        self.donor_branch = _mlp([z_bio_dim] + h[::-1] + [input_dim])
        self.template_branch = _mlp([z_meta_dim, 64, 256, input_dim])
        # alpha gate: how much template overrides donor
        self.alpha_gate = nn.Sequential(nn.Linear(z_meta_dim, 32), nn.GELU(),
                                         nn.Linear(32, input_dim), nn.Sigmoid())
        # Heads on z_meta
        self.head_mod = nn.Linear(1, 1)
        self.head_tis = nn.Linear(1, 1)

    @staticmethod
    def reparam(mu, lv):
        return mu + (0.5 * lv).exp() * torch.randn_like(mu)

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_meta(h), self.lv_meta(h), self.mu_bio(h), self.lv_bio(h)

    def decode(self, z_meta, z_bio):
        x_donor = self.donor_branch(z_bio)
        x_template = self.template_branch(z_meta)
        alpha = self.alpha_gate(z_meta)
        return (1 - alpha) * x_donor + alpha * x_template

    def forward(self, x):
        mu_m, lv_m, mu_b, lv_b = self.encode(x)
        z_m = self.reparam(mu_m, lv_m); z_b = self.reparam(mu_b, lv_b)
        return self.decode(z_m, z_b), mu_m, lv_m, mu_b, lv_b, z_m, z_b


def _cosine(a, b):
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    print(f"[Q27/Run17 decomposed-decoder] device={DEVICE}")
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
    print(f"  paired (donor, tissue) groups: {len(paired_dts)}")

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
        if not paired_pairs:
            continue

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

        model = DecomposedVAE(input_dim=bulk_x.shape[1], z_meta_dim=2, z_bio_dim=50).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)

        bulk_z0_target = -3.0
        sc_z0_target = 3.0
        for ep in range(1, EPOCHS + 1):
            x_hat_b, mu_m_b, lv_m_b, mu_b_b, lv_b_b, z_m_b, z_b_b = model(Xb_pair)
            x_hat_s, mu_m_s, lv_m_s, mu_b_s, lv_b_s, z_m_s, z_b_s = model(Xs_pair)
            recon = nn.functional.mse_loss(x_hat_b, Xb_pair) + nn.functional.mse_loss(x_hat_s, Xs_pair)
            kl_b = (kl_with_free_bits(mu_b_b, lv_b_b, 0.5)
                    + kl_with_free_bits(mu_b_s, lv_b_s, 0.5)) / Xb_pair.size(0)
            pred_b = model.head_mod(z_m_b[:, 0:1]).squeeze(-1)
            pred_s = model.head_mod(z_m_s[:, 0:1]).squeeze(-1)
            sup_mod = (nn.functional.binary_cross_entropy_with_logits(pred_b, m_b_pair_mod)
                       + nn.functional.binary_cross_entropy_with_logits(pred_s, m_s_pair_mod))
            pred_t_b = model.head_tis(z_m_b[:, 1:2]).squeeze(-1)
            pred_t_s = model.head_tis(z_m_s[:, 1:2]).squeeze(-1)
            sup_tis = (nn.functional.mse_loss(pred_t_b, m_b_pair_tis)
                       + nn.functional.mse_loss(pred_t_s, m_s_pair_tis))

            z_m_flip_b = z_m_b.clone(); z_m_flip_b[:, 0] = sc_z0_target
            x_flip_b2s = model.decode(z_m_flip_b, z_b_b)
            paired_b2s = nn.functional.mse_loss(x_flip_b2s, Xs_pair)
            z_m_flip_s = z_m_s.clone(); z_m_flip_s[:, 0] = bulk_z0_target
            x_flip_s2b = model.decode(z_m_flip_s, z_b_s)
            paired_s2b = nn.functional.mse_loss(x_flip_s2b, Xb_pair)

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
                    + LAM_CYCLE * cyc + LAM_LEAK * leak
                    + LAM_DONOR_ID * donor_id)
            opt.zero_grad(); loss.backward(); opt.step()

            if fold == 0 and (ep % 25 == 0 or ep == 1):
                print(f"    ep{ep:3d}  loss={loss.item():.3f}  recon={recon.item():.3f}  "
                      f"paired_b2s={paired_b2s.item():.3f}  donor_id={donor_id.item():.4f}",
                      flush=True)

        model.eval()
        with torch.no_grad():
            x_held_b = torch.from_numpy(bulk_x[held_bulk_i:held_bulk_i+1]).to(DEVICE)
            mu_m_h, _, mu_b_h, _ = model.encode(x_held_b)
            z_m_flip = mu_m_h.clone(); z_m_flip[:, 0] = sc_z0_target
            x_hat_sc = model.decode(z_m_flip, mu_b_h).cpu().numpy()[0]

        pool_idx = np.where(sn_tissue == held_ts)[0]
        cos_to_pool = np.array([_cosine(x_hat_sc, sn_x[i]) for i in pool_idx])
        order = np.argsort(-cos_to_pool)
        ordered_donors = sn_donor[pool_idx][order]
        rank_first_same = next((i for i, dn in enumerate(ordered_donors) if dn == held_dn), None)
        rank_first_same_1based = (rank_first_same + 1) if rank_first_same is not None else None
        n_same_in_pool = int((sn_donor[pool_idx] == held_dn).sum())
        same_donor_ranks = [i + 1 for i, dn in enumerate(ordered_donors) if dn == held_dn]
        topK = {K: float(np.mean(ordered_donors[:K] == held_dn)) for K in [1, 3, 5, 10]}

        print(f"  fold {fold+1}/{len(paired_dts)}: held=({held_dn}, {held_ts}); "
              f"first-same-rank={rank_first_same_1based}; "
              f"top-1={topK[1]:.2f}; top-3={topK[3]:.2f}; mean rank={np.mean(same_donor_ranks):.2f}",
              flush=True)

        fold_results.append({
            "held_donor": held_dn, "held_tissue": held_ts,
            "n_pool": len(pool_idx), "n_same_donor_in_pool": n_same_in_pool,
            "rank_first_same": rank_first_same_1based,
            "mean_rank_same_donor": float(np.mean(same_donor_ranks)),
            "topK_purity": topK,
            "n_paired_train_pairs": len(paired_pairs),
        })

    arr_top1 = np.array([f["topK_purity"][1] for f in fold_results])
    arr_top3 = np.array([f["topK_purity"][3] for f in fold_results])
    arr_top5 = np.array([f["topK_purity"][5] for f in fold_results])
    arr_rank1 = np.array([f["rank_first_same"] for f in fold_results])
    arr_mean_rank = np.array([f["mean_rank_same_donor"] for f in fold_results])
    arr_n_same = np.array([f["n_same_donor_in_pool"] for f in fold_results])
    arr_n_pool = np.array([f["n_pool"] for f in fold_results])
    expected_top1_random = float(np.mean(arr_n_same / arr_n_pool))
    expected_rank1_random = float(np.mean((arr_n_pool - arr_n_same + 1) / (arr_n_same + 1)))
    print(f"\n=== Run 17 LOO summary across {len(fold_results)} folds ===")
    print(f"  mean top-1: {arr_top1.mean():.3f}  (random ~{expected_top1_random:.3f})")
    print(f"  mean top-3: {arr_top3.mean():.3f}")
    print(f"  mean top-5: {arr_top5.mean():.3f}")
    print(f"  mean rank-of-first-same: {arr_rank1.mean():.2f}  (random ~{expected_rank1_random:.2f})")
    print(f"  mean rank-of-all-same:   {arr_mean_rank.mean():.2f}")

    out = {
        "config": {"epochs": EPOCHS, "lr": LR, "lam_paired": LAM_PAIRED,
                   "lam_leak": LAM_LEAK, "lam_cycle": LAM_CYCLE,
                   "lam_donor_id": LAM_DONOR_ID,
                   "architecture": "DecomposedVAE: (1-alpha)*donor_branch(z_bio) + alpha*template(z_meta)"},
        "fold_results": fold_results,
        "summary": {
            "mean_top1": float(arr_top1.mean()),
            "mean_top3": float(arr_top3.mean()),
            "mean_top5": float(arr_top5.mean()),
            "mean_rank_first_same": float(arr_rank1.mean()),
            "mean_rank_all_same": float(arr_mean_rank.mean()),
            "expected_top1_random": expected_top1_random,
            "expected_rank1_random": expected_rank1_random,
        },
    }
    with (OUT_DIR / "q27_eraslan_run17_decompose.json").open("w") as f:
        json.dump(out, f, indent=2)
    print(f"saved → {OUT_DIR / 'q27_eraslan_run17_decompose.json'}")


if __name__ == "__main__":
    main()
