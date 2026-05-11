"""Q34 / Run 22 — CLIP/SimCLR-style contrastive embedding.

The pivot: drop the decoder entirely. Only learn an encoder that puts
paired (bulk_d, sn_d) samples close together in embedding space and
others far apart. This is what scVI / scANVI / scGen / CLIP do for
cross-modality matching.

Architecture:
- single MLP encoder: 11374 -> 1024 -> 512 -> 256 -> z (50)
- normalized cosine sim, InfoNCE temperature 0.1
- optional: small auxiliary recon decoder for stability (lam_recon)

Test: encode held-out bulk → NN among encoded sn samples in tissue.
Goal: beat raw cosine (0.50) and PCA-50 (0.51) baselines.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

PAIRED_NPZ = "/Users/rls/ecs271/data/sc/eraslan/eraslan_paired_plus_gtex.npz"
OUT_DIR = ROOT / "analysis" / "results" / "q20_disentangled"

EPOCHS = 300
LR = 5e-4
TAU = 0.1
LAM_RECON = 0.5     # small recon to stabilize
Z_DIM = 50


def _mlp(dims, dropout=0.1):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class ContrastiveAE(nn.Module):
    def __init__(self, input_dim, z_dim=50):
        super().__init__()
        self.encoder = _mlp([input_dim, 1024, 512, 256, z_dim])
        self.decoder = _mlp([z_dim, 256, 512, 1024, input_dim])

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def _cosine(a, b):
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def info_nce(z_a, z_b, tau=0.1):
    """Symmetric InfoNCE between paired rows of z_a and z_b."""
    a = nn.functional.normalize(z_a, dim=1)
    b = nn.functional.normalize(z_b, dim=1)
    logits_ab = a @ b.t() / tau
    logits_ba = b @ a.t() / tau
    target = torch.arange(z_a.size(0), device=z_a.device)
    return 0.5 * (nn.functional.cross_entropy(logits_ab, target)
                  + nn.functional.cross_entropy(logits_ba, target))


def main():
    print(f"[Q34/Run22 contrastive] device={DEVICE}  z={Z_DIM}  tau={TAU}  lam_recon={LAM_RECON}")
    d = np.load(PAIRED_NPZ, allow_pickle=True)
    bulk_x = d["bulk_x"].astype(np.float32)
    bulk_donor = np.asarray(d["bulk_donor"], dtype=str)
    bulk_tissue = np.asarray(d["bulk_tissue"], dtype=str)
    bulk_is_eraslan = np.asarray(d["bulk_is_eraslan"], dtype=bool)
    sn_x = d["sn_x"].astype(np.float32)
    sn_donor = np.asarray(d["sn_donor"], dtype=str)
    sn_tissue = np.asarray(d["sn_tissue"], dtype=str)

    sn_idx_per_dt = {}
    for i, (dn, ts) in enumerate(zip(sn_donor, sn_tissue)):
        sn_idx_per_dt.setdefault((dn, ts), []).append(i)
    eraslan_idx = np.where(bulk_is_eraslan)[0]
    bulk_idx_per_dt = {(bulk_donor[i], bulk_tissue[i]): i for i in eraslan_idx}
    paired_dts = [(dn, ts) for (dn, ts) in bulk_idx_per_dt if (dn, ts) in sn_idx_per_dt]
    print(f"  paired (donor, tissue) groups: {len(paired_dts)}")

    fold_results = []
    for fold, (held_dn, held_ts) in enumerate(paired_dts):
        held_bulk_i = bulk_idx_per_dt[(held_dn, held_ts)]
        held_sn_rows = set(sn_idx_per_dt[(held_dn, held_ts)])
        train_bulk_mask = np.arange(len(bulk_x)) != held_bulk_i
        train_sn_mask = np.array([i not in held_sn_rows for i in range(len(sn_x))])
        Xb_all = bulk_x[train_bulk_mask]
        donor_b_all = bulk_donor[train_bulk_mask]
        tissue_b_all = bulk_tissue[train_bulk_mask]
        is_eraslan_all = bulk_is_eraslan[train_bulk_mask]
        Xs_tr = sn_x[train_sn_mask]; donor_s_tr = sn_donor[train_sn_mask]; tissue_s_tr = sn_tissue[train_sn_mask]

        sn_by_dt_tr = {}
        for j, (dn, ts) in enumerate(zip(donor_s_tr, tissue_s_tr)):
            sn_by_dt_tr.setdefault((dn, ts), []).append(j)
        paired_pairs = []
        for ii in np.where(is_eraslan_all)[0]:
            for j in sn_by_dt_tr.get((donor_b_all[ii], tissue_b_all[ii]), []):
                paired_pairs.append((ii, j))
        if not paired_pairs:
            continue
        if fold == 0:
            print(f"  fold 1: paired pairs = {len(paired_pairs)}, total bulk = {len(Xb_all)}")

        bulk_pair_idx = np.array([p[0] for p in paired_pairs])
        sn_pair_idx = np.array([p[1] for p in paired_pairs])
        Xb_pair = torch.from_numpy(Xb_all[bulk_pair_idx]).to(DEVICE)
        Xs_pair = torch.from_numpy(Xs_tr[sn_pair_idx]).to(DEVICE)

        model = ContrastiveAE(input_dim=bulk_x.shape[1], z_dim=Z_DIM).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)

        for ep in range(1, EPOCHS + 1):
            z_b = model.encode(Xb_pair)
            z_s = model.encode(Xs_pair)
            x_hat_b = model.decode(z_b)
            x_hat_s = model.decode(z_s)
            recon = nn.functional.mse_loss(x_hat_b, Xb_pair) + nn.functional.mse_loss(x_hat_s, Xs_pair)
            nce = info_nce(z_b, z_s, tau=TAU)
            loss = nce + LAM_RECON * recon
            opt.zero_grad(); loss.backward(); opt.step()
            if fold == 0 and (ep % 50 == 0 or ep == 1):
                with torch.no_grad():
                    a = nn.functional.normalize(z_b, dim=1)
                    b = nn.functional.normalize(z_s, dim=1)
                    sim = a @ b.t()
                    train_top1 = (sim.argmax(dim=1) == torch.arange(z_b.size(0), device=DEVICE)).float().mean().item()
                print(f"    ep{ep:3d}  loss={loss.item():.3f}  recon={recon.item():.3f}  "
                      f"nce={nce.item():.3f}  train_top1={train_top1:.3f}", flush=True)

        model.eval()
        with torch.no_grad():
            x_held_b = torch.from_numpy(bulk_x[held_bulk_i:held_bulk_i+1]).to(DEVICE)
            z_held = model.encode(x_held_b).cpu().numpy()[0]
            sn_full = torch.from_numpy(sn_x).to(DEVICE)
            z_sn_all = model.encode(sn_full).cpu().numpy()

        pool_idx = np.where(sn_tissue == held_ts)[0]
        z_sn_pool = z_sn_all[pool_idx]
        cos = np.array([_cosine(z_held, z_sn_pool[i]) for i in range(len(z_sn_pool))])
        order = np.argsort(-cos)
        ordered_donors = sn_donor[pool_idx][order]
        rank_first_same = next((i for i, dn in enumerate(ordered_donors) if dn == held_dn), None)
        rank_first_same_1based = (rank_first_same + 1) if rank_first_same is not None else None
        n_same_in_pool = int((sn_donor[pool_idx] == held_dn).sum())
        topK = {K: float(np.mean(ordered_donors[:K] == held_dn)) for K in [1, 3, 5, 10]}
        print(f"  fold {fold+1}/{len(paired_dts)}: held=({held_dn}, {held_ts}); "
              f"first-same-rank={rank_first_same_1based}; top-1={topK[1]:.2f}; top-3={topK[3]:.2f}",
              flush=True)
        fold_results.append({
            "held_donor": held_dn, "held_tissue": held_ts,
            "n_pool": len(pool_idx), "n_same_donor_in_pool": n_same_in_pool,
            "rank_first_same": rank_first_same_1based,
            "topK_purity": topK,
        })

    arr_top1 = np.array([f["topK_purity"][1] for f in fold_results])
    arr_top3 = np.array([f["topK_purity"][3] for f in fold_results])
    arr_top5 = np.array([f["topK_purity"][5] for f in fold_results])
    arr_top10 = np.array([f["topK_purity"][10] for f in fold_results])
    arr_rank1 = np.array([f["rank_first_same"] for f in fold_results])
    arr_n_same = np.array([f["n_same_donor_in_pool"] for f in fold_results])
    arr_n_pool = np.array([f["n_pool"] for f in fold_results])
    expected_top1_random = float(np.mean(arr_n_same / arr_n_pool))
    expected_rank1_random = float(np.mean((arr_n_pool - arr_n_same + 1) / (arr_n_same + 1)))
    print(f"\n=== Run 22 (contrastive embedding) LOO summary across {len(fold_results)} folds ===")
    print(f"  mean top-1: {arr_top1.mean():.3f}  (random ~{expected_top1_random:.3f})")
    print(f"  mean top-3: {arr_top3.mean():.3f}")
    print(f"  mean top-5: {arr_top5.mean():.3f}")
    print(f"  mean top-10: {arr_top10.mean():.3f}")
    print(f"  mean rank-of-first-same: {arr_rank1.mean():.2f}  (random ~{expected_rank1_random:.2f})")
    print(f"  Q32 baselines for comparison: raw cosine 0.500, PCA-50 0.510")

    out = {"config": {"epochs": EPOCHS, "lr": LR, "tau": TAU, "z_dim": Z_DIM,
                      "lam_recon": LAM_RECON,
                      "method": "ContrastiveAE: InfoNCE on paired (bulk, sn)"},
           "fold_results": fold_results,
           "summary": {"mean_top1": float(arr_top1.mean()),
                       "mean_top3": float(arr_top3.mean()),
                       "mean_top5": float(arr_top5.mean()),
                       "mean_top10": float(arr_top10.mean()),
                       "mean_rank_first_same": float(arr_rank1.mean()),
                       "expected_top1_random": expected_top1_random,
                       "expected_rank1_random": expected_rank1_random}}
    with (OUT_DIR / "q34_run22_contrastive.json").open("w") as f:
        json.dump(out, f, indent=2)
    print(f"saved → {OUT_DIR / 'q34_run22_contrastive.json'}")


if __name__ == "__main__":
    main()
