"""Q44 — External benchmark: cross-modality tissue classification.

The proposal's Goal #3 is "improve downstream predictive performance by
jointly leveraging bulk and sc." This benchmark tests that directly.

Setup:
1. Train tissue classifier on GTEx bulk (per-(donor, tissue) bulk samples).
2. Evaluate on Eraslan sn pseudobulks: how accurately does the bulk-trained
   classifier predict the right tissue from a sn sample? — this is the
   cross-modality DOMAIN SHIFT baseline.
3. Apply our VAE flip (sn → bulk) before classification. Does flipping
   improve classification accuracy?

If our flip operation preserves enough biology to enable a bulk-trained
classifier to work on sn-derived inputs, then the flip is useful even if
not perfect at donor-level NN.

This complements the donor-NN test: NN tests "donor identity through flip",
this tests "tissue identity through flip" — which is a meaningful proxy
for the proposal's "joint cross-modality utility" goal.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, top_k_accuracy_score
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.disentangled_vae import kl_with_free_bits, hsic_penalty
from analysis.q43_run26_additive_contrastive import AdditiveVAE, info_nce

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

PAIRED_NPZ = "/Users/rls/ecs271/data/sc/eraslan/eraslan_paired_plus_gtex.npz"
OUT_DIR = ROOT / "analysis" / "results" / "q20_disentangled"

EPOCHS = 200
LR = 1e-3
LAM_PAIRED = 5.0
LAM_LEAK = 0.3
LAM_CYCLE = 0.3
LAM_SUP = 1.0
LAM_DONOR_ID = 5.0
LAM_NCE_FLIP = 10.0
NCE_TAU = 0.1
BETA_BIO = 1e-3
Z_BIO_DIM = 5


def _train_additive_vae(Xb_pair, Xs_pair, m_b_tis, m_s_tis, n_genes,
                         bulk_z0, sc_z0):
    model = AdditiveVAE(input_dim=n_genes, z_meta_dim=2, z_bio_dim=Z_BIO_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    m_b_mod = torch.zeros(Xb_pair.size(0), device=DEVICE)
    m_s_mod = torch.ones(Xs_pair.size(0), device=DEVICE)
    for ep in range(1, EPOCHS + 1):
        x_hat_b, mu_m_b, lv_m_b, mu_b_b, lv_b_b, z_m_b, z_b_b = model(Xb_pair)
        x_hat_s, mu_m_s, lv_m_s, mu_b_s, lv_b_s, z_m_s, z_b_s = model(Xs_pair)
        recon = nn.functional.mse_loss(x_hat_b, Xb_pair) + nn.functional.mse_loss(x_hat_s, Xs_pair)
        kl = (kl_with_free_bits(mu_b_b, lv_b_b, 0.5) + kl_with_free_bits(mu_b_s, lv_b_s, 0.5)) / Xb_pair.size(0)
        sup_mod = (nn.functional.binary_cross_entropy_with_logits(model.head_mod(z_m_b[:, 0:1]).squeeze(-1), m_b_mod)
                   + nn.functional.binary_cross_entropy_with_logits(model.head_mod(z_m_s[:, 0:1]).squeeze(-1), m_s_mod))
        sup_tis = (nn.functional.mse_loss(model.head_tis(z_m_b[:, 1:2]).squeeze(-1), m_b_tis)
                   + nn.functional.mse_loss(model.head_tis(z_m_s[:, 1:2]).squeeze(-1), m_s_tis))
        z_m_flip_b = z_m_b.clone(); z_m_flip_b[:, 0] = sc_z0
        x_flip_b2s = model.decode(z_m_flip_b, z_b_b)
        paired_b2s = nn.functional.mse_loss(x_flip_b2s, Xs_pair)
        z_m_flip_s = z_m_s.clone(); z_m_flip_s[:, 0] = bulk_z0
        x_flip_s2b = model.decode(z_m_flip_s, z_b_s)
        paired_s2b = nn.functional.mse_loss(x_flip_s2b, Xb_pair)
        nce_b2s = info_nce(x_flip_b2s, Xs_pair, tau=NCE_TAU)
        nce_s2b = info_nce(x_flip_s2b, Xb_pair, tau=NCE_TAU)
        mu_m_re_b, _, mu_b_re_b, _ = model.encode(x_flip_b2s)
        mu_m_re_s, _, mu_b_re_s, _ = model.encode(x_flip_s2b)
        cyc = nn.functional.mse_loss(mu_b_re_b, mu_b_b.detach()) + nn.functional.mse_loss(mu_b_re_s, mu_b_s.detach())
        mod_col = torch.cat([torch.full((Xb_pair.size(0),1), 0.0, device=DEVICE),
                             torch.full((Xs_pair.size(0),1), 1.0, device=DEVICE)])
        z_b_combined = torch.cat([z_b_b, z_b_s], dim=0)
        leak = hsic_penalty(z_b_combined, mod_col)
        donor_id = nn.functional.mse_loss(mu_b_b, mu_b_s)
        loss = (recon + BETA_BIO * kl + LAM_SUP * (sup_mod + sup_tis)
                + LAM_PAIRED * (paired_b2s + paired_s2b) + LAM_CYCLE * cyc
                + LAM_LEAK * leak + LAM_DONOR_ID * donor_id
                + LAM_NCE_FLIP * (nce_b2s + nce_s2b))
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def main():
    print(f"[Q44 external benchmark] device={DEVICE}")
    d = np.load(PAIRED_NPZ, allow_pickle=True)
    bulk_x = d["bulk_x"].astype(np.float32)
    bulk_donor = np.asarray(d["bulk_donor"], dtype=str)
    bulk_tissue = np.asarray(d["bulk_tissue"], dtype=str)
    bulk_is_eraslan = np.asarray(d["bulk_is_eraslan"], dtype=bool)
    sn_x = d["sn_x"].astype(np.float32)
    sn_donor = np.asarray(d["sn_donor"], dtype=str)
    sn_tissue = np.asarray(d["sn_tissue"], dtype=str)
    print(f"  bulk: {bulk_x.shape} ({bulk_is_eraslan.sum()} Eraslan, {(~bulk_is_eraslan).sum()} extra GTEx)")
    print(f"  sn:   {sn_x.shape}")
    tissues = sorted(set(bulk_tissue))
    tissue_to_idx = {t: i for i, t in enumerate(tissues)}
    n_tissues = len(tissues)
    print(f"  tissues ({n_tissues}): {tissues}")

    # === Baseline 1: train tissue classifier on ALL GTEx bulk (using "extra" GTEx, not Eraslan-paired) ===
    train_bulk_mask = ~bulk_is_eraslan  # use the 4413 extra GTEx donors
    print(f"\n=== Training tissue classifier on {train_bulk_mask.sum()} GTEx bulk samples ===")
    y_train = np.array([tissue_to_idx[t] for t in bulk_tissue[train_bulk_mask]])
    clf = LogisticRegression(max_iter=2000, multi_class="multinomial", C=1.0, n_jobs=-1)
    clf.fit(bulk_x[train_bulk_mask], y_train)
    train_acc = clf.score(bulk_x[train_bulk_mask], y_train)
    print(f"  classifier train acc: {train_acc:.3f}")

    # Held-out Eraslan bulk for sanity check
    y_eraslan_bulk = np.array([tissue_to_idx[t] for t in bulk_tissue[bulk_is_eraslan]])
    eraslan_bulk_acc = clf.score(bulk_x[bulk_is_eraslan], y_eraslan_bulk)
    eraslan_bulk_bal_acc = balanced_accuracy_score(y_eraslan_bulk, clf.predict(bulk_x[bulk_is_eraslan]))
    print(f"  bulk -> tissue (Eraslan bulk):  acc={eraslan_bulk_acc:.3f}  bal_acc={eraslan_bulk_bal_acc:.3f}")

    # === Baseline 2: apply same classifier directly to sn pseudobulks ===
    # CROSS-MODALITY DOMAIN SHIFT — how much does it hurt?
    y_sn = np.array([tissue_to_idx[t] for t in sn_tissue])
    sn_pred_baseline = clf.predict(sn_x)
    sn_acc_baseline = accuracy_score(y_sn, sn_pred_baseline)
    sn_bal_acc_baseline = balanced_accuracy_score(y_sn, sn_pred_baseline)
    sn_top3_baseline = top_k_accuracy_score(y_sn, clf.predict_proba(sn_x), k=3, labels=list(range(n_tissues)))
    print(f"\n  sn -> tissue (DIRECT, no flip):  acc={sn_acc_baseline:.3f}  "
          f"bal_acc={sn_bal_acc_baseline:.3f}  top-3={sn_top3_baseline:.3f}")

    # === Train VAE on paired data (one big multi-tissue model, no LOO) ===
    print(f"\n=== Training AdditiveVAE on ALL Eraslan paired data ===")
    sn_idx_per_dt = {}
    for i, (dn, ts) in enumerate(zip(sn_donor, sn_tissue)):
        sn_idx_per_dt.setdefault((dn, ts), []).append(i)
    eraslan_idx = np.where(bulk_is_eraslan)[0]
    paired_pairs = []
    for ii in eraslan_idx:
        for j in sn_idx_per_dt.get((bulk_donor[ii], bulk_tissue[ii]), []):
            paired_pairs.append((ii, j))
    print(f"  paired pairs: {len(paired_pairs)}")
    bulk_pair_idx = np.array([p[0] for p in paired_pairs])
    sn_pair_idx = np.array([p[1] for p in paired_pairs])
    Xb_pair = torch.from_numpy(bulk_x[bulk_pair_idx]).to(DEVICE)
    Xs_pair = torch.from_numpy(sn_x[sn_pair_idx]).to(DEVICE)
    m_b_tis = torch.tensor([tissue_to_idx[t] for t in bulk_tissue[bulk_pair_idx]],
                           device=DEVICE, dtype=torch.float32) / max(n_tissues - 1, 1)
    m_s_tis = torch.tensor([tissue_to_idx[t] for t in sn_tissue[sn_pair_idx]],
                           device=DEVICE, dtype=torch.float32) / max(n_tissues - 1, 1)
    model = _train_additive_vae(Xb_pair, Xs_pair, m_b_tis, m_s_tis,
                                  bulk_x.shape[1], bulk_z0=-3.0, sc_z0=3.0)
    print(f"  VAE trained.")

    # === Apply VAE flip sn → bulk on the sn samples, classify ===
    model.eval()
    sn_flipped = np.zeros_like(sn_x)
    chunk = 64
    with torch.no_grad():
        for i in range(0, len(sn_x), chunk):
            x = torch.from_numpy(sn_x[i:i+chunk]).to(DEVICE)
            mu_m, _, mu_b, _ = model.encode(x)
            z_m_flip = mu_m.clone(); z_m_flip[:, 0] = -3.0  # flip to bulk
            sn_flipped[i:i+chunk] = model.decode(z_m_flip, mu_b).cpu().numpy()

    sn_pred_flipped = clf.predict(sn_flipped)
    sn_acc_flipped = accuracy_score(y_sn, sn_pred_flipped)
    sn_bal_acc_flipped = balanced_accuracy_score(y_sn, sn_pred_flipped)
    sn_top3_flipped = top_k_accuracy_score(y_sn, clf.predict_proba(sn_flipped), k=3, labels=list(range(n_tissues)))
    print(f"  sn -> tissue (FLIPPED to bulk):  acc={sn_acc_flipped:.3f}  "
          f"bal_acc={sn_bal_acc_flipped:.3f}  top-3={sn_top3_flipped:.3f}")

    # === Per-tissue breakdown ===
    print(f"\n--- Per-tissue: sn → tissue accuracy ---")
    print(f"{'tissue':<22} {'n_sn':>5} {'direct':>8} {'flipped':>8} {'delta':>8}")
    per_tissue = {}
    for t in tissues:
        ti = tissue_to_idx[t]
        mask = (y_sn == ti)
        if mask.sum() == 0: continue
        acc_d = accuracy_score(y_sn[mask], sn_pred_baseline[mask])
        acc_f = accuracy_score(y_sn[mask], sn_pred_flipped[mask])
        per_tissue[t] = {"n_sn": int(mask.sum()), "direct": acc_d, "flipped": acc_f, "delta": acc_f - acc_d}
        print(f"{t:<22} {int(mask.sum()):>5} {acc_d:>8.3f} {acc_f:>8.3f} {acc_f - acc_d:>+8.3f}")

    # Confusion summary
    print(f"\n--- Confusion: which tissue gets predicted? ---")
    from sklearn.metrics import confusion_matrix
    cm_baseline = confusion_matrix(y_sn, sn_pred_baseline, labels=list(range(n_tissues)))
    cm_flipped = confusion_matrix(y_sn, sn_pred_flipped, labels=list(range(n_tissues)))
    print("Baseline confusion (rows=true, cols=pred):")
    print(f"  {'true\\pred':<22} " + " ".join(f"{t[:6]:>7}" for t in tissues))
    for i, t in enumerate(tissues):
        print(f"  {t:<22} " + " ".join(f"{cm_baseline[i][j]:>7}" for j in range(n_tissues)))
    print("Flipped confusion:")
    print(f"  {'true\\pred':<22} " + " ".join(f"{t[:6]:>7}" for t in tissues))
    for i, t in enumerate(tissues):
        print(f"  {t:<22} " + " ".join(f"{cm_flipped[i][j]:>7}" for j in range(n_tissues)))

    out = {
        "baseline": {
            "name": "Logistic Regression on GTEx bulk, applied DIRECTLY to sn pseudobulks",
            "sn_acc": float(sn_acc_baseline),
            "sn_bal_acc": float(sn_bal_acc_baseline),
            "sn_top3": float(sn_top3_baseline),
        },
        "vae_flip": {
            "name": "Same classifier, applied to VAE-flipped (sn -> bulk) pseudobulks",
            "sn_acc": float(sn_acc_flipped),
            "sn_bal_acc": float(sn_bal_acc_flipped),
            "sn_top3": float(sn_top3_flipped),
        },
        "sanity": {
            "eraslan_bulk_acc": float(eraslan_bulk_acc),
            "eraslan_bulk_bal_acc": float(eraslan_bulk_bal_acc),
            "classifier_train_acc": float(train_acc),
        },
        "per_tissue": per_tissue,
        "tissues": tissues,
        "n_train_bulk": int(train_bulk_mask.sum()),
        "n_sn": int(len(sn_x)),
    }
    with (OUT_DIR / "q44_external_benchmark.json").open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved → {OUT_DIR / 'q44_external_benchmark.json'}")


if __name__ == "__main__":
    main()
